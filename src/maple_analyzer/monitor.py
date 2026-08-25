"""Background capture/OCR workers used by the live HUD.

The Tk event loop must never wait for a screen grab or ONNX inference.  The
status worker samples the four bottom-bar fields at the configured interval;
the auxiliary worker scans pickup messages and shortcut slots at a slower,
independent cadence; and the context worker refreshes the visible map/job
labels only every few seconds.  OverlayApp consumes the small immutable
results on its normal Tk timer.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .economy import parse_mesos_amount, parse_slot_count
from .parser import StatSnapshot, parse_fields
from .regions import (
    PICKUP_LINE_BOXES,
    PICKUP_LINE_HEIGHT,
    PICKUP_LINE_TOP_OFFSET,
    SHORTCUT_SLOT_BOXES,
)
from .settings import PotionSlotConfig


AUX_SCAN_MIN_MS = 200
PICKUP_SCAN_MIN_MS = 100
PICKUP_DETECTION_INTERVAL_S = 0.35


class MonitorSource(Protocol):
    def grab_fields(self) -> dict[str, Any]:
        ...

    def grab_auxiliary(self) -> dict[str, Any]:
        ...

    def grab_context(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class StatusReading:
    snapshot: StatSnapshot
    error: str | None = None
    client_size: tuple[int, int] | None = None
    timestamp: float | None = None


@dataclass(frozen=True)
class AuxiliaryReading:
    lines: tuple[tuple[str, float], ...] = ()
    counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    timestamp: float | None = None
    # Kept true by default for compatibility with custom monitor producers;
    # the split-cadence worker sets it false on potion-only scans so an empty
    # line list does not erase the visible mesos feed between pickup scans.
    pickup_scanned: bool = True


@dataclass(frozen=True)
class ContextReading:
    map_name: str | None = None
    job_name: str | None = None
    error: str | None = None


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_CANONICAL_BARRACKS_RE = re.compile(r"第\d+軍營")
_CONTEXT_NOISE = {
    "小地圖", "小地图", "地圖", "地图", "MINI", "MAP", "LV", "Lv", "lv",
}
_JOB_ALIASES = {
    # The small white job label in the status bar is commonly recognized as
    # 快盜/侠盗 by PP-OCR even though the game renders 俠盜.  Keep this narrow
    # correction local to the job field instead of changing general OCR text.
    "快盜": "俠盜",
    "快盗": "俠盜",
    "恢盜": "俠盜",
    "恢盗": "俠盜",
    "侠盗": "俠盜",
    "快咨": "俠盜",
}


def _put_latest(target: queue.Queue, value: object) -> None:
    """Keep a bounded queue fresh; stale frames are less useful than latest."""
    try:
        target.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        target.get_nowait()
    except queue.Empty:
        pass
    try:
        target.put_nowait(value)
    except queue.Full:
        pass


def _line_text(line: object) -> str:
    value = getattr(line, "text", line)
    if isinstance(value, (list, tuple)) and len(value) > 1:
        value = value[1]
    return str(value).strip() if value is not None else ""


def _clean_context_text(value: str) -> str:
    text = re.sub(r"\s+", "", value).strip("|:：·.,。()[]【】")
    return text[:32]


def _normalize_context_text(value: str, *, kind: str) -> str:
    """Normalize the few stable Traditional/Simplified OCR substitutions."""
    text = _clean_context_text(value)
    if kind == "map":
        # The client renders 第3軍營, while RapidOCR sometimes returns the
        # simplified glyphs 军/营 on this very small second-line crop.  管 is
        # another common visual confusion for 營 in the same tiny font.
        text = text.translate(str.maketrans({"军": "軍", "营": "營"}))
        # When the leading 第3 is missed, the same confusion appears as the
        # bare fragment 軍管.  Normalize it before the weak-candidate filter
        # so a partial OCR result is never shown as if it were a real map.
        if "軍" in text and text.endswith("管"):
            text = text[:-1] + "營"
        text = re.sub(r"第(\d+)軍(?:管|営)", r"第\1軍營", text)
    return text


def _compact_map_text(value: str) -> str:
    """Remove OCR separators while keeping the map's Chinese characters."""
    return re.sub(r"[\s/\\|:：·.,。()\[\]【】_\-]+", "", value)


def _canonicalize_map_candidate(value: str) -> str:
    """Return the stable map token when OCR included nearby UI noise."""
    compact = _compact_map_text(value)
    match = _CANONICAL_BARRACKS_RE.search(compact)
    return match.group(0) if match else value


def _is_weak_map_candidate(value: str) -> bool:
    """Reject the common partial result ``軍/營`` as a map name.

    The second mini-map line is only a few pixels high.  Recognition-only OCR
    can confidently return the final two glyphs even when it missed the map's
    identifying number.  Passing that fragment to the drop database produces
    a misleading ``map not found`` result, so it must not replace a previously
    confirmed map name.
    """
    compact = _compact_map_text(value)
    return (
        "軍" in compact
        and any(glyph in compact for glyph in ("營", "管"))
        and not re.search(r"第\d+", compact)
    )


def _has_complete_map_candidate(candidates: list[str]) -> bool:
    return any(_CANONICAL_BARRACKS_RE.search(_compact_map_text(candidate)) for candidate in candidates)


def _select_context_candidate(candidates: list[str], *, kind: str) -> str | None:
    if not candidates:
        return None
    if kind == "map":
        complete = [
            candidate for candidate in candidates
            if _CANONICAL_BARRACKS_RE.search(_compact_map_text(candidate))
        ]
        if complete:
            return max(complete, key=len)
    return candidates[-1]


def _context_candidates(lines: list[object], *, kind: str) -> list[str]:
    candidates: list[str] = []
    for line in lines:
        text = _normalize_context_text(_line_text(line), kind=kind)
        if kind == "map":
            text = _canonicalize_map_candidate(text)
            if _is_weak_map_candidate(text):
                continue
        if kind == "job":
            # A recognition-only pass over the LV crop can return the job and
            # level together, e.g. "快咨 6 8 LV.".  Keep the CJK job prefix
            # and discard the numeric/status suffix before alias correction.
            text = re.split(r"(?:LV|HP|MP|EXP)|\d", text, maxsplit=1, flags=re.IGNORECASE)[0]
            text = text.strip("|:：·.,。()[]【】")
        if not text or text in _CONTEXT_NOISE or text.isdigit():
            continue
        if text.upper().replace(".", "") in _CONTEXT_NOISE:
            continue
        # Job/map labels in this UI are CJK.  Requiring at least two CJK
        # glyphs rejects LV/68 and the tiny decorative arrows around them.
        if len(_CJK_RE.findall(text)) < 2:
            continue
        if kind == "job":
            text = _JOB_ALIASES.get(text, text)
        candidates.append(text)
    return candidates


def extract_context(ocr: Any, regions: dict[str, Any]) -> ContextReading:
    """Best-effort extraction from the fixed visible map/status labels."""
    map_lines: list[object] = []
    map_images = [
        regions.get("map"),
        regions.get("map_wide"),
    ]
    map_images = [image for image in map_images if image is not None]
    for image in map_images:
        # The focused crop is only one line high.  Recognition on an enlarged
        # copy is materially more reliable than asking detection to rediscover
        # a 75x21-pixel label.  The wider retry recovers the leading "第3" when
        # a client has shifted the mini-map text left by a few pixels.
        try:
            from PIL import Image
            resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC", 3)
            enlarged = image.resize((image.width * 8, image.height * 8), resampling)
            map_lines.append(ocr.read_field(enlarged))
        except Exception:
            pass
        try:
            map_lines.append(ocr.read_field(image))
        except Exception:
            pass

    map_candidates = _context_candidates(map_lines, kind="map")
    if not _has_complete_map_candidate(map_candidates) and map_images:
        # A single contrast retry is cheaper than running detection on every
        # context tick and helps when the game window is dimmed or partially
        # scaled by Windows DPI settings.
        try:
            from PIL import Image, ImageEnhance, ImageOps
            image = map_images[-1]
            enhanced = ImageOps.autocontrast(ImageOps.grayscale(image))
            enhanced = ImageEnhance.Contrast(enhanced).enhance(2.5)
            enhanced = enhanced.resize(
                (image.width * 8, image.height * 8),
                getattr(getattr(Image, "Resampling", Image), "BICUBIC", 3),
            ).convert("RGB")
            map_lines.append(ocr.read_field(enhanced))
        except Exception:
            pass

    map_candidates = _context_candidates(map_lines, kind="map")
    if not _has_complete_map_candidate(map_candidates) and map_images:
        # Detection is the final confirmation path.  It runs only when the
        # cheap recognition passes did not find a complete map token.
        for image in reversed(map_images):
            try:
                map_lines.extend(ocr.read_lines(image))
            except Exception:
                pass
            map_candidates = _context_candidates(map_lines, kind="map")
            if _has_complete_map_candidate(map_candidates):
                break

    job_lines: list[object] = []
    if regions.get("job") is not None:
        image = regions["job"]
        # The job label is also a known single-line field.  Try a cheap
        # recognition-only enlarged crop before the slower detector pass.
        try:
            from PIL import Image
            resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC", 3)
            enlarged = image.resize((image.width * 8, image.height * 8), resampling)
            job_lines.append(ocr.read_field(enlarged))
        except Exception:
            pass
        try:
            job_lines.extend(ocr.read_lines(image))
        except Exception:
            pass
    if regions.get("job") is not None and not _context_candidates(job_lines, kind="job"):
        # The job label is small gray text beside LV.  A contrast-enhanced
        # retry is cheap at the 3s context cadence and avoids adding a second
        # heavyweight OCR pass to the 0.3s status loop.
        try:
            from PIL import Image, ImageEnhance, ImageOps
            image = regions["job"]
            enhanced = ImageOps.grayscale(image)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(4)
            enhanced = enhanced.resize(
                (image.width * 8, image.height * 8),
                getattr(Image, "Resampling", Image).LANCZOS,
            ).convert("RGB")
            job_lines.extend(ocr.read_lines(enhanced))
        except Exception:
            pass
    maps = _context_candidates(map_lines, kind="map")
    jobs = _context_candidates(job_lines, kind="job")
    return ContextReading(
        map_name=_select_context_candidate(maps, kind="map"),
        job_name=_select_context_candidate(jobs, kind="job"),
    )


class BackgroundMonitor:
    """Own the expensive capture/OCR work outside Tk's main thread."""

    def __init__(
        self,
        source: MonitorSource,
        ocr: Any,
        *,
        sample_interval_ms: int = 300,
        aux_scan_ms: int = 250,
        pickup_interval_ms: int = 200,
        context_scan_ms: int = 3000,
    ) -> None:
        self.source = source
        self.ocr = ocr
        self.status_queue: queue.Queue[StatusReading] = queue.Queue(maxsize=24)
        self.auxiliary_queue: queue.Queue[AuxiliaryReading] = queue.Queue(maxsize=4)
        self.context_queue: queue.Queue[ContextReading] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._aux_enabled = threading.Event()
        self._aux_request = threading.Event()
        self._context_request = threading.Event()
        self._lock = threading.Lock()
        self._sample_interval_ms = max(200, min(1000, sample_interval_ms))
        self._aux_scan_ms = max(AUX_SCAN_MIN_MS, aux_scan_ms)
        self._pickup_interval_ms = max(PICKUP_SCAN_MIN_MS, min(1000, pickup_interval_ms))
        self._context_scan_ms = max(1500, context_scan_ms)
        self._track_pickup = True
        self._track_potions = True
        self._configured_potion_slots: tuple[PotionSlotConfig, ...] = ()
        self._potion_slots: tuple[PotionSlotConfig, ...] = ()
        self._threads: list[threading.Thread] = []
        self._next_pickup_detection = 0.0

    def start(self) -> None:
        if self._threads:
            return
        self._threads = [
            threading.Thread(target=self._status_loop, name="maple-status-monitor", daemon=True),
            threading.Thread(target=self._auxiliary_loop, name="maple-economy-monitor", daemon=True),
            threading.Thread(target=self._context_loop, name="maple-context-monitor", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._aux_enabled.set()
        for thread in self._threads:
            thread.join(timeout=0.8)
        self._threads.clear()

    def set_sample_interval(self, value_ms: int) -> None:
        with self._lock:
            self._sample_interval_ms = max(200, min(1000, int(value_ms)))

    def set_pickup_interval(self, value_ms: int) -> None:
        with self._lock:
            self._pickup_interval_ms = max(PICKUP_SCAN_MIN_MS, min(1000, int(value_ms)))

    def set_aux_enabled(self, enabled: bool) -> None:
        if enabled:
            self._aux_enabled.set()
            self._aux_request.set()
        else:
            self._aux_enabled.clear()
            self._aux_request.clear()

    def request_auxiliary_scan(self) -> None:
        """Wake the economy worker for a fresh Start/Resume baseline."""
        self._aux_request.set()

    def request_context(self) -> None:
        self._context_request.set()

    def configure_auxiliary(
        self,
        *,
        track_pickup: bool,
        track_potions: bool,
        potion_slots: list[PotionSlotConfig],
    ) -> None:
        with self._lock:
            self._track_pickup = track_pickup
            self._track_potions = track_potions
            configured = tuple(slot for slot in potion_slots if slot.enabled)
            self._configured_potion_slots = configured
            configured_ids = {slot.slot for slot in configured}
            # Once the user has explicitly mapped potion slots, do not let an
            # adjacent unconfigured shortcut cell (for example slot 8) be
            # mistaken for the configured blue-water slot 7 when one OCR
            # crop is incomplete.  Keep the all-slots fallback only for a
            # brand-new configuration with no enabled rows at all.
            self._potion_slots = configured or tuple(
                PotionSlotConfig(slot=slot, kind="both", enabled=True)
                for slot in SHORTCUT_SLOT_BOXES
            )

    def _status_loop(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                field_images = self.source.grab_fields()
                read_fields = getattr(self.ocr, "read_fields", None)
                if callable(read_fields):
                    field_text = read_fields(field_images)
                else:
                    field_text = {
                        name: self.ocr.read_field(image)
                        for name, image in field_images.items()
                    }
                snapshot = parse_fields(field_text)
                error = None
            except RuntimeError as exc:
                snapshot = StatSnapshot(None, None, None, None, None, None, None)
                error = str(exc)
            except Exception as exc:
                snapshot = StatSnapshot(None, None, None, None, None, None, None)
                error = f"OCR: {exc}"
            _put_latest(
                self.status_queue,
                StatusReading(
                    snapshot=snapshot,
                    error=error,
                    client_size=getattr(self.source, "client_size", None),
                    timestamp=time.monotonic(),
                ),
            )
            with self._lock:
                interval = self._sample_interval_ms / 1000
            remaining = max(0.01, interval - (time.perf_counter() - started))
            self._stop.wait(remaining)

    def _auxiliary_loop(self) -> None:
        next_pickup_scan = 0.0
        next_potion_scan = 0.0
        while not self._stop.is_set():
            if not self._aux_enabled.wait(0.1):
                continue
            now = time.monotonic()
            requested = self._aux_request.is_set()
            with self._lock:
                track_pickup = self._track_pickup
                track_potions = self._track_potions
                pickup_interval = self._pickup_interval_ms / 1000
                potion_interval = self._aux_scan_ms / 1000
            next_due = min(
                next_pickup_scan if track_pickup else float("inf"),
                next_potion_scan if track_potions else float("inf"),
            )
            if not requested and now < next_due:
                self._stop.wait(min(0.1, next_due - now))
                continue
            self._aux_request.clear()
            pickup_due = track_pickup and (requested or now >= next_pickup_scan)
            potion_due = track_potions and (requested or now >= next_potion_scan)
            if pickup_due:
                next_pickup_scan = now + pickup_interval
            if potion_due:
                next_potion_scan = now + potion_interval
            try:
                with self._lock:
                    configured_slots = self._configured_potion_slots
                    slots = self._potion_slots
                if not (track_pickup or track_potions):
                    continue
                regions = self.source.grab_auxiliary()
                lines: list[tuple[str, float]] = []
                if pickup_due:
                    lines = self._read_pickup_lines(regions, now)
                counts: dict[str, int] = {}
                if potion_due:
                    read_shortcut_counts = getattr(self.ocr, "read_shortcut_counts", None)
                    if callable(read_shortcut_counts) and regions.get("shortcut") is not None:
                        configured_ids = {slot.slot for slot in configured_slots}
                        blue_ids = {
                            slot.slot
                            for slot in configured_slots
                            if slot.kind in ("mp", "both")
                        }
                        try:
                            detected_counts = read_shortcut_counts(
                                regions["shortcut"], configured_ids, blue_ids
                            )
                        except TypeError:
                            # Compatibility with custom OCR adapters that still
                            # expose the original one/two-argument method.
                            try:
                                detected_counts = read_shortcut_counts(
                                    regions["shortcut"], configured_ids
                                )
                            except TypeError:
                                # Compatibility with custom OCR adapters that
                                # still expose the original one-argument method.
                                detected_counts = read_shortcut_counts(regions["shortcut"])
                        enabled_slots = {slot.slot for slot in slots if slot.enabled}
                        counts = {
                            slot_id: count
                            for slot_id, count in detected_counts.items()
                            if slot_id in enabled_slots
                        }
                    else:
                        for slot in slots:
                            if not slot.enabled:
                                continue
                            image = regions.get(f"shortcut:{slot.slot}")
                            if image is None:
                                continue
                            read_slot_count = getattr(self.ocr, "read_slot_count", None)
                            count = (
                                read_slot_count(image)
                                if callable(read_slot_count)
                                else parse_slot_count(self.ocr.read_field(image))
                            )
                            if count is not None:
                                counts[slot.slot] = count
                _put_latest(
                    self.auxiliary_queue,
                    AuxiliaryReading(
                        tuple(lines), counts,
                        timestamp=time.monotonic(),
                        pickup_scanned=bool(pickup_due),
                    ),
                )
            except RuntimeError as exc:
                _put_latest(
                    self.auxiliary_queue,
                    AuxiliaryReading(error=str(exc), timestamp=time.monotonic()),
                )
            except Exception as exc:
                _put_latest(
                    self.auxiliary_queue,
                    AuxiliaryReading(error=f"OCR: {exc}", timestamp=time.monotonic()),
                )

    def _read_pickup_lines(self, regions: dict[str, Any], now: float) -> list[tuple[str, float]]:
        line_images = [
            (line_id, regions.get(f"pickup:{line_id}"))
            for line_id in PICKUP_LINE_BOXES
            if regions.get(f"pickup:{line_id}") is not None
        ]
        line_images = [
            (line_id, image)
            for line_id, image in line_images
            if self._image_has_content(image)
        ]
        lines = [
            (
                self.ocr.read_field(image),
                int(line_id) * PICKUP_LINE_HEIGHT
                + PICKUP_LINE_TOP_OFFSET
                + PICKUP_LINE_HEIGHT / 2,
            )
            for line_id, image in line_images
        ]
        if (
            not any(parse_mesos_amount(text) is not None for text, _ in lines)
            and now >= self._next_pickup_detection
        ):
            self._next_pickup_detection = now + PICKUP_DETECTION_INTERVAL_S
            detected: list[Any] = []
            for key in ("pickup", "pickup_wide"):
                image = regions.get(key)
                if image is None:
                    continue
                detected.extend(self.ocr.read_lines(image))
                if any(parse_mesos_amount(_line_text(line)) is not None for line in detected):
                    break
            lines = [(_line_text(line), float(getattr(line, "y", 0) or 0)) for line in detected]
        elif not line_images and regions.get("pickup") is not None:
            detected: list[Any] = []
            for key in ("pickup", "pickup_wide"):
                image = regions.get(key)
                if image is None:
                    continue
                detected.extend(self.ocr.read_lines(image))
                if any(parse_mesos_amount(_line_text(line)) is not None for line in detected):
                    break
            lines = [(_line_text(line), float(getattr(line, "y", 0) or 0)) for line in detected]
        return lines

    def _context_loop(self) -> None:
        grab_context = getattr(self.source, "grab_context", None)
        if not callable(grab_context):
            return
        while not self._stop.is_set():
            try:
                regions = grab_context()
                _put_latest(self.context_queue, extract_context(self.ocr, regions))
            except RuntimeError as exc:
                _put_latest(self.context_queue, ContextReading(error=str(exc)))
            except Exception as exc:
                _put_latest(self.context_queue, ContextReading(error=f"OCR: {exc}"))
            deadline = time.monotonic() + self._context_scan_ms / 1000
            while not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if self._context_request.wait(min(0.1, remaining)):
                    self._context_request.clear()
                    break

    @staticmethod
    def _image_has_content(image: Any) -> bool:
        try:
            histogram = image.convert("L").histogram()
            bright_pixels = sum(histogram[110:])
            return bright_pixels >= max(12, int(image.width * image.height * 0.003))
        except Exception:
            return True
