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
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from .economy import mesos_text_needs_full_detection, parse_mesos_amount, parse_slot_count
from .bar_flash import BarFlashDetector
from .diagnostics import log_exception
from .parser import StatSnapshot, parse_fields
from .regions import (
    PICKUP_LINE_BOXES,
    PICKUP_LINE_HEIGHT,
    PICKUP_LINE_TOP_OFFSET,
)
from .settings import PotionSlotConfig


# Shortcut quantities are a separate fast signal from the pickup feed. Keep
# the lower bound below the game's usual 0.3-0.7s potion animation so a single
# use is not hidden behind a slow full-bar OCR pass.
AUX_SCAN_MIN_MS = 150
PICKUP_SCAN_MIN_MS = 100
PICKUP_DETECTION_INTERVAL_S = 0.35
# The EXP percentage is rounded to two decimals. Keep the display guard as
# loose as Session.EXP_TOTAL_BAND, but prevent a structurally valid OCR frame
# from replacing a good same-level value with an impossible total.
EXP_DISPLAY_TOTAL_BAND = 0.25


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
    # Edge-triggered visual events from the game's HP/MP bar frame.  The
    # economy tracker applies the configured potion-slot kind before using
    # these events, so this is evidence, not an unconditional drink count.
    bar_flash: tuple[str, ...] = ()


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


def merge_status_snapshots(previous: StatSnapshot, incoming: StatSnapshot) -> StatSnapshot:
    """Merge one status frame without publishing transient EXP disappearance.

    Parser output is structurally valid before it reaches this boundary, but a
    tiny EXP crop can still produce a plausible number that is lower than the
    previous same-level value or implies a completely different level total.
    Missing fields already carry forward here; the extra monotonic/percentage
    checks keep a single bad frame from making the live EXP value visibly jump
    backwards and then poisoning the next rate calculation.
    """
    merged = StatSnapshot(*(
        new if new is not None else old
        for new, old in zip(vars(incoming).values(), vars(previous).values())
    ))
    if previous.exp_cur is None or incoming.exp_cur is None:
        return merged

    previous_level = previous.level
    incoming_level = incoming.level
    level_increased = (
        previous_level is not None
        and incoming_level is not None
        and incoming_level > previous_level
    )
    confirmed_level_up = level_increased and incoming.exp_cur < previous.exp_cur

    # A level change without the corresponding EXP reset is normally a one-frame
    # level OCR error. Hold the old level so it cannot weaken the same-level
    # EXP guard or make Session bank a phantom level-up.
    if (
        previous_level is not None
        and incoming_level is not None
        and incoming_level != previous_level
        and not confirmed_level_up
    ):
        merged.level = previous_level

    if confirmed_level_up:
        return merged

    # EXP cannot decrease within one level. If the level OCR is missing for the
    # actual level-up, holding this frame is safer than showing an EXP reset;
    # the next frame with a confirmed level will be accepted normally.
    if incoming.exp_cur < previous.exp_cur:
        merged.exp_cur = previous.exp_cur
        merged.exp_pct = previous.exp_pct if previous.exp_pct is not None else incoming.exp_pct
        return merged

    previous_pct = previous.exp_pct
    incoming_pct = incoming.exp_pct
    if previous_pct is None or incoming_pct is None or previous_pct <= 0 or incoming_pct <= 0:
        return merged

    previous_total = previous.exp_cur / (previous_pct / 100)
    incoming_total = incoming.exp_cur / (incoming_pct / 100)
    band = EXP_DISPLAY_TOTAL_BAND + (0.005 / incoming_pct)
    if previous_total > 0 and abs(incoming_total / previous_total - 1) > band:
        # Keep both pieces together. Publishing the new number with the old
        # percentage (or vice versa) would make the next frame look like a
        # real level-total change to the rate/session layer.
        merged.exp_cur = previous.exp_cur
        merged.exp_pct = previous.exp_pct
    return merged


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


def _image_signature(image: Any) -> tuple | None:
    """Build a cheap visual signature for an auxiliary crop.

    Recognition-only OCR is still expensive when it is repeated for twelve
    pickup rows whose pixels did not change. A small grayscale signature lets
    the worker reuse the previous text and reserve OCR for a newly drawn or
    cleared notification.
    """
    try:
        gray = image.convert("L")
        reduced = gray.resize(
            (max(8, min(48, gray.width // 2)), max(6, min(24, gray.height // 2))),
            getattr(Image, "Resampling", Image).BILINEAR,
        )
        return reduced.size, hash(reduced.tobytes())
    except Exception:
        return None


def _pickup_lines_need_detection(lines: list[tuple[str, float]]) -> bool:
    """Require feed detection for missing or structurally weak money text."""
    parsed = [text for text, _ in lines if parse_mesos_amount(text) is not None]
    return not parsed or any(mesos_text_needs_full_detection(text) for text in parsed)


def _clean_context_text(value: str) -> str:
    text = re.sub(r"\s+", "", value).strip("|:：·.,。()[]【】")
    return text[:32]


def _normalize_context_text(value: str, *, kind: str) -> str:
    """Normalize the few stable Traditional/Simplified OCR substitutions."""
    text = _clean_context_text(value)
    if kind == "map":
        # A shifted crop can capture the mini-map tab caption instead of the
        # second line.  These strings are UI chrome, never a real map name;
        # rejecting them is safer than allowing them into the stable candidate
        # streak and the drop lookup request.
        if "小地圖" in text or "小地图" in text:
            return ""
        if text in {"國地圖", "国地图", "地圖國", "地图国"}:
            return ""
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
            candidates = complete
    # A single enlarged/detection retry can still return a plausible but wrong
    # CJK string.  Prefer the mode across all views, with the most recent
    # candidate breaking a tie, instead of selecting the longest or last OCR
    # result unconditionally.
    counts = Counter(candidates)
    highest = max(counts.values())
    winners = {candidate for candidate, count in counts.items() if count == highest}
    for candidate in reversed(candidates):
        if candidate in winners:
            return candidate
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
        # The job crop also contains the orange LV badge on its left.  The
        # numeric model naturally wins that larger badge and returns ``68``
        # instead of the small class label.  Focus on the right-hand text and
        # use the general text reader, which keeps this correction proportional
        # across client sizes and does not hard-code a class name.
        job_reader = getattr(ocr, "read_text_field", None)
        if not callable(job_reader):
            job_reader = ocr.read_field
        job_focus = image.crop((
            max(0, round(image.width * 0.22)),
            0,
            image.width,
            image.height,
        ))
        try:
            from PIL import Image
            resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC", 3)
            enlarged = job_focus.resize(
                (job_focus.width * 8, job_focus.height * 8),
                resampling,
            )
            job_lines.append(job_reader(enlarged))
        except Exception:
            pass
        try:
            job_lines.append(job_reader(job_focus))
        except Exception:
            pass
        # Do not pay for detection when the focused recognition crop already
        # produced a usable class.  This keeps background context available
        # quickly after startup; detection remains the fallback for clients
        # whose tiny class label is not readable in the focused view.
        if not _context_candidates(job_lines, kind="job"):
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
        aux_scan_ms: int = 200,
        pickup_interval_ms: int = 200,
        context_scan_ms: int = 3000,
    ) -> None:
        self.source = source
        self.ocr = ocr
        self.status_queue: queue.Queue[StatusReading] = queue.Queue(maxsize=24)
        self.auxiliary_queue: queue.Queue[AuxiliaryReading] = queue.Queue(maxsize=4)
        self.context_queue: queue.Queue[ContextReading] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._status_enabled = threading.Event()
        self._aux_enabled = threading.Event()
        # Potion and pickup scans have separate cadences. Sharing one event
        # lets either worker clear the other's wake-up request, which makes a
        # Start/Resume refresh nondeterministic.
        self._potion_request = threading.Event()
        self._pickup_request = threading.Event()
        self._context_request = threading.Event()
        self._lock = threading.Lock()
        # RapidOCR/ONNX objects are shared by status, pickup and context
        # workers. Their public wrapper is not safe for concurrent inference;
        # serialize only the OCR section while keeping screen capture and Tk
        # queue delivery off the UI thread.
        self._ocr_lock = threading.RLock()
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
        self._pickup_feed_signature: tuple | None = None
        self._pickup_detection_signature: tuple | None = None
        self._pickup_detected_lines: list[tuple[str, float]] = []
        self._pickup_line_signatures: dict[str, tuple | None] = {}
        self._pickup_line_values: dict[str, str] = {}
        self._bar_flash_detector = BarFlashDetector()

    def start(self) -> None:
        if self._threads:
            return
        self._threads = [
            threading.Thread(
                target=self._worker_entry,
                args=(self._status_loop, "status"),
                name="maple-status-monitor",
                daemon=True,
            ),
            threading.Thread(
                target=self._worker_entry,
                args=(self._auxiliary_loop, "economy"),
                name="maple-economy-monitor",
                daemon=True,
            ),
            threading.Thread(
                target=self._worker_entry,
                args=(self._pickup_loop, "pickup"),
                name="maple-pickup-monitor",
                daemon=True,
            ),
            threading.Thread(
                target=self._worker_entry,
                args=(self._context_loop, "context"),
                name="maple-context-monitor",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def _worker_entry(self, worker, label: str) -> None:
        """Keep one unexpected worker exception from killing monitoring.

        Each loop already handles expected capture/OCR failures locally. This
        outer boundary is for regressions in code outside those inner blocks
        (queue delivery, timing, third-party image objects, or a future edit).
        It logs the complete traceback and retries after a short backoff while
        the user can still see the HUD and stop the session normally.
        """
        if label == "context" and not callable(getattr(self.source, "grab_context", None)):
            # Older/custom capture adapters may not expose the optional
            # background map/job surface.  This worker is optional; do not
            # turn that normal capability gap into an endless crash-log loop.
            return
        while not self._stop.is_set():
            try:
                worker()
            except Exception as exc:
                log_exception(f"monitor worker crashed: {label}", exc)
                self._stop.wait(0.5)
                continue
            if self._stop.is_set():
                return
            # A monitor loop should only return after stop. Treat an
            # unexpected return as recoverable instead of silently losing live
            # OCR for the rest of the session.
            log_exception(
                f"monitor worker exited unexpectedly: {label}",
                RuntimeError("worker returned before monitor stop was requested"),
            )
            self._stop.wait(0.5)

    def stop(self) -> None:
        self._stop.set()
        self._status_enabled.set()
        self._aux_enabled.set()
        for thread in self._threads:
            thread.join(timeout=0.8)
        self._threads.clear()

    def set_sample_interval(self, value_ms: int) -> None:
        with self._lock:
            self._sample_interval_ms = max(200, min(1000, int(value_ms)))

    def set_status_enabled(self, enabled: bool) -> None:
        """Run high-frequency status OCR only while a session is active.

        Context OCR remains independent so map/job detection is available
        before Start. Keeping LV/HP/MP/EXP idle until then removes a large
        source of startup and idle CPU contention without changing the last
        displayed snapshot.
        """
        if enabled:
            self._status_enabled.set()
        else:
            self._status_enabled.clear()

    def set_pickup_interval(self, value_ms: int) -> None:
        with self._lock:
            self._pickup_interval_ms = max(PICKUP_SCAN_MIN_MS, min(1000, int(value_ms)))

    def set_aux_enabled(self, enabled: bool) -> None:
        if enabled:
            self._aux_enabled.set()
            self._potion_request.set()
            self._pickup_request.set()
        else:
            self._aux_enabled.clear()
            self._potion_request.clear()
            self._pickup_request.clear()

    def request_auxiliary_scan(self) -> None:
        """Wake the economy worker for a fresh Start/Resume baseline."""
        self._potion_request.set()
        self._pickup_request.set()

    def request_context(self) -> None:
        self._context_request.set()

    def reset_bar_flash_detection(self) -> None:
        """Start a fresh visual baseline after Start/Resume."""
        self._bar_flash_detector.reset()

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
            # The geometry still defines all eight cells, but OCR/accounting
            # must never scan an unconfigured cell.  An empty configuration is
            # an explicit "do not track shortcut potions" state, not a reason
            # to fall back to all eight cells.
            self._potion_slots = configured

    def _status_loop(self) -> None:
        while not self._stop.is_set():
            if not self._status_enabled.wait(0.1):
                continue
            started = time.perf_counter()
            bar_flash: tuple[str, ...] = ()
            try:
                try:
                    field_images = self.source.grab_fields(include_bar_signals=True)
                except TypeError:
                    # Keep lightweight/custom capture sources compatible with
                    # the original four-field method signature.
                    field_images = self.source.grab_fields()
                ocr_images = {
                    name: image
                    for name, image in field_images.items()
                    if not name.startswith("__bar_")
                }
                bar_images = {
                    name.removeprefix("__bar_"): image
                    for name, image in field_images.items()
                    if name.startswith("__bar_")
                }
                bar_flash = self._bar_flash_detector.update(bar_images)
                with self._ocr_lock:
                    read_fields = getattr(self.ocr, "read_fields", None)
                    if callable(read_fields):
                        field_text = read_fields(ocr_images)
                    else:
                        field_text = {
                            name: self.ocr.read_field(image)
                            for name, image in ocr_images.items()
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
                    bar_flash=bar_flash,
                ),
            )
            with self._lock:
                interval = self._sample_interval_ms / 1000
            remaining = max(0.01, interval - (time.perf_counter() - started))
            self._stop.wait(remaining)

    def _auxiliary_loop(self) -> None:
        next_potion_scan = 0.0
        while not self._stop.is_set():
            if not self._aux_enabled.wait(0.1):
                continue
            now = time.monotonic()
            requested = self._potion_request.is_set()
            with self._lock:
                track_potions = self._track_potions
                potion_interval = self._aux_scan_ms / 1000
                configured_slots = self._configured_potion_slots
            if not track_potions or not configured_slots:
                self._stop.wait(0.1)
                continue
            if not requested and now < next_potion_scan:
                self._stop.wait(min(0.1, next_potion_scan - now))
                continue
            self._potion_request.clear()
            next_potion_scan = now + potion_interval
            try:
                with self._lock:
                    slots = self._potion_slots
                regions = self.source.grab_auxiliary()
                # Quantity changes are the hard real-time signal: a potion
                with self._ocr_lock:
                    counts = self._read_potion_counts(regions, configured_slots, slots)
                # Publish quantity OCR immediately. Pickup detection is
                # intentionally handled by _pickup_loop because its full
                # detector can take hundreds of milliseconds on CPU-only
                # machines. A slow money retry must never delay a 0.2-0.3s
                # shortcut sample or make the quantity appear stale.
                _put_latest(
                    self.auxiliary_queue,
                    AuxiliaryReading(
                        counts=counts,
                        timestamp=time.monotonic(),
                        pickup_scanned=False,
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

    def _read_potion_counts(
        self,
        regions: dict[str, Any],
        configured_slots: tuple[PotionSlotConfig, ...],
        slots: tuple[PotionSlotConfig, ...],
    ) -> dict[str, int]:
        """Read only shortcut quantities for the high-frequency worker."""
        # The settings page is the source of truth for which cells contain
        # trackable potions.  Do not send the complete bar (or every cell crop)
        # to OCR when the user has not enabled a row yet.
        if not configured_slots:
            return {}
        read_shortcut_counts = getattr(self.ocr, "read_shortcut_counts", None)
        if callable(read_shortcut_counts) and regions.get("shortcut") is not None:
            # Pass only the cells explicitly enabled in Settings.  ``slots``
            # is retained in the signature for adapters created before this
            # rule was introduced, but it is intentionally not used as a
            # fallback source of OCR work.
            observed_slots = configured_slots
            configured_ids = {slot.slot for slot in observed_slots}
            blue_ids = {
                slot.slot
                for slot in observed_slots
                if slot.kind in ("mp", "both")
            }
            slot_images = {
                slot.slot: regions[f"shortcut:{slot.slot}"]
                for slot in observed_slots
                if regions.get(f"shortcut:{slot.slot}") is not None
            }
            try:
                try:
                    detected_counts = read_shortcut_counts(
                        regions["shortcut"], configured_ids, blue_ids,
                        allow_full_validation=False,
                        slot_images=slot_images,
                    )
                except TypeError:
                    try:
                        # Compatibility with adapters that have the newer
                        # slot arguments but not the direct-cell keyword.
                        detected_counts = read_shortcut_counts(
                            regions["shortcut"], configured_ids, blue_ids,
                            allow_full_validation=False,
                        )
                    except TypeError:
                        # Compatibility with older adapters that expose only
                        # the positional slot arguments.
                        detected_counts = read_shortcut_counts(
                            regions["shortcut"], configured_ids, blue_ids
                        )
            except TypeError:
                # Compatibility with custom OCR adapters that still expose
                # the original one/two-argument method.
                try:
                    detected_counts = read_shortcut_counts(
                        regions["shortcut"], configured_ids
                    )
                except TypeError:
                    detected_counts = read_shortcut_counts(regions["shortcut"])
            enabled_slots = {slot.slot for slot in slots if slot.enabled}
            return {
                slot_id: count
                for slot_id, count in detected_counts.items()
                if slot_id in enabled_slots
            }

        counts: dict[str, int] = {}
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
        return counts

    def _pickup_loop(self) -> None:
        """Track the pickup feed independently from high-frequency potions."""
        next_pickup_scan = 0.0
        while not self._stop.is_set():
            if not self._aux_enabled.wait(0.1):
                continue
            now = time.monotonic()
            requested = self._pickup_request.is_set()
            with self._lock:
                track_pickup = self._track_pickup
                pickup_interval = self._pickup_interval_ms / 1000
            if not track_pickup:
                self._stop.wait(0.1)
                continue
            if not requested and now < next_pickup_scan:
                self._stop.wait(min(0.1, next_pickup_scan - now))
                continue
            self._pickup_request.clear()
            next_pickup_scan = now + pickup_interval
            try:
                regions = self.source.grab_auxiliary()
                with self._ocr_lock:
                    lines = self._read_pickup_lines(regions, now)
                _put_latest(
                    self.auxiliary_queue,
                    AuxiliaryReading(
                        lines=tuple(lines),
                        timestamp=time.monotonic(),
                        pickup_scanned=True,
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
        feed_signature = _image_signature(regions.get("pickup"))
        feed_changed = feed_signature != self._pickup_feed_signature
        self._pickup_feed_signature = feed_signature

        # A full notification detector is the expensive fallback. Reuse its
        # result while the same toast stack remains on screen instead of
        # rerunning detection every 350ms and starving shortcut OCR.
        if (
            not feed_changed
            and feed_signature is not None
            and self._pickup_detection_signature == feed_signature
        ):
            return list(self._pickup_detected_lines)

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
        read_text_field = getattr(self.ocr, "read_text_field", None)
        current_line_signatures = {
            line_id: _image_signature(image)
            for line_id, image in line_images
        }
        lines: list[tuple[str, float]] = []
        for line_id, image in line_images:
            signature = current_line_signatures.get(line_id)
            if (
                self._pickup_line_signatures.get(line_id) == signature
                and line_id in self._pickup_line_values
            ):
                text = self._pickup_line_values[line_id]
            else:
                text = (
                    read_text_field(image)
                    if callable(read_text_field)
                    else self.ocr.read_field(image)
                )
                self._pickup_line_values[line_id] = text
            lines.append(
                (
                    text,
                    int(line_id) * PICKUP_LINE_HEIGHT
                    + PICKUP_LINE_TOP_OFFSET
                    + PICKUP_LINE_HEIGHT / 2,
                )
            )
        self._pickup_line_signatures = current_line_signatures
        self._pickup_line_values = {
            line_id: self._pickup_line_values.get(line_id, "")
            for line_id in current_line_signatures
        }

        needs_detection = _pickup_lines_need_detection(lines)
        detection_already_attempted = (
            feed_signature is not None
            and self._pickup_detection_signature == feed_signature
        )
        if (
            needs_detection
            and feed_signature is not None
            and not detection_already_attempted
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
            self._pickup_detection_signature = feed_signature
            self._pickup_detected_lines = list(lines)
        elif (
            not line_images
            and regions.get("pickup") is not None
            and self._image_has_content(regions["pickup"])
            and feed_signature is not None
            and not detection_already_attempted
            and now >= self._next_pickup_detection
        ):
            detected: list[Any] = []
            for key in ("pickup", "pickup_wide"):
                image = regions.get(key)
                if image is None:
                    continue
                detected.extend(self.ocr.read_lines(image))
                if any(parse_mesos_amount(_line_text(line)) is not None for line in detected):
                    break
            lines = [(_line_text(line), float(getattr(line, "y", 0) or 0)) for line in detected]
            self._pickup_detection_signature = feed_signature
            self._pickup_detected_lines = list(lines)
        elif feed_changed:
            # A new frame without a completed detector retry is intentionally
            # represented by the cheap row reads. Never expose a stale full
            # detector result from the previous toast stack.
            self._pickup_detected_lines = []
        return lines

    def _context_loop(self) -> None:
        grab_context = getattr(self.source, "grab_context", None)
        if not callable(grab_context):
            return
        while not self._stop.is_set():
            try:
                regions = grab_context()
                with self._ocr_lock:
                    reading = extract_context(self.ocr, regions)
                _put_latest(self.context_queue, reading)
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
