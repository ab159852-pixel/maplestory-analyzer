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
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image, ImageChops, ImageOps

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
# A changed notification frame is tiny compared with a full game capture.  A
# longer queue preserves the exact 100-200ms sequence while status/potion OCR
# briefly owns the shared model; unchanged frames are coalesced below so this
# does not grow continuously while the feed is idle.
PICKUP_FRAME_QUEUE_SIZE = 32
PICKUP_RESULT_QUEUE_SIZE = 64
PICKUP_UNCHANGED_RETRY_SECONDS = 0.30
PICKUP_VISUAL_CACHE_SIZE = 96
PICKUP_REFERENCE_FEED_HEIGHT = 195
# The potion and pickup workers share one screen grab.  A very small cache
# prevents both workers from asking WGC/mss for the same frame when their
# deadlines overlap, without making the quantity older than one redraw.
AUX_CAPTURE_CACHE_SECONDS = 0.05
# The EXP percentage is rounded to two decimals. Keep the display guard as
# loose as Session.EXP_TOTAL_BAND, but prevent a structurally valid OCR frame
# from replacing a good same-level value with an impossible total.
EXP_DISPLAY_TOTAL_BAND = 0.25


def _lower_current_worker_priority() -> None:
    """Keep OCR workers from competing with desktop/game input on Windows."""
    try:
        import ctypes
        import sys

        if sys.platform != "win32":
            return
        # THREAD_PRIORITY_BELOW_NORMAL = -1. The OCR worker still runs at the
        # configured cadence, but Windows can schedule mouse/UI work first
        # when the game is minimized or another app is in the foreground.
        ctypes.windll.kernel32.SetThreadPriority(
            ctypes.windll.kernel32.GetCurrentThread(),
            -1,
        )
    except Exception:
        # Priority is an optimization only; it must never prevent monitoring.
        return


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
    # One scan already contains several independent recognition views.  Let
    # the UI publish a value immediately when those views corroborate it (or
    # when it contains an unambiguous numbered/floor marker), while retaining
    # the cross-scan stability gate for a lone generic OCR guess.
    map_confirmed: bool = False
    job_confirmed: bool = False
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
# MapleStory appends an ASCII/Unicode Roman numeral to several map names.  A
# recognition pass over the narrow mini-map crop can drop that suffix, so it
# must be treated as a stronger candidate than the same map name without it.
_MAP_ROMAN_SUFFIX_RE = re.compile(r"[IVX]+$", re.IGNORECASE)
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


def _pickup_capture_signature(image: Any) -> tuple | None:
    """Sign only the right-side toast surface, excluding animated map pixels."""
    try:
        width, height = image.size
        left = max(0, min(width - 1, round(width * 0.12)))
        return _image_signature(image.crop((left, 0, width, height)))
    except Exception:
        return _image_signature(image)


def _pickup_row_fingerprint(image: Any) -> tuple | None:
    """Return a position-independent fingerprint for one rendered toast row.

    The same message moves upward whenever a newer pickup arrives.  Hashing by
    fixed row number therefore re-ran OCR for every visible line on every
    scroll.  A thresholded glyph crop stays identical after that movement and
    lets the worker OCR only the genuinely new row.
    """
    try:
        gray = ImageOps.grayscale(image)
        mask = gray.point(lambda value: 255 if value >= 100 else 0)
        bounds = mask.getbbox()
        if bounds is None:
            return None
        glyphs = mask.crop(bounds)
        return glyphs.size, glyphs.tobytes()
    except Exception:
        return _image_signature(image)


def _segment_pickup_money_rows(image: Any) -> tuple[list[tuple[Any, float]], bool]:
    """Locate white pickup rows without invoking the OCR detector.

    MapleStory renders bonus/EXP notifications in yellow and the mesos pickup
    toast in white on the right-side black surface.  Horizontal projection is
    both faster and more accurate than twelve fixed 16px crops: the live font
    is spaced about 14px apart and scales with the game viewport.  ``bool`` is
    false only when the pixels do not resemble a line stack, allowing legacy
    adapters and synthetic tests to retain the old fixed-row fallback.
    """
    try:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width < 16 or height < 16:
            return [], False
    except Exception:
        return [], False

    scale = max(0.5, height / PICKUP_REFERENCE_FEED_HEIGHT)
    left = max(0, min(width - 1, round(width * 0.12)))
    min_bright = max(4, round(width * 0.014))
    # Let Pillow perform colour projection in native code.  The old Python
    # pixel loop was already faster than OCR at 1366x768, but at 2K it could
    # consume 50-70ms by itself and erode the 100ms capture budget.
    _hue, saturation, value = rgb.convert("HSV").split()
    bright_mask = value.point(lambda pixel: 255 if pixel >= 120 else 0)
    neutral_mask = saturation.point(lambda pixel: 255 if pixel <= 100 else 0)
    white_mask = ImageChops.multiply(bright_mask, neutral_mask)
    surface_width = width - left
    resampling = getattr(Image, "Resampling", Image)

    def projected_counts(mask: Any) -> list[int]:
        projection = mask.crop((left, 0, width, height)).resize(
            (1, height), resampling.BOX
        )
        get_values = getattr(projection, "get_flattened_data", projection.getdata)
        return [round(value * surface_width / 255) for value in get_values()]

    bright_counts = projected_counts(bright_mask)
    white_counts = projected_counts(white_mask)
    row_stats = list(zip(bright_counts, white_counts))
    active_rows = [
        y for y, bright in enumerate(bright_counts)
        if bright >= min_bright
    ]

    # A genuinely empty feed is a valid, confident observation.
    if not active_rows:
        return [], True

    max_gap = max(1, round(1.5 * scale))
    bands: list[list[int]] = []
    for y in active_rows:
        if not bands or y - bands[-1][-1] > max_gap + 1:
            bands.append([y])
        else:
            bands[-1].append(y)

    min_height = max(3, round(5 * scale))
    max_height = max(min_height, round(18 * scale))
    pad = max(2, round(2 * scale))
    valid_band_seen = False
    candidates: list[tuple[Any, float]] = []
    for band in bands:
        top = band[0]
        bottom = band[-1]
        band_height = bottom - top + 1
        if not min_height <= band_height <= max_height:
            continue
        valid_band_seen = True
        bright = sum(row_stats[y][0] for y in band)
        white = sum(row_stats[y][1] for y in band)
        # White money text is effectively 100% low-saturation in observed
        # game frames; 55% leaves room for antialiasing and capture scaling.
        if white < max(4, round(min_bright * scale)) or white / max(1, bright) < 0.55:
            continue
        crop_top = max(0, top - pad)
        crop_bottom = min(height, bottom + pad + 1)
        candidates.append((rgb.crop((0, crop_top, width, crop_bottom)), (top + bottom) / 2))

    if not valid_band_seen:
        return [], False
    return candidates, True


def _explicit_non_mesos_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text)).lower()
    return any(token in compact for token in (
        "經驗", "经验", "經值", "验值", "exp", "bonus", "增益", "通行證", "通行证",
    ))


def _canonical_mesos_text(amount: int) -> str:
    return f"獲取楓幣。(+{amount})"


def _normalize_pickup_row_for_ocr(image: Any) -> Any:
    """Normalize scaled game fonts to the recognizer's fast 32px height."""
    try:
        width, height = image.size
        if height <= 0 or 28 <= height <= 36:
            return image
        target_height = 32
        target_width = max(1, round(width * target_height / height))
        resampling = getattr(Image, "Resampling", Image)
        return image.resize((target_width, target_height), resampling.LANCZOS)
    except Exception:
        return image


def _pickup_lines_need_detection(lines: list[tuple[str, float]]) -> bool:
    """Require feed detection for missing or structurally weak money text."""
    parsed = [text for text, _ in lines if parse_mesos_amount(text) is not None]
    return not parsed or any(mesos_text_needs_full_detection(text) for text in parsed)


def _clean_context_text(value: str) -> str:
    text = re.sub(r"\s+", "", value).strip("|:：·.,。()[]【】")
    return text[:32]


def _normalize_context_text(value: str, *, kind: str) -> str:
    """Normalize the few stable Traditional/Simplified OCR substitutions."""
    # NFKC converts the game's full-width/Unicode Roman floor glyphs (for
    # example Ⅱ) into the ASCII form used by the drop database (II), while
    # leaving the Traditional Chinese map title intact.
    text = unicodedata.normalize("NFKC", _clean_context_text(value))
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


def _has_explicit_map_floor(value: str) -> bool:
    """Return whether a map candidate keeps its visible Roman floor suffix."""
    compact = _compact_map_text(value)
    return len(_CJK_RE.findall(compact)) >= 2 and bool(_MAP_ROMAN_SUFFIX_RE.search(compact))


def _has_strong_map_candidate(candidates: list[str]) -> bool:
    """Return whether OCR found a complete numbered map or explicit floor."""
    return _has_complete_map_candidate(candidates) or any(
        _has_explicit_map_floor(candidate) for candidate in candidates
    )


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
        else:
            # When both views produce ``寺院通道`` and one view produces the
            # actual ``寺院通道II``, a plain mode vote would systematically
            # discard the floor marker.  Prefer the explicit floor candidate;
            # it carries information the shorter candidate demonstrably lost.
            with_floor = [
                candidate for candidate in candidates
                if _has_explicit_map_floor(candidate)
            ]
            if with_floor:
                candidates = with_floor
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


def _context_candidate_is_confirmed(
    candidates: list[str],
    selected: str | None,
    *,
    kind: str,
) -> bool:
    """Return whether one scan has enough evidence to publish immediately."""
    if not selected:
        return False
    if kind == "map":
        # A numbered barracks name or explicit Roman floor cannot be produced
        # by merely dropping a glyph from a neighbouring label.  These are the
        # two real tiny-map cases where waiting another 2s made drop lookup feel
        # broken even though the first OCR pass was already complete.
        return _has_strong_map_candidate([selected])
    if Counter(candidates).get(selected, 0) >= 2:
        return True
    # The alias table is intentionally narrow and currently canonicalizes only
    # the repeatedly observed thief label.  A recognized canonical alias is
    # stronger than an arbitrary two-CJK-glyph string from a single view.
    return selected in set(_JOB_ALIASES.values())


def extract_context(ocr: Any, regions: dict[str, Any]) -> ContextReading:
    """Best-effort extraction from the fixed visible map/status labels."""
    map_lines: list[object] = []
    map_images = [
        regions.get("map"),
        regions.get("map_wide"),
    ]
    map_images = [image for image in map_images if image is not None]
    # The supplied 2K game frame preserves ``寺院通道Ⅱ`` in the focused native
    # crop, while enlarging it turns the floor marker into ``川`` or removes it.
    # Read that one cheap native crop first and stop immediately when it already
    # carries a numbered map/floor suffix.  Enlarged/wide retries remain for
    # clients whose native text is too small (for example 第3軍營).
    native_map_images_read = 0
    if map_images:
        try:
            map_lines.append(ocr.read_field(map_images[0]))
            native_map_images_read = 1
        except Exception:
            pass

    map_candidates = _context_candidates(map_lines, kind="map")
    if not _has_strong_map_candidate(map_candidates):
        for image in map_images:
            try:
                from PIL import Image
                resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC", 3)
                enlarged = image.resize((image.width * 8, image.height * 8), resampling)
                map_lines.append(ocr.read_field(enlarged))
            except Exception:
                pass
            map_candidates = _context_candidates(map_lines, kind="map")
            if _has_strong_map_candidate(map_candidates):
                break

    map_candidates = _context_candidates(map_lines, kind="map")
    if not _has_strong_map_candidate(map_candidates):
        # The wider native crop is a cheap secondary vote for generic map
        # names.  Do not repeat the focused native crop already read above.
        for image in map_images[native_map_images_read:]:
            try:
                map_lines.append(ocr.read_field(image))
            except Exception:
                pass
            map_candidates = _context_candidates(map_lines, kind="map")
            if _has_strong_map_candidate(map_candidates):
                break
    if not _has_strong_map_candidate(map_candidates) and map_images:
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
    if not _has_strong_map_candidate(map_candidates) and map_images:
        # Detection is the final confirmation path.  It runs only when the
        # cheap recognition passes did not find a complete map token.
        for image in reversed(map_images):
            try:
                map_lines.extend(ocr.read_lines(image))
            except Exception:
                pass
            map_candidates = _context_candidates(map_lines, kind="map")
            if _has_strong_map_candidate(map_candidates):
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
        # Native-first is both faster and more accurate on the supplied 2K
        # frame (俠盜); the enlarged view also captures the character name and
        # costs another ~650ms.  Keep it only as a fallback for tiny clients.
        try:
            job_lines.append(job_reader(job_focus))
        except Exception:
            pass
        if not _context_candidates(job_lines, kind="job"):
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
    selected_map = _select_context_candidate(maps, kind="map")
    selected_job = _select_context_candidate(jobs, kind="job")
    return ContextReading(
        map_name=selected_map,
        job_name=selected_job,
        map_confirmed=_context_candidate_is_confirmed(
            maps,
            selected_map,
            kind="map",
        ),
        job_confirmed=_context_candidate_is_confirmed(
            jobs,
            selected_job,
            kind="job",
        ),
    )


class BackgroundMonitor:
    """Own the expensive capture/OCR work outside Tk's main thread."""

    def __init__(
        self,
        source: MonitorSource,
        ocr: Any,
        *,
        sample_interval_ms: int = 300,
        aux_scan_ms: int = 150,
        pickup_interval_ms: int = 200,
        context_scan_ms: int = 3000,
    ) -> None:
        self.source = source
        self.ocr = ocr
        self.status_queue: queue.Queue[StatusReading] = queue.Queue(maxsize=24)
        # Pickup snapshots are event-bearing, unlike status frames where only
        # the newest value matters.  Keep enough tiny OCR results for the
        # 300ms Tk drain to consume a burst without evicting intermediate
        # messages that represent real income.
        self.auxiliary_queue: queue.Queue[AuxiliaryReading] = queue.Queue(
            maxsize=PICKUP_RESULT_QUEUE_SIZE
        )
        # Potion readings must not be evicted by the much noisier pickup-feed
        # stream.  The UI drains both queues in timestamp order.
        self.potion_queue: queue.Queue[AuxiliaryReading] = queue.Queue(maxsize=8)
        self.context_queue: queue.Queue[ContextReading] = queue.Queue(maxsize=2)
        # Screen sampling and pickup OCR must be independent.  At 100-200ms a
        # detector pass can outlive the next sample deadline; retaining a
        # short sequence of already-captured feed frames prevents those OCR
        # milliseconds from becoming blind time.
        self._pickup_frame_queue: queue.Queue[tuple[float, dict[str, Any]]] = (
            queue.Queue(maxsize=PICKUP_FRAME_QUEUE_SIZE)
        )
        self._stop = threading.Event()
        self._status_enabled = threading.Event()
        # Stopped/paused readouts are display-only: they can show a bounded
        # target-window frame while the game is idle, but a running session
        # must continue to receive only new WGC presentations for accounting.
        self._status_allow_stale = False
        self._aux_enabled = threading.Event()
        # The stopped/paused shortcut inventory is a display baseline, not a
        # billable signal. It may use a recent target frame so the user sees
        # the actual quantities before Start; active potion/pickup accounting
        # always clears this flag and waits for a new presentation.
        self._auxiliary_allow_stale = False
        self._pickup_enabled = threading.Event()
        # Potion and pickup scans have separate cadences. Sharing one event
        # lets either worker clear the other's wake-up request, which makes a
        # Start/Resume refresh nondeterministic.
        self._potion_request = threading.Event()
        self._pickup_request = threading.Event()
        self._context_request = threading.Event()
        # Startup/manual context scans briefly take priority over status and
        # pickup OCR so map/job lookup cannot starve behind 0.15-0.3s workers.
        # Potion quantity remains the highest-priority billing signal.
        self._context_priority = threading.Event()
        self._context_priority.set()
        self._potion_scan_active = threading.Event()
        self._aux_capture_lock = threading.Lock()
        self._aux_capture_cache: tuple[float, bool, dict[str, Any]] | None = None
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
        self._pickup_visual_text_cache: dict[tuple, str] = {}
        self._pickup_capture_frame_signature: tuple | None = None
        self._pickup_capture_last_queued_at = 0.0
        self._pickup_scan_confident = True
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
                args=(self._pickup_capture_loop, "pickup-capture"),
                name="maple-pickup-capture",
                daemon=True,
            ),
            threading.Thread(
                target=self._worker_entry,
                args=(self._pickup_loop, "pickup-ocr"),
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
        _lower_current_worker_priority()
        if label == "context" and not callable(getattr(self.source, "grab_context", None)):
            # Older/custom capture adapters may not expose the optional
            # background map/job surface.  This worker is optional; do not
            # turn that normal capability gap into an endless crash-log loop.
            context_priority = getattr(self, "_context_priority", None)
            if context_priority is not None:
                context_priority.clear()
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

    def stop(self, *, total_timeout: float = 0.8) -> None:
        """Signal every worker and release capture handles within one budget."""
        self._stop.set()
        self._status_enabled.set()
        self._aux_enabled.set()
        self._pickup_enabled.set()
        self._potion_request.set()
        self._pickup_request.set()
        self._context_request.set()
        context_priority = getattr(self, "_context_priority", None)
        if context_priority is not None:
            context_priority.set()
        self._potion_scan_active.clear()
        deadline = time.monotonic() + max(0.0, float(total_timeout))

        # Native WGC/WinRT teardown can itself wait on a frame callback.  Run
        # it off the Tk thread and give it part of the same bounded budget;
        # closing the capture source also wakes workers currently blocked in a
        # frame request.
        close_source = getattr(self.source, "close", None)
        if callable(close_source):
            close_thread = threading.Thread(
                target=close_source,
                name="maple-capture-close",
                daemon=True,
            )
            close_thread.start()
            close_thread.join(timeout=min(0.25, max(0.0, deadline - time.monotonic())))

        for thread in self._threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
        self._threads.clear()

    def set_sample_interval(self, value_ms: int) -> None:
        with self._lock:
            self._sample_interval_ms = max(200, min(1000, int(value_ms)))

    def set_status_enabled(self, enabled: bool, *, allow_stale: bool = False) -> None:
        """Enable live status OCR, optionally in display-only stale mode."""
        with self._lock:
            self._status_allow_stale = bool(enabled and allow_stale)
        if enabled:
            self._status_enabled.set()
        else:
            self._status_enabled.clear()

    def set_pickup_interval(self, value_ms: int) -> None:
        with self._lock:
            self._pickup_interval_ms = max(PICKUP_SCAN_MIN_MS, min(1000, int(value_ms)))

    def set_aux_enabled(
        self,
        enabled: bool,
        *,
        allow_stale: bool = False,
        pickup_enabled: bool = True,
    ) -> None:
        with self._aux_capture_lock:
            # A cached stopped-state image must never cross the Start/Resume
            # boundary into a billable potion/pickup scan.
            self._aux_capture_cache = None
        with self._lock:
            self._auxiliary_allow_stale = bool(enabled and allow_stale)
        if enabled:
            self._aux_enabled.set()
            self._potion_request.set()
            if pickup_enabled:
                self._pickup_enabled.set()
                self._pickup_request.set()
            else:
                self._pickup_enabled.clear()
                self._pickup_request.clear()
                self._clear_pickup_frames()
        else:
            self._aux_enabled.clear()
            self._pickup_enabled.clear()
            self._potion_request.clear()
            self._pickup_request.clear()
            self._clear_pickup_frames()

    def request_auxiliary_scan(self) -> None:
        """Wake the economy worker for a fresh Start/Resume baseline."""
        self._potion_request.set()
        self._pickup_request.set()

    def request_context(self) -> None:
        self._context_priority.set()
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

    def _grab_auxiliary_cached(self, *, allow_stale: bool = False) -> dict[str, Any]:
        """Capture one short-lived auxiliary frame for both workers."""
        now = time.monotonic()
        with self._aux_capture_lock:
            cached = self._aux_capture_cache
            if (
                cached is not None
                and cached[1] == allow_stale
                and now - cached[0] <= AUX_CAPTURE_CACHE_SECONDS
            ):
                return cached[2]
            try:
                regions = self.source.grab_auxiliary(allow_stale=allow_stale)
            except TypeError:
                # Keep minimal/custom sources created before display-only
                # inventory support working; their original method remains a
                # fresh-frame implementation.
                regions = self.source.grab_auxiliary()
            self._aux_capture_cache = (time.monotonic(), allow_stale, regions)
            return regions

    def _status_loop(self) -> None:
        while not self._stop.is_set():
            if not self._status_enabled.wait(0.1):
                continue
            # A configured shortcut change is the billing signal and has a
            # shorter deadline than the status display.  Do not let a status
            # frame begin while the potion worker has a requested/active
            # sample; skipping one status frame is preferable to making a
            # 0.3-0.7s drink update wait behind HP/MP/EXP OCR.
            with self._lock:
                potion_priority = (
                    self._track_potions and bool(self._configured_potion_slots)
                )
                allow_stale = bool(getattr(self, "_status_allow_stale", False))
            if potion_priority and (
                self._potion_request.is_set() or self._potion_scan_active.is_set()
            ):
                self._stop.wait(0.01)
                continue
            context_priority = getattr(self, "_context_priority", None)
            if context_priority is not None and context_priority.is_set():
                self._stop.wait(0.01)
                continue
            started = time.perf_counter()
            bar_flash: tuple[str, ...] = ()
            try:
                try:
                    field_images = self.source.grab_fields(
                        include_bar_signals=True,
                        allow_stale=allow_stale,
                    )
                except TypeError:
                    # Keep lightweight/custom capture sources compatible with
                    # the original method signatures.
                    try:
                        field_images = self.source.grab_fields(
                            include_bar_signals=True
                        )
                    except TypeError:
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
                # A potion scan may have started while this frame was being
                # captured.  It owns the lock priority; discard this status
                # frame instead of waiting on the lock and delaying billing.
                if not self._ocr_lock.acquire(blocking=False):
                    self._stop.wait(0.01)
                    continue
                try:
                    read_fields = getattr(self.ocr, "read_fields", None)
                    if callable(read_fields):
                        field_text = read_fields(ocr_images)
                    else:
                        field_text = {
                            name: self.ocr.read_field(image)
                            for name, image in ocr_images.items()
                        }
                finally:
                    self._ocr_lock.release()
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
                display_only = bool(getattr(self, "_status_allow_stale", False))
            # Re-OCRing an unchanged status bar five times per second wastes
            # CPU before a session has even started. The value is still
            # visible promptly, and a running session always restores the
            # user's configured live cadence above.
            if display_only:
                interval = max(interval, 0.75)
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
                allow_stale = bool(getattr(self, "_auxiliary_allow_stale", False))
            if not track_potions or not configured_slots:
                self._potion_request.clear()
                self._stop.wait(0.1)
                continue
            if not requested and now < next_potion_scan:
                self._stop.wait(min(0.1, next_potion_scan - now))
                continue
            self._potion_request.clear()
            # Schedule from the *start* of this sample.  Scheduling after OCR
            # made the effective cadence ``configured interval + inference
            # time``; a 150ms setting could therefore update only every
            # 400-700ms and miss several bottle changes.  If OCR itself takes
            # longer than the interval, the next loop runs immediately.
            next_potion_scan = now + potion_interval
            self._potion_scan_active.set()
            try:
                with self._lock:
                    slots = self._potion_slots
                regions = self._grab_auxiliary_cached(allow_stale=allow_stale)
                # Quantity changes are the hard real-time signal: a potion
                with self._ocr_lock:
                    counts = self._read_potion_counts(regions, configured_slots, slots)
                # Publish quantity OCR immediately. Pickup detection is
                # intentionally handled by _pickup_loop because its full
                # detector can take hundreds of milliseconds on CPU-only
                # machines. A slow money retry must never delay a 0.2-0.3s
                # shortcut sample or make the quantity appear stale.
                _put_latest(
                    self.potion_queue,
                    AuxiliaryReading(
                        counts=counts,
                        timestamp=time.monotonic(),
                        pickup_scanned=False,
                    ),
                )
            except RuntimeError as exc:
                _put_latest(
                    self.potion_queue,
                    AuxiliaryReading(error=str(exc), timestamp=time.monotonic()),
                )
            except Exception as exc:
                _put_latest(
                    self.potion_queue,
                    AuxiliaryReading(error=f"OCR: {exc}", timestamp=time.monotonic()),
                )
            finally:
                self._potion_scan_active.clear()

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
            # Preserve Settings order when constructing the batch.  The OCR
            # result is keyed by slot, but deterministic ordering makes the
            # two-cell (for example 6/7) initial scan reproducible for engines
            # that return only the successfully decoded batch entries.
            configured_ids = tuple(slot.slot for slot in observed_slots)
            blue_ids = tuple(
                slot.slot
                for slot in observed_slots
                if slot.kind in ("mp", "both")
            )
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
                        live=True,
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

    def _clear_pickup_frames(self) -> None:
        self._pickup_capture_frame_signature = None
        self._pickup_capture_last_queued_at = 0.0
        frame_queue = getattr(self, "_pickup_frame_queue", None)
        if frame_queue is None:
            return
        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                return

    def _queue_pickup_frame(
        self,
        captured_at: float,
        regions: dict[str, Any],
    ) -> None:
        """Keep the newest feed sequence without blocking screen sampling."""
        item = (captured_at, regions)
        try:
            self._pickup_frame_queue.put_nowait(item)
            return
        except queue.Full:
            # A later toast stack contains the still-visible older entries as
            # well as the newest pickup.  Drop one oldest frame rather than
            # blocking capture behind OCR and losing the latest stack.
            try:
                self._pickup_frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._pickup_frame_queue.put_nowait(item)
        except queue.Full:
            pass

    def _pickup_capture_loop(self) -> None:
        """Capture pickup pixels at the configured cadence, independent of OCR."""
        next_pickup_scan = 0.0
        while not self._stop.is_set():
            if not self._pickup_enabled.wait(0.1):
                continue
            now = time.monotonic()
            requested = self._pickup_request.is_set()
            with self._lock:
                track_pickup = self._track_pickup
                pickup_interval = self._pickup_interval_ms / 1000
                allow_stale = bool(getattr(self, "_auxiliary_allow_stale", False))
            if not track_pickup:
                self._pickup_request.clear()
                self._clear_pickup_frames()
                self._stop.wait(0.1)
                continue
            if not requested and now < next_pickup_scan:
                self._stop.wait(min(0.05, next_pickup_scan - now))
                continue
            self._pickup_request.clear()
            next_pickup_scan = now + pickup_interval
            try:
                regions = self._grab_auxiliary_cached(allow_stale=allow_stale)
            except RuntimeError as exc:
                _put_latest(
                    self.auxiliary_queue,
                    AuxiliaryReading(
                        error=str(exc),
                        timestamp=time.monotonic(),
                        pickup_scanned=False,
                    ),
                )
            except Exception as exc:
                _put_latest(
                    self.auxiliary_queue,
                    AuxiliaryReading(
                        error=f"OCR: {exc}",
                        timestamp=time.monotonic(),
                        pickup_scanned=False,
                    ),
                )
            else:
                pickup = regions.get("pickup")
                signature = _pickup_capture_signature(pickup)
                last_signature = getattr(self, "_pickup_capture_frame_signature", None)
                last_queued_at = getattr(self, "_pickup_capture_last_queued_at", 0.0)
                should_queue = (
                    requested
                    or signature != last_signature
                    or now - last_queued_at >= PICKUP_UNCHANGED_RETRY_SECONDS
                )
                self._pickup_capture_frame_signature = signature
                if should_queue:
                    self._pickup_capture_last_queued_at = now
                    # The OCR worker now derives scaled rows directly from the
                    # feed image.  Do not retain shortcut cells and twelve
                    # duplicate row crops in a burst queue.
                    pickup_regions = {"pickup": pickup} if pickup is not None else regions
                    self._queue_pickup_frame(now, pickup_regions)

    def _pickup_loop(self) -> None:
        """OCR the captured pickup sequence without controlling its cadence."""
        pending: tuple[float, dict[str, Any]] | None = None
        while not self._stop.is_set():
            if not self._pickup_enabled.wait(0.1):
                pending = None
                continue
            with self._lock:
                track_pickup = self._track_pickup
            if not track_pickup:
                pending = None
                self._stop.wait(0.1)
                continue
            if pending is None:
                try:
                    pending = self._pickup_frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
            captured_at, regions = pending
            # A potion quantity frame has priority over the optional full
            # pickup detector.  The latter can invoke detector OCR and take
            # hundreds of milliseconds; never let it hold the shared model
            # lock in front of a configured shortcut cell.
            if self._potion_scan_active.is_set():
                self._stop.wait(0.01)
                continue
            context_priority = getattr(self, "_context_priority", None)
            if context_priority is not None and context_priority.is_set():
                # Preserve ``pending``; context gets one bounded recognition
                # turn, then this exact captured pickup frame resumes.
                self._stop.wait(0.01)
                continue
            if not self._ocr_lock.acquire(blocking=False):
                # Keep this captured frame pending.  Discarding it here would
                # recreate the exact blind spot the capture queue is meant to
                # remove whenever status/context OCR owns the shared model.
                self._stop.wait(0.01)
                continue
            try:
                try:
                    lines = self._read_pickup_lines(
                        regions,
                        captured_at,
                        allow_full_detection=self._pickup_frame_queue.empty(),
                    )
                finally:
                    self._ocr_lock.release()
                _put_latest(
                    self.auxiliary_queue,
                    AuxiliaryReading(
                        lines=tuple(lines),
                        timestamp=captured_at,
                        pickup_scanned=getattr(self, "_pickup_scan_confident", True),
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
            finally:
                pending = None

    def _read_pickup_lines(
        self,
        regions: dict[str, Any],
        now: float,
        *,
        allow_full_detection: bool = True,
    ) -> list[tuple[str, float]]:
        self._pickup_scan_confident = True
        feed_signature = _image_signature(regions.get("pickup"))
        feed_changed = feed_signature != self._pickup_feed_signature
        self._pickup_feed_signature = feed_signature

        dynamic_rows, segmentation_valid = _segment_pickup_money_rows(regions.get("pickup"))
        if segmentation_valid:
            return self._read_dynamic_pickup_rows(dynamic_rows)

        # A full notification detector is the expensive fallback. Reuse its
        # result while the same toast stack remains on screen instead of
        # rerunning detection every 350ms and starving shortcut OCR.
        if (
            not feed_changed
            and feed_signature is not None
            and self._pickup_detection_signature == feed_signature
            and self._pickup_detected_lines
            and any(
                parse_mesos_amount(text) is not None
                for text, _ in self._pickup_detected_lines
            )
        ):
            # An empty/garbled detector result is deliberately not cached as a
            # replacement for the fixed row reads below.  The old early return
            # did exactly that: one detector miss made every later pass return
            # an empty list until the whole feed image changed, so mesos
            # income appeared to stop even though the visible row was readable.
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
            cached_text = self._pickup_line_values.get(line_id, "")
            if (
                self._pickup_line_signatures.get(line_id) == signature
                and line_id in self._pickup_line_values
                and bool(str(cached_text).strip())
            ):
                text = cached_text
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
            and allow_full_detection
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
            detected_lines = [
                (_line_text(line), float(getattr(line, "y", 0) or 0))
                for line in detected
            ]
            self._pickup_detection_signature = feed_signature
            if any(parse_mesos_amount(text) is not None for text, _ in detected_lines):
                self._pickup_detected_lines = list(detected_lines)
                lines = detected_lines
            else:
                # Keep a weak but parseable fixed-row candidate when the
                # heavyweight detector failed to return a money line.  It is
                # safer than replacing real visible text with an empty frame.
                self._pickup_detected_lines = []
        elif (
            not line_images
            and regions.get("pickup") is not None
            and self._image_has_content(regions["pickup"])
            and feed_signature is not None
            and not detection_already_attempted
            and allow_full_detection
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

    def _read_dynamic_pickup_rows(
        self,
        rows: list[tuple[Any, float]],
    ) -> list[tuple[str, float]]:
        """Read only new white rows and reuse text after the stack scrolls."""
        cache = getattr(self, "_pickup_visual_text_cache", {})
        result: list[tuple[str, float]] = []
        uncertain = False
        for image, y in rows:
            fingerprint = _pickup_row_fingerprint(image)
            text = cache.get(fingerprint, "") if fingerprint is not None else ""
            if not text.strip():
                text = self._read_dynamic_pickup_row(image)
                if (
                    fingerprint is not None
                    and text.strip()
                    and (
                        not mesos_text_needs_full_detection(text)
                        or _explicit_non_mesos_text(text)
                    )
                ):
                    cache[fingerprint] = text

            amount = parse_mesos_amount(text)
            if amount is not None and not mesos_text_needs_full_detection(text):
                result.append((text, y))
            elif not _explicit_non_mesos_text(text):
                # Do not turn an OCR miss into a false empty-feed boundary.
                # The unchanged-frame retry will read this still-visible row
                # again in 300ms without making the event tracker forget it.
                uncertain = True

        while len(cache) > PICKUP_VISUAL_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        self._pickup_visual_text_cache = cache
        # A partially readable stack is not a complete feed boundary.  If one
        # visible money row is retained while another briefly fails OCR,
        # publishing only ``result`` makes MesosFeedTracker forget the missed
        # row; when it reappears on the next frame it is then counted twice.
        # Keep the previous visible state until every candidate row is either
        # parsed or explicitly known to be non-mesos text.
        self._pickup_scan_confident = not uncertain
        return result

    def _read_dynamic_pickup_row(self, image: Any) -> str:
        """Confirm one candidate row with two cheap, row-sized OCR passes."""
        normalized = _normalize_pickup_row_for_ocr(image)
        read_text_field = getattr(self.ocr, "read_text_field", None)
        first = (
            read_text_field(normalized)
            if callable(read_text_field)
            else self.ocr.read_field(normalized)
        )
        first_amount = parse_mesos_amount(first)
        if first_amount is not None and not mesos_text_needs_full_detection(first):
            return first
        if _explicit_non_mesos_text(first):
            return first

        # Detector OCR over the whole 286x195 feed costs hundreds of
        # milliseconds.  Over one already-located 15-30px row it is normally
        # ~20-30ms and preserves the complete right-aligned amount.
        try:
            detected = self.ocr.read_lines(normalized)
        except Exception:
            detected = []
        detected_texts = [_line_text(line) for line in detected if _line_text(line)]
        for candidate in detected_texts:
            amount = parse_mesos_amount(candidate)
            if amount is not None and not mesos_text_needs_full_detection(candidate):
                return candidate
        for candidate in detected_texts:
            amount = parse_mesos_amount(candidate)
            if amount is not None and first_amount == amount:
                # Recognition-only and detector paths independently agree on
                # the number.  Canonicalize the marker so this confirmed row
                # never falls into the expensive whole-feed fallback.
                return _canonical_mesos_text(amount)
        for candidate in detected_texts:
            if _explicit_non_mesos_text(candidate):
                return candidate
        return first

    def _context_loop(self) -> None:
        grab_context = getattr(self.source, "grab_context", None)
        if not callable(grab_context):
            return
        while not self._stop.is_set():
            context_complete = False
            try:
                regions = grab_context()
                # Context is useful before Start, but it is not a real-time
                # signal.  Never make the potion worker wait for a full CJK
                # detector pass just because a scheduled map refresh landed
                # at the same instant.
                priority_event = getattr(self, "_context_priority", None)
                priority = bool(priority_event is not None and priority_event.is_set())
                if self._potion_scan_active.is_set():
                    self._stop.wait(0.02)
                    continue
                acquired = (
                    self._ocr_lock.acquire(timeout=0.25)
                    if priority
                    else self._ocr_lock.acquire(blocking=False)
                )
                if not acquired:
                    self._stop.wait(0.02)
                    continue
                try:
                    reading = extract_context(self.ocr, regions)
                finally:
                    self._ocr_lock.release()
                context_complete = bool(reading.map_name and reading.job_name)
                if priority_event is not None:
                    if context_complete:
                        priority_event.clear()
                    else:
                        # Startup/drop lookup is still missing useful context.
                        # Keep one bounded-priority retry instead of falling
                        # into the normal 3s cadence after an empty OCR pass.
                        priority_event.set()
                _put_latest(self.context_queue, reading)
            except RuntimeError as exc:
                priority_event = getattr(self, "_context_priority", None)
                if priority_event is not None:
                    # WGC can recover after the game finishes initializing or
                    # the compositor restarts.  Retain the request and retry
                    # promptly; the target-only capture path never substitutes
                    # foreground HUD pixels while waiting.
                    priority_event.set()
                _put_latest(self.context_queue, ContextReading(error=str(exc)))
            except Exception as exc:
                priority_event = getattr(self, "_context_priority", None)
                if priority_event is not None:
                    priority_event.set()
                _put_latest(self.context_queue, ContextReading(error=f"OCR: {exc}"))
            retry_seconds = self._context_scan_ms / 1000
            if not context_complete:
                retry_seconds = min(retry_seconds, 1.0)
            deadline = time.monotonic() + retry_seconds
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
