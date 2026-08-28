"""Window capture abstraction.

`WindowCapture.grab_panel()` returns a PIL image of just the stat panel region.
Two implementations:

- `GameWindowCapture` (Windows only): finds the MapleStory window by title via
  pywin32 and prefers Windows Graphics Capture for the target HWND, so a
  foreground window cannot replace the game's pixels. It falls back to an
  `mss` desktop grab when Windows Graphics Capture is unavailable.
- `StaticImageCapture` (any platform): replays a single image file every call.
  Used for dev/testing on machines without the game running (e.g. this repo's
  Linux dev environment) -- proves out the OCR/parse/rate/overlay code without
  needing Windows + a live client.

Call `get_capture()` to get whichever is appropriate for the current platform;
pass `sample_path=` to force StaticImageCapture regardless of platform (useful
for demos/tests on Windows too).
"""
from __future__ import annotations

import contextlib
import sys
import threading
import time
from pathlib import Path
from typing import Protocol

from PIL import Image

from .regions import (
    AUXILIARY_BOXES,
    BAR_BOXES,
    CONTEXT_BOXES,
    Box,
    FIELD_BOXES,
    REFERENCE_WINDOW_SIZE,
    PICKUP_LINE_BOXES,
    SHORTCUT_BOX,
    STAT_PANEL_BOX,
    scale_box,
    scale_shortcut_box,
    scale_top_left_box,
    scale_window_box_to_client,
    scale_window_shortcut_box_to_client,
    scale_window_top_left_box_to_client,
    shortcut_slot_boxes_for_parent,
)


# Raised (as a RuntimeError message) when another window sits over the stat
# panel. Routine and recoverable, so it travels the same path as the
# minimized/not-found states -- see overlay._do_tick and _localize_error.
PANEL_OBSCURED = "stat panel is obscured"


def set_process_dpi_awareness() -> None:
    """Use physical pixels for Win32 client rectangles and captured frames.

    ``GetClientRect``/``ClientToScreen`` and Windows Graphics Capture must
    describe the same coordinate space.  A system-DPI-aware process can get
    logical coordinates while WGC returns physical pixels, which shifts every
    OCR crop on a monitor whose scale is not 100%.  Per-monitor-v2 is the
    strongest available context and the older shcore API is kept as a
    compatibility fallback for older Windows builds.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = (HANDLE)-4
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        pass


def _crop_frame_to_client(
    image: Image.Image,
    client_rect: tuple[int, int, int, int],
    window_rect: tuple[int, int, int, int],
    *,
    item_size: tuple[int, int] | None = None,
) -> Image.Image | None:
    """Convert a WGC frame into client-relative pixels.

    A ``GraphicsCaptureItem`` may expose either the client surface or the
    complete HWND depending on the Windows version/window class.  Never
    infer that from the frame *size* alone: the old implementation created a
    client-sized pool for a full-window item, so a same-sized frame could
    still contain a resized title bar.  The item size plus the actual HWND
    geometry lets us distinguish the two cases and scale the crop precisely.
    """
    client_left, client_top, client_right, client_bottom = client_rect
    window_left, window_top, window_right, window_bottom = window_rect
    client_width = client_right - client_left
    client_height = client_bottom - client_top
    window_width = window_right - window_left
    window_height = window_bottom - window_top
    if min(client_width, client_height, window_width, window_height) <= 0:
        return None
    client_size = (client_width, client_height)
    window_size = (window_width, window_height)

    # A client-only item is already in the coordinate system expected by
    # regions.py.  Normalize a rare pool-size mismatch without introducing a
    # second coordinate conversion.
    if item_size == client_size:
        if image.size == client_size:
            return image
        return image.resize(client_size, Image.Resampling.BILINEAR)

    # When there is no non-client frame, the direct-sized result is safe only
    # if the HWND itself has the same dimensions.  This intentionally rejects
    # the old ``full window resized to client size`` shortcut.
    if image.size == client_size and window_size == client_size:
        return image

    # A full-window WGC frame is anchored at the HWND's top-left.  Scale the
    # client offset independently on each axis so a DPI transition or a
    # compositor frame with a one-pixel rounding difference cannot drift the
    # right/bottom edge of the crop.
    if item_size not in (None, window_size) and image.size != window_size:
        return None
    scale_x = image.width / window_width
    scale_y = image.height / window_height
    crop_left = round((client_left - window_left) * scale_x)
    crop_top = round((client_top - window_top) * scale_y)
    crop_right = round((client_right - window_left) * scale_x)
    crop_bottom = round((client_bottom - window_top) * scale_y)
    if not (
        0 <= crop_left < crop_right <= image.width
        and 0 <= crop_top < crop_bottom <= image.height
    ):
        return None
    cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    if cropped.size != client_size:
        cropped = cropped.resize(client_size, Image.Resampling.BILINEAR)
    return cropped


def _pickup_boxes_for_client(window_size: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """Extend right-side pickup crops into the wide-client letterbox.

    MapleStory keeps the notification feed flush with the actual client edge
    on wide captures, while the reference viewport ends at x=1351.  The
    normal reference/static crop must stay unchanged for tests; live captures
    get a small right extension so the ``(+275)`` amount is not clipped.
    """
    boxes = dict(AUXILIARY_BOXES)
    if window_size[0] > REFERENCE_WINDOW_SIZE[0]:
        for name in ("pickup", "pickup_wide"):
            left, top, _right, bottom = boxes[name]
            boxes[name] = (left, top, REFERENCE_WINDOW_SIZE[0] + 37, bottom)
    return boxes


def _shortcut_scan_box(
    client_size: tuple[int, int],
    *,
    window_size: tuple[int, int] | None = None,
    client_offset: tuple[int, int] = (0, 0),
) -> Box:
    """Return a small search area around the expected shortcut frame.

    The shortcut frame is detected from its chrome before the eight cells are
    cropped.  The search margin is intentionally larger than the historical
    crop error, but still much smaller than a full client capture.
    """
    expected = (
        scale_window_shortcut_box_to_client(
            SHORTCUT_BOX, client_size, window_size, client_offset
        )
        if window_size is not None
        else scale_shortcut_box(SHORTCUT_BOX, client_size)
    )
    pad_x = max(18, round(expected.width * 0.24))
    pad_y = max(18, round(expected.height * 0.30))
    return Box(
        max(0, expected.left - pad_x),
        max(0, expected.top - pad_y),
        min(client_size[0], expected.right + pad_x),
        min(client_size[1], expected.bottom + pad_y),
    )


def _edge_peaks(values, start: int, threshold: float) -> list[tuple[int, float]]:
    """Find separated local edge peaks without importing a CV dependency."""
    peaks: list[tuple[int, float]] = []
    for index in range(1, len(values) - 1):
        value = float(values[index])
        if value < threshold or value < float(values[index - 1]) or value < float(values[index + 1]):
            continue
        position = start + index
        if peaks and position - peaks[-1][0] <= 2:
            if value > peaks[-1][1]:
                peaks[-1] = (position, value)
        else:
            peaks.append((position, value))
    return peaks


def detect_shortcut_frame(image: Image.Image, expected: Box | None = None) -> Box:
    """Locate the visible shortcut frame and return one calibrated parent box.

    Resolution alone is not a reliable scale source here: Windows DPI, the
    game's client size, and letterboxing can each change by a few pixels.  The
    frame has two strong full-height vertical edges and two strong full-width
    horizontal edges.  Finding those edges in a bounded neighbourhood lets
    the same eight-cell geometry follow the actual rendered game UI.

    If a compositor frame is mid-transition or the border is not visible, the
    deterministic resolution transform remains the safe fallback.
    """
    if expected is None:
        expected = scale_shortcut_box(SHORTCUT_BOX, image.size)
    expected = Box(
        max(0, min(image.width - 1, expected.left)),
        max(0, min(image.height - 1, expected.top)),
        max(1, min(image.width, expected.right)),
        max(1, min(image.height, expected.bottom)),
    )
    if expected.right <= expected.left or expected.bottom <= expected.top:
        return expected
    try:
        import numpy as np

        pad_x = max(18, round(expected.width * 0.24))
        pad_y = max(18, round(expected.height * 0.30))
        x0 = max(1, expected.left - pad_x)
        y0 = max(1, expected.top - pad_y)
        x1 = min(image.width - 1, expected.right + pad_x)
        y1 = min(image.height - 1, expected.bottom + pad_y)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return expected
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        vertical = np.abs(np.diff(gray[y0:y1, x0:x1], axis=1)).mean(axis=0)
        horizontal = np.abs(np.diff(gray[y0:y1, x0:x1], axis=0)).mean(axis=1)
        if not len(vertical) or not len(horizontal):
            return expected

        x_peaks = _edge_peaks(vertical, x0 + 1, max(18.0, float(vertical.max()) * 0.55))
        y_peaks = _edge_peaks(horizontal, y0 + 1, max(18.0, float(horizontal.max()) * 0.55))
        expected_width = max(1, expected.width)
        expected_height = max(1, expected.height)
        expected_center_x = (expected.left + expected.right) / 2
        expected_center_y = (expected.top + expected.bottom) / 2

        x_candidates: list[tuple[float, int, int]] = []
        for left, left_score in x_peaks:
            for right, right_score in x_peaks:
                width = right - left
                if width < expected_width * 0.65 or width > expected_width * 1.45:
                    continue
                score = (
                    left_score
                    + right_score
                    - abs(width - expected_width) * 0.55
                    - abs((left + right) / 2 - expected_center_x) * 0.35
                )
                x_candidates.append((score, left, right))

        y_candidates: list[tuple[float, int, int]] = []
        for top, top_score in y_peaks:
            for bottom, bottom_score in y_peaks:
                height = bottom - top
                if height < expected_height * 0.60 or height > expected_height * 1.45:
                    continue
                score = (
                    top_score
                    + bottom_score
                    - abs(height - expected_height) * 0.65
                    - abs((top + bottom) / 2 - expected_center_y) * 0.40
                )
                y_candidates.append((score, top, bottom))

        if not x_candidates or not y_candidates:
            return expected
        _, left, right = max(x_candidates)
        _, top, bottom = max(y_candidates)
        if right - left < 20 or bottom - top < 20:
            return expected
        return Box(left, top, right, bottom)
    except Exception:
        # Numpy is bundled in the normal build, but capture must remain usable
        # with a minimal source environment as well.
        return expected


def _shortcut_parent_regions(source: Image.Image, frame: Box) -> dict[str, Image.Image]:
    """Crop one frame and all eight cells from the exact same parent box."""
    parent = source.crop(frame.as_tuple())
    local_parent = Box(0, 0, parent.width, parent.height)
    slots = shortcut_slot_boxes_for_parent(local_parent)
    regions = {"shortcut": parent}
    regions.update({
        f"shortcut:{slot}": parent.crop(box.as_tuple())
        for slot, box in slots.items()
    })
    return regions


def field_sample_points(
    client_size: tuple[int, int],
    *,
    window_size: tuple[int, int] | None = None,
    client_offset: tuple[int, int] = (0, 0),
) -> list[tuple[int, int]]:
    """Client-relative points to probe for occlusion: the four corners of each
    FIELD_BOX, inset by a pixel so a corner lands inside its own box.

    Sampling per *field* rather than the panel as a whole is deliberate. A
    window clipping only the MP digits is the dangerous case -- the 'MP' label
    stays readable so the value still parses, just wrong -- and a panel-level
    check with a few points can miss it.
    """
    return box_sample_points(
        FIELD_BOXES.values(),
        client_size,
        window_size=window_size,
        client_offset=client_offset,
    )


def box_sample_points(
    boxes,
    client_size: tuple[int, int],
    *,
    window_size: tuple[int, int] | None = None,
    client_offset: tuple[int, int] = (0, 0),
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for box in boxes:
        if window_size is None:
            mapper = scale_shortcut_box if box == SHORTCUT_BOX else scale_box
            b = mapper(box, client_size)
        elif box == SHORTCUT_BOX:
            b = scale_window_shortcut_box_to_client(
                box, client_size, window_size, client_offset
            )
        else:
            b = scale_window_box_to_client(
                box, client_size, window_size, client_offset
            )
        points += [
            (b.left + 1, b.top + 1), (b.right - 2, b.top + 1),
            (b.left + 1, b.bottom - 2), (b.right - 2, b.bottom - 2),
        ]
    return points


def panel_is_obscured(sample_points, game_hwnd: int, window_at) -> bool:
    """True if any sample point belongs to a window other than the game.

    `window_at(x, y)` returns the *root* window at a screen point; injected so
    this stays testable without a Win32 desktop. Any single covered point
    counts -- there is no threshold, because partial coverage corrupts values
    rather than merely hiding them.
    """
    return any(window_at(x, y) != game_hwnd for x, y in sample_points)


class WindowCapture(Protocol):
    def grab_full(self) -> Image.Image:
        """Full client-area frame."""
        ...

    def grab_panel(self) -> Image.Image:
        """Just the stat panel crop, scaled to the current client size."""
        ...

    def grab_fields(self, *, include_bar_signals: bool = False) -> dict[str, Image.Image]:
        """One crop per FIELD_BOXES entry ('LV'/'HP'/'MP'/'EXP'), for
        recognition-only OCR -- see ocr.py's read_field().

        ``include_bar_signals`` is used by the live monitor to request the
        same-frame HP/MP bar crops without changing the four-field contract
        used by tests and custom capture sources.
        """
        ...

    def grab_auxiliary(self) -> dict[str, Image.Image]:
        """Capture economy regions ('pickup' and 'shortcut')."""
        ...

    def grab_context(self) -> dict[str, Image.Image]:
        """Capture low-frequency map/job context regions."""
        ...


class StaticImageCapture:
    """Dev/demo stand-in: always returns (a copy of) one image from disk."""

    def __init__(self, path: str | Path):
        self._image = Image.open(path).convert("RGB")
        self.shortcut_frame: Box | None = None

    def grab_full(self) -> Image.Image:
        return self._image.copy()

    def grab_panel(self) -> Image.Image:
        box = scale_box(STAT_PANEL_BOX, self._image.size)
        return self._image.crop(box.as_tuple())

    def grab_fields(self, *, include_bar_signals: bool = False) -> dict[str, Image.Image]:
        fields = {
            name: self._image.crop(scale_box(box, self._image.size).as_tuple())
            for name, box in FIELD_BOXES.items()
        }
        if include_bar_signals:
            fields.update({
                f"__bar_{resource}": self._image.crop(
                    scale_box(box, self._image.size).as_tuple()
                )
                for resource, box in BAR_BOXES.items()
            })
        return fields

    def grab_auxiliary(self) -> dict[str, Image.Image]:
        capture_boxes = _pickup_boxes_for_client(self._image.size)
        regions = {
            name: self._image.crop(scale_box(capture_boxes[name], self._image.size).as_tuple())
            for name in capture_boxes
            if name != "shortcut"
        }
        pickup = regions["pickup"]
        pickup_parent = scale_box(capture_boxes["pickup"], self._image.size)
        reference_width = AUXILIARY_BOXES["pickup"][2] - AUXILIARY_BOXES["pickup"][0]
        reference_height = AUXILIARY_BOXES["pickup"][3] - AUXILIARY_BOXES["pickup"][1]
        actual_width = pickup_parent.right - pickup_parent.left
        actual_height = pickup_parent.bottom - pickup_parent.top
        for line, raw_box in PICKUP_LINE_BOXES.items():
            left_box, top_box, right_box, bottom_box = raw_box
            regions[f"pickup:{line}"] = pickup.crop((
                round(left_box * actual_width / reference_width),
                round(top_box * actual_height / reference_height),
                round(right_box * actual_width / reference_width),
                round(bottom_box * actual_height / reference_height),
            ))
        # The demo image never changes.  Keep the same calibrated frame for
        # subsequent auxiliary reads so the test path has the same timing
        # characteristics as the live capture path.
        if self.shortcut_frame is None:
            expected = scale_shortcut_box(SHORTCUT_BOX, self._image.size)
            self.shortcut_frame = detect_shortcut_frame(self._image, expected)
        regions.update(_shortcut_parent_regions(self._image, self.shortcut_frame))
        return regions

    def grab_context(self) -> dict[str, Image.Image]:
        return {
            name: self._image.crop(scale_top_left_box(box, self._image.size).as_tuple())
            for name, box in CONTEXT_BOXES.items()
        }


class GameWindowCapture:
    """Real capture: locates the game window by title and grabs its client area."""

    def __init__(self, title_substring: str = "新楓之谷", process_name: str = "Maplestory"):
        if sys.platform != "win32":
            raise RuntimeError("GameWindowCapture requires Windows (pywin32 + real desktop)")
        # Must run before pywin32/mss query geometry, not after the first
        # capture.  Otherwise the first frame can establish a logical-pixel
        # coordinate system that remains wrong after the process is made DPI
        # aware.
        set_process_dpi_awareness()
        import mss
        import win32api
        import win32con
        import win32gui
        import win32process

        self._win32gui = win32gui
        self._win32process = win32process
        self._win32api = win32api
        self._win32con = win32con
        self._mss = mss.mss()
        self._title_substring = title_substring
        self._process_name = process_name.lower()
        self._hwnd: int | None = None
        # Last client size seen by grab_fields, for the overlay to log. Every
        # crop in regions.py is scaled from this, so it is the single most
        # useful number when diagnosing a bad read from a log after the fact
        # -- and the one thing missing from every capture taken so far.
        self.client_size: tuple[int, int] | None = None
        # Reference boxes are measured from the complete top-level window
        # screenshot. Live backends return a client-only image, so retain the
        # actual outer size and client origin inside it for every crop.
        self.window_size: tuple[int, int] | None = None
        self.client_offset: tuple[int, int] | None = None
        self.shortcut_frame: Box | None = None
        # The shortcut frame is UI geometry, not per-frame content.  Reuse the
        # calibrated client-relative box until the game client size changes;
        # detecting edges on every 150ms potion sample was unnecessary CPU work
        # and made quantity updates fall behind the game's drink animation.
        self._shortcut_frame_client_size: tuple[int, int] | None = None
        self._shortcut_frame_geometry_key: tuple | None = None
        # Status, economy, and context workers share one mss desktop grabber.
        # Serialize only the short screen-capture sections; OCR remains fully
        # independent and can never block the Tk thread.
        self._capture_lock = threading.Lock()
        # Prefer Windows Graphics Capture so foreground windows (including the
        # analyzer HUD) cannot replace the game's pixels.  If WinRT/D3D is not
        # available, the existing mss desktop path remains a fallback.
        self._graphics_capture = None
        self._graphics_capture_size: tuple[int, int] | None = None
        self._graphics_capture_disabled = False
        self._graphics_capture_retry_at = 0.0
        self.capture_backend = "windows-graphics"
        self.graphics_capture_error: str | None = None
        # PrintWindow is a second compositor-independent fallback for Windows
        # builds where the Graphics Capture service is disabled.  It remains
        # available for full/context previews, but live status/economy OCR
        # never accepts it: accelerated game clients can return a valid-looking
        # bitmap that never changes.
        self._print_window_disabled = False
        self._print_window_retry_at = 0.0
        self.print_window_error: str | None = None

    def _remember_capture_geometry(
        self,
        client_rect: tuple[int, int, int, int],
        window_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Cache the physical window/client relationship used by live crops."""
        client_left, client_top, client_right, client_bottom = client_rect
        client_size = (
            client_right - client_left,
            client_bottom - client_top,
        )
        if window_rect is None:
            try:
                hwnd = self._hwnd or self._find_window()
                window_rect = self._window_rect_on_screen(hwnd)
            except Exception:
                window_rect = client_rect
        window_left, window_top, window_right, window_bottom = window_rect
        window_size = (
            window_right - window_left,
            window_bottom - window_top,
        )
        if min(*client_size, *window_size) <= 0:
            # The caller will report the original invalid-geometry error; do
            # not replace a valid previous geometry with a zero-sized frame.
            return
        self.client_size = client_size
        self.window_size = window_size
        self.client_offset = (
            client_left - window_left,
            client_top - window_top,
        )

    def _live_box(
        self,
        raw_box: tuple[int, int, int, int],
        client_size: tuple[int, int],
        *,
        top_left: bool = False,
        shortcut: bool = False,
    ) -> Box:
        """Map one reference box using the latest live HWND geometry."""
        window_size = self.window_size or client_size
        client_offset = self.client_offset or (0, 0)
        if shortcut:
            return scale_window_shortcut_box_to_client(
                raw_box, client_size, window_size, client_offset
            )
        if top_left:
            return scale_window_top_left_box_to_client(
                raw_box, client_size, window_size, client_offset
            )
        return scale_window_box_to_client(
            raw_box, client_size, window_size, client_offset
        )

    def _live_geometry_key(self, client_size: tuple[int, int]) -> tuple:
        return client_size, self.window_size, self.client_offset

    def _owning_process_name(self, hwnd: int) -> str:
        # Title alone isn't a reliable match: e.g. a browser tab for a wiki page
        # about the game can also contain the title substring. Require the
        # window's actual owning process to match too.
        try:
            _, pid = self._win32process.GetWindowThreadProcessId(hwnd)
            # PROCESS_VM_READ is denied for this game (anti-tamper protection) --
            # PROCESS_QUERY_LIMITED_INFORMATION alone is enough for
            # GetModuleFileNameEx and works even on protected processes.
            handle = self._win32api.OpenProcess(
                self._win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            try:
                path = self._win32process.GetModuleFileNameEx(handle, 0)
                return path.rsplit("\\", 1)[-1]
            finally:
                self._win32api.CloseHandle(handle)
        except Exception:
            return ""

    def _is_match(self, hwnd: int) -> bool:
        title = self._win32gui.GetWindowText(hwnd)
        if self._title_substring not in title:
            return False
        owner = self._owning_process_name(hwnd).lower()
        # Anti-tamper protected clients can deny GetModuleFileNameEx even
        # though the title is an exact game-window match.  Do not make that
        # harmless query failure look like "game window not found".
        return not owner or self._process_name in owner

    def _find_window(self) -> int:
        # IsWindow() alone isn't enough: if the game process exits, Windows can
        # recycle its hwnd number for an unrelated window, and IsWindow() stays
        # True for that new window -- silently capturing garbage instead of
        # erroring. Re-check title+process on every call to catch that.
        if self._hwnd and self._win32gui.IsWindow(self._hwnd) and self._is_match(self._hwnd):
            return self._hwnd
        self._hwnd = None

        found: list[int] = []

        def _cb(hwnd: int, _):
            if self._is_match(hwnd):
                found.append(hwnd)

        self._win32gui.EnumWindows(_cb, None)
        if not found:
            raise RuntimeError(f"No window found with title containing {self._title_substring!r}")
        self._hwnd = found[0]
        return self._hwnd

    def _client_rect_on_screen(self) -> tuple[int, int, int, int]:
        hwnd = self._find_window()
        if self._win32gui.IsIconic(hwnd):
            # Minimized windows report a client rect around (-32000, -32000)
            # with zero size -- mss.grab() throws a raw ScreenShotError on
            # that instead of anything actionable. Fail clearly here so
            # callers (the overlay) can show 'game minimized' and retry,
            # same as the 'game not found' case.
            raise RuntimeError("game window is minimized")
        left, top, right, bottom = self._win32gui.GetClientRect(hwnd)
        left, top = self._win32gui.ClientToScreen(hwnd, (left, top))
        right, bottom = self._win32gui.ClientToScreen(hwnd, (right, bottom))
        if right <= left or bottom <= top:
            raise RuntimeError("game window has an invalid client size")
        return left, top, right, bottom

    def _window_rect_on_screen(self, hwnd: int) -> tuple[int, int, int, int]:
        """Return the visible HWND frame in the same physical space as client_rect.

        ``GetWindowRect`` can include an invisible resize border on modern
        Windows.  DWM's extended frame bounds omit that border and therefore
        line up with the compositor frame used by WGC; the Win32 rectangle is
        retained as a safe fallback for older systems and test doubles.
        """
        rect = tuple(int(value) for value in self._win32gui.GetWindowRect(hwnd))
        try:
            import ctypes
            from ctypes import wintypes

            frame = wintypes.RECT()
            result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
                hwnd,
                9,  # DWMWA_EXTENDED_FRAME_BOUNDS
                ctypes.byref(frame),
                ctypes.sizeof(frame),
            )
            candidate = (
                int(frame.left), int(frame.top), int(frame.right), int(frame.bottom)
            )
            if result == 0 and candidate[2] > candidate[0] and candidate[3] > candidate[1]:
                return candidate
        except Exception:
            pass
        return rect

    def _try_graphics_frame(
        self, client_rect: tuple[int, int, int, int]
    ) -> Image.Image | None:
        """Return a compositor-independent frame, or select desktop fallback."""
        if self._graphics_capture_disabled:
            # A transient Graphics Capture service/driver failure should not
            # permanently downgrade the app to desktop pixels.  Retry after
            # a short cooldown so an OBS-like HWND capture can recover after
            # the game finishes initializing or the compositor restarts.
            if time.monotonic() < getattr(self, "_graphics_capture_retry_at", 0.0):
                return None
            self._graphics_capture_disabled = False
        try:
            from .graphics_capture import WindowsGraphicsCapture

            hwnd = self._hwnd or self._find_window()
            client_width = client_rect[2] - client_rect[0]
            client_height = client_rect[3] - client_rect[1]
            client_size = (client_width, client_height)
            window_rect = self._window_rect_on_screen(hwnd)
            self._remember_capture_geometry(client_rect, window_rect)
            capture_size = (
                window_rect[2] - window_rect[0],
                window_rect[3] - window_rect[1],
            )
            if self._graphics_capture is None or getattr(self, "_graphics_capture_size", None) != capture_size:
                # A resized game window needs a new frame pool.  Reusing the
                # old pool makes WGC return the old bitmap size and the former
                # code permanently disabled the backend after that mismatch.
                old_capture = self._graphics_capture
                self._graphics_capture = None
                self._graphics_capture_size = None
                if old_capture is not None:
                    with contextlib.suppress(Exception):
                        old_capture.close()
                # Let WGC use the GraphicsCaptureItem's native physical size.
                # Passing client_size here was incorrect for a top-level HWND:
                # Windows resized the full window (including its non-client
                # frame) into a client-sized bitmap before OCR saw it. The
                # outer size is only a fallback for a zero-sized item during
                # the first compositor frame.
                self._graphics_capture = WindowsGraphicsCapture(hwnd, capture_size)
                self._graphics_capture_size = capture_size
            image = self._graphics_capture.grab(timeout=1.2)
            item_size = getattr(self._graphics_capture, "item_size", None)
            frame = _crop_frame_to_client(
                image,
                client_rect,
                window_rect,
                item_size=item_size,
            )
            if frame is not None:
                self.graphics_capture_error = None
                self.capture_backend = "windows-graphics"
                return frame
            raise RuntimeError(
                f"graphics capture size {image.size} does not match client "
                f"{client_size}; window={capture_size}; item={item_size}"
            )
        except Exception as exc:
            self.graphics_capture_error = str(exc.__cause__ or exc)
            self._graphics_capture_disabled = True
            self._graphics_capture_retry_at = time.monotonic() + 5.0
            graphics = self._graphics_capture
            self._graphics_capture = None
            self._graphics_capture_size = None
            if graphics is not None:
                with contextlib.suppress(Exception):
                    graphics.close()
            self.capture_backend = "desktop"
            return None

    def _try_print_window_frame(
        self, client_rect: tuple[int, int, int, int]
    ) -> Image.Image | None:
        """Capture the target HWND without reading the visible desktop.

        ``PrintWindow(PW_CLIENTONLY | PW_RENDERFULLCONTENT)`` asks the target
        window to render into an off-screen bitmap. It is not supported by
        every DirectX renderer, so a failed/blank result simply returns None
        and the normal visible-desktop path remains available.
        """
        if self._print_window_disabled:
            if time.monotonic() < self._print_window_retry_at:
                return None
            self._print_window_disabled = False

        hwnd = self._hwnd or self._find_window()
        width = client_rect[2] - client_rect[0]
        height = client_rect[3] - client_rect[1]
        if width <= 0 or height <= 0:
            return None

        source_dc = None
        memory_dc = None
        bitmap = None
        old_bitmap = None
        window_dc = None
        try:
            import ctypes
            import win32ui

            window_dc = self._win32gui.GetWindowDC(hwnd)
            if not window_dc:
                raise RuntimeError("GetWindowDC returned no device context")
            source_dc = win32ui.CreateDCFromHandle(window_dc)
            memory_dc = source_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(source_dc, width, height)
            old_bitmap = memory_dc.SelectObject(bitmap)
            flags = 0x00000001 | 0x00000002  # PW_CLIENTONLY | PW_RENDERFULLCONTENT
            rendered = ctypes.windll.user32.PrintWindow(
                hwnd, memory_dc.GetSafeHdc(), flags
            )
            bits = bitmap.GetBitmapBits(True)
            if not rendered or len(bits) < width * height * 4:
                raise RuntimeError("PrintWindow returned an empty frame")

            image = Image.frombuffer(
                "RGB", (width, height), bits, "raw", "BGRX", 0, 1
            ).copy()
            # A black/empty PrintWindow response is common for unsupported
            # accelerated renderers. Reject it instead of feeding blank OCR.
            gray = image.convert("L")
            histogram = gray.histogram()
            if sum(histogram[32:]) < max(100, int(width * height * 0.001)):
                raise RuntimeError("PrintWindow returned a blank frame")
            self.print_window_error = None
            self.capture_backend = "print-window"
            return image
        except Exception as exc:
            self.print_window_error = str(exc)
            self._print_window_disabled = True
            self._print_window_retry_at = time.monotonic() + 5.0
            return None
        finally:
            if memory_dc is not None and old_bitmap is not None:
                with contextlib.suppress(Exception):
                    memory_dc.SelectObject(old_bitmap)
            if bitmap is not None:
                with contextlib.suppress(Exception):
                    bitmap.DeleteObject()
            if memory_dc is not None:
                with contextlib.suppress(Exception):
                    memory_dc.DeleteDC()
            if window_dc is not None:
                with contextlib.suppress(Exception):
                    self._win32gui.ReleaseDC(hwnd, window_dc)

    def _try_window_frame(
        self, client_rect: tuple[int, int, int, int]
    ) -> Image.Image | None:
        """Try compositor capture backends in priority order."""
        self._remember_capture_geometry(client_rect)
        graphics = self._try_graphics_frame(client_rect)
        if graphics is not None:
            return graphics
        return self._try_print_window_frame(client_rect)

    def _obscured_live_error(self) -> str:
        """Explain why a covered live panel cannot be read safely.

        PrintWindow is intentionally excluded from the live fields path.  A
        stale off-screen bitmap is worse than a visible error because it makes
        the session record false EXP, mesos, HP loss, and potion counts.
        """
        detail = self.graphics_capture_error
        if detail:
            return f"{PANEL_OBSCURED}; Windows Graphics Capture unavailable: {detail}"
        return PANEL_OBSCURED

    def grab_full(self) -> Image.Image:
        client_rect = self._client_rect_on_screen()
        window_frame = self._try_window_frame(client_rect)
        if window_frame is not None:
            self.client_size = window_frame.size
            return window_frame
        left, top, right, bottom = client_rect
        self.client_size = (right - left, bottom - top)
        shot = self._mss.grab({"left": left, "top": top, "width": right - left, "height": bottom - top})
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def grab_panel(self) -> Image.Image:
        client_rect = self._client_rect_on_screen()
        window_frame = self._try_window_frame(client_rect)
        if window_frame is not None:
            self.client_size = window_frame.size
            return window_frame.crop(
                self._live_box(STAT_PANEL_BOX, window_frame.size).as_tuple()
            )
        left, top, right, bottom = client_rect
        client_size = (right - left, bottom - top)
        self._remember_capture_geometry(client_rect)
        box = self._live_box(STAT_PANEL_BOX, client_size)
        shot = self._mss.grab({
            "left": left + box.left,
            "top": top + box.top,
            "width": box.right - box.left,
            "height": box.bottom - box.top,
        })
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def _root_window_at(self, x: int, y: int) -> int:
        # GA_ROOT (2) resolves a child window/control to its top-level owner,
        # so the game's own children don't read as something covering it.
        return self._win32gui.GetAncestor(self._win32gui.WindowFromPoint((x, y)), 2)

    def grab_fields(self, *, include_bar_signals: bool = False) -> dict[str, Image.Image]:
        with self._capture_lock:
            return self._grab_fields(include_bar_signals=include_bar_signals)

    def _grab_fields(self, *, include_bar_signals: bool = False) -> dict[str, Image.Image]:
        client_rect = self._client_rect_on_screen()
        self._remember_capture_geometry(client_rect)
        # WGC is the live, compositor-independent path.  When it is
        # unavailable, try the visible desktop before PrintWindow: classic
        # DirectX clients can return a perfectly valid-looking but stale
        # PrintWindow bitmap, which would freeze EXP/HP/MP at the first frame.
        graphics = self._try_graphics_frame(client_rect)
        if graphics is not None:
            self.client_size = graphics.size
            fields = {
                name: graphics.crop(self._live_box(box, graphics.size).as_tuple())
                for name, box in FIELD_BOXES.items()
            }
            if include_bar_signals:
                fields.update({
                    f"__bar_{resource}": graphics.crop(
                        self._live_box(box, graphics.size).as_tuple()
                    )
                    for resource, box in BAR_BOXES.items()
                })
            return fields

        # One screen grab covering the whole panel (mss itself is cheap, ~3.5ms
        # measured -- see VERSIONS.md/overlay.py timing notes), then slice each
        # field out of that single in-memory image rather than four separate
        # mss.grab() calls.
        left, top, right, bottom = client_rect
        client_size = (right - left, bottom - top)
        self.client_size = client_size

        # mss grabs the screen *region* where the panel sits, not the game's
        # own pixels, so anything on top of it is what would reach OCR. Refuse
        # the frame instead of reading someone else's window (~0.03ms measured
        # for the whole check, against ~60ms of OCR).
        points = [
            (left + x, top + y)
            for x, y in field_sample_points(
                client_size,
                window_size=self.window_size,
                client_offset=self.client_offset or (0, 0),
            )
        ]
        if panel_is_obscured(points, self._hwnd, self._root_window_at):
            raise RuntimeError(self._obscured_live_error())

        self.capture_backend = "desktop"
        panel_box = self._live_box(STAT_PANEL_BOX, client_size)
        shot = self._mss.grab({
            "left": left + panel_box.left,
            "top": top + panel_box.top,
            "width": panel_box.right - panel_box.left,
            "height": panel_box.bottom - panel_box.top,
        })
        panel = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        fields = {}
        for name, box in FIELD_BOXES.items():
            field_box = self._live_box(box, client_size)
            local = (
                field_box.left - panel_box.left, field_box.top - panel_box.top,
                field_box.right - panel_box.left, field_box.bottom - panel_box.top,
            )
            fields[name] = panel.crop(local)
        if include_bar_signals:
            for resource, box in BAR_BOXES.items():
                bar_box = self._live_box(box, client_size)
                fields[f"__bar_{resource}"] = panel.crop((
                    bar_box.left - panel_box.left,
                    bar_box.top - panel_box.top,
                    bar_box.right - panel_box.left,
                    bar_box.bottom - panel_box.top,
                ))
        return fields

    def grab_auxiliary(self) -> dict[str, Image.Image]:
        with self._capture_lock:
            return self._grab_auxiliary()

    def _grab_auxiliary(self) -> dict[str, Image.Image]:
        client_rect = self._client_rect_on_screen()
        self._remember_capture_geometry(client_rect)
        graphics = self._try_graphics_frame(client_rect)
        if graphics is not None:
            client_size = graphics.size
            self.client_size = client_size
            capture_window_size = self.window_size or client_size
            capture_boxes = _pickup_boxes_for_client(capture_window_size)
            regions: dict[str, Image.Image] = {
                name: graphics.crop(
                    self._live_box(capture_boxes[name], client_size).as_tuple()
                )
                for name in capture_boxes
                if name != "shortcut"
            }
            expected = self._live_box(
                SHORTCUT_BOX, client_size, shortcut=True
            )
            cached = self.shortcut_frame
            geometry_key = self._live_geometry_key(client_size)
            cached_key = getattr(self, "_shortcut_frame_geometry_key", None)
            cached_valid = (
                isinstance(cached, Box)
                and cached_key == geometry_key
                and 0 <= cached.left < cached.right <= client_size[0]
                and 0 <= cached.top < cached.bottom <= client_size[1]
            )
            if not cached_valid:
                self.shortcut_frame = detect_shortcut_frame(graphics, expected)
                self._shortcut_frame_client_size = client_size
                self._shortcut_frame_geometry_key = geometry_key
            regions.update(_shortcut_parent_regions(graphics, self.shortcut_frame))
        else:
            left, top, right, bottom = client_rect
            client_size = (right - left, bottom - top)
            self.client_size = client_size

            # Refuse auxiliary OCR only for the desktop fallback. Graphics
            # Capture already reads the game's own compositor surface.
            points = [
                (left + x, top + y)
                for x, y in box_sample_points(
                    AUXILIARY_BOXES.values(),
                    client_size,
                    window_size=self.window_size,
                    client_offset=self.client_offset or (0, 0),
                )
            ]
            if panel_is_obscured(points, self._hwnd, self._root_window_at):
                raise RuntimeError(self._obscured_live_error())
            else:
                self.capture_backend = "desktop"
                capture_window_size = self.window_size or client_size
                capture_boxes = _pickup_boxes_for_client(capture_window_size)
                regions = {}
                for name, raw_box in capture_boxes.items():
                    if name == "shortcut":
                        continue
                    box = self._live_box(raw_box, client_size)
                    shot = self._mss.grab({
                        "left": left + box.left,
                        "top": top + box.top,
                        "width": box.right - box.left,
                        "height": box.bottom - box.top,
                    })
                    regions[name] = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

                # The desktop fallback cannot use the game's hidden pixels,
                # so capture a small expanded neighbourhood and detect the
                # frame inside it.  This is still far cheaper than grabbing
                # the whole desktop and it prevents a few DPI pixels from
                # moving the eight quantity crops into adjacent cells.
                scan_box = _shortcut_scan_box(
                    client_size,
                    window_size=self.window_size,
                    client_offset=self.client_offset or (0, 0),
                )
                scan_shot = self._mss.grab({
                    "left": left + scan_box.left,
                    "top": top + scan_box.top,
                    "width": scan_box.right - scan_box.left,
                    "height": scan_box.bottom - scan_box.top,
                })
                scan_image = Image.frombytes(
                    "RGB", scan_shot.size, scan_shot.bgra, "raw", "BGRX"
                )
                expected = self._live_box(
                    SHORTCUT_BOX, client_size, shortcut=True
                )
                expected_local = Box(
                    expected.left - scan_box.left,
                    expected.top - scan_box.top,
                    expected.right - scan_box.left,
                    expected.bottom - scan_box.top,
                )
                cached = self.shortcut_frame
                geometry_key = self._live_geometry_key(client_size)
                cached_key = getattr(self, "_shortcut_frame_geometry_key", None)
                cached_valid = (
                    isinstance(cached, Box)
                    and cached_key == geometry_key
                    and scan_box.left <= cached.left < cached.right <= scan_box.right
                    and scan_box.top <= cached.top < cached.bottom <= scan_box.bottom
                )
                if cached_valid:
                    frame_local = Box(
                        cached.left - scan_box.left,
                        cached.top - scan_box.top,
                        cached.right - scan_box.left,
                        cached.bottom - scan_box.top,
                    )
                else:
                    frame_local = detect_shortcut_frame(scan_image, expected_local)
                    self.shortcut_frame = Box(
                        frame_local.left + scan_box.left,
                        frame_local.top + scan_box.top,
                        frame_local.right + scan_box.left,
                        frame_local.bottom + scan_box.top,
                    )
                    self._shortcut_frame_client_size = client_size
                    self._shortcut_frame_geometry_key = geometry_key
                regions.update(_shortcut_parent_regions(scan_image, frame_local))

        pickup = regions["pickup"]
        reference_feed_width = AUXILIARY_BOXES["pickup"][2] - AUXILIARY_BOXES["pickup"][0]
        reference_feed_height = AUXILIARY_BOXES["pickup"][3] - AUXILIARY_BOXES["pickup"][1]
        actual_feed_size = pickup.size
        for line, raw_box in PICKUP_LINE_BOXES.items():
            left_box, top_box, right_box, bottom_box = raw_box
            regions[f"pickup:{line}"] = pickup.crop((
                round(left_box * actual_feed_size[0] / reference_feed_width),
                round(top_box * actual_feed_size[1] / reference_feed_height),
                round(right_box * actual_feed_size[0] / reference_feed_width),
                round(bottom_box * actual_feed_size[1] / reference_feed_height),
            ))
        return regions

    def grab_context(self) -> dict[str, Image.Image]:
        with self._capture_lock:
            return self._grab_context()

    def _grab_context(self) -> dict[str, Image.Image]:
        client_rect = self._client_rect_on_screen()
        self._remember_capture_geometry(client_rect)
        window_frame = self._try_window_frame(client_rect)
        if window_frame is not None:
            self.client_size = window_frame.size
            return {
                name: window_frame.crop(
                    self._live_box(
                        box,
                        window_frame.size,
                        top_left=name in {"map", "map_wide"},
                    ).as_tuple()
                )
                for name, box in CONTEXT_BOXES.items()
            }

        left, top, right, bottom = client_rect
        client_size = (right - left, bottom - top)
        self.client_size = client_size
        regions: dict[str, Image.Image] = {}
        for name, raw_box in CONTEXT_BOXES.items():
            box = self._live_box(
                raw_box,
                client_size,
                top_left=name in {"map", "map_wide"},
            )
            shot = self._mss.grab({
                "left": left + box.left,
                "top": top + box.top,
                "width": box.right - box.left,
                "height": box.bottom - box.top,
            })
            regions[name] = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return regions


def get_capture(sample_path: str | Path | None = None) -> WindowCapture:
    if sample_path is not None:
        return StaticImageCapture(sample_path)
    if sys.platform == "win32":
        return GameWindowCapture()
    raise RuntimeError(
        "Real game capture requires Windows. Pass sample_path= to use "
        "StaticImageCapture for dev/testing on this platform."
    )
