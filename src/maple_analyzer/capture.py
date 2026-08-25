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
    CONTEXT_BOXES,
    FIELD_BOXES,
    REFERENCE_CLIENT_SIZE,
    PICKUP_LINE_BOXES,
    SHORTCUT_BOX,
    SHORTCUT_SLOT_BOXES,
    STAT_PANEL_BOX,
    scale_box,
    scale_top_left_box,
)


# Raised (as a RuntimeError message) when another window sits over the stat
# panel. Routine and recoverable, so it travels the same path as the
# minimized/not-found states -- see overlay._do_tick and _localize_error.
PANEL_OBSCURED = "stat panel is obscured"


def _pickup_boxes_for_client(client_size: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """Extend right-side pickup crops into the wide-client letterbox.

    MapleStory keeps the notification feed flush with the actual client edge
    on wide captures, while the reference viewport ends at x=1351.  The
    normal reference/static crop must stay unchanged for tests; live captures
    get a small right extension so the ``(+275)`` amount is not clipped.
    """
    boxes = dict(AUXILIARY_BOXES)
    if client_size[0] > REFERENCE_CLIENT_SIZE[0]:
        for name in ("pickup", "pickup_wide"):
            left, top, _right, bottom = boxes[name]
            boxes[name] = (left, top, REFERENCE_CLIENT_SIZE[0] + 37, bottom)
    return boxes


def field_sample_points(client_size: tuple[int, int]) -> list[tuple[int, int]]:
    """Client-relative points to probe for occlusion: the four corners of each
    FIELD_BOX, inset by a pixel so a corner lands inside its own box.

    Sampling per *field* rather than the panel as a whole is deliberate. A
    window clipping only the MP digits is the dangerous case -- the 'MP' label
    stays readable so the value still parses, just wrong -- and a panel-level
    check with a few points can miss it.
    """
    return box_sample_points(FIELD_BOXES.values(), client_size)


def box_sample_points(boxes, client_size: tuple[int, int]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for box in boxes:
        b = scale_box(box, client_size)
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

    def grab_fields(self) -> dict[str, Image.Image]:
        """One crop per FIELD_BOXES entry ('LV'/'HP'/'MP'/'EXP'), for
        recognition-only OCR -- see ocr.py's read_field()."""
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

    def grab_full(self) -> Image.Image:
        return self._image.copy()

    def grab_panel(self) -> Image.Image:
        box = scale_box(STAT_PANEL_BOX, self._image.size)
        return self._image.crop(box.as_tuple())

    def grab_fields(self) -> dict[str, Image.Image]:
        return {
            name: self._image.crop(scale_box(box, self._image.size).as_tuple())
            for name, box in FIELD_BOXES.items()
        }

    def grab_auxiliary(self) -> dict[str, Image.Image]:
        regions = {
            name: self._image.crop(scale_box(box, self._image.size).as_tuple())
            for name, box in AUXILIARY_BOXES.items()
        }
        pickup = regions["pickup"]
        reference_width = AUXILIARY_BOXES["pickup"][2] - AUXILIARY_BOXES["pickup"][0]
        reference_height = AUXILIARY_BOXES["pickup"][3] - AUXILIARY_BOXES["pickup"][1]
        for line, raw_box in PICKUP_LINE_BOXES.items():
            left_box, top_box, right_box, bottom_box = raw_box
            regions[f"pickup:{line}"] = pickup.crop((
                round(left_box * pickup.width / reference_width),
                round(top_box * pickup.height / reference_height),
                round(right_box * pickup.width / reference_width),
                round(bottom_box * pickup.height / reference_height),
            ))
        regions.update({
            f"shortcut:{slot}": self._image.crop(scale_box(box, self._image.size).as_tuple())
            for slot, box in SHORTCUT_SLOT_BOXES.items()
        })
        return regions

    def grab_context(self) -> dict[str, Image.Image]:
        return {
            name: self._image.crop(scale_box(box, self._image.size).as_tuple())
            for name, box in CONTEXT_BOXES.items()
        }


class GameWindowCapture:
    """Real capture: locates the game window by title and grabs its client area."""

    def __init__(self, title_substring: str = "新楓之谷", process_name: str = "Maplestory"):
        if sys.platform != "win32":
            raise RuntimeError("GameWindowCapture requires Windows (pywin32 + real desktop)")
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
        return left, top, right, bottom

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
            capture_size = (client_width, client_height)
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
                self._graphics_capture = WindowsGraphicsCapture(
                    hwnd, capture_size
                )
                self._graphics_capture_size = capture_size
            image = self._graphics_capture.grab(timeout=1.2)
            if image.size == capture_size:
                self.capture_backend = "windows-graphics"
                return image

            # Some Windows builds include the non-client frame in a window
            # capture. Convert it back to the same client-relative coordinate
            # system used by regions.py before handing it to OCR.
            window_left, window_top, _window_right, _window_bottom = self._win32gui.GetWindowRect(hwnd)
            offset_x = client_rect[0] - window_left
            offset_y = client_rect[1] - window_top
            crop_box = (
                offset_x,
                offset_y,
                offset_x + client_width,
                offset_y + client_height,
            )
            if (
                offset_x >= 0
                and offset_y >= 0
                and crop_box[2] <= image.width
                and crop_box[3] <= image.height
            ):
                cropped = image.crop(crop_box)
                if cropped.size == (client_width, client_height):
                    self.capture_backend = "windows-graphics"
                    return cropped
            raise RuntimeError(
                f"graphics capture size {image.size} does not match client "
                f"{capture_size}"
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
            return window_frame.crop(scale_box(STAT_PANEL_BOX, window_frame.size).as_tuple())
        left, top, right, bottom = client_rect
        client_size = (right - left, bottom - top)
        self.client_size = client_size
        box = scale_box(STAT_PANEL_BOX, client_size)
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

    def grab_fields(self) -> dict[str, Image.Image]:
        with self._capture_lock:
            return self._grab_fields()

    def _grab_fields(self) -> dict[str, Image.Image]:
        client_rect = self._client_rect_on_screen()
        # WGC is the live, compositor-independent path.  When it is
        # unavailable, try the visible desktop before PrintWindow: classic
        # DirectX clients can return a perfectly valid-looking but stale
        # PrintWindow bitmap, which would freeze EXP/HP/MP at the first frame.
        graphics = self._try_graphics_frame(client_rect)
        if graphics is not None:
            self.client_size = graphics.size
            return {
                name: graphics.crop(scale_box(box, graphics.size).as_tuple())
                for name, box in FIELD_BOXES.items()
            }

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
        points = [(left + x, top + y) for x, y in field_sample_points(client_size)]
        if panel_is_obscured(points, self._hwnd, self._root_window_at):
            raise RuntimeError(self._obscured_live_error())

        self.capture_backend = "desktop"
        panel_box = scale_box(STAT_PANEL_BOX, client_size)
        shot = self._mss.grab({
            "left": left + panel_box.left,
            "top": top + panel_box.top,
            "width": panel_box.right - panel_box.left,
            "height": panel_box.bottom - panel_box.top,
        })
        panel = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        fields = {}
        for name, box in FIELD_BOXES.items():
            field_box = scale_box(box, client_size)
            local = (
                field_box.left - panel_box.left, field_box.top - panel_box.top,
                field_box.right - panel_box.left, field_box.bottom - panel_box.top,
            )
            fields[name] = panel.crop(local)
        return fields

    def grab_auxiliary(self) -> dict[str, Image.Image]:
        with self._capture_lock:
            return self._grab_auxiliary()

    def _grab_auxiliary(self) -> dict[str, Image.Image]:
        client_rect = self._client_rect_on_screen()
        graphics = self._try_graphics_frame(client_rect)
        if graphics is not None:
            client_size = graphics.size
            self.client_size = client_size
            capture_boxes = _pickup_boxes_for_client(client_size)
            regions: dict[str, Image.Image] = {
                name: graphics.crop(scale_box(capture_boxes[name], client_size).as_tuple())
                for name in capture_boxes
            }
        else:
            left, top, right, bottom = client_rect
            client_size = (right - left, bottom - top)
            self.client_size = client_size

            # Refuse auxiliary OCR only for the desktop fallback. Graphics
            # Capture already reads the game's own compositor surface.
            points = [(left + x, top + y) for x, y in box_sample_points(AUXILIARY_BOXES.values(), client_size)]
            if panel_is_obscured(points, self._hwnd, self._root_window_at):
                raise RuntimeError(self._obscured_live_error())
            else:
                self.capture_backend = "desktop"
                capture_boxes = _pickup_boxes_for_client(client_size)
                regions = {}
                for name, raw_box in capture_boxes.items():
                    box = scale_box(raw_box, client_size)
                    shot = self._mss.grab({
                        "left": left + box.left,
                        "top": top + box.top,
                        "width": box.right - box.left,
                        "height": box.bottom - box.top,
                    })
                    regions[name] = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        pickup = regions["pickup"]
        pickup_parent = scale_box(_pickup_boxes_for_client(client_size)["pickup"], client_size)
        reference_feed_width = AUXILIARY_BOXES["pickup"][2] - AUXILIARY_BOXES["pickup"][0]
        reference_feed_height = AUXILIARY_BOXES["pickup"][3] - AUXILIARY_BOXES["pickup"][1]
        actual_feed_size = (pickup_parent.right - pickup_parent.left, pickup_parent.bottom - pickup_parent.top)
        for line, raw_box in PICKUP_LINE_BOXES.items():
            left_box, top_box, right_box, bottom_box = raw_box
            regions[f"pickup:{line}"] = pickup.crop((
                round(left_box * actual_feed_size[0] / reference_feed_width),
                round(top_box * actual_feed_size[1] / reference_feed_height),
                round(right_box * actual_feed_size[0] / reference_feed_width),
                round(bottom_box * actual_feed_size[1] / reference_feed_height),
            ))
        shortcut = regions["shortcut"]
        parent = scale_box(SHORTCUT_BOX, client_size)
        for slot, raw_box in SHORTCUT_SLOT_BOXES.items():
            box = scale_box(raw_box, client_size)
            local = (
                box.left - parent.left,
                box.top - parent.top,
                box.right - parent.left,
                box.bottom - parent.top,
            )
            regions[f"shortcut:{slot}"] = shortcut.crop(local)
        return regions

    def grab_context(self) -> dict[str, Image.Image]:
        with self._capture_lock:
            return self._grab_context()

    def _grab_context(self) -> dict[str, Image.Image]:
        client_rect = self._client_rect_on_screen()
        window_frame = self._try_window_frame(client_rect)
        if window_frame is not None:
            self.client_size = window_frame.size
            return {
                name: window_frame.crop(
                    scale_top_left_box(box, window_frame.size).as_tuple()
                )
                for name, box in CONTEXT_BOXES.items()
            }

        left, top, right, bottom = client_rect
        client_size = (right - left, bottom - top)
        self.client_size = client_size
        regions: dict[str, Image.Image] = {}
        for name, raw_box in CONTEXT_BOXES.items():
            box = scale_top_left_box(raw_box, client_size)
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
