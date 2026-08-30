"""Tests for the compositor-backed window capture adapter."""
from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import pytest

from PIL import Image

from maple_analyzer.capture import (
    LIVE_GRAPHICS_SHARED_FRAME_SECONDS,
    GameWindowCapture,
    _crop_frame_to_client,
)
from maple_analyzer.graphics_capture import GraphicsCaptureError, WindowsGraphicsCapture


def _capture_stub(
    *,
    frame: Image.Image,
    hwnd: int = 123,
    window_rect: tuple[int, int, int, int] | None = None,
    item_size: tuple[int, int] | None = None,
):
    capture = GameWindowCapture.__new__(GameWindowCapture)
    capture._graphics_capture = None
    capture._graphics_capture_size = None
    capture._graphics_capture_disabled = False
    capture._hwnd = hwnd
    capture.capture_backend = "windows-graphics"
    capture.graphics_capture_error = None
    capture._window_rect_on_screen = lambda _hwnd: window_rect or (0, 0, frame.width, frame.height)
    capture._win32gui = SimpleNamespace(
        GetWindowRect=lambda _hwnd: window_rect or (0, 0, frame.width, frame.height)
    )

    class FakeGraphicsCapture:
        def __init__(self, received_hwnd, capture_size=None):
            assert received_hwnd == hwnd
            expected_size = (
                (window_rect[2] - window_rect[0], window_rect[3] - window_rect[1])
                if window_rect is not None
                else (frame.width, frame.height)
            )
            assert capture_size == expected_size
            self.item_size = item_size

        def grab(self, timeout=1.2):
            assert timeout == 0.25
            return frame.copy()

        def close(self):
            pass

    return capture, FakeGraphicsCapture


def test_graphics_frame_prefers_same_size_client_frame(monkeypatch):
    image = Image.new("RGB", (100, 60), "#123456")
    capture, fake = _capture_stub(frame=image)
    monkeypatch.setattr("maple_analyzer.graphics_capture.WindowsGraphicsCapture", fake)

    result = GameWindowCapture._try_graphics_frame(capture, (20, 30, 120, 90))

    assert result is not None
    assert result.size == (100, 60)
    assert result.getpixel((0, 0)) == (18, 52, 86)
    assert capture.capture_backend == "windows-graphics"


def test_graphics_frame_crops_non_client_frame_to_client_coordinates(monkeypatch):
    image = Image.new("RGB", (130, 80), "#000000")
    image.paste("#abcdef", (10, 10, 110, 70))
    capture, fake = _capture_stub(frame=image)
    monkeypatch.setattr("maple_analyzer.graphics_capture.WindowsGraphicsCapture", fake)

    result = GameWindowCapture._try_graphics_frame(capture, (10, 10, 110, 70))

    assert result is not None
    assert result.size == (100, 60)
    assert result.getpixel((0, 0)) == (171, 205, 239)
    assert capture.capture_backend == "windows-graphics"


def test_graphics_frame_scales_native_window_frame_before_client_crop(monkeypatch):
    image = Image.new("RGB", (260, 160), "#000000")
    image.paste("#abcdef", (20, 20, 220, 140))
    capture, fake = _capture_stub(
        frame=image,
        window_rect=(0, 0, 260, 160),
        item_size=(260, 160),
    )
    monkeypatch.setattr("maple_analyzer.graphics_capture.WindowsGraphicsCapture", fake)

    result = GameWindowCapture._try_graphics_frame(capture, (20, 20, 220, 140))

    assert result is not None
    assert result.size == (200, 120)
    assert result.getpixel((0, 0)) == (171, 205, 239)


def test_native_graphics_crop_can_preserve_physical_pixels():
    """A DPI-scaled WGC bitmap must not be reduced before OCR."""
    image = Image.new("RGB", (520, 320), "#000000")
    image.paste("#abcdef", (40, 40, 440, 280))

    resized = _crop_frame_to_client(
        image,
        (20, 20, 220, 140),
        (0, 0, 260, 160),
    )
    native = _crop_frame_to_client(
        image,
        (20, 20, 220, 140),
        (0, 0, 260, 160),
        preserve_native=True,
    )

    assert resized is not None and resized.size == (200, 120)
    assert native is not None and native.size == (400, 240)
    assert native.getpixel((20, 20)) == (171, 205, 239)


def test_native_graphics_frame_rescales_cached_geometry():
    """Later live boxes must use the same physical-pixel space as WGC."""
    capture = GameWindowCapture.__new__(GameWindowCapture)
    capture.window_size = (260, 160)
    capture.client_offset = (20, 20)
    capture.client_size = (200, 120)

    capture._rescale_geometry_for_frame((200, 120), (400, 240))

    assert capture.client_size == (400, 240)
    assert capture.window_size == (520, 320)
    assert capture.client_offset == (40, 40)


def test_graphics_path_keeps_native_frame_and_maps_boxes_in_that_space(monkeypatch):
    """Exercise the same mismatch through GameWindowCapture's WGC path."""
    image = Image.new("RGB", (520, 320), "#000000")
    capture, fake = _capture_stub(
        frame=image,
        window_rect=(0, 0, 260, 160),
        item_size=(260, 160),
    )
    monkeypatch.setattr("maple_analyzer.graphics_capture.WindowsGraphicsCapture", fake)

    result = GameWindowCapture._try_graphics_frame(
        capture,
        (20, 20, 220, 140),
    )

    assert result is not None and result.size == (400, 240)
    assert capture.client_size == (400, 240)
    assert capture.window_size == (520, 320)
    assert capture.client_offset == (40, 40)


def test_graphics_frame_accepts_a_client_only_graphics_item(monkeypatch):
    image = Image.new("RGB", (100, 60), "#123456")
    capture, fake = _capture_stub(
        frame=image,
        window_rect=(0, 0, 130, 80),
        item_size=(100, 60),
    )
    monkeypatch.setattr("maple_analyzer.graphics_capture.WindowsGraphicsCapture", fake)

    result = GameWindowCapture._try_graphics_frame(capture, (10, 10, 110, 70))

    assert result is not None
    assert result.size == (100, 60)
    assert result.getpixel((0, 0)) == (18, 52, 86)


def test_graphics_frame_disables_broken_backend_for_desktop_fallback(monkeypatch):
    capture, _fake = _capture_stub(frame=Image.new("RGB", (100, 60)))

    class BrokenGraphicsCapture:
        def __init__(self, *_args):
            raise RuntimeError("WGC unavailable")

    monkeypatch.setattr(
        "maple_analyzer.graphics_capture.WindowsGraphicsCapture",
        BrokenGraphicsCapture,
    )

    assert GameWindowCapture._try_graphics_frame(capture, (20, 30, 120, 90)) is None
    assert capture._graphics_capture_disabled is True
    assert capture.capture_backend == "desktop"
    assert "WGC unavailable" in capture.graphics_capture_error


def test_graphics_frame_keeps_wgc_session_after_a_normal_no_frame_wait(monkeypatch):
    """A quiet game surface is not a backend failure or a five-second outage."""
    image = Image.new("RGB", (100, 60), "#123456")
    capture, _fake = _capture_stub(frame=image)

    class IntermittentGraphicsCapture:
        def __init__(self, *_args):
            self.item_size = None
            self.calls = 0
            self.closed = False

        def grab(self, timeout=1.2):
            assert timeout == 0.25
            self.calls += 1
            if self.calls == 1:
                raise GraphicsCaptureError(
                    "Windows Graphics Capture did not return a new frame"
                )
            return image.copy()

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "maple_analyzer.graphics_capture.WindowsGraphicsCapture",
        IntermittentGraphicsCapture,
    )

    assert GameWindowCapture._try_graphics_frame(capture, (20, 30, 120, 90)) is None
    active = capture._graphics_capture
    assert active is not None
    assert capture._graphics_capture_disabled is False
    assert capture.capture_backend == "windows-graphics-waiting"

    result = GameWindowCapture._try_graphics_frame(capture, (20, 30, 120, 90))
    assert result is not None
    assert capture._graphics_capture is active
    assert active.closed is False


def test_live_workers_share_only_the_same_fresh_graphics_presentation(monkeypatch):
    """One WGC presentation must initialize all workers that wake together."""
    image = Image.new("RGB", (100, 60), "#123456")
    capture, _fake = _capture_stub(frame=image)

    class OneFrameGraphicsCapture:
        def __init__(self, *_args):
            self.item_size = None
            self.calls = 0

        def grab(self, timeout=1.2):
            assert timeout == 0.25
            self.calls += 1
            if self.calls == 1:
                return image.copy()
            raise GraphicsCaptureError(
                "Windows Graphics Capture did not return a new frame"
            )

        def close(self):
            pass

    monkeypatch.setattr(
        "maple_analyzer.graphics_capture.WindowsGraphicsCapture",
        OneFrameGraphicsCapture,
    )

    first = GameWindowCapture._try_graphics_frame(capture, (20, 30, 120, 90))
    sibling = GameWindowCapture._try_graphics_frame(capture, (20, 30, 120, 90))

    assert first is not None
    assert sibling is not None
    assert sibling.getpixel((0, 0)) == (18, 52, 86)
    assert capture.capture_backend == "windows-graphics-shared"
    assert capture._graphics_capture.calls == 1

    # This is a concurrent-consumer hand-off, not a stale live cache.  The
    # next scheduled live scan must go back to WGC and visibly wait for a
    # presentation when the short hand-off window has passed.
    capture._last_live_graphics_frame_at = (
        time.monotonic() - LIVE_GRAPHICS_SHARED_FRAME_SECONDS - 0.01
    )
    assert GameWindowCapture._try_graphics_frame(capture, (20, 30, 120, 90)) is None
    assert capture._graphics_capture.calls == 2


def test_graphics_grab_never_returns_a_stale_last_frame():
    """A stalled frame stream must be visible to the capture fallback.

    Returning the previous bitmap here makes every OCR field appear healthy
    while EXP/HP/MP and economy values are frozen at the first frame.
    """
    capture = WindowsGraphicsCapture.__new__(WindowsGraphicsCapture)
    capture._frame_ready = threading.Event()
    capture._take_pending_frame = lambda: None

    with pytest.raises(GraphicsCaptureError, match="new frame"):
        capture.grab(timeout=0.05)


def test_graphics_grab_returns_a_recent_frame_only_when_context_opts_in():
    capture = WindowsGraphicsCapture.__new__(WindowsGraphicsCapture)
    capture._lock = threading.Lock()
    capture._frame_ready = threading.Event()
    capture._take_pending_frame = lambda: None
    capture._last_image = Image.new("RGB", (3, 2), "#123456")
    capture._last_image_at = time.monotonic()

    result = capture.grab(
        timeout=0.05,
        allow_stale=True,
        max_stale_seconds=1.0,
    )

    assert result.size == (3, 2)
    assert result.getpixel((0, 0)) == (18, 52, 86)
    assert capture._last_grab_was_stale is True


def test_graphics_display_grab_can_use_a_recent_target_frame_without_waiting():
    """Idle HUD display must not hold the capture lock for a missing redraw."""
    capture = WindowsGraphicsCapture.__new__(WindowsGraphicsCapture)
    capture._lock = threading.Lock()
    capture._frame_ready = threading.Event()
    capture._take_pending_frame = lambda: None
    capture._last_image = Image.new("RGB", (3, 2), "#123456")
    capture._last_image_at = time.monotonic()

    started = time.monotonic()
    result = capture.grab(
        timeout=1.0,
        allow_stale=True,
        max_stale_seconds=1.0,
        prefer_stale=True,
    )

    assert result.size == (3, 2)
    assert time.monotonic() - started < 0.1


def test_context_keeps_its_last_fresh_target_frame_during_wgc_reconnect():
    capture = GameWindowCapture.__new__(GameWindowCapture)
    capture._remember_capture_geometry = lambda _rect: None
    capture._try_graphics_frame = lambda _rect, **_kwargs: None
    capture._try_print_window_frame = lambda _rect: None
    capture._last_context_frame = Image.new("RGB", (100, 60), "#123456")
    capture._last_context_frame_client_size = (100, 60)

    result = capture._try_window_frame(
        (0, 0, 100, 60),
        allow_stale_graphics=True,
    )

    assert result is not None
    assert result.getpixel((0, 0)) == (18, 52, 86)
    assert capture.capture_backend == "windows-graphics-context-cache"


def test_live_window_frame_never_uses_the_context_cache():
    capture = GameWindowCapture.__new__(GameWindowCapture)
    capture._remember_capture_geometry = lambda _rect: None
    capture._try_graphics_frame = lambda _rect, **_kwargs: None
    capture._try_print_window_frame = lambda _rect: None
    capture._last_context_frame = Image.new("RGB", (100, 60), "#123456")
    capture._last_context_frame_client_size = (100, 60)

    assert capture._try_window_frame((0, 0, 100, 60)) is None
