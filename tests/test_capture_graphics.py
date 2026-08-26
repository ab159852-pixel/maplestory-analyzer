"""Tests for the compositor-backed window capture adapter."""
from __future__ import annotations

from types import SimpleNamespace
import threading

import pytest

from PIL import Image

from maple_analyzer.capture import GameWindowCapture
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
            assert timeout == 1.2
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
