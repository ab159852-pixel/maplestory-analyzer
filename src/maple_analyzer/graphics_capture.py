"""Windows Graphics Capture backend for unobscured window capture.

The normal ``mss`` path reads pixels from the desktop compositor.  A browser,
the analyzer HUD, or any other foreground window can therefore cover the game
and become the pixels OCR sees.  Windows Graphics Capture targets the game's
HWND directly and keeps a small frame pool, which is the same class of capture
used by modern recording/streaming tools.

This module is imported lazily by :mod:`maple_analyzer.capture` so static-image
tests and non-Windows development do not need WinRT installed.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Any

import numpy as np
from PIL import Image


class GraphicsCaptureError(RuntimeError):
    """Raised when Windows Graphics Capture cannot provide a frame."""


class WindowsGraphicsCapture:
    """Persistent Windows Graphics Capture session for one top-level HWND."""

    def __init__(
        self, hwnd: int, capture_size: tuple[int, int] | None = None
    ) -> None:
        try:
            from winrt.windows.ai.machinelearning import (
                LearningModelDevice,
                LearningModelDeviceKind,
            )
            from winrt.windows.graphics.capture import Direct3D11CaptureFramePool
            from winrt.windows.graphics.capture.interop import create_for_window
            from winrt.windows.graphics import SizeInt32
            from winrt.windows.graphics.directx import DirectXPixelFormat
        except Exception as exc:  # pragma: no cover - depends on Windows install
            raise GraphicsCaptureError("Windows Graphics Capture dependencies unavailable") from exc

        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._pending_frame: Any | None = None
        # The live status/economy paths must never reuse this image: a stalled
        # capture would otherwise look like healthy, frozen telemetry.  The
        # low-frequency map/job caller can opt in to a bounded reuse window,
        # because it is better to keep a confirmed map label than turn it into
        # "not detected" simply because the static game scene did not present
        # another compositor frame.
        self._last_image: Image.Image | None = None
        self._last_image_at = 0.0
        self._closed = False

        try:
            # LearningModelDevice creates a D3D11 device without requiring a
            # camera or a visible picker window.  The device must be created on
            # the same worker thread that owns the capture session.
            device = LearningModelDevice(
                LearningModelDeviceKind.DIRECTX_HIGH_PERFORMANCE
            ).direct3d11_device
            self._item = create_for_window(hwnd)
            item_size = self._item.size
            item_width = int(item_size.width)
            item_height = int(item_size.height)
            self._item_size = (
                (item_width, item_height)
                if item_width > 0 and item_height > 0
                else None
            )
            # A few Win32 window classes report a zero GraphicsCaptureItem
            # size until their first compositor frame.  The HWND frame size
            # supplied by GameWindowCapture is only a fallback in that case;
            # using it unconditionally would resize a full-window item before
            # the client crop gets a chance to remove its title bar.
            if self._item_size is not None:
                width, height = self._item_size
            else:
                width, height = capture_size or (item_width, item_height)
            if width <= 0 or height <= 0:
                raise GraphicsCaptureError(
                    f"invalid capture size {(width, height)} for HWND {hwnd}"
                )
            self._capture_size = (int(width), int(height))
            self._frame_pool = Direct3D11CaptureFramePool.create_free_threaded(
                device,
                DirectXPixelFormat.B8_G8_R8_A8_UINT_NORMALIZED,
                2,
                SizeInt32(*self._capture_size),
            )
            self._frame_token = self._frame_pool.add_frame_arrived(
                self._on_frame_arrived
            )
            self._session = self._frame_pool.create_capture_session(self._item)
            # These properties prevent an artificial yellow capture
            # border/cursor from entering OCR. They are optional on older
            # Windows builds; failure to set one must not disable HWND capture.
            with contextlib.suppress(Exception):
                self._session.is_border_required = False
            with contextlib.suppress(Exception):
                self._session.is_cursor_capture_enabled = False
            self._session.start_capture()
        except Exception as exc:
            self.close()
            raise GraphicsCaptureError("unable to start Windows Graphics Capture") from exc

    @property
    def size(self) -> tuple[int, int]:
        return self._capture_size

    @property
    def item_size(self) -> tuple[int, int] | None:
        """Native physical size of the GraphicsCaptureItem, if available."""
        return self._item_size

    def _on_frame_arrived(self, sender: Any, _args: Any) -> None:
        if self._closed:
            return
        try:
            frame = sender.try_get_next_frame()
        except Exception:
            return
        if frame is None:
            return
        old_frame = None
        with self._lock:
            old_frame = self._pending_frame
            self._pending_frame = frame
            self._frame_ready.set()
        if old_frame is not None:
            with contextlib.suppress(Exception):
                old_frame.close()

    def _take_pending_frame(self) -> Any | None:
        with self._lock:
            frame = self._pending_frame
            self._pending_frame = None
            if frame is None:
                self._frame_ready.clear()
            return frame

    def grab(
        self,
        timeout: float = 0.8,
        *,
        allow_stale: bool = False,
        max_stale_seconds: float = 0.0,
    ) -> Image.Image:
        """Return a newly arrived frame.

        The default always requires a new frame.  This protects EXP, HP/MP,
        and economy tracking from treating a stalled capture as live data.
        The map/job context caller may explicitly opt in to a recent cached
        target-window frame; it is low-frequency metadata, not accounting,
        and receives a new frame first whenever one arrives.
        """
        frame = self._take_pending_frame()
        if frame is None:
            self._frame_ready.wait(max(0.05, timeout))
            frame = self._take_pending_frame()

        if frame is None:
            if allow_stale:
                with self._lock:
                    last_image = getattr(self, "_last_image", None)
                    last_at = float(getattr(self, "_last_image_at", 0.0))
                    age = time.monotonic() - last_at
                    if (
                        last_image is not None
                        and max_stale_seconds > 0
                        and 0 <= age <= max_stale_seconds
                    ):
                        return last_image.copy()
            raise GraphicsCaptureError(
                "Windows Graphics Capture did not return a new frame"
            )

        try:
            image = self._frame_to_image(frame)
        except Exception as exc:
            raise GraphicsCaptureError("unable to convert Windows Graphics Capture frame") from exc
        finally:
            with contextlib.suppress(Exception):
                frame.close()

        with self._lock:
            self._last_image = image.copy()
            self._last_image_at = time.monotonic()
        return image

    @staticmethod
    def _frame_to_image(frame: Any) -> Image.Image:
        from winrt.windows.graphics.imaging import (
            BitmapBufferAccessMode,
            SoftwareBitmap,
        )

        async def copy_surface():
            return await SoftwareBitmap.create_copy_from_surface_async(frame.surface)

        bitmap = asyncio.run(copy_surface())
        try:
            buffer = bitmap.lock_buffer(BitmapBufferAccessMode.READ)
            description = buffer.get_plane_description(0)
            reference = buffer.create_reference()
            raw = np.frombuffer(reference, dtype=np.uint8).copy()
            start = int(description.start_index)
            stride = int(description.stride)
            width = int(description.width)
            height = int(description.height)
            plane = raw[start : start + stride * height]
            rows = np.frombuffer(plane, dtype=np.uint8).reshape(height, stride)
            bgra = rows[:, : width * 4].reshape(height, width, 4)
            # The capture pool uses B8G8R8A8_UINT_NORMALIZED.
            rgb = bgra[:, :, :3][:, :, ::-1].copy()
            return Image.fromarray(rgb, mode="RGB")
        finally:
            with contextlib.suppress(Exception):
                bitmap.close()

    def close(self) -> None:
        self._closed = True
        with self._lock:
            pending = self._pending_frame
            self._pending_frame = None
            self._last_image = None
            self._last_image_at = 0.0
            self._frame_ready.set()
        if pending is not None:
            with contextlib.suppress(Exception):
                pending.close()
        for attribute in ("_session", "_frame_pool"):
            value = getattr(self, attribute, None)
            if value is not None:
                with contextlib.suppress(Exception):
                    value.close()
                setattr(self, attribute, None)
