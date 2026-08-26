"""Small, dependency-free runtime diagnostics for the packaged HUD.

The Windows build is a windowed PyInstaller application, so an uncaught
exception normally has no console in which to leave a useful traceback.  This
module installs process, thread, Tk callback, and faulthandler diagnostics in
the real entrypoint.  Logging is deliberately best-effort: a failed log write
must never become the reason the analyzer exits.
"""
from __future__ import annotations

import faulthandler
import os
from pathlib import Path
import sys
import tempfile
import threading
import traceback
from typing import TextIO


LOG_FILE_NAME = "MapleStoryAnalyzer-crash.log"
_FAULT_HANDLE: TextIO | None = None
_INSTALLED = False
_ORIGINAL_SYS_EXCEPTHOOK = sys.excepthook
_ORIGINAL_THREADING_EXCEPTHOOK = threading.excepthook


def crash_log_path() -> Path:
    """Return the stable per-user temp path used by the packaged app."""
    return Path(tempfile.gettempdir()) / LOG_FILE_NAME


def _write_block(title: str, lines: list[str]) -> None:
    try:
        path = crash_log_path()
        with path.open("a", encoding="utf-8", errors="replace") as output:
            output.write("\n===== " + title + " =====\n")
            output.writelines(lines)
            if not lines or not lines[-1].endswith("\n"):
                output.write("\n")
    except Exception:
        # Diagnostics are intentionally non-fatal.
        pass


def log_exception(title: str, exc: BaseException) -> None:
    """Append one exception and traceback without ever propagating."""
    try:
        lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    except Exception:
        lines = [f"{type(exc).__name__}: {exc!r}\n"]
    _write_block(title, lines)


def log_message(title: str, message: object) -> None:
    """Append a short diagnostic message without requiring an exception."""
    _write_block(title, [f"{message!r}\n"])


def _sys_exception_hook(exc_type, exc_value, exc_traceback) -> None:
    try:
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    except Exception:
        lines = [f"{exc_type!r}: {exc_value!r}\n"]
    _write_block("uncaught main-thread exception", lines)
    # In a console/source run, preserve Python's normal traceback.  The
    # windowed build has no useful stderr and should not open an unhelpful
    # native error dialog after the traceback is safely saved.
    if not getattr(sys, "frozen", False):
        try:
            _ORIGINAL_SYS_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
        except Exception:
            pass


def _thread_exception_hook(args: threading.ExceptHookArgs) -> None:
    try:
        name = getattr(args.thread, "name", "unknown-thread")
        lines = traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        _write_block(f"uncaught worker exception: {name}", lines)
    except Exception:
        pass
    if not getattr(sys, "frozen", False):
        try:
            _ORIGINAL_THREADING_EXCEPTHOOK(args)
        except Exception:
            pass


def install_exception_logging() -> Path:
    """Install idempotent process/thread/fault diagnostics and return the path."""
    global _FAULT_HANDLE, _INSTALLED
    path = crash_log_path()
    if _INSTALLED:
        return path
    _INSTALLED = True
    try:
        # Keep the file useful rather than allowing a long-running test build
        # to grow forever.  Rename is recoverable and only affects our own log.
        if path.is_file() and path.stat().st_size > 2 * 1024 * 1024:
            path.replace(path.with_suffix(".previous.log"))
    except Exception:
        pass
    sys.excepthook = _sys_exception_hook
    threading.excepthook = _thread_exception_hook
    try:
        _FAULT_HANDLE = path.open("a", encoding="utf-8", errors="replace")
        faulthandler.enable(file=_FAULT_HANDLE, all_threads=True)
    except Exception:
        _FAULT_HANDLE = None
    return path


def install_tk_exception_logging(root) -> None:
    """Route Tk callback exceptions to the same file without crashing Tk."""
    def report_callback_exception(exc_type, exc_value, exc_traceback):
        try:
            _write_block(
                "Tk callback exception",
                traceback.format_exception(exc_type, exc_value, exc_traceback),
            )
        except Exception:
            pass

    try:
        root.report_callback_exception = report_callback_exception
    except Exception as exc:
        log_exception("unable to install Tk exception handler", exc)

