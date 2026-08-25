"""Keep tkinter discoverable when the build machine has no GUI desktop."""
from __future__ import annotations

from pathlib import Path
import sys


def pre_find_module_path(hook_api) -> None:
    # PyInstaller's built-in hook empties search_dirs when its isolated
    # Tk() probe cannot create a window.  That probe is not a valid reason to
    # omit tkinter: the application will create its window on the user's
    # Windows desktop, and the Tcl/Tk runtime is bundled explicitly by the
    # project spec.
    lib_dir = Path(sys.base_prefix) / "Lib"
    if (lib_dir / "tkinter").is_dir():
        hook_api.search_dirs = [str(lib_dir)]
