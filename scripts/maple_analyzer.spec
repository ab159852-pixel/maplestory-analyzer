# PyInstaller spec for the MapleStoryAnalyzer HUD.
#
# Build on Windows, from the repo root, inside the project venv:
#   .venv\Scripts\pyinstaller scripts\maple_analyzer.spec --noconfirm
#
# Output: dist\MapleStoryAnalyzer\MapleStoryAnalyzer.exe (one-folder build --
# faster startup than --onefile, and rapidocr's ONNX models are large enough
# that unpacking them to a temp dir on every launch isn't worth it).

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all

block_cipher = None
repo_root = Path(SPECPATH).resolve().parent

# Tk calculates widget sizes in logical pixels.  Without an explicit
# per-monitor DPI declaration, Windows can scale the child widgets after Tk
# has laid them out while leaving the outer window at its unscaled size; that
# is what caused the right-side HUD/context buttons to be clipped on this
# machine.  Keep the manifest in the build definition so the .exe behaves the
# same as the source run at 100%, 125%, and 150% display scaling.
DPI_MANIFEST = r'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity version="1.0.41.0" processorArchitecture="*" name="MapleStoryAnalyzer" type="win32"/>
  <description>Maple Insight live efficiency HUD</description>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
    </windowsSettings>
  </application>
</assembly>'''

datas = []
binaries = []
hiddenimports = []

# ``icon=`` embeds the executable's shell icon, while Tk needs the same ICO
# available at runtime for the title bar/taskbar icon.
datas.append((str(repo_root / "assets" / "maple_insight.ico"), "."))
datas.append((str(repo_root / "assets" / "maple_insight_dark_blue.json"), "."))
datas.append((
    str(repo_root / "assets" / "paddle_models" / "en_PP-OCRv4_mobile_rec" / "inference.onnx"),
    "paddle_models/en_PP-OCRv4_mobile_rec",
))

for pkg in ("customtkinter", "rapidocr_onnxruntime", "winrt", "mss"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# CustomTkinter imports ThemeManager at module import time and that manager
# loads ``blue.json`` before OverlayApp can switch to the project-owned theme.
# PyInstaller's collect_all has not been reliable across CustomTkinter/Python
# combinations, so preserve every built-in theme explicitly.  Missing one of
# these files makes the frozen exe fail before the first window is painted.
import customtkinter

builtin_theme_dir = Path(customtkinter.__file__).resolve().parent / "assets" / "themes"
for theme_path in builtin_theme_dir.glob("*.json"):
    datas.append((str(theme_path), "customtkinter/assets/themes"))

# customtkinter is built on top of tkinter.  Some portable Python runtimes
# make tkinter.Tk() unavailable to PyInstaller's isolated probe (for example
# when the build process has no interactive desktop), which makes PyInstaller
# silently exclude tkinter and its Tcl/Tk runtime.  That produces an exe that
# fails immediately with "No module named 'tkinter'".  Add the complete Tk
# runtime explicitly so the build is independent of that probe.
python_root = Path(sys.base_prefix)
tkinter_root = python_root / "Lib" / "tkinter"
tcl_root = python_root / "tcl"
dll_root = python_root / "DLLs"

hiddenimports += [
    # graphics_capture.py is imported only when a real HWND is available,
    # so PyInstaller cannot discover it from the normal import graph.
    "maple_analyzer.graphics_capture",
    "winrt.windows.ai.machinelearning",
    "winrt.windows.graphics",
    "winrt.windows.graphics.capture",
    "winrt.windows.graphics.capture.interop",
    "winrt.windows.graphics.directx",
    "winrt.windows.graphics.imaging",
    "tkinter",
    "_tkinter",
    "tkinter.constants",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.simpledialog",
    "tkinter.ttk",
    # pywin32 modules are imported lazily by capture.py on Windows, so keep
    # them explicit even when the module graph does not follow that branch.
    "win32api",
    "win32con",
    "win32gui",
    "win32process",
    "win32ui",
    "pywintypes",
    "pythoncom",
]

if not tkinter_root.is_dir() or not tcl_root.is_dir():
    raise SystemExit(
        f"Python Tcl/Tk runtime was not found below {python_root}. "
        "Use a standard Windows Python installation to build the app."
    )

# The runtime hook supplied by PyInstaller looks for these exact destination
# names when setting TCL_LIBRARY and TK_LIBRARY in the frozen process.
for source, destination in (
    (tcl_root / "tcl8.6", "_tcl_data"),
    (tcl_root / "tk8.6", "_tk_data"),
):
    if source.is_dir():
        datas.append((str(source), destination))

# Include the extension and its native Tcl/Tk libraries beside the frozen
# executable.  The glob keeps this compatible with Python 3.10–3.13 while
# avoiding unrelated DLLs in the Python installation.
for pattern in ("_tkinter*.pyd", "tcl*.dll", "tk*.dll"):
    for source in dll_root.glob(pattern):
        binaries.append((str(source), "."))

a = Analysis(
    [str(repo_root / "scripts" / "run_overlay.py")],
    pathex=[str(repo_root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(repo_root / "scripts" / "pyinstaller_hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MapleStoryAnalyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(repo_root / "assets" / "maple_insight.ico"),
    manifest=DPI_MANIFEST,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MapleStoryAnalyzer",
)
