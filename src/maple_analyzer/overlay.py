"""Always-on-top HUD: capture -> OCR -> parse -> session -> redraw, on a timer.

Per-tick timing (2026-08-17 rework, measured against the live game): the
original whole-panel detection+recognition OCR pass was ~600-680ms/tick --
detection (finding text regions) was the entire cost, not recognition. Since
regions.py's FIELD_BOXES already pins down exactly where each field's text
is, detection was pure waste; switched to four small recognition-only calls
(no detection stage) on individually pre-cropped fields, ~15ms each, ~60ms
total. Capture itself is ~3.5ms. `_tick()` now also computes its own elapsed
work time and schedules the next call at `TARGET_MS - elapsed`, floored at
0, instead of the old fixed post-delay (which added TARGET_MS on top of
whatever the work took, so it could never reach the target rate no matter
how fast OCR got) -- this is what actually makes the real cycle approach
TARGET_MS rather than merely bound the *added* delay to it.

UI (2026-08-17 rework, restyled same day per an HTML design preview the user
approved): CustomTkinter, three tabs (Live/History/Settings). Status/session
timer live in their own strip at the top of Live (a pill + a chip, not just
another text row); stats and session info sit in aligned grids with tabular
numerals; History renders each session as a card, not scrollback text. See
~/.claude/notes/maplestory-analyzer/final-spec-2026-08-17.md Section 3 for
the full spec. This module still only calls Session's public methods and
reads StatSnapshot/SessionSummary fields -- the capture/OCR/parser engine
(capture.py/ocr.py/parser.py/regions.py) is untouched by this rework, per
the hard UI/engine separation rule in that same doc.

Settings + i18n (2026-08-17, later same day): all UI-layer settings live in
one `Settings` struct (settings.py) instead of scattered instance attributes,
so a future persistence layer can load/save it wholesale. All user-facing
strings route through `self._t(key)` into i18n.py's translation table (English
+ Traditional Chinese, zh default) instead of literals inline here -- static
widgets built once register themselves in `self._i18n_labels` so a language
switch can walk the list and reconfigure every one of them, tabs get renamed
via CTkTabview.rename(), and History cards (built dynamically per session) are
simply torn down and rebuilt from `self._session_history` on switch.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, simpledialog
from typing import Protocol

import customtkinter as ctk

from .capture import PANEL_OBSCURED
from .drop_lookup import (
    DropLookupError,
    MapDropSummary,
    DropItem,
    fetch_map_drop_summary,
    format_probability,
    format_quantity,
    MAPS_PAGE_URL,
    monster_page_url,
    normalize_map_name,
)
from .economy import EconomyTracker, parse_mesos_amount, parse_slot_count
from .i18n import Lang, t
from .monitor import BackgroundMonitor
from .ocr import StatPanelOcr
from .parser import StatSnapshot, parse_fields
from .rate import Session, SessionSummary
from .regions import (
    PICKUP_LINE_BOXES,
    PICKUP_LINE_HEIGHT,
    PICKUP_LINE_TOP_OFFSET,
    SHORTCUT_SLOT_BOXES,
)
from .settings import PotionSlotConfig, Settings
from .storage import export_history_csv, load_history, load_settings, save_history, save_settings
from .updates import UpdateError, UpdateInfo, check_for_update, download_update, schedule_update
from .version import APP_DISPLAY_NAME, APP_VERSION

# The console's codepage (e.g. cp950 Traditional Chinese) can't represent
# every character OCR might misread out of the game's UI -- printing one
# used to raise UnicodeEncodeError and silently kill the tick loop (see
# _tick's try/except below for the other half of this fix). errors="replace"
# swaps unencodable characters for '?' instead of crashing.
if sys.stdout is None:
    # PyInstaller's windowed build (console=False) has no stdout/stderr at
    # all (both are None), which crashes not just .reconfigure() below but
    # every bare print() elsewhere in this module (tick-error/debug
    # logging) the moment they run. Swap in a no-op sink so those stay
    # harmless instead of taking down the app.
    #
    # encoding/errors are NOT optional here: open() defaults to the locale
    # codepage with errors='strict', i.e. cp950 on this zh-TW machine. The
    # PP-OCR recognition dictionary is largely *Simplified* Chinese, so a
    # garbage read (game window obscured, floating damage numbers over the
    # panel) routinely produces characters Big5/cp950 cannot encode -- and
    # printing one raised UnicodeEncodeError straight through the sink,
    # killing the tick loop. Same errors="replace" the console path below
    # has always had; the windowed build was the only place missing it.
    sys.stdout = sys.stderr = open(
        os.devnull, "w", encoding="utf-8", errors="replace"
    )
else:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr is not None:
        # Tk writes uncaught-callback tracebacks here, and a traceback can
        # carry the same unencodable OCR text in its repr.
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

if sys.platform == "win32":
    # Without declaring DPI awareness, Windows scales the whole rendered
    # window as a bitmap after the fact -- Tk still thinks the window is
    # e.g. 420 logical px, but the OS-scaled result doesn't match, and
    # widget content ends up clipped past the visible window edge (observed
    # live: value labels cut off mid-digit). Declaring per-monitor-v2
    # awareness lets Windows and Tk agree on actual pixel dimensions instead.
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        pass

TARGET_MS = 300  # target full status/OCR cycle -- 3.33Hz
# Pickup toasts are brief lower-right notifications. Keep the fallback path
# fast too; the threaded monitor uses Settings.pickup_interval_ms directly.
AUX_SCAN_MS = 200
PICKUP_DETECTION_MS = 200
POTION_PROJECTION_MIN_SECONDS = 60.0  # avoid a first-drink spike in a short sample
SCALE_STEP_PCT = 10
SCALE_MIN_PCT = 50
SCALE_MAX_PCT = 150

# Live tab's Pause/Resume/Start + Restart button row -- see _apply_run_state.
BUTTON_HEIGHT = 34
STOPPED_BUTTON_WIDTH = 96  # Start alone, centered -- smaller than the two-button width

# Color tokens: an obsidian/sapphire base with restrained teal, violet and
# gold accents.  The extra border/raised tokens keep cards visually layered
# instead of making the whole window read as one flat dark rectangle.
# Night-pink glass palette.  CustomTkinter does not provide per-widget alpha
# compositing, so the jelly/glow effect is built from close translucent-like
# surfaces, bright hairline borders, and high-contrast hover/selected layers.
BG = "#0f0915"
SURFACE = "#1e1326"
SURFACE_2 = "#281832"
SURFACE_RAISED = "#33203e"
SURFACE_ELEVATED = "#42284d"
BORDER = "#784866"
BORDER_SOFT = "#432842"
INK = "#fff1fa"
INK_DIM = "#e0b6d1"
INK_FAINT = "#a47b99"
ACCENT = "#ff8dcc"
ACCENT_INK = "#351126"
VIOLET = "#ff79c8"
VIOLET_HOVER = "#ffb0e1"
TAB_SURFACE = "#160d1e"
TAB_UNSELECTED = "#321a38"
TAB_HOVER = "#633052"
GLOW_BORDER = "#c764a5"
HP_COLOR = "#ff789d"
MP_COLOR = "#8cb4ff"
EXP_COLOR = "#ffd27a"
OK_COLOR = "#78efc2"
TRACK_BG = "#3c1e31"

# Chrome text (tabs, headers, buttons, switches, kv labels -- anything that
# can carry translated content) picks its font family from the active
# language via OverlayApp._font(); Segoe UI has no real Traditional Chinese
# glyphs of its own (falls back to a system CJK font Windows picks for you,
# inconsistent with the rest of the UI), so zh uses Microsoft JhengHei
# (Windows' standard Traditional Chinese UI font) instead.
_FONT_FAMILY: dict[Lang, str] = {"en": "Segoe UI", "zh": "Microsoft JhengHei"}

# Fixed English-only chrome that never carries translated text (the game's
# own on-screen abbreviations LV/HP/MP/EXP, and the +/- scale stepper) stays
# on a plain Segoe UI tuple -- no language switching needed for pure ASCII.
_FONT_LABEL = ("Segoe UI", 10, "bold")
_FONT_UI_BOLD = ("Segoe UI", 13, "bold")

# Pure-numeric value labels (HP/MP/EXP readouts, session EXP diffs, history
# card numbers) stay on Consolas regardless of language -- they never render
# CJK text, and Consolas' monospacing is what keeps tabular digits aligned.
_FONT_MONO = ("Consolas", 12)
_FONT_MONO_SM = ("Consolas", 10)
_FONT_MONO_BOLD = ("Consolas", 12, "bold")

FLOATING_METRIC_SPECS = (
    ("proj_exp", "kv_proj_exp_interval", EXP_COLOR),
    ("eta", "kv_eta", EXP_COLOR),
    ("proj_mesos", "kv_mesos_projected", EXP_COLOR),
    ("proj_potion_cost", "kv_potion_cost_projected", OK_COLOR),
    ("level", "settings_show_level", EXP_COLOR),
    ("hp", "settings_show_hp", HP_COLOR),
    ("mp", "settings_show_mp", MP_COLOR),
    ("exp", "settings_show_exp", EXP_COLOR),
    ("exp_diff", "settings_show_exp_diff", EXP_COLOR),
    ("exp_rate", "settings_show_exp_rate", EXP_COLOR),
    ("mesos", "kv_mesos", EXP_COLOR),
    ("hp_potions", "kv_hp_potions", HP_COLOR),
    ("mp_potions", "kv_mp_potions", MP_COLOR),
    ("shortcut_inventory", "kv_shortcut_inventory", OK_COLOR),
    ("shared_potions", "kv_shared_potions", EXP_COLOR),
    ("hp_recovery", "kv_hp_recovery", HP_COLOR),
    ("mp_recovery", "kv_mp_recovery", MP_COLOR),
    ("hp_recovery_savings", "kv_hp_recovery_savings", HP_COLOR),
    ("mp_recovery_savings", "kv_mp_recovery_savings", MP_COLOR),
    ("hp_loss", "settings_show_hp_loss", HP_COLOR),
    ("mp_loss", "settings_show_mp_loss", MP_COLOR),
)


def _persist_settings(settings: Settings) -> None:
    """Best-effort preference save; a disk problem must not stop the HUD."""
    with contextlib.suppress(Exception):
        save_settings(settings)


def _persist_history(history: list[SessionSummary]) -> None:
    """Best-effort history save; rendering and tracking remain the priority."""
    with contextlib.suppress(Exception):
        save_history(history)


def _maybe_persist_settings(app: object) -> None:
    if getattr(app, "_state_persistence", False):
        _persist_settings(getattr(app, "_settings"))


def _maybe_persist_history(app: object) -> None:
    if getattr(app, "_state_persistence", False):
        _persist_history(getattr(app, "_session_history"))


class PanelSource(Protocol):
    def grab_fields(self) -> dict:
        ...

    def grab_auxiliary(self) -> dict:
        ...


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _fmt_loss(loss: int) -> str:
    return f"-{loss}" if loss > 0 else "0"


def _fmt_summary(s: SessionSummary, index: int) -> str:
    # Console/debug log only, not shown in the UI -- deliberately left in
    # plain English regardless of self._settings.language.
    diff = s.exp_diff
    diff_s = f"+{diff:,}" if diff is not None else "?"
    pct_diff = s.exp_pct_diff
    pct_s = f" (+{pct_diff:.2f}%)" if pct_diff is not None else ""
    start_s = f"{s.start_exp:,}" if s.start_exp is not None else "?"
    end_s = f"{s.end_exp:,}" if s.end_exp is not None else "?"
    dur_min = s.duration_s / 60
    if s.interval_minutes is not None and abs(dur_min - s.interval_minutes) > 0.05:
        dur_s = f"{dur_min:.1f}m of {s.interval_minutes:.0f}m, restarted early"
    else:
        dur_s = f"{dur_min:.1f}m"
    return (
        f"#{index} ({dur_s}): "
        f"EXP {start_s} -> {end_s} ({diff_s}{pct_s})  "
        f"HP {_fmt_loss(s.hp_loss)}  MP {_fmt_loss(s.mp_loss)}"
    )


class OverlayApp:
    def __init__(self, source: PanelSource):
        self._source = source
        # RapidOCR loads ONNX models and can take several seconds on first
        # start.  Constructing it before Tk's mainloop made the application
        # look frozen.  The model is now created on a daemon worker while the
        # shell/HUD remains responsive.
        self._ocr: StatPanelOcr | None = None
        self._ocr_loading = True
        self._ocr_error: str | None = None
        self._ocr_result: tuple[StatPanelOcr | None, str | None] | None = None
        self._monitor: BackgroundMonitor | None = None
        self._session = Session()
        self._state_persistence = True
        # Preferences and finalized sessions survive an app restart.  Both
        # loaders validate their own fields and fall back safely, so a bad
        # optional state file cannot prevent the OCR HUD from opening.
        self._settings = load_settings()
        if self._settings.track_potions and "shortcut_inventory" not in self._settings.floating_fields:
            self._settings.floating_fields.append("shortcut_inventory")
        self._session_history: list[SessionSummary] = load_history()
        self._economy = EconomyTracker(
            self._settings.potion_slots,
            self._settings.potion_recovery_hp_default,
            self._settings.potion_recovery_mp_default,
        )

        self._last: StatSnapshot = StatSnapshot(None, None, None, None, None, None, None)
        # Newest-first: History cards are inserted at index 0 rather than
        # appended, so this tracks the card widgets in display order (index 0
        # = topmost/newest) to pack each new one with before=.
        self._history_cards: list[ctk.CTkFrame] = []
        # Static widgets whose text is a plain translated string (no
        # per-tick data baked in) register themselves here as they're built,
        # so _apply_language() can walk this list and reconfigure every one
        # instead of _build_*_tab needing to be re-run from scratch.
        self._i18n_labels: list[tuple[ctk.CTkBaseClass, str, int, bool]] = []
        # Guards _do_tick's finalize-on-timeout check against the rename
        # dialog's nested event loop -- see _do_tick and _on_rename_clicked.
        self._modal_open = False
        # "running" / "paused" / "stopped" -- see _on_pause_button_clicked and
        # _finalize_and_maybe_stop. "stopped" reached via the timer is
        # implemented by pausing the already-running Session (its clock
        # freezes and record() no-ops, exactly what "stopped" needs) rather
        # than adding a third Session state; starting "stopped" here needs no
        # such call since nothing has fed this fresh Session a tick yet --
        # _do_tick simply doesn't call session.record() until Start is
        # clicked, so it can't begin calibrating or accumulating unasked.
        #
        # Starts stopped rather than tracking immediately on launch -- opening
        # the app (or the .exe) shouldn't silently start a session before the
        # user has actually arrived at the game and decided to track.
        self._run_state = "stopped"
        # Last capture failure message, so _do_tick can log state changes
        # instead of repeating the same line every 2s retry.
        self._last_capture_error: str | None = None
        self._last_aux_error: str | None = None
        self._last_client_size: tuple[int, int] | None = None
        # The first shortcut OCR result after Start/Resume is an inventory
        # baseline, never a potion-use event.
        self._potion_baseline_pending = True
        self._potion_baseline_samples: list[dict[str, int]] = []
        self._last_logged_shortcut_counts: dict[str, int] | None = None
        self._next_aux_scan = 0.0
        # Full pickup-feed detection is much slower than recognition-only
        # line crops.  It is a recovery path for clients where the measured
        # line rhythm misses the glyphs, not something to run on every 0.3s
        # status tick.
        self._next_pickup_detection = 0.0
        self._floating_mode = False
        self._saved_topmost = self._settings.topmost
        self._detected_job_name: str | None = None
        self._detected_map_name: str | None = None
        self._context_error: str | None = None
        self._session_job_name: str | None = None
        self._session_map_name: str | None = None
        # Drop lookup is deliberately lazy.  The public map/drop scripts are
        # several megabytes; keeping them out of startup is important for the
        # same reason OCR loading is deferred, and the result queue means a
        # slow network can never occupy Tk's event thread.
        self._drop_lookup_cache: dict[str, MapDropSummary] = {}
        self._drop_lookup_queue: queue.Queue[tuple[str, MapDropSummary | None, str | None]] = queue.Queue(maxsize=4)
        self._drop_lookup_requested_map: str | None = None
        self._drop_lookup_summary: MapDropSummary | None = None
        self._drop_lookup_loading_map: str | None = None
        self._drop_lookup_error: str | None = None
        self._drop_lookup_expanded = False
        self._drop_detail_expanded: set[str] = set()
        self._update_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=4)
        self._update_checking = False
        self._update_manual_check = False
        self._update_prompted_version: str | None = None

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        # Applied before the window/widgets are built so the default window
        # size below is already at the configured scale, not built at 100%
        # then rescaled after the fact.
        ctk.set_widget_scaling(self._settings.scale_pct / 100)
        ctk.set_window_scaling(self._settings.scale_pct / 100)

        self.root = ctk.CTk()
        self.root.title(APP_DISPLAY_NAME)
        self.root.attributes("-topmost", self._settings.topmost)
        self.root.attributes("-alpha", 1.0)
        self.root.configure(fg_color=BG)
        # 760 logical pixels leaves enough horizontal breathing room for the
        # card actions even when Windows/Tk lays widgets out at 125% DPI.
        # Live content remains vertically scrollable, so the larger canvas is
        # used for hierarchy rather than squeezing more rows into view.
        self._full_geometry = "760x900+40+40"
        self.root.geometry(self._full_geometry)
        self.root.minsize(420, 520)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._shell = ctk.CTkFrame(self.root, fg_color="transparent")
        self._shell.pack(fill="both", expand=True, padx=14, pady=14)
        shell = self._shell
        header = ctk.CTkFrame(
            shell, fg_color=SURFACE, corner_radius=18, height=72,
            border_width=1, border_color=BORDER,
        )
        header.pack(fill="x", padx=(0, 64), pady=(0, 8))
        header.pack_propagate(False)
        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.pack(side="left", fill="both", expand=True, padx=(16, 4))
        ctk.CTkLabel(
            brand, text="MSA", width=42, height=42, corner_radius=13,
            fg_color=VIOLET, text_color="#111326", font=("Segoe UI", 14, "bold"),
        ).pack(side="left", pady=14)
        title_box = ctk.CTkFrame(brand, fg_color="transparent")
        title_box.pack(side="left", fill="both", expand=True, padx=(12, 0))
        ctk.CTkLabel(
            title_box, text="MAPLE INSIGHT", anchor="w",
            text_color=INK, font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", pady=(14, 0))
        ctk.CTkLabel(
            title_box, text="CONTROL CENTER  ·  LIVE 0.3s SAMPLING", anchor="w",
            text_color=INK_FAINT, font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        self._hud_mode_button = ctk.CTkButton(
            header, text="HUD", command=self._toggle_floating_mode,
            width=76, height=30, corner_radius=10,
            fg_color=SURFACE_ELEVATED, hover_color=BORDER, text_color=INK,
            font=("Segoe UI", 10, "bold"),
        )
        self._hud_mode_button.pack(side="right", padx=(4, 16), pady=21)

        self._tabview = ctk.CTkTabview(
            shell, fg_color=BG, segmented_button_fg_color=SURFACE,
            segmented_button_selected_color=VIOLET,
            segmented_button_selected_hover_color=VIOLET_HOVER,
            segmented_button_unselected_color=SURFACE,
            segmented_button_unselected_hover_color=SURFACE_ELEVATED,
        )
        # Give the three primary destinations a deliberate control-center
        # treatment instead of the stock compact segmented-button look.
        self._tabview._segmented_button.configure(
            height=38,
            corner_radius=14,
            fg_color=TAB_SURFACE,
            selected_color=VIOLET,
            selected_hover_color=VIOLET_HOVER,
            unselected_color=TAB_UNSELECTED,
            unselected_hover_color=TAB_HOVER,
            border_width=1,
            text_color=INK,
            font=self._font(11, bold=True),
        )
        self._tabview._segmented_button.grid_configure(padx=6, pady=(0, 8))
        self._tabview.pack(fill="both", expand=True)
        # CTkTabview's tab name doubles as its segmented-button label and its
        # internal dict key -- there's no separate "id" to address a tab by,
        # so the translated string itself is the key. rename() (used by
        # _apply_language) swaps the key/label together and keeps the
        # frame/selection intact; this dict just tracks the current name per
        # logical tab so rename() always has both the old and new string.
        self._tab_names = {
            "live": t("tab_live", self._settings.language),
            "history": t("tab_history", self._settings.language),
            "settings": t("tab_settings", self._settings.language),
        }
        for name in self._tab_names.values():
            self._tabview.add(name)

        self._build_live_tab(self._tabview.tab(self._tab_names["live"]))
        self._build_history_tab(self._tabview.tab(self._tab_names["history"]))
        self._build_settings_tab(self._tabview.tab(self._tab_names["settings"]))
        self._tabview.set(self._tab_names["live"])  # CTkTabview defaults to the last-added tab otherwise
        self._rebuild_history_cards()
        self._apply_visibility()
        self._apply_run_state()
        self._build_floating_bar()

        # Start the model loader only after the first paint opportunity.  The
        # user gets a real window immediately instead of staring at a frozen
        # blank process while ONNX Runtime initializes.
        self.root.after(10, self._start_ocr_loader)
        # Drop lookup has its own lightweight UI-side pump.  It must not rely
        # solely on the OCR tick: users can open the lookup before pressing
        # Start, while the monitor is still loading or unavailable.
        self.root.after(100, self._poll_drop_lookup_results)
        # Updating is deliberately delayed until the first paint so a network
        # outage can never slow startup or make the initial window feel frozen.
        self.root.after(1500, self._start_update_check)
        self.root.after(250, self._poll_update_results)
        self._tick()

    # ---- i18n ------------------------------------------------------------

    def _t(self, key: str, **kwargs: object) -> str:
        return t(key, self._settings.language, **kwargs)

    def _localize_error(self, message: str) -> str:
        """Translate the known capture.py RuntimeError messages (game
        minimized / not found / stat panel covered) shown via _set_status_error --
        these are routine, expected states, not exceptional ones, so they
        deserve a real translation rather than leaking capture.py's raw
        English text into a zh-language UI. Anything unrecognized (a real
        bug, not a known game-window state) passes through unchanged."""
        if message == "game window is minimized":
            return self._t("status_error_minimized")
        if message.startswith("No window found with title containing"):
            return self._t("status_error_not_found")
        if message == PANEL_OBSCURED or message.startswith(f"{PANEL_OBSCURED};"):
            return self._t("status_error_obscured")
        return message

    def _font(self, size: int, bold: bool = False) -> tuple:
        """Chrome-text font at the given size, in the active language's font
        family (see _FONT_FAMILY). Use for any widget that renders translated
        text; pure-numeric value labels should use the module-level
        _FONT_MONO* constants instead (see their docstring)."""
        family = _FONT_FAMILY[self._settings.language]
        return (family, size, "bold") if bold else (family, size)

    def _scale_header_text(self) -> str:
        return self._t("settings_window_scale") + f" — {self._settings.scale_pct}%"

    def _interval_header_text(self) -> str:
        return self._t("settings_session_interval") + f" — {self._settings.window_min} {self._t('unit_min')}"

    def _sampling_header_text(self) -> str:
        return self._t(
            "settings_sampling_value", seconds=self._settings.sample_interval_ms / 1000
        )

    def _pickup_sampling_header_text(self) -> str:
        return self._t(
            "settings_pickup_sampling_value",
            seconds=self._settings.pickup_interval_ms / 1000,
        )

    def _opacity_header_text(self) -> str:
        return self._t("settings_floating_opacity") + f" — {self._settings.floating_opacity_pct}%"

    def _i18n(self, widget: ctk.CTkBaseClass, key: str, size: int, bold: bool = True) -> ctk.CTkBaseClass:
        """Set a widget's text + font from a translation key and register it
        for re-translation on language switch. Use for any widget whose text
        is *only* the translated string (no per-tick value baked in) --
        widgets that mix in live data (timer, status pill, kv values) instead
        call self._t(...)/self._font(...) directly wherever they're
        re-rendered every tick."""
        widget.configure(text=self._t(key), font=self._font(size, bold))
        self._i18n_labels.append((widget, key, size, bold))
        return widget

    def _apply_language(self, lang: Lang) -> None:
        if lang == self._settings.language:
            return
        self._settings.language = lang

        for logical, key in (("live", "tab_live"), ("history", "tab_history"), ("settings", "tab_settings")):
            old_name = self._tab_names[logical]
            new_name = self._t(key)
            if new_name != old_name:
                self._tabview.rename(old_name, new_name)
                self._tab_names[logical] = new_name

        for widget, key, size, bold in self._i18n_labels:
            widget.configure(text=self._t(key), font=self._font(size, bold))

        self._status_pill.configure(font=self._font(9, bold=True))
        self._timer_label.configure(font=self._font(10, bold=True))
        self._tabview._segmented_button.configure(font=self._font(11, bold=True))
        self._scale_header_label.configure(text=self._scale_header_text(), font=self._font(11, bold=True))
        self._interval_header_label.configure(text=self._interval_header_text(), font=self._font(11, bold=True))
        self._sampling_header_label.configure(text=self._sampling_header_text(), font=self._font(10, bold=True))
        self._pickup_sampling_header_label.configure(
            text=self._pickup_sampling_header_text(), font=self._font(10, bold=True)
        )
        self._floating_header_label.configure(text=self._opacity_header_text(), font=self._font(10, bold=True))
        self._hud_mode_button.configure(
            text=self._t("hud_button_exit" if self._floating_mode else "hud_button_enter")
        )
        # _pause_button's text depends on _run_state, not just language, so it
        # isn't in _i18n_labels -- _apply_run_state() re-derives it from
        # scratch, which also happens to pick up the new language/font.
        self._apply_run_state()

        # History cards mix translated chrome (SESSION #N, HP/MP LOSS) with
        # per-session data and aren't worth tracking widget-by-widget --
        # tearing down and rebuilding from the data we already keep is
        # simpler and this only happens on an explicit language switch, and
        # _append_history_card already picks up the new language/font.
        self._rebuild_history_cards()

        self._render(self._last)  # refreshes status pill / timer text immediately
        self._refresh_floating_metric_labels()
        self._render_drop_lookup()

        # ---- tab construction ------------------------------------------------

    def _start_ocr_loader(self) -> None:
        if not self._ocr_loading or getattr(self, "_ocr_thread_started", False):
            return
        self._ocr_thread_started = True
        threading.Thread(target=self._ocr_worker, name="maple-ocr-loader", daemon=True).start()
        self.root.after(50, self._poll_ocr_loader)

    def _ocr_worker(self) -> None:
        try:
            engine = StatPanelOcr()
        except Exception as exc:
            self._ocr_result = (None, str(exc))
        else:
            self._ocr_result = (engine, None)

    # ---- updates -----------------------------------------------------------

    def _start_update_check(self, *, manual: bool = False) -> None:
        """Start a non-blocking release check for the packaged app only."""
        if not getattr(sys, "frozen", False) or self._update_checking:
            if manual and not getattr(sys, "frozen", False):
                with self._modal():
                    messagebox.showinfo(
                        self._t("update_title"),
                        self._t("update_dev_build"),
                        parent=self.root,
                    )
            return
        self._update_checking = True
        self._update_manual_check = manual
        threading.Thread(
            target=self._update_check_worker,
            name="maple-update-check",
            daemon=True,
        ).start()

    def _on_check_updates_clicked(self) -> None:
        self._start_update_check(manual=True)

    def _update_check_worker(self) -> None:
        try:
            info = check_for_update()
        except UpdateError as exc:
            result: tuple[str, object] = ("check_error", str(exc))
        except Exception as exc:
            result = ("check_error", str(exc))
        else:
            result = ("available", info)
        with contextlib.suppress(queue.Full):
            self._update_queue.put_nowait(result)

    def _update_download_worker(self, info: UpdateInfo) -> None:
        try:
            path = download_update(info)
        except UpdateError as exc:
            result: tuple[str, object] = ("download_error", str(exc))
        except Exception as exc:
            result = ("download_error", str(exc))
        else:
            result = ("downloaded", (info, path))
        with contextlib.suppress(queue.Full):
            self._update_queue.put_nowait(result)

    def _offer_update(self, info: UpdateInfo) -> None:
        if self._update_prompted_version == info.version:
            return
        self._update_prompted_version = info.version
        notes = info.notes.strip()
        if len(notes) > 700:
            notes = notes[:700].rstrip() + "…"
        prompt = self._t(
            "update_available",
            version=info.version,
            current=APP_VERSION,
            notes=notes or self._t("update_no_notes"),
        )
        with self._modal():
            accepted = messagebox.askyesno(
                self._t("update_title"), prompt, parent=self.root
            )
        if not accepted:
            return
        self._update_checking = True
        threading.Thread(
            target=self._update_download_worker,
            args=(info,),
            name="maple-update-download",
            daemon=True,
        ).start()

    def _poll_update_results(self) -> None:
        try:
            while True:
                kind, payload = self._update_queue.get_nowait()
                if kind == "check_error":
                    was_manual = self._update_manual_check
                    self._update_checking = False
                    self._update_manual_check = False
                    if was_manual:
                        with self._modal():
                            messagebox.showerror(
                                self._t("update_title"),
                                self._t("update_failed", detail=str(payload)),
                                parent=self.root,
                            )
                elif kind == "available":
                    was_manual = self._update_manual_check
                    self._update_checking = False
                    self._update_manual_check = False
                    if isinstance(payload, UpdateInfo):
                        self._offer_update(payload)
                    elif was_manual:
                        with self._modal():
                            messagebox.showinfo(
                                self._t("update_title"),
                                self._t("update_current", version=APP_VERSION),
                                parent=self.root,
                            )
                elif kind == "download_error":
                    self._update_checking = False
                    with self._modal():
                        messagebox.showerror(
                            self._t("update_title"),
                            self._t("update_failed", detail=str(payload)),
                            parent=self.root,
                        )
                elif kind == "downloaded":
                    self._update_checking = False
                    info, path = payload
                    if not isinstance(info, UpdateInfo):
                        continue
                    with self._modal():
                        accepted = messagebox.askyesno(
                            self._t("update_ready_title"),
                            self._t("update_ready", version=info.version),
                            parent=self.root,
                        )
                    if not accepted:
                        with contextlib.suppress(OSError):
                            path.unlink()
                        continue
                    try:
                        schedule_update(path)
                    except UpdateError as exc:
                        with self._modal():
                            messagebox.showerror(
                                self._t("update_title"),
                                self._t("update_failed", detail=str(exc)),
                                parent=self.root,
                            )
                    else:
                        self._on_close()
        except queue.Empty:
            pass
        finally:
            with contextlib.suppress(Exception):
                self.root.after(250, self._poll_update_results)

    def _poll_ocr_loader(self) -> None:
        result = self._ocr_result
        if result is None:
            if self._ocr_loading:
                self.root.after(50, self._poll_ocr_loader)
            return
        self._ocr, self._ocr_error = result
        self._ocr_loading = False
        if self._ocr_error:
            self._log(f"[{time.strftime('%H:%M:%S')}] OCR load error: {self._ocr_error}")
        elif self._ocr is not None:
            self._monitor = BackgroundMonitor(
                self._source,
                self._ocr,
                sample_interval_ms=self._settings.sample_interval_ms,
                pickup_interval_ms=self._settings.pickup_interval_ms,
            )
            self._monitor.configure_auxiliary(
                track_pickup=self._settings.track_pickup_messages,
                track_potions=self._settings.track_potions,
                potion_slots=self._settings.potion_slots,
            )
            self._set_monitor_aux_enabled(self._run_state == "running")
            self._monitor.start()
        self._render(self._last)

    def _build_floating_bar(self) -> None:
        """Build the compact horizontal work HUD once, then toggle visibility."""
        self._floating_bar = ctk.CTkFrame(
            self.root, fg_color=SURFACE, corner_radius=18,
            border_width=1, border_color=BORDER,
        )
        self._floating_bar.pack_forget()
        self._floating_bar.grid_columnconfigure(1, weight=1)

        brand = ctk.CTkFrame(self._floating_bar, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="nsw", padx=(12, 6), pady=10)
        ctk.CTkLabel(
            brand, text="MSA", width=34, height=30, corner_radius=10,
            fg_color=ACCENT, text_color=ACCENT_INK, font=("Segoe UI", 12, "bold"),
        ).pack(side="left", pady=1)
        identity = ctk.CTkFrame(brand, fg_color="transparent")
        identity.pack(side="left", padx=(8, 0))
        ctk.CTkLabel(
            identity, text="LIVE", text_color=INK, font=("Segoe UI", 9, "bold"), anchor="w",
        ).pack(anchor="w")
        self._floating_context_label = ctk.CTkLabel(
            identity, text=self._t("context_detecting"), text_color=INK_DIM,
            font=self._font(8), anchor="w",
        )
        self._floating_context_label.pack(anchor="w")

        metric_strip = ctk.CTkFrame(self._floating_bar, fg_color="transparent")
        metric_strip.grid(row=0, column=1, sticky="ew", pady=7)
        metric_specs = FLOATING_METRIC_SPECS
        self._floating_metric_specs = metric_specs
        self._floating_metric_frames: dict[str, ctk.CTkFrame] = {}
        self._floating_metric_labels: dict[str, ctk.CTkLabel] = {}
        self._floating_metric_values: dict[str, ctk.CTkLabel] = {}
        for index, (key, label_key, color) in enumerate(metric_specs):
            metric = ctk.CTkFrame(
                metric_strip, fg_color=SURFACE_2, corner_radius=10,
                border_width=1, border_color=BORDER_SOFT,
            )
            metric.grid(row=0, column=index, sticky="ns", padx=(0 if index == 0 else 4, 0))
            label = ctk.CTkLabel(metric, text_color=INK_FAINT, anchor="w")
            self._i18n(label, label_key, size=8, bold=True)
            label.pack(anchor="w", padx=9, pady=(4, 0))
            self._floating_metric_labels[key] = label
            value = ctk.CTkLabel(
                metric, text="--", text_color=color, font=_FONT_MONO_BOLD, anchor="w",
            )
            value.pack(anchor="w", padx=9, pady=(0, 4))
            self._floating_metric_frames[key] = metric
            self._floating_metric_values[key] = value

        self._floating_restore_button = ctk.CTkButton(
            self._floating_bar, command=self._toggle_floating_mode,
            width=58, height=28, corner_radius=9, fg_color=ACCENT,
            hover_color="#7ff2e0", text_color=ACCENT_INK,
        )
        self._i18n(self._floating_restore_button, "hud_button_exit", size=9, bold=True)
        self._floating_restore_button.grid(row=0, column=2, sticky="e", padx=(6, 12), pady=10)
        self._apply_floating_visibility()
        self._refresh_floating_metric_labels()

    def _set_alpha(self, opacity_pct: int) -> None:
        with contextlib.suppress(Exception):
            self.root.attributes("-alpha", max(0.45, min(1.0, opacity_pct / 100)))

    def _enter_floating_mode(self) -> None:
        if not self._floating_mode:
            self._saved_topmost = self._settings.topmost
        self._floating_mode = True
        # The tab switcher is useful for configuration but noisy while
        # grinding.  Floating mode keeps only the Live tab visible; the HUD
        # button restores the full application chrome when needed.
        with contextlib.suppress(Exception):
            self._tabview.set(self._tab_names["live"])
            self._tabview._segmented_button.grid_remove()
        with contextlib.suppress(Exception):
            self._shell.pack_forget()
            self._floating_bar.pack(fill="x", padx=10, pady=10)
            self.root.geometry("1100x92+40+40")
            self.root.minsize(700, 70)
        with contextlib.suppress(Exception):
            self.root.attributes("-topmost", True)
        self._set_alpha(self._settings.floating_opacity_pct)
        self._hud_mode_button.configure(
            text=self._t("hud_button_exit"), fg_color=ACCENT, text_color=ACCENT_INK
        )

    def _leave_floating_mode(self) -> None:
        self._floating_mode = False
        with contextlib.suppress(Exception):
            self._floating_bar.pack_forget()
            self._shell.pack(fill="both", expand=True, padx=14, pady=14)
            self.root.geometry(self._full_geometry)
            self.root.minsize(420, 520)
        with contextlib.suppress(Exception):
            self._tabview._segmented_button.grid()
        self._set_alpha(100)
        with contextlib.suppress(Exception):
            self.root.attributes("-topmost", self._saved_topmost)
        self._hud_mode_button.configure(
            text=self._t("hud_button_enter"), fg_color=SURFACE_ELEVATED, text_color=INK
        )

    def _toggle_floating_mode(self) -> None:
        if self._floating_mode:
            self._leave_floating_mode()
        else:
            self._enter_floating_mode()

    # ---- tab construction ------------------------------------------------

    def _build_live_tab(self, parent) -> None:
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        # The Live tab is intentionally scrollable.  It lets the overview
        # remain spacious and premium at startup while the optional drop list
        # can expand without pushing the primary controls off-screen.
        scroll = ctk.CTkScrollableFrame(
            parent, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=BORDER, scrollbar_button_hover_color=INK_FAINT,
        )
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        # CustomTkinter's scrollable canvas reserves a scrollbar column but
        # older Windows/Tk DPI combinations can still report the inner canvas
        # a few dozen pixels wider than the viewport.  A small intentional
        # right inset keeps card values/actions inside the visible canvas at
        # 100% and 125% scaling instead of letting them be clipped.
        content = ctk.CTkFrame(scroll, fg_color="transparent")
        content.grid(row=0, column=0, sticky="ew", padx=(0, 112))
        content.grid_columnconfigure(0, weight=1)
        self._live_scroll = scroll
        self._live_content = content
        parent = content
        parent.grid_columnconfigure(0, weight=1)

        # Status + session timer share one row, both shrunk down (smaller
        # font/padding than the rest of the chrome) so a longer localized
        # status string (the capture-error states from _localize_error run
        # much longer than "Tracking"/"追蹤中") still leaves room for the
        # timer instead of pushing the Restart button out of the window.
        # wraplength caps the status pill's own width so it wraps to a
        # second line rather than growing sideways into the timer's column.
        strip = ctk.CTkFrame(parent, fg_color="transparent")
        strip.grid(row=0, column=0, sticky="ew", padx=(2, 64), pady=(2, 3))
        strip.grid_columnconfigure(0, weight=1)
        strip.grid_columnconfigure(1, weight=0)

        self._status_pill = ctk.CTkLabel(
            strip, text=self._t("status_loading"), corner_radius=999, fg_color=SURFACE_2,
            text_color=EXP_COLOR, font=self._font(9, bold=True), padx=8, pady=2,
            anchor="w", justify="left", wraplength=180,
        )
        self._status_pill.grid(row=0, column=0, sticky="w")

        # Mixes translated chrome ("left"/"剩餘") with the countdown digits,
        # so it needs the language-aware font (self._font), not the fixed
        # digits-only _FONT_MONO_BOLD -- unlike the pure-numeric value labels.
        self._timer_label = ctk.CTkLabel(
            strip, text="--:--", corner_radius=999, fg_color=SURFACE_2,
            text_color=INK, font=self._font(10, bold=True), padx=8, pady=2,
        )
        self._timer_label.grid(row=0, column=1, sticky="e")

        # Context card: map/job are low-frequency OCR signals, so they stay
        # visually separate from the 0.3s combat values.  This also gives the
        # floating HUD a compact identity for the current grinding spot.
        context_card = ctk.CTkFrame(
            parent, fg_color=SURFACE_2, corner_radius=16,
            border_width=1, border_color=BORDER,
        )
        context_card.grid(row=1, column=0, sticky="ew", padx=2, pady=(0, 3))
        context_card.grid_columnconfigure(0, weight=1)
        context_card.grid_columnconfigure(1, weight=0)
        context_header = ctk.CTkFrame(context_card, fg_color="transparent")
        context_header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=(12, 64), pady=(5, 0))
        self._i18n(
            ctk.CTkLabel(context_header, text_color=ACCENT, anchor="w"),
            "context_header", size=9, bold=True,
        ).pack(side="left")
        ctk.CTkLabel(
            context_header, text="⌖", text_color=VIOLET,
            font=("Segoe UI Symbol", 12, "bold"), width=16,
        ).pack(side="left", padx=(0, 5))
        self._context_refresh_button = ctk.CTkButton(
            context_header,
            width=72, height=22, corner_radius=8, fg_color=SURFACE_ELEVATED,
            hover_color=BORDER, text_color=INK_DIM, font=self._font(8, bold=True),
            command=self._refresh_context,
        )
        self._i18n(self._context_refresh_button, "context_refresh", size=8, bold=True)
        self._context_refresh_button.pack(side="right")
        self._context_value_labels: dict[str, ctk.CTkLabel] = {}
        for row, key, i18n_key in (
            (1, "job", "kv_job"),
            (2, "map", "kv_map"),
        ):
            label = ctk.CTkLabel(context_card, text_color=INK_DIM, anchor="w")
            self._i18n(label, i18n_key, size=9, bold=False)
            label.grid(row=row, column=0, sticky="w", padx=(14, 6), pady=(2, 2))
            value = ctk.CTkLabel(
                context_card, text=self._t("context_detecting"), font=self._font(9, bold=True),
                text_color=INK, anchor="e",
            )
            value.grid(row=row, column=1, sticky="e", padx=(6, 14), pady=(2, 2))
            self._context_value_labels[key] = value

        # Primary action row stays near the identity card, where the user can
        # start tracking before browsing any optional detail.
        button_row = ctk.CTkFrame(parent, fg_color="transparent")
        button_row.grid(row=2, column=0, sticky="ew", padx=2, pady=(3, 6))
        button_row.grid_columnconfigure(0, weight=1)
        button_row.grid_columnconfigure(1, weight=1)

        self._pause_button = ctk.CTkButton(
            button_row, command=self._on_pause_button_clicked,
            fg_color=VIOLET, hover_color=VIOLET_HOVER, text_color="#111326",
            corner_radius=11, height=34,
        )
        self._pause_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self._restart_button = ctk.CTkButton(
            button_row, command=self._on_restart_clicked,
            fg_color=ACCENT, text_color=ACCENT_INK, hover_color="#84f2e3",
            corner_radius=11, height=34,
        )
        self._i18n(self._restart_button, "restart_button", size=12, bold=True)
        self._restart_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        self._build_drop_lookup_card(parent, row=3)

        # Stat grid: label | mini bar | tabular value, aligned via one grid
        # rather than independently left-justified label:value text. Labels
        # (LV/HP/MP/EXP) are the game's own on-screen abbreviations -- see
        # i18n.py's docstring for why these are not translated.
        stats = ctk.CTkFrame(
            parent, fg_color=SURFACE, corner_radius=16,
            border_width=1, border_color=BORDER,
        )
        stats.grid(row=4, column=0, sticky="ew", padx=2, pady=(0, 6))
        stats.grid_columnconfigure(0, weight=0)
        stats.grid_columnconfigure(1, weight=1)
        stats.grid_columnconfigure(2, weight=0)

        self._stat_rows: dict[str, tuple] = {}
        self._value_labels: dict[str, ctk.CTkLabel] = {}
        self._bars: dict[str, ctk.CTkProgressBar] = {}

        self._i18n(
            ctk.CTkLabel(stats, text_color=ACCENT, anchor="w"),
            "live_snapshot_header", size=9, bold=True,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(9, 3))

        def add_stat_row(row: int, key: str, label_text: str, color: str, with_bar: bool) -> None:
            lbl = ctk.CTkLabel(stats, text=label_text, font=_FONT_LABEL, text_color=color, anchor="w")
            lbl.grid(row=row, column=0, sticky="w", padx=(14, 6), pady=(3, 3))
            value = ctk.CTkLabel(stats, text="--", font=_FONT_MONO, text_color=INK, anchor="e")
            value.grid(row=row, column=2, sticky="e", padx=(6, 14), pady=(3, 3))
            bar = None
            if with_bar:
                bar = ctk.CTkProgressBar(stats, height=5, progress_color=color, fg_color=SURFACE_2)
                bar.set(0)
                bar.grid(row=row, column=1, sticky="ew", padx=6, pady=(3, 3))
                self._bars[key] = bar
            self._stat_rows[key] = (lbl, bar, value)
            self._value_labels[key] = value

        add_stat_row(1, "level", "LV", EXP_COLOR, with_bar=False)
        add_stat_row(2, "hp", "HP", HP_COLOR, with_bar=True)
        add_stat_row(3, "mp", "MP", MP_COLOR, with_bar=True)
        add_stat_row(4, "exp", "EXP", EXP_COLOR, with_bar=True)

        # Session info: label | tabular value, same alignment discipline.
        session_card = ctk.CTkFrame(
            parent, fg_color=SURFACE, corner_radius=16,
            border_width=1, border_color=BORDER,
        )
        session_card.grid(row=5, column=0, sticky="ew", padx=2, pady=(0, 6))
        session_card.grid_columnconfigure(0, weight=1)
        session_card.grid_columnconfigure(1, weight=0)

        self._kv_rows: dict[str, tuple] = {}

        self._i18n(
            ctk.CTkLabel(session_card, text_color=ACCENT, anchor="w"),
            "session_forecast_header", size=9, bold=True,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(9, 3))

        def add_kv_row(row: int, key: str, i18n_key: str) -> None:
            lbl = ctk.CTkLabel(session_card, text_color=INK_DIM, anchor="w")
            self._i18n(lbl, i18n_key, size=11, bold=False)
            lbl.grid(row=row, column=0, sticky="w", padx=(14, 6), pady=(3, 3))
            value = ctk.CTkLabel(session_card, text="--", font=_FONT_MONO, text_color=INK, anchor="e")
            value.grid(row=row, column=1, sticky="e", padx=(6, 14), pady=(3, 3))
            self._kv_rows[key] = (lbl, value)
            self._value_labels[key] = value

        add_kv_row(1, "startexp", "kv_start_exp")
        add_kv_row(2, "expdiff", "kv_exp_diff")
        add_kv_row(3, "exprate", "kv_exp_rate")
        add_kv_row(4, "eta", "kv_eta")
        add_kv_row(5, "projexp", "kv_proj_exp")
        add_kv_row(6, "hploss", "kv_hp_loss")
        add_kv_row(7, "mploss", "kv_mp_loss")

        economy_card = ctk.CTkFrame(
            parent, fg_color=SURFACE_2, corner_radius=16,
            border_width=1, border_color=BORDER,
        )
        economy_card.grid(row=6, column=0, sticky="ew", padx=2, pady=(0, 6))
        economy_card.grid_columnconfigure(0, weight=1)
        economy_card.grid_columnconfigure(1, weight=0)
        self._economy_rows: dict[str, tuple] = {}

        self._i18n(
            ctk.CTkLabel(economy_card, text_color=ACCENT, anchor="w"),
            "economy_header", size=9, bold=True,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(9, 3))

        def add_economy_row(row: int, key: str, i18n_key: str) -> None:
            label = ctk.CTkLabel(economy_card, text_color=INK_DIM, anchor="w")
            self._i18n(label, i18n_key, size=10, bold=False)
            label.grid(row=row, column=0, sticky="w", padx=(14, 6), pady=(3, 3))
            value = ctk.CTkLabel(economy_card, text="--", font=_FONT_MONO_SM, text_color=INK, anchor="e")
            value.grid(row=row, column=1, sticky="e", padx=(6, 14), pady=(3, 3))
            self._economy_rows[key] = (label, value)
            self._value_labels[key] = value

        add_economy_row(1, "shortcut_inventory", "kv_shortcut_inventory")
        add_economy_row(2, "mesos", "kv_mesos")
        add_economy_row(3, "hp_potions", "kv_hp_potions")
        add_economy_row(4, "mp_potions", "kv_mp_potions")
        add_economy_row(5, "shared_potions", "kv_shared_potions")
        add_economy_row(6, "hp_recovery", "kv_hp_recovery")
        add_economy_row(7, "mp_recovery", "kv_mp_recovery")
        add_economy_row(8, "hp_recovery_savings", "kv_hp_recovery_savings")
        add_economy_row(9, "mp_recovery_savings", "kv_mp_recovery_savings")

    def _build_drop_lookup_card(self, parent, row: int) -> None:
        """Build the lazy map -> monster -> drop quick-view card."""
        self._drop_lookup_card = ctk.CTkFrame(
            parent, fg_color=SURFACE_RAISED, corner_radius=16,
            border_width=1, border_color=BORDER,
        )
        self._drop_lookup_card.grid(row=row, column=0, sticky="ew", padx=2, pady=(0, 6))
        self._drop_lookup_card.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self._drop_lookup_card, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(9, 3))
        header.grid_columnconfigure(0, weight=1)
        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="ew")
        self._i18n(
            ctk.CTkLabel(title_box, text_color=ACCENT, anchor="w"),
            "drop_lookup_header", size=9, bold=True,
        ).pack(side="left")
        self._drop_lookup_map_label = ctk.CTkLabel(
            title_box, text=self._t("context_unknown"), text_color=INK_FAINT,
            font=self._font(9), anchor="w",
        )
        self._drop_lookup_map_label.pack(side="left", padx=(9, 0))

        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.grid(row=0, column=1, sticky="e", padx=(6, 0))
        self._drop_lookup_source_button = ctk.CTkButton(
            actions, width=30, height=24, corner_radius=8,
            fg_color=SURFACE_ELEVATED, hover_color=BORDER, text="↗",
            text_color=INK_DIM, font=("Segoe UI Symbol", 11, "bold"),
            command=self._open_drop_map_source,
        )
        self._drop_lookup_source_button.pack(side="left", padx=(0, 4))
        self._drop_lookup_button = ctk.CTkButton(
            actions, width=82, height=24, corner_radius=8,
            fg_color=ACCENT, hover_color="#84f2e3", text_color=ACCENT_INK,
            font=self._font(8, bold=True), command=self._on_drop_lookup_clicked,
        )
        self._i18n(self._drop_lookup_button, "drop_lookup_button", size=8, bold=True)
        self._drop_lookup_button.pack(side="left", padx=(0, 4))
        self._drop_lookup_toggle_button = ctk.CTkButton(
            actions, width=30, height=24, corner_radius=8,
            fg_color=SURFACE_ELEVATED, hover_color=BORDER, text="⌄",
            text_color=INK, font=("Segoe UI Symbol", 11, "bold"),
            command=self._toggle_drop_lookup_card,
        )
        self._drop_lookup_toggle_button.pack(side="left")

        self._drop_lookup_status_label = ctk.CTkLabel(
            self._drop_lookup_card, text=self._t("drop_lookup_hint"),
            text_color=INK_DIM, anchor="w", justify="left",
            font=self._font(9), wraplength=450,
        )
        self._drop_lookup_status_label.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))

        self._drop_lookup_body = ctk.CTkFrame(self._drop_lookup_card, fg_color="transparent")
        self._drop_lookup_body.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self._drop_lookup_body.grid_remove()
        self._drop_lookup_rows_frame = ctk.CTkScrollableFrame(
            self._drop_lookup_body, height=250, fg_color=BG, corner_radius=12,
            scrollbar_button_color=BORDER, scrollbar_button_hover_color=INK_FAINT,
        )
        self._drop_lookup_rows_frame.pack(fill="both", expand=True)

    def _open_drop_source(self, url: str) -> None:
        with contextlib.suppress(Exception):
            webbrowser.open_new_tab(url)

    def _open_drop_map_source(self) -> None:
        summary = getattr(self, "_drop_lookup_summary", None)
        if summary is not None:
            self._open_drop_source(summary.source_url)
            return
        self._open_drop_source(MAPS_PAGE_URL)

    def _toggle_drop_lookup_card(self) -> None:
        self._drop_lookup_expanded = not self._drop_lookup_expanded
        self._apply_drop_lookup_expanded()
        if not self._drop_lookup_expanded:
            return

        # Opening the card is itself an obvious request to see the data.  The
        # separate button remains available for refresh, but the first open
        # should never leave an empty panel waiting for a second click.
        map_name = self._current_map_name()
        map_key = normalize_map_name(map_name) if map_name else None
        if map_key and (
            map_key != self._drop_lookup_requested_map
            or (
                self._drop_lookup_summary is None
                and self._drop_lookup_loading_map is None
            )
        ):
            self._on_drop_lookup_clicked()

    def _apply_drop_lookup_expanded(self) -> None:
        body = getattr(self, "_drop_lookup_body", None)
        toggle = getattr(self, "_drop_lookup_toggle_button", None)
        if body is None or toggle is None:
            return
        if self._drop_lookup_expanded:
            body.grid()
            toggle.configure(text="⌃")
        else:
            body.grid_remove()
            toggle.configure(text="⌄")

    def _on_drop_lookup_clicked(self) -> None:
        map_name = self._current_map_name()
        if not map_name:
            self._drop_lookup_expanded = True
            self._drop_lookup_error = None
            self._render_drop_lookup()
            return

        map_key = normalize_map_name(map_name)
        self._drop_lookup_requested_map = map_key
        cached = self._drop_lookup_cache.get(map_key)
        if cached is not None:
            self._drop_lookup_summary = cached
            self._drop_lookup_error = None
            self._drop_lookup_expanded = True
            self._render_drop_lookup()
            return
        if self._drop_lookup_loading_map == map_key:
            self._drop_lookup_expanded = True
            self._render_drop_lookup()
            return

        self._drop_lookup_summary = None
        self._drop_lookup_error = None
        self._drop_lookup_loading_map = map_key
        self._drop_lookup_expanded = True
        self._render_drop_lookup()
        threading.Thread(
            target=self._drop_lookup_worker,
            args=(map_name, map_key),
            name="maple-drop-lookup",
            daemon=True,
        ).start()

    def _drop_lookup_worker(self, map_name: str, map_key: str) -> None:
        try:
            summary = fetch_map_drop_summary(map_name)
            error = None if summary is not None else f"map not found: {map_name}"
        except DropLookupError as exc:
            summary = None
            error = str(exc)
        except Exception as exc:
            summary = None
            error = str(exc)
        with contextlib.suppress(queue.Full):
            self._drop_lookup_queue.put_nowait((map_key, summary, error))

    def _poll_drop_lookup_results(self) -> None:
        """Drain lookup results even when the live monitor is not running."""
        try:
            self._drain_drop_lookup_results()
        except Exception as exc:
            # Keep this independent pump alive if a malformed third-party row
            # ever reaches the renderer.  The normal tick logger remains the
            # authoritative path for OCR/runtime errors.
            self._log(f"[{time.strftime('%H:%M:%S')}] drop lookup render error: {exc!r}")
        finally:
            with contextlib.suppress(Exception):
                self.root.after(100, self._poll_drop_lookup_results)

    def _drain_drop_lookup_results(self) -> None:
        result_queue = getattr(self, "_drop_lookup_queue", None)
        if result_queue is None:
            return
        changed = False
        context_changed = False
        while True:
            try:
                map_key, summary, error = result_queue.get_nowait()
            except queue.Empty:
                break
            if summary is not None:
                self._drop_lookup_cache[map_key] = summary
            if map_key != self._drop_lookup_requested_map:
                continue
            if summary is not None:
                # A successful fuzzy lookup has a canonical database name.
                # Promote it back into the context display and session record
                # so the user sees 遺跡之墓Ⅳ instead of the partial OCR text
                # 遺之墓IV, and future history rows remain searchable.
                settings = getattr(self, "_settings", None)
                detected = getattr(self, "_detected_map_name", None)
                if (
                    settings is not None
                    and getattr(settings, "auto_context", False)
                    and detected
                    and normalize_map_name(detected) == map_key
                ):
                    canonical_name = summary.map_name
                    canonical_key = normalize_map_name(canonical_name)
                    if canonical_key:
                        self._detected_map_name = canonical_name
                        self._drop_lookup_cache[canonical_key] = summary
                        self._drop_lookup_requested_map = canonical_key
                        if (
                            getattr(self, "_session_map_name", None)
                            and normalize_map_name(self._session_map_name) == map_key
                        ):
                            self._session_map_name = canonical_name
                        context_changed = True
            self._drop_lookup_loading_map = None
            self._drop_lookup_summary = summary
            self._drop_lookup_error = error
            changed = True
        if changed:
            if context_changed:
                self._render_context()
            self._render_drop_lookup()

    def _render_drop_lookup_header(self) -> None:
        """Update only the cheap header when context OCR changes."""
        label = getattr(self, "_drop_lookup_map_label", None)
        if label is None:
            return
        map_name = self._current_map_name()
        label.configure(text=map_name or self._t("context_unknown"))
        map_key = normalize_map_name(map_name) if map_name else None
        if map_key == self._drop_lookup_requested_map:
            return
        self._drop_lookup_requested_map = map_key
        self._drop_lookup_summary = self._drop_lookup_cache.get(map_key) if map_key else None
        self._drop_lookup_loading_map = None
        self._drop_lookup_error = None
        self._drop_detail_expanded.clear()
        self._render_drop_lookup()
        # If the panel was opened before context OCR finished, the map becomes
        # available later.  Treat that first map transition as the same lookup
        # request as opening the panel, instead of leaving the user on the
        # placeholder hint forever.
        if self._drop_lookup_expanded and map_key:
            self._on_drop_lookup_clicked()

    def _render_drop_lookup(self) -> None:
        if not hasattr(self, "_drop_lookup_status_label"):
            return
        map_name = self._current_map_name()
        map_key = normalize_map_name(map_name) if map_name else None
        summary = self._drop_lookup_summary if map_key == self._drop_lookup_requested_map else None
        loading = map_key is not None and self._drop_lookup_loading_map == map_key

        self._drop_lookup_map_label.configure(text=map_name or self._t("context_unknown"))
        self._drop_lookup_source_button.configure(
            state="normal" if (summary is not None or map_name) else "disabled"
        )
        self._drop_lookup_button.configure(
            state="disabled" if loading else "normal",
            text=self._t("drop_lookup_loading_short" if loading else "drop_lookup_button"),
            font=self._font(8, bold=True),
        )
        self._apply_drop_lookup_expanded()

        if not map_name:
            self._drop_lookup_status_label.configure(text=self._t("drop_lookup_no_map"), text_color=INK_DIM)
            self._clear_drop_lookup_rows()
            return
        if loading:
            self._drop_lookup_status_label.configure(
                text=self._t("drop_lookup_loading", map=map_name), text_color=ACCENT
            )
            self._clear_drop_lookup_rows()
            return
        if self._drop_lookup_error:
            self._drop_lookup_status_label.configure(
                text=self._t("drop_lookup_error", detail=self._drop_lookup_error), text_color=HP_COLOR
            )
            self._clear_drop_lookup_rows()
            return
        if summary is None:
            # Keep the expanded panel self-healing if a context refresh or a
            # transient UI redraw cleared the request state.  The worker now
            # returns an explicit error for a real no-match, so this guard does
            # not create an endless retry loop.
            if self._drop_lookup_expanded:
                self._on_drop_lookup_clicked()
                return
            self._drop_lookup_status_label.configure(text=self._t("drop_lookup_hint"), text_color=INK_DIM)
            self._clear_drop_lookup_rows()
            return

        self._drop_lookup_status_label.configure(
            text=self._t(
                "drop_lookup_summary",
                map=summary.map_name,
                monsters=len(summary.monsters),
                generated=summary.generated_at or "—",
            ),
            text_color=INK_DIM,
        )
        self._render_drop_lookup_rows(summary)

    def _clear_drop_lookup_rows(self) -> None:
        rows = getattr(self, "_drop_lookup_rows_frame", None)
        if rows is None:
            return
        for child in rows.winfo_children():
            child.destroy()

    def _render_drop_lookup_rows(self, summary: MapDropSummary) -> None:
        rows = getattr(self, "_drop_lookup_rows_frame", None)
        if rows is None:
            return
        self._clear_drop_lookup_rows()
        rows.grid_columnconfigure(0, weight=1)
        if not summary.monsters:
            ctk.CTkLabel(
                rows, text=self._t("drop_lookup_no_monsters"), text_color=INK_FAINT,
                font=self._font(9), anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=10, pady=10)
            return
        for index, monster in enumerate(summary.monsters):
            key = f"{summary.map_id}:{monster.monster_id}"
            expanded = key in self._drop_detail_expanded
            card = ctk.CTkFrame(
                rows, fg_color=SURFACE, corner_radius=11,
                border_width=1, border_color=BORDER_SOFT,
            )
            card.grid(row=index, column=0, sticky="ew", pady=(0, 6))
            card.grid_columnconfigure(0, weight=1)
            header = ctk.CTkFrame(card, fg_color="transparent")
            header.grid(row=0, column=0, sticky="ew", padx=8, pady=7)
            header.grid_columnconfigure(0, weight=1)
            toggle = ctk.CTkButton(
                header,
                text=f"{'⌄' if expanded else '›'}  {monster.name}  Lv.{monster.level or '—'}",
                anchor="w", height=25, fg_color="transparent", hover_color=SURFACE_ELEVATED,
                text_color=INK, font=self._font(10, bold=True),
                command=lambda item_key=key: self._toggle_drop_monster(item_key),
            )
            toggle.grid(row=0, column=0, sticky="ew")
            ctk.CTkButton(
                header, text="↗", width=28, height=25, corner_radius=7,
                fg_color=SURFACE_ELEVATED, hover_color=BORDER, text_color=INK_DIM,
                font=("Segoe UI Symbol", 10, "bold"),
                command=lambda url=monster.source_url: self._open_drop_source(url),
            ).grid(row=0, column=1, padx=(5, 0))
            ctk.CTkLabel(
                card,
                text=self._t(
                    "drop_lookup_monster_meta",
                    spawns=monster.spawn_count,
                    drops=len(monster.drops),
                ),
                text_color=INK_FAINT, anchor="w", font=self._font(8),
            ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
            if expanded:
                detail = ctk.CTkFrame(card, fg_color="transparent")
                detail.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
                detail.grid_columnconfigure(0, weight=1)
                if not monster.drops:
                    ctk.CTkLabel(
                        detail, text=self._t("drop_lookup_no_drops"), text_color=INK_FAINT,
                        font=self._font(9), anchor="w",
                    ).grid(row=0, column=0, sticky="ew", padx=6, pady=5)
                for drop_index, drop in enumerate(monster.drops):
                    self._render_drop_item_row(detail, drop_index, drop)

    def _render_drop_item_row(self, parent, row: int, drop: DropItem) -> None:
        item_row = ctk.CTkFrame(parent, fg_color=SURFACE_2, corner_radius=8)
        item_row.grid(row=row, column=0, sticky="ew", pady=(0, 3))
        item_row.grid_columnconfigure(0, weight=1)
        info = ctk.CTkFrame(item_row, fg_color="transparent")
        info.grid(row=0, column=0, sticky="ew", padx=(9, 4), pady=5)
        ctk.CTkLabel(
            info, text=drop.name, text_color=INK, anchor="w",
            font=self._font(9, bold=True),
        ).pack(anchor="w", fill="x")
        meta = " · ".join(part for part in (drop.category, drop.subcategory, drop.source_label) if part)
        ctk.CTkLabel(
            info, text=meta or self._t("drop_lookup_source_unknown"), text_color=INK_FAINT,
            anchor="w", font=self._font(8),
        ).pack(anchor="w", fill="x")
        rate = "  ".join(part for part in (format_probability(drop.probability), format_quantity(drop.min_quantity, drop.max_quantity)) if part)
        ctk.CTkLabel(
            item_row, text=rate, text_color=EXP_COLOR if drop.probability is not None else INK_FAINT,
            anchor="e", font=("Consolas", 9, "bold"),
        ).grid(row=0, column=1, sticky="e", padx=(4, 9), pady=5)

    def _toggle_drop_monster(self, key: str) -> None:
        if key in self._drop_detail_expanded:
            self._drop_detail_expanded.remove(key)
        else:
            self._drop_detail_expanded.add(key)
        self._render_drop_lookup()

    def _build_history_tab(self, parent) -> None:
        toolbar = ctk.CTkFrame(parent, fg_color="transparent")
        toolbar.pack(fill="x", padx=2, pady=(2, 4))
        toolbar.grid_columnconfigure(0, weight=1)
        toolbar.grid_columnconfigure(1, weight=1)

        self._export_history_button = ctk.CTkButton(
            toolbar, command=self._on_export_history_clicked,
            fg_color=ACCENT, hover_color="#7ff2e0", text_color=ACCENT_INK,
            corner_radius=9, height=28,
        )
        self._i18n(self._export_history_button, "history_export_button", size=11, bold=True)
        self._export_history_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self._clear_history_button = ctk.CTkButton(
            toolbar, command=self._on_clear_history_clicked,
            fg_color=SURFACE_2, hover_color=TRACK_BG, text_color=HP_COLOR,
            corner_radius=9, height=28,
        )
        self._i18n(self._clear_history_button, "history_clear_button", size=11, bold=True)
        self._clear_history_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        self._history_overview_label = ctk.CTkLabel(
            parent, text_color=INK_DIM, anchor="w", justify="left", wraplength=260,
            font=self._font(10),
        )
        self._history_overview_label.pack(fill="x", padx=8, pady=(0, 5))

        self._history_frame = ctk.CTkScrollableFrame(parent, fg_color=BG, label_text="")
        self._history_frame.pack(fill="both", expand=True, padx=2, pady=(0, 2))
        self._history_empty_label = ctk.CTkLabel(
            self._history_frame, text_color=INK_FAINT,
        )
        self._i18n(self._history_empty_label, "history_empty", size=13, bold=False)
        self._history_empty_label.pack(pady=24)

    def _build_settings_tab(self, parent) -> None:
        # Scrollable: at some WINDOW SCALE values the settings content is
        # taller than the window, and a plain .pack() into the tab would
        # just clip the overflow with no way to reach it -- a scrollbar
        # keeps every option reachable regardless of scale/window size.
        scroll = ctk.CTkScrollableFrame(parent, fg_color=BG, label_text="")
        scroll.pack(fill="both", expand=True)

        window_card = ctk.CTkFrame(
            scroll, fg_color=SURFACE, corner_radius=14,
            border_width=1, border_color=BORDER,
        )
        window_card.pack(fill="x", padx=2, pady=(2, 3))

        # Value lives in the section header, not squeezed into the control
        # row -- at narrow window widths (esp. with the scrollbar eating
        # horizontal space) a fixed-width label at the end of a packed row
        # was getting clipped to invisible. The header always has room.
        self._scale_header_label = ctk.CTkLabel(
            window_card, text=self._scale_header_text(),
            anchor="w", text_color=INK_DIM, font=self._font(10, bold=True),
        )
        self._scale_header_label.pack(fill="x", padx=12, pady=(5, 0))
        scale_row = ctk.CTkFrame(window_card, fg_color="transparent")
        scale_row.pack(fill="x", padx=12, pady=(0, 3))
        # A +/- stepper instead of a slider -- a small draggable handle at
        # this widget size was fiddly to land on an exact value; discrete
        # SCALE_STEP_PCT taps are precise and don't need fine motor control.
        ctk.CTkButton(
            scale_row, text="-", width=36, command=lambda: self._on_scale_step(-SCALE_STEP_PCT),
            fg_color=SURFACE_2, hover_color=TRACK_BG, text_color=INK, font=_FONT_UI_BOLD,
        ).pack(side="left")
        ctk.CTkButton(
            scale_row, text="+", width=36, command=lambda: self._on_scale_step(SCALE_STEP_PCT),
            fg_color=SURFACE_2, hover_color=TRACK_BG, text_color=INK, font=_FONT_UI_BOLD,
        ).pack(side="left", padx=(6, 0))

        self._topmost_var = tk.BooleanVar(value=self._settings.topmost)
        self._i18n(ctk.CTkSwitch(
            window_card, variable=self._topmost_var, text_color=INK,
            progress_color=ACCENT, button_color=INK_DIM, button_hover_color=ACCENT,
            command=self._on_topmost_changed,
        ), "settings_always_on_top", size=11, bold=False).pack(fill="x", padx=12, pady=(0, 3))

        lang_row = ctk.CTkFrame(window_card, fg_color="transparent")
        lang_row.pack(fill="x", padx=12, pady=(0, 4))
        self._i18n(
            ctk.CTkLabel(lang_row, anchor="w", text_color=INK_DIM), "settings_language", size=10, bold=True
        ).pack(side="left")
        self._lang_button = ctk.CTkSegmentedButton(
            lang_row, values=["中文", "EN"], command=self._on_language_button_changed,
            selected_color=ACCENT, selected_hover_color="#7ff2e0", text_color=INK,
        )
        self._lang_button.set("中文" if self._settings.language == "zh" else "EN")
        self._lang_button.pack(side="right")

        self._update_button = ctk.CTkButton(
            window_card, command=self._on_check_updates_clicked,
            fg_color=SURFACE_2, hover_color=TRACK_BG, text_color=INK,
            corner_radius=9, height=28,
        )
        self._i18n(self._update_button, "settings_check_updates", size=10, bold=True)
        self._update_button.pack(fill="x", padx=12, pady=(0, 7))

        hud_card = ctk.CTkFrame(
            scroll, fg_color=SURFACE, corner_radius=14,
            border_width=1, border_color=BORDER,
        )
        hud_card.pack(fill="x", padx=2, pady=(0, 3))
        self._i18n(
            ctk.CTkLabel(hud_card, anchor="w", text_color=INK_DIM),
            "settings_hud", size=10, bold=True,
        ).pack(fill="x", padx=12, pady=(5, 0))
        self._floating_on_start_var = tk.BooleanVar(value=self._settings.floating_on_start)
        self._i18n(ctk.CTkSwitch(
            hud_card, variable=self._floating_on_start_var, text_color=INK,
            progress_color=ACCENT, button_color=INK_DIM, button_hover_color=ACCENT,
            command=self._on_floating_on_start_changed,
        ), "settings_floating_on_start", size=11, bold=False).pack(fill="x", padx=12, pady=(0, 2))
        self._floating_header_label = ctk.CTkLabel(
            hud_card, text=self._opacity_header_text(), anchor="w",
            text_color=INK_DIM, font=self._font(10, bold=True),
        )
        self._floating_header_label.pack(fill="x", padx=12, pady=(1, 0))
        self._floating_opacity_slider = ctk.CTkSlider(
            hud_card, from_=45, to=100, number_of_steps=55,
            command=self._on_opacity_changed,
            progress_color=ACCENT, button_color=ACCENT, button_hover_color="#7ff2e0",
        )
        self._floating_opacity_slider.set(self._settings.floating_opacity_pct)
        self._floating_opacity_slider.pack(fill="x", padx=12, pady=(0, 7))
        self._i18n(
            ctk.CTkLabel(hud_card, anchor="w", text_color=INK_DIM),
            "settings_display_fields", size=10, bold=True,
        ).pack(fill="x", padx=12, pady=(0, 0))
        floating_grid = ctk.CTkFrame(hud_card, fg_color="transparent")
        floating_grid.pack(fill="x", padx=10, pady=(0, 5))
        floating_grid.grid_columnconfigure((0, 1), weight=1, uniform="floating")
        self._floating_field_vars: dict[str, tk.BooleanVar] = {}
        for index, (key, label_key, _color) in enumerate(FLOATING_METRIC_SPECS):
            row, column = divmod(index, 2)
            var = tk.BooleanVar(value=key in self._settings.floating_fields)
            self._floating_field_vars[key] = var
            self._i18n(ctk.CTkCheckBox(
                floating_grid, variable=var, text_color=INK,
                fg_color=ACCENT, hover_color="#7ff2e0", border_color=INK_FAINT,
                command=lambda k=key, v=var: self._on_floating_field_changed(k, v),
            ), label_key, size=9, bold=False).grid(
                row=row, column=column, sticky="w", padx=2, pady=1,
            )

        context_settings_card = ctk.CTkFrame(
            scroll, fg_color=SURFACE, corner_radius=14,
            border_width=1, border_color=BORDER,
        )
        context_settings_card.pack(fill="x", padx=2, pady=(0, 3))
        self._i18n(
            ctk.CTkLabel(context_settings_card, anchor="w", text_color=INK_DIM),
            "settings_context", size=10, bold=True,
        ).pack(fill="x", padx=12, pady=(5, 0))
        self._auto_context_var = tk.BooleanVar(value=self._settings.auto_context)
        self._i18n(ctk.CTkSwitch(
            context_settings_card, variable=self._auto_context_var, text_color=INK,
            progress_color=ACCENT, button_color=INK_DIM, button_hover_color=ACCENT,
            command=self._on_auto_context_changed,
        ), "settings_auto_context", size=11, bold=False).pack(fill="x", padx=12, pady=(0, 3))
        context_grid = ctk.CTkFrame(context_settings_card, fg_color="transparent")
        context_grid.pack(fill="x", padx=12, pady=(0, 2))
        context_grid.grid_columnconfigure(1, weight=1)
        for row, label_key, attr in (
            (0, "settings_job_override", "job_name_override"),
            (1, "settings_map_override", "map_name_override"),
        ):
            self._i18n(
                ctk.CTkLabel(context_grid, anchor="w", text_color=INK_DIM),
                label_key, size=9, bold=False,
            ).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            entry = ctk.CTkEntry(context_grid, width=140)
            entry.insert(0, getattr(self._settings, attr))
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            setattr(self, f"_{attr}_entry", entry)
        self._i18n(ctk.CTkButton(
            context_settings_card, command=self._on_apply_context_settings,
            fg_color=ACCENT, hover_color="#7ff2e0", text_color=ACCENT_INK,
            corner_radius=8, height=25,
        ), "settings_apply_context", size=10, bold=True).pack(fill="x", padx=12, pady=(2, 7))

        card = ctk.CTkFrame(
            scroll, fg_color=SURFACE, corner_radius=14,
            border_width=1, border_color=BORDER,
        )
        card.pack(fill="x", padx=2, pady=(0, 0))

        self._interval_header_label = ctk.CTkLabel(
            card, text=self._interval_header_text(),
            anchor="w", text_color=INK_DIM, font=self._font(10, bold=True),
        )
        self._interval_header_label.pack(fill="x", padx=12, pady=(5, 0))
        slider_row = ctk.CTkFrame(card, fg_color="transparent")
        slider_row.pack(fill="x", padx=12, pady=(0, 2))
        slider = ctk.CTkSlider(
            slider_row, from_=1, to=60, number_of_steps=59, command=self._on_interval_changed,
            progress_color=ACCENT, button_color=ACCENT, button_hover_color="#7ff2e0",
        )
        slider.set(self._settings.window_min)
        slider.pack(fill="x", expand=True)

        self._i18n(
            ctk.CTkLabel(card, anchor="w", text_color=INK_DIM), "settings_display_fields", size=10, bold=True
        ).pack(fill="x", padx=12, pady=(2, 0))

        # A two-column checklist is denser and much easier to scan than a
        # long stack of switches.  Every live value has its own preference,
        # which is what makes the floating HUD genuinely user-configurable.
        display_grid = ctk.CTkFrame(card, fg_color="transparent")
        display_grid.pack(fill="x", padx=10, pady=(0, 5))
        display_grid.grid_columnconfigure((0, 1), weight=1, uniform="display")
        self._switch_vars: dict[str, tk.BooleanVar] = {}
        display_options = (
            ("level", "settings_show_level", "show_level"),
            ("hp", "settings_show_hp", "show_hp"),
            ("mp", "settings_show_mp", "show_mp"),
            ("exp", "settings_show_exp", "show_exp"),
            ("exp_pct", "settings_show_exp_pct", "show_exp_pct"),
            ("exp_diff", "settings_show_exp_diff", "show_exp_diff"),
            ("exp_rate", "settings_show_exp_rate", "show_exp_rate"),
            ("eta", "settings_show_eta", "show_eta"),
            ("proj_exp", "settings_show_proj_exp", "show_proj_exp"),
            ("hp_loss", "settings_show_hp_loss", "show_hp_loss"),
            ("mp_loss", "settings_show_mp_loss", "show_mp_loss"),
            ("mesos", "kv_mesos", "show_mesos"),
            ("hp_potions", "kv_hp_potions", "show_hp_potions"),
            ("mp_potions", "kv_mp_potions", "show_mp_potions"),
            ("shared_potions", "kv_shared_potions", "show_shared_potions"),
            ("hp_recovery", "kv_hp_recovery", "show_hp_recovery"),
            ("mp_recovery", "kv_mp_recovery", "show_mp_recovery"),
            ("hp_recovery_savings", "kv_hp_recovery_savings", "show_hp_recovery_savings"),
            ("mp_recovery_savings", "kv_mp_recovery_savings", "show_mp_recovery_savings"),
        )
        for index, (key, i18n_key, attr) in enumerate(display_options):
            row, column = divmod(index, 2)
            var = tk.BooleanVar(value=getattr(self._settings, attr))
            self._switch_vars[key] = var
            self._i18n(ctk.CTkCheckBox(
                display_grid, variable=var, text_color=INK,
                fg_color=ACCENT, hover_color="#7ff2e0", border_color=INK_FAINT,
                command=lambda k=key, a=attr, v=var: self._on_switch_changed(k, a, v),
            ), i18n_key, size=10, bold=False).grid(
                row=row, column=column, sticky="w", padx=2, pady=2,
            )

        # SESSION: behaviour switches, not display toggles -- neither one
        # hides/shows a widget, so they bypass _on_switch_changed/_apply_visibility
        # entirely (see _on_auto_stop_changed/_on_save_on_restart_changed).
        self._i18n(
            ctk.CTkLabel(card, anchor="w", text_color=INK_DIM), "settings_session", size=10, bold=True
        ).pack(fill="x", padx=12, pady=(3, 0))

        self._auto_stop_var = tk.BooleanVar(value=self._settings.auto_stop)
        self._i18n(ctk.CTkSwitch(
            card, variable=self._auto_stop_var, text_color=INK,
            progress_color=ACCENT, button_color=INK_DIM, button_hover_color=ACCENT,
            command=self._on_auto_stop_changed,
        ), "settings_auto_stop", size=11, bold=False).pack(fill="x", padx=12, pady=0)

        self._save_on_restart_var = tk.BooleanVar(value=self._settings.save_on_restart)
        self._i18n(ctk.CTkSwitch(
            card, variable=self._save_on_restart_var, text_color=INK,
            progress_color=ACCENT, button_color=INK_DIM, button_hover_color=ACCENT,
            command=self._on_save_on_restart_changed,
        ), "settings_save_on_restart", size=11, bold=False).pack(fill="x", padx=12, pady=(0, 4))

        sampling_card = ctk.CTkFrame(
            scroll, fg_color=SURFACE, corner_radius=14,
            border_width=1, border_color=BORDER,
        )
        sampling_card.pack(fill="x", padx=2, pady=(3, 3))
        self._sampling_header_label = ctk.CTkLabel(
            sampling_card, text=self._sampling_header_text(),
            anchor="w", text_color=INK_DIM, font=self._font(10, bold=True),
        )
        self._sampling_header_label.pack(fill="x", padx=12, pady=(5, 0))
        self._sampling_slider = ctk.CTkSlider(
            sampling_card, from_=0.2, to=1.0, number_of_steps=8,
            command=self._on_sampling_changed,
            progress_color=ACCENT, button_color=ACCENT, button_hover_color="#7ff2e0",
        )
        self._sampling_slider.set(self._settings.sample_interval_ms / 1000)
        self._sampling_slider.pack(fill="x", padx=12, pady=(0, 7))

        economy_card = ctk.CTkFrame(
            scroll, fg_color=SURFACE, corner_radius=14,
            border_width=1, border_color=BORDER,
        )
        economy_card.pack(fill="x", padx=2, pady=(0, 2))
        self._i18n(
            ctk.CTkLabel(economy_card, anchor="w", text_color=INK_DIM),
            "settings_economy", size=10, bold=True,
        ).pack(fill="x", padx=12, pady=(5, 0))

        self._pickup_sampling_header_label = ctk.CTkLabel(
            economy_card, text=self._pickup_sampling_header_text(),
            anchor="w", text_color=INK_DIM, font=self._font(10, bold=True),
        )
        self._pickup_sampling_header_label.pack(fill="x", padx=12, pady=(4, 0))
        self._pickup_sampling_slider = ctk.CTkSlider(
            economy_card, from_=0.1, to=1.0, number_of_steps=9,
            command=self._on_pickup_sampling_changed,
            progress_color=ACCENT, button_color=ACCENT, button_hover_color="#8bfff0",
        )
        self._pickup_sampling_slider.set(self._settings.pickup_interval_ms / 1000)
        self._pickup_sampling_slider.pack(fill="x", padx=12, pady=(0, 5))

        self._track_pickup_var = tk.BooleanVar(value=self._settings.track_pickup_messages)
        self._i18n(ctk.CTkSwitch(
            economy_card, variable=self._track_pickup_var, text_color=INK,
            progress_color=ACCENT, button_color=INK_DIM, button_hover_color=ACCENT,
            command=self._on_track_pickup_changed,
        ), "settings_track_pickup", size=11, bold=False).pack(fill="x", padx=12, pady=0)

        self._track_potions_var = tk.BooleanVar(value=self._settings.track_potions)
        self._i18n(ctk.CTkSwitch(
            economy_card, variable=self._track_potions_var, text_color=INK,
            progress_color=ACCENT, button_color=INK_DIM, button_hover_color=ACCENT,
            command=self._on_track_potions_changed,
        ), "settings_track_potions", size=11, bold=False).pack(fill="x", padx=12, pady=0)

        fallback_grid = ctk.CTkFrame(economy_card, fg_color="transparent")
        fallback_grid.pack(fill="x", padx=12, pady=(2, 2))
        fallback_grid.grid_columnconfigure(0, weight=1)
        for row, (label_key, attr, entry_attr) in enumerate((
            ("settings_default_recovery_hp", "potion_recovery_hp_default", "_default_recovery_hp_entry"),
            ("settings_default_recovery_mp", "potion_recovery_mp_default", "_default_recovery_mp_entry"),
        )):
            self._i18n(
                ctk.CTkLabel(fallback_grid, anchor="w", text_color=INK_DIM),
                label_key, size=10, bold=False,
            ).grid(row=row, column=0, sticky="w", pady=1)
            entry = ctk.CTkEntry(fallback_grid, width=72, justify="right")
            entry.insert(0, str(getattr(self._settings, attr)))
            entry.grid(row=row, column=1, sticky="e", pady=1)
            setattr(self, entry_attr, entry)

        self._i18n(
            ctk.CTkLabel(economy_card, anchor="w", text_color=INK_FAINT),
            "settings_potion_slots_hint", size=9, bold=False,
        ).pack(fill="x", padx=12, pady=(0, 4))

        table = ctk.CTkFrame(economy_card, fg_color="transparent")
        table.pack(fill="x", padx=12, pady=(0, 3))
        for column, key in enumerate(("settings_potion_slot", "settings_potion_name", "settings_potion_cost", "settings_potion_recovery", "settings_potion_kind")):
            table.grid_columnconfigure(column, weight=1 if column == 1 else 0)
            self._i18n(
                ctk.CTkLabel(table, anchor="w", text_color=INK_FAINT), key, size=8, bold=True
            ).grid(row=0, column=column, sticky="w", padx=(0, 3))

        configured_slots = {slot.slot: slot for slot in self._settings.potion_slots}
        self._potion_entries: dict[str, tuple] = {}
        kind_labels = [self._t("settings_hp_short"), self._t("settings_mp_short"), self._t("settings_both_short")]
        for row, slot_id in enumerate(SHORTCUT_SLOT_BOXES, start=1):
            config = configured_slots.get(slot_id, PotionSlotConfig(slot=slot_id, enabled=False))
            ctk.CTkLabel(table, text=slot_id, font=_FONT_MONO_SM, text_color=INK_DIM, width=24).grid(
                row=row, column=0, sticky="w", padx=(0, 3), pady=1
            )
            name_entry = ctk.CTkEntry(table, width=92)
            name_entry.insert(0, config.name)
            name_entry.grid(row=row, column=1, sticky="ew", padx=(0, 3), pady=1)
            cost_entry = ctk.CTkEntry(table, width=58, justify="right")
            if config.cost:
                cost_entry.insert(0, str(config.cost))
            cost_entry.grid(row=row, column=2, sticky="ew", padx=(0, 3), pady=1)
            recovery_entry = ctk.CTkEntry(table, width=62, justify="right")
            if config.recovery:
                recovery_entry.insert(0, str(config.recovery))
            recovery_entry.grid(row=row, column=3, sticky="ew", padx=(0, 3), pady=1)
            kind_menu = ctk.CTkOptionMenu(table, values=kind_labels, width=66)
            kind_menu.set({"hp": kind_labels[0], "mp": kind_labels[1], "both": kind_labels[2]}.get(config.kind, kind_labels[0]))
            kind_menu.grid(row=row, column=4, sticky="ew", pady=1)
            self._potion_entries[slot_id] = (name_entry, cost_entry, recovery_entry, kind_menu)

        self._i18n(ctk.CTkButton(
            economy_card, command=self._on_apply_potion_settings,
            fg_color=ACCENT, hover_color="#7ff2e0", text_color=ACCENT_INK,
            corner_radius=9, height=28,
        ), "settings_apply_potions", size=11, bold=True).pack(fill="x", padx=12, pady=(2, 7))

    # ---- settings callbacks ------------------------------------------------

    def _on_scale_step(self, delta: int) -> None:
        pct = max(SCALE_MIN_PCT, min(SCALE_MAX_PCT, self._settings.scale_pct + delta))
        if pct == self._settings.scale_pct:
            return
        self._settings.scale_pct = pct
        self._scale_header_label.configure(text=self._scale_header_text())
        # CTk's own scaling knobs: widget_scaling resizes fonts/padding/etc,
        # window_scaling resizes the geometry set via .geometry() -- both are
        # needed together, otherwise widgets end up mismatched against the
        # window size. Both apply live to the already-open root window.
        factor = pct / 100
        ctk.set_widget_scaling(factor)
        ctk.set_window_scaling(factor)
        _maybe_persist_settings(self)

    def _on_topmost_changed(self) -> None:
        self._settings.topmost = self._topmost_var.get()
        self._saved_topmost = self._settings.topmost
        self.root.attributes("-topmost", True if self._floating_mode else self._settings.topmost)
        _maybe_persist_settings(self)

    def _on_floating_on_start_changed(self) -> None:
        self._settings.floating_on_start = self._floating_on_start_var.get()
        _maybe_persist_settings(self)

    def _on_opacity_changed(self, value: float) -> None:
        self._settings.floating_opacity_pct = max(45, min(100, round(float(value))))
        self._floating_header_label.configure(text=self._opacity_header_text())
        if self._floating_mode:
            self._set_alpha(self._settings.floating_opacity_pct)
        _maybe_persist_settings(self)

    def _refresh_context(self) -> None:
        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            monitor.request_context()
        self._context_error = None
        self._render_context()

    def _current_job_name(self) -> str | None:
        # A manual fallback is authoritative when supplied.  This is useful
        # for clients whose class label is stylized or unavailable to OCR;
        # without this order a misread player name could overwrite the user's
        # explicit class choice on the next context refresh.
        if self._settings.job_name_override:
            return self._settings.job_name_override
        if self._settings.auto_context and self._detected_job_name:
            return self._detected_job_name
        return None

    def _current_map_name(self) -> str | None:
        if self._settings.auto_context and self._detected_map_name:
            return self._detected_map_name
        return self._settings.map_name_override or None

    def _render_context(self) -> None:
        values = getattr(self, "_context_value_labels", {})
        if not values:
            return
        job_label = values.get("job")
        map_label = values.get("map")
        if job_label is not None:
            job_label.configure(text=self._current_job_name() or self._t("context_unknown"))
        if map_label is not None:
            map_label.configure(text=self._current_map_name() or self._t("context_unknown"))
        floating_context = getattr(self, "_floating_context_label", None)
        if floating_context is not None:
            job = self._current_job_name() or self._t("context_unknown")
            map_name = self._current_map_name() or self._t("context_unknown")
            floating_context.configure(text=f"{job}  ·  {map_name}")
        self._render_drop_lookup_header()

    def _configure_monitor(self) -> None:
        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            monitor.configure_auxiliary(
                track_pickup=self._settings.track_pickup_messages,
                track_potions=self._settings.track_potions,
                potion_slots=self._settings.potion_slots,
            )

    def _set_monitor_aux_enabled(self, enabled: bool) -> None:
        monitor = getattr(self, "_monitor", None)
        if enabled and getattr(self._settings, "track_potions", False):
            economy = getattr(self, "_economy", None)
            if economy is not None:
                economy.begin_quick_slot_baseline()
            self._potion_baseline_pending = True
            self._potion_baseline_samples.clear()
            self._last_logged_shortcut_counts = None
        if monitor is not None:
            monitor.set_aux_enabled(enabled)
            if enabled:
                request_scan = getattr(monitor, "request_auxiliary_scan", None)
                if callable(request_scan):
                    request_scan()

    def _record_auxiliary_counts(self, counts: dict[str, int], now: float) -> None:
        economy = getattr(self, "_economy", None)
        if economy is None:
            return
        if self._potion_baseline_pending:
            if not counts:
                return
            # The first read is calibration, not accounting.  Require each
            # slot to repeat the same value in two consecutive auxiliary
            # frames before making it the session baseline.  This prevents a
            # single adjacent-cell OCR merge (e.g. 89 -> 895) from defining
            # the starting inventory for the entire session.
            self._potion_baseline_samples.append(dict(counts))
            if len(self._potion_baseline_samples) > 3:
                del self._potion_baseline_samples[:-3]
            stable: dict[str, int] = {}
            for slot_id, count in counts.items():
                confirmations = sum(
                    sample.get(slot_id) == count
                    for sample in self._potion_baseline_samples
                )
                if confirmations >= 2:
                    stable[slot_id] = count
            if not stable:
                return
            economy.prime_quick_slot_counts(stable)
            self._potion_baseline_pending = False
            self._last_logged_shortcut_counts = dict(stable)
            visible = ", ".join(f"{slot}={count}" for slot, count in sorted(stable.items()))
            self._log(f"[{time.strftime('%H:%M:%S')}] potion baseline: {visible}")
            return
        if counts != self._last_logged_shortcut_counts:
            visible = ", ".join(f"{slot}={count}" for slot, count in sorted(counts.items()))
            self._log(f"[{time.strftime('%H:%M:%S')}] potion counts: {visible or 'unreadable'}")
            self._last_logged_shortcut_counts = dict(counts)
        uses = economy.record_quick_slot_counts(counts, now)
        if uses:
            self._log(f"[{time.strftime('%H:%M:%S')}] potion use confirmed: {uses}")

    def _start_new_session(self) -> None:
        begin_fresh = getattr(self._session, "begin_fresh", None)
        if callable(begin_fresh):
            begin_fresh()
        else:
            # Compatibility for lightweight test/custom Session objects.
            self._session.start()
        self._session_job_name = self._current_job_name()
        self._session_map_name = self._current_map_name()

    def _on_auto_context_changed(self) -> None:
        self._settings.auto_context = self._auto_context_var.get()
        if self._settings.auto_context:
            self._refresh_context()
        _maybe_persist_settings(self)

    def _on_apply_context_settings(self) -> None:
        for attr in ("job_name_override", "map_name_override"):
            entry = getattr(self, f"_{attr}_entry", None)
            if entry is not None:
                setattr(self._settings, attr, entry.get().strip()[:32])
        self._render_context()
        _maybe_persist_settings(self)

    def _on_language_button_changed(self, value: str) -> None:
        self._apply_language("zh" if value == "中文" else "en")
        _maybe_persist_settings(self)

    def _on_interval_changed(self, value: float) -> None:
        self._settings.window_min = round(value)
        self._interval_header_label.configure(text=self._interval_header_text())
        self._refresh_floating_metric_labels()
        # Doesn't retroactively affect the currently-running session's
        # already-baked-in target -- takes effect for the *next* session,
        # same as the interval_minutes recorded on SessionSummary.finalize().
        _maybe_persist_settings(self)

    def _on_sampling_changed(self, value: float) -> None:
        self._settings.sample_interval_ms = max(200, min(1000, round(float(value) * 10) * 100))
        self._sampling_header_label.configure(text=self._sampling_header_text())
        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            monitor.set_sample_interval(self._settings.sample_interval_ms)
        _maybe_persist_settings(self)

    def _on_pickup_sampling_changed(self, value: float) -> None:
        self._settings.pickup_interval_ms = max(
            100, min(1000, round(float(value) * 10) * 100)
        )
        label = getattr(self, "_pickup_sampling_header_label", None)
        if label is not None:
            label.configure(text=self._pickup_sampling_header_text())
        monitor = getattr(self, "_monitor", None)
        setter = getattr(monitor, "set_pickup_interval", None)
        if callable(setter):
            setter(self._settings.pickup_interval_ms)
        _maybe_persist_settings(self)

    def _on_track_pickup_changed(self) -> None:
        self._settings.track_pickup_messages = self._track_pickup_var.get()
        self._configure_monitor()
        self._apply_visibility()
        self._render(self._last)
        _maybe_persist_settings(self)

    def _on_track_potions_changed(self) -> None:
        self._settings.track_potions = self._track_potions_var.get()
        self._configure_monitor()
        self._apply_visibility()
        self._render(self._last)
        _maybe_persist_settings(self)

    @staticmethod
    def _entry_nonnegative(entry, default: int = 0) -> int:
        try:
            return max(0, int(entry.get().replace(",", "").strip() or default))
        except (AttributeError, TypeError, ValueError):
            return default

    def _on_apply_potion_settings(self) -> None:
        default_recovery_hp = self._entry_nonnegative(self._default_recovery_hp_entry)
        default_recovery_mp = self._entry_nonnegative(self._default_recovery_mp_entry)
        kind_by_label = {
            self._t("settings_hp_short"): "hp",
            self._t("settings_mp_short"): "mp",
            self._t("settings_both_short"): "both",
        }
        slots: list[PotionSlotConfig] = []
        for slot_id, (name_entry, cost_entry, recovery_entry, kind_menu) in self._potion_entries.items():
            name = name_entry.get().strip()
            cost = self._entry_nonnegative(cost_entry)
            recovery = self._entry_nonnegative(recovery_entry)
            slots.append(PotionSlotConfig(
                slot=slot_id,
                name=name,
                kind=kind_by_label.get(kind_menu.get(), "hp"),
                cost=cost,
                recovery=recovery,
                enabled=bool(name or cost or recovery),
            ))
        self._settings.potion_recovery_hp_default = default_recovery_hp
        self._settings.potion_recovery_mp_default = default_recovery_mp
        self._settings.potion_slots = slots
        self._economy.configure(slots, default_recovery_hp, default_recovery_mp)
        self._configure_monitor()
        _maybe_persist_settings(self)
        self._render(self._last)

    def _on_switch_changed(self, key: str, attr: str, var: tk.BooleanVar) -> None:
        setattr(self._settings, attr, var.get())
        if key != "exp_pct":  # visibility-affecting; exp_pct only changes rendered text
            self._apply_visibility()
        self._render(self._last)  # immediate feedback
        _maybe_persist_settings(self)

    def _on_auto_stop_changed(self) -> None:
        self._settings.auto_stop = self._auto_stop_var.get()
        _maybe_persist_settings(self)

    def _on_save_on_restart_changed(self) -> None:
        self._settings.save_on_restart = self._save_on_restart_var.get()
        _maybe_persist_settings(self)

    def _on_floating_field_changed(self, key: str, variable: tk.BooleanVar) -> None:
        selected = [
            field_key for field_key, _label_key, _color in FLOATING_METRIC_SPECS
            if self._floating_field_vars.get(field_key, tk.BooleanVar(value=False)).get()
        ]
        # Preserve the canonical metric order, which keeps the horizontal bar
        # visually stable while a user toggles optional values.
        self._settings.floating_fields = selected
        self._apply_floating_visibility()
        _maybe_persist_settings(self)

    def _apply_floating_visibility(self) -> None:
        selected = set(getattr(self._settings, "floating_fields", ()))
        for key, frame in getattr(self, "_floating_metric_frames", {}).items():
            if key in selected:
                frame.grid()
            else:
                frame.grid_remove()

    def _refresh_floating_metric_labels(self) -> None:
        labels = getattr(self, "_floating_metric_labels", {})
        minutes = self._settings.window_min
        for key, translation_key in {
            "proj_exp": "hud_proj_exp_interval",
            "proj_mesos": "hud_mesos_interval",
            "proj_potion_cost": "hud_potion_cost_interval",
        }.items():
            label = labels.get(key)
            if label is not None:
                label.configure(
                    text=self._t(translation_key, minutes=minutes),
                    font=self._font(8, bold=True),
                )

    def _apply_visibility(self) -> None:
        s = self._settings
        visible_stats = {"level": s.show_level, "hp": s.show_hp, "mp": s.show_mp, "exp": s.show_exp}
        for key, (lbl, bar, value) in self._stat_rows.items():
            widgets = [lbl, value] + ([bar] if bar else [])
            for w in widgets:
                w.grid() if visible_stats[key] else w.grid_remove()

        visible_kv = {
            "startexp": s.show_exp_diff,
            "expdiff": s.show_exp_diff,
            "exprate": s.show_exp_rate,
            "eta": s.show_eta,
            "projexp": s.show_proj_exp,
            "hploss": s.show_hp_loss,
            "mploss": s.show_mp_loss,
        }
        for key, (lbl, value) in self._kv_rows.items():
            for w in (lbl, value):
                w.grid() if visible_kv[key] else w.grid_remove()

        visible_economy = {
            "shortcut_inventory": s.track_potions,
            "mesos": s.track_pickup_messages and s.show_mesos,
            "hp_potions": s.track_potions and s.show_hp_potions,
            "mp_potions": s.track_potions and s.show_mp_potions,
            "shared_potions": s.track_potions and s.show_shared_potions,
            "hp_recovery": s.track_potions and s.show_hp_recovery,
            "mp_recovery": s.track_potions and s.show_mp_recovery,
            "hp_recovery_savings": s.track_potions and s.show_hp_recovery_savings,
            "mp_recovery_savings": s.track_potions and s.show_mp_recovery_savings,
        }
        for key, (lbl, value) in getattr(self, "_economy_rows", {}).items():
            for w in (lbl, value):
                w.grid() if visible_economy[key] else w.grid_remove()

    # ---- tick loop ---------------------------------------------------------

    def _tick(self) -> None:
        # Every path through this method must reschedule -- this loop is the
        # only thing driving the HUD, so an exception escaping before
        # self.root.after(...) freezes it permanently on stale data. The
        # window itself stays responsive, which makes the failure especially
        # confusing: buttons still click, Restart Session still "works", and
        # nothing ever updates again.
        #
        # Hence both the except *and* the finally. The original try/except
        # wasn't enough on its own: in the release .exe an unencodable OCR
        # character raised UnicodeEncodeError out of _do_tick's debug print,
        # and the handler's own `print(... {e!r})` re-raised on the same
        # unencodable text, so the reschedule below was never reached (see
        # the stdout-sink note at the top of this module for that trigger's
        # actual fix, and _log for why logging can no longer raise at all).
        # `finally` is what makes the loop survive the *next* such bug.
        next_delay = getattr(getattr(self, "_settings", None), "sample_interval_ms", TARGET_MS)
        try:
            next_delay = self._do_tick()
        except Exception as e:
            self._log(f"[{time.strftime('%H:%M:%S')}] tick error: {e!r}")
            with contextlib.suppress(Exception):
                self._set_status_error(self._t("status_error_unknown", detail=str(e)))
        finally:
            self.root.after(next_delay, self._tick)

    @staticmethod
    def _log(message: str) -> None:
        """Debug logging must never be able to kill the tick loop -- it is the
        least important thing this app does and has already taken the whole
        HUD down once (see _tick)."""
        with contextlib.suppress(Exception):
            print(message, flush=True)

    def _drain_background(self, *, include_auxiliary_when_paused: bool = False) -> None:
        """Apply worker results on Tk's thread without doing OCR here.

        ``include_auxiliary_when_paused`` is used only by the final boundary
        flush.  A last shortcut quantity is still useful for final accounting
        after the user has paused, while normal paused ticks must remain
        side-effect free.
        """
        monitor = getattr(self, "_monitor", None)
        if monitor is None:
            return

        status_readings = []
        while True:
            try:
                status_readings.append(monitor.status_queue.get_nowait())
            except queue.Empty:
                break

        auxiliary_readings = []
        while True:
            try:
                auxiliary_readings.append(monitor.auxiliary_queue.get_nowait())
            except queue.Empty:
                break

        # Status and shortcut workers run independently.  Apply their results
        # in capture order so a potion quantity drop is registered before the
        # HP/MP increase caused by that same drink is classified.
        events = [
            (reading.timestamp if reading.timestamp is not None else time.monotonic(), 0, reading)
            for reading in status_readings
        ] + [
            (reading.timestamp if reading.timestamp is not None else time.monotonic(), 1, reading)
            for reading in auxiliary_readings
        ]
        events.sort(key=lambda event: (event[0], event[1]))
        economy = getattr(self, "_economy", None)
        for event_now, event_kind, reading in events:
            if event_kind == 0:
                if reading.error:
                    if reading.error != self._last_capture_error:
                        self._log(f"[{time.strftime('%H:%M:%S')}] capture unavailable: {reading.error}")
                        self._last_capture_error = reading.error
                    self._set_status_error(self._localize_error(reading.error.removeprefix("OCR: ")))
                    continue
                if self._last_capture_error is not None:
                    self._log(f"[{time.strftime('%H:%M:%S')}] capture recovered")
                    self._last_capture_error = None
                if reading.client_size is not None and reading.client_size != self._last_client_size:
                    self._log(
                        f"[{time.strftime('%H:%M:%S')}] client size: "
                        f"{reading.client_size[0]}x{reading.client_size[1]}"
                    )
                    self._last_client_size = reading.client_size
                snap = reading.snapshot
                merged = StatSnapshot(*(
                    new if new is not None else old
                    for new, old in zip(vars(snap).values(), vars(self._last).values())
                ))
                self._last = merged
                if self._run_state == "running":
                    self._session.record(
                        merged.exp_cur, merged.hp_cur, merged.mp_cur, merged.exp_pct,
                        hp_max=merged.hp_max, mp_max=merged.mp_max, level=merged.level,
                    )
                    if economy is not None:
                        hp_recovery, mp_recovery = economy.record_stats(
                            merged.hp_cur, merged.mp_cur, event_now
                        )
                        self._session.add_recovery_evidence("hp", hp_recovery)
                        self._session.add_recovery_evidence("mp", mp_recovery)
                continue

            if reading.error:
                if reading.error != self._last_aux_error:
                    self._log(f"[{time.strftime('%H:%M:%S')}] auxiliary OCR error: {reading.error}")
                    self._last_aux_error = reading.error
                continue
            self._last_aux_error = None
            if economy is not None and (
                self._run_state == "running" or include_auxiliary_when_paused
            ):
                if getattr(reading, "pickup_scanned", True):
                    economy.record_pickup_lines(reading.lines, event_now)
                self._record_auxiliary_counts(dict(reading.counts), event_now)

        while True:
            try:
                reading = monitor.context_queue.get_nowait()
            except queue.Empty:
                break
            self._context_error = reading.error
            if reading.map_name:
                self._detected_map_name = reading.map_name
            if reading.job_name:
                self._detected_job_name = reading.job_name
            if self._run_state == "running":
                if self._session_map_name is None and self._current_map_name():
                    self._session_map_name = self._current_map_name()
                if self._session_job_name is None and self._current_job_name():
                    self._session_job_name = self._current_job_name()
        self._render_context()

    def _do_tick(self) -> int:
        # Production uses BackgroundMonitor.  The synchronous fallback keeps
        # OverlayApp's small no-display unit-test stubs useful and is never
        # selected by the real window after OCR loading completes.
        getattr(self, "_drain_drop_lookup_results", lambda: None)()
        if getattr(self, "_monitor", None) is not None:
            self._drain_background()
            self._maybe_finalize_on_timeout()
            self._render(self._last)
            return 50
        sync = getattr(self, "_do_tick_sync", None)
        return sync() if callable(sync) else OverlayApp._do_tick_sync(self)

    def _do_tick_sync(self) -> int:
        t0 = time.perf_counter()
        if self._ocr is None:
            # Keep the Tk event loop alive while ONNX Runtime loads.  No
            # capture or OCR work is attempted until the worker hands back a
            # ready engine, so the shell remains clickable during startup.
            self._render(self._last)
            return 250 if self._ocr_loading else 1000
        try:
            field_images = self._source.grab_fields()
        except RuntimeError as e:
            # Game window gone (closed/crashed), minimized, or the stat panel
            # is covered by another window -- don't crash the HUD, show it
            # plainly and keep retrying at a slower pace in case it clears.
            #
            # Logged on *transition* only: this path produces no other output,
            # so a persistently obscured panel used to leave a completely
            # empty log with nothing to diagnose from -- but logging every
            # 2s retry would bury the real ticks.
            if str(e) != self._last_capture_error:
                self._log(f"[{time.strftime('%H:%M:%S')}] capture unavailable: {e}")
                self._last_capture_error = str(e)
            self._set_status_error(self._localize_error(str(e)))
            # The session clock is wall-clock time (Session.elapsed()), not
            # tick-driven, so it keeps running even while OCR can't read the
            # panel (game window covered, alt-tabbed away, minimized). Both
            # of these used to be skipped entirely on this path: the timer
            # chip froze at its last-rendered text even though the real
            # countdown kept going underneath, and a session whose window
            # stayed blocked past its interval would never auto-finalize at
            # all, silently overrunning forever.
            self._update_timer_label()
            self._maybe_finalize_on_timeout()
            return 2000
        if self._last_capture_error is not None:
            self._log(f"[{time.strftime('%H:%M:%S')}] capture recovered")
            self._last_capture_error = None

        # Every crop is scaled from the client size (regions.py), so a log
        # without it can't explain a bad read -- and a mid-session resize is
        # exactly the kind of thing that moves the panel out from under the
        # boxes. Logged once at startup and again on any change.
        client_size = getattr(self._source, "client_size", None)
        if client_size is not None and client_size != self._last_client_size:
            self._log(f"[{time.strftime('%H:%M:%S')}] client size: {client_size[0]}x{client_size[1]}")
            self._last_client_size = client_size
        read_fields = getattr(self._ocr, "read_fields", None)
        if callable(read_fields):
            field_text = read_fields(field_images)
        else:
            field_text = {name: self._ocr.read_field(img) for name, img in field_images.items()}
        snap = parse_fields(field_text)
        self._log(f"[{time.strftime('%H:%M:%S')}] fields={field_text}")
        self._log(f"          -> {snap}")
        # A single tick occasionally misses a field (combat effects/floating
        # damage numbers over the HP/MP bars, transient OCR confidence dips) --
        # observed live: HP briefly read as None while MP/EXP/LV parsed fine on
        # the same frame. Carry forward the last known value per field instead
        # of flickering to '--' on every miss; a field that's genuinely gone
        # (e.g. OCR permanently broken) will just show stale data, which is a
        # more honest failure mode than a blank field for a live number.
        merged = StatSnapshot(*(
            new if new is not None else old
            for new, old in zip(vars(snap).values(), vars(self._last).values())
        ))
        self._last = merged
        # hp_max/mp_max are passed purely so Session can sanity-check them --
        # a tick whose max doesn't match the rest of the session was misparsed
        # (see rate.py's _LossTracker) and is dropped before it can inflate the
        # loss totals.
        #
        # There used to be a "does this frame even look like the stat panel?"
        # gate here (reject the tick unless LV parsed). It was removed after
        # ablating it against both live captures: it changed the totals by
        # exactly zero, because rate.py already rejects those same frames one
        # layer down -- and it carried a real risk of its own, since a broken
        # LV crop would have stopped a session recording anything at all.
        # tests/test_captured_regression.py replays the real failure through
        # this path with no gate in front of it.
        # Gated on run_state rather than relying on Session's own pause/no-op
        # behaviour: while "stopped" the Session may never have been started
        # at all (see _run_state's docstring in __init__), and feeding it
        # ticks here would silently begin calibrating/tracking a session the
        # user hasn't asked for yet.
        if self._run_state == "running":
            self._session.record(
                merged.exp_cur, merged.hp_cur, merged.mp_cur, merged.exp_pct,
                hp_max=merged.hp_max, mp_max=merged.mp_max, level=merged.level,
            )
            economy = getattr(self, "_economy", None)
            if economy is not None:
                self._scan_auxiliary()
                hp_recovery, mp_recovery = economy.record_stats(merged.hp_cur, merged.mp_cur)
                self._session.add_recovery_evidence("hp", hp_recovery)
                self._session.add_recovery_evidence("mp", mp_recovery)

        self._maybe_finalize_on_timeout()

        self._render(merged)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return max(0, int(self._settings.sample_interval_ms - elapsed_ms))

    def _scan_auxiliary(self) -> None:
        """Read pickup/shortcut regions without blocking the status loop."""
        source = getattr(self, "_source", None)
        economy = getattr(self, "_economy", None)
        grab_auxiliary = getattr(source, "grab_auxiliary", None)
        if (
            economy is None
            or not callable(grab_auxiliary)
            or not (self._settings.track_pickup_messages or self._settings.track_potions)
        ):
            return
        now = time.monotonic()
        if now < self._next_aux_scan:
            return
        # The status bar remains a 0.3s loop, but economy OCR used to run 12
        # pickup rows plus 8 shortcut slots on every status tick.  That could
        # consume the whole Tk frame budget and make the window look hung.
        fallback_interval = min(AUX_SCAN_MS, self._settings.pickup_interval_ms)
        self._next_aux_scan = now + fallback_interval / 1000
        try:
            regions = grab_auxiliary()
        except RuntimeError as exc:
            if str(exc) != self._last_aux_error:
                self._log(f"[{time.strftime('%H:%M:%S')}] auxiliary capture unavailable: {exc}")
                self._last_aux_error = str(exc)
            return
        except Exception as exc:
            if repr(exc) != self._last_aux_error:
                self._log(f"[{time.strftime('%H:%M:%S')}] auxiliary capture error: {exc!r}")
                self._last_aux_error = repr(exc)
            return
        if self._last_aux_error is not None:
            self._log(f"[{time.strftime('%H:%M:%S')}] auxiliary capture recovered")
            self._last_aux_error = None

        if self._settings.track_pickup_messages:
            try:
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
                if line_images:
                    lines = [
                        (
                            self._ocr.read_field(image),
                            int(line_id) * PICKUP_LINE_HEIGHT
                            + PICKUP_LINE_TOP_OFFSET
                            + PICKUP_LINE_HEIGHT / 2,
                        )
                        for line_id, image in line_images
                    ]
                    # Some clients render the feed a few pixels away from the
                    # reference rhythm.  If the cheap fixed rows did not
                    # expose a 楓幣 token, retry the full feed with detection
                    # at most once per second.  This keeps the normal 0.3s
                    # loop fast while making mesos collection resilient to a
                    # small DPI/font/layout offset.
                    if (
                        not any(parse_mesos_amount(text) is not None for text, _ in lines)
                        and now >= self._next_pickup_detection
                    ):
                        self._next_pickup_detection = now + PICKUP_DETECTION_MS / 1000
                        detected = []
                        for key in ("pickup", "pickup_wide"):
                            image = regions.get(key)
                            if image is None:
                                continue
                            detected.extend(self._ocr.read_lines(image))
                            if any(
                                parse_mesos_amount(getattr(line, "text", str(line))) is not None
                                for line in detected
                            ):
                                break
                        lines = detected
                else:
                    # Compatibility path for third-party/custom capture
                    # sources that only provide the original full feed crop.
                    lines = self._ocr.read_lines(regions.get("pickup")) if regions.get("pickup") is not None else []
                economy.record_pickup_lines(lines, now)
            except Exception as exc:
                self._log(f"[{time.strftime('%H:%M:%S')}] pickup OCR error: {exc!r}")

        if self._settings.track_potions:
            counts: dict[str, int] = {}
            configured = [slot for slot in self._settings.potion_slots if slot.enabled]
            configured_ids = {slot.slot for slot in configured}
            slots = configured + [
                PotionSlotConfig(slot=slot, kind="both", enabled=True)
                for slot in SHORTCUT_SLOT_BOXES
                if slot not in configured_ids
            ]
            read_shortcut_counts = getattr(self._ocr, "read_shortcut_counts", None)
            if callable(read_shortcut_counts) and regions.get("shortcut") is not None:
                try:
                    configured_ids = {slot.slot for slot in configured if slot.enabled}
                    try:
                        detected_counts = read_shortcut_counts(regions["shortcut"], configured_ids)
                    except TypeError:
                        detected_counts = read_shortcut_counts(regions["shortcut"])
                    enabled_slots = {slot.slot for slot in slots if slot.enabled}
                    counts = {
                        slot_id: count
                        for slot_id, count in detected_counts.items()
                        if slot_id in enabled_slots
                    }
                except Exception as exc:
                    self._log(f"[{time.strftime('%H:%M:%S')}] shortcut OCR error: {exc!r}")
            else:
                for slot in slots:
                    image = regions.get(f"shortcut:{slot.slot}")
                    if image is None:
                        continue
                    try:
                        read_slot_count = getattr(self._ocr, "read_slot_count", None)
                        count = (
                            read_slot_count(image)
                            if callable(read_slot_count)
                            else parse_slot_count(self._ocr.read_field(image))
                        )
                    except Exception as exc:
                        self._log(f"[{time.strftime('%H:%M:%S')}] shortcut OCR error: {exc!r}")
                        continue
                    if count is not None:
                        counts[slot.slot] = count
            self._record_auxiliary_counts(counts, now)

    @staticmethod
    def _image_has_content(image) -> bool:
        """Cheap brightness gate for dark pickup/shortcut crops.

        Blank feed rows are almost entirely black.  Skipping their OCR call
        saves most of the auxiliary work while preserving any row containing
        the bright yellow/white notification glyphs.
        """
        try:
            histogram = image.convert("L").histogram()
            bright_pixels = sum(histogram[110:])
            return bright_pixels >= max(12, int(image.width * image.height * 0.003))
        except Exception:
            # A custom capture source may not be a PIL image; let OCR handle
            # it rather than treating an unknown object as an empty crop.
            return True

    def _update_timer_label(self) -> None:
        """Split out of _render so the capture-error path in _do_tick can
        keep the countdown moving without running a full render against
        stale/absent OCR data."""
        if self._run_state == "stopped":
            # A stopped session (including the very first one, before Start
            # is ever clicked) has no countdown running -- showing a static
            # "10:00" the whole time would look like a stuck timer rather
            # than a genuinely inactive one.
            self._timer_label.configure(text="--:--")
            return
        remaining = max(0.0, self._settings.window_min * 60 - self._session.elapsed())
        # A countdown should not lose a whole second merely because the OCR
        # callback took a few milliseconds past the boundary.  Ceiling gives
        # the user the conventional display (5.01s elapsed still reads 9:55),
        # while the next tick naturally advances it to 9:54.
        remaining_whole = max(0, math.ceil(remaining))
        remaining_s = f"{remaining_whole // 60}:{remaining_whole % 60:02d}"
        self._timer_label.configure(text=self._t("timer_left", time=remaining_s))

    def _flush_economy_before_boundary(self) -> None:
        """Take one last fast shortcut scan before pause/restart/close.

        The economy worker is asynchronous by design, so the newest OCR result
        can still be in flight when the user presses Pause or closes the HUD.
        Give that requested scan a short, bounded window, drain it on Tk's
        thread, then reconcile the final visible quantities against the
        session baseline.  This removes the common "last few potions missing"
        result without making normal 300ms ticks wait for OCR.
        """
        economy = getattr(self, "_economy", None)
        monitor = getattr(self, "_monitor", None)
        if economy is None or monitor is None or not self._settings.track_potions:
            return
        was_running = self._run_state == "running"
        if not was_running:
            monitor.set_aux_enabled(True)
        self._drain_background(include_auxiliary_when_paused=True)
        request_scan = getattr(monitor, "request_auxiliary_scan", None)
        if callable(request_scan):
            request_scan()
        deadline = time.monotonic() + 1.25
        while time.monotonic() < deadline:
            self._drain_background(include_auxiliary_when_paused=True)
            time.sleep(0.04)
        self._drain_background(include_auxiliary_when_paused=True)
        if not was_running:
            monitor.set_aux_enabled(False)
        snapshot = economy.snapshot
        uses = economy.reconcile_quick_slot_counts(
            snapshot.shortcut_current,
            time.monotonic(),
        )
        if uses:
            self._log(f"[{time.strftime('%H:%M:%S')}] final potion reconciliation: {uses}")

    def _maybe_finalize_on_timeout(self) -> None:
        # Skipped while a rename dialog is open: simpledialog.askstring blocks
        # via a nested Tk event loop but doesn't stop self.root.after() timers
        # from firing, so without this guard a session could finalize and
        # insert a new history card underneath the open modal mid-edit. Also
        # skipped outright unless actually running: elapsed() is frozen while
        # paused/stopped anyway, so this wouldn't fire either way, but being
        # explicit here means it can't ever race a state change mid-tick.
        #
        # Called from both branches of _do_tick (capture success and capture
        # failure) -- Session.elapsed() is wall-clock time, not tick-driven,
        # so a session must still be able to hit its interval and finalize
        # even while the game window is covered/minimized for the whole
        # window, not just while OCR happens to be succeeding.
        if not self._modal_open and self._run_state == "running" \
                and self._session.elapsed() >= self._settings.window_min * 60:
            self._finalize_and_maybe_stop()

    def _commit_session_to_history(self) -> None:
        # Shared by the timer rollover and a manual restart with
        # save_on_restart on -- exactly one code path commits, so two
        # triggers landing on the same tick can't double-log.
        # Skip logging if the session never got a real EXP reading (restart
        # clicked immediately after launch, before OCR produced anything --
        # a '? -> ?' entry would just be noise), or if essentially no time
        # passed (rapid double-click on the restart button after real data
        # already exists -- start() carries the last known values forward,
        # so a second click 50ms later would otherwise log a valid-looking
        # but meaningless 0-duration, 0-diff entry).
        if self._session.start_exp is not None and self._session.elapsed() >= 1.0:
            getattr(self, "_flush_economy_before_boundary", lambda: None)()
            summary = self._session.finalize(self._settings.window_min)
            summary = dataclasses.replace(
                summary,
                job_name=getattr(self, "_session_job_name", None) or getattr(self, "_current_job_name", lambda: None)(),
                map_name=getattr(self, "_session_map_name", None) or getattr(self, "_current_map_name", lambda: None)(),
            )
            economy = getattr(self, "_economy", None)
            if economy is not None:
                snapshot = economy.snapshot
                summary = dataclasses.replace(
                    summary,
                    mesos=snapshot.mesos,
                    potion_uses=snapshot.potion_uses,
                    potion_cost=snapshot.potion_cost,
                    hp_potion_uses=snapshot.hp_potion_uses,
                    hp_potion_cost=snapshot.hp_potion_cost,
                    mp_potion_uses=snapshot.mp_potion_uses,
                    mp_potion_cost=snapshot.mp_potion_cost,
                    shared_potion_uses=snapshot.shared_potion_uses,
                    shared_potion_cost=snapshot.shared_potion_cost,
                    hp_recovery_natural=snapshot.hp_recovery_natural,
                    hp_recovery_potion=snapshot.hp_recovery_potion,
                    mp_recovery_natural=snapshot.mp_recovery_natural,
                    mp_recovery_potion=snapshot.mp_recovery_potion,
                    hp_recovery_savings=snapshot.hp_recovery_savings,
                    mp_recovery_savings=snapshot.mp_recovery_savings,
                    potion_breakdown=snapshot.potion_breakdown,
                )
            self._session_history.append(summary)
            self._log(f"[{time.strftime('%H:%M:%S')}] {_fmt_summary(summary, len(self._session_history))}")
            self._append_history_card(summary, len(self._session_history))
            getattr(self, "_update_history_overview", lambda: None)()
            _maybe_persist_history(self)

    def _finalize_and_maybe_stop(self) -> None:
        """Commit one interval, then continue or stop per the user's setting."""
        self._commit_session_to_history()
        if self._settings.auto_stop:
            # Reuses Session.pause() rather than adding a third Session
            # state: it freezes elapsed() at exactly this instant and makes
            # record() a no-op, which is exactly what "stopped" needs, and
            # nothing else in rate.py has to know "stopped" exists.
            self._session.pause()
            self._run_state = "stopped"
            getattr(self, "_set_monitor_aux_enabled", lambda _enabled: None)(False)
            self._apply_run_state()
        else:
            getattr(self, "_start_new_session", self._session.start)()
            getattr(self, "_set_monitor_aux_enabled", lambda _enabled: None)(True)
            economy = getattr(self, "_economy", None)
            if economy is not None:
                economy.reset()
                economy.configure(
                    self._settings.potion_slots,
                    self._settings.potion_recovery_hp_default,
                    self._settings.potion_recovery_mp_default,
                )
            if self._settings.floating_on_start:
                getattr(self, "_enter_floating_mode", lambda: None)()

    def _on_restart_clicked(self) -> None:
        if self._settings.save_on_restart:
            self._commit_session_to_history()
        getattr(self, "_start_new_session", self._session.start)()  # resets pause state too, so a restart from "paused" lands in "running"
        self._run_state = "running"
        getattr(self, "_set_monitor_aux_enabled", lambda _enabled: None)(True)
        economy = getattr(self, "_economy", None)
        if economy is not None:
            economy.reset()
            economy.configure(
                self._settings.potion_slots,
                self._settings.potion_recovery_hp_default,
                self._settings.potion_recovery_mp_default,
            )
        if self._settings.floating_on_start:
            getattr(self, "_enter_floating_mode", lambda: None)()
        self._apply_run_state()
        self._render(self._last)  # immediate feedback, don't wait for next tick

    def _on_pause_button_clicked(self) -> None:
        """One button, three roles depending on _run_state -- see
        _apply_run_state for how its label/command follow that state."""
        if self._run_state == "running":
            getattr(self, "_flush_economy_before_boundary", lambda: None)()
            self._session.pause()
            self._run_state = "paused"
            getattr(self, "_set_monitor_aux_enabled", lambda _enabled: None)(False)
        elif self._run_state == "paused":
            self._session.resume()
            self._run_state = "running"
            getattr(self, "_set_monitor_aux_enabled", lambda _enabled: None)(True)
        else:  # "stopped" -- already committed to History by _finalize_and_maybe_stop
            getattr(self, "_start_new_session", self._session.start)()
            self._run_state = "running"
            getattr(self, "_set_monitor_aux_enabled", lambda _enabled: None)(True)
            economy = getattr(self, "_economy", None)
            if economy is not None:
                economy.reset()
                economy.configure(
                    self._settings.potion_slots,
                    self._settings.potion_recovery_hp_default,
                    self._settings.potion_recovery_mp_default,
                )
            if self._settings.floating_on_start:
                getattr(self, "_enter_floating_mode", lambda: None)()
        self._apply_run_state()
        self._render(self._last)  # immediate feedback, don't wait for next tick

    def _apply_run_state(self) -> None:
        label_key = {"running": "pause_button", "paused": "resume_button", "stopped": "start_button"}[self._run_state]
        self._pause_button.configure(text=self._t(label_key), font=self._font(12, bold=True))
        # A Restart with nothing running/paused to restart from doesn't mean
        # anything -- Start (the pause button's role while stopped) already
        # covers beginning the next session. As the sole button in the row
        # it's centered and shrunk rather than stretched across both
        # columns the way the two-button running/paused layout is.
        if self._run_state == "stopped":
            self._restart_button.grid_remove()
            self._pause_button.configure(width=STOPPED_BUTTON_WIDTH, height=BUTTON_HEIGHT)
            self._pause_button.grid(row=0, column=0, columnspan=2, sticky="", padx=0)
        else:
            self._pause_button.configure(width=140, height=BUTTON_HEIGHT)  # CTkButton's own default width
            self._pause_button.grid(row=0, column=0, columnspan=1, sticky="ew", padx=(0, 3))
            self._restart_button.grid(row=0, column=1, columnspan=1, sticky="ew", padx=(3, 0))

    def _rebuild_history_cards(self) -> None:
        self._update_history_overview()
        for card in self._history_cards:
            card.destroy()
        self._history_cards.clear()
        if not self._session_history:
            # _append_history_card only ever pack_forget()s this label (on
            # the first card added) -- nothing re-packs it once the list is
            # emptied again (e.g. via Clear History), so do it explicitly.
            self._history_empty_label.pack(pady=24)
            return
        # Cards are always inserted at the top (newest-first) -- rebuilding
        # oldest-first via _append_history_card reproduces the exact same
        # final order without needing separate "rebuild" layout logic.
        for index, summary in enumerate(self._session_history, start=1):
            self._append_history_card(summary, index)

    def _update_history_overview(self) -> None:
        """Render a compact, weighted efficiency summary above History."""
        label = getattr(self, "_history_overview_label", None)
        if label is None:
            return
        if not self._session_history:
            label.configure(text=self._t("history_overview_empty"), font=self._font(10))
            if hasattr(self, "_export_history_button"):
                self._export_history_button.configure(state="disabled")
            return

        total_exp = sum(max(0, summary.exp_diff or 0) for summary in self._session_history)
        total_seconds = sum(max(0.0, summary.duration_s) for summary in self._session_history)
        average = total_exp * 3600 / total_seconds if total_seconds > 0 else None
        rates = [rate for summary in self._session_history if (rate := summary.exp_per_hour) is not None]
        best = max(rates) if rates else None
        rate_text = f"{average:,.0f}" if average is not None else "--"
        best_text = f"{best:,.0f}" if best is not None else "--"
        label.configure(
            text=self._t(
                "history_overview", count=len(self._session_history), average=rate_text, best=best_text
            ),
            font=self._font(10),
        )
        if hasattr(self, "_export_history_button"):
            self._export_history_button.configure(state="normal")

    def _append_history_card(self, summary: SessionSummary, index: int) -> None:
        self._history_empty_label.pack_forget()

        card = ctk.CTkFrame(
            self._history_frame, fg_color=SURFACE, corner_radius=14,
            border_width=1, border_color=BORDER,
        )
        # Newest-first: pack before the current top card (if any) rather than
        # appending, so the most recently finalized session is always the
        # first thing visible in the scrollable frame.
        if self._history_cards:
            card.pack(fill="x", pady=(0, 8), before=self._history_cards[0])
        else:
            card.pack(fill="x", pady=(0, 8))
        self._history_cards.insert(0, card)

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.pack(fill="x", padx=12, pady=(10, 0))
        title_label = ctk.CTkLabel(
            head, text=summary.name or self._t("history_session", n=index), font=self._font(10, bold=True),
            text_color=INK_FAINT, cursor="hand2",
        )
        title_label.pack(side="left")
        title_label.bind("<Button-1>", lambda _e, i=index, lbl=title_label: self._on_rename_clicked(i, lbl))

        # Packed before dur_text below so it lands rightmost -- pack(side="right")
        # stacks from the outer edge inward in packing order, so whichever
        # side="right" widget is packed first ends up furthest right.
        ctk.CTkButton(
            head, text="×", width=22, height=18, command=lambda i=index: self._on_delete_history_clicked(i),
            fg_color="transparent", hover_color=SURFACE_2, text_color=INK_FAINT, font=_FONT_UI_BOLD,
        ).pack(side="right")

        dur_min = summary.duration_s / 60
        # Mixes translated chrome ("restarted early"/提前重啟) with the
        # duration number when applicable, so this needs the language-aware
        # font -- the plain "10.0m" case doesn't strictly need it, but the
        # widget is rebuilt wholesale on language switch anyway either way.
        unit = self._t("unit_min_short")
        if summary.interval_minutes is not None and abs(dur_min - summary.interval_minutes) > 0.05:
            dur_text = self._t(
                "history_duration_early",
                dur=f"{dur_min:.1f}",
                target=summary.interval_minutes,
                unit=unit,
                label=self._t("history_restarted_early"),
            )
            dur_color = EXP_COLOR
            dur_font = self._font(11)
        else:
            dur_text, dur_color, dur_font = f"{dur_min:.1f}{unit}", INK_DIM, _FONT_MONO_SM
        ctk.CTkLabel(head, text=dur_text, font=dur_font, text_color=dur_color).pack(side="right")

        timestamp = ctk.CTkFrame(card, fg_color="transparent")
        timestamp.pack(fill="x", padx=12, pady=(0, 4))
        start_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(summary.start_time))
        end_ts = time.strftime("%H:%M:%S", time.localtime(summary.end_time))
        ctk.CTkLabel(
            timestamp, text=f"{start_ts} → {end_ts}", font=_FONT_MONO_SM, text_color=INK_FAINT,
        ).pack(side="left")

        if summary.job_name or summary.map_name:
            ctk.CTkLabel(
                card,
                text=self._t(
                    "history_context_line",
                    job=summary.job_name or self._t("context_unknown"),
                    map=summary.map_name or self._t("context_unknown"),
                ),
                font=self._font(9, bold=True), text_color=ACCENT, anchor="w",
            ).pack(fill="x", padx=12, pady=(0, 5))

        rng = ctk.CTkFrame(card, fg_color="transparent")
        rng.pack(fill="x", padx=12, pady=(0, 8))
        start_s = f"{summary.start_exp:,}" if summary.start_exp is not None else "?"
        end_s = f"{summary.end_exp:,}" if summary.end_exp is not None else "?"
        diff = summary.exp_diff
        diff_s = f"+{diff:,}" if diff is not None else "?"
        pct_diff = summary.exp_pct_diff
        ctk.CTkLabel(rng, text=start_s, font=_FONT_MONO, text_color=INK).pack(side="left")
        ctk.CTkLabel(rng, text=" → ", font=_FONT_MONO, text_color=INK_FAINT).pack(side="left")
        ctk.CTkLabel(rng, text=end_s, font=_FONT_MONO, text_color=INK).pack(side="left")
        ctk.CTkLabel(rng, text=f"  {diff_s}", font=_FONT_MONO, text_color=EXP_COLOR).pack(side="left")
        if pct_diff is not None:
            ctk.CTkLabel(rng, text=f" (+{pct_diff:.2f}%)", font=_FONT_MONO_SM, text_color=INK_DIM).pack(side="left")

        economy_line = ctk.CTkLabel(
            card,
            text=self._t(
                "history_economy_line",
                mesos=summary.mesos,
                hp_uses=summary.hp_potion_uses,
                hp_cost=summary.hp_potion_cost,
                mp_uses=summary.mp_potion_uses,
                mp_cost=summary.mp_potion_cost,
                shared_uses=summary.shared_potion_uses,
                shared_cost=summary.shared_potion_cost,
            ),
            font=self._font(9),
            text_color=INK_DIM,
            anchor="w",
        )
        economy_line.pack(fill="x", padx=12, pady=(0, 7))
        ctk.CTkLabel(
            card,
            text=self._t(
                "history_recovery_savings",
                hp=f"{summary.hp_recovery_savings:,.1f}",
                mp=f"{summary.mp_recovery_savings:,.1f}",
            ),
            font=self._font(9),
            text_color=INK_DIM,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 7))

        mini = ctk.CTkFrame(card, fg_color="transparent")
        mini.pack(fill="x", padx=12, pady=(0, 10))
        mini.grid_columnconfigure((0, 1), weight=1, uniform="mini")

        def mini_stat(col: int, label: str, value: str, color: str) -> None:
            box = ctk.CTkFrame(
                mini, fg_color=SURFACE_2, corner_radius=9,
                border_width=1, border_color=BORDER_SOFT,
            )
            box.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 4, 0))
            ctk.CTkLabel(box, text=label, font=self._font(9, bold=True), text_color=INK_FAINT, anchor="w").pack(
                fill="x", padx=8, pady=(6, 0)
            )
            ctk.CTkLabel(box, text=value, font=_FONT_MONO_SM, text_color=color, anchor="w").pack(
                fill="x", padx=8, pady=(0, 6)
            )

        mini_stat(0, self._t("history_hp_loss"), _fmt_loss(summary.hp_loss), HP_COLOR if summary.hp_loss > 0 else INK_FAINT)
        mini_stat(1, self._t("history_mp_loss"), _fmt_loss(summary.mp_loss), MP_COLOR if summary.mp_loss > 0 else INK_FAINT)

    @contextlib.contextmanager
    def _modal(self):
        """Run a blocking dialog. Two things have to happen around one:

        1. `_modal_open` tells _do_tick not to finalize a session while a
           dialog is up -- askstring/askyesno block on a *nested* Tk event
           loop, which does not stop self.root.after() timers from firing,
           so a session could otherwise roll over and insert a history card
           underneath the open modal mid-edit.
        2. -topmost has to come off for the duration. Tk dialogs are not
           topmost themselves, so with the HUD pinned above everything the
           dialog renders *behind* it -- while still holding a grab on all
           input. The app looks frozen (clicks on the HUD, including Restart
           Session, do nothing) with no visible cause, and stays that way
           until the invisible dialog is found and dismissed.
        """
        self._modal_open = True
        was_topmost = self._settings.topmost
        if was_topmost:
            self.root.attributes("-topmost", False)
        try:
            yield
        finally:
            self._modal_open = False
            if was_topmost:
                self.root.attributes("-topmost", True)

    def _on_rename_clicked(self, index: int, label: ctk.CTkLabel) -> None:
        # index is 1-based. session_history is no longer strictly append-only
        # (see _on_delete_history_clicked), but deleting any entry rebuilds
        # every card from scratch via _rebuild_history_cards(), so a *live*
        # card's index - 1 is always still correct: it can only go stale by
        # having its own card destroyed and recreated with the new one first.
        current = self._session_history[index - 1]
        with self._modal():
            new_name = simpledialog.askstring(
                self._t("rename_dialog_title"), self._t("rename_dialog_prompt"),
                initialvalue=current.name or self._t("history_session", n=index),
                parent=self.root,
            )
        if new_name is None:
            return  # cancelled
        new_name = new_name.strip()
        updated = dataclasses.replace(current, name=new_name or None)
        self._session_history[index - 1] = updated
        label.configure(text=updated.name or self._t("history_session", n=index))
        _maybe_persist_history(self)

    def _on_export_history_clicked(self) -> None:
        if not self._session_history:
            with self._modal():
                messagebox.showinfo(
                    self._t("history_export_title"), self._t("history_export_no_data"), parent=self.root
                )
            return

        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        with self._modal():
            destination = filedialog.asksaveasfilename(
                parent=self.root,
                title=self._t("history_export_title"),
                defaultextension=".csv",
                filetypes=filetypes,
                initialfile="maplestory-history.csv",
            )
        if not destination:
            return
        try:
            export_history_csv(self._session_history, destination, language=self._settings.language)
        except (OSError, UnicodeError) as exc:
            with self._modal():
                messagebox.showerror(
                    self._t("history_export_title"),
                    self._t("history_export_failed", detail=str(exc)),
                    parent=self.root,
                )
            return
        with self._modal():
            messagebox.showinfo(
                self._t("history_export_title"),
                self._t("history_export_success", n=len(self._session_history)),
                parent=self.root,
            )

    def _on_delete_history_clicked(self, index: int) -> None:
        summary = self._session_history[index - 1]
        with self._modal():  # see _do_tick's guard comment on _modal()
            confirmed = messagebox.askyesno(
                self._t("history_delete_confirm_title"),
                self._t(
                    "history_delete_confirm_prompt",
                    name=summary.name or self._t("history_session", n=index),
                ),
                parent=self.root,
            )
        if not confirmed:
            return
        del self._session_history[index - 1]
        # Every remaining card's 1-based index shifts once one entry is
        # removed -- rebuild from scratch rather than patching indices in
        # place, same as _on_clear_history_clicked already does.
        self._rebuild_history_cards()
        _maybe_persist_history(self)

    def _on_clear_history_clicked(self) -> None:
        if not self._session_history:
            return
        with self._modal():  # see _do_tick's guard comment on _modal()
            confirmed = messagebox.askyesno(
                self._t("history_clear_confirm_title"),
                self._t("history_clear_confirm_prompt", n=len(self._session_history)),
                parent=self.root,
            )
        if not confirmed:
            return
        self._session_history.clear()
        self._rebuild_history_cards()
        _maybe_persist_history(self)

    def _set_status_error(self, text: str) -> None:
        self._status_pill.configure(text=text, fg_color=SURFACE_2, text_color=HP_COLOR)

    # ---- render --------------------------------------------------------

    def _render(self, snap: StatSnapshot) -> None:
        economy_snapshot = None
        self._value_labels["level"].configure(text=str(snap.level) if snap.level is not None else "--")

        if snap.hp_cur is not None:
            self._value_labels["hp"].configure(text=f"{snap.hp_cur}/{snap.hp_max}")
            if snap.hp_max:
                self._bars["hp"].set(max(0.0, min(1.0, snap.hp_cur / snap.hp_max)))
        else:
            self._value_labels["hp"].configure(text="--")

        if snap.mp_cur is not None:
            self._value_labels["mp"].configure(text=f"{snap.mp_cur}/{snap.mp_max}")
            if snap.mp_max:
                self._bars["mp"].set(max(0.0, min(1.0, snap.mp_cur / snap.mp_max)))
        else:
            self._value_labels["mp"].configure(text="--")

        pct = f"  ({snap.exp_pct:.2f}%)" if snap.exp_pct is not None and self._settings.show_exp_pct else ""
        if snap.exp_cur is not None:
            self._value_labels["exp"].configure(text=f"{snap.exp_cur:,}{pct}")
            if snap.exp_pct is not None:
                self._bars["exp"].set(max(0.0, min(1.0, snap.exp_pct / 100)))
        else:
            self._value_labels["exp"].configure(text="--")

        start_exp = self._session.start_exp
        self._value_labels["startexp"].configure(text=f"{start_exp:,}" if start_exp is not None else "--")

        self._update_timer_label()

        exp_diff = self._session.exp_diff
        # Total EXP required for the current level isn't shown directly by the
        # game, but can be derived from any single tick that has both the
        # absolute value and percentage: total = cur / (pct/100). Anchoring
        # off the current tick (rather than diffing OCR'd percentages
        # directly) is more robust since per-level EXP totals are constant,
        # while independently-read percentages carry their own OCR noise on
        # top of the cur value's.
        total_exp = snap.exp_cur / (snap.exp_pct / 100) if snap.exp_cur and snap.exp_pct else None

        if exp_diff is not None:
            pct_s = f"  (+{exp_diff / total_exp * 100:.2f}%)" if total_exp and self._settings.show_exp_pct else ""
            self._value_labels["expdiff"].configure(text=f"+{exp_diff:,}{pct_s}")
        else:
            self._value_labels["expdiff"].configure(text="--")

        exp_rate = self._session.exp_per_hour
        self._value_labels["exprate"].configure(
            text=f"{exp_rate:,.0f}" if exp_rate is not None else "--",
            text_color=EXP_COLOR if exp_rate is not None else INK,
        )

        # ETA to level up: current session's EXP/sec rate, projected against
        # the EXP still needed (total - cur). Needs a few seconds of session
        # data first -- extrapolating off a 1-2 second sample swings wildly.
        elapsed = self._session.elapsed()
        eta_s = None
        if exp_diff and exp_diff > 0 and elapsed > 3 and total_exp and snap.exp_cur:
            rate_per_sec = exp_diff / elapsed
            remaining_exp = total_exp - snap.exp_cur
            if rate_per_sec > 0:
                eta_s = remaining_exp / rate_per_sec
        self._value_labels["eta"].configure(text=_fmt_duration(eta_s) if eta_s is not None else "--")

        # Projected session total: current rate extrapolated across the full
        # window setting, not just what's elapsed so far -- see
        # Session.projected_exp (same 3s/positive-gain guard as ETA above,
        # for the same reason).
        proj = self._session.projected_exp(self._settings.window_min * 60)
        if proj is not None:
            proj_pct_s = f"  (+{proj / total_exp * 100:.2f}%)" if total_exp and self._settings.show_exp_pct else ""
            self._value_labels["projexp"].configure(text=f"+{proj:,}{proj_pct_s}")
        else:
            self._value_labels["projexp"].configure(text="--")

        hp_loss, mp_loss = self._session.hp_loss, self._session.mp_loss
        self._value_labels["hploss"].configure(
            text=_fmt_loss(hp_loss), text_color=HP_COLOR if hp_loss > 0 else INK_FAINT
        )
        self._value_labels["mploss"].configure(
            text=_fmt_loss(mp_loss), text_color=MP_COLOR if mp_loss > 0 else INK_FAINT
        )

        economy = getattr(self, "_economy", None)
        if economy is not None:
            economy_snapshot = economy.snapshot
            if economy_snapshot.shortcut_baseline_ready:
                # Pair each slot's initial quantity with its latest trusted
                # quantity.  This makes a potion drop visible immediately and
                # gives the user a direct sanity check for OCR baselines.
                inventory_values = []
                for slot_id, initial in economy_snapshot.shortcut_baseline.items():
                    current = economy_snapshot.shortcut_current.get(slot_id, initial)
                    inventory_values.append(f"{slot_id}:{initial:,}→{current:,}")
                inventory_text = " · ".join(inventory_values) or self._t("potion_inventory_pending")
            else:
                inventory_text = self._t("potion_inventory_pending")
            self._value_labels["shortcut_inventory"].configure(
                text=inventory_text,
                text_color=OK_COLOR if economy_snapshot.shortcut_baseline_ready else EXP_COLOR,
            )
            self._value_labels["mesos"].configure(
                text=f"+{economy_snapshot.mesos:,}",
                text_color=EXP_COLOR if economy_snapshot.mesos else INK,
            )
            self._value_labels["hp_potions"].configure(
                text=self._t(
                    "potion_compact",
                    uses=economy_snapshot.hp_potion_uses,
                    cost=economy_snapshot.hp_potion_cost,
                ),
                text_color=HP_COLOR if economy_snapshot.hp_potion_uses else INK,
            )
            self._value_labels["mp_potions"].configure(
                text=self._t(
                    "potion_compact",
                    uses=economy_snapshot.mp_potion_uses,
                    cost=economy_snapshot.mp_potion_cost,
                ),
                text_color=MP_COLOR if economy_snapshot.mp_potion_uses else INK,
            )
            self._value_labels["shared_potions"].configure(
                text=self._t(
                    "potion_compact",
                    uses=economy_snapshot.shared_potion_uses,
                    cost=economy_snapshot.shared_potion_cost,
                ),
                text_color=EXP_COLOR if economy_snapshot.shared_potion_uses else INK,
            )
            self._value_labels["hp_recovery"].configure(
                text=self._t(
                    "recovery_compact",
                    natural=f"{economy_snapshot.hp_recovery_natural:,}",
                    potion=f"{economy_snapshot.hp_recovery_potion:,}",
                ),
                text_color=HP_COLOR if economy_snapshot.hp_recovery_potion else INK,
            )
            self._value_labels["mp_recovery"].configure(
                text=self._t(
                    "recovery_compact",
                    natural=f"{economy_snapshot.mp_recovery_natural:,}",
                    potion=f"{economy_snapshot.mp_recovery_potion:,}",
                ),
                text_color=MP_COLOR if economy_snapshot.mp_recovery_potion else INK,
            )
            self._value_labels["hp_recovery_savings"].configure(
                text=self._t(
                    "recovery_savings_compact",
                    amount=f"{economy_snapshot.hp_recovery_savings:,.1f}",
                ),
                text_color=HP_COLOR if economy_snapshot.hp_recovery_savings else INK,
            )
            self._value_labels["mp_recovery_savings"].configure(
                text=self._t(
                    "recovery_savings_compact",
                    amount=f"{economy_snapshot.mp_recovery_savings:,.1f}",
                ),
                text_color=MP_COLOR if economy_snapshot.mp_recovery_savings else INK,
            )

        getattr(self, "_render_floating", lambda *_args: None)(snap, proj, eta_s, economy_snapshot)

        # Pause/stop/calibration are user- or engine-driven states that take
        # priority over the activity-based idle/tracking read below -- e.g. a
        # paused session with real HP/MP/EXP movement in its history isn't
        # "Idle", it's "Paused".
        if getattr(self, "_ocr_loading", False):
            self._status_pill.configure(
                text=self._t("status_loading"), fg_color=SURFACE_2, text_color=EXP_COLOR
            )
        elif getattr(self, "_ocr_error", None):
            self._status_pill.configure(
                text=self._t("status_ocr_failed"), fg_color=SURFACE_2, text_color=HP_COLOR
            )
        elif getattr(self, "_last_capture_error", None):
            self._status_pill.configure(
                text=self._localize_error(self._last_capture_error.removeprefix("OCR: ")),
                fg_color=SURFACE_2, text_color=HP_COLOR,
            )
        elif self._run_state == "paused":
            self._status_pill.configure(text=self._t("status_paused"), fg_color=SURFACE_2, text_color=EXP_COLOR)
        elif self._run_state == "stopped":
            self._status_pill.configure(text=self._t("status_stopped"), fg_color=SURFACE_2, text_color=INK_DIM)
        elif self._run_state == "running" and self._settings.track_potions and getattr(self, "_potion_baseline_pending", False):
            self._status_pill.configure(
                text=self._t("status_potion_baseline"), fg_color=SURFACE_2, text_color=EXP_COLOR
            )
        elif self._session.is_calibrating:
            self._status_pill.configure(text=self._t("status_calibrating"), fg_color=SURFACE_2, text_color=EXP_COLOR)
        else:
            # Idle only if NONE of HP/MP/EXP have changed recently within this
            # session -- any one of them moving counts as activity, not idle.
            idle = hp_loss == 0 and mp_loss == 0 and (exp_diff or 0) == 0
            if idle:
                self._status_pill.configure(text=self._t("status_idle"), fg_color=SURFACE_2, text_color=INK_DIM)
            else:
                self._status_pill.configure(text=self._t("status_tracking"), fg_color=TRACK_BG, text_color=OK_COLOR)

    def _render_floating(self, snap: StatSnapshot, projected_exp: int | None, eta_s: float | None, economy_snapshot) -> None:
        values = getattr(self, "_floating_metric_values", None)
        if not values:
            return
        elapsed = self._session.elapsed()
        target_s = self._settings.window_min * 60

        def projected(counter: int, *, minimum_elapsed: float = 3.0) -> str:
            if counter <= 0 or elapsed < minimum_elapsed:
                return "--"
            return f"{int(counter / elapsed * target_s):,}"

        mesos = getattr(economy_snapshot, "mesos", 0) if economy_snapshot is not None else 0
        potion_cost = getattr(economy_snapshot, "potion_cost", 0) if economy_snapshot is not None else 0
        projected_mesos = projected(mesos)
        # A single confirmed drink in the first few seconds is real data, but
        # dividing it by a tiny elapsed time produces a misleading interval
        # estimate.  Keep the actual cost in the detailed panel and wait for a
        # short observation window before projecting it to ten minutes.
        projected_cost = projected(potion_cost, minimum_elapsed=POTION_PROJECTION_MIN_SECONDS)
        hp_potions = getattr(economy_snapshot, "hp_potion_uses", 0) if economy_snapshot is not None else 0
        mp_potions = getattr(economy_snapshot, "mp_potion_uses", 0) if economy_snapshot is not None else 0
        shared_potions = getattr(economy_snapshot, "shared_potion_uses", 0) if economy_snapshot is not None else 0
        hp_recovery = getattr(economy_snapshot, "hp_recovery_natural", 0) if economy_snapshot is not None else 0
        mp_recovery = getattr(economy_snapshot, "mp_recovery_natural", 0) if economy_snapshot is not None else 0
        hp_recovery_savings = getattr(economy_snapshot, "hp_recovery_savings", 0.0) if economy_snapshot is not None else 0.0
        mp_recovery_savings = getattr(economy_snapshot, "mp_recovery_savings", 0.0) if economy_snapshot is not None else 0.0
        if economy_snapshot is not None and economy_snapshot.shortcut_baseline_ready:
            inventory_values = [
                f"{slot_id}:{economy_snapshot.shortcut_current.get(slot_id, initial):,}"
                for slot_id, initial in economy_snapshot.shortcut_baseline.items()
            ]
            shortcut_inventory = " · ".join(inventory_values) or "--"
        else:
            shortcut_inventory = self._t("potion_inventory_pending")
        exp_diff = self._session.exp_diff
        exp_rate = self._session.exp_per_hour
        rendered = {
            "proj_exp": f"+{projected_exp:,}" if projected_exp is not None else "--",
            "eta": _fmt_duration(eta_s) if eta_s is not None else "--",
            "proj_mesos": f"+{projected_mesos}" if projected_mesos != "--" else "--",
            "proj_potion_cost": f"-{projected_cost}" if projected_cost != "--" else "--",
            "level": str(snap.level) if snap.level is not None else "--",
            "hp": f"{snap.hp_cur}/{snap.hp_max}" if snap.hp_cur is not None else "--",
            "mp": f"{snap.mp_cur}/{snap.mp_max}" if snap.mp_cur is not None else "--",
            "exp": f"{snap.exp_cur:,}" if snap.exp_cur is not None else "--",
            "exp_diff": f"+{exp_diff:,}" if exp_diff is not None else "--",
            "exp_rate": f"{exp_rate:,.0f}" if exp_rate is not None else "--",
            "mesos": f"+{mesos:,}",
            "hp_potions": self._t("potion_compact", uses=hp_potions, cost=getattr(economy_snapshot, "hp_potion_cost", 0) if economy_snapshot is not None else 0),
            "mp_potions": self._t("potion_compact", uses=mp_potions, cost=getattr(economy_snapshot, "mp_potion_cost", 0) if economy_snapshot is not None else 0),
            "shortcut_inventory": shortcut_inventory,
            "shared_potions": self._t("potion_compact", uses=shared_potions, cost=getattr(economy_snapshot, "shared_potion_cost", 0) if economy_snapshot is not None else 0),
            "hp_recovery": f"+{hp_recovery:,}",
            "mp_recovery": f"+{mp_recovery:,}",
            "hp_recovery_savings": f"{hp_recovery_savings:,.1f}",
            "mp_recovery_savings": f"{mp_recovery_savings:,.1f}",
            "hp_loss": _fmt_loss(self._session.hp_loss),
            "mp_loss": _fmt_loss(self._session.mp_loss),
        }
        for key, label in values.items():
            label.configure(text=rendered.get(key, "--"))
        self._render_context()

    def _on_close(self) -> None:
        """Flush the latest preferences/history before closing the HUD."""
        if self._run_state in ("running", "paused"):
            self._commit_session_to_history()
        monitor = getattr(self, "_monitor", None)
        if monitor is not None:
            monitor.stop()
        _maybe_persist_settings(self)
        _maybe_persist_history(self)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
