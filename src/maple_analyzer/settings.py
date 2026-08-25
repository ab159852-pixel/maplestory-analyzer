"""UI-layer settings, as a single struct.

Deliberately a plain dataclass with JSON-primitive fields only (str/int/bool),
same reasoning as rate.py's SessionSummary -- this is the shape a future
persistence layer (see ~/.claude/notes/maplestory-analyzer/ui-plan-2026-08-17.md)
would load/save wholesale, e.g. `json.dumps(dataclasses.asdict(settings))`.
overlay.py should read/write through a single `self._settings` instance rather
than scattering individual attributes across OverlayApp, so that swapping in
disk persistence later is "load one struct, save one struct" instead of a
field-by-field migration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .i18n import Lang


@dataclass
class PotionSlotConfig:
    """User mapping for one consumable slot in the in-game shortcut bar."""

    slot: str
    name: str = ""
    kind: str = "hp"  # hp, mp, or both
    cost: int = 0
    recovery: int = 0
    enabled: bool = True


def default_potion_slots() -> list[PotionSlotConfig]:
    # Empty rows must not be treated as active slots.  The user can still
    # configure all eight rows in Settings; until then, a blank OCR read must
    # never become a zero-cost potion usage event.
    return [PotionSlotConfig(slot=str(index), enabled=False) for index in range(1, 9)]


@dataclass
class Settings:
    window_min: int = 10
    # Individual live-HUD fields.  The older show_hp/show_mp/show_exp names
    # remain the persisted/public settings for those three core bars; the
    # additional flags let a compact floating HUD hide everything the user
    # does not need while grinding.
    show_level: bool = True
    show_hp: bool = True
    show_mp: bool = True
    show_exp: bool = True
    show_exp_pct: bool = True
    show_exp_diff: bool = True
    show_exp_rate: bool = True
    show_eta: bool = True
    show_proj_exp: bool = True
    show_hp_loss: bool = True
    show_mp_loss: bool = True
    show_mesos: bool = True
    show_hp_potions: bool = True
    show_mp_potions: bool = True
    show_shared_potions: bool = True
    show_hp_recovery: bool = True
    show_mp_recovery: bool = True
    show_hp_recovery_savings: bool = True
    show_mp_recovery_savings: bool = True
    topmost: bool = True
    scale_pct: int = 100
    language: Lang = "zh"
    # Whether the timer rolling over finalizes+commits to History and then
    # STOPS.  The default is off: every interval is recorded and the next one
    # starts automatically, so pressing Start begins continuous monitoring.
    auto_stop: bool = False
    # Whether the manual Restart button commits the in-progress session to
    # History before discarding it. Governs Restart only -- auto_stop's
    # timer-driven finalize always commits regardless of this. Default on
    # per user request (2026-08-23, revised from the original off default):
    # a restarted session's progress should be kept unless the user opts
    # out, and per-entry deletion (see OverlayApp._on_delete_history_clicked)
    # is the escape hatch for a throwaway entry now that one exists.
    save_on_restart: bool = True
    # The live game's status bar is sampled at 300ms.  This is intentionally a
    # setting so a slower machine can trade responsiveness for CPU later,
    # without another code change or release rebuild.
    sample_interval_ms: int = 300
    track_pickup_messages: bool = True
    track_potions: bool = True
    # Per-resource fallback recovery used when a potion slot's quantity OCR is
    # unavailable.  Keeping HP and MP separate prevents a fixed HP heal from
    # classifying a natural/MP recovery as the wrong potion type.
    potion_recovery_hp_default: int = 0
    potion_recovery_mp_default: int = 0
    # HUD presentation.  Alpha is applied only after Start when
    # floating_on_start is enabled, so the settings window is fully readable.
    floating_on_start: bool = True
    floating_opacity_pct: int = 84
    # Visible game context.  OCR is attempted automatically; these optional
    # values are a safe manual fallback when a custom UI scale/font hides the
    # tiny map/job labels.
    auto_context: bool = True
    job_name_override: str = ""
    map_name_override: str = ""
    # Compact horizontal HUD defaults to the four user-requested planning
    # metrics.  Extra fields can be enabled independently without changing
    # the full Live tab's visibility settings.
    floating_fields: list[str] = field(default_factory=lambda: [
        "proj_exp", "eta", "proj_mesos", "proj_potion_cost",
        "shortcut_inventory",
    ])
    potion_slots: list[PotionSlotConfig] = field(default_factory=default_potion_slots)
