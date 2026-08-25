"""Small, recoverable persistence layer for user preferences and history.

The executable is commonly placed in a read-only or disposable folder, so
state must never be written beside the executable.  This module keeps the
format deliberately boring: versioned UTF-8 JSON for settings/history and a
UTF-8-with-BOM CSV export that opens cleanly in Excel on Windows.

All public helpers accept an optional ``data_dir``.  The application uses the
per-user directory selected by :func:`app_data_dir`; tests and future import
tools can pass a temporary directory without touching a real profile.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Iterable

from .rate import SessionSummary
from .settings import PotionSlotConfig, Settings, default_potion_slots

STORE_VERSION = 3
SETTINGS_FILENAME = "settings.json"
HISTORY_FILENAME = "history.json"


def app_data_dir() -> Path:
    """Return the per-user directory used by the packaged application.

    ``LOCALAPPDATA`` is the right home for machine-local HUD state.  The
    ``APPDATA`` fallback keeps the tool usable on older Windows setups and in
    non-Windows development environments.
    """
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / "MapleStoryAnalyzer"
    return Path.home() / ".maplestory-analyzer"


def _path(data_dir: str | Path | None, filename: str) -> Path:
    return Path(data_dir) / filename if data_dir is not None else app_data_dir() / filename


def _atomic_write(path: Path, text: str) -> None:
    """Write one file without leaving a half-written JSON document behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        # ``replace`` removes the temporary path on success.  Cleanup is
        # best-effort so an interrupted write never hides the original file.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        # A corrupt optional state file should recover to defaults, not stop
        # the OCR HUD from starting.  The next successful save repairs it.
        return None


def _settings_payload(settings: Settings) -> dict[str, object]:
    return {
        "version": STORE_VERSION,
        "settings": dataclasses.asdict(settings),
    }


def save_settings(settings: Settings, data_dir: str | Path | None = None) -> None:
    """Persist settings atomically to the selected user-data directory."""
    path = _path(data_dir, SETTINGS_FILENAME)
    _atomic_write(path, json.dumps(_settings_payload(settings), ensure_ascii=False, indent=2) + "\n")


def _bounded_int(value: object, default: int, lower: int, upper: int) -> int:
    # bool is an int subclass but is never a meaningful numeric preference.
    if isinstance(value, bool):
        return default
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return max(lower, min(upper, number))


def _bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def load_settings(data_dir: str | Path | None = None) -> Settings:
    """Load settings, validating each field independently.

    Unknown keys are ignored so a future version can add preferences without
    making an older executable unusable.  A malformed field falls back to the
    current default while valid sibling fields are retained.
    """
    settings = Settings()
    raw = _read_json(_path(data_dir, SETTINGS_FILENAME))
    if not isinstance(raw, dict):
        return settings
    values = raw.get("settings", raw)
    if not isinstance(values, dict):
        return settings
    try:
        stored_version = int(raw.get("version", 1))
    except (TypeError, ValueError):
        stored_version = 1

    settings.window_min = _bounded_int(values.get("window_min"), settings.window_min, 1, 60)
    settings.scale_pct = _bounded_int(values.get("scale_pct"), settings.scale_pct, 50, 150)
    settings.sample_interval_ms = _bounded_int(
        values.get("sample_interval_ms"), settings.sample_interval_ms, 200, 1000
    )
    settings.pickup_interval_ms = _bounded_int(
        values.get("pickup_interval_ms"), settings.pickup_interval_ms, 100, 1000
    )
    legacy_recovery = _bounded_int(values.get("potion_recovery_default"), 0, 0, 1_000_000)
    settings.potion_recovery_hp_default = _bounded_int(
        values.get("potion_recovery_hp_default"), legacy_recovery, 0, 1_000_000
    )
    settings.potion_recovery_mp_default = _bounded_int(
        values.get("potion_recovery_mp_default"), legacy_recovery, 0, 1_000_000
    )
    settings.floating_opacity_pct = _bounded_int(
        values.get("floating_opacity_pct"), settings.floating_opacity_pct, 45, 100
    )
    settings.language = values.get("language") if values.get("language") in ("zh", "en") else settings.language

    for field_name in (
        "show_level", "show_hp", "show_mp", "show_exp", "show_exp_pct", "show_exp_diff",
        "show_exp_rate", "show_eta", "show_proj_exp", "show_hp_loss", "show_mp_loss",
        "show_mesos", "show_hp_potions", "show_mp_potions", "show_shared_potions",
        "show_hp_recovery", "show_mp_recovery", "show_hp_recovery_savings",
        "show_mp_recovery_savings",
        "topmost", "auto_stop", "save_on_restart", "track_pickup_messages", "track_potions",
        "floating_on_start", "auto_context",
    ):
        setattr(settings, field_name, _bool(values.get(field_name), getattr(settings, field_name)))

    # v1 shipped with auto_stop=True and therefore made a Start click look
    # like a hard ten-minute limit.  The v2 default is continuous ten-minute
    # history rollover; migrate old persisted preferences to that behavior.
    if stored_version < 2:
        settings.auto_stop = False

    for field_name in ("job_name_override", "map_name_override"):
        value = values.get(field_name)
        if isinstance(value, str):
            setattr(settings, field_name, value.strip()[:32])
    raw_floating = values.get("floating_fields")
    if isinstance(raw_floating, list):
        allowed_floating = {
            "proj_exp", "eta", "proj_mesos", "proj_potion_cost", "level", "hp", "mp",
            "exp", "exp_diff", "exp_rate", "mesos", "hp_potions", "mp_potions",
            "shared_potions", "hp_recovery", "mp_recovery", "hp_recovery_savings",
            "mp_recovery_savings", "hp_loss", "mp_loss", "shortcut_inventory",
        }
        settings.floating_fields = [
            str(item) for item in raw_floating
            if str(item) in allowed_floating
        ]

    raw_slots = values.get("potion_slots")
    if isinstance(raw_slots, list):
        slots: list[PotionSlotConfig] = []
        for index, raw_slot in enumerate(raw_slots[:12], start=1):
            if not isinstance(raw_slot, dict):
                continue
            slot = str(raw_slot.get("slot", index)).strip()[:12] or str(index)
            name = str(raw_slot.get("name", "")).strip()[:40]
            kind = raw_slot.get("kind", "hp")
            kind = kind if kind in ("hp", "mp", "both") else "hp"
            slots.append(PotionSlotConfig(
                slot=slot,
                name=name,
                kind=kind,
                cost=_bounded_int(raw_slot.get("cost"), 0, 0, 1_000_000_000),
                recovery=_bounded_int(raw_slot.get("recovery"), 0, 0, 1_000_000),
                enabled=_bool(raw_slot.get("enabled"), True),
            ))
        settings.potion_slots = slots
    elif not settings.potion_slots:
        settings.potion_slots = default_potion_slots()
    return settings


def _summary_payload(summary: SessionSummary) -> dict[str, object]:
    return dataclasses.asdict(summary)


def save_history(history: Iterable[SessionSummary], data_dir: str | Path | None = None) -> None:
    """Persist finalized sessions in chronological order."""
    payload = {
        "version": STORE_VERSION,
        "sessions": [_summary_payload(summary) for summary in history],
    }
    path = _path(data_dir, HISTORY_FILENAME)
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: object) -> int | None:
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _nonnegative_float(value: object) -> float | None:
    number = _finite_number(value)
    if number is None or number < 0:
        return None
    return number


def _decode_summary(raw: object) -> SessionSummary | None:
    if not isinstance(raw, dict):
        return None
    start_time = _finite_number(raw.get("start_time"))
    end_time = _finite_number(raw.get("end_time"))
    hp_loss = _nonnegative_int(raw.get("hp_loss"))
    mp_loss = _nonnegative_int(raw.get("mp_loss"))
    if start_time is None or end_time is None or end_time < start_time or hp_loss is None or mp_loss is None:
        return None

    start_exp = _nonnegative_int(raw.get("start_exp")) if raw.get("start_exp") is not None else None
    end_exp = _nonnegative_int(raw.get("end_exp")) if raw.get("end_exp") is not None else None
    total_exp = _finite_number(raw.get("total_exp")) if raw.get("total_exp") is not None else None
    interval = _finite_number(raw.get("interval_minutes")) if raw.get("interval_minutes") is not None else None
    exp_gained = _nonnegative_int(raw.get("exp_gained")) if raw.get("exp_gained") is not None else None
    mesos = _nonnegative_int(raw.get("mesos", 0))
    potion_uses = _nonnegative_int(raw.get("potion_uses", 0))
    potion_cost = _nonnegative_int(raw.get("potion_cost", 0))
    # History written before HP/MP classification existed only has a total.
    # Keep that information visible as an unclassified/shared bucket instead
    # of silently dropping it when the new UI loads the old JSON file.  New
    # rows always carry the explicit split fields (including zero), so they
    # are not affected by this migration fallback.
    hp_potion_uses = _nonnegative_int(raw.get("hp_potion_uses", 0))
    hp_potion_cost = _nonnegative_int(raw.get("hp_potion_cost", 0))
    mp_potion_uses = _nonnegative_int(raw.get("mp_potion_uses", 0))
    mp_potion_cost = _nonnegative_int(raw.get("mp_potion_cost", 0))
    shared_potion_uses = _nonnegative_int(raw.get("shared_potion_uses", potion_uses or 0))
    shared_potion_cost = _nonnegative_int(raw.get("shared_potion_cost", potion_cost or 0))
    hp_recovery_natural = _nonnegative_int(raw.get("hp_recovery_natural", 0))
    hp_recovery_potion = _nonnegative_int(raw.get("hp_recovery_potion", 0))
    mp_recovery_natural = _nonnegative_int(raw.get("mp_recovery_natural", 0))
    mp_recovery_potion = _nonnegative_int(raw.get("mp_recovery_potion", 0))
    hp_recovery_savings = _nonnegative_float(raw.get("hp_recovery_savings", 0.0))
    mp_recovery_savings = _nonnegative_float(raw.get("mp_recovery_savings", 0.0))
    if None in (
        mesos, potion_uses, potion_cost, hp_potion_uses, hp_potion_cost,
        mp_potion_uses, mp_potion_cost, shared_potion_uses, shared_potion_cost,
        hp_recovery_natural,
        hp_recovery_potion, mp_recovery_natural, mp_recovery_potion,
        hp_recovery_savings, mp_recovery_savings,
    ):
        return None
    breakdown_raw = raw.get("potion_breakdown")
    breakdown = None
    if isinstance(breakdown_raw, dict):
        breakdown = {
            str(name): count
            for name, value in breakdown_raw.items()
            if (count := _nonnegative_int(value)) is not None
        }
    name = raw.get("name")
    if name is not None and not isinstance(name, str):
        name = None
    job_name = raw.get("job_name")
    if job_name is not None and not isinstance(job_name, str):
        job_name = None
    map_name = raw.get("map_name")
    if map_name is not None and not isinstance(map_name, str):
        map_name = None

    return SessionSummary(
        start_time=start_time,
        end_time=end_time,
        start_exp=start_exp,
        end_exp=end_exp,
        hp_loss=hp_loss,
        mp_loss=mp_loss,
        total_exp=total_exp,
        interval_minutes=interval,
        exp_gained=exp_gained,
        name=name,
        job_name=job_name,
        map_name=map_name,
        mesos=mesos,
        potion_uses=potion_uses,
        potion_cost=potion_cost,
        hp_potion_uses=hp_potion_uses,
        hp_potion_cost=hp_potion_cost,
        mp_potion_uses=mp_potion_uses,
        mp_potion_cost=mp_potion_cost,
        shared_potion_uses=shared_potion_uses,
        shared_potion_cost=shared_potion_cost,
        hp_recovery_natural=hp_recovery_natural,
        hp_recovery_potion=hp_recovery_potion,
        mp_recovery_natural=mp_recovery_natural,
        mp_recovery_potion=mp_recovery_potion,
        hp_recovery_savings=hp_recovery_savings,
        mp_recovery_savings=mp_recovery_savings,
        potion_breakdown=breakdown,
    )


def load_history(data_dir: str | Path | None = None) -> list[SessionSummary]:
    """Load valid history rows, skipping only malformed rows."""
    raw = _read_json(_path(data_dir, HISTORY_FILENAME))
    if isinstance(raw, dict):
        rows = raw.get("sessions", [])
    else:
        # Accept a plain list as a small migration courtesy for early builds.
        rows = raw
    if not isinstance(rows, list):
        return []
    return [summary for row in rows if (summary := _decode_summary(row)) is not None]


def _csv_value(value: object | None) -> object:
    return "" if value is None else value


def export_history_csv(
    history: Iterable[SessionSummary],
    destination: str | Path,
    *,
    language: str = "zh",
) -> None:
    """Export history in an Excel-friendly CSV format.

    The numeric columns remain machine-readable while the first row is
    localized for the current UI language.  A UTF-8 BOM is intentional:
    Windows Excel otherwise guesses the legacy system code page for Chinese.
    """
    if language == "en":
        headers = [
            "Session", "Start", "End", "Duration (min)", "Start EXP", "End EXP",
            "EXP gained", "EXP / hour", "EXP %", "HP loss", "MP loss", "Interval (min)",
            "Job", "Map",
            "Mesos", "HP potion uses", "HP potion cost", "MP potion uses", "MP potion cost",
            "Shared potion uses", "Shared potion cost", "Potion uses (total)", "Potion cost (total)",
            "Natural HP recovery", "Potion HP recovery",
            "Natural MP recovery", "Potion MP recovery", "HP recovery saved", "MP recovery saved",
        ]
    else:
        headers = [
            "紀錄", "開始時間", "結束時間", "時長（分鐘）", "起始經驗值", "結束經驗值",
            "經驗值增加", "每小時經驗值", "經驗值百分比", "HP 損失", "MP 損失", "區間（分鐘）",
            "職業", "地圖",
            "楓幣收入", "HP 藥水次數", "HP 藥水成本", "MP 藥水次數", "MP 藥水成本",
            "共用藥水次數", "共用藥水成本", "藥水總次數", "藥水總成本", "自然回復 HP", "藥水回復 HP",
            "自然回復 MP", "藥水回復 MP", "自然／技能節省 HP 楓幣", "自然／技能節省 MP 楓幣",
        ]

    with Path(destination).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for index, summary in enumerate(history, start=1):
            exp_diff = summary.exp_diff
            rate = summary.exp_per_hour
            writer.writerow([
                summary.name or f"Session #{index}",
                summary.start_time,
                summary.end_time,
                round(summary.duration_s / 60, 3),
                _csv_value(summary.start_exp),
                _csv_value(summary.end_exp),
                _csv_value(exp_diff),
                round(rate, 3) if rate is not None else "",
                round(summary.exp_pct_diff, 4) if summary.exp_pct_diff is not None else "",
                summary.hp_loss,
                summary.mp_loss,
                _csv_value(summary.interval_minutes),
                _csv_value(summary.job_name),
                _csv_value(summary.map_name),
                summary.mesos,
                summary.hp_potion_uses,
                summary.hp_potion_cost,
                summary.mp_potion_uses,
                summary.mp_potion_cost,
                summary.shared_potion_uses,
                summary.shared_potion_cost,
                summary.potion_uses,
                summary.potion_cost,
                summary.hp_recovery_natural,
                summary.hp_recovery_potion,
                summary.mp_recovery_natural,
                summary.mp_recovery_potion,
                round(summary.hp_recovery_savings, 1),
                round(summary.mp_recovery_savings, 1),
            ])
