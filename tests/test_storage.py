"""Persistence/export tests use a temporary directory only."""
from __future__ import annotations

import json

from maple_analyzer.rate import SessionSummary
from maple_analyzer.settings import Settings
from maple_analyzer.storage import export_history_csv, load_history, load_settings, save_history, save_settings


def _summary(name: str = "Forest") -> SessionSummary:
    return SessionSummary(
        start_time=100.0,
        end_time=460.0,
        start_exp=10_000,
        end_exp=10_900,
        hp_loss=120,
        mp_loss=80,
        total_exp=100_000,
        interval_minutes=10,
        exp_gained=900,
        name=name,
        job_name="俠盜",
        map_name="第3軍營",
        mesos=12_345,
        potion_uses=7,
        potion_cost=350,
        hp_potion_uses=4,
        hp_potion_cost=100,
        mp_potion_uses=2,
        mp_potion_cost=200,
        shared_potion_uses=1,
        shared_potion_cost=50,
        hp_recovery_natural=80,
        hp_recovery_potion=500,
        mp_recovery_natural=60,
        mp_recovery_potion=140,
        hp_recovery_savings=96.0,
        mp_recovery_savings=126.0,
        potion_breakdown={"Red Potion": 7},
    )


def test_settings_round_trip(tmp_path):
    original = Settings(
        window_min=25,
        show_hp=False,
        show_mp=True,
        show_exp=True,
        show_exp_pct=False,
        show_eta=False,
        show_proj_exp=True,
        topmost=False,
        scale_pct=120,
        language="en",
        auto_stop=False,
        save_on_restart=False,
        auto_context=True,
        job_name_override="",
        map_name_override="",
        floating_fields=["proj_exp", "eta", "proj_mesos", "proj_potion_cost"],
    )
    save_settings(original, tmp_path)

    restored = load_settings(tmp_path)

    assert restored == original


def test_invalid_settings_are_clamped_without_losing_valid_fields(tmp_path):
    (tmp_path / "settings.json").write_text(
        json.dumps({
            "settings": {
                "window_min": 999,
                "scale_pct": 1,
                "language": "en",
                "show_hp": False,
                "topmost": "yes",
            }
        }),
        encoding="utf-8",
    )

    restored = load_settings(tmp_path)

    assert restored.window_min == 60
    assert restored.scale_pct == 50
    assert restored.language == "en"
    assert restored.show_hp is False
    assert restored.topmost is True
    assert restored.show_mp is True


def test_legacy_settings_migrate_to_continuous_interval_rollover(tmp_path):
    (tmp_path / "settings.json").write_text(
        json.dumps({"version": 1, "settings": {"auto_stop": True}}),
        encoding="utf-8",
    )

    restored = load_settings(tmp_path)

    assert restored.auto_stop is False


def test_history_round_trip_and_malformed_rows_are_skipped(tmp_path):
    save_history([_summary()], tmp_path)
    payload = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    payload["sessions"].append({"start_time": "not a timestamp"})
    (tmp_path / "history.json").write_text(json.dumps(payload), encoding="utf-8")

    restored = load_history(tmp_path)

    assert len(restored) == 1
    assert restored[0] == _summary()


def test_csv_export_is_excel_friendly_and_contains_efficiency(tmp_path):
    destination = tmp_path / "history.csv"
    export_history_csv([_summary()], destination, language="zh")

    raw = destination.read_bytes()
    text = raw.decode("utf-8-sig")

    assert raw.startswith(b"\xef\xbb\xbf")
    assert "每小時經驗值" in text
    assert "楓幣收入" in text
    assert "HP 藥水次數" in text
    assert "MP 藥水成本" in text
    assert "自然／技能節省 HP 楓幣" in text
    assert "12345" in text
    assert "500" in text
    assert "96.0" in text
    assert "900" in text
    assert "9000.0" in text  # 900 EXP in six minutes -> 9,000 EXP/hour
    assert "俠盜" in text
    assert "第3軍營" in text
