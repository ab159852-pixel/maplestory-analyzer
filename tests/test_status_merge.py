"""Regression tests for the live status snapshot merge boundary."""

from maple_analyzer.monitor import merge_status_snapshots
from maple_analyzer.parser import StatSnapshot


def _snapshot(*, level=69, exp_cur=100_000, exp_pct=10.0):
    return StatSnapshot(
        level=level,
        hp_cur=2_000,
        hp_max=2_200,
        mp_cur=1_000,
        mp_max=1_400,
        exp_cur=exp_cur,
        exp_pct=exp_pct,
    )


def test_same_level_exp_drop_is_held_for_display():
    previous = _snapshot(exp_cur=121_000, exp_pct=12.1)
    incoming = _snapshot(exp_cur=12_100, exp_pct=1.21)

    merged = merge_status_snapshots(previous, incoming)

    assert merged.exp_cur == 121_000
    assert merged.exp_pct == 12.1


def test_same_level_inconsistent_exp_total_is_held():
    previous = _snapshot(exp_cur=100_000, exp_pct=10.0)
    incoming = _snapshot(exp_cur=130_000, exp_pct=10.0)

    merged = merge_status_snapshots(previous, incoming)

    assert merged.exp_cur == previous.exp_cur
    assert merged.exp_pct == previous.exp_pct


def test_confirmed_level_up_accepts_exp_reset():
    previous = _snapshot(level=69, exp_cur=990_000, exp_pct=99.0)
    incoming = _snapshot(level=70, exp_cur=10_000, exp_pct=1.0)

    merged = merge_status_snapshots(previous, incoming)

    assert merged.level == 70
    assert merged.exp_cur == 10_000
    assert merged.exp_pct == 1.0


def test_missing_exp_field_carries_forward():
    previous = _snapshot(exp_cur=100_000, exp_pct=10.0)
    incoming = _snapshot(exp_cur=None, exp_pct=None)

    merged = merge_status_snapshots(previous, incoming)

    assert merged.exp_cur == previous.exp_cur
    assert merged.exp_pct == previous.exp_pct
