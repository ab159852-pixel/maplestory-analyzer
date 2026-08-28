"""Regression tests for multi-cell shortcut inventory calibration."""
from __future__ import annotations

from maple_analyzer.economy import EconomyTracker
from maple_analyzer.overlay import OverlayApp
from maple_analyzer.settings import PotionSlotConfig, Settings


class _EconomySpy:
    def __init__(self):
        self.baselines = []

    def prime_quick_slot_counts(self, counts, *, now=None):
        self.baselines.append((dict(counts), now))


def _app_for_baseline() -> object:
    app = OverlayApp.__new__(OverlayApp)
    app._settings = Settings(
        potion_slots=[
            PotionSlotConfig(slot="6", kind="hp", enabled=True),
            PotionSlotConfig(slot="7", kind="mp", enabled=True),
        ]
    )
    app._economy = _EconomySpy()
    app._potion_baseline_pending = True
    app._potion_baseline_samples = []
    app._last_logged_shortcut_counts = None
    app._log = lambda _message: None
    return app


def test_initial_baseline_waits_for_every_enabled_shortcut_slot():
    app = _app_for_baseline()

    # Slot 6 is readable first.  It must not close the baseline before slot 7
    # has supplied a confirmed value of its own.
    app._record_auxiliary_counts({"6": 2676}, now=1.0)
    app._record_auxiliary_counts({"6": 2676}, now=2.0)
    assert app._potion_baseline_pending is True
    assert app._economy.baselines == []

    # The second cell can arrive on later frames; the retained sample window
    # should combine the two partial results into one complete baseline.
    app._record_auxiliary_counts({"6": 2676, "7": 1875}, now=3.0)
    app._record_auxiliary_counts({"6": 2676, "7": 1875}, now=4.0)

    assert app._potion_baseline_pending is False
    assert app._economy.baselines == [
        ({"6": 2676, "7": 1875}, 4.0)
    ]


def test_initial_baseline_does_not_commit_a_slot_that_is_only_seen_once():
    app = _app_for_baseline()

    app._record_auxiliary_counts({"6": 2676, "7": 1875}, now=1.0)
    app._record_auxiliary_counts({"6": 2676}, now=2.0)
    app._record_auxiliary_counts({"6": 2676}, now=3.0)

    assert app._potion_baseline_pending is True
    assert app._economy.baselines == []


def test_initial_baseline_does_not_block_forever_when_one_slot_stays_unreadable():
    app = _app_for_baseline()
    app._potion_baseline_started_at = 0.0

    app._record_auxiliary_counts({"6": 2676}, now=1.0)
    app._record_auxiliary_counts({"6": 2676}, now=2.0)
    app._record_auxiliary_counts({"6": 2676}, now=6.0)

    # Slot 6 can start accounting after the bounded grace period.  Slot 7 is
    # not charged when it eventually becomes readable; EconomyTracker adds a
    # fresh per-slot baseline for it.
    assert app._potion_baseline_pending is False
    assert app._economy.baselines == [
        ({"6": 2676}, 6.0)
    ]


def test_late_slot_is_added_without_becoming_a_potion_charge():
    app = _app_for_baseline()
    app._economy = EconomyTracker(app._settings.potion_slots)
    app._potion_baseline_started_at = 0.0

    app._record_auxiliary_counts({"6": 2676}, now=1.0)
    app._record_auxiliary_counts({"6": 2676}, now=2.0)
    app._record_auxiliary_counts({"6": 2676}, now=6.0)

    uses = app._economy.record_quick_slot_counts(
        {"6": 2676, "7": 1875}, now=7.0, immediate=True
    )

    assert uses == 0
    assert app._economy.snapshot.shortcut_baseline == {
        "6": 2676,
        "7": 1875,
    }
