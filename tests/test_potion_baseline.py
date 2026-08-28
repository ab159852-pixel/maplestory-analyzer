"""Regression tests for multi-cell shortcut inventory calibration."""
from __future__ import annotations

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
