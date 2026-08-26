import pytest

from maple_analyzer.economy import EconomyTracker, MesosFeedTracker, MesosObservation, parse_mesos_amount, parse_slot_count
from maple_analyzer.settings import PotionSlotConfig


def test_mesos_parser_anchors_on_pickup_marker_and_ignores_exp():
    assert parse_mesos_amount("獲取楓幣。(+1,234)") == 1234
    assert parse_mesos_amount("+123楓幣") == 123
    # RapidOCR can drop 楓幣 from the money template on a busy feed.
    assert parse_mesos_amount("獲取。(+144)") == 144
    assert parse_mesos_amount("取楓。(+144)") == 144
    assert parse_mesos_amount("瘦取椋略。(+134)") == 134
    assert parse_mesos_amount("高級楓之谷通行證Bonus經驗值。(+19)") is None
    assert parse_mesos_amount("增益經驗值。(+19)") is None
    assert parse_mesos_amount("獲得經驗值。(+144)") is None
    assert parse_slot_count("Ctrl 2,048") == 2048


def test_mesos_parser_handles_fullwidth_digits_and_common_ocr_money_glyphs():
    assert parse_mesos_amount("獲取椋略。(＋１，２３４)") == 1234
    assert parse_mesos_amount("取楓略。(+275)") == 275
    assert parse_slot_count("Ctrl ２，６７６") == 2676


def test_mesos_feed_counts_new_lines_once_while_they_remain_visible():
    tracker = MesosFeedTracker()

    assert tracker.update([MesosObservation(144, 100)]) == 0  # establish the visible feed baseline
    assert tracker.update([MesosObservation(144, 100)]) == 0
    assert tracker.update([MesosObservation(144, 100), MesosObservation(300, 120)]) == 300
    assert tracker.total == 300
    assert tracker.events == 1
    # Empty feed is the boundary that allows a later identical pickup to count.
    tracker.update([])
    assert tracker.update([MesosObservation(144, 100)]) == 144


def test_shortcut_drop_is_the_only_source_of_cost_and_classifies_recovery():
    tracker = EconomyTracker(
        [PotionSlotConfig(slot="F1", name="Red Potion", kind="hp", cost=25, recovery=50)]
    )
    tracker.record_quick_slot_counts({"F1": 10}, now=0)
    tracker.record_quick_slot_counts({"F1": 9}, now=1)
    assert tracker.snapshot.potion_uses == 0  # one low OCR frame is held
    tracker.record_quick_slot_counts({"F1": 9}, now=1.75)
    assert tracker.snapshot.potion_uses == 1
    assert tracker.snapshot.potion_cost == 25
    tracker.record_stats(100, 200, now=2)
    tracker.record_stats(150, 200, now=2.2)

    snapshot = tracker.snapshot
    assert snapshot.potion_uses == 1
    assert snapshot.potion_cost == 25
    assert snapshot.hp_recovery_potion == 50
    assert snapshot.hp_recovery_natural == 0
    assert snapshot.hp_recovery_savings == 0


def test_bar_flash_can_confirm_one_frame_drop_but_never_creates_cost_alone():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="6", name="HP Potion", kind="hp", cost=25),
    ])
    tracker.record_quick_slot_counts({"6": 10}, now=0)
    tracker.record_bar_flash(("hp",), now=1)
    assert tracker.snapshot.potion_uses == 0

    # The quantity drop is visible only once.  A matching conservative bar
    # flash is allowed to confirm exactly one bottle.
    tracker.record_quick_slot_counts({"6": 9}, now=1.0)

    snapshot = tracker.snapshot
    assert snapshot.hp_potion_uses == 1
    assert snapshot.hp_potion_cost == 25


def test_monotonic_multi_potion_drop_counts_without_repeated_quantity_frame():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="F1", name="Red Potion", kind="hp", cost=25),
    ])
    tracker.record_quick_slot_counts({"F1": 1180}, now=0)
    # Without an intermediate trusted frame, the aggregate 1180 -> 1178
    # transition is ambiguous and is rejected instead of charging two uses.
    tracker.record_quick_slot_counts({"F1": 1179}, now=1)
    tracker.record_quick_slot_counts({"F1": 1178}, now=1.75)

    assert tracker.snapshot.potion_uses == 0
    assert tracker.snapshot.potion_cost == 0


def test_hp_and_mp_slots_are_classified_independently():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="1", name="Red Potion", kind="hp", cost=25),
        PotionSlotConfig(slot="2", name="Blue Potion", kind="mp", cost=40),
    ])
    tracker.record_quick_slot_counts({"1": 1180, "2": 2037}, now=0)
    tracker.record_quick_slot_counts({"1": 1179, "2": 2036}, now=1)
    tracker.record_quick_slot_counts({"1": 1178, "2": 2035}, now=1.75)

    snapshot = tracker.snapshot
    assert snapshot.hp_potion_uses == 0
    assert snapshot.hp_potion_cost == 0
    assert snapshot.mp_potion_uses == 0
    assert snapshot.mp_potion_cost == 0


def test_drop_candidate_survives_slow_enhanced_ocr_retry():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="2", name="White Potion", kind="hp", cost=25),
    ])
    tracker.record_quick_slot_counts({"2": 100}, now=0)
    tracker.record_quick_slot_counts({"2": 99}, now=1)
    tracker.record_quick_slot_counts({"2": 99}, now=4.5)

    assert tracker.snapshot.hp_potion_uses == 1


def test_matching_hp_mp_recovery_confirms_a_one_frame_quantity_drop():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="2", name="White Potion", kind="hp", cost=25, recovery=300),
        PotionSlotConfig(slot="3", name="Blue Potion", kind="mp", cost=40, recovery=300),
    ])
    tracker.record_quick_slot_counts({"2": 100, "3": 200}, now=0)
    tracker.record_stats(500, 800, now=0)
    tracker.record_quick_slot_counts({"2": 99, "3": 199}, now=1)
    # The next shortcut frame is missed, but both heals identify the two
    # configured one-frame quantity drops.
    tracker.record_stats(800, 1100, now=1.5)

    snapshot = tracker.snapshot
    assert snapshot.hp_potion_uses == 1
    assert snapshot.mp_potion_uses == 1
    assert snapshot.hp_recovery_potion == 300
    assert snapshot.mp_recovery_potion == 300


def test_recovery_without_slot_drop_never_creates_potion_cost_and_estimates_savings():
    tracker = EconomyTracker(
        [
            PotionSlotConfig(slot="F1", name="Red Potion", kind="hp", cost=25),
            PotionSlotConfig(slot="F2", name="Blue Potion", kind="mp", cost=100),
        ],
        default_recovery_hp=50,
        default_recovery_mp=30,
    )
    tracker.record_stats(100, 200, now=0)
    tracker.record_stats(150, 230, now=0.3)
    tracker.record_stats(160, 240, now=0.6)

    snapshot = tracker.snapshot
    assert snapshot.potion_uses == 0
    assert snapshot.potion_cost == 0
    assert snapshot.hp_potion_uses == 0
    assert snapshot.mp_potion_uses == 0
    assert snapshot.hp_recovery_potion == 0
    assert snapshot.hp_recovery_natural == 60
    assert snapshot.hp_recovery_savings == 72.0
    assert snapshot.mp_recovery_potion == 0
    assert snapshot.mp_recovery_natural == 40
    assert snapshot.mp_recovery_savings == 84.0


def test_quantity_increase_only_resets_baseline_and_a_later_drop_counts():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="F1", name="Red Potion", kind="hp", cost=25, recovery=50),
    ], allow_in_session_restock=True)

    tracker.record_quick_slot_counts({"F1": 10}, now=0)
    # A restock is accepted only after three identical frames so a joined
    # neighbour cell cannot redefine the inventory baseline.
    tracker.record_quick_slot_counts({"F1": 12}, now=1)
    tracker.record_quick_slot_counts({"F1": 12}, now=1.5)
    tracker.record_quick_slot_counts({"F1": 12}, now=1.75)  # restock
    assert tracker.snapshot.potion_uses == 0
    tracker.record_quick_slot_counts({"F1": 11}, now=2)
    assert tracker.snapshot.potion_uses == 0
    tracker.record_quick_slot_counts({"F1": 11}, now=2.75)

    assert tracker.snapshot.potion_uses == 1
    assert tracker.snapshot.potion_cost == 25


def test_recovery_after_confirmed_drop_is_potion_even_when_amount_is_capped():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="F1", name="Red Potion", kind="hp", cost=25, recovery=50),
    ])
    tracker.record_quick_slot_counts({"F1": 10}, now=0)
    tracker.record_quick_slot_counts({"F1": 9}, now=1)
    tracker.record_quick_slot_counts({"F1": 9}, now=1.75)
    tracker.record_stats(100, 200, now=2)
    tracker.record_stats(130, 200, now=2.2)  # capped/partial heal, not exactly 50

    snapshot = tracker.snapshot
    assert snapshot.potion_uses == 1
    assert snapshot.potion_cost == 25
    # A partial heal is no longer attributed to a potion: without an exact
    # configured amount it is indistinguishable from a skill/natural tick.
    assert snapshot.hp_recovery_potion == 0
    assert snapshot.hp_recovery_natural == 30
    assert snapshot.hp_recovery_savings == 36.0


def test_one_frame_lower_ocr_is_not_a_potion_use():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="F1", name="Red Potion", kind="hp", cost=25),
    ])
    tracker.record_quick_slot_counts({"F1": 1180}, now=0)
    tracker.record_quick_slot_counts({"F1": 180}, now=1)
    tracker.record_quick_slot_counts({"F1": 1180}, now=1.75)

    assert tracker.snapshot.potion_uses == 0
    assert tracker.snapshot.potion_cost == 0


def test_large_shortcut_jump_is_ignored_instead_of_counting_hundreds_of_potions():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="F1", name="Red Potion", kind="hp", cost=25),
    ])
    tracker.record_quick_slot_counts({"F1": 1180}, now=0)
    tracker.record_quick_slot_counts({"F1": 180}, now=1)
    tracker.record_quick_slot_counts({"F1": 180}, now=1.75)
    tracker.record_quick_slot_counts({"F1": 1179}, now=2.5)

    assert tracker.snapshot.potion_uses == 0
    assert tracker.snapshot.potion_cost == 0


def test_blank_potion_settings_still_count_unknown_slot_as_shared_use():
    tracker = EconomyTracker([])

    tracker.record_quick_slot_counts({"1": 20}, now=0)
    tracker.record_quick_slot_counts({"1": 19}, now=1)
    tracker.record_quick_slot_counts({"1": 19}, now=1.75)

    snapshot = tracker.snapshot
    assert snapshot.shared_potion_uses == 1
    assert snapshot.shared_potion_cost == 0


def test_start_baseline_does_not_charge_current_hp_or_mp_inventory():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="1", kind="hp", cost=320),
        PotionSlotConfig(slot="2", kind="mp", cost=180),
    ])
    tracker.prime_quick_slot_counts({"1": 18, "2": 24})
    tracker.record_quick_slot_counts({"1": 18, "2": 24}, now=0)

    assert tracker.snapshot.potion_uses == 0
    assert tracker.snapshot.hp_potion_uses == 0
    assert tracker.snapshot.mp_potion_uses == 0


def test_snapshot_exposes_initial_and_current_shortcut_quantities():
    tracker = EconomyTracker([])
    tracker.prime_quick_slot_counts({"1": 1180, "2": 2037})
    tracker.record_quick_slot_counts({"1": 1179, "2": 2037}, now=1)

    # One lower OCR frame is only a candidate; the live quantity remains
    # stable until the same decrease is observed again.
    assert tracker.snapshot.shortcut_current == {"1": 1180, "2": 2037}
    tracker.record_quick_slot_counts({"1": 1179, "2": 2037}, now=1.75)

    snapshot = tracker.snapshot
    assert snapshot.shortcut_baseline_ready is True
    assert snapshot.shortcut_baseline == {"1": 1180, "2": 2037}
    assert snapshot.shortcut_current == {"1": 1179, "2": 2037}


def test_truncated_shortcut_ocr_never_jumps_inventory_to_a_suffix():
    tracker = EconomyTracker([PotionSlotConfig(slot="7", kind="mp", cost=604)])
    tracker.prime_quick_slot_counts({"7": 116})

    # 116 -> 6 is a missing-leading-digit OCR result, not 110 potion uses.
    tracker.record_quick_slot_counts({"7": 6}, now=1)
    assert tracker.snapshot.shortcut_current == {"7": 116}
    assert tracker.snapshot.mp_potion_uses == 0

    # A real one-item decrease must still be accepted after confirmation.
    tracker.record_quick_slot_counts({"7": 115}, now=1.5)
    tracker.record_quick_slot_counts({"7": 115}, now=2)
    assert tracker.snapshot.shortcut_current == {"7": 115}
    assert tracker.snapshot.mp_potion_uses == 1


def test_one_frame_neighbour_cell_merge_never_publishes_as_restock():
    tracker = EconomyTracker([PotionSlotConfig(slot="7", kind="mp", cost=604)])
    tracker.prime_quick_slot_counts({"7": 89})

    # The neighbouring cell can be joined to the MP count for one frame.
    # That is not a restock and must not leak into the visible inventory.
    tracker.record_quick_slot_counts({"7": 895}, now=1)
    assert tracker.snapshot.shortcut_current == {"7": 89}
    tracker.record_quick_slot_counts({"7": 89}, now=1.5)
    assert tracker.snapshot.shortcut_current == {"7": 89}
    assert tracker.snapshot.mp_potion_uses == 0


def test_restock_requires_three_identical_frames():
    tracker = EconomyTracker(
        [PotionSlotConfig(slot="7", kind="mp", cost=604)],
        allow_in_session_restock=True,
    )
    tracker.prime_quick_slot_counts({"7": 89})

    tracker.record_quick_slot_counts({"7": 94}, now=1)
    tracker.record_quick_slot_counts({"7": 94}, now=1.5)
    assert tracker.snapshot.shortcut_current == {"7": 89}
    tracker.record_quick_slot_counts({"7": 94}, now=2)
    assert tracker.snapshot.shortcut_current == {"7": 94}
    assert tracker.snapshot.mp_potion_uses == 0


def test_final_reconciliation_commits_missed_quantity_drop_once():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="2", name="White Potion", kind="hp", cost=320),
        PotionSlotConfig(slot="3", name="Blue Potion", kind="mp", cost=604),
    ])
    tracker.prime_quick_slot_counts({"2": 1578, "3": 229}, now=0)
    # OCR only confirmed part of the real HP stack decrease during the live
    # session; the final visible inventory is authoritative at the boundary.
    tracker.record_quick_slot_counts({"2": 1577, "3": 228}, now=1)
    tracker.record_quick_slot_counts({"2": 1577, "3": 228}, now=1.75)
    assert tracker.snapshot.hp_potion_uses == 1
    assert tracker.snapshot.mp_potion_uses == 1

    uses = tracker.reconcile_quick_slot_counts({"2": 1545, "3": 229}, now=20)

    assert uses == 32
    assert tracker.snapshot.hp_potion_uses == 33
    assert tracker.snapshot.mp_potion_uses == 1
    assert tracker.snapshot.hp_potion_cost == 10_560


def test_fast_bulk_shortcut_drop_is_rejected_by_realistic_drink_rate():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="7", name="Blue Potion", kind="mp", cost=604),
    ])
    tracker.prime_quick_slot_counts({"7": 1200}, now=0)

    # 80 drinks in half a second cannot happen in the game and is an OCR
    # suffix/crop error, not a real inventory event.
    tracker.record_quick_slot_counts({"7": 1120}, now=0.5)
    tracker.record_quick_slot_counts({"7": 1120}, now=0.75)

    assert tracker.snapshot.shortcut_current == {"7": 1200}
    assert tracker.snapshot.mp_potion_uses == 0
    assert tracker.snapshot.mp_potion_cost == 0


def test_live_session_ignores_shortcut_restock_without_refill_action():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="7", name="Blue Potion", kind="mp", cost=604),
    ])
    tracker.prime_quick_slot_counts({"7": 86}, now=0)

    tracker.record_quick_slot_counts({"7": 91}, now=1)
    tracker.record_quick_slot_counts({"7": 91}, now=1.5)
    tracker.record_quick_slot_counts({"7": 91}, now=2)

    assert tracker.snapshot.shortcut_current == {"7": 86}
    assert tracker.snapshot.mp_potion_uses == 0


def test_unconfigured_recovery_is_not_attributed_to_a_pending_potion():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="F1", name="Red Potion", kind="hp", cost=25),
    ])
    tracker.prime_quick_slot_counts({"F1": 10}, now=0)
    tracker.record_quick_slot_counts({"F1": 9}, now=1)
    tracker.record_quick_slot_counts({"F1": 9}, now=1.75)
    tracker.record_stats(100, 200, now=2)
    tracker.record_stats(160, 200, now=2.2)

    snapshot = tracker.snapshot
    assert snapshot.hp_potion_uses == 1
    assert snapshot.hp_recovery_potion == 0
    assert snapshot.hp_recovery_natural == 60


@pytest.mark.parametrize("kind", ["hp", "mp", "both"])
def test_two_from_eighty_two_is_rejected_for_every_potion_kind(kind):
    tracker = EconomyTracker([
        PotionSlotConfig(slot="7", name="Potion", kind=kind, cost=100),
    ])
    tracker.prime_quick_slot_counts({"7": 82}, now=0)

    tracker.record_quick_slot_counts({"7": 2}, now=1)
    tracker.record_quick_slot_counts({"7": 2}, now=1.2)
    uses = tracker.reconcile_quick_slot_counts({"7": 2}, now=1.4)

    assert uses == 0
    assert tracker.snapshot.shortcut_current == {"7": 82}
    assert tracker.snapshot.potion_cost == 0
