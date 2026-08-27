import pytest

from maple_analyzer.economy import (
    EconomyTracker,
    MesosFeedTracker,
    MesosObservation,
    mesos_text_needs_full_detection,
    parse_mesos_amount,
    parse_slot_count,
)
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


def test_weak_mesos_fallback_is_confirmed_by_full_feed_detection():
    assert mesos_text_needs_full_detection("獲取。(+144)") is True
    assert mesos_text_needs_full_detection("獲取楓幣。(+144)") is False
    assert mesos_text_needs_full_detection("高級楓之谷通行證Bonus經驗值。(+5)") is True


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


def test_fast_auxiliary_samples_confirm_a_real_quantity_change_after_rate_window():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="1", name="Red Potion", kind="hp", cost=25),
    ])
    tracker.prime_quick_slot_counts({"1": 1000}, now=0)

    # The overlay can sample the same shortcut every 0.3s. The first lower
    # frame is a candidate; the repeated value at 0.6s is the confirmation.
    tracker.record_quick_slot_counts({"1": 999}, now=0.30)
    assert tracker.snapshot.hp_potion_uses == 0
    tracker.record_quick_slot_counts({"1": 999}, now=0.60)

    assert tracker.snapshot.hp_potion_uses == 1
    assert tracker.snapshot.hp_potion_cost == 25


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
    # A fast drink animation can expose a monotonic sequence without repeating
    # one quantity.  Charge the net transition once the second lower frame
    # corroborates it.
    tracker.record_quick_slot_counts({"F1": 1179}, now=1)
    tracker.record_quick_slot_counts({"F1": 1178}, now=1.75)

    assert tracker.snapshot.potion_uses == 2
    assert tracker.snapshot.potion_cost == 50


def test_hp_and_mp_slots_are_classified_independently():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="1", name="Red Potion", kind="hp", cost=25),
        PotionSlotConfig(slot="2", name="Blue Potion", kind="mp", cost=40),
    ])
    tracker.record_quick_slot_counts({"1": 1180, "2": 2037}, now=0)
    tracker.record_quick_slot_counts({"1": 1179, "2": 2036}, now=1)
    tracker.record_quick_slot_counts({"1": 1178, "2": 2035}, now=1.75)

    snapshot = tracker.snapshot
    assert snapshot.hp_potion_uses == 2
    assert snapshot.hp_potion_cost == 50
    assert snapshot.mp_potion_uses == 2
    assert snapshot.mp_potion_cost == 80


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


def test_one_frame_status_ocr_outlier_is_not_counted_as_natural_recovery():
    tracker = EconomyTracker([])

    tracker.record_stats(1000, 800, now=0, hp_max=2210, mp_max=1407)
    # A lost leading digit is a common OCR failure. The next correct frame
    # must return to the trusted baseline without creating fake savings.
    tracker.record_stats(100, 80, now=0.3, hp_max=2210, mp_max=1407)
    tracker.record_stats(1000, 800, now=0.6, hp_max=2210, mp_max=1407)

    snapshot = tracker.snapshot
    assert snapshot.hp_recovery_natural == 0
    assert snapshot.mp_recovery_natural == 0
    assert snapshot.hp_recovery_savings == 0
    assert snapshot.mp_recovery_savings == 0


def test_large_natural_recovery_requires_a_second_consistent_frame():
    tracker = EconomyTracker([])

    tracker.record_stats(200, 100, now=0, hp_max=2210, mp_max=1407)
    assert tracker.record_stats(1100, 700, now=0.3, hp_max=2210, mp_max=1407) == (0, 0)
    assert tracker.snapshot.hp_recovery_natural == 0
    assert tracker.snapshot.mp_recovery_natural == 0

    assert tracker.record_stats(1100, 700, now=0.6, hp_max=2210, mp_max=1407) == (900, 600)
    snapshot = tracker.snapshot
    assert snapshot.hp_recovery_natural == 900
    assert snapshot.mp_recovery_natural == 600
    assert snapshot.hp_recovery_savings == 1080.0
    assert snapshot.mp_recovery_savings == 1260.0


def test_impossible_status_value_does_not_poison_recovery_baseline():
    tracker = EconomyTracker([])

    tracker.record_stats(1000, 700, now=0, hp_max=2210, mp_max=1407)
    tracker.record_stats(9999, 9999, now=0.3, hp_max=2210, mp_max=1407)
    assert tracker.snapshot.hp_recovery_natural == 0
    assert tracker.snapshot.mp_recovery_natural == 0

    tracker.record_stats(1010, 710, now=0.6, hp_max=2210, mp_max=1407)
    assert tracker.snapshot.hp_recovery_natural == 10
    assert tracker.snapshot.mp_recovery_natural == 10


def test_rebound_after_one_frame_low_ocr_outlier_is_not_natural_recovery():
    tracker = EconomyTracker([])

    tracker.record_stats(1000, 700, now=0, hp_max=2210, mp_max=1407)
    tracker.record_stats(100, 70, now=0.3, hp_max=2210, mp_max=1407)
    # A slightly damaged but otherwise valid next frame is still close to the
    # trusted baseline; it corroborates that 100/70 was a bad OCR frame.
    tracker.record_stats(900, 650, now=0.6, hp_max=2210, mp_max=1407)
    tracker.record_stats(1000, 700, now=0.9, hp_max=2210, mp_max=1407)

    snapshot = tracker.snapshot
    assert snapshot.hp_recovery_natural == 0
    assert snapshot.mp_recovery_natural == 0
    assert snapshot.hp_recovery_savings == 0
    assert snapshot.mp_recovery_savings == 0


def test_rebound_after_one_frame_high_ocr_outlier_is_not_natural_recovery():
    tracker = EconomyTracker([])

    tracker.record_stats(1000, 700, now=0, hp_max=2210, mp_max=1407)
    tracker.record_stats(2000, 1300, now=0.3, hp_max=2210, mp_max=1407)
    tracker.record_stats(1100, 750, now=0.6, hp_max=2210, mp_max=1407)
    tracker.record_stats(1000, 700, now=0.9, hp_max=2210, mp_max=1407)

    snapshot = tracker.snapshot
    assert snapshot.hp_recovery_natural == 0
    assert snapshot.mp_recovery_natural == 0
    assert snapshot.hp_recovery_savings == 0
    assert snapshot.mp_recovery_savings == 0


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


def test_large_shortcut_jump_is_provisional_and_reconciles_to_later_quantity():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="F1", name="Red Potion", kind="hp", cost=25),
    ])
    tracker.record_quick_slot_counts({"F1": 1180}, now=0)
    # A missed OCR interval can expose a large real-looking drop.  It is
    # allowed into the provisional ledger instead of being discarded by an
    # arbitrary four-bottle cap.
    tracker.record_quick_slot_counts({"F1": 180}, now=1)
    tracker.record_quick_slot_counts({"F1": 180}, now=1.75)
    assert tracker.snapshot.hp_potion_uses == 1000

    # When the correct quantity returns, the provisional overcharge is
    # reversed and the final cost follows the latest stable inventory.
    tracker.record_quick_slot_counts({"F1": 1180}, now=2.5)
    tracker.record_quick_slot_counts({"F1": 1180}, now=3.0)

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
    assert tracker.snapshot.shortcut_observed == {"1": 1179, "2": 2037}
    tracker.record_quick_slot_counts({"1": 1179, "2": 2037}, now=1.75)

    snapshot = tracker.snapshot
    assert snapshot.shortcut_baseline_ready is True
    assert snapshot.shortcut_baseline == {"1": 1180, "2": 2037}
    assert snapshot.shortcut_current == {"1": 1179, "2": 2037}
    assert snapshot.shortcut_observed == {"1": 1179, "2": 2037}


def test_large_shortcut_drop_is_reversible_when_a_later_frame_restores_quantity():
    tracker = EconomyTracker([PotionSlotConfig(slot="7", kind="mp", cost=604)])
    tracker.prime_quick_slot_counts({"7": 116})

    # The lower value is allowed to become a provisional charge.  If it was
    # really a missing-leading-digit OCR result, the later stable quantity
    # restores the ledger instead of leaving a permanent false cost.
    tracker.record_quick_slot_counts({"7": 6}, now=1)
    assert tracker.snapshot.shortcut_current == {"7": 116}
    assert tracker.snapshot.mp_potion_uses == 0
    tracker.record_quick_slot_counts({"7": 6}, now=1.5)
    assert tracker.snapshot.shortcut_current == {"7": 6}
    assert tracker.snapshot.mp_potion_uses == 110

    tracker.record_quick_slot_counts({"7": 116}, now=2)
    tracker.record_quick_slot_counts({"7": 116}, now=2.5)
    assert tracker.snapshot.shortcut_current == {"7": 116}
    assert tracker.snapshot.mp_potion_uses == 0


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


def test_large_shortcut_drop_is_billed_then_can_be_reversed():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="7", name="Blue Potion", kind="mp", cost=604),
    ])
    tracker.prime_quick_slot_counts({"7": 1200}, now=0)

    # Do not permanently discard a large lower reading.  It is provisional and
    # can be corrected when a later frame returns toward the baseline.
    tracker.record_quick_slot_counts({"7": 1120}, now=0.5)
    tracker.record_quick_slot_counts({"7": 1120}, now=0.75)

    assert tracker.snapshot.shortcut_current == {"7": 1120}
    assert tracker.snapshot.mp_potion_uses == 80
    assert tracker.snapshot.mp_potion_cost == 80 * 604

    tracker.record_quick_slot_counts({"7": 1200}, now=1.0)
    tracker.record_quick_slot_counts({"7": 1200}, now=1.5)

    assert tracker.snapshot.shortcut_current == {"7": 1200}
    assert tracker.snapshot.mp_potion_uses == 0
    assert tracker.snapshot.mp_potion_cost == 0


def test_bulk_shortcut_drop_is_accepted_after_stable_frames():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="7", name="Blue Potion", kind="mp", cost=604),
    ])
    tracker.prime_quick_slot_counts({"7": 1850}, now=0)

    # A missed interval can expose five consumed bottles at once. Two
    # identical frames accept the complete drop instead of applying the old
    # hard four-bottle cap.
    tracker.record_quick_slot_counts({"7": 1845}, now=0.3)
    tracker.record_quick_slot_counts({"7": 1845}, now=0.6)

    assert tracker.snapshot.shortcut_current == {"7": 1845}
    assert tracker.snapshot.mp_potion_uses == 5
    assert tracker.snapshot.mp_potion_cost == 5 * 604


def test_provisional_overcharge_converges_to_later_correct_quantity():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="7", name="Blue Potion", kind="mp", cost=604),
    ])
    tracker.prime_quick_slot_counts({"7": 100}, now=0)

    # OCR temporarily reads 40, then later recovers the actual 80. The ledger
    # must settle at the baseline-to-corrected difference: 20 bottles.
    tracker.record_quick_slot_counts({"7": 40}, now=1)
    tracker.record_quick_slot_counts({"7": 40}, now=1.5)
    assert tracker.snapshot.mp_potion_uses == 60

    tracker.record_quick_slot_counts({"7": 80}, now=2)
    tracker.record_quick_slot_counts({"7": 80}, now=2.5)

    assert tracker.snapshot.shortcut_current == {"7": 80}
    assert tracker.snapshot.mp_potion_uses == 20
    assert tracker.snapshot.mp_potion_cost == 20 * 604


def test_correcting_a_potion_drop_also_reverses_its_recovery_attribution():
    tracker = EconomyTracker([
        PotionSlotConfig(
            slot="F1", name="Red Potion", kind="hp", cost=25, recovery=50
        ),
    ])
    tracker.prime_quick_slot_counts({"F1": 10}, now=0)
    tracker.record_quick_slot_counts({"F1": 9}, now=1)
    tracker.record_quick_slot_counts({"F1": 9}, now=1.5)
    tracker.record_stats(100, 200, now=2)
    tracker.record_stats(150, 200, now=2.2)

    assert tracker.snapshot.hp_recovery_potion == 50
    assert tracker.snapshot.hp_recovery_natural == 0

    # The entire drink was an OCR false positive.  Its cost and the recovery
    # classification must be rolled back together.
    tracker.record_quick_slot_counts({"F1": 10}, now=3)
    tracker.record_quick_slot_counts({"F1": 10}, now=3.5)

    snapshot = tracker.snapshot
    assert snapshot.potion_uses == 0
    assert snapshot.potion_cost == 0
    assert snapshot.hp_recovery_potion == 0
    assert snapshot.hp_recovery_natural == 50
    assert snapshot.hp_recovery_savings == 60.0


def test_substituted_two_digit_quantity_is_reversed_by_a_stable_correction():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="7", name="Blue Potion", kind="mp", cost=604),
    ])
    tracker.prime_quick_slot_counts({"7": 1487}, now=0)

    # 1487 -> 1467 is a plausible four-digit OCR substitution.  It may be
    # charged provisionally, but a later stable 1487 must reverse the 20-bottle
    # difference.
    tracker.record_quick_slot_counts({"7": 1467}, now=10)
    tracker.record_quick_slot_counts({"7": 1467}, now=10.5)

    assert tracker.snapshot.shortcut_current == {"7": 1467}
    assert tracker.snapshot.mp_potion_uses == 20
    assert tracker.snapshot.mp_potion_cost == 20 * 604

    tracker.record_quick_slot_counts({"7": 1487}, now=11)
    tracker.record_quick_slot_counts({"7": 1487}, now=11.5)

    assert tracker.snapshot.shortcut_current == {"7": 1487}
    assert tracker.snapshot.mp_potion_uses == 0
    assert tracker.snapshot.mp_potion_cost == 0


def test_provisional_multi_item_drop_can_be_reversed_by_stable_quantity_return():
    tracker = EconomyTracker([
        PotionSlotConfig(slot="1", name="Red Potion", kind="hp", cost=25),
    ])
    tracker.prime_quick_slot_counts({"1": 1000}, now=0)

    # A small multi-item OCR jump is allowed only after the normal gates, but
    # remains marked as suspicious until the later stable frame confirms it.
    tracker.record_quick_slot_counts({"1": 996}, now=10)
    tracker.record_quick_slot_counts({"1": 996}, now=10.5)
    assert tracker.snapshot.hp_potion_uses == 4
    assert tracker.snapshot.hp_potion_cost == 100

    # The quantity returning to the session baseline twice proves the earlier
    # multi-item value was a substituted digit. Reverse both use and cost.
    tracker.record_quick_slot_counts({"1": 1000}, now=11)
    tracker.record_quick_slot_counts({"1": 1000}, now=11.5)
    assert tracker.snapshot.shortcut_current == {"1": 1000}
    assert tracker.snapshot.hp_potion_uses == 0
    assert tracker.snapshot.hp_potion_cost == 0


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
def test_large_drop_and_restore_converge_for_every_potion_kind(kind):
    tracker = EconomyTracker([
        PotionSlotConfig(slot="7", name="Potion", kind=kind, cost=100),
    ])
    tracker.prime_quick_slot_counts({"7": 82}, now=0)

    tracker.record_quick_slot_counts({"7": 2}, now=1)
    tracker.record_quick_slot_counts({"7": 2}, now=1.2)
    assert tracker.snapshot.shortcut_current == {"7": 2}
    assert tracker.snapshot.potion_uses == 80

    tracker.record_quick_slot_counts({"7": 82}, now=1.5)
    uses = tracker.record_quick_slot_counts({"7": 82}, now=2.0)

    assert uses == 0
    assert tracker.snapshot.shortcut_current == {"7": 82}
    assert tracker.snapshot.potion_cost == 0
