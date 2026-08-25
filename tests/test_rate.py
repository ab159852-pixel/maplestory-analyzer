"""Session/SessionSummary tests -- pure logic, no OCR/images."""
import dataclasses

from maple_analyzer.rate import Session, SessionSummary


def test_start_exp_set_on_first_record():
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200)
    assert s.start_exp == 1000
    assert s.exp_diff == 0


def test_exp_diff_tracks_gain():
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200)
    s.record(exp_cur=1500, hp_cur=500, mp_cur=200)
    assert s.exp_diff == 500


def test_hp_mp_loss_only_accumulates_on_decrease():
    # Normal-sized moves are within _LossTracker.OUTLIER_FRACTION and are
    # taken immediately -- the noise guards add no lag to ordinary play.
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200)
    s.record(exp_cur=1000, hp_cur=400, mp_cur=150)  # lost 100 HP, 50 MP
    s.record(exp_cur=1000, hp_cur=450, mp_cur=180)  # healed -- no loss added
    s.record(exp_cur=1000, hp_cur=300, mp_cur=180)  # lost another 150 HP
    assert s.hp_loss == 250
    assert s.mp_loss == 50


def test_sustained_loss_is_counted_in_full():
    s = Session(require_calibration=False)
    for hp in (824, 700, 600, 500, 500):
        s.record(exp_cur=1000, hp_cur=hp, mp_cur=200, hp_max=824)
    assert s.hp_loss == 324  # 824 -> 500, all of it


def test_large_real_drop_lands_one_tick_late():
    """A one-shot big enough to look like a misread is held for one tick, then
    committed in full once the next reading corroborates it."""
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=824, mp_cur=200, hp_max=824)
    s.record(exp_cur=1000, hp_cur=90, mp_cur=200, hp_max=824)  # held
    assert s.hp_loss == 0
    s.record(exp_cur=1000, hp_cur=90, mp_cur=200, hp_max=824)  # corroborated
    assert s.hp_loss == 734


def test_recovery_evidence_recovers_damage_hidden_by_outlier_guard():
    """A large damage frame can be held while the next potion heal is valid.
    The economy worker supplies that upward delta so HP loss is not reduced to
    the start/end difference."""
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=2210, mp_cur=200, hp_max=2210)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200, hp_max=2210)  # held
    s.record(exp_cur=1000, hp_cur=2210, mp_cur=200, hp_max=2210)  # held frame discarded
    s.add_recovery_evidence("hp", 1710)

    assert s.hp_loss == 1710


def test_alternating_misreads_book_nothing():
    """The regime that broke a median-of-3 filter: every other tick corrupt.
    Real MP here only regenerates, so the truth is zero loss."""
    s = Session(require_calibration=False)
    for mp in [1663, 3, 1663, 16, 1663, 166, 1663] * 20:
        s.record(exp_cur=1000, hp_cur=500, mp_cur=mp, mp_max=2816)
    assert s.mp_loss == 0


def test_phantom_high_read_is_bounded_not_unbounded():
    """A misread that reads *high* costs nothing that tick, then books a
    phantom loss when the next correct read "drops" back. The expensive
    version of this ('1663/2816' -> '16632/816') is rejected outright by the
    max check. What remains is a high read that is still <= max and inside the
    tolerance band -- indistinguishable from drinking a potion, so it is
    accepted. Pinned here to record that the damage is bounded by the size of
    the bar rather than accumulating without limit as it used to."""
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1663, mp_max=2816)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=2700, mp_max=2816)  # phantom high
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1663, mp_max=2816)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1663, mp_max=2816)
    assert 0 < s.mp_loss <= 2816


def test_single_tick_misread_books_no_loss():
    """The reported bug: an idle character accumulating huge MP 'loss'.
    '1663 -> 3 -> 1663' is a misread, not a drop."""
    s = Session(require_calibration=False)
    for mp in (1663, 3, 1663, 16, 1663, 166, 1663):
        s.record(exp_cur=1000, hp_cur=500, mp_cur=mp, mp_max=2816)
    assert s.mp_loss == 0


def test_misparsed_max_rejects_the_whole_tick():
    """'1663/2816' misread as '16632/816' reads high, then books a huge
    phantom loss when the next correct read 'drops' back. The max mismatch
    (816 != 2816) is the tell, so the tick never reaches the loss math."""
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1663, mp_max=2816)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=16632, mp_max=816)  # rejected
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1663, mp_max=2816)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1663, mp_max=2816)
    assert s.mp_loss == 0


def test_real_max_change_is_accepted_once_corroborated():
    """A level-up genuinely raises max -- the guard must not wedge shut.
    Accompanied by a level bump, 2 corroborating ticks suffice (see
    _LossTracker._accept_max's level-aware corroboration); with no level
    bump at all it takes 3 (test_level_change_without_a_level_bump_needs_a_third_tick)."""
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1663, mp_max=2816, level=44)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1700, mp_max=3000, level=45)  # held
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1700, mp_max=3000, level=45)  # corroborated -- level bump halves the wait
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1600, mp_max=3000, level=45)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1500, mp_max=3000, level=45)
    assert s.mp_loss == 200  # tracking normally again at the new max


def test_max_change_without_a_level_bump_needs_a_third_tick():
    """The same change, but with no level info at all -- static OCR garbage
    from a covered panel repeats an identical wrong max every tick with no
    level movement either, so an unaccompanied max change is held one tick
    longer before it's trusted."""
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1663, mp_max=2816)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1700, mp_max=3000)  # candidate, 1/3
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1700, mp_max=3000)  # 2/3 -- not yet enough
    assert s.mp_loss == 0
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1600, mp_max=3000)  # 3/3 -- corroborated, and this tick's own drop counts
    s.record(exp_cur=1000, hp_cur=500, mp_cur=1500, mp_max=3000)
    assert s.mp_loss == 163  # 1663 -> 1600 (63) -> 1500 (100); the held 1700 reading never counted


def test_missing_reading_does_not_corrupt_loss_tracking():
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200)
    s.record(exp_cur=None, hp_cur=None, mp_cur=None)  # a tick that missed everything
    s.record(exp_cur=1000, hp_cur=450, mp_cur=200)
    assert s.hp_loss == 50


def test_finalize_produces_correct_summary():
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200, exp_pct=10.0)
    s.record(exp_cur=1200, hp_cur=400, mp_cur=200)
    summary = s.finalize(interval_minutes=5, now=s._start_time + 60)
    assert summary.start_exp == 1000
    assert summary.end_exp == 1200
    assert summary.exp_diff == 200
    assert summary.hp_loss == 100
    assert summary.mp_loss == 0
    assert summary.duration_s == 60
    assert summary.interval_minutes == 5
    assert summary.name is None
    assert summary.total_exp == 10000  # 1000 / (10.0/100)
    assert summary.exp_pct_diff == 2.0  # 200/10000 * 100


def test_exp_per_hour_normalizes_current_and_finalized_sessions():
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200)
    start = s._start_time
    s.record(exp_cur=1500, hp_cur=500, mp_cur=200)
    summary = s.finalize(now=start + 60)

    assert summary.exp_per_hour == 30_000
    # The live property uses the real elapsed clock; it only needs to be
    # present and positive after a positive gain, not to depend on sleeping.
    assert s.exp_per_hour is not None
    assert s.exp_per_hour > 0


def test_zero_gain_live_session_reports_zero_instead_of_missing_data():
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200)
    s._start_time -= 120
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200)

    assert s.exp_diff == 0
    assert s.exp_per_hour == 0
    assert s.projected_exp(600) == 0


def test_exp_per_hour_is_none_for_empty_or_zero_duration_summary():
    empty = SessionSummary(
        start_time=1.0, end_time=1.0, start_exp=None, end_exp=None,
        hp_loss=0, mp_loss=0, total_exp=None, interval_minutes=None,
    )
    assert empty.exp_per_hour is None


def test_restart_carries_forward_last_values():
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200)
    s.record(exp_cur=1200, hp_cur=400, mp_cur=180)
    s.start()  # simulates restart -- new session should baseline off last-known values
    assert s.start_exp == 1200
    assert s.hp_loss == 0
    assert s.mp_loss == 0
    s.record(exp_cur=1200, hp_cur=350, mp_cur=180)
    assert s.hp_loss == 50  # loss measured from the carried-forward baseline, not 0


def test_begin_fresh_uses_first_post_start_exp_for_projection():
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200)
    s.record(exp_cur=1800, hp_cur=500, mp_cur=200)

    s.begin_fresh()
    assert s.start_exp is None
    s.record(exp_cur=5000, hp_cur=500, mp_cur=200)
    assert s.start_exp == 5000
    assert s.exp_diff == 0

    s._start_time -= 10
    s.record(exp_cur=5100, hp_cur=500, mp_cur=200)
    assert s.projected_exp(600) == 6000


def test_summary_is_renamable_via_dataclasses_replace():
    # SessionSummary is frozen -- the History tab's rename feature works by
    # replacing the stored summary, not mutating it in place.
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=200)
    summary = s.finalize()
    renamed = dataclasses.replace(summary, name="grinding spot A")
    assert renamed.name == "grinding spot A"
    assert summary.name is None  # original untouched
    assert renamed.start_exp == summary.start_exp  # everything else preserved


def test_implausible_max_is_never_adopted_however_often_it_repeats():
    """When a window covers the panel the OCR garbage is *static* -- the same
    wrong text every tick -- so corroboration alone cannot reject it. Live
    capture (2026-08-17): '2816' misread as '281616' installed a bogus max,
    after which a bogus cur of 28163 passed every check and booked 25,347 of
    phantom loss the moment the panel came back."""
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=2816, mp_max=2816)
    for _ in range(20):  # static garbage, repeated
        s.record(exp_cur=1000, hp_cur=500, mp_cur=28163, mp_max=281616)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=2816, mp_max=2816)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=2816, mp_max=2816)
    assert s.mp_loss == 0


def test_level_up_still_raises_max_within_the_plausible_band():
    s = Session(require_calibration=False)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=2816, mp_max=2816, level=44)
    s.record(exp_cur=1000, hp_cur=500, mp_cur=2900, mp_max=3000, level=45)  # held
    s.record(exp_cur=1000, hp_cur=500, mp_cur=2900, mp_max=3000, level=45)  # corroborated -- level bump halves the wait
    s.record(exp_cur=1000, hp_cur=500, mp_cur=2800, mp_max=3000, level=45)
    assert s.mp_loss == 100
