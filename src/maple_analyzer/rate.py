"""Fixed-epoch session tracking: values accumulate from an explicit start
point until the session is finalized and a new one begins, rather than a
sliding window.

A rolling window (the original design) shrinks in a confusing way once it's
been running longer than the window: e.g. a big EXP gain 4 minutes ago ages
out of a 5-min window even though nothing changed just now, making the
displayed diff decrease with no corresponding in-game event. A session fixes
that -- the start values (EXP, HP, MP) are set once and held constant until
`finalize()` is called, so 'EXP diff' unambiguously means 'since the session
started', full stop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionSummary:
    """A finalized, immutable record of one completed session.

    Every field is a JSON/SQLite-friendly value and there are no references to
    Session, StatSnapshot, or anything OCR/UI-related -- this is deliberately
    the shape persistence stores and the UI reads, decoupled from how it is
    produced or displayed.
    See ~/.claude/notes/maplestory-analyzer/ui-plan-2026-08-17.md for the
    plan to add that persistence layer and a session-history browser on top
    of this struct, without touching the capture/OCR/parser engine.
    """

    start_time: float
    end_time: float
    start_exp: int | None
    end_exp: int | None
    hp_loss: int
    mp_loss: int
    # EXP required for the whole current level, derived once during the
    # session from a tick that had both cur and pct (cur / (pct/100)) -- see
    # Session._total_exp. None if no such tick occurred during the session.
    total_exp: float | None
    # The session-length *setting* in effect when this session ended -- not
    # necessarily equal to duration_s. They differ whenever a session is
    # manually restarted before the timer fires, or (once the settings UI
    # exists) the interval setting is changed between sessions -- recording
    # it here means a saved/displayed session stays self-describing even if
    # the live setting has since changed.
    interval_minutes: float | None
    # EXP actually gained over the session, accumulated tick by tick so it
    # stays correct across level-ups -- end_exp - start_exp is wrong the moment
    # a session spans one, since the game's counter resets to ~0. None only for
    # summaries built before this existed, which fall back to the subtraction.
    exp_gained: int | None = None
    # User-assigned label, e.g. "grinding spot A". None until renamed via the
    # History tab -- UI-layer concern only, the engine never sets this.
    name: str | None = None
    # Low-frequency game context captured by the overlay.  These are optional
    # so old history JSON/CSV rows remain readable and future upload schemas
    # can add more context without changing the EXP engine.
    job_name: str | None = None
    map_name: str | None = None
    # Economy/recovery fields are optional so older history files remain
    # readable and old callers constructing SessionSummary do not change.
    mesos: int = 0
    potion_uses: int = 0
    potion_cost: int = 0
    hp_potion_uses: int = 0
    hp_potion_cost: int = 0
    mp_potion_uses: int = 0
    mp_potion_cost: int = 0
    shared_potion_uses: int = 0
    shared_potion_cost: int = 0
    hp_recovery_natural: int = 0
    hp_recovery_potion: int = 0
    mp_recovery_natural: int = 0
    mp_recovery_potion: int = 0
    # Estimated mesos saved by non-potion recovery.  These are floats because
    # the fixed reference prices are 1.2 mesos/HP and 2.1 mesos/MP.
    hp_recovery_savings: float = 0.0
    mp_recovery_savings: float = 0.0
    potion_breakdown: dict[str, int] | None = None

    @property
    def exp_diff(self) -> int | None:
        if self.exp_gained is not None:
            return self.exp_gained
        if self.start_exp is None or self.end_exp is None:
            return None
        return self.end_exp - self.start_exp

    @property
    def exp_pct_diff(self) -> float | None:
        diff = self.exp_diff
        if diff is None or not self.total_exp:
            return None
        return diff / self.total_exp * 100

    @property
    def exp_per_hour(self) -> float | None:
        """EXP gained normalized to one hour for comparing sessions.

        A finalized session can be very short (for example, a user may hit
        Restart while testing the HUD), so zero-length rows deliberately
        return ``None`` instead of producing an infinite or misleading rate.
        """
        diff = self.exp_diff
        if diff is None or self.duration_s <= 0:
            return None
        return max(0, diff) * 3600 / self.duration_s

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time


class _LossTracker:
    """Accumulates the downward side of one stat's per-tick deltas, with two
    guards against OCR noise.

    Why guards are needed here specifically: HP/MP loss is a *running total*
    that only ever increases, so a single bad reading is baked in permanently
    -- unlike EXP, which is end-minus-start and self-corrects on the next
    tick. Observed live: an idle character (zero real MP spend) accumulated
    142,258 MP of "loss", ~50x the character's max MP, purely from misreads.

    Guard 1 -- max stability. `max` is constant except at level-up, so a tick
    reporting a different one was misparsed and is discarded whole. This
    catches the expensive modes, where a misread '/' shifts a digit across the
    divider: '1663/2816' -> '16632/816' reads *high*, costs nothing that tick,
    then books a 14,969 phantom loss when the next correct read "drops" back.
    A genuinely new max (level-up) is accepted once a second tick corroborates
    it, so this can't wedge the tracker permanently -- or, without a level
    bump alongside it, once a third tick does (see _accept_max: a level-up
    IS the corroboration that a max change is real, so it needs one less tick
    than a max change with no level movement at all).

    Guard 2 -- outlier hold. A reading within OUTLIER_FRACTION of max from the
    last accepted value is normal play and is taken immediately (no lag). A
    reading further out than that has to be corroborated by a second reading
    near it before it becomes the new baseline; otherwise it is held aside and
    forgotten the moment a normal reading arrives.

    Two earlier designs failed here and are worth not repeating:

    - Guarding only *drops* leaks badly (48k of phantom loss in a 3-minute
      idle simulation), because a phantom *high* read poisons the baseline
      instantly and the true values then "drop" back to reality and
      corroborate each other perfectly. The guard must be symmetric.
    - A median-of-3 despike is symmetric but only survives *isolated* spikes.
      Under the alternating good/bad pattern real OCR produces
      (1663, 3, 1663, 16, ...) the median window itself is half garbage and
      the filter passes the noise straight through.

    Outlier hold survives both: the alternating case never corroborates, and
    a genuine large change (a one-shot, a full-heal) simply lands one tick
    late. Normal-sized changes aren't delayed at all.

    Cost: a real, large, single-tick dip that fully recovers before the next
    read isn't counted -- but at 2Hz such a dip was already invisible half the
    time. A *persistently* misread value is booked once, then becomes the
    baseline; bounded, unlike the old unbounded ratchet.

    Establishing `max` in the first place is a separate concern, owned by
    Session's calibration (see its docstring) -- this class never adopts a
    max on its own. `confirm_max()` is the only way `_max` is set from
    nothing; before that call, `record()` treats every tick as unusable and
    is a no-op, exactly like a tick with no maximum at all.
    """

    # Fraction of max MP/HP a single 500ms tick may move before the reading
    # is treated as suspect. Generous on purpose: this only decides what needs
    # corroboration, not what counts, so a real big hit still lands (one tick
    # late) while digit-truncation misreads -- which are always wrong by most
    # of the bar -- are the ones held.
    OUTLIER_FRACTION = 0.5

    def __init__(self) -> None:
        self.loss = 0
        self._loss_floor = 0
        self._last: int | None = None       # last accepted value
        self._segment_start: int | None = None
        self._gross_loss = 0
        self._recovery_evidence = 0
        self._max: int | None = None        # established max for this stat
        self._max_candidate: int | None = None
        self._max_candidate_count = 0
        self._max_candidate_level_bumped = False
        self._last_level: int | None = None
        self._candidate: int | None = None  # outlier awaiting corroboration

    def confirm_max(self, maximum: int) -> None:
        """Called once by Session's calibration, when `maximum` has been
        corroborated by independent means (see Session._calibrate_hp_max /
        _calibrate_mp_max) -- this is the only place `_max` is ever set from
        nothing. `_last` is deliberately left alone: the very next record()
        call baselines off its `cur` with no loss booked, same as the first
        tick of a stat that's never seen a max at all."""
        self._max = maximum

    def reset(self, last: int | None) -> None:
        """New session: zero the total, keep the established max (it survives
        session boundaries), baseline off the last known value."""
        self.loss = 0
        self._loss_floor = 0
        self._last = last
        self._segment_start = last
        self._gross_loss = 0
        self._recovery_evidence = 0
        self._candidate = None

    def rebaseline(self, cur: int | None) -> None:
        """Re-anchor without booking a loss -- used when resuming from a
        pause (see Session.resume): whatever happened while paused is
        invisible to the session, so the first post-resume reading must not
        be diffed against the pre-pause value."""
        if cur is not None:
            self._loss_floor = self.loss
            self._last = cur
            self._segment_start = cur
            self._gross_loss = 0
            self._recovery_evidence = 0
        self._candidate = None

    def add_recovery_evidence(self, amount: int) -> None:
        """Add an upward HP/MP delta observed by the economy tracker.

        The economy worker and the status worker have different OCR guards.
        If the status tracker holds a large damage frame as suspicious but the
        next potion heal is valid, this evidence lets the loss total recover
        the hidden damage instead of reporting only the endpoint difference.
        """
        if amount <= 0 or self._segment_start is None or self._last is None:
            return
        self._recovery_evidence += amount
        self._recompute_loss()

    def record(self, cur: int | None, maximum: int | None = None, level: int | None = None) -> None:
        if cur is None:
            return
        if maximum is not None:
            accepted = self._accept_max(maximum, level)
            if level is not None:
                self._last_level = level
            if not accepted:
                return  # misparsed tick -- don't let it near the loss math
            if cur > maximum:
                return  # cur can never exceed max; this reading is garbage
        if self._last is None:
            self._last = cur
            self._segment_start = cur
            return
        tolerance = (self._max or self._last) * self.OUTLIER_FRACTION
        if abs(cur - self._last) <= tolerance:
            self._commit(cur)
        elif self._candidate is not None and abs(cur - self._candidate) <= tolerance:
            self._commit(cur)  # corroborated -- a real jump after all
        else:
            self._candidate = cur  # hold; a normal reading next tick discards it

    def _commit(self, cur: int) -> None:
        if cur < self._last:
            self._gross_loss += self._last - cur
        self._last = cur
        self._candidate = None
        self._recompute_loss()

    def _recompute_loss(self) -> None:
        if self._segment_start is None or self._last is None:
            self.loss = self._loss_floor
            return
        endpoint_estimate = max(
            0,
            self._segment_start - self._last + self._recovery_evidence,
        )
        self.loss = self._loss_floor + max(self._gross_loss, endpoint_estimate)

    # A level-up nudges max HP/MP; nothing in the game multiplies it. A
    # proposed max outside this factor of the established one is garbage and
    # is never adopted, no matter how many ticks repeat it -- corroboration
    # alone is not enough, because when a window covers the panel the OCR
    # garbage is *static*: the identical wrong text every tick. That is how a
    # max of 281616 (from '281616' misread out of '2816') got installed in a
    # live capture, after which a bogus cur of 28163 passed every check and
    # booked 25,347 of phantom loss when the panel came back.
    MAX_CHANGE_FACTOR = 2.0

    def _accept_max(self, maximum: int, level: int | None) -> bool:
        if maximum <= 0:
            return False
        if self._max is None:
            # Session's calibration (require_calibration=True) always calls
            # confirm_max() before this tracker ever sees a real record()
            # call, so this branch only fires with calibration off -- same
            # unconditional first-adopt as before that feature existed.
            self._max = maximum
            return True
        if maximum == self._max:
            self._max_candidate = None
            self._max_candidate_count = 0
            return True
        if not (self._max / self.MAX_CHANGE_FACTOR <= maximum <= self._max * self.MAX_CHANGE_FACTOR):
            return False  # implausible -- never adopt, however often it repeats
        level_bumped = (
            level is not None and self._last_level is not None and level > self._last_level
        )
        if maximum != self._max_candidate:
            self._max_candidate = maximum
            self._max_candidate_count = 1
            # Captured at the moment the candidate first appears, not
            # re-evaluated every tick after -- a level-up IS the
            # corroboration, so it only counts for the change it accompanies,
            # not for an unrelated max blip that happens to follow one.
            self._max_candidate_level_bumped = level_bumped
        else:
            self._max_candidate_count += 1
        required = 2 if self._max_candidate_level_bumped else 3
        if self._max_candidate_count >= required:
            self._max = maximum
            self._max_candidate = None
            self._max_candidate_count = 0
            return True
        return False


class Session:
    """See the module docstring for the fixed-epoch design.

    Calibration (require_calibration=True, the default): max HP, max MP, and
    the starting EXP are not trusted from a single reading. Each is
    established independently after two valid 0.3s frames corroborate it, and
    nothing is recorded into the session until all three are confirmed.
    is_calibrating reports
    this state; while true, elapsed() reads 0 and the HUD should show
    "Calibrating...".

    This exists because the first screenshot after launch/focus can be wrong
    -- see mp-loss-investigation-2026-08-17.md for the live case: a covered
    panel misread max MP as 281616 from 2816, which then locked out the real
    value forever (MAX_CHANGE_FACTOR rejects anything more than 2x away from
    an established max, and there was nothing to re-establish it with).

    Repetition alone is a weak acceptance test. When a window covers the
    panel the OCR garbage is *static* -- the identical wrong text every tick
    -- so N-identical-reads certifies it exactly as happily as it certifies
    the truth. Two earlier guard designs in _LossTracker failed this exact
    way (see its docstring). _calib_liveness still gives a changing live frame
    extra weight, but it is no longer needed to make an idle character wait
    for a third sample. The structural parser and capture occlusion checks
    remain the guards against malformed or covered frames.

    This is NOT a claim that two static ticks of well-formed-but-wrong data can
    never calibrate -- nothing at this layer has ground truth to check a
    number against in isolation, that's what parser.py's structural filters
    (a missing '[' etc.) and capture.py's PANEL_OBSCURED occlusion probe are
    for, upstream of this. What calibration actually protects against, and
    is tested against real captured frames for
    (test_calibration_locks_onto_the_truth_before_real_garbage_arrives): a
    single bad frame arriving among otherwise-good ones, and a bad max
    change post-calibration outliving its own corroboration window -- the
    two concrete bugs this whole mechanism exists to fix.

    Once confirmed, calibration does not repeat for the lifetime of this
    Session object -- start() (restart, timer rollover) carries the
    already-confirmed max/baseline forward exactly as before, with no delay.
    A level-up's max change goes through _LossTracker's own (separate,
    always-on) corroboration instead, never through this calibration path
    again.
    """

    # Two valid 0.3s frames are enough for the short startup calibration. The
    # parser's structural checks and capture occlusion guard already reject
    # malformed/covered frames; making the UI wait three frames on top of
    # model loading made Start feel unresponsive. A changing live frame still
    # receives the liveness weight, while an idle character confirms after
    # the second identical valid frame.
    CALIB_TARGET = 2

    def __init__(self, require_calibration: bool = True) -> None:
        self._require_calibration = require_calibration
        self._calibrated = not require_calibration

        self._start_time: float | None = None
        # Wall-clock timestamps are persisted in history and remain the public
        # session timebase.  Keep a monotonic companion for the very short
        # live-rate window where Windows can return the same wall-clock tick for
        # two back-to-back OCR frames.
        self._start_perf: float | None = None
        self._start_exp: int | None = None
        self._hp = _LossTracker()
        self._mp = _LossTracker()
        self._exp_cur: int | None = None
        self._hp_cur: int | None = None
        self._mp_cur: int | None = None
        self._total_exp: float | None = None
        # EXP is measured per *level segment* -- see _record_exp for why it is
        # deliberately not a tick-by-tick accumulator.
        self._banked = 0                      # gain from levels completed this session
        self._segment_start: int | None = None  # EXP at the start of the current level
        self._last_exp: int | None = None
        self._last_level: int | None = None
        self._last_implied_total: float | None = None  # see _exp_reading_is_trusted

        # Calibration state -- see the class docstring. Inert (never
        # consulted) when require_calibration is False.
        self._hp_max_calibrated = self._calibrated
        self._mp_max_calibrated = self._calibrated
        self._exp_calibrated = self._calibrated
        self._hp_calib_max: int | None = None
        self._hp_calib_count = 0
        self._mp_calib_max: int | None = None
        self._mp_calib_count = 0
        self._exp_calib_baseline: int | None = None
        self._exp_calib_total: float | None = None
        self._exp_calib_count = 0
        self._calib_last_snapshot: tuple | None = None

        # Pause state (see pause()/resume()).
        self._paused = False
        self._pause_started_at: float | None = None
        self._paused_total = 0.0
        self._resume_pending = False

    def start(self, now: float | None = None) -> None:
        """Begin a new session. Carries forward whatever EXP/HP/MP values are
        already known as the new baseline (so a level-up-triggered or
        timer-triggered restart doesn't wait a tick to re-establish it) --
        unless calibration has never completed, in which case there is
        nothing yet to carry forward and the clock stays off until it does."""
        if self._require_calibration and not self._calibrated:
            self._start_time = None
            self._start_perf = None
        else:
            self._start_time = now if now is not None else time.time()
            self._start_perf = time.perf_counter()
        self._start_exp = self._exp_cur
        self._hp.reset(self._hp_cur)
        self._mp.reset(self._mp_cur)
        self._total_exp = None  # re-derived fresh -- could differ after a level-up
        self._banked = 0
        self._segment_start = self._exp_cur
        self._last_exp = self._exp_cur
        self._paused = False
        self._pause_started_at = None
        self._paused_total = 0.0
        self._resume_pending = False

    def begin_fresh(self) -> None:
        """Start the next session from its first post-start live sample.

        ``start()`` intentionally carries the last accepted values forward for
        callers that already have a trustworthy current frame.  The overlay's
        Start button can be pressed after a previous run has stopped, though,
        and carrying that old EXP value would make the new interval's
        projection depend on the previous session.  Clear the per-session
        baselines while retaining calibrated HP/MP maxima; ``record()`` will
        establish the clock and all baselines from the first fresh frame.
        """
        self._start_time = None
        self._start_perf = None
        self._start_exp = None
        self._hp_cur = None
        self._mp_cur = None
        self._exp_cur = None
        self._total_exp = None
        self._banked = 0
        self._segment_start = None
        self._last_exp = None
        self._last_level = None
        self._last_implied_total = None
        self._hp.reset(None)
        self._mp.reset(None)
        self._paused = False
        self._pause_started_at = None
        self._paused_total = 0.0
        self._resume_pending = False
        if not self._calibrated:
            # A restart during startup should not inherit partial candidates.
            self._hp_calib_max = None
            self._hp_calib_count = 0
            self._mp_calib_max = None
            self._mp_calib_count = 0
            self._exp_calib_baseline = None
            self._exp_calib_total = None
            self._exp_calib_count = 0
            self._calib_last_snapshot = None

    # ---- pause/resume -------------------------------------------------

    def pause(self, now: float | None = None) -> None:
        """Freeze the session: elapsed() stops advancing and record() becomes
        a no-op, but the caller keeps polling/rendering live OCR values
        independently of Session -- pausing never stops the HUD updating,
        only what gets counted. No-op if not running or already paused."""
        if self._paused or self._start_time is None:
            return
        self._paused = True
        self._pause_started_at = now if now is not None else time.time()

    def resume(self, now: float | None = None) -> None:
        """Unfreeze. The next record() call re-baselines HP/MP/EXP off
        whatever it reads rather than diffing against the pre-pause values --
        see _rebaseline_after_resume. A level-up that happened entirely
        during the pause can't be reconstructed (no ticks were seen); it is
        absorbed into the new baseline with nothing banked, the same known
        gap as a session that misses a level-up's percentage reading."""
        if not self._paused:
            return
        now = now if now is not None else time.time()
        if self._pause_started_at is not None:
            self._paused_total += now - self._pause_started_at
        self._paused = False
        self._pause_started_at = None
        self._resume_pending = True

    @property
    def is_paused(self) -> bool:
        return self._paused

    def _rebaseline_after_resume(self, exp_cur: int | None, hp_cur: int | None, mp_cur: int | None) -> None:
        if exp_cur is not None:
            gained_during_pause = (exp_cur - self._exp_cur) if self._exp_cur is not None else 0
            # Only ever push segment_start forward -- a level-up during the
            # pause makes this look like a large drop, which the max(0, ...)
            # here correctly does NOT treat as a negative gain to claw back;
            # it just leaves the stale segment_start behind, i.e. the
            # documented "nothing banked" gap for that case.
            if self._segment_start is not None:
                self._segment_start += max(0, gained_during_pause)
            self._exp_cur = exp_cur
            self._last_exp = exp_cur
        if hp_cur is not None:
            self._hp.rebaseline(hp_cur)
            self._hp_cur = hp_cur
        if mp_cur is not None:
            self._mp.rebaseline(mp_cur)
            self._mp_cur = mp_cur

    # ---- calibration ----------------------------------------------------

    def _calib_liveness(self, hp_cur, mp_cur, exp_cur, level) -> bool:
        """Return whether the current valid frame differs from the prior one.

        A changing live frame receives a small liveness bonus, while static
        valid data still reaches the two-frame calibration target.
        """
        current = (hp_cur, mp_cur, exp_cur, level)
        previous = self._calib_last_snapshot
        self._calib_last_snapshot = current
        if previous is None:
            return False
        return any(a is not None and b is not None and a != b for a, b in zip(previous, current))

    def _calibrate_hp_max(self, hp_cur, hp_max, live: bool) -> None:
        if hp_cur is None or hp_max is None or hp_max <= 0 or hp_cur > hp_max:
            return
        weight = 2 if live else 1
        if hp_max == self._hp_calib_max:
            self._hp_calib_count += weight
        else:
            self._hp_calib_max = hp_max
            self._hp_calib_count = weight
        if self._hp_calib_count >= self.CALIB_TARGET:
            self._hp_max_calibrated = True

    def _calibrate_mp_max(self, mp_cur, mp_max, live: bool) -> None:
        if mp_cur is None or mp_max is None or mp_max <= 0 or mp_cur > mp_max:
            return
        weight = 2 if live else 1
        if mp_max == self._mp_calib_max:
            self._mp_calib_count += weight
        else:
            self._mp_calib_max = mp_max
            self._mp_calib_count = weight
        if self._mp_calib_count >= self.CALIB_TARGET:
            self._mp_max_calibrated = True

    def _calibrate_exp_baseline(self, exp_cur, exp_pct, live: bool) -> None:
        """Corroborate by *consistency*, not equality -- EXP moves every tick
        during real play, so unlike max HP/MP the candidate value itself
        isn't expected to repeat. What must repeat is the implied level total
        (cur / (pct/100)), the same cross-check _exp_reading_is_trusted uses
        post-calibration. The baseline stays pinned to the FIRST reading of a
        corroborating streak, not the latest -- so once confirmed, EXP gained
        between that first reading and the confirming one is still counted
        rather than lost to the calibration wait."""
        if exp_cur is None:
            return
        implied = exp_cur / (exp_pct / 100) if exp_pct else None
        weight = 2 if live else 1
        if self._exp_calib_baseline is None:
            self._exp_calib_baseline = exp_cur
            self._exp_calib_total = implied
            self._exp_calib_count = weight
        else:
            ok = exp_cur >= self._exp_calib_baseline
            if ok and implied is not None and self._exp_calib_total is not None:
                band = Session.EXP_TOTAL_BAND + (0.005 / exp_pct)
                ok = abs(implied / self._exp_calib_total - 1) <= band
            if ok:
                self._exp_calib_count += weight
                if implied is not None:
                    self._exp_calib_total = implied
            else:
                self._exp_calib_baseline = exp_cur
                self._exp_calib_total = implied
                self._exp_calib_count = weight
        if self._exp_calib_count >= self.CALIB_TARGET:
            self._exp_calibrated = True

    @property
    def is_calibrating(self) -> bool:
        return self._require_calibration and not self._calibrated

    def record(
        self, exp_cur: int | None, hp_cur: int | None, mp_cur: int | None, exp_pct: float | None = None,
        hp_max: int | None = None, mp_max: int | None = None, level: int | None = None,
    ) -> None:
        """hp_max/mp_max/level are optional but strongly recommended: the maxes
        enable _LossTracker's max-stability guard (which catches the
        high-magnitude OCR misreads), and the level is what tells a level-up
        apart from a misread when EXP drops -- see _record_exp."""
        if self._paused:
            return

        if self._resume_pending:
            self._resume_pending = False
            self._rebaseline_after_resume(exp_cur, hp_cur, mp_cur)
            return  # this tick only re-anchors; accounting resumes next tick

        if self._require_calibration and not self._calibrated:
            live = self._calib_liveness(hp_cur, mp_cur, exp_cur, level)
            if not self._hp_max_calibrated:
                self._calibrate_hp_max(hp_cur, hp_max, live)
            if not self._mp_max_calibrated:
                self._calibrate_mp_max(mp_cur, mp_max, live)
            if not self._exp_calibrated:
                self._calibrate_exp_baseline(exp_cur, exp_pct, live)
            if not (self._hp_max_calibrated and self._mp_max_calibrated and self._exp_calibrated):
                return
            self._calibrated = True
            self._hp.confirm_max(self._hp_calib_max)
            self._mp.confirm_max(self._mp_calib_max)
            self._start_exp = self._exp_calib_baseline
            self._segment_start = self._exp_calib_baseline
            self._last_exp = self._exp_calib_baseline
            self._start_time = time.time()
            self._start_perf = time.perf_counter()
            # Fall through -- this confirming tick's own readings are still
            # real data, not spent purely on calibration.

        if self._start_time is None:
            self.start()
        if self._start_exp is None and exp_cur is not None:
            self._start_exp = exp_cur
        self._hp.record(hp_cur, hp_max, level)
        self._mp.record(mp_cur, mp_max, level)
        if hp_cur is not None:
            self._hp_cur = hp_cur
        if mp_cur is not None:
            self._mp_cur = mp_cur
        self._record_exp(exp_cur, exp_pct, level)

    def add_recovery_evidence(self, kind: str, amount: int) -> None:
        """Feed recovery observed by the separate economy OCR worker."""
        if kind == "hp":
            self._hp.add_recovery_evidence(amount)
        elif kind == "mp":
            self._mp.add_recovery_evidence(amount)

    # A reading may deviate this far from the level total established by the
    # previous tick before it is treated as garbage. Deliberately loose: it is
    # meant to catch order-of-magnitude nonsense, not to police OCR jitter.
    EXP_TOTAL_BAND = 0.25

    def _exp_reading_is_trusted(self, exp_cur: int, exp_pct: float | None, level: int | None) -> bool:
        """Cross-check `cur` against `pct`.

        They are two independent OCR readings of the same quantity, so their
        ratio -- the level's total EXP, constant within a level -- validates
        them against each other. 'EXP S255[1 12%]' (a 5 read as an S) implies a
        total of 22,768 where every good reading agrees on ~468,500.

        Compared against the *previous accepted tick*, never against a total
        learned from the data: in the live capture the bad value was the
        majority (379 of 429 ticks), so anything that learned would have
        learned 22,768 and rejected every correct reading afterwards.

        This mainly protects finalize(). Segments (see _record_exp) already
        make a bad frame transient, but a summary freezes one instant, and a
        garbage frame captured there is written to History permanently.
        """
        if level is not None and level != self._last_level:
            self._last_implied_total = None  # new level, new total: re-baseline
        if exp_pct:
            implied = exp_cur / (exp_pct / 100)
            if self._last_implied_total:
                # pct is rounded to 2dp, so at small pct the implied total is
                # numerically unstable -- half a least-significant digit is
                # 0.005/pct in relative terms, i.e. +-50% at pct=0.01. A fixed
                # band would reject every legitimate reading after a level-up.
                band = self.EXP_TOTAL_BAND + (0.005 / exp_pct)
                if abs(implied / self._last_implied_total - 1) > band:
                    return False
            self._last_implied_total = implied
            return True
        # No percentage to check against: fall back to the one bound that needs
        # no cross-reference -- a single 500ms tick cannot gain a whole level.
        if self._total_exp and self._last_exp is not None:
            if abs(exp_cur - self._last_exp) > self._total_exp:
                return False
        return True

    def _record_exp(self, exp_cur: int | None, exp_pct: float | None, level: int | None) -> None:
        """Track EXP gained, per level *segment*.

        Within a level this is plain end-minus-start, which is the important
        property: it depends only on the current reading, so a garbage frame
        shows a wrong number for one tick and then self-corrects. Only a
        level-up banks a segment and opens a new one.

        It is deliberately NOT a tick-by-tick accumulator. That was tried
        (2026-08-19) to handle level-ups and it regressed badly: summing every
        rise means one absurd reading is baked in forever, exactly the ratchet
        that makes HP/MP loss fragile. A single garbage frame -- 'EXP101332182',
        no brackets, no percentage -- booked +101,322,049 of phantom gain in
        one tick and it never came back. end-minus-start showed +16,058 over
        the same run.

        HP/MP cannot be done this way and must accumulate: loss is a path
        integral, not a difference. A character who takes 5,000 damage and
        potions back to full has the same endpoints as one who stood still.
        That is why the guards in _LossTracker exist there and are not needed
        here.
        """
        if exp_cur is None:
            return
        if not self._exp_reading_is_trusted(exp_cur, exp_pct, level):
            return  # garbage -- treat it as an unreadable field and carry forward
        # Captured before the update below: _total_exp is re-derived every tick
        # from cur/pct, so by the time we see the reset it already describes
        # the *new* level. The level just finished has to be measured with the
        # pre-update value, or every level-up under-counts by a level.
        previous_total = self._total_exp

        if self._segment_start is None:
            self._segment_start = exp_cur

        levelled = (
            level is not None
            and self._last_level is not None
            and level > self._last_level
            and self._last_exp is not None
            and exp_cur < self._last_exp
        )
        if levelled:
            # Bank the level just finished, then start the new segment at 0 --
            # whatever is already banked into the new level counts as gain.
            # Requiring *both* a level increase and an EXP reset keeps a
            # one-off level misread from banking a phantom segment.
            if previous_total:
                self._banked += max(0, int(previous_total - self._segment_start))
            # Without a percentage reading the finished level's total is
            # unknown, so its remainder is dropped rather than invented: an
            # under-count, never a fabricated number.
            self._segment_start = 0

        self._last_exp = exp_cur
        self._exp_cur = exp_cur
        if level is not None:
            self._last_level = level
        if exp_cur and exp_pct:
            self._total_exp = exp_cur / (exp_pct / 100)

    def elapsed(self, now: float | None = None) -> float:
        """Pause-adjusted: frozen at the pause instant while paused, and
        resumes counting from where it left off once resume() runs -- see
        pause()/resume()."""
        if self._start_time is None:
            return 0.0
        if now is None:
            now = time.time()
            wall_elapsed = now - self._start_time
            # Some Windows clocks have millisecond-scale granularity.  A pair
            # of immediate OCR frames can therefore have positive EXP gain but
            # a reported wall duration of exactly zero.  Use the monotonic
            # companion only for that degenerate live case; explicit `now`
            # values used by history/pause tests keep their deterministic wall
            # clock semantics.
            if wall_elapsed <= 0 and self._start_perf is not None and not self._paused:
                return max(0.0, time.perf_counter() - self._start_perf)
        else:
            wall_elapsed = now - self._start_time
        end = self._pause_started_at if (self._paused and self._pause_started_at is not None) else now
        return max(0.0, (end - self._start_time) - self._paused_total)

    @property
    def start_exp(self) -> int | None:
        return self._start_exp

    @property
    def exp_diff(self) -> int | None:
        """EXP gained since the session started, spanning level-ups (see
        _record_exp). Clamped at 0: within a level EXP only rises, so a
        negative here means a misread, and showing 0 for a tick beats
        rendering a negative behind the '+' the HUD prints."""
        if self._start_exp is None or self._exp_cur is None or self._segment_start is None:
            return None
        return max(0, self._banked + (self._exp_cur - self._segment_start))

    @property
    def hp_loss(self) -> int:
        return self._hp.loss

    @property
    def mp_loss(self) -> int:
        return self._mp.loss

    @property
    def total_exp(self) -> float | None:
        return self._total_exp

    @property
    def exp_per_hour(self) -> float | None:
        """Current session EXP rate, normalized to one hour."""
        diff = self.exp_diff
        elapsed = self.elapsed()
        # A valid session with no EXP gain is still a real zero rate. Returning
        # None here made an idle character look indistinguishable from a
        # broken capture/OCR pipeline in the HUD (which rendered both as "--").
        if diff is None or elapsed <= 0:
            return None
        return diff * 3600 / elapsed

    def projected_exp(self, window_s: float, now: float | None = None) -> int | None:
        """EXP this session would total if the current rate held for the
        whole `window_s` -- rate is exp_diff / elapsed(), which is already
        pause-adjusted, so pausing doesn't deflate the projection. None
        before there's enough signal to extrapolate from (same 3s/positive
        gain guard the level-up ETA uses -- a 1-2s sample swings wildly)."""
        diff = self.exp_diff
        elapsed = self.elapsed(now)
        if diff is None or elapsed <= 3:
            return None
        # Timestamp subtraction can leave a mathematically integral result a
        # few ulps below the integer (for example 5999.999999999).  Rounding
        # keeps the displayed projection stable without changing its unit.
        return int(round(diff / elapsed * window_s))

    def finalize(self, interval_minutes: float | None = None, now: float | None = None) -> SessionSummary:
        end_time = now if now is not None else time.time()
        return SessionSummary(
            start_time=self._start_time if self._start_time is not None else end_time,
            end_time=end_time,
            start_exp=self._start_exp,
            end_exp=self._exp_cur,
            hp_loss=self._hp.loss,
            mp_loss=self._mp.loss,
            total_exp=self._total_exp,
            interval_minutes=interval_minutes,
            exp_gained=self.exp_diff,
        )
