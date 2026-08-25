"""Event-driven economy and recovery tracking.

The game does not expose a cumulative mesos counter in the visible HUD.  The
right-hand pickup feed is therefore treated as a stream of short-lived OCR
events.  Potion usage is charged only from a configured shortcut-slot quantity
drop.  HP/MP increases are a separate signal: they are labelled as potion
recovery only when they follow a confirmed slot drop, otherwise they remain
natural/skill recovery and become a saved-cost estimate.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .regions import SHORTCUT_SLOT_BOXES
from .settings import PotionSlotConfig

_MESOS_RE = re.compile(r"[+]?(\d[\d,]*)")
_INTEGER_RE = re.compile(r"\d[\d,]*")
RECOVERY_MATCH_TOLERANCE = 0.15
RECOVERY_MIN_TOLERANCE = 3
# The shortcut OCR and status OCR run on separate worker cadences.  A drink
# can therefore be confirmed after the status frame that contains its heal;
# keep the marker alive long enough for that frame to arrive.
POTION_PENDING_SECONDS = 5.0
# Shortcut OCR is sampled about every 0.75s.  A lower value must survive two
# samples before it becomes a consumption event; one-frame digit loss during
# the drink animation must not become several potions.  A held shortcut key
# can still consume a larger batch between auxiliary scans, so only jumps that
# look like a missing leading digit are rejected.
SLOT_CONFIRMATIONS_REQUIRED = 2
# Enhanced full-bar retries can take longer than one auxiliary interval on a
# CPU-only machine.  Keep a one-frame lower candidate through that delay so a
# later identical/lower read can still confirm it.
SLOT_CANDIDATE_MAX_GAP_SECONDS = 5.0
MAX_SLOT_DROP_PER_SCAN = 99
# A final inventory reconciliation may cover a long interval, so it is not
# limited by the per-scan held-key guard above.  Values larger than this are
# more likely to be a missing OCR digit than a real single-session depletion.
MAX_SLOT_RECONCILE_DROP = 999
HP_RECOVERY_MESOS_PER_POINT = 1.2
MP_RECOVERY_MESOS_PER_POINT = 2.1
MESOS_Y_MATCH_PX = 52.0


def _is_probable_truncated_count(previous: int, current: int) -> bool:
    """Return whether OCR dropped one or more leading quantity digits."""
    previous_text = str(previous)
    current_text = str(current)
    return (
        len(current_text) < len(previous_text)
        and bool(current_text)
        and previous_text.endswith(current_text)
    )


def _is_probable_leading_digit_recovery(previous: int, current: int) -> bool:
    """Return whether a later frame restored a digit lost in the baseline."""
    previous_text = str(previous)
    current_text = str(current)
    return len(current_text) > len(previous_text) and current_text.endswith(previous_text)


def parse_mesos_amount(text: str) -> int | None:
    """Extract a mesos amount from a pickup line.

    The exact punctuation varies between clients/OCR frames, so the parser
    anchors on the distinctive 楓幣 token and takes the last numeric group
    after it.  Some RapidOCR frames drop part or all of 楓幣 and return the
    game's money template as ``取楓。(+144)`` or ``獲取。(+144)`` instead; that
    narrow fallback is accepted only for a 楓/取 token or 獲取/獲得 line which
    does not contain an experience marker.
    EXP lines are intentionally ignored even when they contain a number in
    parentheses.
    """
    compact = re.sub(r"\s+", "", text).replace(",", "")
    # The OCR model occasionally emits simplified characters or confuses
    # 楓 with 椋 in the small pickup toast.
    compact = (
        compact.replace("楓币", "楓幣")
        .replace("枫幣", "楓幣")
        .replace("枫币", "楓幣")
        .replace("椋幣", "楓幣")
        .replace("椋币", "楓幣")
    )
    marker = compact.find("楓幣")
    fallback_without_marker = False
    if marker < 0:
        # This is deliberately not a generic "any line with (+number)"
        # fallback: item counts and EXP notifications also use parentheses.
        # The pickup verb is the remaining stable part of the game's money
        # template in the observed failure mode.
        if (
            not any(token in compact for token in (
                "經驗", "经验", "經值", "验值", "EXP", "exp", "Bonus", "增益"
            ))
            and any(token in compact for token in ("獲取", "獲得", "取", "拾取"))
        ):
            fallback_without_marker = True
        else:
            return None
    if fallback_without_marker:
        matches = _MESOS_RE.findall(compact)
    else:
        # The normal template puts the amount after 楓幣, but OCR sometimes
        # reverses the two fragments (``+123 楓幣``).
        after = _MESOS_RE.findall(compact[marker + len("楓幣"):])
        before = _MESOS_RE.findall(compact[:marker])
        matches = after or before
    if not matches:
        return None
    try:
        value = int(matches[-1].replace(",", ""))
    except ValueError:
        return None
    return value if value >= 0 else None


def parse_slot_count(text: str) -> int | None:
    """Read the last integer in one configured shortcut-slot crop."""
    matches = _INTEGER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


@dataclass(frozen=True)
class MesosObservation:
    amount: int
    y: float | None = None


@dataclass(frozen=True)
class EconomySnapshot:
    mesos: int
    mesos_events: int
    potion_uses: int
    potion_cost: int
    hp_potion_uses: int
    hp_potion_cost: int
    mp_potion_uses: int
    mp_potion_cost: int
    shared_potion_uses: int
    shared_potion_cost: int
    hp_recovery_natural: int
    hp_recovery_potion: int
    mp_recovery_natural: int
    mp_recovery_potion: int
    hp_recovery_savings: float
    mp_recovery_savings: float
    potion_breakdown: dict[str, int]
    # Visible shortcut quantities are part of the live snapshot so the user
    # can verify the baseline before trusting potion-use totals.
    shortcut_baseline: dict[str, int] = field(default_factory=dict)
    shortcut_current: dict[str, int] = field(default_factory=dict)
    shortcut_baseline_ready: bool = False


@dataclass
class _VisibleMesos:
    amount: int
    y: float | None


@dataclass
class _PendingPotion:
    config: PotionSlotConfig
    expires_at: float
    matched_kinds: set[str] = field(default_factory=set)


class MesosFeedTracker:
    """Count only newly appearing pickup messages, not persistent OCR lines."""

    def __init__(self) -> None:
        self.total = 0
        self.events = 0
        self._visible: list[_VisibleMesos] = []
        self._initialized = False

    def reset(self) -> None:
        self.total = 0
        self.events = 0
        self._visible.clear()
        self._initialized = False

    def update(self, observations: Iterable[MesosObservation], now: float | None = None) -> int:
        del now  # kept in the signature for a future fade-time model
        current = list(observations)
        if not current:
            # A genuinely empty feed is a strong boundary: an identical line
            # appearing after this point is a new pickup, even if the amount
            # happens to match the previous one.
            self._visible.clear()
            self._initialized = True
            return 0

        if not self._initialized:
            self._visible = [_VisibleMesos(item.amount, item.y) for item in current]
            self._initialized = True
            return 0

        unmatched = list(self._visible)
        new_total = 0
        new_events = 0
        next_visible: list[_VisibleMesos] = []
        for observation in current:
            match_index = self._find_match(unmatched, observation)
            if match_index is None:
                new_total += observation.amount
                new_events += 1
            else:
                unmatched.pop(match_index)
            next_visible.append(_VisibleMesos(observation.amount, observation.y))
        self._visible = next_visible
        self.total += new_total
        self.events += new_events
        return new_total

    @staticmethod
    def _find_match(visible: list[_VisibleMesos], observation: MesosObservation) -> int | None:
        candidates = [
            (index, item)
            for index, item in enumerate(visible)
            if item.amount == observation.amount
            and (
                item.y is None
                or observation.y is None
                or abs(item.y - observation.y) <= MESOS_Y_MATCH_PX
            )
        ]
        if not candidates:
            return None
        if observation.y is None:
            return candidates[0][0]
        return min(
            candidates,
            key=lambda pair: abs((pair[1].y if pair[1].y is not None else observation.y) - observation.y),
        )[0]


def _line_text_and_y(line: object) -> tuple[str, float | None]:
    if isinstance(line, str):
        return line, None
    text = getattr(line, "text", None)
    y = getattr(line, "y", None)
    if isinstance(text, str):
        return text, float(y) if isinstance(y, (int, float)) else None
    if isinstance(line, (tuple, list)) and line:
        raw_text = line[0]
        raw_y = line[1] if len(line) > 1 else None
        return str(raw_text), float(raw_y) if isinstance(raw_y, (int, float)) else None
    return "", None


class EconomyTracker:
    """Per-session mesos, potion, and recovery accounting."""

    def __init__(
        self,
        slots: Iterable[PotionSlotConfig],
        default_recovery_hp: int = 0,
        default_recovery_mp: int = 0,
    ) -> None:
        self._slots: list[PotionSlotConfig] = []
        self._by_slot: dict[str, PotionSlotConfig] = {}
        self._default_recovery_hp = max(0, int(default_recovery_hp))
        self._default_recovery_mp = max(0, int(default_recovery_mp))
        self._mesos = MesosFeedTracker()
        self.reset()
        self.configure(slots, default_recovery_hp, default_recovery_mp)

    def configure(
        self,
        slots: Iterable[PotionSlotConfig],
        default_recovery_hp: int | None = None,
        default_recovery_mp: int | None = None,
    ) -> None:
        previous_by_slot = self._by_slot
        configured = [slot for slot in slots if slot.enabled]
        # A blank settings table should still detect quantity drops. Unknown
        # slots are classified as shared and have zero cost until the user
        # enters a name/price/type. Once any slot is explicitly configured,
        # however, only those slots are authoritative; neighboring cells must
        # not be allowed to masquerade as a configured blue/HP potion.
        configured_by_slot = {slot.slot: slot for slot in configured}
        self._slots = configured or [
            PotionSlotConfig(slot=slot, kind="both", enabled=True)
            for slot in SHORTCUT_SLOT_BOXES
        ]
        self._by_slot = {slot.slot: slot for slot in self._slots}
        # A settings edit can replace the item in a slot while an old OCR
        # quantity is still in the queue.  Establish a fresh baseline instead
        # of charging the first reading from the new configuration as a use.
        if previous_by_slot != self._by_slot:
            self._slot_counts.clear()
            self._slot_charged.clear()
            self._shortcut_baseline.clear()
            self._shortcut_observed.clear()
            self._slot_candidates.clear()
            self._pending_potions.clear()
        if default_recovery_hp is not None:
            self._default_recovery_hp = max(0, int(default_recovery_hp))
        if default_recovery_mp is not None:
            self._default_recovery_mp = max(0, int(default_recovery_mp))

    def begin_quick_slot_baseline(self) -> None:
        """Discard the previous inventory baseline without clearing totals.

        Start/Resume must sample the currently visible HP/MP quantities first.
        Otherwise a quantity change while the session was stopped or paused
        can be charged as a new potion use when monitoring resumes.
        """
        self._slot_counts.clear()
        self._slot_charged = {}
        self._shortcut_baseline.clear()
        self._shortcut_observed.clear()
        self._slot_candidates.clear()
        self._pending_potions.clear()

    def prime_quick_slot_counts(self, counts: dict[str, int]) -> None:
        """Set the first visible inventory quantities without counting drops."""
        self.begin_quick_slot_baseline()
        for slot_id, count in counts.items():
            if count >= 0:
                self._shortcut_baseline[slot_id] = count
                self._slot_counts[slot_id] = count
                self._shortcut_observed[slot_id] = count
                self._slot_charged[slot_id] = 0

    def reset(self) -> None:
        self._mesos.reset()
        self._potion_uses = 0
        self._potion_cost = 0
        self._hp_potion_uses = 0
        self._hp_potion_cost = 0
        self._mp_potion_uses = 0
        self._mp_potion_cost = 0
        self._shared_potion_uses = 0
        self._shared_potion_cost = 0
        self._potion_breakdown: Counter[str] = Counter()
        self._hp_recovery_natural = 0
        self._hp_recovery_potion = 0
        self._mp_recovery_natural = 0
        self._mp_recovery_potion = 0
        self._hp_recovery_savings = 0.0
        self._mp_recovery_savings = 0.0
        self._last_hp: int | None = None
        self._last_mp: int | None = None
        self._slot_counts: dict[str, int] = {}
        self._slot_charged = {}
        self._shortcut_baseline: dict[str, int] = {}
        self._shortcut_observed: dict[str, int] = {}
        self._slot_candidates: dict[str, tuple[int, int, float]] = {}
        self._pending_potions: list[_PendingPotion] = []

    def record_pickup_lines(self, lines: Iterable[object], now: float | None = None) -> int:
        observations: list[MesosObservation] = []
        for line in lines:
            text, y = _line_text_and_y(line)
            amount = parse_mesos_amount(text)
            if amount is not None:
                observations.append(MesosObservation(amount, y))
        return self._mesos.update(observations, now)

    def record_quick_slot_counts(self, counts: dict[str, int], now: float | None = None) -> int:
        """Register potion use from observed shortcut quantity decreases.

        A quantity increase is only a new baseline (for example, a dropped
        stack or a restock).  It never offsets a previous cost and it never
        creates income.  A decrease must be visible in two consecutive OCR
        samples and must be a plausible-sized change.  This makes the
        shortcut bar the authoritative source for potion consumption while
        filtering the transient digit loss that often occurs during drinking.
        """
        timestamp = time.monotonic() if now is None else now
        valid_counts = {slot_id: count for slot_id, count in counts.items() if count >= 0}
        if not valid_counts:
            return 0
        # Keep direct users of EconomyTracker safe too.  The overlay normally
        # calls prime_quick_slot_counts explicitly so the UI can show that the
        # first read was a baseline, but a tracker used on its own should have
        # the same no-charge first-read behavior.
        if not self._shortcut_baseline:
            self.prime_quick_slot_counts(valid_counts)
            return 0
        uses = 0
        for slot_id, current in valid_counts.items():
            previous = self._slot_counts.get(slot_id)
            if previous is None:
                self._slot_counts[slot_id] = current
                self._shortcut_observed[slot_id] = current
                self._slot_candidates.pop(slot_id, None)
                continue
            if current >= previous:
                # A restock/pickup is a new baseline.  Returning to the old
                # quantity also cancels a one-frame low OCR candidate.
                self._slot_counts[slot_id] = current
                self._shortcut_observed[slot_id] = current
                if (
                    current > previous
                    and self._slot_charged.get(slot_id, 0) == 0
                    and _is_probable_leading_digit_recovery(previous, current)
                ):
                    # The first baseline may have lost a leading digit.  Do
                    # not leave the UI showing a false initial quantity.
                    self._shortcut_baseline[slot_id] = current
                self._slot_candidates.pop(slot_id, None)
                continue

            if _is_probable_truncated_count(previous, current):
                # Keep the trusted and displayed values stable; a suffix such
                # as 6/16 is a crop error, not a real drop from 116.
                self._slot_candidates.pop(slot_id, None)
                continue

            drop = previous - current
            if drop > MAX_SLOT_DROP_PER_SCAN:
                # Do not turn a missing digit (e.g. 1180 -> 180) into hundreds
                # of potion uses.  Keep the last trusted baseline until OCR
                # produces a plausible stable value.
                self._slot_candidates.pop(slot_id, None)
                continue

            candidate = self._slot_candidates.get(slot_id)
            candidate_is_recent = (
                candidate is not None
                and timestamp - candidate[2] <= SLOT_CANDIDATE_MAX_GAP_SECONDS
            )
            if candidate_is_recent and current <= candidate[0]:
                # A held potion key can produce 1180 -> 1179 -> 1178 rather
                # than repeating 1179.  A monotonic lower sequence is still
                # two-frame evidence; count from the last trusted quantity.
                confirmations = candidate[1] + 1
            else:
                confirmations = 1
            self._slot_candidates[slot_id] = (current, confirmations, timestamp)
            if confirmations >= SLOT_CONFIRMATIONS_REQUIRED:
                self._slot_candidates.pop(slot_id, None)
                self._slot_counts[slot_id] = current
                self._shortcut_observed[slot_id] = current
                confirmed_drop = previous - current
                if confirmed_drop <= MAX_SLOT_DROP_PER_SCAN:
                    uses += self._commit_slot_drop(slot_id, confirmed_drop, timestamp)
        return uses

    def _commit_slot_drop(self, slot_id: str, count: int, now: float) -> int:
        slot = self._by_slot.get(slot_id)
        if slot is None or count <= 0:
            return 0
        self._slot_charged[slot_id] = self._slot_charged.get(slot_id, 0) + count
        self._register_potion_use(slot, count, now)
        return count

    def reconcile_quick_slot_counts(self, counts: dict[str, int], now: float | None = None) -> int:
        """Reconcile the final visible inventory before pause/restart/close.

        Normal tracking intentionally waits for stable per-scan evidence.  A
        final OCR result is different: the user expects the session total to
        match the actual shortcut quantities at the moment they ended it.  A
        stable final quantity can therefore commit the portion of the
        baseline-to-final drop that earlier OCR frames missed, while
        ``_slot_charged`` prevents double counting already confirmed drinks.
        """
        timestamp = time.monotonic() if now is None else now
        valid_counts = {slot_id: count for slot_id, count in counts.items() if count >= 0}
        if not valid_counts:
            return 0
        if not self._shortcut_baseline:
            self.prime_quick_slot_counts(valid_counts)
            return 0
        uses = 0
        for slot_id, current in valid_counts.items():
            baseline = self._shortcut_baseline.get(slot_id)
            if baseline is None:
                self._slot_counts[slot_id] = current
                self._shortcut_observed[slot_id] = current
                self._shortcut_baseline[slot_id] = current
                self._slot_charged[slot_id] = 0
                continue
            if current >= baseline:
                # A restock is not consumption.  Keep the latest visible
                # quantity as the reference for future scans.
                self._slot_counts[slot_id] = current
                self._shortcut_observed[slot_id] = current
                self._slot_candidates.pop(slot_id, None)
                continue
            if _is_probable_truncated_count(self._slot_counts.get(slot_id, baseline), current):
                continue
            total_drop = baseline - current
            already_charged = self._slot_charged.get(slot_id, 0)
            missing = total_drop - already_charged
            if not (0 < missing <= MAX_SLOT_RECONCILE_DROP):
                continue
            self._slot_counts[slot_id] = current
            self._shortcut_observed[slot_id] = current
            self._slot_candidates.pop(slot_id, None)
            uses += self._commit_slot_drop(slot_id, missing, timestamp)
        return uses

    def _register_potion_use(self, slot: PotionSlotConfig, count: int = 1, now: float | None = None) -> None:
        self._potion_uses += count
        cost = max(0, slot.cost) * count
        self._potion_cost += cost
        if slot.kind == "hp":
            self._hp_potion_uses += count
            self._hp_potion_cost += cost
        elif slot.kind == "mp":
            self._mp_potion_uses += count
            self._mp_potion_cost += cost
        else:
            # A shared HP/MP potion is kept separate instead of charging its
            # price to both categories.  This preserves the invariant that
            # category costs add up to the total cost.
            self._shared_potion_uses += count
            self._shared_potion_cost += cost
        self._potion_breakdown[slot.name or slot.slot] += count
        expires_at = (time.monotonic() if now is None else now) + POTION_PENDING_SECONDS
        for _ in range(min(count, 100)):
            self._pending_potions.append(_PendingPotion(slot, expires_at))

    def record_stats(
        self, hp_cur: int | None, mp_cur: int | None, now: float | None = None
    ) -> tuple[int, int]:
        """Record current HP/MP and return the observed upward deltas.

        The live session uses these deltas as recovery evidence as well.  A
        large damage read can be held by the OCR noise guard while a later
        potion heal is still visible; returning the evidence lets the rate
        tracker account for that otherwise hidden damage.
        """
        timestamp = time.monotonic() if now is None else now
        self._pending_potions = [item for item in self._pending_potions if item.expires_at >= timestamp]
        hp_recovery = 0
        mp_recovery = 0
        if hp_cur is not None and self._last_hp is not None and hp_cur > self._last_hp:
            hp_recovery = hp_cur - self._last_hp
            self._confirm_candidate_from_recovery("hp", hp_recovery, timestamp)
            self._record_recovery("hp", hp_recovery, timestamp)
        if mp_cur is not None and self._last_mp is not None and mp_cur > self._last_mp:
            mp_recovery = mp_cur - self._last_mp
            self._confirm_candidate_from_recovery("mp", mp_recovery, timestamp)
            self._record_recovery("mp", mp_recovery, timestamp)
        if hp_cur is not None:
            self._last_hp = hp_cur
        if mp_cur is not None:
            self._last_mp = mp_cur
        return hp_recovery, mp_recovery

    def _confirm_candidate_from_recovery(self, kind: str, amount: int, now: float) -> None:
        """Confirm a one-frame slot drop when the matching heal is visible.

        The shortcut OCR may see ``2037 -> 2036`` once and then miss the next
        frame.  When the configured MP potion's recovery appears at the same
        time, that is stronger evidence than waiting for a repeated quantity.
        Only configured slots with a positive recovery value use this path;
        blank/unknown slots remain protected from natural-regeneration false
        positives.
        """
        if amount <= 0:
            return
        exact: list[tuple[str, int, int]] = []
        for slot_id, (current, _confirmations, candidate_at) in self._slot_candidates.items():
            if now - candidate_at > POTION_PENDING_SECONDS:
                continue
            previous = self._slot_counts.get(slot_id)
            slot = self._by_slot.get(slot_id)
            if previous is None or slot is None or current >= previous:
                continue
            if slot.kind not in (kind, "both"):
                continue
            expected = slot.recovery or self._default_recovery(kind)
            if expected > 0 and _recovery_matches(expected, amount):
                exact.append((slot_id, previous - current, current))
        if not exact:
            return
        slot_id, drop, current = exact[0]
        self._slot_candidates.pop(slot_id, None)
        self._slot_counts[slot_id] = current
        self._commit_slot_drop(slot_id, drop, now)

    def _record_recovery(self, kind: str, amount: int, now: float) -> None:
        pending_index = self._matching_pending(kind, amount)
        if pending_index is not None:
            pending = self._pending_potions[pending_index]
            self._add_potion_recovery(kind, amount)
            # A shared potion may produce two recovery observations (one HP,
            # one MP) for the same use.  Keep the pending marker until both
            # kinds have matched so the second observation is classified as
            # potion recovery without creating a second potion use.
            if pending.config.kind == "both":
                pending.matched_kinds.add(kind)
                if pending.matched_kinds >= {"hp", "mp"}:
                    self._pending_potions.pop(pending_index)
            else:
                self._pending_potions.pop(pending_index)
            return

        # No shortcut quantity drop means no potion use and no potion cost.
        # The same rule prevents a natural regen tick or a healing skill from
        # being converted into a drink merely because its value resembles a
        # configured potion amount.
        self._add_non_potion_recovery(kind, amount)

    def _matching_pending(self, kind: str, amount: int) -> int | None:
        fallback: int | None = None
        for index, pending in enumerate(self._pending_potions):
            if pending.config.kind not in (kind, "both"):
                continue
            if kind in pending.matched_kinds:
                continue
            if _recovery_matches(pending.config.recovery or self._default_recovery(kind), amount):
                return index
            # A confirmed shortcut quantity drop is stronger evidence than a
            # configured heal amount: game recovery may be capped by max HP,
            # delayed across OCR frames, or the settings value may be blank.
            # Keep this as a second pass so an exact configured match wins.
            if fallback is None:
                fallback = index
        return fallback

    def _default_recovery(self, kind: str) -> int:
        return self._default_recovery_hp if kind == "hp" else self._default_recovery_mp

    def _add_potion_recovery(self, kind: str, amount: int) -> None:
        if kind == "hp":
            self._hp_recovery_potion += amount
        else:
            self._mp_recovery_potion += amount

    def _add_non_potion_recovery(self, kind: str, amount: int) -> None:
        if kind == "hp":
            self._hp_recovery_natural += amount
            self._hp_recovery_savings += amount * HP_RECOVERY_MESOS_PER_POINT
        else:
            self._mp_recovery_natural += amount
            self._mp_recovery_savings += amount * MP_RECOVERY_MESOS_PER_POINT

    @property
    def snapshot(self) -> EconomySnapshot:
        return EconomySnapshot(
            mesos=self._mesos.total,
            mesos_events=self._mesos.events,
            potion_uses=self._potion_uses,
            potion_cost=self._potion_cost,
            hp_potion_uses=self._hp_potion_uses,
            hp_potion_cost=self._hp_potion_cost,
            mp_potion_uses=self._mp_potion_uses,
            mp_potion_cost=self._mp_potion_cost,
            shared_potion_uses=self._shared_potion_uses,
            shared_potion_cost=self._shared_potion_cost,
            hp_recovery_natural=self._hp_recovery_natural,
            hp_recovery_potion=self._hp_recovery_potion,
            mp_recovery_natural=self._mp_recovery_natural,
            mp_recovery_potion=self._mp_recovery_potion,
            hp_recovery_savings=round(self._hp_recovery_savings, 1),
            mp_recovery_savings=round(self._mp_recovery_savings, 1),
            potion_breakdown=dict(self._potion_breakdown),
            shortcut_baseline=dict(self._shortcut_baseline),
            # Current is the latest trusted quantity.  A raw OCR suffix such
            # as 6 from 116 must never leak into the UI while a decrease is
            # being validated.
            shortcut_current=dict(self._slot_counts),
            shortcut_baseline_ready=bool(self._shortcut_baseline),
        )


def _recovery_matches(expected: int, actual: int) -> bool:
    if expected <= 0 or actual <= 0:
        return False
    tolerance = max(RECOVERY_MIN_TOLERANCE, int(expected * RECOVERY_MATCH_TOLERANCE))
    return abs(expected - actual) <= tolerance
