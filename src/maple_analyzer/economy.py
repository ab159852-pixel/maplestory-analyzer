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

from .regions import (
    MAX_SHORTCUT_QUANTITY,
    MAX_SHORTCUT_SINGLE_SAMPLE_DROP,
    SHORTCUT_SLOT_BOXES,
)
from .settings import PotionSlotConfig

_MESOS_RE = re.compile(r"[+]?(\d[\d,]*)")
_INTEGER_RE = re.compile(r"\d[\d,]*")
RECOVERY_MATCH_TOLERANCE = 0.15
RECOVERY_MIN_TOLERANCE = 3
# The shortcut OCR and status OCR run on separate worker cadences.  A drink
# can therefore be confirmed after the status frame that contains its heal;
# keep the marker alive long enough for that frame to arrive.
POTION_PENDING_SECONDS = 5.0
# A lower value must survive two samples before it becomes a provisional
# consumption event.  This keeps a single redraw from charging immediately,
# while allowing a later higher quantity to correct an overcharge.
SLOT_CONFIRMATIONS_REQUIRED = 2
# Enhanced full-bar retries can take longer than one auxiliary interval on a
# CPU-only machine.  Keep a one-frame lower candidate through that delay so a
# later identical/lower read can still confirm it.
SLOT_CANDIDATE_MAX_GAP_SECONDS = 5.0
# Keep the old name as the boundary between ordinary and bulk drops for
# compatibility with diagnostics/tests. It is no longer a hard maximum: a
# genuine 5-10 bottle decrease is admitted by the time/multi-frame gates.
MAX_SLOT_DROP_PER_SCAN = MAX_SHORTCUT_SINGLE_SAMPLE_DROP
SLOT_CORRECTION_CONFIRMATIONS_REQUIRED = 2
# An increase is not a potion event.  The live session does not refill slots,
# so upward changes are rejected by default.  The optional restock path below
# remains available for isolated callers/tests that explicitly need it.
SLOT_INCREASE_CONFIRMATIONS_REQUIRED = 3
# During a live test the user does not refill shortcut slots.  Any upward
# quantity change is therefore treated as an OCR artifact and never replaces
# the trusted inventory baseline.
MAX_IN_SESSION_RESTOCK_DELTA = 5
# Retained for compatibility with diagnostic callers of the old rate helper.
# Live accounting is now reversible rather than rate-blocked.
POTION_MIN_INTERVAL_SECONDS = 0.3
POTION_RATE_TOLERANCE_SECONDS = 0.06
# HP/MP OCR can briefly lose a leading digit or read a visual effect as part
# of the status value. Small changes are useful recovery evidence immediately,
# but a large jump must be seen twice before it affects natural-recovery
# savings. Configured potion heals are accepted immediately when their exact
# pending amount matches.
RECOVERY_OUTLIER_FRACTION = 0.35
RECOVERY_MIN_OUTLIER_THRESHOLD = 120
RECOVERY_CANDIDATE_MAX_GAP_SECONDS = 2.0
RESOURCE_MAX_CHANGE_FACTOR = 2.0
# HP/MP bar flashes are an optional third signal.  They are intentionally not
# used as a standalone cost source: a shortcut quantity decrease is still
# required.  A flash may, however, confirm a one-frame quantity drop when the
# next auxiliary OCR frame is missed. Keep this close to the independent
# 0.2/0.3s worker cadence so a late OCR artefact cannot borrow an old flash.
BAR_FLASH_WINDOW_SECONDS = 1.25
# Bulk confirmation may arrive after several drinks and several OCR frames.
# Keep edge-triggered flashes long enough to corroborate a multi-bottle drop,
# without making them a standalone source of potion uses.
BULK_BAR_FLASH_WINDOW_SECONDS = POTION_PENDING_SECONDS
MAX_RECENT_BAR_FLASHES = 32
# A final inventory reconciliation may cover a long interval.  The quantity
# itself is already bounded to four digits, so do not add a smaller arbitrary
# drop ceiling that would lose a legitimate near-empty stack.
MAX_SLOT_RECONCILE_DROP = MAX_SHORTCUT_QUANTITY
HP_RECOVERY_MESOS_PER_POINT = 1.2
MP_RECOVERY_MESOS_PER_POINT = 2.1
MESOS_Y_MATCH_PX = 52.0


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
    digit_translation = str.maketrans(
        "０１２３４５６７８９＋，",
        "0123456789+,",
    )
    compact = re.sub(r"\s+", "", str(text)).translate(digit_translation)
    compact = compact.replace(",", "")
    # The OCR model occasionally emits simplified characters or confuses
    # 楓 with 椋 in the small pickup toast.
    compact = (
        compact.replace("楓币", "楓幣")
        .replace("枫幣", "楓幣")
        .replace("枫币", "楓幣")
        .replace("椋幣", "楓幣")
        .replace("椋币", "楓幣")
        .replace("楓略", "楓幣")
        .replace("椋略", "楓幣")
        .replace("楓弊", "楓幣")
        .replace("椋弊", "楓幣")
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


def mesos_text_needs_full_detection(text: str) -> bool:
    """Return whether a parsed mesos candidate needs a wider OCR check.

    Recognition-only row OCR is intentionally cheap, but it can return a
    plausible amount after losing the distinctive ``楓幣`` marker (for
    example ``獲取。(+14)`` when the actual toast says ``+144``).  Treat that
    fallback as a candidate only: the caller should confirm it with the full
    notification-feed detector before handing it to the event tracker.  A
    candidate with a normalized 楓幣 marker is structurally stronger and can
    stay on the fast path.
    """
    if parse_mesos_amount(text) is None:
        return True
    compact = re.sub(r"\s+", "", str(text))
    compact = (
        compact.replace("楓币", "楓幣")
        .replace("枫幣", "楓幣")
        .replace("枫币", "楓幣")
        .replace("椋幣", "楓幣")
        .replace("椋币", "楓幣")
        .replace("楓略", "楓幣")
        .replace("椋略", "楓幣")
        .replace("楓弊", "楓幣")
        .replace("椋弊", "楓幣")
    )
    return "楓幣" not in compact


def parse_slot_count(text: str) -> int | None:
    """Read one valid four-digit-or-less shortcut quantity.

    OCR can merge a neighbouring cell or attach an unrelated UI number to the
    quantity.  The game domain has a strict upper bound, so a five-digit result
    is invalid input rather than a plausible inventory value.
    """
    normalized = str(text).translate(str.maketrans("０１２３４５６７８９，", "0123456789,"))
    matches = _INTEGER_RE.findall(normalized.replace(",", ""))
    if not matches:
        return None
    try:
        value = int(matches[-1])
    except ValueError:
        return None
    return value if 0 <= value <= MAX_SHORTCUT_QUANTITY else None


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
    # Latest plausible OCR observation. This may differ from
    # ``shortcut_current`` for one or two frames while the economy validator
    # confirms that a decrease is a real drink; it is display-only and must
    # never be used to calculate cost.
    shortcut_observed: dict[str, int] = field(default_factory=dict)
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
        allow_in_session_restock: bool = False,
    ) -> None:
        self._slots: list[PotionSlotConfig] = []
        self._by_slot: dict[str, PotionSlotConfig] = {}
        self._default_recovery_hp = max(0, int(default_recovery_hp))
        self._default_recovery_mp = max(0, int(default_recovery_mp))
        self._allow_in_session_restock = bool(allow_in_session_restock)
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
            self._slot_last_accepted_at.clear()
            self._slot_last_sample_at.clear()
            self._shortcut_baseline.clear()
            self._shortcut_observed.clear()
            self._slot_candidates.clear()
            self._pending_potions.clear()
            # Existing totals remain part of the session history, but their
            # old slot links must not be used to reverse a newly configured
            # potion row.
            self._potion_recovery_events.clear()
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
        self._slot_last_accepted_at.clear()
        self._slot_last_sample_at.clear()
        self._shortcut_baseline.clear()
        self._shortcut_observed.clear()
        self._slot_candidates.clear()
        self._pending_potions.clear()
        # No charged slot drop can be rolled back across a new
        # start/resume baseline, so discard only the attribution links while
        # keeping the already accumulated recovery totals.
        self._potion_recovery_events.clear()
        # Auxiliary monitoring is disabled while paused/stopped, but the
        # status worker keeps producing frames. Re-anchor HP/MP when it is
        # enabled again so time spent outside the session is never counted as
        # natural recovery on the first resumed frame.
        self._last_hp = None
        self._last_mp = None
        self._recovery_candidates.clear()
        # A flash captured before Start/Resume belongs to the old visual
        # baseline.  Carrying it over could confirm an unrelated OCR drop in
        # the new interval, so the evidence must have the same boundary as
        # the shortcut inventory baseline.
        self._recent_bar_flashes = {"hp": [], "mp": []}

    def prime_quick_slot_counts(
        self, counts: dict[str, int], now: float | None = None
    ) -> None:
        """Set the first visible inventory quantities without counting drops."""
        self.begin_quick_slot_baseline()
        timestamp = time.monotonic() if now is None else now
        for slot_id, count in counts.items():
            if isinstance(count, int) and 0 <= count <= MAX_SHORTCUT_QUANTITY:
                self._shortcut_baseline[slot_id] = count
                self._slot_counts[slot_id] = count
                self._shortcut_observed[slot_id] = count
                self._slot_charged[slot_id] = 0
                self._slot_last_accepted_at[slot_id] = timestamp
                self._slot_last_sample_at[slot_id] = timestamp

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
        self._resource_max: dict[str, int] = {}
        self._recovery_candidates: dict[str, tuple[int, float]] = {}
        self._slot_counts: dict[str, int] = {}
        self._slot_charged = {}
        self._slot_last_accepted_at: dict[str, float] = {}
        self._slot_last_sample_at: dict[str, float] = {}
        self._shortcut_baseline: dict[str, int] = {}
        self._shortcut_observed: dict[str, int] = {}
        self._slot_candidates: dict[str, tuple[int, int, float]] = {}
        self._pending_potions: list[_PendingPotion] = []
        # Keep the source slot for potion-recovery totals.  If an OCR drop is
        # later corrected upward, its already matched HP/MP recovery must be
        # reversible along with the potion cost.
        self._potion_recovery_events: dict[str, list[tuple[str, int]]] = {}
        self._recent_bar_flashes: dict[str, list[float]] = {"hp": [], "mp": []}

    def record_bar_flash(
        self, resources: Iterable[str], now: float | None = None
    ) -> None:
        """Remember conservative HP/MP bar-flash evidence.

        The bar detector is edge-triggered and emits at most one resource per
        flash.  Keeping the evidence briefly bridges the independent status
        (0.3s) and shortcut (0.2s) worker cadences.  It never creates a
        potion use by itself; the quantity must still decrease in a valid
        configured slot.
        """
        timestamp = time.monotonic() if now is None else now
        if isinstance(resources, str):
            resources = (resources,)
        resource_names = {str(value).lower() for value in resources}
        for kind in ("hp", "mp"):
            if kind not in resource_names:
                continue
            values = self._recent_bar_flashes.setdefault(kind, [])
            values.append(timestamp)
            self._recent_bar_flashes[kind] = [
                value for value in values
                if timestamp - value <= BULK_BAR_FLASH_WINDOW_SECONDS
            ][-MAX_RECENT_BAR_FLASHES:]

    def _flash_kind_for_slot(self, slot: PotionSlotConfig) -> tuple[str, ...]:
        if slot.kind == "hp":
            return ("hp",)
        if slot.kind == "mp":
            return ("mp",)
        return ("hp", "mp")

    def _consume_matching_bar_flash(
        self, slot: PotionSlotConfig, timestamp: float
    ) -> bool:
        """Consume one nearby flash, if it matches the potion's type."""
        for kind in self._flash_kind_for_slot(slot):
            values = self._recent_bar_flashes.get(kind, [])
            eligible = [
                (index, value)
                for index, value in enumerate(values)
                if abs(timestamp - value) <= BAR_FLASH_WINDOW_SECONDS
            ]
            if not eligible:
                continue
            index, _value = min(eligible, key=lambda pair: abs(timestamp - pair[1]))
            values.pop(index)
            return True
        return False

    def _has_matching_bar_flash(
        self, slot: PotionSlotConfig, timestamp: float
    ) -> bool:
        return any(
            any(abs(timestamp - value) <= BAR_FLASH_WINDOW_SECONDS for value in self._recent_bar_flashes.get(kind, ()))
            for kind in self._flash_kind_for_slot(slot)
        )

    def _matching_bar_flash_count(
        self,
        slot: PotionSlotConfig,
        timestamp: float,
        *,
        since: float | None = None,
    ) -> int:
        """Count recent flashes that can corroborate this slot's drop."""
        lower_bound = timestamp - BULK_BAR_FLASH_WINDOW_SECONDS
        if since is not None:
            lower_bound = max(lower_bound, since)
        return sum(
            1
            for kind in self._flash_kind_for_slot(slot)
            for value in self._recent_bar_flashes.get(kind, ())
            if lower_bound <= value <= timestamp
        )

    def _consume_matching_bar_flashes(
        self,
        slot: PotionSlotConfig,
        timestamp: float,
        count: int,
        *,
        since: float | None = None,
    ) -> int:
        """Consume up to ``count`` matching flash markers after a commit."""
        remaining = max(0, int(count))
        consumed = 0
        lower_bound = timestamp - BULK_BAR_FLASH_WINDOW_SECONDS
        if since is not None:
            lower_bound = max(lower_bound, since)
        for kind in self._flash_kind_for_slot(slot):
            if remaining <= 0:
                break
            values = self._recent_bar_flashes.get(kind, [])
            eligible = [
                index
                for index, value in enumerate(values)
                if lower_bound <= value <= timestamp
            ]
            for index in reversed(eligible[:remaining]):
                values.pop(index)
                remaining -= 1
                consumed += 1
                if remaining <= 0:
                    break
        return consumed

    def record_pickup_lines(self, lines: Iterable[object], now: float | None = None) -> int:
        observations: list[MesosObservation] = []
        for line in lines:
            text, y = _line_text_and_y(line)
            amount = parse_mesos_amount(text)
            if amount is not None:
                observations.append(MesosObservation(amount, y))
        return self._mesos.update(observations, now)

    def record_quick_slot_counts(
        self,
        counts: dict[str, int],
        now: float | None = None,
        *,
        immediate: bool = False,
    ) -> int:
        """Register potion use from observed shortcut quantity decreases.

    In the live session, upward quantity changes are not potion events.  A
    decrease normally must be corroborated by a second nearby lower OCR sample
    (the value may step down again during a rapid drink animation) before it
    becomes a provisional charge.  The live monitor can opt into ``immediate``
    mode after the numeric colour/threshold consensus has already passed: the
    cost is then visible on the first valid lower frame, while the existing
    stable upward-correction path can roll it back if that frame was OCR noise.
    A later stable increase toward the session baseline can always reverse a
    provisional charge when the lower value was an OCR error.
        """
        timestamp = time.monotonic() if now is None else now
        valid_counts = {
            slot_id: count
            for slot_id, count in counts.items()
            if isinstance(count, int) and 0 <= count <= MAX_SHORTCUT_QUANTITY
        }
        if not valid_counts:
            return 0
        # Keep direct users of EconomyTracker safe too.  The overlay normally
        # calls prime_quick_slot_counts explicitly so the UI can show that the
        # first read was a baseline, but a tracker used on its own should have
        # the same no-charge first-read behavior.
        if not self._shortcut_baseline:
            self.prime_quick_slot_counts(valid_counts, now=timestamp)
            return 0
        uses = 0
        for slot_id, current in valid_counts.items():
            # Keep the latest OCR timestamp for diagnostics/reconciliation,
            # but do not use it as the consumption-rate origin. OCR may see
            # the same pending quantity every 0.2-0.3s while the game only
            # changed the stack once; measuring from that last frame made a
            # real 0.3-0.7s potion change stay blocked forever.
            self._slot_last_sample_at[slot_id] = timestamp
            if slot_id not in self._shortcut_baseline:
                # One configured cell can be temporarily unreadable while a
                # neighbouring cell is already stable. Establish a baseline
                # per slot when that cell first becomes visible; otherwise a
                # later drop in that slot would have no reference quantity
                # and could never be charged.
                self._shortcut_baseline[slot_id] = current
                self._slot_counts[slot_id] = current
                self._shortcut_observed[slot_id] = current
                self._slot_charged[slot_id] = 0
                self._slot_last_accepted_at[slot_id] = timestamp
                self._slot_candidates.pop(slot_id, None)
                continue
            previous = self._slot_counts.get(slot_id)
            if previous is None:
                self._slot_counts[slot_id] = current
                self._shortcut_observed[slot_id] = current
                self._slot_candidates.pop(slot_id, None)
                continue
            if current == previous:
                # A stable return to the trusted value cancels either kind of
                # one-frame OCR candidate.
                self._slot_counts[slot_id] = current
                self._shortcut_observed[slot_id] = current
                self._slot_candidates.pop(slot_id, None)
                continue

            if current > previous:
                # Refilling is outside the live-test model. An upward change
                # is therefore never allowed to redefine the trusted value;
                # this blocks neighbouring-cell merges such as 86 -> 865.
                # A value between the last trusted count and the session
                # baseline is different: it is a correction to an earlier
                # provisional lower OCR read, so the matching charge is
                # rolled back after the same confirmation gate.  This is what
                # makes 100 -> 40 -> 80 settle at 20 bottles, not 60.
                if not self._allow_in_session_restock:
                    baseline = self._shortcut_baseline.get(slot_id)
                    charged = self._slot_charged.get(slot_id, 0)
                    if (
                        baseline is None
                        or current > baseline
                        or charged <= 0
                    ):
                        self._slot_candidates.pop(slot_id, None)
                        continue
                    candidate = self._slot_candidates.get(slot_id)
                    candidate_is_recent = (
                        candidate is not None
                        and timestamp - candidate[2] <= SLOT_CANDIDATE_MAX_GAP_SECONDS
                    )
                    if candidate_is_recent and candidate[0] == current:
                        confirmations = candidate[1] + 1
                        candidate_started_at = candidate[2]
                    else:
                        confirmations = 1
                        candidate_started_at = timestamp
                    self._slot_candidates[slot_id] = (
                        current,
                        confirmations,
                        candidate_started_at,
                    )
                    self._shortcut_observed[slot_id] = current
                    if confirmations < SLOT_CORRECTION_CONFIRMATIONS_REQUIRED:
                        continue
                    rollback = min(
                        current - previous,
                        self._slot_charged.get(slot_id, 0),
                    )
                    self._slot_candidates.pop(slot_id, None)
                    if rollback <= 0:
                        continue
                    self._slot_counts[slot_id] = current
                    self._shortcut_observed[slot_id] = current
                    self._slot_last_accepted_at[slot_id] = timestamp
                    self._rollback_slot_drop(slot_id, rollback)
                    continue
                if current - previous > MAX_IN_SESSION_RESTOCK_DELTA:
                    self._slot_candidates.pop(slot_id, None)
                    continue
                candidate = self._slot_candidates.get(slot_id)
                candidate_is_recent = (
                    candidate is not None
                    and timestamp - candidate[2] <= SLOT_CANDIDATE_MAX_GAP_SECONDS
                )
                if candidate_is_recent and candidate[0] == current:
                    confirmations = candidate[1] + 1
                else:
                    confirmations = 1
                self._slot_candidates[slot_id] = (current, confirmations, timestamp)
                if confirmations < SLOT_INCREASE_CONFIRMATIONS_REQUIRED:
                    continue
                self._slot_candidates.pop(slot_id, None)
                self._slot_counts[slot_id] = current
                self._shortcut_observed[slot_id] = current
                self._slot_last_accepted_at[slot_id] = timestamp
                if (
                    self._slot_charged.get(slot_id, 0) == 0
                    and _is_probable_leading_digit_recovery(previous, current)
                ):
                    # The first baseline may have lost a leading digit.  Do
                    # not leave the UI showing a false initial quantity.
                    self._shortcut_baseline[slot_id] = current
                continue

            # A pending increase is not evidence for a decrease.  This
            # matters when one bad frame is followed by the real quantity.
            candidate = self._slot_candidates.get(slot_id)
            if candidate is not None and candidate[0] > previous:
                self._slot_candidates.pop(slot_id, None)

            drop = previous - current

            # Publish a plausible lower observation immediately so the UI can
            # prove that OCR saw a quantity change. The trusted quantity and
            # cost ledger still wait for the confirmation/rate gates below.
            self._shortcut_observed[slot_id] = current

            candidate = self._slot_candidates.get(slot_id)
            candidate_is_recent = (
                candidate is not None
                and timestamp - candidate[2] <= SLOT_CANDIDATE_MAX_GAP_SECONDS
            )
            if candidate_is_recent and candidate[0] == current:
                # The same lower value in two frames is the strongest
                # confirmation signal.
                confirmations = candidate[1] + 1
                candidate_started_at = candidate[2]
            elif candidate_is_recent and current < candidate[0]:
                # A fast drink animation can expose a monotonic sequence such
                # as 1359 -> 1358 -> 1357 without ever repeating one number.
                # Treat that as one corroborated aggregate transition and
                # charge only the net drop from the last trusted quantity.
                # The later upward-correction path remains able to reverse an
                # OCR substitution if the sequence was not real.
                confirmations = candidate[1] + 1
                candidate_started_at = candidate[2]
            else:
                confirmations = 1
                candidate_started_at = timestamp
            self._slot_candidates[slot_id] = (
                current,
                confirmations,
                candidate_started_at,
            )
            slot = self._by_slot.get(slot_id)
            flash_confirmed = bool(
                slot is not None
                and drop == 1
                and self._has_matching_bar_flash(slot, timestamp)
            ) if slot is not None else False
            bulk_flash_confirmed = bool(
                slot is not None
                and drop > MAX_SLOT_DROP_PER_SCAN
                and self._matching_bar_flash_count(
                    slot,
                    timestamp,
                    since=self._slot_last_accepted_at.get(slot_id),
                ) >= drop
            ) if slot is not None else False
            # Every valid lower reading is provisional and reversible.  A
            # large decrease may be a real group of drinks, or it may be an
            # OCR frame such as 100 -> 40.  Do not discard either case here:
            # two identical frames make the observation billable, while a
            # later stable increase toward the session baseline rolls back
            # the overcharge.
            confirmations_required = 1 if immediate else SLOT_CONFIRMATIONS_REQUIRED
            if (
                confirmations >= confirmations_required
                or flash_confirmed
                or bulk_flash_confirmed
            ):
                confirmed_drop = previous - current
                self._slot_candidates.pop(slot_id, None)
                if slot is not None:
                    if bulk_flash_confirmed:
                        self._consume_matching_bar_flashes(
                            slot,
                            timestamp,
                            confirmed_drop,
                            since=self._slot_last_accepted_at.get(slot_id),
                        )
                    elif flash_confirmed:
                        self._consume_matching_bar_flash(slot, timestamp)
                self._slot_counts[slot_id] = current
                self._shortcut_observed[slot_id] = current
                self._slot_last_accepted_at[slot_id] = timestamp
                uses += self._commit_slot_drop(slot_id, confirmed_drop, timestamp)
        return uses

    def _allowed_drop_for_slot(
        self,
        slot_id: str,
        timestamp: float,
        *,
        reference_at: float | None = None,
    ) -> int:
        """Return the legacy rate estimate for diagnostic callers.

        ``_slot_last_sample_at`` advances on every OCR frame, including a
        repeated pending candidate. It is therefore not a valid rate-limit
        origin. ``reference_at`` remains available for compatibility with
        older callers. Live accounting intentionally does not use this value
        as a hard block; its ledger is corrected by later quantity increases.
        """
        last = (
            reference_at
            if reference_at is not None
            else self._slot_last_accepted_at.get(slot_id)
        )
        if last is None:
            last = self._slot_last_sample_at.get(slot_id)
        if last is None or timestamp < last:
            # Keep deterministic/unit-test callers that use synthetic times
            # before a real monotonic baseline backwards compatible.  Live
            # overlay calls always prime with the same monotonic clock.
            return MAX_SLOT_DROP_PER_SCAN
        elapsed = timestamp - last
        if elapsed + POTION_RATE_TOLERANCE_SECONDS < POTION_MIN_INTERVAL_SECONDS:
            return 0
        return max(1, int((elapsed + POTION_RATE_TOLERANCE_SECONDS) / POTION_MIN_INTERVAL_SECONDS))

    def _commit_slot_drop(self, slot_id: str, count: int, now: float) -> int:
        slot = self._by_slot.get(slot_id)
        if slot is None or count <= 0:
            return 0
        self._slot_charged[slot_id] = self._slot_charged.get(slot_id, 0) + count
        self._register_potion_use(slot, count, now)
        return count

    def _rollback_slot_drop(self, slot_id: str, count: int) -> int:
        """Reverse a previously charged OCR drop after a stable correction.

        Quantity OCR is allowed to be optimistic for display, but the cost
        ledger must be reversible when a later stable frame proves that a
        substituted digit was charged.  Recovery amounts matched to those
        provisional potion uses are rolled back at the same time.
        """
        slot = self._by_slot.get(slot_id)
        charged = self._slot_charged.get(slot_id, 0)
        if slot is None or count <= 0 or charged <= 0:
            return 0
        amount = min(count, charged)
        self._slot_charged[slot_id] = charged - amount
        if self._slot_charged[slot_id] <= 0:
            self._slot_charged.pop(slot_id, None)

        cost = max(0, slot.cost) * amount
        self._potion_uses = max(0, self._potion_uses - amount)
        self._potion_cost = max(0, self._potion_cost - cost)
        if slot.kind == "hp":
            self._hp_potion_uses = max(0, self._hp_potion_uses - amount)
            self._hp_potion_cost = max(0, self._hp_potion_cost - cost)
        elif slot.kind == "mp":
            self._mp_potion_uses = max(0, self._mp_potion_uses - amount)
            self._mp_potion_cost = max(0, self._mp_potion_cost - cost)
        else:
            self._shared_potion_uses = max(0, self._shared_potion_uses - amount)
            self._shared_potion_cost = max(0, self._shared_potion_cost - cost)

        # Keep at most the number of recovery events that can still be
        # explained by the slot's remaining charged uses. Any event that no
        # longer has a matching use is reclassified as natural/skill recovery
        # instead of disappearing from the recovery totals.
        self._trim_potion_recovery_to_uses(
            slot_id,
            self._slot_charged.get(slot_id, 0),
        )

        breakdown_key = slot.name or slot.slot
        remaining_breakdown = self._potion_breakdown.get(breakdown_key, 0) - amount
        if remaining_breakdown > 0:
            self._potion_breakdown[breakdown_key] = remaining_breakdown
        else:
            self._potion_breakdown.pop(breakdown_key, None)

        # Remove the newest pending recovery markers for this slot as well;
        # otherwise a false charge could still relabel the next natural heal as
        # a potion recovery after its cost has already been corrected.
        remaining = amount
        for index in range(len(self._pending_potions) - 1, -1, -1):
            if self._pending_potions[index].config.slot != slot_id:
                continue
            self._pending_potions.pop(index)
            remaining -= 1
            if remaining <= 0:
                break
        return amount

    def _trim_potion_recovery_to_uses(self, slot_id: str, max_uses: int) -> None:
        """Keep recovery attribution within the corrected use count.

        A shared HP/MP potion can produce one event for each resource, so the
        per-kind limit is intentionally applied independently.  Events that
        no longer fit the corrected count are moved to natural/skill recovery
        so the recovery total and saved-cost estimate remain conserved.
        """
        events = self._potion_recovery_events.get(slot_id)
        if not events:
            return
        max_uses = max(0, int(max_uses))
        for kind in ("hp", "mp"):
            matching_indices = [
                index
                for index, (event_kind, _amount) in enumerate(events)
                if event_kind == kind
            ]
            overflow = max(0, len(matching_indices) - max_uses)
            for index in reversed(matching_indices[-overflow:] if overflow else []):
                event_kind, event_amount = events[index]
                if event_kind != kind:
                    break
                events.pop(index)
                if kind == "hp":
                    self._hp_recovery_potion = max(
                        0, self._hp_recovery_potion - event_amount
                    )
                else:
                    self._mp_recovery_potion = max(
                        0, self._mp_recovery_potion - event_amount
                    )
                self._add_non_potion_recovery(kind, event_amount)
        if not events:
            self._potion_recovery_events.pop(slot_id, None)

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
        valid_counts = {
            slot_id: count
            for slot_id, count in counts.items()
            if isinstance(count, int) and 0 <= count <= MAX_SHORTCUT_QUANTITY
        }
        if not valid_counts:
            return 0
        if not self._shortcut_baseline:
            self.prime_quick_slot_counts(valid_counts, now=timestamp)
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
            trusted = self._slot_counts.get(slot_id, baseline)
            if (
                baseline > current > trusted
                and self._slot_charged.get(slot_id, 0) > 0
            ):
                rollback = min(
                    current - trusted,
                    self._slot_charged.get(slot_id, 0),
                )
                if rollback > 0:
                    self._slot_counts[slot_id] = current
                    self._shortcut_observed[slot_id] = current
                    self._slot_candidates.pop(slot_id, None)
                    self._slot_last_accepted_at[slot_id] = timestamp
                    self._rollback_slot_drop(slot_id, rollback)
                continue
            if current >= baseline:
                # The live test does not refill slots.  Ignore upward OCR
                # jumps at the session boundary too; otherwise one bad final
                # frame can redefine the baseline and corrupt the cost.
                if self._allow_in_session_restock:
                    self._slot_counts[slot_id] = current
                    self._shortcut_observed[slot_id] = current
                self._slot_candidates.pop(slot_id, None)
                continue
            total_drop = baseline - current
            already_charged = self._slot_charged.get(slot_id, 0)
            missing = total_drop - already_charged
            if not (0 < missing <= MAX_SLOT_RECONCILE_DROP):
                continue
            slot = self._by_slot.get(slot_id)
            accepted_before = self._slot_last_accepted_at.get(slot_id)
            bulk_flash_confirmed = bool(
                slot is not None
                and missing > MAX_SLOT_DROP_PER_SCAN
                and self._matching_bar_flash_count(
                    slot,
                    timestamp,
                    since=accepted_before,
                ) >= missing
            ) if slot is not None else False
            # The final visible quantity is the best correction point.  Do
            # not reject a large or suffix-shaped change here: any amount
            # already charged for that OCR path is subtracted from the
            # baseline difference, so the ledger converges to the latest
            # stable value rather than remaining permanently inflated.
            self._slot_counts[slot_id] = current
            self._shortcut_observed[slot_id] = current
            self._slot_candidates.pop(slot_id, None)
            self._slot_last_accepted_at[slot_id] = timestamp
            self._slot_last_sample_at[slot_id] = timestamp
            if bulk_flash_confirmed and slot is not None:
                self._consume_matching_bar_flashes(
                    slot,
                    timestamp,
                    missing,
                    since=accepted_before,
                )
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
        self,
        hp_cur: int | None,
        mp_cur: int | None,
        now: float | None = None,
        *,
        hp_max: int | None = None,
        mp_max: int | None = None,
    ) -> tuple[int, int]:
        """Record current HP/MP and return only trusted upward deltas.

        A raw OCR sequence such as ``1000 -> 100 -> 1000`` used to become a
        false 900 HP natural recovery. Keep a trusted resource baseline and
        hold large moves until the new value is corroborated. A configured
        potion heal with an exact pending recovery amount remains immediate,
        so this guard does not make potion statistics feel delayed.
        """
        timestamp = time.monotonic() if now is None else now
        self._pending_potions = [
            item for item in self._pending_potions
            if item.expires_at >= timestamp
        ]
        hp_recovery = self._record_resource_stat("hp", hp_cur, hp_max, timestamp)
        mp_recovery = self._record_resource_stat("mp", mp_cur, mp_max, timestamp)
        return hp_recovery, mp_recovery

    def _record_resource_stat(
        self,
        kind: str,
        current: int | None,
        maximum: int | None,
        timestamp: float,
    ) -> int:
        """Accept one HP/MP sample and return a trusted recovery delta."""
        if current is None or current < 0:
            return 0

        effective_max = self._update_resource_max(kind, maximum, current)
        if effective_max is not None and current > effective_max:
            # Never move the trusted baseline to an impossible value. This is
            # the stateful accounting boundary, so it remains protected even
            # if a compatibility OCR adapter bypasses parser.py's check.
            self._recovery_candidates.pop(kind, None)
            return 0

        previous = self._last_hp if kind == "hp" else self._last_mp
        if previous is None:
            self._set_last_resource(kind, current)
            self._recovery_candidates.pop(kind, None)
            return 0

        if current == previous:
            # Returning to the trusted value cancels a one-frame OCR outlier.
            self._recovery_candidates.pop(kind, None)
            return 0

        difference = current - previous
        threshold = self._recovery_outlier_threshold(previous, effective_max)

        candidate = self._recovery_candidates.get(kind)
        if (
            candidate is not None
            and timestamp - candidate[1] <= RECOVERY_CANDIDATE_MAX_GAP_SECONDS
            and abs(candidate[0] - previous) > threshold
            and abs(difference) <= threshold
        ):
            # A common OCR failure is ``1000 -> 100 -> 900`` (or the mirrored
            # high spike). The third value is close enough to the trusted
            # baseline to prove that the middle value was an outlier, not a
            # real damage event. Do not commit the small rebound as a fake
            # natural heal; keep the original baseline intact.
            self._recovery_candidates.pop(kind, None)
            return 0

        # An exact configured potion recovery can confirm a one-frame
        # shortcut drop even when it is larger than the natural-recovery band.
        if difference > 0:
            self._confirm_candidate_from_recovery(kind, difference, timestamp)
            if self._matching_pending(kind, difference) is not None:
                self._recovery_candidates.pop(kind, None)
                self._set_last_resource(kind, current)
                self._record_recovery(kind, difference, timestamp)
                return difference

        if abs(difference) <= threshold:
            self._recovery_candidates.pop(kind, None)
            self._set_last_resource(kind, current)
            if difference > 0:
                self._record_recovery(kind, difference, timestamp)
                return difference
            return 0

        candidate = self._recovery_candidates.get(kind)
        if (
            candidate is not None
            and timestamp - candidate[1] <= RECOVERY_CANDIDATE_MAX_GAP_SECONDS
            and abs(current - candidate[0]) <= threshold
        ):
            # The same large move survived a second sample. Accept the current
            # value; only an upward move contributes to recovery/savings.
            self._recovery_candidates.pop(kind, None)
            self._set_last_resource(kind, current)
            if difference > 0:
                self._record_recovery(kind, difference, timestamp)
                return difference
            return 0

        # Hold the first large move. If the next frame returns to the old
        # value, the equality branch above clears it and emits nothing. If it
        # persists, the next sample corroborates it.
        self._recovery_candidates[kind] = (current, timestamp)
        return 0

    def _update_resource_max(
        self, kind: str, maximum: int | None, current: int
    ) -> int | None:
        """Keep a plausible HP/MP maximum for stateful OCR validation."""
        known = self._resource_max.get(kind)
        if isinstance(maximum, int) and maximum > 0:
            if known is None or (
                known / RESOURCE_MAX_CHANGE_FACTOR <= maximum
                <= known * RESOURCE_MAX_CHANGE_FACTOR
            ):
                known = maximum
                self._resource_max[kind] = maximum
        # No max is available for a few compatibility callers/tests. Their
        # relative baseline still receives the outlier guard below; a future
        # structured HP/MP pair will add the stronger physical ceiling.
        return known

    def _recovery_outlier_threshold(
        self, previous: int, maximum: int | None
    ) -> int:
        reference = maximum if maximum is not None else max(previous, 1)
        return max(
            RECOVERY_MIN_OUTLIER_THRESHOLD,
            int(reference * RECOVERY_OUTLIER_FRACTION),
        )

    def _set_last_resource(self, kind: str, value: int) -> None:
        if kind == "hp":
            self._last_hp = value
        else:
            self._last_mp = value

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
            drop = previous - current
            expected = slot.recovery or self._default_recovery(kind)
            if expected > 0 and _recovery_matches(expected, amount):
                exact.append((slot_id, drop, current))
        if not exact:
            return
        slot_id, drop, current = exact[0]
        self._slot_candidates.pop(slot_id, None)
        self._slot_counts[slot_id] = current
        self._shortcut_observed[slot_id] = current
        self._slot_last_accepted_at[slot_id] = now
        self._slot_last_sample_at[slot_id] = now
        self._commit_slot_drop(slot_id, drop, now)

    def _record_recovery(self, kind: str, amount: int, now: float) -> None:
        pending_index = self._matching_pending(kind, amount)
        if pending_index is not None:
            pending = self._pending_potions[pending_index]
            self._add_potion_recovery(kind, amount, pending.config.slot)
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
        for index, pending in enumerate(self._pending_potions):
            if pending.config.kind not in (kind, "both"):
                continue
            if kind in pending.matched_kinds:
                continue
            expected = pending.config.recovery or self._default_recovery(kind)
            # Without an explicit recovery amount there is no safe way to
            # distinguish a potion heal from natural regeneration or a skill.
            # Requiring a configured amount prevents one false potion drop
            # from relabelling the next large HP/MP increase as potion output.
            if expected > 0 and _recovery_matches(expected, amount):
                return index
        return None

    def _default_recovery(self, kind: str) -> int:
        return self._default_recovery_hp if kind == "hp" else self._default_recovery_mp

    def _add_potion_recovery(
        self, kind: str, amount: int, slot_id: str | None = None
    ) -> None:
        if kind == "hp":
            self._hp_recovery_potion += amount
        else:
            self._mp_recovery_potion += amount
        if slot_id is not None:
            self._potion_recovery_events.setdefault(slot_id, []).append(
                (kind, amount)
            )

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
            # Current is the latest accepted quantity used by the reversible
            # ledger. ``shortcut_observed`` can temporarily be a newer OCR
            # candidate while a decrease or its upward correction is being
            # confirmed.
            shortcut_current=dict(self._slot_counts),
            shortcut_observed=dict(self._shortcut_observed),
            shortcut_baseline_ready=bool(self._shortcut_baseline),
        )


def _recovery_matches(expected: int, actual: int) -> bool:
    if expected <= 0 or actual <= 0:
        return False
    tolerance = max(RECOVERY_MIN_TOLERANCE, int(expected * RECOVERY_MATCH_TOLERANCE))
    return abs(expected - actual) <= tolerance
