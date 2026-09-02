r"""Passing and yielding, as a state machine with no physics in it.

The rest of this package is a hydrodynamics model. This module is
deliberately not: it strips the river down to **position along the course
and which side of the line you are on**, so the Head of the Charles
passing rules can be implemented and tested without a single force
calculation getting in the way.

That separation is the point. The rules are a discrete, stateful,
adversarial system -- who declared what, when, and whether the obligation
has been discharged -- and debugging a state machine through a 6-DOF
simulation is a bad way to spend an afternoon. Physics can be layered back
on later; correctness of the rules is settled here.

The rules being implemented
---------------------------
From the Head Of The Charles Regatta *Rules and Guidelines*:

* The **Passer** has the right to pass on the side of its choice, if and
  when a safe pass can be accomplished.
* The Passer **should declare** their desired line when they close to
  within **one boat length of open water**.
* The **Passee** must have yielded by the time the Passer has closed to
  within **half a boat length of open water**, given adequate room and
  time.
* Once the Passee has yielded the line indicated, **their obligation is
  satisfied**; both crews then own the safety of the rest of the pass.
* Failing to yield costs **60 s**, then **120 s**, then disqualification.

And the part that is not in the rulebook but is how it actually works: a
yield is not a momentary gesture. Once a crew has moved over they **stay**
there while the boats are overlapped. A Passer who draws alongside but
cannot finish the pass does not get to make them move again -- the Passee
holds that side until the Passer either clears ahead or drops back far
enough to start a fresh approach. If the passed crew later comes back and
starts closing, the roles reverse and the obligation is now the other
way.

"Open water" means clear water
------------------------------
Every distance in the rules is **open water**: the gap between the
Passee's stern and the Passer's bow, not between bow balls. Two boats with
their bow balls one length apart have *zero* open water and are already
overlapping. Getting this wrong makes every threshold in the rulebook fire
a full boat length too early, which is the single easiest way to
mis-implement head racing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["Side", "Phase", "PassingRules", "Entry", "Encounter",
           "HeadRace", "RaceLog"]

#: Length of an eight, m. Every rule distance is quoted in these.
EIGHT_LENGTH = 17.3
#: How far a yielding crew moves over, m -- "a boat width".
BOAT_WIDTH = 3.5


class Side(Enum):
    """Which side of the river, from the coxswain's point of view."""

    PORT = +1
    STARBOARD = -1

    @property
    def opposite(self) -> "Side":
        return Side.STARBOARD if self is Side.PORT else Side.PORT


class Phase(Enum):
    """Where a pair of boats is in the passing sequence."""

    #: Too far apart to matter.
    CLEAR = "clear"
    #: Passer inside one length of open water and has named a side.
    DECLARED = "declared"
    #: Passee has moved over; obligation discharged.
    YIELDED = "yielded"
    #: Hulls overlapping. The yield persists through this.
    OVERLAPPED = "overlapped"
    #: Passer is fully past. Nothing owed either way.
    COMPLETE = "complete"


@dataclass(frozen=True)
class PassingRules:
    """The thresholds, all in metres of **open water**."""

    boat_length: float = EIGHT_LENGTH
    #: Passer declares a side inside this much open water.
    declare_at: float = 1.0 * EIGHT_LENGTH
    #: Passee must have finished yielding by this much open water.
    yield_by: float = 0.5 * EIGHT_LENGTH
    #: How far over a yield moves the passed crew.
    yield_width: float = BOAT_WIDTH
    #: Open water at which a pass counts as complete and obligations end.
    clear_at: float = 1.0 * EIGHT_LENGTH
    #: Open water beyond which a stalled encounter resets, so a fresh
    #: approach has to be declared again.
    reset_at: float = 3.0 * EIGHT_LENGTH
    #: Escalating penalties for failing to yield, s.
    penalties: Tuple[float, ...] = (60.0, 120.0)
    #: How fast a crew can actually move over, m/s of lateral travel.
    yield_rate: float = 0.35
    #: Safety factor on the time a yield physically takes, used to decide
    #: whether a crew had "adequate room and time".
    grace: float = 1.3

    @property
    def adequate_time(self) -> float:
        """Seconds a crew must be given before a non-yield is an offence.

        The rulebook conditions the penalty on there being "adequate room
        and time to yield", and without that condition the model punishes
        the physically impossible: a role reversal begins with the boats
        already overlapped, the declaration fires at negative open water,
        and the yield threshold is breached on the very next step.  That
        produced ten penalties in a field where every crew was complying.
        """
        return self.grace * self.yield_width / max(self.yield_rate, 1e-6)

    def penalty_for(self, offence: int) -> Optional[float]:
        """Seconds for the ``offence``-th failure, or ``None`` for a DQ."""
        if offence <= len(self.penalties):
            return self.penalties[offence - 1]
        return None


@dataclass
class Entry:
    """One crew: a bow number, a start time, and a speed."""

    bow: int
    start: float                    # s after the first boat
    speed: float                    # m/s, constant in this model
    name: str = ""
    #: Lateral offset from the racing line, m. Positive to port.
    lateral: float = 0.0
    #: Where the crew *wants* to be, which a yield overrides.
    preferred_lateral: float = 0.0
    penalty: float = 0.0
    offences: int = 0
    disqualified: bool = False
    finished: Optional[float] = None
    #: Encounters in which this crew currently owes a yield, by other bow.
    owes: Dict[int, Side] = field(default_factory=dict)

    #: ``(station, lateral) -> m/s``.  When given, the crew's speed is
    #: read from the river at every step instead of being a constant, and
    #: :attr:`speed` becomes only the starting guess.  This is what
    #: couples the rules to the physics: a crew forced off the deep line
    #: by a yield slows down, which is the cost the rulebook does not
    #: mention and the state machine alone cannot see.
    speed_fn: object = None
    #: Integrated station, used when ``speed_fn`` is set.
    station: float = 0.0
    #: Metres of progress given up to yields, for accounting.
    lost_to_yield: float = 0.0

    def position(self, time: float) -> float:
        """Bow-ball station along the course at ``time``, m.

        Constant speed has a closed form and is kept for it: the rule
        tests want an exactly reproducible geometry, and integrating a
        constant would only add rounding.  With ``speed_fn`` the station
        is advanced by :meth:`HeadRace.run` instead and simply read here.
        """
        if self.speed_fn is not None:
            return self.station
        return max(0.0, (time - self.start)) * self.speed

    def current_speed(self) -> float:
        if self.speed_fn is None:
            return self.speed
        return float(self.speed_fn(self.station, self.lateral))

    @property
    def label(self) -> str:
        return self.name or ("bow %d" % self.bow)


@dataclass
class Encounter:
    """The state of one ordered pair, passer behind passee."""

    passer: int
    passee: int
    phase: Phase = Phase.CLEAR
    side: Optional[Side] = None       # side the passer chose to pass on
    #: Whether the passed crew is going to comply at all.  Drawn once,
    #: at declaration, so a crew cannot dither.
    complying: bool = True
    declared_at: Optional[float] = None
    #: Open water at the previous step, so "closing" can be distinguished
    #: from "close".
    previous_gap: Optional[float] = None
    yielded_at: Optional[float] = None
    completed_at: Optional[float] = None
    penalised: bool = False


@dataclass
class RaceLog:
    """Everything that happened, in order, as plain records."""

    events: List[dict] = field(default_factory=list)

    def add(self, time: float, kind: str, **detail) -> None:
        record = {"time": round(float(time), 2), "event": kind}
        record.update(detail)
        self.events.append(record)

    def of_kind(self, kind: str) -> List[dict]:
        return [e for e in self.events if e["event"] == kind]

    def __len__(self) -> int:
        return len(self.events)


class HeadRace:
    """A field of boats on a one-dimensional course, obeying the rules.

    No hydrodynamics, no steering dynamics, no river. Boats hold constant
    speed and move sideways at a finite rate when they have to yield. What
    is modelled carefully is *only* the rule logic.
    """

    def __init__(self, entries: List[Entry], length: float = 4800.0,
                 rules: PassingRules = None, compliance: float = 1.0,
                 seed: int = 0):
        self.entries = {entry.bow: entry for entry in entries}
        self.length = float(length)
        self.rules = rules or PassingRules()
        #: Probability that a crew yields when it should. Below one, crews
        #: miss yields and the penalty machinery gets exercised -- which is
        #: the only way to test that half of the rulebook.
        self.compliance = float(compliance)
        self.random = np.random.default_rng(seed)
        self.encounters: Dict[Tuple[int, int], Encounter] = {}
        self.log = RaceLog()

    # -- geometry ---------------------------------------------------------
    def open_water(self, ahead: Entry, behind: Entry, time: float) -> float:
        """Clear water between the leader's stern and the follower's bow.

        Negative means the hulls overlap. This is the quantity every rule
        threshold is written in, and it is **not** the bow-to-bow gap.
        """
        return (ahead.position(time) - self.rules.boat_length
                - behind.position(time))

    def _pair(self, passer: int, passee: int) -> Encounter:
        key = (passer, passee)
        if key not in self.encounters:
            self.encounters[key] = Encounter(passer=passer, passee=passee)
        return self.encounters[key]

    # -- the rules --------------------------------------------------------
    def _declare(self, encounter: Encounter, time: float) -> None:
        passer = self.entries[encounter.passer]
        passee = self.entries[encounter.passee]
        # The passer takes the side with more room, which is the only
        # sensible default when there is no river in the model.
        encounter.side = (Side.PORT if passee.lateral <= 0.0
                          else Side.STARBOARD)
        encounter.phase = Phase.DECLARED
        encounter.declared_at = time
        encounter.complying = bool(self.random.random() < self.compliance)
        if encounter.complying:
            passee.owes[passer.bow] = encounter.side
        self.log.add(time, "declare", passer=passer.bow, passee=passee.bow,
                     side=encounter.side.name,
                     open_water=round(self.open_water(passee, passer, time), 2))

    def _target_lateral(self, entry: Entry) -> float:
        """Where a crew should be, given every yield it currently owes.

        A crew can owe more than one at once in a tight field. Taking the
        most extreme demand rather than the sum keeps it on the water: two
        crews asking for a yield to port do not require two boat widths.
        """
        if not entry.owes:
            return entry.preferred_lateral
        demands = [-side.value * self.rules.yield_width
                   for side in entry.owes.values()]
        furthest = max(demands, key=abs)
        return furthest

    def _advance_lateral(self, entry: Entry, dt: float) -> None:
        target = self._target_lateral(entry)
        step = self.rules.yield_rate * dt
        gap = target - entry.lateral
        if abs(gap) <= step:
            entry.lateral = target
        else:
            entry.lateral += np.sign(gap) * step

    def _has_yielded(self, encounter: Encounter) -> bool:
        passee = self.entries[encounter.passee]
        wanted = -encounter.side.value * self.rules.yield_width
        return abs(passee.lateral - wanted) < 0.2

    def _penalise(self, encounter: Encounter, time: float) -> None:
        passee = self.entries[encounter.passee]
        passee.offences += 1
        encounter.penalised = True
        seconds = self.rules.penalty_for(passee.offences)
        if seconds is None:
            passee.disqualified = True
            self.log.add(time, "disqualified", crew=passee.bow,
                         offences=passee.offences)
        else:
            passee.penalty += seconds
            self.log.add(time, "penalty", crew=passee.bow, seconds=seconds,
                         offence=passee.offences, passer=encounter.passer)

    def _update_pair(self, passer: Entry, passee: Entry, time: float) -> None:
        gap = self.open_water(passee, passer, time)
        encounter = self._pair(passer.bow, passee.bow)

        if encounter.phase is Phase.COMPLETE:
            return

        # Stalled and drifted apart: the approach is over and a fresh one
        # has to be declared.  This is the "drop back and re-initiate"
        # case, and without it a crew would owe a yield for the rest of
        # the race after one brief approach.
        if gap > self.rules.reset_at and encounter.phase is not Phase.CLEAR:
            passee.owes.pop(passer.bow, None)
            self.log.add(time, "reset", passer=passer.bow, passee=passee.bow)
            self.encounters[(passer.bow, passee.bow)] = Encounter(
                passer=passer.bow, passee=passee.bow)
            return

        if encounter.phase is Phase.CLEAR:
            # **Closing, not merely near.**  A boat that is dropping back
            # is not attempting a pass, and letting it declare produced a
            # nonsense the coupled model exposed at once: a crew that had
            # just been overtaken immediately demanded a yield from the
            # crew that passed it, purely because the two were still
            # within a length of each other.  The rulebook is about a
            # boat "attempting to pass"; falling away is not that.
            #
            # A pair must be seen TWICE before it can declare.  Treating
            # the first sighting as closing re-admitted the same bug by
            # the back door: a role reversal creates a fresh encounter
            # that has no previous gap, so the boat being dropped got one
            # free declaration and pinned the crew that had just passed it
            # off its line for the rest of the race.
            closing = (encounter.previous_gap is not None
                       and gap < encounter.previous_gap - 1e-9)
            encounter.previous_gap = gap
            if gap <= self.rules.declare_at and closing:
                self._declare(encounter, time)
            return

        if encounter.phase is Phase.DECLARED:
            if self._has_yielded(encounter):
                encounter.phase = Phase.YIELDED
                encounter.yielded_at = time
                self.log.add(time, "yield", passer=passer.bow,
                             passee=passee.bow,
                             side=encounter.side.name,
                             open_water=round(gap, 2))
            elif (gap < self.rules.yield_by and not encounter.penalised
                  and encounter.declared_at is not None
                  and time - encounter.declared_at
                  >= self.rules.adequate_time):
                self._penalise(encounter, time)
            return

        if encounter.phase in (Phase.YIELDED, Phase.OVERLAPPED):
            if gap < 0.0:
                encounter.phase = Phase.OVERLAPPED
            # Complete only when the passer is genuinely clear AHEAD.
            if (passer.position(time) - passee.position(time)
                    > self.rules.boat_length + self.rules.clear_at):
                encounter.phase = Phase.COMPLETE
                encounter.completed_at = time
                passee.owes.pop(passer.bow, None)
                self.log.add(time, "complete", passer=passer.bow,
                             passee=passee.bow)

    # -- running it -------------------------------------------------------
    def run(self, dt: float = 0.5, limit: float = 3000.0) -> RaceLog:
        time = 0.0
        order = sorted(self.entries.values(), key=lambda e: e.bow)
        while time < limit:
            for entry in order:
                if entry.finished is None and entry.position(time) >= self.length:
                    entry.finished = time
                    self.log.add(time, "finish", crew=entry.bow)

            racing = [e for e in order
                      if e.finished is None and not e.disqualified
                      and e.position(time) > 0.0]
            for behind in racing:
                for ahead in racing:
                    if ahead.bow == behind.bow:
                        continue
                    # Only a boat genuinely astern can be the passer.
                    if ahead.position(time) <= behind.position(time):
                        continue
                    self._update_pair(behind, ahead, time)

            # **Retire encounters whose passer has gone by.**  The loop
            # above stops visiting a pair the moment the passer draws
            # ahead, so a pass that succeeds is never marked COMPLETE and
            # the passed crew is never released from its yield.  Coupled
            # to a speed field that showed up immediately: a crew held a
            # 3.5 m offset for the remaining two thirds of the race and
            # lost 187 m to it, having been passed once.
            for (passer_bow, passee_bow), encounter in \
                    list(self.encounters.items()):
                if encounter.phase is Phase.COMPLETE:
                    continue
                passer = self.entries[passer_bow]
                passee = self.entries[passee_bow]
                if passer.position(time) - passee.position(time) > \
                        self.rules.boat_length + self.rules.clear_at:
                    encounter.phase = Phase.COMPLETE
                    encounter.completed_at = time
                    passee.owes.pop(passer_bow, None)
                    self.log.add(time, "complete", passer=passer_bow,
                                 passee=passee_bow)

            for entry in racing:
                self._advance_lateral(entry, dt)

            # Advance the physics-driven crews along the river.  Their
            # speed depends on where they are laterally, so this has to
            # come *after* the yields have moved them: a crew pushed off
            # the deep line is slower for the rest of the step, which is
            # the whole point of coupling the two models.
            for entry in racing:
                if entry.speed_fn is None:
                    continue
                free = float(entry.speed_fn(entry.station,
                                            entry.preferred_lateral))
                actual = entry.current_speed()
                entry.station += actual * dt
                entry.lost_to_yield += (free - actual) * dt

            for entry in order:
                if entry.speed_fn is not None and entry.finished is None \
                        and entry.station <= 0.0 and time >= entry.start:
                    entry.station = 1e-6

            if all(e.finished is not None or e.disqualified
                   for e in order):
                break
            time += dt
        return self.log

    # -- results ----------------------------------------------------------
    def results(self) -> List[dict]:
        out = []
        for entry in sorted(self.entries.values(), key=lambda e: e.bow):
            raw = (entry.finished - entry.start
                   if entry.finished is not None else float("nan"))
            out.append({
                "bow": entry.bow, "name": entry.label,
                "raw": raw, "penalty": entry.penalty,
                "official": raw + entry.penalty,
                "offences": entry.offences,
                "disqualified": entry.disqualified,
            })
        return out


def build_field(n: int, interval: float = 15.0, mean_speed: float = 4.23,
                spread: float = 0.06, seed: int = 0,
                names: List[str] = None) -> List[Entry]:
    """A field started at fixed intervals with randomised speeds.

    ``spread`` is the fractional standard deviation of boat speed across
    the field. Six percent is about what separates the middle of a masters
    event, and it is what makes passes happen at all: with identical
    speeds nobody ever closes and the rulebook never fires.
    """
    rng = np.random.default_rng(seed)
    speeds = mean_speed * (1.0 + rng.normal(0.0, spread, size=n))
    return [Entry(bow=i + 1, start=i * float(interval),
                  speed=float(max(s, 0.5 * mean_speed)),
                  name=(names[i] if names and i < len(names) else ""))
            for i, s in enumerate(speeds)]
