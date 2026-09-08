r"""The squad's anonymous erg roster, and boats built out of it.

``data/squad_roster.csv`` is written by ``scripts/build_squad_roster.py``
from the club's training spreadsheet, with the names removed and the
identifying numbers blurred before anything is committed.  See that
script for what is kept and why; this module is only the reading end.

The point of it
---------------
A simulator needs opposition, and opposition invented out of nothing is
opposition that proves nothing.  These are real 5k scores from a real
masters squad, so a boat crewed from this roster pulls what people that
age and weight actually pull -- including the part that is hardest to
invent, which is the spread.  The squad runs from 19:48 to well past
25 minutes over the 5k, and a field where everybody is average is not a
field anybody recognises.

The one thing the sheet does not carry
--------------------------------------
Height.  It logs weight, because the erg's own weight adjustment needs
it, and nobody ever wrote down a stature.  So a roster rower's height
is **not a measurement** and is flagged as such wherever it is shown:
:data:`ASSUMED_STATURE` stands in a population mean, and every rower
built from the roster carries ``stature_estimated=True``.

That flag matters more than it looks.  Stature sets every link length
in :class:`~coxswain.crew.kinematics.JointDrivenRower`, so it moves the
crew's centre-of-mass travel and with it the hull's speed fluctuation.
A roster boat's *power* is measured; its *body geometry* is a guess,
and the two should never be quoted with the same confidence.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

LB = 0.45359237
INCH = 0.0254

#: Population mean stature standing in for a number the sheet does not
#: have.  NHANES adult US means, not a rowing-specific figure: masters
#: rowers do skew taller, but by an amount nobody here has measured, and
#: a made-up correction is worse than a stated approximation.
ASSUMED_STATURE = {"women": 1.63, "men": 1.75}

#: Where the roster lives, relative to the repository root.
ROSTER_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data", "squad_roster.csv")


@dataclass(frozen=True)
class RosterEntry:
    """One anonymous rower: what they weigh, and what they pull."""

    id: str
    squad: str
    age: Optional[int]
    pounds: Optional[float]
    #: Median 5k, ``mm:ss``.  What they hold, not what they once did.
    erg_5k: str
    #: Fastest 5k on the sheet.
    erg_5k_best: str
    #: How many 5k pieces the median rests on -- one is an anecdote.
    pieces: int

    @property
    def stature(self) -> float:
        return ASSUMED_STATURE.get(self.squad, 1.70)

    @property
    def mass(self) -> float:
        return float(self.pounds or 0.0) * LB

    @property
    def seconds(self) -> float:
        minutes, secs = self.erg_5k.split(":")
        return float(minutes) * 60.0 + float(secs)

    @property
    def watts(self) -> float:
        """Concept2's own relation, ``P = 2.80 / pace^3``, pace in s/m."""
        return 2.80 / (self.seconds / 5000.0) ** 3


_CACHE = {}


def load_roster(path: str = None) -> Tuple[RosterEntry, ...]:
    """Every rower on the roster, in file order (which is shuffled)."""
    path = path or ROSTER_PATH
    if path in _CACHE:
        return _CACHE[path]
    if not os.path.exists(path):
        _CACHE[path] = ()
        return ()
    entries = []
    with io.open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            entries.append(RosterEntry(
                id=row["id"], squad=row["squad"],
                age=int(row["age"]) if row["age"] else None,
                pounds=float(row["pounds"]) if row["pounds"] else None,
                erg_5k=row["erg_5k"], erg_5k_best=row["erg_5k_best"],
                pieces=int(row["pieces"])))
    _CACHE[path] = tuple(entries)
    return _CACHE[path]


def select(squad: str = "women", min_age: int = None, max_age: int = None,
           roster: Sequence[RosterEntry] = None) -> Tuple[RosterEntry, ...]:
    """The roster filtered to one squad and age band, fastest first.

    ``min_age`` is inclusive, which is how masters bands read: a "60+"
    event is 60 and over, and a rower turning 60 that calendar year is
    in it.
    """
    entries = load_roster() if roster is None else roster
    picked = [e for e in entries if e.squad == squad and e.pounds]
    if min_age is not None:
        picked = [e for e in picked if e.age is not None and e.age >= min_age]
    if max_age is not None:
        picked = [e for e in picked if e.age is not None and e.age <= max_age]
    return tuple(sorted(picked, key=lambda e: e.seconds))


def crew_of(seats: int, squad: str = "women", min_age: int = None,
            max_age: int = None, offset: int = 0,
            roster: Sequence[RosterEntry] = None
            ) -> Tuple[RosterEntry, ...]:
    """``seats`` rowers off the roster, ``offset`` places down the order.

    ``offset`` is how a whole field gets built out of one squad: 0 is
    the fastest boat that band can make, and stepping the offset walks
    down through crews that are each internally plausible -- rowers of
    similar speed sit together, the way selection actually puts them.
    Taking four at random instead would give every boat the squad's
    mean and none of them its range.
    """
    pool = select(squad, min_age, max_age, roster)
    if len(pool) < seats:
        return pool
    offset = max(0, min(int(offset), len(pool) - seats))
    return pool[offset:offset + seats]


def feet_inches(stature: float) -> Tuple[int, float]:
    """Metres to whole feet and remaining inches, for the rig editor."""
    total = stature / INCH
    return int(total // 12), round(total - 12 * int(total // 12), 1)
