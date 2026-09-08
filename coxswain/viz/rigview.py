r"""The boat as a plan you can edit: who sits where, and on which side.

A rig is a picture before it is a table.  A coxswain asked what a bucket
rig is will draw one -- four seats down a hull, riggers sticking out
alternately or not -- and will never recite ``(-1, +1, +1, -1)``.  So
this carries the crew as a *seating plan*, and the menu draws it as one:
the hull down the middle, a rigger out of each seat on the side that
seat actually rows, and each rower's numbers in a box on their own side
of the boat with a line back to their seat.

That side matters more than it looks.  A rower's side is the single
thing most often got wrong when a lineup is typed in, and it is
invisible in a list of names -- but in a plan, a starboard rower with a
port rigger is instantly and obviously wrong.

Kept free of pygame, like :mod:`coxswain.viz.menu`, so the layout can be
tested without a display.  The drawing takes the geometry this computes
and does nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__all__ = ["Rower", "Lineup", "PRESETS", "SHELLS", "RIGS", "rig_sides",
           "plan_geometry", "erg_watts", "erg_stamp"]

LB = 0.45359237
INCH = 0.0254

#: ``key -> (label, seats, coxed, bow_loaded)``
SHELLS = {
    "4+": ("Coxed four", 4, True, True),
    "8+": ("Eight", 8, True, False),
    "2x": ("Double scull", 2, False, False),
    "1x": ("Single scull", 1, False, False),
}

#: Rig names per seat count.  The values are the model's own convention:
#: ``+1`` port, ``-1`` starboard, read from the STROKE seat toward bow.
RIGS = {
    4: ("standard", "starboard stroke", "bucket, stbd stroke",
        "bucket, port stroke", "tandem, port stroke", "tandem, stbd stroke"),
    8: ("standard", "starboard stroke", "german", "italian", "battleship",
        "tandem"),
    2: ("sculling",),
    1: ("sculling",),
}


def erg_watts(stamp: str) -> Optional[float]:
    """Concept2's own relation, ``P = 2.80 / pace^3``, pace in s/m.

    Confirmed against the squad's own spreadsheet, whose stored watts
    agree with this to 0.3 W across every woman with a current test.
    """
    try:
        minutes, seconds = str(stamp).split(":")
        total = int(minutes) * 60 + float(seconds)
    except Exception:
        return None
    if total <= 0:
        return None
    return 2.80 / (total / 5000.0) ** 3


def erg_stamp(watts: float) -> str:
    """The inverse: what 5 km time a power corresponds to."""
    total = (2.80 / float(watts)) ** (1.0 / 3.0) * 5000.0
    return "%d:%04.1f" % (int(total // 60), total % 60)


@dataclass
class Rower:
    """One seat's occupant.  ``side`` is ``+1`` port, ``-1`` starboard."""

    name: str = ""
    pounds: float = 0.0
    feet: int = 0
    inches: float = 0.0
    erg_5k: str = ""
    side: int = +1

    @property
    def mass(self) -> float:
        return float(self.pounds) * LB

    @property
    def stature(self) -> float:
        return (int(self.feet) * 12 + float(self.inches)) * INCH

    @property
    def watts(self) -> Optional[float]:
        return erg_watts(self.erg_5k) if self.erg_5k else None


def rig_sides(shell: str, rig: str) -> Tuple[int, ...]:
    """The sides a named rig gives, stroke first.

    Read straight from :mod:`coxswain.boats.rig` rather than duplicated,
    so the menu cannot offer a rig the boat builder does not know.
    """
    from ..boats.rig import RIG_PATTERNS, RIG_PATTERNS_4

    seats = SHELLS[shell][1]
    if seats == 4:
        return tuple(RIG_PATTERNS_4.get(rig, RIG_PATTERNS_4["standard"]))
    if seats == 8:
        return tuple(RIG_PATTERNS.get(rig, RIG_PATTERNS["standard"]))
    return tuple([0] * seats)          # sculling: both hands, no side


@dataclass
class Lineup:
    """A shell, a rig, and the people in it."""

    shell: str = "4+"
    rig: str = "standard"
    rowers: List[Rower] = field(default_factory=list)
    cox_name: str = ""
    cox_pounds: float = 0.0
    cox_feet: int = 0
    cox_inches: float = 0.0
    name: str = "New lineup"

    # -- structure --------------------------------------------------------
    @property
    def seats(self) -> int:
        return SHELLS[self.shell][1]

    @property
    def coxed(self) -> bool:
        return SHELLS[self.shell][2]

    def seat_label(self, index: int) -> str:
        """Stroke, then numbers down the boat, then bow."""
        if index == 0:
            return "stroke"
        if index == self.seats - 1:
            return "bow"
        return str(self.seats - index)

    # -- editing ----------------------------------------------------------
    def apply_rig(self) -> "Lineup":
        """Set every rower's side from the named rig."""
        sides = rig_sides(self.shell, self.rig)
        for index, rower in enumerate(self.rowers[:len(sides)]):
            rower.side = sides[index]
        return self

    def switch_side(self, index: int) -> "Lineup":
        """Move one rower across the boat.

        The rig name stops describing the boat as soon as a seat is
        moved by hand, so it becomes ``"custom"``.  Leaving it reading
        "standard" over a rig that is not standard is exactly the quiet
        lie this plan exists to prevent.
        """
        if 0 <= index < len(self.rowers):
            self.rowers[index].side = -self.rowers[index].side
            if tuple(r.side for r in self.rowers) != rig_sides(self.shell,
                                                               self.rig):
                self.rig = "custom"
        return self

    def set_rig(self, rig: str) -> "Lineup":
        self.rig = rig
        return self.apply_rig()

    def set_shell(self, shell: str) -> "Lineup":
        """Change hull, keeping as many rowers as still have a seat."""
        self.shell = shell
        wanted = SHELLS[shell][1]
        while len(self.rowers) < wanted:
            self.rowers.append(Rower(side=+1))
        del self.rowers[wanted:]
        if self.rig not in RIGS.get(wanted, ()):
            self.rig = RIGS.get(wanted, ("standard",))[0]
        return self.apply_rig()

    def balanced(self) -> bool:
        """Equal port and starboard, which a sweep boat must be."""
        if self.seats in (1, 2):
            return True
        return sum(r.side for r in self.rowers) == 0

    # -- numbers ----------------------------------------------------------
    def mean_watts(self) -> Optional[float]:
        values = [r.watts for r in self.rowers if r.watts]
        return sum(values) / len(values) if values else None

    def crew_mass(self) -> float:
        return sum(r.mass for r in self.rowers)

    def cox_mass(self) -> float:
        return float(self.cox_pounds) * LB


def plan_geometry(lineup: Lineup, width: float, height: float) -> dict:
    """Where to draw everything, in pixels, for a top-down plan.

    Returns the hull outline, a mark per seat, the rigger each seat
    carries and where its label box hangs.  All of it here rather than
    in the renderer, so the layout is testable and the drawing is only
    drawing.
    """
    seats = max(lineup.seats, 1)
    hull_len = height * 0.74
    top = (height - hull_len) * 0.5
    centre = width * 0.5
    half_beam = max(width * 0.022, 5.0)
    rigger = max(width * 0.075, 26.0)

    # Stroke sits at the STERN, which is the bottom of the plan: the boat
    # is drawn travelling up the screen, the way a course map runs, so
    # "up" is the direction the boat is going and bow is at the top.
    spacing = hull_len / (seats + 1.0)
    marks = []
    for index in range(seats):
        y = top + hull_len - spacing * (index + 1.0)
        rower = (lineup.rowers[index] if index < len(lineup.rowers)
                 else Rower())
        side = rower.side or +1
        # Port is the left hand of a coxswain facing the bow, and the bow
        # is up the screen, so port draws on the left.
        tip = centre - side * rigger
        marks.append({
            "index": index,
            "label": lineup.seat_label(index),
            "rower": rower,
            "seat": (centre, y),
            "rigger": ((centre - side * half_beam, y), (tip, y)),
            "side": side,
            "box_anchor": (tip, y),
            "box_side": -1 if side > 0 else +1,
        })

    cox = None
    if lineup.coxed:
        bow_loaded = SHELLS[lineup.shell][3]
        cox = (centre, top + 0.03 * hull_len if bow_loaded
               else top + hull_len - 0.03 * hull_len)
    return {
        "hull": (centre, top, half_beam, hull_len),
        "bow": (centre, top),
        "stern": (centre, top + hull_len),
        "marks": marks,
        "cox": cox,
        "cox_bow_loaded": SHELLS[lineup.shell][3] if lineup.coxed else False,
        "title": "%s -- %s" % (SHELLS[lineup.shell][0], lineup.rig),
    }


def _hocr_four() -> Lineup:
    """The Women's Veteran 60+ four this project is built around.

    Bucket rigged with a starboard stroke -- S-P-P-S read from stroke --
    in a bow-loader, so the coxswain lies down in the bow.
    """
    crew = [
        Rower("Marilyn", 120.0, 5, 5.5, "23:25"),
        Rower("Alex", 120.0, 5, 2.0, "23:41"),
        Rower("Lea", 125.0, 5, 3.5, "23:07"),
        Rower("Sheila", 155.0, 5, 3.0, "22:20"),
    ]
    lineup = Lineup(shell="4+", rig="bucket, stbd stroke", rowers=crew,
                    cox_name="you", cox_pounds=160.0, cox_feet=5,
                    cox_inches=8.0, name="HOCR 4+")
    return lineup.apply_rig()


#: Saved boats, by name.  Callables, so each load is a fresh copy and
#: editing one does not quietly edit the preset.
PRESETS = {"HOCR 4+": _hocr_four}
