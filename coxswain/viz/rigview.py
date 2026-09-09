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

import re
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
    #: Years.  0 means "not given", and then nothing about this rower
    #: is aged -- see :mod:`coxswain.crew.ageing`.
    age: int = 0
    #: This rower's own skill and experience, 0 novice to 1 ideal.
    #: Negative means "use the crew slider", which is what an untouched
    #: rower has: the Rowers menu still moves the whole crew, and these
    #: only override the seats somebody has typed a number into.
    skill: float = -1.0
    experience: float = -1.0
    #: True when the height is a population mean standing in for a
    #: number nobody measured -- roster rowers, whose sheet logs weight
    #: and erg scores but never a stature.  Shown with a "~" so the
    #: picture does not claim to know something it does not.
    stature_estimated: bool = False

    @property
    def mass(self) -> float:
        return float(self.pounds) * LB

    @property
    def stature(self) -> float:
        return (int(self.feet) * 12 + float(self.inches)) * INCH

    @property
    def watts(self) -> Optional[float]:
        return erg_watts(self.erg_5k) if self.erg_5k else None


#: The fields a rower can be typed into, in the order the editor walks
#: them, as ``(attribute, label)``.  Height is two fields because that
#: is how people say it -- "five foot three" -- and asking for metres or
#: total inches makes everyone stop and do arithmetic at the keyboard.
FIELDS = (("name", "name"),
          ("pounds", "weight, lb"),
          ("feet", "height, ft"),
          ("inches", "height, in"),
          ("erg_5k", "5k erg, m:ss"),
          ("age", "age, years"),
          ("skill", "skill 0-1"),
          ("experience", "experience 0-1"))

#: What the erg field will accept: ``m:ss`` or ``mm:ss`` with an
#: optional tenth, or nothing at all.
_ERG = re.compile(r"^(\d{1,2}):([0-5]\d)(\.\d)?$")


def field_text(rower: "Rower", index: int) -> str:
    """The current value of a field, as the text one would edit."""
    attr = FIELDS[index][0]
    value = getattr(rower, attr)
    if attr == "pounds":
        return "" if not value else "%g" % float(value)
    if attr == "feet":
        return "" if not value and not rower.inches else "%d" % int(value)
    if attr == "inches":
        return "" if not value and not rower.feet else "%g" % float(value)
    if attr == "age":
        return "" if not value else "%d" % int(value)
    if attr in ("skill", "experience"):
        # Negative is "use the crew slider", and shows as blank rather
        # than as a number nobody typed.
        return "" if value is None or float(value) < 0.0 else "%g" % float(value)
    return str(value or "")


def parse_field(index: int, text: str):
    """The typed text as the field's value, or ``None`` if it is not one.

    Refusing is the whole job: a weight of "abc" must not become 0 lb,
    which reads as a real number and quietly changes the boat.
    """
    attr = FIELDS[index][0]
    text = text.strip()
    if attr == "name":
        return text
    if attr == "erg_5k":
        if text == "":
            return ""
        return text if _ERG.match(text) else None
    if text == "":
        if attr in ("skill", "experience"):
            return -1.0             # back to the crew slider
        return 0.0 if attr not in ("feet", "age") else 0
    try:
        value = float(text)
    except ValueError:
        return None
    if value < 0.0:
        return None
    if attr == "feet":
        return int(value)
    if attr == "age":
        return int(value) if 10 <= value <= 110 else None
    if attr in ("skill", "experience"):
        return float(value) if value <= 1.0 else None
    if attr == "inches" and value >= 12.0:
        return None                     # that is another foot
    return value


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
    def to_dict(self) -> dict:
        """Plain data, for presets.json."""
        return {
            "name": self.name, "shell": self.shell, "rig": self.rig,
            "cox": {"name": self.cox_name, "pounds": self.cox_pounds,
                    "feet": self.cox_feet, "inches": self.cox_inches},
            "rowers": [{"name": r.name, "pounds": r.pounds, "feet": r.feet,
                        "inches": r.inches, "erg_5k": r.erg_5k,
                        "side": r.side, "age": r.age, "skill": r.skill,
                        "experience": r.experience,
                        "stature_estimated": bool(r.stature_estimated)}
                       for r in self.rowers],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Lineup":
        """The inverse.  Sides are taken as saved -- a saved custom rig
        is custom -- and anything missing gets the field's default, so
        a file from an older build still loads."""
        cox = data.get("cox") or {}
        rowers = [Rower(name=str(r.get("name", "")),
                        pounds=float(r.get("pounds", 0.0)),
                        feet=int(r.get("feet", 0)),
                        inches=float(r.get("inches", 0.0)),
                        erg_5k=str(r.get("erg_5k", "")),
                        side=int(r.get("side", +1)),
                        age=int(r.get("age", 0) or 0),
                        skill=float(r.get("skill", -1.0)),
                        experience=float(r.get("experience", -1.0)),
                        stature_estimated=bool(r.get("stature_estimated",
                                                     False)))
                  for r in data.get("rowers", [])]
        lineup = cls(shell=str(data.get("shell", "4+")),
                     rig=str(data.get("rig", "standard")), rowers=rowers,
                     cox_name=str(cox.get("name", "")),
                     cox_pounds=float(cox.get("pounds", 0.0)),
                     cox_feet=int(cox.get("feet", 0)),
                     cox_inches=float(cox.get("inches", 0.0)),
                     name=str(data.get("name", "Saved lineup")))
        if lineup.rig != "custom":
            lineup.apply_rig()
        return lineup

    def apply_rig(self) -> "Lineup":
        """Set every rower's side from the named rig."""
        sides = rig_sides(self.shell, self.rig)
        for index, rower in enumerate(self.rowers[:len(sides)]):
            rower.side = sides[index]
        return self

    def commit_edit(self, seat: int, index: int, text: str) -> bool:
        """Set one field of one rower from typed text.

        Returns whether it took.  A height typed in is a measurement,
        so it clears ``stature_estimated`` -- the flag means "nobody
        measured this", and somebody just did.
        """
        if not (0 <= seat < len(self.rowers)):
            return False
        value = parse_field(index, text)
        if value is None:
            return False
        rower = self.rowers[seat]
        attr = FIELDS[index][0]
        setattr(rower, attr, value)
        if attr in ("feet", "inches"):
            rower.stature_estimated = False
        return True

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
    # Ages are the age band the crew race in -- Veteran 60+ -- rather
    # than four measured birthdays, so the physiology is banded, not
    # personal.  Type real ones in and they are used instead.
    crew = [
        Rower("Marilyn", 120.0, 5, 5.5, "23:25", age=62),
        Rower("Alex", 120.0, 5, 2.0, "23:41", age=62),
        Rower("Lea", 125.0, 5, 3.5, "23:07", age=62),
        Rower("Sheila", 155.0, 5, 3.0, "22:20", age=62),
    ]
    lineup = Lineup(shell="4+", rig="bucket, stbd stroke", rowers=crew,
                    cox_name="you", cox_pounds=160.0, cox_feet=5,
                    cox_inches=8.0, name="HOCR 4+")
    return lineup.apply_rig()


def _squad_boat(shell: str, rig: str, label: str, squad: str = "women",
                min_age: int = None, max_age: int = None, offset: int = 0):
    """A boat crewed off the anonymous roster; see :mod:`..crew.roster`.

    Returns a callable, because :data:`PRESETS` holds builders rather
    than boats -- and because the roster file may not be present, in
    which case this hands back an empty shell rather than raising in
    the middle of a menu.
    """
    def build() -> Lineup:
        from ..crew.roster import crew_of, feet_inches

        seats = SHELLS[shell][1]
        picked = crew_of(seats, squad, min_age, max_age, offset)
        crew = []
        for index in range(seats):
            if index >= len(picked):
                crew.append(Rower())
                continue
            entry = picked[index]
            feet, inches = feet_inches(entry.stature)
            crew.append(Rower(entry.id, entry.pounds or 0.0, feet, inches,
                              entry.erg_5k, stature_estimated=True))
        # Stroke is the seat that sets the rhythm, so the roster's
        # fastest goes there and the order runs down from it.  That is
        # not how every crew is seated -- plenty of coaches put their
        # strongest at 3 -- but it is the arrangement that needs no
        # further justification, and any seat can be swapped by hand.
        lineup = Lineup(shell=shell, rig=rig, rowers=crew,
                        cox_name="cox", cox_pounds=125.0, cox_feet=5,
                        cox_inches=4.0, name=label)
        return lineup.apply_rig()

    return build


def _pink_ribbon() -> Lineup:
    """Genevieve's Pink Ribbon: a men's coxed four on a standard rig.

    Port stroke, alternating -- P S P S read from the stroke seat --
    in a bow-loader.  Four six-footers and a 73 kg coxswain lying down
    in the bow; the crew mean is 233 W, a 19:06 5k.
    """
    crew = [
        Rower("Kevin", 182.0, 6, 0.0, "19:39"),
        Rower("Mark", 182.0, 6, 0.0, "18:55"),
        Rower("Scott", 220.0, 6, 0.0, "18:26"),
        Rower("Evan", 180.0, 6, 0.0, "19:30"),
    ]
    lineup = Lineup(shell="4+", rig="standard", rowers=crew,
                    cox_name="you", cox_pounds=161.0, cox_feet=5,
                    cox_inches=8.0, name="Genevieve's Pink Ribbon")
    return lineup.apply_rig()


#: Saved boats, by name.  Callables, so each load is a fresh copy and
#: editing one does not quietly edit the preset.
#:
#: The squad boats are built from ``data/squad_roster.csv`` -- real erg
#: scores from a real masters squad, with the identities removed.  They
#: exist so a practice race has opposition that pulls what people that
#: age actually pull, rather than four copies of an invented average.
#: The boat the editor opens on.  Definition order in PRESETS is the
#: order Load preset cycles through; this is first in both.
DEFAULT_PRESET = "HOCR 4+"

PRESETS = {
    "HOCR 4+": _hocr_four,
    "Genevieve's Pink Ribbon": _pink_ribbon,
    "Squad W 60+ 4+": _squad_boat("4+", "bucket, stbd stroke",
                                  "Squad W 60+ 4+", "women", min_age=60),
    "Squad W 50s 4+": _squad_boat("4+", "standard", "Squad W 50s 4+",
                                  "women", min_age=50, max_age=59),
    "Squad W open 8+": _squad_boat("8+", "standard", "Squad W open 8+",
                                   "women"),
    "Squad M open 8+": _squad_boat("8+", "standard", "Squad M open 8+",
                                   "men"),
}


# ---------------------------------------------------------------------------
# Drawing, and the only part that knows about pygame
# ---------------------------------------------------------------------------

#: Shared with :mod:`coxswain.viz.menu` so the two screens match.
INK = (233, 240, 245)
DIM = (150, 164, 176)
PICK = (255, 146, 72)
PANEL = (12, 17, 21, 188)
HULL = (196, 202, 206)
PORT_RED = (196, 92, 88)
STBD_GREEN = (96, 176, 118)


def side_colour(side: int):
    """Port red, starboard green -- the navigation convention.

    Worth using rather than inventing one: a rower already knows which
    side is which by that colour, so the plan reads without a key.
    """
    return PORT_RED if side > 0 else STBD_GREEN


def hull_half_beam(fraction: float) -> float:
    """Half-beam as a fraction of maximum, bow (0) to stern (1).

    Fine at both ends, full amidships.  A rectangle would read as a
    barge, and the taper is what makes the two ends legible in a plan
    that has no other cue about which is which.
    """
    return max(0.0, (4.0 * fraction * (1.0 - fraction)) ** 0.55)


def hull_outline(centre, top, half_beam, hull_len, steps: int = 26):
    """The hull as a closed polygon, down one side and back up the other."""
    points = []
    for step in range(steps + 1):
        f = step / steps
        points.append((centre - half_beam * hull_half_beam(f),
                       top + f * hull_len))
    for step in range(steps, -1, -1):
        f = step / steps
        points.append((centre + half_beam * hull_half_beam(f),
                       top + f * hull_len))
    return points


def rower_lines(rower: "Rower") -> list:
    """What a rower's box says, top line first."""
    lines = [rower.name or "(empty)"]
    if rower.pounds:
        lines.append("%.0f lb   %s%d ft %.1f in"
                     % (rower.pounds, "~" if rower.stature_estimated else "",
                        rower.feet, rower.inches))
    if rower.erg_5k:
        watts = rower.watts
        lines.append("5k %s%s" % (rower.erg_5k,
                                  "   %.0f W" % watts if watts else ""))
    extra = []
    if rower.age:
        extra.append("age %d" % int(rower.age))
    if rower.skill is not None and float(rower.skill) >= 0.0:
        extra.append("skill %.2f" % float(rower.skill))
    if rower.experience is not None and float(rower.experience) >= 0.0:
        extra.append("exp %.2f" % float(rower.experience))
    if extra:
        lines.append("   ".join(extra))
    return lines


def edit_lines(rower: "Rower", index: int, buffer: str) -> list:
    """The box while a field is being typed: every field on its own
    row, the live one showing the buffer with a caret.  Showing all of
    them is what tells the eye which one Tab will go to next."""
    lines = [rower.name or "(empty)"]
    for at, (attr, label) in enumerate(FIELDS):
        if at == index:
            lines.append("%s: %s_" % (label, buffer))
        else:
            lines.append("%s: %s" % (label, field_text(rower, at) or "-"))
    return lines


def draw_plan(surface, lineup: "Lineup", font, small, size,
              selected: int = -1, editing=None) -> dict:
    """Draw the boat from above, with each rower's numbers beside them.

    Returns the box rectangles by seat index, so the caller can
    hit-test a click without recomputing the layout.
    """
    import pygame

    width, height = size
    plan_w = width * 0.62
    plan = plan_geometry(lineup, plan_w, height)
    centre, top, half_beam, hull_len = plan["hull"]

    title = font.render(plan["title"], True, INK)
    surface.blit(title, ((plan_w - title.get_width()) * 0.5,
                         max(top - 46, 6)))
    if not lineup.balanced():
        warn = small.render("unbalanced -- a sweep boat needs equal sides",
                            True, PICK)
        surface.blit(warn, ((plan_w - warn.get_width()) * 0.5,
                            max(top - 24, 28)))

    outline = hull_outline(centre, top, half_beam, hull_len)
    pygame.draw.polygon(surface, (28, 36, 42), outline)
    pygame.draw.polygon(surface, HULL, outline, 2)

    boxes = {}
    for mark in plan["marks"]:
        colour = side_colour(mark["side"])
        (x0, y0), (x1, y1) = mark["rigger"]
        pygame.draw.line(surface, colour, (x0, y0), (x1, y1), 3)
        pygame.draw.circle(surface, colour, (int(x1), int(y1)), 5)
        seat_x, seat_y = int(mark["seat"][0]), int(mark["seat"][1])
        pygame.draw.circle(surface, (18, 24, 29), (seat_x, seat_y), 6)
        pygame.draw.circle(surface, HULL, (seat_x, seat_y), 6, 1)

        if editing is not None and editing[0] == mark["index"]:
            lines = edit_lines(mark["rower"], editing[1], editing[2])
        else:
            lines = rower_lines(mark["rower"])
        box_w = max(small.size(line)[0] for line in lines) + 18
        box_h = 8 + len(lines) * (small.get_height() + 2)
        gap = 16
        if mark["box_side"] < 0:
            box_x = mark["box_anchor"][0] - gap - box_w
        else:
            box_x = mark["box_anchor"][0] + gap
        rect = pygame.Rect(int(box_x),
                           int(mark["box_anchor"][1] - box_h * 0.5),
                           int(box_w), int(box_h))

        link_x = rect.right if mark["box_side"] < 0 else rect.left
        pygame.draw.line(surface, DIM, mark["box_anchor"],
                         (link_x, rect.centery), 1)

        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill(PANEL)
        surface.blit(panel, rect.topleft)
        chosen = mark["index"] == selected
        pygame.draw.rect(surface, PICK if chosen else colour, rect,
                         2 if chosen else 1)
        live = (editing[1] + 1 if editing is not None
                and editing[0] == mark["index"] else -1)
        for row, line in enumerate(lines):
            shade = PICK if row == live else (INK if row == 0 else DIM)
            surface.blit(small.render(line, True, shade),
                         (rect.x + 9,
                          rect.y + 4 + row * (small.get_height() + 2)))

        # The seat name goes on the hull's EMPTY side -- opposite the
        # rigger -- and clear of the gunwale.  Offset by a fixed few
        # pixels it landed inside the hull, where the dark fill ate it.
        tag = small.render(mark["label"], True, DIM)
        clear = half_beam + 7
        tag_x = (seat_x + clear if mark["box_side"] < 0
                 else seat_x - clear - tag.get_width())
        surface.blit(tag, (tag_x, seat_y - tag.get_height() * 0.5))
        boxes[mark["index"]] = rect

    if plan["cox"] is not None:
        cx, cy = plan["cox"]
        pygame.draw.circle(surface, (232, 196, 96), (int(cx), int(cy)), 7)
        pygame.draw.circle(surface, (18, 24, 29), (int(cx), int(cy)), 7, 1)
        where = ("bow-loader, lying down" if plan["cox_bow_loaded"]
                 else "stern")
        tag = small.render("%s  (%s)" % (lineup.cox_name or "cox", where),
                           True, DIM)
        # Beside the marker, not above it: centred over a bow-loader the
        # label sat on the hull's own tip and the two fought.
        surface.blit(tag, (cx + 14, cy - tag.get_height() * 0.5))

    mean = lineup.mean_watts()
    footer = []
    if mean:
        footer.append("crew mean %.0f W  (5k %s)" % (mean, erg_stamp(mean)))
    footer.append("crew %.0f kg + cox %.0f kg"
                  % (lineup.crew_mass(), lineup.cox_mass()))
    for row, line in enumerate(footer):
        text = small.render(line, True, DIM)
        surface.blit(text, ((plan_w - text.get_width()) * 0.5,
                            top + hull_len + 14
                            + row * (small.get_height() + 3)))
    return boxes


#: The rows the side pane offers, as ``(key, label)``.
PANE_ROWS = (("preset", "Load preset"),
             ("shell", "Shell"),
             ("rig", "Rig"),
             ("switch", "Switch a rower's side"),
             ("edit", "Edit a rower"),
             ("save", "Save as preset"),
             ("done", "Done"))


def pane_values(lineup: "Lineup") -> list:
    """What each pane row currently reads."""
    return [lineup.name, SHELLS[lineup.shell][0], lineup.rig,
            "pick a seat", "pick a seat, then type", "enter, then a name",
            ""]


def draw_side_pane(surface, lineup: "Lineup", font, small, size,
                   cursor: int = 0) -> list:
    """The pane of actions beside the plan.  Returns its row rectangles."""
    import pygame

    width, height = size
    left = width * 0.64
    pane_w = width - left - 18
    pane_h = height * 0.56
    top = height * 0.20

    panel = pygame.Surface((int(pane_w), int(pane_h)), pygame.SRCALPHA)
    panel.fill(PANEL)
    surface.blit(panel, (int(left), int(top)))
    pygame.draw.rect(surface, (70, 82, 92),
                     pygame.Rect(int(left), int(top), int(pane_w),
                                 int(pane_h)), 1)
    surface.blit(font.render("Boat", True, INK), (left + 16, top + 14))

    rects, values = [], pane_values(lineup)
    y = top + 56
    for index, ((_key, label), value) in enumerate(zip(PANE_ROWS, values)):
        picked = index == cursor
        height_used = small.get_height() + (small.get_height() + 4
                                            if value else 0)
        rect = pygame.Rect(int(left + 8), int(y - 5), int(pane_w - 16),
                           int(height_used + 10))
        if picked:
            pygame.draw.rect(surface, (30, 40, 48), rect)
        surface.blit(small.render(label, True, PICK if picked else INK),
                     (left + 18, y))
        if value:
            text = small.render(value, True, DIM)
            surface.blit(text, (left + pane_w - 18 - text.get_width(),
                                y + small.get_height() + 2))
        y += height_used + 16
        rects.append(rect)
    return rects
