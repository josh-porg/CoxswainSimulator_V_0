r"""Menus for the trainer: pick a boat and a course, and pause.

Two things are wanted here and they pull in different directions.  A
coxswain being handed this to try should not have to know what a
``--race`` flag is; and the person developing it should still be able to
drive everything from the command line.  So the menu is a **model** --
a list of settings and where the cursor is -- with no drawing and no
pygame in it, and the renderer is a separate function.  The model is
what the tests exercise; it is also what makes a second front end (a
launcher, a web build) possible without moving any logic.

Nothing here starts a simulation.  :class:`Setup` is read once, before
the world is built, and :class:`PauseMenu` hands back an *action* for
the caller to carry out.  Keeping the menu ignorant of the simulator is
what stops the two becoming one object that neither can be tested
without.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = ["Choice", "Menu", "boat_choices", "course_choices",
           "setup_menu", "pause_menu", "build_boat",
           "start_music", "stop_music", "music_path"]


#: Shells a coxswain might sit in, as ``(key, label, seats, coxed)``.
#: ``seats`` is how many are pulling, which is what the audio and the
#: rig need; ``coxed`` decides whether there is anyone to steer.
BOATS = (
    ("4+", "Coxed four", 4, True),
    ("8+", "Eight", 8, True),
    ("2x", "Double scull", 2, False),
    ("1x", "Single scull", 1, False),
)

#: Courses, as ``(key, label, blurb)``.
COURSES = (
    ("charles", "Head of the Charles",
     "4.8 km, Boston.  Six bridges and the Weeks turn."),
    ("totl", "Tail of the Lake",
     "4.0 km, Lake Union.  Downtown Seattle down the length of it."),
    ("hotl", "Head of the Lake",
     "4.8 km, Seattle.  Portage Bay, the Montlake Cut, the Big Turn."),
)


@dataclass
class Choice:
    """One row of a menu: a label, some options, and which is picked."""

    key: str
    label: str
    options: Sequence[Tuple]
    index: int = 0
    #: ``None`` for a setting; a string for an action row like "Start".
    action: Optional[str] = None
    #: Optional ``(low, high, step)`` for a numeric row.
    numeric: Optional[Tuple[float, float, float]] = None
    value: float = 0.0
    unit: str = ""

    @property
    def selection(self):
        if self.numeric is not None:
            return self.value
        if not self.options:
            return None
        return self.options[self.index % len(self.options)]

    @property
    def shown(self) -> str:
        """What the row reads as, right of its label."""
        if self.action is not None:
            return ""
        if self.numeric is not None:
            step = self.numeric[2]
            digits = 0 if step >= 1.0 else 1
            return "%.*f %s" % (digits, self.value, self.unit)
        chosen = self.selection
        return "" if chosen is None else str(chosen[1])

    def adjust(self, direction: int) -> None:
        if self.action is not None:
            return
        if self.numeric is not None:
            low, high, step = self.numeric
            self.value = min(max(self.value + direction * step, low), high)
        elif self.options:
            self.index = (self.index + direction) % len(self.options)


@dataclass
class Menu:
    """A cursor over some :class:`Choice` rows."""

    title: str
    rows: List[Choice]
    cursor: int = 0
    #: Set when an action row is chosen; the caller reads and clears it.
    chosen: Optional[str] = None

    def move(self, direction: int) -> None:
        if not self.rows:
            return
        self.cursor = (self.cursor + direction) % len(self.rows)

    def adjust(self, direction: int) -> None:
        if self.rows:
            self.rows[self.cursor].adjust(direction)

    def enter(self) -> Optional[str]:
        """Activate the row under the cursor; returns its action, if any."""
        if not self.rows:
            return None
        row = self.rows[self.cursor]
        if row.action is not None:
            self.chosen = row.action
            return row.action
        # Entering a setting row steps it, which is what a mouse-less
        # user expects when the arrows are also the adjust keys.
        row.adjust(1)
        return None

    def by_key(self, key: str) -> Optional[Choice]:
        for row in self.rows:
            if row.key == key:
                return row
        return None

    def settings(self) -> dict:
        """Every non-action row, as ``{key: value}``."""
        out = {}
        for row in self.rows:
            if row.action is not None:
                continue
            if row.numeric is not None:
                out[row.key] = row.value
            else:
                chosen = row.selection
                out[row.key] = None if chosen is None else chosen[0]
        return out


def boat_choices():
    return [(key, label) for key, label, _seats, _coxed in BOATS]


def course_choices():
    return [(key, label) for key, label, _blurb in COURSES]


def setup_menu(boat: str = "4+", course: str = "charles",
               rate: float = 30.0, wind: float = 5.0) -> Menu:
    """The menu shown before anything is built."""
    boats = boat_choices()
    courses = course_choices()
    rows = [
        Choice("boat", "Boat", boats,
               index=max([i for i, b in enumerate(boats) if b[0] == boat]
                         + [0])),
        Choice("race", "Course", courses,
               index=max([i for i, c in enumerate(courses) if c[0] == course]
                         + [0])),
        Choice("rate", "Stroke rate", (), numeric=(16.0, 44.0, 1.0),
               value=float(rate), unit="spm"),
        Choice("wind", "Wind", (), numeric=(0.0, 16.0, 1.0),
               value=float(wind), unit="m/s"),
        Choice("start", "Push off", (), action="start"),
        Choice("quit", "Quit", (), action="quit"),
    ]
    return Menu("Coxswain", rows)


def pause_menu(rate: float = 30.0, wind: float = 5.0) -> Menu:
    """The menu shown mid-outing.

    Rate and wind are live here: both can be changed without rebuilding
    the world, because neither touches the terrain.  Course and boat are
    not, because changing either means building a new world, so those
    send you back to the start.
    """
    rows = [
        Choice("resume", "Back to it", (), action="resume"),
        Choice("rate", "Stroke rate", (), numeric=(16.0, 44.0, 1.0),
               value=float(rate), unit="spm"),
        Choice("wind", "Wind", (), numeric=(0.0, 16.0, 1.0),
               value=float(wind), unit="m/s"),
        Choice("restart", "Restart this course", (), action="restart"),
        Choice("setup", "Change boat or course", (), action="setup"),
        Choice("quit", "Quit", (), action="quit"),
    ]
    return Menu("Paused", rows)


def build_boat(key: str, rate: float, catalog=None):
    """A boat from a menu key, or the nearest thing the catalog has.

    The catalog does not necessarily carry every shell in :data:`BOATS`,
    and a menu that offers a boat the code cannot build is worse than a
    short menu -- so this reports what it actually made.
    Returns ``(boat, made_key)``.
    """
    if catalog is None:
        from ..boats import catalog as catalog_module

        catalog = catalog_module
    crew = dict(rower_mass=68.0, rower_stature=1.70)
    # Only what the catalog actually exports: a menu offering a boat the
    # code cannot build is worse than a short menu.
    makers = {
        "4+": ("coxed_four", dict(coxswain_mass=68.0)),
        "8+": ("eight", dict(coxswain_mass=68.0)),
        "2x": ("double_scull", {}),
        "1x": ("single_scull", {}),
    }
    name, extra = makers.get(key, makers["4+"])
    for candidate, kwargs, made in ((name, extra, key),
                                    ("coxed_four",
                                     dict(coxswain_mass=68.0), "4+")):
        maker = getattr(catalog, candidate, None)
        if maker is None:
            continue
        try:
            return maker(rate=float(rate), **crew, **kwargs), made
        except TypeError:
            try:
                return maker(rate=float(rate)), made
            except Exception:
                continue
        except Exception:
            continue
    raise SystemExit("the catalog has no boat this menu can build")


# ---------------------------------------------------------------------------
# Drawing, and the only part that knows about pygame
# ---------------------------------------------------------------------------

#: Colours, dark enough to read over a rendered river.
INK = (233, 240, 245)
DIM = (150, 164, 176)
PICK = (255, 146, 72)
PANEL = (12, 17, 21, 226)


#: Menu music.  Mixkit item 754, "Romantic 03", under the Mixkit Free
#: License: free in commercial and non-commercial work with no
#: attribution required, but the asset may not be redistributed on its
#: own or sold as the substance of a product.  Bundled as background
#: audio in a free trainer it is a permitted use; it is credited here
#: anyway, because provenance that lives only in someone's memory is
#: provenance that is lost.
MUSIC_FILE = "menu_music.mp3"


def music_path():
    """Absolute path to the menu music, wherever it is being run from."""
    from ..core.resources import data_path

    return data_path("coxswain", "data", MUSIC_FILE)


def start_music(volume: float = 0.45) -> bool:
    """Begin looping the menu music.  ``False`` if it could not.

    Deliberately forgiving: a machine with no sound device, a mixer
    another part of the program already owns, or a missing file are all
    reasons to have no music and none of them are reasons to refuse to
    show the menu.
    """
    import os

    import pygame

    path = music_path()
    if not os.path.exists(path):
        return False
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.set_volume(float(volume))
        pygame.mixer.music.play(-1)                 # loop
    except Exception:
        return False
    return True


def stop_music(fade_ms: int = 600) -> None:
    """Fade the menu music out, if it is playing."""
    import pygame

    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.fadeout(int(fade_ms))
    except Exception:
        pass


def blurb_for(menu: "Menu") -> str:
    """A line of help for whatever the cursor is on."""
    row = menu.rows[menu.cursor] if menu.rows else None
    if row is None:
        return ""
    if row.key == "race":
        chosen = row.selection
        for key, _label, blurb in COURSES:
            if chosen is not None and key == chosen[0]:
                return blurb
    if row.key == "boat":
        chosen = row.selection
        for key, _label, seats, coxed in BOATS:
            if chosen is not None and key == chosen[0]:
                return ("%d rowing, %s" %
                        (seats, "you steer from the boat" if coxed
                         else "no coxswain -- you steer it yourself"))
    if row.key == "rate":
        return "The crew's rating.  Changes the whole cycle, including sound."
    if row.key == "wind":
        return "Sets the chop through the JONSWAP relations, as the report does."
    return ""


def draw_menu(surface, menu: "Menu", font, small, size,
              footer: str = "arrows move and change  .  enter selects  "
                            ".  escape backs out") -> None:
    """Paint ``menu`` onto a pygame surface with alpha.

    Kept apart from the model on purpose: everything about *what* the
    menu contains is testable without a display, and this function is
    the only part that needs one.
    """
    import pygame

    width, height = size
    surface.fill((0, 0, 0, 0))
    rows = len(menu.rows)
    panel_w = min(int(width * 0.62), 560)
    panel_h = 132 + rows * 40
    left = (width - panel_w) // 2
    top = (height - panel_h) // 2
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill(PANEL)
    pygame.draw.rect(panel, (70, 82, 92), panel.get_rect(), 1)
    surface.blit(panel, (left, top))

    surface.blit(font.render(menu.title, True, INK), (left + 28, top + 24))
    y = top + 74
    for index, row in enumerate(menu.rows):
        picked = index == menu.cursor
        colour = PICK if picked else (INK if row.action is None else DIM)
        if picked:
            pygame.draw.rect(surface, (30, 40, 48),
                             (left + 14, y - 6, panel_w - 28, 34))
        surface.blit(font.render(row.label, True, colour), (left + 28, y))
        shown = row.shown
        if shown:
            text = font.render(shown, True, colour)
            surface.blit(text, (left + panel_w - 28 - text.get_width(), y))
        if picked and row.action is None:
            for glyph, at in (("<", left + panel_w - 200), (">", left + 200)):
                surface.blit(small.render(glyph, True, DIM), (at, y + 4))
        y += 40

    hint = blurb_for(menu)
    if hint:
        text = small.render(hint, True, DIM)
        surface.blit(text, ((width - text.get_width()) // 2,
                            top + panel_h - 46))
    text = small.render(footer, True, DIM)
    surface.blit(text, ((width - text.get_width()) // 2,
                        top + panel_h - 26))


def handle_key(menu: "Menu", key, keys=None) -> Optional[str]:
    """Feed one pygame key to a menu; returns an action or ``None``.

    Takes the key *constants* module so the model file need not import
    pygame at all when nothing is drawing.
    """
    import pygame

    if key in (pygame.K_UP, pygame.K_w):
        menu.move(-1)
    elif key in (pygame.K_DOWN, pygame.K_s):
        menu.move(1)
    elif key in (pygame.K_LEFT, pygame.K_a):
        menu.adjust(-1)
    elif key in (pygame.K_RIGHT, pygame.K_d):
        menu.adjust(1)
    elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
        return menu.enter()
    return None
