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
           "start_music", "stop_music", "music_path",
           "chart_surface", "draw_controls", "CONTROLS",
           "options_menu", "quality_settings", "QUALITY", "QUALITY_TIERS",
           "Tier", "tier_settings", "AUDIO_MODES",
           "WEATHERS", "weather_choices", "weather_menu", "rowers_menu"]


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
        namer = getattr(self, "shown_as", None)
        if namer is not None:
            # A slider whose number means nothing to the reader.  "0.75"
            # is not a crew; "elite" is.
            return "%s" % namer(self.value)
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


#: Sound modes, as ``(key, label)``.
AUDIO_MODES = (("full", "Full"), ("events", "Events only"), ("off", "Off"))

#: Graphics presets, as ``(key, label, water_divisions, trees)``.
#:
#: Splash droplets at the catch are HIGH only.  They are the least
#: important thing on the screen -- a coxswain has never once steered by
#: them -- and they are the only per-frame geometry that is rebuilt and
#: re-uploaded every frame regardless of how little of it there is.  On
#: a part that is struggling, they are the first thing to go.
#:
#: Three things are traded.  The water grid is a quarter of a million
#: triangles at Standard and the only part of the scene rebuilt every
#: frame.  And **Minimal keeps the original water shader** -- a Fresnel
#: mix between a deep colour and a flat sky, opaque, reflecting nothing
#: -- while Standard and High run the second pass: screen-space
#: refraction with depth absorption, and a reflection distorted by the
#: waves.  Minimal also drops the distant trees.
from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class Tier:
    """Everything a graphics setting decides, in one place.

    Each field is a cost the renderer pays every frame or at start-up,
    and every one of them scales down a tier at a time -- so an old
    laptop with integrated graphics has a setting that is actually
    cheaper, not merely a smaller water patch with the same 1.3 M
    triangles behind it.

    ``ultra`` is the floor: flat water, no screen-space reflection or
    refraction, one shadow tap from a smaller baked map, a plain
    distance fog, no distant skyline, every tree an impostor, a 400 m
    reach at a coarser terrain step, no multisampling, and the scene
    drawn at full size -- a smaller buffer cost more than it saved
    on the integrated part it was measured on.  It is meant to run on an
    integrated GPU from 2015.
    """

    key: str
    label: str
    water_divisions: int         # the moving water patch, per side
    trees: str                   # "full" | "impostor" | "off"
    rich_water: bool             # screen-space reflection + refraction
    particles: bool              # catch and drag splash
    shadow: str                  # "pcf" | "single" | "off"
    shadow_size: int             # baked map, one side; 0 = auto
    reflect_steps: int           # SSR march length; 0 = sky only
    fog: str                     # "full" | "simple"
    skyline: bool                # the distant-city ring
    reach: float                 # metres either side of the course
    step: float                  # terrain step, m
    samples: int                 # MSAA on the scene buffer
    #: Scene drawn at this fraction of the window, then stretched.  1.0
    #: on every tier: measured on an Intel UHD, drawing at 0.75 into an
    #: off-screen buffer cost MORE than the pixels saved (44.7 ms against
    #: 27.8) -- the extra pass and copy outweigh fill on a part whose
    #: bottleneck is not fill.  Kept as a lever (``--render-scale``) for
    #: a GPU where the balance differs; nothing defaults to it until one
    #: is measured.
    render_scale: float
    exact_within: float          # exact water normals inside this, m
    physics_hz: float = 60.0     # integrator rate; 60 == 100 to 0.2 mm

    #: Simplex on the overcast dome, and therefore in every water
    #: pixel's reflection.  Off at the low tiers: it was a
    #: measurable share of a 5-7 ms water pass on an integrated
    #: part, for a texture those tiers are not there to show.
    sky_detail: bool = True
    #: The integrator: "heun" at two derivative evaluations a step,
    #: "rk4" at four.  Heun at 60 Hz is the same boat to 3 cm over
    #: 367 m; High keeps RK4 because it is the reference and can
    #: afford it.
    physics_scheme: str = "rk4"

QUALITY_TIERS = (
    Tier("ultra", "Ultra minimal (lowest)", 80, "impostor", False, False, "single",
         2048, 0, "simple", False, 400.0, 12.0, 0, 1.0, 5.0, 50.0, sky_detail=False, physics_scheme="heun"),
    Tier("minimal", "Minimal", 120, "impostor", False, False, "single",
         2048, 0, "simple", False, 600.0, 10.0, 0, 1.0, 7.0, 60.0, sky_detail=False, physics_scheme="heun"),
    Tier("standard", "Standard", 240, "full", True, False, "pcf",
         0, 10, "full", True, 900.0, 8.0, 2, 1.0, 16.0, 60.0, physics_scheme="heun"),
    Tier("high", "High", 420, "full", True, True, "pcf",
         0, 16, "full", True, 900.0, 8.0, 4, 1.0, 26.0, 100.0),
)

#: The old four-tuple view, kept for the callers that read it.
QUALITY = tuple((t.key, t.label, t.water_divisions, t.trees != "off",
                 t.rich_water, t.particles) for t in QUALITY_TIERS)


def tier_settings(key: str) -> Tier:
    """The full :class:`Tier` for a preset key; ``standard`` if unknown."""
    for tier in QUALITY_TIERS:
        if tier.key == key:
            return tier
    return QUALITY_TIERS[2]


def audio_choices():
    return [(key, label) for key, label in AUDIO_MODES]


def quality_choices():
    return [(key, label) for key, label, _d, _t, _r, _p in QUALITY]


def quality_settings(key: str):
    """``(water_divisions, trees, rich_water, particles)`` for a preset."""
    for name, _label, divisions, trees, rich, particles in QUALITY:
        if name == key:
            return divisions, bool(trees), bool(rich), bool(particles)
    return 340, True, True, False


#: Weather, as ``(key, label)``.  What the air is doing, which on a
#: river is most of what there is to look at.
WEATHERS = (("clear", "Clear"), ("hazy", "Hazy"),
            ("overcast", "Overcast"), ("fog", "Fog"))


def weather_choices():
    return [(key, label) for key, label in WEATHERS]


def rowers_menu(skill: float = 0.55, balance: float = 0.55) -> Menu:
    """The crew itself: how good they are, not what they are sitting in.

    Separate from the boat because it is a different question.  The boat
    is what you were given; the crew is who turned up, and a coxswain
    knows perfectly well that the same four can be a different boat on a
    different morning.
    """
    from ..crew.variability import skill_label

    rows = [
        Choice("skill", "Crew skill", (), numeric=(0.0, 1.0, 0.05),
               value=float(skill), unit=""),
        Choice("balance", "Balance", (), numeric=(0.0, 1.0, 0.05),
               value=float(balance), unit=""),
        Choice("back", "Back", (), action="back"),
    ]
    rows[0].shown_as = skill_label
    rows[1].shown_as = skill_label
    return Menu("Rowers", rows)


def weather_menu(weather: str = "hazy", wind: float = 5.0) -> Menu:
    """The air, on its own.

    Kept apart from graphics and sound because it is not a preference
    about how the program draws: it is a condition of the outing, like
    the course.  Wind belongs here and nowhere else -- it sets the chop,
    the micro-ripple, the shelter behind the hull and which way the boat
    gets pushed.
    """
    rows = [
        Choice("weather", "Sky", weather_choices(),
               index=max([i for i, w in enumerate(WEATHERS)
                          if w[0] == weather] + [0])),
        Choice("wind", "Wind", (), numeric=(0.0, 16.0, 1.0),
               value=float(wind), unit="m/s"),
        Choice("back", "Back", (), action="back"),
    ]
    return Menu("Weather", rows)


REPORT_CHOICES = (("off", "Off"), ("on", "On"))


def options_menu(audio: str = "full", quality: str = "standard",
                 weather: str = "hazy", wind: float = 5.0,
                 report: str = "off", updates: str = "on") -> Menu:
    """Graphics and sound, reached from either menu."""
    modes = audio_choices()
    grades = quality_choices()
    rows = [
        Choice("audio", "Sound", modes,
               index=max([i for i, m in enumerate(modes) if m[0] == audio]
                         + [0])),
        Choice("quality", "Graphics", grades,
               index=max([i for i, q in enumerate(grades) if q[0] == quality]
                         + [0])),
        Choice("report", "Send performance reports", list(REPORT_CHOICES),
               index=1 if report == "on" else 0),
        Choice("updates", "Check for updates", list(REPORT_CHOICES),
               index=1 if updates == "on" else 0),
        Choice("back", "Back", (), action="back"),
    ]
    return Menu("Graphics and sound", rows)


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
        Choice("start", "Push off", (), action="start"),
        Choice("rowers", "Rowers", (), action="rowers"),
        Choice("crew", "Rig and crew", (), action="crew"),
        Choice("weather", "Weather", (), action="weather"),
        Choice("options", "Graphics and sound", (), action="options"),
        Choice("controls", "Controls", (), action="controls"),
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
        Choice("restart", "Restart this course", (), action="restart"),
        Choice("setup", "Change boat or course", (), action="setup"),
        Choice("rowers", "Rowers", (), action="rowers"),
        Choice("weather", "Weather", (), action="weather"),
        Choice("options", "Graphics and sound", (), action="options"),
        Choice("controls", "Controls", (), action="controls"),
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
PANEL = (12, 17, 21, 188)


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


#: The keys, in one place, so the menu and the docs cannot disagree.
CONTROLS = (
    ("Steering", ""),
    ("left / right, A / D", "the stick, by key"),
    ("mouse", "the stick, with mouse steering on"),
    ("M", "hand steering between mouse and keys"),
    ("C", "centre the stick"),
    ("", ""),
    ("Crew", ""),
    ("W / E", "pressure split -- more work on one side"),
    ("", ""),
    ("Crew", ""),
    ("Up / Down", "call for more or less power"),
    ("", "the crew give what you ask until the reserve is gone"),
    ("View", ""),
    ("V", "look over your shoulder"),
    ("F1", "free camera -- fly around to inspect the course"),
    ("  WASD / QE", "move and rise, while flying"),
    ("  shift", "faster, while flying"),
    ("", ""),
    ("Session", ""),
    ("Esc", "the menu: rate, wind, restart, change boat"),
    ("Space", "freeze"),
    ("R", "restart the course"),
    ("Q", "quit"),
)

#: Which file the depth soundings come from, per course.
CHART_DATA = {
    "charles": ("data", "charles_isobaths.csv"),
    "totl": ("coxswain", "data", "lake_union_depth.npz"),
    "hotl": ("coxswain", "data", "lake_union_depth.npz"),
}

_CHARTS: dict = {}


def chart_surface(race: str, size):
    """A faint bathymetric chart of the course, to sit behind the menu.

    Real soundings rather than a picture of some water: the Charles
    isobaths and the Lake Union depth grid are already in the build,
    because the courses are laid out against them.  A flat panel of
    colour behind the menu is not *wrong*, but this costs one pass over
    a point cloud and tells you something true about where you are
    about to row.

    Cached per course and size.  Returns ``None`` if the data is not
    there, and the caller falls back to plain colour.
    """
    import numpy as np
    import pygame

    key = (race, tuple(size))
    if key in _CHARTS:
        return _CHARTS[key]

    from ..core.resources import data_path

    parts = CHART_DATA.get(race)
    surface = None
    try:
        path = data_path(*parts) if parts else None
        if path is None or not __import__("os").path.exists(path):
            raise FileNotFoundError(path)
        if path.endswith(".npz"):
            blob = np.load(path)
            points = np.asarray(blob["depth_xy"], dtype=float)
            depth = np.asarray(blob["depth"], dtype=float)
        else:
            raw = np.genfromtxt(path, delimiter=",", names=True)
            points = np.column_stack([raw["lon"], raw["lat"]]).astype(float)
            depth = np.asarray(raw["depth_m"], dtype=float)

        width, height = size
        surface = pygame.Surface(size)
        surface.fill((13, 18, 23))
        low, high = np.nanmin(points, axis=0), np.nanmax(points, axis=0)
        span = np.maximum(high - low, 1e-9)
        scale = 0.92 * min(width / span[0], height / span[1])
        centre = 0.5 * (low + high)
        screen_x = (width / 2 + (points[:, 0] - centre[0]) * scale)
        screen_y = (height / 2 - (points[:, 1] - centre[1]) * scale)

        deep = np.nanmax(depth) if np.isfinite(depth).any() else 1.0
        shade = np.clip(np.abs(depth) / max(abs(deep), 1e-6), 0.0, 1.0)
        inside = ((screen_x >= 0) & (screen_x < width)
                  & (screen_y >= 0) & (screen_y < height))
        for x, y, value in zip(screen_x[inside], screen_y[inside],
                               shade[inside]):
            tone = (int(26 + 34 * (1.0 - value)),
                    int(46 + 58 * (1.0 - value)),
                    int(62 + 74 * (1.0 - value)))
            surface.fill(tone, (int(x), int(y), 2, 2))
    except Exception:
        surface = None

    _CHARTS[key] = surface
    return surface


def draw_controls(surface, font, small, size) -> None:
    """The key list, over whatever is behind it."""
    import pygame

    width, height = size
    surface.fill((0, 0, 0, 0))
    panel_w = min(int(width * 0.78), 620)
    panel_h = min(int(height * 0.88), 34 * len(CONTROLS) + 96)
    left, top = (width - panel_w) // 2, (height - panel_h) // 2
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill(PANEL)
    pygame.draw.rect(panel, (70, 82, 92), panel.get_rect(), 1)
    surface.blit(panel, (left, top))

    surface.blit(font.render("Controls", True, INK), (left + 28, top + 20))
    y = top + 62
    step = max(18, (panel_h - 108) // max(len(CONTROLS), 1))
    for key, what in CONTROLS:
        if not key and not what:
            y += step // 2
            continue
        if not what:                      # a heading
            surface.blit(small.render(key.upper(), True, PICK),
                         (left + 28, y))
        else:
            surface.blit(small.render(key, True, INK), (left + 44, y))
            surface.blit(small.render(what, True, DIM), (left + 250, y))
        y += step
    text = small.render("any key to go back", True, DIM)
    surface.blit(text, ((width - text.get_width()) // 2,
                        top + panel_h - 28))


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
        return ("Sets the chop through the JONSWAP relations, and the "
                "ripple with it.  At nothing, glass.")
    if row.key == "skill":
        return ("How consistent the crew is, stroke to stroke.  Elite and "
                "junior are measured; novice is extrapolated.")
    if row.key == "balance":
        return ("How well they sit the boat.  A novice crew checks it "
                "down; an experienced one holds it level through the "
                "recovery.")
    if row.key == "audio":
        return ("Full is the measured envelope; Events is catches and "
                "finishes only.")
    if row.key == "crew":
        return "Who sits where, on which side, and how the boat is rigged."
    if row.key == "quality":
        return ("Ultra minimal runs on integrated graphics; High wants a "
                "gaming card.  Applies on the next start.")
    if row.key == "updates":
        return ("Ask GitHub once at start whether a newer release exists, "
                "and say so here.  Nothing is downloaded; the link is "
                "the same one you were sent.")
    if row.key == "report":
        return ("At the end of a session, send frame times, GPU and tier "
                "home -- numbers and product names only, never a name or "
                "a path.  Remembered.")
    if row.key == "weather":
        return ("Sky, visibility and how the light scatters.  Fog takes "
                "the far bank out.")
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
