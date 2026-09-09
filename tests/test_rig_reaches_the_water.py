r"""The rig you picked is the rig that rows.

Reported from the boat: a bucket-rigged four was selected and a
standard-rigged one rendered.  It was not the renderer -- the renderer
draws whatever ``boat.rig`` says.  The lineup never got that far: the
setup menu put it in its return value and ``main`` read ``boat``,
``race`` and ``rate`` out of that dict and nothing else, so the plan
was drawn, discarded, and the catalogue's default crew raced.

The tests that were supposed to cover this asserted that the two ends
existed in the source -- ``picked["lineup"] = lineup_last`` and
``lineup=getattr(args, "lineup", None)`` -- and both did.  They were
simply not connected to each other.  So these check the connection by
running it, not by reading it.
"""

from __future__ import annotations

import argparse
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from coxswain.boats.rig import RIG_PATTERNS, RIG_PATTERNS_4   # noqa: E402
from coxswain.viz.menu import build_boat                      # noqa: E402
from coxswain.viz.rigview import PRESETS, RIGS, Lineup, Rower  # noqa: E402


def _move(here: int, there: int, pygame):
    """Key presses that take a wrapping menu cursor from ``here`` to
    ``there`` -- the short way round, and never a full lap."""
    step = there - here
    key = pygame.K_DOWN if step > 0 else pygame.K_UP
    return [key] * abs(step)


def drawn_sides(boat):
    """The sides the renderer will actually draw, off the built boat."""
    return tuple(int(lock.side)
                 for seat in boat.rig.seats for lock in seat.oarlocks)


@pytest.mark.parametrize("rig", RIGS[4])
def test_every_four_rig_rows_as_itself(rig):
    lineup = PRESETS["HOCR 4+"]()
    lineup.set_rig(rig)
    boat, _made = build_boat("4+", 30.0, lineup=lineup)
    assert drawn_sides(boat) == tuple(RIG_PATTERNS_4[rig]), rig
    assert drawn_sides(boat) == tuple(r.side for r in lineup.rowers), rig


@pytest.mark.parametrize("rig", RIGS[8])
def test_every_eight_rig_rows_as_itself(rig):
    lineup = Lineup(shell="8+", rig=rig,
                    rowers=[Rower("r%d" % i, 170.0, 6, 0.0, "18:30")
                            for i in range(8)]).apply_rig()
    boat, _made = build_boat("8+", 32.0, lineup=lineup)
    assert drawn_sides(boat) == tuple(RIG_PATTERNS[rig]), rig


def test_a_bucket_four_is_not_a_standard_four():
    """The report, stated as the thing that must not happen again."""
    bucket = PRESETS["HOCR 4+"]()
    bucket.set_rig("bucket, stbd stroke")
    boat, _made = build_boat("4+", 30.0, lineup=bucket)
    assert drawn_sides(boat) == (-1, +1, +1, -1)
    assert drawn_sides(boat) != tuple(RIG_PATTERNS_4["standard"])


def test_the_setup_menu_hands_the_lineup_to_the_race(monkeypatch):
    """Drive the real setup menu, pick a rig, leave, and go.

    One event per frame, as the rig-editor test does: posting them all
    at once drains the queue in a single iteration and proves nothing.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((900, 560))
    import fpv
    from coxswain.viz.menu import setup_menu

    rows = [row.key for row in setup_menu().rows]
    from coxswain.viz.rigview import PANE_ROWS

    pane = [k for k, _label in PANE_ROWS]
    # into the crew editor; down to Rig and change it; down to Done and
    # take it; then back up to Start and go.  Counted off the tables
    # rather than guessed, so inserting a row cannot silently turn this
    # into a walk that ends somewhere else and skips.
    script = ([pygame.K_DOWN] * rows.index("crew") + [pygame.K_RETURN]
              + [pygame.K_DOWN] * pane.index("rig") + [pygame.K_RIGHT]
              + [pygame.K_DOWN] * (pane.index("done") - pane.index("rig"))
              + [pygame.K_RETURN]
              # and back to Start.  A SIGNED step from where the cursor
              # actually is: the menu wraps, so "press Up as many times
              # as there are rows" returns to exactly where it started,
              # which is how the first version of this walked onto
              # Controls and skipped instead of failing.
              + _move(rows.index("crew"), rows.index("start"), pygame)
              + [pygame.K_RETURN])

    step = {"i": 0}
    original = pygame.event.get

    def feed():
        i = step["i"]; step["i"] += 1
        if i >= len(script):
            return [pygame.event.Event(pygame.QUIT)]
        return [pygame.event.Event(pygame.KEYDOWN, key=script[i])]

    args = argparse.Namespace(
        boat="4+", race="charles", rate=30.0, wind=5.0, audio="full",
        quality="standard", weather="hazy", skill=0.55, balance=0.55,
        no_sound=True, report="off", updates="off", minimap="on",
        bonus="off", bonus_unlocked=False, fullscreen="off")
    pygame.event.get = feed
    try:
        picked = fpv.run_setup_menu(pygame.display.get_surface(), args)
    finally:
        pygame.event.get = original
        pygame.quit()

    assert picked is not None, "the scripted walk never reached Start"
    assert "lineup" in picked, "the menu dropped the lineup on the floor"
    lineup = picked["lineup"]
    assert lineup is not None, "the editor's boat never left the menu"
    # and that lineup builds the boat it describes
    boat, _made = build_boat(lineup.shell, 30.0, lineup=lineup)
    assert drawn_sides(boat) == tuple(r.side for r in lineup.rowers)
    # the walk pressed Right on the Rig row, so it is NOT the preset's
    assert lineup.rig != PRESETS["HOCR 4+"]().rig, lineup.rig


def test_opening_a_submenu_does_not_throw_the_lineup_away():
    """``args.lineup`` was assigned from a sub-menu's own settings,
    which have no lineup in them -- so Graphics and sound wiped it."""
    text = open(os.path.join(ROOT, "scripts", "fpv.py"),
                encoding="utf-8").read()
    # exactly one assignment, and it is in main, off the menu's return
    assert text.count("args.lineup = picked.get(\"lineup\")") == 1
    at = text.index("args.lineup = picked.get(\"lineup\")")
    main_at = text.index("picked = run_setup_menu(screen, args)")
    assert at > main_at, "the assignment must be main's, not a submenu's"
    assert "args.lineup = None" not in text
