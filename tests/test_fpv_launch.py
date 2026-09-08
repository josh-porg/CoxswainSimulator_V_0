r"""The launch path that opens a window, which ``--shot`` never touches.

This file exists because of a real escape.  ``main`` imports pygame
inside itself, and the setup-menu block added later ran *above* that
import -- which makes ``pygame`` a local read before assignment, so
every ordinary launch died with::

    UnboundLocalError: cannot access local variable 'pygame'

Nothing caught it.  The packaging tests all pass ``--shot``, and that
skips the menu entirely, so the one path a coxswain actually uses was
the one path never run.  These tests drive it headless through SDL's
dummy video driver instead.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

pygame = pytest.importorskip("pygame")


@pytest.fixture()
def dummy_video():
    """A pygame display that needs no screen."""
    previous = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    try:
        yield pygame.display.set_mode((320, 200))
    finally:
        pygame.quit()
        if previous is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = previous


def test_setup_menu_runs_and_can_be_closed(dummy_video):
    """The menu draws and returns, rather than raising on a bare name.

    Closing the window returns ``None``; the point is that it gets far
    enough to *have* a return value.
    """
    import fpv

    args = fpv.main.__globals__["argparse"].Namespace(
        boat="4+", race="charles", rate=30.0, wind=5.0, audio="full",
        quality="standard", weather="hazy", no_sound=True)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert fpv.run_setup_menu(dummy_video, args) is None


def test_setup_menu_hands_back_the_choices(dummy_video):
    """Enter on "Push off" yields settings the caller can use."""
    import fpv

    args = fpv.main.__globals__["argparse"].Namespace(
        boat="8+", race="totl", rate=28.0, wind=3.0, audio="full",
        quality="standard", weather="hazy", no_sound=True)
    # Walk to "Push off" by name rather than by counting keystrokes:
    # counting broke the moment a row was added, which is a test
    # reporting the menu's shape rather than its behaviour.
    from coxswain.viz.menu import setup_menu
    start_row = [row.key for row in setup_menu().rows].index("start")
    for _ in range(start_row):
        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_DOWN))
    pygame.event.post(pygame.event.Event(
        pygame.KEYDOWN, key=pygame.K_RETURN))
    picked = fpv.run_setup_menu(dummy_video, args)
    assert picked is not None, "enter on the action row should start"
    assert picked["boat"] == "8+" and picked["race"] == "totl"
    assert picked["rate"] == 28.0


def test_main_opens_the_menu_without_an_unbound_name(dummy_video,
                                                    monkeypatch):
    """``main`` must survive the lines that run *before* the menu.

    The bug lived in ``main`` itself -- ``pygame.init()`` above the
    function's own ``import pygame`` -- so stubbing the menu out still
    exercises it, and does so without building a world or needing a GL
    context.  Returning ``None`` from the stub is the "window closed"
    path, which makes ``main`` return 0 immediately.
    """
    import fpv

    monkeypatch.setattr(fpv, "run_setup_menu", lambda screen, args: None)
    assert fpv.main(["--race", "totl"]) == 0


def test_loading_screen_repaints_and_reports_failure(dummy_video):
    """The build runs on a thread and the screen keeps drawing.

    Two things matter and both have bitten already.  The window must
    repaint, because a frozen one is titled "not responding" by Windows
    and reads as a crash.  And a build that raises must raise *here*
    rather than being swallowed on the worker thread, leaving a bar
    sliding forever over a program that has already failed.

    This also covers the same unbound-name trap as the tests above:
    fpv.py has no module-level pygame, so every function that draws
    needs its own import, and run_loading was written without one.
    """
    import time

    import fpv

    frames = {"n": 0}
    original = pygame.display.flip

    def counted():
        frames["n"] += 1
        original()

    pygame.display.flip = counted
    try:
        assert fpv.run_loading(dummy_video, "Building",
                               lambda: time.sleep(0.6) or "built") == "built"
        assert frames["n"] > 8, frames["n"]

        class Boom(Exception):
            pass

        def explode():
            raise Boom("build failed")

        with pytest.raises(Boom):
            fpv.run_loading(dummy_video, "Building", explode)
    finally:
        pygame.display.flip = original


def test_weather_and_back_do_not_crash_the_setup_menu(dummy_video):
    """Open Weather from the setup menu, come Back, then push off.

    "Back" from the weather menu crashed the game: the setup menu is
    rebuilt from its own remembered settings, and it looked there for a
    wind it no longer carries -- wind had moved to the weather menu --
    so every exit from Weather was a KeyError.  This walks that exact
    path by key, so the row order can change without the test lying.
    """
    import fpv
    from coxswain.viz.menu import setup_menu, weather_menu

    args = fpv.main.__globals__["argparse"].Namespace(
        boat="4+", race="charles", rate=30.0, wind=5.0, audio="full",
        quality="standard", weather="hazy", no_sound=True)

    def rows_of(menu):
        return [row.key for row in menu.rows]

    def press(key):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))

    # Down to "Weather", Enter.
    for _ in range(rows_of(setup_menu()).index("weather")):
        press(pygame.K_DOWN)
    press(pygame.K_RETURN)
    # Down to "Wind", turn it up one, then down to "Back", Enter.
    weather_rows = rows_of(weather_menu())
    for _ in range(weather_rows.index("wind")):
        press(pygame.K_DOWN)
    press(pygame.K_RIGHT)
    for _ in range(weather_rows.index("back") - weather_rows.index("wind")):
        press(pygame.K_DOWN)
    press(pygame.K_RETURN)
    # Back on the setup menu: down to "Push off", Enter.
    for _ in range(rows_of(setup_menu()).index("start")):
        press(pygame.K_DOWN)
    press(pygame.K_RETURN)

    picked = fpv.run_setup_menu(dummy_video, args)
    assert picked is not None, "the menu should have started, not died"
    assert picked["boat"] == "4+" and picked["race"] == "charles"
    # And the wind the weather menu set survived the trip.
    assert args.wind == 6.0


def test_headless_draws_every_frame_so_catches_can_fire():
    """``--shot`` with ``--frames`` must draw each step, not just once.

    Catches -- and both effects keyed off one, the puddle trail and the
    splash -- are detected by watching the stroke phase WRAP between one
    draw() call and the next.  Advancing N steps and drawing once can
    never see a wrap: ``draw.last_phase`` starts at 0.0 and the single
    phase it is compared against is never negative, so the condition is
    dead.  For a long time a ``--shot`` of a boat mid-stroke was quietly
    incapable of showing either effect, and it looked like the effects
    were broken rather than the harness.
    """
    with open(os.path.join(ROOT, "scripts", "fpv.py"),
              encoding="utf-8") as handle:
        source = handle.read()
    block = source[source.rindex("if headless:"):]
    block = block[:block.index("from PIL import Image")]
    loop = block[block.index("for _ in range(int(args.frames"):]
    assert "draw(" in loop, (
        "the headless frame loop must call draw() each step, or no "
        "catch can ever be detected in a --shot")
