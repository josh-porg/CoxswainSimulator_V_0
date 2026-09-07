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
        boat="4+", race="charles", rate=30.0, wind=5.0)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert fpv.run_setup_menu(dummy_video, args) is None


def test_setup_menu_hands_back_the_choices(dummy_video):
    """Enter on "Push off" yields settings the caller can use."""
    import fpv

    args = fpv.main.__globals__["argparse"].Namespace(
        boat="8+", race="totl", rate=28.0, wind=3.0)
    for _ in range(4):                      # down to the "Push off" row
        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_DOWN))
    pygame.event.post(pygame.event.Event(
        pygame.KEYDOWN, key=pygame.K_RETURN))
    picked = fpv.run_setup_menu(dummy_video, args)
    assert picked is not None, "enter on the action row should start"
    assert picked["boat"] == "8+" and picked["race"] == "totl"
    assert picked["rate"] == 28.0 and picked["wind"] == 3.0


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
