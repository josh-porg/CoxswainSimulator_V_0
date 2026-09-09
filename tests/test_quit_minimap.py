r"""Q asks before it quits, and the minimap is a map.

Q sits between W and E on the keyboard a coxswain steers with, so a
slipped finger used to end the race.  The minimap draws under the
HUD's change key so a moving boat does not put the HUD back to an
upload every frame.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def source(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def test_the_quit_menu_offers_no_first_and_yes_second():
    from coxswain.viz.menu import confirm_quit_menu

    menu = confirm_quit_menu()
    keys = [row.key for row in menu.rows]
    actions = [row.action for row in menu.rows]
    assert keys == ["resume", "quit_yes"]
    assert actions == ["resume", "quit_yes"]
    # the cursor starts on "no": enter by reflex keeps you in the boat
    assert menu.rows[0].action == "resume"


def test_q_and_the_pause_menu_route_through_the_confirmation():
    text = source("scripts", "fpv.py")
    q = text.index("elif event.key == pygame.K_q and freecam is None:")
    body = text[q:q + 300]
    assert "confirm_quit_menu()" in body
    assert "running = False" not in body, "Q must not quit outright"
    # the pause menu's "quit" opens the same question; only "quit_yes" ends
    assert 'elif action == "quit":' in text
    assert 'elif action in ("setup", "quit_yes"):' in text
    assert 'elif action in ("setup", "quit"):' not in text


def test_the_minimap_is_offered_remembered_and_switchable():
    text = source("scripts", "fpv.py")
    menu = source("coxswain", "viz", "menu.py")
    assert '"--no-minimap"' in text
    assert 'Choice("minimap", "Minimap"' in menu
    assert '_settings.load().get("minimap") == "off"' in text
    assert text.count("_settings.update(minimap=args.minimap)") == text.count(
        "_settings.update(updates=args.updates)")


def test_the_minimap_is_part_of_the_hud_key_only_when_on_and_quantised():
    text = source("scripts", "fpv.py")
    assert "_hud_key = (tuple(lines), knob, _map_key)" in text
    # `pose`, not `state`.  This test asserted `state` -- the name that
    # does not exist in that scope -- and so pinned the v0.12 crash in
    # place instead of catching it.  A source-text test is only worth
    # having if the text it demands is the text that works.
    assert "int(pose[0] / 2.0), int(pose[1] / 2.0)" in text
    assert "int(math.degrees(pose[5]) / 5.0)" in text
    assert "state[0]" not in text[text.index("_map_key ="):
                                  text.index("_hud_key =")]
    assert "if _map_on else None" in text
    # drawn inside the change-gated compose, before the upload
    compose = text.index("if _hud_changed:")
    upload = text.index("hud_texture.write(_surface_bytes(pygame, overlay))",
                        compose)
    draw = text.index("draw_minimap(pygame, overlay, course, scene.buoys, pose",
                      compose)
    assert compose < draw < upload


@pytest.fixture()
def pygame_surface():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    pygame.init()
    yield pygame, pygame.Surface((1180, 680), pygame.SRCALPHA)
    pygame.quit()


def test_the_map_draws_in_its_box_with_the_boat_where_it_is(pygame_surface):
    import fpv

    pygame, overlay = pygame_surface
    course = np.column_stack([np.linspace(0, 4000, 50), np.linspace(0, 500, 50)])
    buoys = np.array([[1000.0, 125.0], [3000.0, 375.0]])
    state = np.zeros(12)
    state[0], state[1], state[5] = 2000.0, 250.0, 0.3
    fpv.draw_minimap(pygame, overlay, course, buoys, state, (1180, 680))
    box = fpv.MINIMAP_SIZE
    x0 = 1180 - box - fpv.MINIMAP_MARGIN
    y0 = fpv.MINIMAP_MARGIN
    arr = pygame.surfarray.pixels_alpha(overlay)
    inside = arr[x0:x0 + box, y0:y0 + box]
    outside = arr.copy(); outside[x0:x0 + box, y0:y0 + box] = 0
    assert inside.max() > 0, "nothing drawn"
    assert outside.max() == 0, "the map leaked outside its box"
    # the boat arrow is a bright mark near the middle of the course line
    rgb = pygame.surfarray.pixels3d(overlay)
    bright = np.argwhere((rgb[x0:x0 + box, y0:y0 + box, 0] > 220)
                         & (rgb[x0:x0 + box, y0:y0 + box, 1] > 220))
    assert len(bright), "no boat marker"
    centre = bright.mean(axis=0)
    assert abs(centre[0] - box / 2) < box * 0.2
