r"""The rig editor's label boxes fit between the seats.

Each box is centred on its seat, so two boxes on the same side overlap
as soon as one is taller than the seat pitch.  Adding age, skill and
experience pushed a four's box to 84 px in an 83 px pitch and the
existing test caught it -- but only for a four.  An eight has 46 px a
seat and was already overlapping with three lines, unnoticed, because
nothing ever drew one with a full crew in it.
"""

from __future__ import annotations

import os

import pytest

from coxswain.viz.rigview import (PRESETS, Lineup, Rower, draw_plan,
                                  plan_geometry)


@pytest.fixture()
def fonts():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((900, 560))
    yield (pygame,
           pygame.font.SysFont("dejavusans,arial", 20),
           pygame.font.SysFont("dejavusans,arial", 14))
    pygame.quit()


def _full_eight():
    crew = [Rower("Rower %d" % (i + 1), 165.0, 5, 11.0, "19:%02d" % (30 + i),
                  age=40 + i, skill=0.6, experience=0.5) for i in range(8)]
    return Lineup(shell="8+", rig="standard", rowers=crew).apply_rig()


def _no_overlaps(boxes):
    pairs = [(a, b) for a in boxes for b in boxes if a < b]
    return [(a, b) for a, b in pairs if boxes[a].colliderect(boxes[b])]


def test_the_plan_publishes_the_pitch_it_lays_seats_out_on():
    for lineup, seats in ((PRESETS["HOCR 4+"](), 4), (_full_eight(), 8)):
        plan = plan_geometry(lineup, 900.0, 560.0)
        hull_len = plan["hull"][3]
        assert plan["pitch"] == pytest.approx(hull_len / (seats + 1.0))
        ys = sorted(mark["seat"][1] for mark in plan["marks"])
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        assert all(g == pytest.approx(plan["pitch"]) for g in gaps)


def test_a_four_with_every_field_filled_in_still_fits(fonts):
    pygame, font, small = fonts
    lineup = PRESETS["HOCR 4+"]()
    for rower in lineup.rowers:
        rower.skill, rower.experience = 0.8, 0.7
    surface = pygame.Surface((900, 560), pygame.SRCALPHA)
    boxes = draw_plan(surface, lineup, font, small, (900, 560))
    assert len(boxes) == 4
    assert _no_overlaps(boxes) == []


def test_an_eight_with_a_full_crew_fits_too(fonts):
    """The case that was already broken before anything was added."""
    pygame, font, small = fonts
    surface = pygame.Surface((900, 560), pygame.SRCALPHA)
    boxes = draw_plan(surface, _full_eight(), font, small, (900, 560))
    assert len(boxes) == 8
    assert _no_overlaps(boxes) == []


def test_no_box_is_taller_than_the_space_it_has(fonts):
    pygame, font, small = fonts
    for lineup in (PRESETS["HOCR 4+"](), _full_eight()):
        pitch = plan_geometry(lineup, 900 * 0.62, 560)["pitch"]
        surface = pygame.Surface((900, 560), pygame.SRCALPHA)
        boxes = draw_plan(surface, lineup, font, small, (900, 560))
        for index, rect in boxes.items():
            assert rect.height <= pitch, (lineup.shell, index, rect.height,
                                          pitch)


def test_the_name_keeps_its_own_line_however_tight_it_gets(fonts):
    """Merging must never fold the name into the numbers: it is what
    tells you whose box this is."""
    from coxswain.viz.rigview import rower_lines

    pygame, font, small = fonts
    lineup = _full_eight()
    surface = pygame.Surface((900, 560), pygame.SRCALPHA)
    boxes = draw_plan(surface, lineup, font, small, (900, 560))
    # An eight's 46 px a seat holds two tight lines, not two loose
    # ones, so the bound is the glyphs and not the padded leading.
    assert all(rect.height >= 2 * small.get_height()
               for rect in boxes.values())
    assert rower_lines(lineup.rowers[0])[0] == "Rower 1"


def test_typing_keeps_every_field_on_its_own_line(fonts):
    """The edited box is not compressed -- you cannot edit a field you
    cannot see -- so it is allowed to be taller than the pitch."""
    pygame, font, small = fonts
    lineup = PRESETS["HOCR 4+"]()
    surface = pygame.Surface((900, 560), pygame.SRCALPHA)
    plain = draw_plan(surface, lineup, font, small, (900, 560))
    edited = draw_plan(surface, lineup, font, small, (900, 560),
                       editing=(0, 1, "125"))
    assert edited[0].height > plain[0].height
    assert edited[1].height == plain[1].height, "only the edited one grows"
