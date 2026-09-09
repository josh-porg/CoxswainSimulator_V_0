r"""Age, skill and experience per rower, and full screen.

The crew sliders move everybody at once.  These are the per-seat
overrides: a seat left alone still follows the slider, and one with a
number typed into it does not.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from coxswain.viz.rigview import (FIELDS, PRESETS, Lineup, Rower, field_text,
                                  parse_field, rower_lines)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def _index(name):
    return [f[0] for f in FIELDS].index(name)


def test_the_editor_offers_the_three_new_fields():
    for name in ("age", "skill", "experience"):
        assert name in [f[0] for f in FIELDS]
    assert [f[0] for f in FIELDS][:5] == ["name", "pounds", "feet", "inches",
                                          "erg_5k"], "the old order is kept"


def test_an_untouched_rower_defers_to_the_crew_slider():
    rower = Rower("nobody", 150.0, 5, 10.0, "")
    assert rower.age == 0
    assert rower.skill < 0.0 and rower.experience < 0.0
    # and shows blank rather than a number nobody typed
    assert field_text(rower, _index("skill")) == ""
    assert field_text(rower, _index("experience")) == ""
    assert field_text(rower, _index("age")) == ""


def test_the_fields_refuse_what_is_not_a_value():
    assert parse_field(_index("age"), "62") == 62
    assert parse_field(_index("age"), "150") is None
    assert parse_field(_index("age"), "3") is None
    assert parse_field(_index("age"), "abc") is None
    assert parse_field(_index("age"), "") == 0
    assert parse_field(_index("skill"), "0.8") == pytest.approx(0.8)
    assert parse_field(_index("skill"), "1") == pytest.approx(1.0)
    assert parse_field(_index("skill"), "2") is None
    assert parse_field(_index("skill"), "-1") is None
    # blank puts it back to the crew slider, which is NOT zero: zero is
    # "novice", and a cleared field must not silently make somebody one
    assert parse_field(_index("skill"), "") == -1.0
    assert parse_field(_index("experience"), "") == -1.0


def test_the_box_shows_them_only_when_they_are_set():
    rower = Rower("A", 150.0, 5, 10.0, "20:00")
    assert not any("age" in line for line in rower_lines(rower))
    rower.age, rower.skill = 62, 0.8
    line = [l for l in rower_lines(rower) if "age" in l][0]
    assert "age 62" in line and "skill 0.80" in line and "exp" not in line


def test_they_survive_the_round_trip():
    lineup = PRESETS["HOCR 4+"]()
    lineup.rowers[0].skill = 0.9
    lineup.rowers[1].experience = 0.2
    data = lineup.to_dict()
    back = Lineup.from_dict(data)
    assert back.to_dict() == data
    assert back.rowers[0].skill == pytest.approx(0.9)
    assert back.rowers[1].experience == pytest.approx(0.2)
    assert back.rowers[0].age == lineup.rowers[0].age
    # a file written before these existed still loads
    old = Lineup.from_dict({"name": "old", "shell": "4+",
                            "rowers": [{"name": "a"}]})
    assert old.rowers[0].age == 0 and old.rowers[0].skill == -1.0


def test_the_boat_carries_the_per_seat_numbers():
    from coxswain.viz.menu import build_boat

    lineup = PRESETS["HOCR 4+"]()
    lineup.rowers[0].skill = 0.9
    lineup.rowers[1].experience = 0.2
    boat, _made = build_boat("4+", 30.0, lineup=lineup)
    assert boat.seat_skill[0] == pytest.approx(0.9)
    assert np.all(boat.seat_skill[1:] < 0.0)
    assert boat.seat_experience[1] == pytest.approx(0.2)
    assert boat.seat_ages == [62, 62, 62, 62]


def test_a_steadier_seat_scatters_less_and_an_untouched_one_is_untouched():
    from coxswain.crew.variability import for_skill, seat_scale_for_skill

    scale = seat_scale_for_skill([0.9, -1.0, -1.0, 0.0], 0.55)
    assert scale[0] < 1.0, "an elite seat in a club crew is steadier"
    assert scale[1] == 1.0 and scale[2] == 1.0, "untouched is exactly 1.0"
    assert scale[3] > 1.0, "a novice seat in a club crew is worse"
    # nobody set anything: None, so the caller takes the untouched path
    assert seat_scale_for_skill([-1.0] * 4, 0.55) is None
    # the scale is the ratio of that skill's scatter to the crew's
    assert scale[0] == pytest.approx(
        for_skill(0.9).power_sigma / for_skill(0.55).power_sigma)


def test_the_scatter_scales_the_deviation_not_the_value():
    """Scaling the drawn power itself would move the seat's MEAN, which
    is a different rower, not a steadier one."""
    from coxswain.boats import catalog
    from coxswain.crew.variability import for_skill

    boat = catalog.coxed_four(rate=30.0)
    variability = for_skill(0.35)
    variability.apply(boat, seat_scale=np.zeros(boat.n_seats))
    assert np.allclose(boat.power_scales, 1.0), "zero scatter is exactly 1.0"
    assert np.allclose(boat.phase_offsets, 0.0)


def test_the_crews_experience_averages_the_seats_that_have_one():
    text = source("scripts", "fpv.py")
    assert "_seat_exp[_seat_exp >= 0.0]" in text
    assert "float(_given.mean()) if len(_given) else float(args.balance)" in text
    assert "cox.balance = balance_for_experience(boat, _crew_exp)" in text


# ---------------------------------------------------------------------------
# full screen
# ---------------------------------------------------------------------------
def test_fullscreen_is_offered_remembered_and_toggled():
    text = source("scripts", "fpv.py")
    menu = source("coxswain", "viz", "menu.py")
    assert '"--fullscreen"' in text
    assert 'Choice("fullscreen", "Full screen"' in menu
    assert '_settings.load().get("fullscreen") == "on"' in text
    assert "elif event.key == pygame.K_F11:" in text
    assert "_settings.update(fullscreen=args.fullscreen)" in text
    assert "draw.hud_last = None          # recompose at the new size" in text
    assert "F11 full screen" in text, "the controls card says so"


def test_the_flags_are_right_and_a_refusal_falls_back_to_a_window():
    import sys

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    pygame = pytest.importorskip("pygame")
    import fpv

    windowed = fpv.display_flags(pygame, "off")
    full = fpv.display_flags(pygame, "on")
    assert windowed == pygame.OPENGL | pygame.DOUBLEBUF
    assert full & pygame.FULLSCREEN and full & pygame.SCALED
    assert full & pygame.OPENGL and full & pygame.DOUBLEBUF
    # the fallback is in the code that opens it
    text = source("scripts", "fpv.py")
    body = text[text.index("def set_display_mode("):]
    body = body[:body.index("\n\n\n")]
    assert "no fullscreen mode here" in body
    assert 'args.fullscreen = "off"' in body, "and it stops claiming to be on"
