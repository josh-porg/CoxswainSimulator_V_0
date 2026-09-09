r"""The seating plan: who is on which side, and where it draws.

The side a rower is on is the thing most often got wrong when a lineup
is typed in, and it is invisible in a list of names.  These hold the
plan honest so the picture cannot disagree with the boat.
"""

from __future__ import annotations

import pytest

from coxswain.viz.rigview import (PRESETS, RIGS, SHELLS, Lineup, Rower,
                                  erg_stamp, erg_watts, plan_geometry,
                                  rig_sides)


def test_the_bucket_preset_is_actually_a_bucket():
    """S-P-P-S from the stroke seat, with a starboard stroke.

    The rig this project's own boat is rigged in.  A bucket puts the
    middle pair on the same side, which is the whole point of it: it
    moves the stagger couple that alternate rigging leaves.
    """
    lineup = PRESETS["HOCR 4+"]()
    sides = tuple(rower.side for rower in lineup.rowers)
    assert sides == (-1, +1, +1, -1), sides          # S P P S
    assert lineup.rowers[0].name == "Marilyn"        # stroke
    assert lineup.rowers[-1].name == "Sheila"        # bow
    assert lineup.balanced()
    # and the preset agrees with the boat builder's own table
    assert sides == rig_sides("4+", "bucket, stbd stroke")


def test_presets_hand_back_a_fresh_copy():
    """Editing a loaded lineup must not edit the preset.

    They are callables for exactly this reason -- a shared mutable
    default would mean the second load of a preset was whatever the
    first load had been changed into.
    """
    first = PRESETS["HOCR 4+"]()
    first.rowers[0].name = "someone else"
    first.switch_side(0)
    second = PRESETS["HOCR 4+"]()
    assert second.rowers[0].name == "Marilyn"
    assert second.rig == "bucket, stbd stroke"


def test_switching_a_seat_stops_the_rig_claiming_to_be_standard():
    """A hand-moved seat is no longer the named rig, and must say so."""
    lineup = PRESETS["HOCR 4+"]()
    assert lineup.rig == "bucket, stbd stroke"
    lineup.switch_side(3)
    assert lineup.rig == "custom"
    assert not lineup.balanced(), "three on one side is not a sweep boat"
    # putting it back does NOT silently reclaim the name; the sides do
    lineup.switch_side(3)
    assert lineup.balanced()
    assert tuple(r.side for r in lineup.rowers) == rig_sides(
        "4+", "bucket, stbd stroke")


def test_changing_shell_keeps_the_crew_it_can_seat():
    """An eight needs eight seats; a double keeps two of the four."""
    lineup = PRESETS["HOCR 4+"]()
    lineup.set_shell("8+")
    assert len(lineup.rowers) == 8
    assert lineup.rowers[0].name == "Marilyn", "the four should survive"
    assert lineup.rig in RIGS[8]
    lineup.set_shell("2x")
    assert len(lineup.rowers) == 2
    assert lineup.balanced(), "a double has no sides to balance"


def test_every_shell_offers_only_rigs_it_can_take():
    for shell, (_label, seats, _coxed, _bow) in SHELLS.items():
        for rig in RIGS[seats]:
            sides = rig_sides(shell, rig)
            assert len(sides) == seats, (shell, rig, sides)
            if seats > 2:
                assert sum(sides) == 0, (shell, rig, "sweep must balance")


def test_the_plan_puts_riggers_on_the_side_the_rower_rows():
    """The check a coxswain makes by eye, made by the test.

    Bow is up the plan, so port -- the left hand of a coxswain facing
    the bow -- draws on the left.  A starboard rower with a port rigger
    is the error this whole picture exists to make obvious.
    """
    lineup = PRESETS["HOCR 4+"]()
    plan = plan_geometry(lineup, 900.0, 560.0)
    centre = plan["hull"][0]
    for mark in plan["marks"]:
        tip_x = mark["rigger"][1][0]
        if mark["side"] > 0:                       # port
            assert tip_x < centre, mark["label"]
        else:                                      # starboard
            assert tip_x > centre, mark["label"]
        # the label box hangs on the same side as its rigger
        assert (mark["box_side"] < 0) == (mark["side"] > 0)


def test_the_plan_runs_stroke_at_the_stern_and_bow_at_the_top():
    lineup = PRESETS["HOCR 4+"]()
    plan = plan_geometry(lineup, 900.0, 560.0)
    ys = [mark["seat"][1] for mark in plan["marks"]]
    assert ys == sorted(ys, reverse=True), "stroke is nearest the stern"
    assert plan["bow"][1] < plan["stern"][1], "bow is up the screen"


def test_a_bow_loader_puts_the_coxswain_in_the_bow():
    """Where the coxswain is drawn has to match the hull.

    A four in this catalogue is a bow-loader and an eight is not, and
    the difference is the whole view from the seat.
    """
    four = plan_geometry(PRESETS["HOCR 4+"](), 900.0, 560.0)
    assert four["cox_bow_loaded"]
    assert four["cox"][1] < four["marks"][0]["seat"][1]

    eight = PRESETS["HOCR 4+"]()
    eight.set_shell("8+")
    plan = plan_geometry(eight, 900.0, 560.0)
    assert not plan["cox_bow_loaded"]
    assert plan["cox"][1] > plan["marks"][0]["seat"][1], "stern-coxed"


def test_erg_conversion_round_trips_and_matches_the_squad_sheet():
    """Concept2's own relation, checked against real stored values.

    These three are from the squad's spreadsheet, which stores its own
    watts beside the times -- so this is a check against their numbers,
    not against my arithmetic.
    """
    for stamp, stored in (("21:27.0", 164), ("19:25.1", 221),
                          ("19:46.2", 210)):
        assert erg_watts(stamp) == pytest.approx(stored, abs=0.5)

    # And back again.  Only to the precision the stamp carries: erg
    # times are quoted to a tenth of a second, and at these paces a
    # tenth is about 0.02% of the power -- so a round trip that agreed
    # to floating point would mean the stamp was keeping digits nobody
    # writes down.
    assert erg_stamp(erg_watts("23:07.0")) == "23:07.0"
    assert erg_watts(erg_stamp(131.2)) == pytest.approx(131.2, rel=1e-3)


def test_a_blank_erg_is_not_a_zero():
    """An unmeasured rower must not read as one pulling nothing."""
    rower = Rower("nobody", 130.0, 5, 6.0, "")
    assert rower.watts is None
    assert erg_watts("nonsense") is None
    lineup = Lineup(shell="4+", rowers=[rower])
    assert lineup.mean_watts() is None


# ---------------------------------------------------------------------------
# the drawing, headless
# ---------------------------------------------------------------------------
@pytest.fixture()
def fonts():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((900, 560))
    yield (pygame,
           pygame.font.SysFont("dejavusans,arial", 20),
           pygame.font.SysFont("dejavusans,arial", 14))
    pygame.quit()


def test_the_boxes_do_not_collide_and_sit_on_the_right_side(fonts):
    """Four boxes hung off a narrow hull is where a plan gets crowded."""
    pygame, font, small = fonts
    from coxswain.viz.rigview import draw_plan

    lineup = PRESETS["HOCR 4+"]()
    surface = pygame.Surface((900, 560), pygame.SRCALPHA)
    boxes = draw_plan(surface, lineup, font, small, (900, 560))

    assert set(boxes) == set(range(lineup.seats))
    pairs = [(a, b) for a in boxes for b in boxes if a < b]
    for a, b in pairs:
        assert not boxes[a].colliderect(boxes[b]), (a, b)

    centre = plan_geometry(lineup, 900 * 0.62, 560)["hull"][0]
    for index, rower in enumerate(lineup.rowers):
        if rower.side > 0:                      # port draws left
            assert boxes[index].centerx < centre, index
        else:
            assert boxes[index].centerx > centre, index


def test_the_plan_redraws_for_a_different_shell(fonts):
    """Switching hull must not leave four boxes on an eight."""
    pygame, font, small = fonts
    from coxswain.viz.rigview import draw_plan

    lineup = PRESETS["HOCR 4+"]()
    lineup.set_shell("8+")
    surface = pygame.Surface((900, 560), pygame.SRCALPHA)
    boxes = draw_plan(surface, lineup, font, small, (900, 560))
    assert len(boxes) == 8


def test_the_hull_is_tapered_and_closed():
    """A rectangle reads as a barge, and gives no cue which end is which."""
    from coxswain.viz.rigview import hull_half_beam, hull_outline

    assert hull_half_beam(0.5) == pytest.approx(1.0)
    assert hull_half_beam(0.0) == pytest.approx(0.0)
    assert hull_half_beam(1.0) == pytest.approx(0.0)
    points = hull_outline(100.0, 10.0, 12.0, 400.0)
    assert len(points) > 20
    xs = [p[0] for p in points]
    assert min(xs) >= 100.0 - 12.0 - 1e-9
    assert max(xs) <= 100.0 + 12.0 + 1e-9


def test_an_empty_seat_still_draws_a_box(fonts):
    """Adding seats must not crash on rowers who are not there yet."""
    pygame, font, small = fonts
    from coxswain.viz.rigview import rower_lines

    lineup = PRESETS["HOCR 4+"]()
    lineup.set_shell("8+")
    empty = lineup.rowers[-1]
    assert empty.name == ""
    assert rower_lines(empty) == ["(empty)"]


def test_the_editor_is_reachable_from_the_menu_and_edits_the_boat(fonts):
    """Drive the real setup menu and watch what it hands the renderer.

    Events are fed ONE PER FRAME.  Posting them all at once drains the
    whole queue in a single iteration, so the editor opens and closes
    before anything is drawn and the test passes while proving nothing --
    which is exactly what the first version of this did.
    """
    import argparse
    import os
    import sys

    pygame, font, small = fonts
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts"))
    import fpv
    from coxswain.viz.menu import setup_menu

    rows = [row.key for row in setup_menu().rows]
    assert "crew" in rows, "the menu must offer the rig editor"
    script = ([pygame.K_DOWN] * rows.index("crew") + [pygame.K_RETURN]
              + [pygame.K_DOWN, pygame.K_DOWN, pygame.K_RIGHT]
              + [pygame.K_DOWN, pygame.K_RIGHT, pygame.K_RETURN]
              + [None, pygame.K_ESCAPE, pygame.K_ESCAPE])

    step = {"i": 0}
    original_get = pygame.event.get

    def feed():
        index = step["i"]
        step["i"] += 1
        if index >= len(script):
            return [pygame.event.Event(pygame.QUIT)]
        key = script[index]
        return ([] if key is None
                else [pygame.event.Event(pygame.KEYDOWN, key=key)])

    seen = []
    original_draw = fpv.draw_plan

    def spy(surface, lineup, *args, **kwargs):
        seen.append((lineup.rig, tuple(r.side for r in lineup.rowers)))
        return original_draw(surface, lineup, *args, **kwargs)

    pygame.event.get = feed
    fpv.draw_plan = spy
    try:
        fpv.run_setup_menu(pygame.display.get_surface(),
                           argparse.Namespace(
                               boat="4+", race="charles", rate=30.0,
                               wind=5.0, audio="full", quality="standard",
                               weather="hazy", skill=0.55, balance=0.55,
                               no_sound=True))
    finally:
        pygame.event.get = original_get
        fpv.draw_plan = original_draw

    assert seen, "the editor never drew, so nothing here was exercised"
    # It opens on the preset...
    assert seen[0][0] == "bucket, stbd stroke"
    assert seen[0][1] == (-1, +1, +1, -1)
    # ...and the keys actually changed the boat.
    assert seen[-1][0] == "custom", seen[-1]
    assert seen[-1][1] != seen[0][1]


# ---------------------------------------------------------------------------
# typing a rower in
# ---------------------------------------------------------------------------
def test_typed_fields_parse_and_bad_ones_are_refused():
    """A weight of "abc" must not become 0 lb and quietly change the boat."""
    from coxswain.viz.rigview import FIELDS, parse_field

    index = {attr: i for i, (attr, _label) in enumerate(FIELDS)}
    assert parse_field(index["name"], "  Zed ") == "Zed"
    assert parse_field(index["pounds"], "135") == 135.0
    assert parse_field(index["pounds"], "abc") is None
    assert parse_field(index["pounds"], "-4") is None
    assert parse_field(index["feet"], "5") == 5
    assert parse_field(index["inches"], "3.5") == 3.5
    assert parse_field(index["inches"], "12") is None, "that is another foot"
    assert parse_field(index["erg_5k"], "22:10") == "22:10"
    assert parse_field(index["erg_5k"], "22:10.4") == "22:10.4"
    assert parse_field(index["erg_5k"], "22:70") is None
    assert parse_field(index["erg_5k"], "fast") is None
    assert parse_field(index["erg_5k"], "") == ""


def test_commit_edit_changes_the_rower_and_a_typed_height_is_measured():
    from coxswain.viz.rigview import FIELDS, field_text, rower_lines

    index = {attr: i for i, (attr, _label) in enumerate(FIELDS)}
    lineup = PRESETS["Squad W 60+ 4+"]()
    rower = lineup.rowers[0]
    assert rower.stature_estimated, "roster heights start as guesses"

    assert lineup.commit_edit(0, index["name"], "Zed")
    assert rower.name == "Zed"
    assert not lineup.commit_edit(0, index["pounds"], "heavy")
    assert lineup.commit_edit(0, index["pounds"], "140")
    assert rower.pounds == 140.0
    # Typing a height is a measurement, so the "~" must go.
    assert lineup.commit_edit(0, index["feet"], "5")
    assert lineup.commit_edit(0, index["inches"], "6")
    assert not rower.stature_estimated
    assert "~" not in rower_lines(rower)[1]
    # And the field text round-trips what was typed.
    assert field_text(rower, index["pounds"]) == "140"
    assert field_text(rower, index["inches"]) == "6"
    # Out-of-range seats are refused, not raised.
    assert not lineup.commit_edit(99, index["name"], "nobody")


def test_a_rower_can_be_typed_in_from_the_menu(fonts):
    """Drive the real setup menu: pick a seat, type a name, commit.

    One event per frame, as the other menu test explains -- posting them
    all at once drains the queue in one iteration and the editor never
    draws.  Typed characters arrive as KEYDOWN events carrying a
    ``unicode``; the feed sets it the way SDL would.
    """
    import argparse
    import os
    import sys

    pygame, font, small = fonts
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts"))
    import fpv
    from coxswain.viz.menu import setup_menu
    from coxswain.viz.rigview import PANE_ROWS

    rows = [row.key for row in setup_menu().rows]
    pane = [key for key, _label in PANE_ROWS]
    K = pygame
    script = ([K.K_DOWN] * rows.index("crew") + [K.K_RETURN]
              + [K.K_DOWN] * pane.index("edit")       # down to "Edit"
              + [K.K_RIGHT]                            # pick seat 0
              + [K.K_RETURN]                           # start typing
              + [K.K_BACKSPACE] * 12                   # clear the name
              + [("Z", K.K_z), ("e", K.K_e), ("d", K.K_d)]
              + [K.K_TAB]                              # commit, to weight
              + [K.K_BACKSPACE] * 6
              + [("1", K.K_1), ("4", K.K_4), ("0", K.K_0)]
              + [K.K_RETURN]                           # commit, stop
              + [None, K.K_ESCAPE, K.K_ESCAPE])

    step = {"i": 0}
    original_get = pygame.event.get

    def feed():
        index = step["i"]
        step["i"] += 1
        if index >= len(script):
            return [pygame.event.Event(pygame.QUIT)]
        item = script[index]
        if item is None:
            return []
        if isinstance(item, tuple):
            char, key = item
            return [pygame.event.Event(pygame.KEYDOWN, key=key,
                                       unicode=char)]
        return [pygame.event.Event(pygame.KEYDOWN, key=item, unicode="")]

    seen = []
    original_draw = fpv.draw_plan

    def spy(surface, lineup, *args, **kwargs):
        seen.append((lineup.rowers[0].name, lineup.rowers[0].pounds,
                     kwargs.get("editing")))
        return original_draw(surface, lineup, *args, **kwargs)

    pygame.event.get = feed
    fpv.draw_plan = spy
    try:
        fpv.run_setup_menu(pygame.display.get_surface(),
                           argparse.Namespace(
                               boat="4+", race="charles", rate=30.0,
                               wind=5.0, audio="full", quality="standard",
                               weather="hazy", skill=0.55, balance=0.55,
                               no_sound=True))
    finally:
        pygame.event.get = original_get
        fpv.draw_plan = original_draw

    assert seen, "the editor never drew"
    assert seen[0][0] == "Marilyn"
    # The box was drawn WHILE typing, with the buffer in it...
    typed = [e for _n, _p, e in seen if e is not None]
    assert typed, "draw_plan never saw an editing state"
    assert any(e[2] == "Zed" for e in typed), typed
    # ...and the lineup ended up changed.
    assert seen[-1][0] == "Zed", seen[-1]
    assert seen[-1][1] == 140.0, seen[-1]
    assert seen[-1][2] is None, "editing must end on Enter"


def test_genevieves_pink_ribbon_is_the_boat_in_the_picture():
    """P S P S from a port stroke, four six-footers, the ergs as typed."""
    lineup = PRESETS["Genevieve's Pink Ribbon"]()
    assert lineup.shell == "4+" and lineup.rig == "standard"
    assert [r.name for r in lineup.rowers] == ["Kevin", "Mark", "Scott", "Evan"]
    assert tuple(r.side for r in lineup.rowers) == (+1, -1, +1, -1)
    assert tuple(r.side for r in lineup.rowers) == rig_sides("4+", "standard")
    assert [r.erg_5k for r in lineup.rowers] == ["19:39", "18:55", "18:26", "19:30"]
    assert all(r.feet == 6 and r.inches == 0.0 for r in lineup.rowers)
    assert [r.pounds for r in lineup.rowers] == [182.0, 182.0, 220.0, 180.0]
    watts = [r.watts for r in lineup.rowers]
    assert abs(sum(watts) / 4 - 233.0) < 1.5
    assert abs(lineup.cox_pounds * 0.45359237 - 73.0) < 0.5
    assert lineup.balanced()
