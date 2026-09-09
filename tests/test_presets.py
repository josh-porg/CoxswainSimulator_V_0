r"""Saved boats, and the editor's boat reaching the race.

Two promises.  A lineup typed into the rig editor survives a round trip
through presets.json and comes back as the same boat.  And the boat
that races IS that lineup -- these rowers at these masses on this rig
with these power ratios -- which until this change it never was: the
editor drew a plan and the catalogue's default crew rowed the race.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from coxswain.viz.rigview import PRESETS, Lineup, Rower


def test_a_lineup_round_trips_through_plain_data():
    lineup = PRESETS["HOCR 4+"]()
    lineup.switch_side(0)                 # make it custom, to be sure
    lineup.rowers[1].name = "Someone Typed"
    data = lineup.to_dict()
    back = Lineup.from_dict(data)
    assert back.to_dict() == data
    assert back.rig == "custom"
    assert [r.side for r in back.rowers] == [r.side for r in lineup.rowers]
    assert back.rowers[1].name == "Someone Typed"
    assert back.cox_pounds == lineup.cox_pounds


def test_a_file_from_an_older_build_still_loads():
    """Missing fields take their defaults rather than raising."""
    back = Lineup.from_dict({"name": "old", "rowers": [{"name": "a"}]})
    assert back.name == "old" and back.shell == "4+"
    assert back.rowers[0].pounds == 0.0 and back.rowers[0].erg_5k == ""


def test_the_store_saves_replaces_lists_and_deletes(tmp_path):
    from coxswain.viz import presets

    path = str(tmp_path / "presets.json")
    assert presets.load(path) == []
    a = PRESETS["HOCR 4+"]().to_dict(); a["name"] = "Tuesday crew"
    assert presets.save(a, path)
    assert presets.names(path) == ["Tuesday crew"]
    a2 = dict(a); a2["rig"] = "custom"
    assert presets.save(a2, path)                 # same name: replaced
    assert presets.names(path) == ["Tuesday crew"]
    assert presets.get("Tuesday crew", path)["rig"] == "custom"
    b = dict(a); b["name"] = "Race day"
    assert presets.save(b, path)
    assert sorted(presets.names(path)) == ["Race day", "Tuesday crew"]
    assert presets.delete("Tuesday crew", path)
    assert presets.names(path) == ["Race day"]
    # a blank name is refused, and a broken file is an empty list
    assert not presets.save({"name": "  "}, path)
    (tmp_path / "presets.json").write_text("{not json", encoding="utf-8")
    assert presets.load(path) == []


def test_the_editor_offers_a_save_row_that_names_the_lineup():
    from coxswain.viz.rigview import PANE_ROWS, pane_values

    keys = [k for k, _label in PANE_ROWS]
    assert "save" in keys and keys.index("save") < keys.index("done")
    assert len(pane_values(PRESETS["HOCR 4+"]())) == len(PANE_ROWS)


def test_the_lineup_builds_its_own_boat():
    """Masses, rig sides, coxswain, and power ratios from the ergs."""
    from coxswain.viz.menu import build_boat

    lineup = PRESETS["HOCR 4+"]()
    boat, made = build_boat("4+", 30.0, lineup=lineup)
    assert made == "4+"
    LB = 0.45359237
    masses = [m.rower.anthropometry.mass for m in boat.crew]
    assert masses == pytest.approx([r.pounds * LB for r in lineup.rowers],
                                   rel=1e-6)
    sides = [int(l.side) for s in boat.rig.seats for l in s.oarlocks]
    assert sides == [r.side for r in lineup.rowers]        # S P P S
    assert boat.rig.coxswain_mass == pytest.approx(lineup.cox_pounds * LB)
    ratios = np.asarray(boat.seat_ratios)
    assert ratios.mean() == pytest.approx(1.0)
    watts = np.array([r.watts for r in lineup.rowers])
    assert ratios == pytest.approx(watts / watts.mean())
    # the strongest erg is the strongest seat
    assert int(np.argmax(ratios)) == int(np.argmax(watts))


def test_a_custom_rig_builds_with_its_own_sides():
    """Balanced but not a named pattern: the tuple goes straight through."""
    from coxswain.viz.menu import build_boat

    lineup = PRESETS["HOCR 4+"]()
    lineup.switch_side(0)
    lineup.switch_side(1)                        # P S P S from S P P S
    assert lineup.rig == "custom" and lineup.balanced()
    boat, _made = build_boat("4+", 30.0, lineup=lineup)
    sides = [int(l.side) for s in boat.rig.seats for l in s.oarlocks]
    assert sides == [r.side for r in lineup.rowers]


def test_an_unbalanced_lineup_falls_back_to_the_catalogue_boat(capsys):
    """The rig code refuses three on one side, by design.  The trainer
    must still start -- with the catalogue crew, and saying so."""
    from coxswain.viz.menu import build_boat

    lineup = PRESETS["HOCR 4+"]()
    lineup.switch_side(3)                        # three on one side
    assert not lineup.balanced()
    boat, made = build_boat("4+", 30.0, lineup=lineup)
    assert made == "4+" and boat.n_seats == 4
    assert "could not build the lineup" in capsys.readouterr().out


def test_a_lineup_without_ergs_rows_evenly():
    from coxswain.viz.menu import build_boat

    lineup = Lineup(shell="8+", rowers=[Rower("r%d" % i, 150.0, 5, 10.0, "")
                                        for i in range(8)]).apply_rig()
    boat, _made = build_boat("8+", 32.0, lineup=lineup)
    assert boat.n_seats == 8
    assert np.allclose(boat.seat_ratios, 1.0)


def test_the_call_scales_the_crew_and_keeps_each_seats_ratio():
    """The coxswain's call moves the crew mean; the ergs set the spread."""
    text = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts", "fpv.py"), encoding="utf-8").read()
    assert 'getattr(boat, "seat_ratios", np.ones(boat.n_seats))' in text
    assert "np.full(boat.n_seats, scale)" not in text
    assert text.count('getattr(boat, "seat_ratios", np.ones(boat.n_seats))') == 2, "both power paths"
    # and the setup menu hands the lineup on, so the race gets it
    assert 'picked["lineup"] = lineup_last' in text
    assert "lineup=getattr(args, \"lineup\", None)" in text
