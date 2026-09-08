r"""The anonymous squad roster, and the boats built from it.

Two separate things are being held here.  One is ordinary: does the
loader read the file and pick sensible crews.  The other is not -- the
roster exists *because* the source spreadsheet belongs to real people
who did not agree to be in a game, so "there are no names in this file"
is a promise the tests have to keep, not a comment.
"""

from __future__ import annotations

import csv
import io
import os
import re

import pytest

from coxswain.crew.roster import (ASSUMED_STATURE, ROSTER_PATH, crew_of,
                                  feet_inches, load_roster, select)

pytestmark = pytest.mark.skipif(not os.path.exists(ROSTER_PATH),
                                reason="roster not built on this checkout")


def test_the_roster_carries_no_identities():
    """The whole point of the anonymiser, checked on the shipped file.

    Ids only, and only the columns the anonymiser writes.  If a name
    column ever reappears -- a rebuild against a changed sheet, someone
    editing the CSV by hand -- this is what catches it before it ships.
    """
    with io.open(ROSTER_PATH, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert set(reader.fieldnames) == {
            "id", "squad", "age", "pounds", "erg_5k", "erg_5k_best", "pieces"}
        rows = list(reader)

    assert rows, "an empty roster would pass every other test here"
    for row in rows:
        assert re.fullmatch(r"[WM]\d{2,}", row["id"]), row["id"]
        assert row["squad"] in ("women", "men")
        # No cell anywhere may hold anything that reads as a name.
        for key, value in row.items():
            if key in ("id", "squad", "erg_5k", "erg_5k_best"):
                continue
            assert not re.search(r"[A-Za-z]", value), (key, value)


def test_the_blurring_actually_happened():
    """Weights to five pounds, times to the second.

    Not cosmetic: an exact weight beside a time to the tenth is close
    to a unique key for anyone holding the original sheet.
    """
    for entry in load_roster():
        if entry.pounds:
            assert entry.pounds % 5 == 0, entry
        assert re.fullmatch(r"\d{1,2}:\d{2}", entry.erg_5k), entry
        assert re.fullmatch(r"\d{1,2}:\d{2}", entry.erg_5k_best), entry


def test_a_best_is_never_slower_than_the_median():
    for entry in load_roster():
        best = [int(p) for p in entry.erg_5k_best.split(":")]
        median = [int(p) for p in entry.erg_5k.split(":")]
        assert best[0] * 60 + best[1] <= median[0] * 60 + median[1], entry


def test_selection_respects_the_age_band_and_orders_by_speed():
    """A 60+ event is 60 and over -- the bound is inclusive."""
    band = select("women", min_age=60)
    assert band, "the squad has 60+ women; if not, the sheet changed"
    assert all(entry.age >= 60 for entry in band)
    assert all(entry.squad == "women" for entry in band)
    times = [entry.seconds for entry in band]
    assert times == sorted(times), "fastest first"


def test_offset_walks_down_the_order_without_running_off_the_end():
    pool = select("women")
    first = crew_of(4, "women")
    assert [e.id for e in first] == [e.id for e in pool[:4]]
    later = crew_of(4, "women", offset=3)
    assert [e.id for e in later] == [e.id for e in pool[3:7]]
    # Asking past the end gives the last full crew, not a short one.
    clamped = crew_of(4, "women", offset=10_000)
    assert len(clamped) == 4
    assert [e.id for e in clamped] == [e.id for e in pool[-4:]]


def test_a_band_too_thin_for_the_boat_gives_what_it_has():
    """Better a short crew than a crew padded with invented rowers."""
    thin = crew_of(8, "women", min_age=60)
    assert len(thin) == len(select("women", min_age=60))


def test_watts_follow_the_concept2_relation():
    """Faster is stronger, and by the cube of the pace."""
    entries = select("women")
    assert entries[0].watts > entries[-1].watts
    fast, slow = entries[0], entries[-1]
    ratio = (slow.seconds / fast.seconds) ** 3
    assert fast.watts / slow.watts == pytest.approx(ratio, rel=1e-9)


def test_height_is_flagged_as_a_guess_everywhere_it_appears():
    """The sheet has no statures.  Nothing may pretend otherwise."""
    from coxswain.viz.rigview import PRESETS, rower_lines

    lineup = PRESETS["Squad W 60+ 4+"]()
    assert lineup.rowers, "the preset seated nobody"
    for rower in lineup.rowers:
        if not rower.name:
            continue
        assert rower.stature_estimated, rower.name
        assert "~" in rower_lines(rower)[1], rower_lines(rower)

    # ...and the crew this project actually owns is measured, so it
    # must NOT be marked.
    for rower in PRESETS["HOCR 4+"]().rowers:
        assert not rower.stature_estimated
        assert "~" not in rower_lines(rower)[1]


def test_every_roster_preset_fills_its_boat_and_keeps_its_rig():
    from coxswain.viz.rigview import PRESETS, SHELLS, rig_sides

    for name, builder in PRESETS.items():
        if not name.startswith("Squad"):
            continue
        lineup = builder()
        assert len(lineup.rowers) == SHELLS[lineup.shell][1], name
        assert tuple(r.side for r in lineup.rowers) == rig_sides(
            lineup.shell, lineup.rig), name
        assert lineup.balanced(), name
        assert lineup.name == name


def test_feet_inches_round_trips_through_the_assumed_stature():
    for squad, stature in ASSUMED_STATURE.items():
        feet, inches = feet_inches(stature)
        back = (feet * 12 + inches) * 0.0254
        assert back == pytest.approx(stature, abs=0.003), squad
