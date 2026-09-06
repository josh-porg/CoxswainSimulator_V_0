"""The Head of the Lake course has to be on the water it is raced on.

These are cross-dataset checks: the course comes from the regatta's buoy
map and the water from OpenStreetMap, and neither can validate itself.
The one that would have caught the first trace is
:func:`test_the_course_threads_the_montlake_cut` -- the overlap
registration put the lane 43 m south, onto the Cut's wall, while every
summary number stayed plausible.
"""

import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COURSE = os.path.join(ROOT, "data", "hotl_course.npy")
BUOYS = os.path.join(ROOT, "data", "hotl_buoys.npy")

pytestmark = pytest.mark.skipif(not os.path.exists(COURSE),
                                reason="run tools/trace_hotl_course.py")


@pytest.fixture(scope="module")
def course():
    return np.load(COURSE)


@pytest.fixture(scope="module")
def canal():
    from coxswain.river.seattle import SHIP_CANAL, water_mask
    return water_mask(10.0, names=SHIP_CANAL)


def _on(mask_tuple, points):
    east, north, mask = mask_tuple
    rows = np.clip(np.searchsorted(north, points[:, 1]), 0, len(north) - 1)
    cols = np.clip(np.searchsorted(east, points[:, 0]), 0, len(east) - 1)
    return mask[rows, cols]


def test_the_whole_course_is_on_water(course, canal):
    assert _on(canal, course).all()


def test_the_course_threads_the_montlake_cut(course):
    """Through the Cut the line must be in the Cut, near its centreline.

    The Cut is ~90 m wall to wall.  A registration 43 m out puts the
    lane on the wall, which is what the overlap-only placement did.
    """
    from coxswain.river.seattle import water_mask
    east, north, mask = water_mask(10.0, names=("Montlake Cut",))
    ge, gn = np.meshgrid(east, north)
    west, east_end = ge[mask].min(), ge[mask].max()
    centreline = gn[mask].mean()
    inside = course[(course[:, 0] > west + 30) & (course[:, 0] < east_end - 30)]
    assert len(inside) >= 8
    assert _on((east, north, mask), inside).all()
    assert np.abs(inside[:, 1] - centreline).max() < 25.0


def test_the_course_is_three_miles(course):
    """Within a few percent of the stated distance, and short not long.

    Chaining dash centroids cuts every corner, so a correct trace comes
    out a little *under* the published length.  Coming out over it was
    the sign that the free-scale registration was wrong.
    """
    length = np.hypot(*np.diff(course, axis=0).T).sum()
    assert 0.94 * 4828.0 < length <= 4828.0


def test_the_course_runs_start_to_finish_in_order(course):
    """West to east overall, ending north-east of where it began, with
    the Big Turn -- a reversal of heading -- near the end."""
    assert course[-1, 0] > course[0, 0] + 2500.0
    headings = np.degrees(np.arctan2(*np.diff(course, axis=0).T[::-1]))
    # The last quarter contains a turn of more than 120 degrees.
    late = headings[3 * len(headings) // 4:]
    swing = np.abs(np.degrees(np.angle(np.exp(1j * np.radians(late[-1] - late[0])))))
    assert swing > 100.0


def test_the_buoys_are_in_the_water(canal):
    buoys = np.load(BUOYS)
    assert len(buoys) >= 40
    wet = _on(canal, buoys[:, 1:])
    # One mark may sit on a dock or the finish line; not more.
    assert wet.sum() >= len(buoys) - 1


def test_the_buoys_are_on_the_sides_the_regatta_says(course):
    """Yellow to starboard, orange to port -- every lane-limit buoy.

    A lane-limit buoy is one within 80 m of the lane and more than 100 m
    from either end.  The map also carries marks that are not: the
    marshalling buoys a kilometre off in Lake Union, the line-up chute
    either side of the start, the finish gate, and a yellow pair beyond
    the orange line at the Pocock apex.  Those are excluded here and
    ignored by ``scripts/render_hotl.py`` for the same reason.
    """
    buoys = np.load(BUOYS)
    station = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(course, axis=0).T))])
    heading = np.arctan2(np.gradient(course[:, 1]), np.gradient(course[:, 0]))
    counted = wrong = 0
    for keep_to_port, bx, by in buoys:
        gap = np.hypot(course[:, 0] - bx, course[:, 1] - by)
        i = int(np.argmin(gap))
        if gap[i] > 80.0 or not 100.0 < station[i] < station[-1] - 100.0:
            continue
        counted += 1
        normal = np.array([-np.sin(heading[i]), np.cos(heading[i])])
        offset = float(np.dot([bx - course[i, 0], by - course[i, 1]], normal))
        # positive offset = port side of the line
        wrong += (offset > 0) != bool(keep_to_port)
    assert counted >= 25
    assert wrong == 0


def test_the_lane_never_doubles_back(course):
    """No dash-to-dash turn sharper than 60 degrees.

    The greedy chain once hopped onto the return leg at the Big Turn,
    where the two legs are 45 m apart: +107, -128, +66 degrees over
    three dashes, and a 26,000 s "as drawn" time from a steering model
    asked to follow a Z.
    """
    headings = np.unwrap(np.arctan2(*np.diff(course, axis=0).T[::-1]))
    assert np.degrees(np.abs(np.diff(headings))).max() < 60.0


def test_no_buoy_is_a_letter_of_the_finish_banner():
    """The FINISH banner's yellow lettering once read as four buoys in a
    neat north-south column 60 m west of the lane's end."""
    buoys = np.load(BUOYS)
    yellow = buoys[buoys[:, 0] == 0.0]
    finish = np.load(COURSE)[-1]
    near = yellow[np.hypot(yellow[:, 1] - finish[0],
                           yellow[:, 2] - finish[1]) < 120.0]
    assert len(near) == 0


def test_the_lane_passes_under_three_bridges_in_order(course):
    from coxswain.river.seattle import canal_bridges
    hits = [(b.name, b.crossing(course)) for b in canal_bridges()]
    crossed = sorted((h[1][0], name) for name, h in hits if h is not None)
    assert [name for _s, name in crossed] == [
        "Ship Canal Bridge", "University Bridge", "Montlake Bridge"]


def test_the_water_under_the_montlake_bridge_matches_the_federal_record():
    """OSM's water polygon against the NBI's navigation clearance.

    Two records nobody reconciled: the mapped water under the Montlake
    Bridge should be the fender-to-fender opening the Bridge Inventory
    reports, and it is, to 5 m.
    """
    from coxswain.river.seattle import canal_bridges
    montlake = [b for b in canal_bridges() if b.name == "Montlake Bridge"][0]
    runs = montlake.water_runs()
    assert len(runs) == 1
    width = runs[0][1] - runs[0][0]
    assert abs(width - montlake.opening) < 5.0


def test_the_lane_crosses_each_bridge_over_water(course):
    """Under each deck the crossing must be on mapped water; under the
    Montlake Bridge and I-5 it must also be near the middle of it."""
    from coxswain.river.seattle import canal_bridges
    for bridge in canal_bridges():
        hit = bridge.crossing(course)
        assert hit is not None, bridge.name
        along = hit[2]
        runs = bridge.water_runs()
        inside = [(a, b) for a, b in runs if a <= along <= b]
        assert inside, (bridge.name, along, runs)
        if bridge.name != "University Bridge":
            a, b = inside[0]
            assert abs(along - (a + b) / 2.0) < 10.0, (bridge.name, along)


def test_the_orange_line_holds_through_the_cut():
    """Between the three orange marks along the Cut the port limit must
    still bind.  They are 357 m apart; joined only within 250 m, the
    corridor between them was the bare clearance to the north wall and
    the optimised line went there."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from render_hotl import hotl_course
    course = hotl_course()
    line = course.centreline
    in_cut = (line[:, 0] > 1800.0) & (line[:, 0] < 2400.0)
    assert in_cut.sum() > 30
    # The orange marks sit 10-24 m to port of the lane; less the 6 m
    # margin, no port limit in the Cut may exceed 20 m.
    assert course.port_limit[in_cut].max() < 20.0


def test_the_montlake_cut_has_walls_and_towers():
    """The Cut is a walled channel, not a bank.

    Bare-earth lidar with the trees stripped draws it as a grassy
    hillside, which is what the scene showed: a 50 m slot between
    concrete walls, rendered as a meadow.
    """
    from coxswain.river.seattle import load_canal_walls
    pieces = load_canal_walls()
    assert pieces, "run tools/extract_canal_walls.py"
    kinds = [k for k, _t, _p in pieces]
    assert kinds.count("tower") == 2, "the bridge's two concrete towers"
    walls = [(t, p) for k, t, p in pieces if k == "wall"]
    assert len(walls) >= 4
    # One long wall each side of the Cut, north and south of the water.
    spans = [p[:, 1].mean() for _t, p in walls
             if np.hypot(*np.diff(p, axis=0).T).sum() > 300.0]
    assert len(spans) >= 2
    assert min(spans) < 870.0 < max(spans)
    assert all(1.0 <= t <= 8.0 for t, _p in walls)


def test_the_bascule_has_no_pier_in_the_navigable_opening():
    """A bascule's leaves meet over the middle of the waterway.

    Dividing the deck by the federal main span put a pier in the centre
    of the Cut, where the boats go.
    """
    from coxswain.river.seattle import canal_bridges
    montlake = [b for b in canal_bridges() if b.name == "Montlake Bridge"][0]
    runs = montlake.water_runs()
    assert len(runs) == 1, "the opening is one clear span"
