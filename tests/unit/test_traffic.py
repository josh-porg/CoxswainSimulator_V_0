"""A boat ahead, and the lateral structure of what it leaves behind."""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.hydro.wake import PuddleWake, blade_track
from coxswain.river.traffic import LeadBoat, TrafficWake


@pytest.fixture(scope="module")
def boat():
    return catalog.eight(rate=32.0)


@pytest.fixture(scope="module")
def wake(boat):
    track = blade_track(boat)
    puddle = PuddleWake(drag=258.0, speed=4.23,
                        period=boat.timing.period, n_blades=8)
    return TrafficWake(puddle=puddle, track=track, follower_track=track)


def test_directly_astern_is_a_penalty(wake):
    assert wake.drag_factor(0.0, 63.0) > 1.0


def test_well_off_the_line_is_clean(wake):
    assert wake.drag_factor(20.0, 63.0) == pytest.approx(1.0, abs=1e-4)


def test_the_wake_has_TWO_lateral_peaks_not_one(wake):
    """The structure that makes this a line-choice problem at all.

    Their puddles sit at +/- the blade track and yours sweep the same
    distance either side of you, so coincidences happen at zero offset AND
    at twice the blade track.  A single-peaked wake would say "just move
    over"; a double-peaked one says move over *the right amount*.
    """
    offsets = np.linspace(0.0, 12.0, 241)
    factor = np.array([wake.drag_factor(o, 63.0) for o in offsets])
    interior = factor[1:-1]
    peaks = [i + 1 for i in range(len(interior))
             if factor[i + 1] >= factor[i] and factor[i + 1] >= factor[i + 2]]
    # the peak at zero is at the boundary, so count it separately
    assert factor[0] > factor[10]
    far = [offsets[i] for i in peaks if offsets[i] > 3.0]
    assert far, "expected a second peak away from the centreline"
    assert far[0] == pytest.approx(2.0 * wake.track, abs=1.0)


def test_the_clean_window_sits_between_the_peaks(wake):
    """And it is genuinely clean, not merely less bad."""
    astern = wake.drag_factor(0.0, 63.0)
    window = wake.drag_factor(wake.track, 63.0)
    second = wake.drag_factor(2.0 * wake.track, 63.0)
    assert window < astern and window < second
    assert (window - 1.0) < 0.15 * (astern - 1.0)


def test_the_wake_weakens_with_a_longer_interval(wake):
    close = wake.drag_factor(0.0, 10.0 * 4.23)
    far = wake.drag_factor(0.0, 60.0 * 4.23)
    assert 1.0 < far < close


def test_separation_is_measured_from_the_leaders_track(boat):
    """Not from the centreline.

    The leader's own line is an offset too, so the wake must move with it.
    Measuring from the centreline instead would paint the wake down the
    middle of the river no matter where the boat ahead actually rowed --
    which is exactly the mistake that makes traffic look like it never
    matters.
    """
    from coxswain.river.charles import charles_course
    from coxswain.river.route import Route

    course = charles_course()
    stations = np.linspace(0.0, course.length, 5)
    shifted = Route(stations, np.full(5, 12.0), name="lead")
    lead = LeadBoat.build(shifted, course, boat, drag=258.0,
                          interval=15.0, speed=4.23)
    probe = np.array([course.length * 0.5])
    # On the leader's own line: penalised.  On the centreline: clean.
    on_them = lead.drag_factor_along(probe, np.array([12.0]))
    on_centre = lead.drag_factor_along(probe, np.array([0.0]))
    assert float(on_them[0]) > 1.0
    assert float(on_centre[0]) == pytest.approx(1.0, abs=1e-4)


def test_the_gap_follows_from_the_interval(boat):
    from coxswain.river.charles import charles_course
    from coxswain.river.route import Route

    course = charles_course()
    lead = LeadBoat.build(Route.centreline(course), course, boat,
                          drag=258.0, interval=15.0, speed=4.23)
    assert lead.gap == pytest.approx(15.0 * 4.23)


def test_the_evaluator_charges_for_traffic(boat):
    """End to end: the same line is slower with a boat ahead on it."""
    from coxswain.river.charles import charles_course
    from coxswain.river.route import Route, RouteEvaluator

    course = charles_course()
    line = Route.centreline(course)
    lead = LeadBoat.build(line, course, boat, drag=258.0, interval=15.0,
                          speed=4.23)
    clean = RouteEvaluator(course, boat=boat).evaluate(line)
    dirty = RouteEvaluator(course, boat=boat).with_traffic(lead).evaluate(line)
    assert dirty.elapsed_clean > clean.elapsed_clean


def test_traffic_is_off_by_default(boat):
    from coxswain.river.charles import charles_course
    from coxswain.river.route import RouteEvaluator

    assert RouteEvaluator(charles_course(), boat=boat).traffic is None
