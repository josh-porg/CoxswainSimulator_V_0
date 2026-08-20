"""Tests for the surveyed Charles bathymetry and the continuity flow model.

Data provenance is in ``coxswain/river/charles.py``; these tests check that
what was loaded is self-consistent and that the flow model reproduces the
qualitative behaviour the source reports.
"""

import numpy as np
import pytest

from coxswain.river import charles
from coxswain.river.course import Course, DepthField

pytestmark = pytest.mark.skipif(
    not __import__("os").path.exists(charles.isobath_path()),
    reason="Charles bathymetry CSV not present",
)


@pytest.fixture(scope="module")
def course():
    return charles.charles_course()


# --------------------------------------------------------------------------
# bathymetry
# --------------------------------------------------------------------------
def test_isobaths_cover_the_lower_charles():
    points, depths = charles.load_isobaths()
    assert len(points) > 5000
    # the surveyed reach is ~10 km east-west
    assert np.ptp(points[:, 0]) > 8000.0


def test_isobath_depths_span_the_surveyed_range():
    """CRAB's chart runs from the 1 ft contour to about 34 ft."""
    _, depths = charles.load_isobaths()
    assert depths.min() == pytest.approx(0.30, abs=0.02)
    assert 10.0 < depths.max() < 11.0


def test_median_depth_is_plausible_for_the_basin():
    _, depths = charles.load_isobaths()
    assert 2.0 < np.median(depths) < 6.0


def test_depth_field_is_marked_as_survey_data():
    field = charles.charles_depth_field()
    assert field.is_survey


def test_depth_field_interpolates_within_the_surveyed_area():
    field = charles.charles_depth_field()
    points, _ = charles.load_isobaths()
    centre = points.mean(axis=0)
    value = field(centre[0], centre[1])
    assert 0.4 < value < 11.0


# --------------------------------------------------------------------------
# discharge record
# --------------------------------------------------------------------------
def test_discharge_table_has_twelve_months():
    assert charles.load_discharge().shape == (12, 6)


def test_discharge_percentiles_are_ordered():
    """p10 <= median <= p90, and min <= everything <= max, every month."""
    table = charles.load_discharge()
    for month in range(12):
        mean, p10, median, p90, low, high = table[month]
        assert low <= p10 <= median <= p90 <= high
        assert low <= mean <= high


def test_spring_carries_far_more_water_than_late_summer():
    """The seasonal signal that makes a single 'current' figure useless."""
    assert charles.monthly_discharge(3) > 5 * charles.monthly_discharge(8)


def test_october_discharge_matches_the_gauge_record():
    """Head of the Charles month, USGS 01104500, 1931-2026."""
    assert charles.monthly_discharge(10) == pytest.approx(2.84, abs=0.1)
    assert charles.monthly_discharge(10, "p90") == pytest.approx(11.4, abs=0.5)


def test_monthly_discharge_rejects_bad_input():
    with pytest.raises(ValueError, match="month must be"):
        charles.monthly_discharge(13)
    with pytest.raises(ValueError, match="statistic must be"):
        charles.monthly_discharge(10, "typical")


# --------------------------------------------------------------------------
# course
# --------------------------------------------------------------------------
def test_course_is_survey_backed_and_passes_the_guard(course):
    assert course.is_survey
    course.require_survey()


def test_course_length_matches_the_surveyed_reach(course):
    """CRAB charted ~14.5 km; the thalweg through it is a little shorter."""
    assert 8000.0 < course.length < 16000.0


def test_thalweg_is_deeper_than_the_geometric_middle():
    """The point of tracing the deep channel rather than the centre.

    The Charles has shoals well inside its banks, so the middle of the
    water is not the navigable line.
    """
    depth = charles.charles_depth_field()
    spine = charles.thalweg(depth=depth)
    points, _ = charles.load_isobaths()

    east = points[:, 0]
    deep, middle = [], []
    for x, y in spine[::4]:
        band = points[np.abs(east - x) < 120.0]
        if len(band) < 30:
            continue
        deep.append(float(depth(x, y)))
        middle.append(float(depth(x, np.median(band[:, 1]))))
    assert np.mean(deep) > np.mean(middle)


# --------------------------------------------------------------------------
# continuity flow model
# --------------------------------------------------------------------------
def test_cross_section_area_is_positive_everywhere(course):
    flow = charles.ContinuityFlow(course)
    _, area, _ = flow.profile(20)
    assert np.all(area > 0.0)


def test_flow_speed_is_discharge_over_area(course):
    flow = charles.ContinuityFlow(course, discharge=10.0)
    station = np.array([course.length * 0.4])
    area = flow.cross_section_area(station)
    np.testing.assert_allclose(flow.speed(station), 10.0 / area, rtol=1e-9)


def test_flow_speed_scales_linearly_with_discharge(course):
    station = np.array([course.length * 0.5])
    slow = charles.ContinuityFlow(course, discharge=3.0).speed(station)
    fast = charles.ContinuityFlow(course, discharge=30.0).speed(station)
    np.testing.assert_allclose(fast, 10.0 * slow, rtol=1e-9)


def test_basin_is_nearly_slack_at_typical_october_discharge(course):
    """The headline result: at race conditions the Charles barely flows.

    October's median 2.8 m3/s spread over a basin cross-section of
    hundreds of square metres is centimetres per second -- negligible
    against a 5 m/s shell.
    """
    flow = charles.ContinuityFlow(course,
                                  discharge=charles.monthly_discharge(10))
    _, _, speed = flow.profile(30)
    # The median is the headline: over most of the reach the water barely
    # moves.  The maximum is checked separately and is deliberately looser,
    # because the channel genuinely pinches to ~6 m of navigable half-width
    # at the tightest bridge and continuity says the water must speed up
    # there.  A single-threshold test on the maximum was only passing
    # before because the channel was a flat 55 m ribbon that had no
    # narrows in it.
    assert np.median(speed) < 0.05
    assert speed.max() < 0.30


def test_october_flow_concentrates_at_the_narrows(course):
    """Slack over the reach, faster where the channel pinches.

    Continuity with a real cross-section says the two go together; a
    constant-width channel cannot show it at all.
    """
    flow = charles.ContinuityFlow(course,
                                  discharge=charles.monthly_discharge(10))
    station, _, speed = flow.profile(60)
    half_width = np.array([course.half_width_at(s) for s in station])
    # rank rather than threshold: many stations sit exactly at the width
    # cap, so a quartile threshold can select an empty set
    order = np.argsort(half_width)
    narrowest, widest = order[:10], order[-10:]
    assert speed[narrowest].mean() > speed[widest].mean()


def test_flood_flow_becomes_significant(course):
    """At the October maximum of record the narrows do matter.

    0.6 m/s against a 5.5 m/s shell is an 11% effect, and because it
    concentrates where the channel is constricted it is route-dependent --
    which is the whole reason to carry a flow field at all.
    """
    flow = charles.ContinuityFlow(
        course, discharge=charles.monthly_discharge(10, "max"))
    _, _, speed = flow.profile(30)
    assert speed.max() > 0.3


def test_flow_is_fastest_where_the_section_is_smallest(course):
    flow = charles.ContinuityFlow(course, discharge=5.0)
    station, area, speed = flow.profile(30)
    assert np.argmin(area) == np.argmax(speed)


def test_current_field_points_downstream(course):
    """Downstream is towards the start of the course, which is laid out
    the way a crew rows it -- bow-first up the river."""
    flow = charles.ContinuityFlow(course, discharge=20.0)
    field = flow.as_current_field()
    station = course.length * 0.5
    point = course.position_at(np.array(station))
    velocity = field(point[0], point[1])
    heading = float(course.heading_at(np.array(station)))
    along = np.array([np.cos(heading), np.sin(heading)])
    assert np.dot(velocity, along) < 0.0


def test_current_field_magnitude_matches_the_profile(course):
    flow = charles.ContinuityFlow(course, discharge=20.0)
    field = flow.as_current_field()
    station = course.length * 0.5
    point = course.position_at(np.array(station))
    speed = float(np.hypot(*field(point[0], point[1])))
    assert speed == pytest.approx(float(flow.speed(np.array(station))[0]),
                                  rel=0.25)


def test_course_carries_a_current_field(course):
    assert not course.current.is_still


# --------------------------------------------------------------------------
# lateral variation -- the part that creates a line choice
# --------------------------------------------------------------------------
def test_lateral_profile_conserves_discharge(course):
    """The distribution refines Q/A, it must not contradict it.

    By construction integral(u h dy) == Q at every section.
    """
    flow = charles.ContinuityFlow(course, discharge=12.0)
    for fraction in (0.2, 0.4, 0.6, 0.8):
        offsets, depth, speed = flow.lateral_profile(fraction * course.length)
        assert np.trapezoid(speed * depth, offsets) == pytest.approx(
            12.0, rel=1e-9)


def test_water_runs_faster_in_the_deep_channel(course):
    """Fast water in the channel, slack over the shoals.

    This is the whole reason a crew has a line to choose: upstream you
    want the slack near the bank, downstream you want the thread.
    """
    flow = charles.ContinuityFlow(course, discharge=20.0)
    offsets, depth, speed = flow.lateral_profile(0.45 * course.length)
    assert np.argmax(speed) == np.argmax(depth)
    assert np.argmin(speed) == np.argmin(depth)


def test_lateral_speed_follows_the_depth_exponent(course):
    """u proportional to h^(2/3): Manning's slope and roughness cancel."""
    flow = charles.ContinuityFlow(course, discharge=20.0)
    _, depth, speed = flow.lateral_profile(0.45 * course.length)
    ratio = speed / depth ** flow.velocity_exponent
    assert np.std(ratio) / np.mean(ratio) < 1e-9


def test_lateral_spread_is_large_enough_to_matter(course):
    """A factor of ~3 across the section at a shoaled cross-section."""
    flow = charles.ContinuityFlow(course, discharge=20.0)
    _, _, speed = flow.lateral_profile(0.45 * course.length)
    assert speed.max() / speed.min() > 1.5


def test_section_mean_lies_between_bank_and_channel_speeds(course):
    flow = charles.ContinuityFlow(course, discharge=20.0)
    station = 0.45 * course.length
    _, _, speed = flow.lateral_profile(station)
    mean = float(flow.speed(np.array(station))[0])
    assert speed.min() <= mean <= speed.max()


def test_current_field_varies_across_the_channel(course):
    """Regression guard: the field used to be laterally uniform.

    Along-river variation alone is the wrong half -- it tells a route
    optimiser the river has no line in it.
    """
    flow = charles.ContinuityFlow(course, discharge=20.0)
    field = flow.as_current_field()
    station = 0.45 * course.length
    half = float(course.half_width_at(station))

    speeds = []
    for fraction in (-0.9, -0.3, 0.3, 0.9):
        point = course.offset_position(np.array(station),
                                       np.array(fraction * half))
        speeds.append(float(np.hypot(*field(point[0], point[1]))))
    assert max(speeds) / min(speeds) > 1.3


def test_uniform_mode_really_is_uniform_across_the_channel(course):
    flow = charles.ContinuityFlow(course, discharge=20.0)
    field = flow.as_current_field(lateral=False)
    station = 0.45 * course.length
    half = float(course.half_width_at(station))

    speeds = []
    for fraction in (-0.9, 0.0, 0.9):
        point = course.offset_position(np.array(station),
                                       np.array(fraction * half))
        speeds.append(float(np.hypot(*field(point[0], point[1]))))
    # Not bit-identical: the three sample points sit at slightly different
    # nearest-stations on a dense centreline, and the section mean varies
    # along the reach.  What matters is that none of the variation comes
    # from the lateral position.
    assert (max(speeds) - min(speeds)) / max(speeds) < 1e-3


def test_lateral_model_disagrees_with_the_uniform_one_at_the_bank(course):
    """Quantifies what the uniform model hides.

    At the bank the uniform model overstates the adverse current a crew
    rowing upstream would meet.
    """
    flow = charles.ContinuityFlow(course, discharge=20.0)
    lateral = flow.as_current_field()
    uniform = flow.as_current_field(lateral=False)
    station = 0.45 * course.length
    point = course.offset_position(np.array(station),
                                   np.array(0.9 * course.half_width_at(station)))
    near_bank = float(np.hypot(*lateral(point[0], point[1])))
    section_mean = float(np.hypot(*uniform(point[0], point[1])))
    assert near_bank < 0.75 * section_mean
