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
    assert speed.max() < 0.10


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
