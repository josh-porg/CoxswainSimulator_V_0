"""Unit tests for river depth, current and course geometry."""

import numpy as np
import pytest

from coxswain.river.course import (
    Course,
    CurrentField,
    DepthField,
    charles_river_sketch,
    local_tangent_plane,
)


@pytest.fixture
def straight_course():
    return Course(
        centreline=np.array([[0.0, 0.0], [1000.0, 0.0]]),
        half_width=30.0,
        depth=DepthField.uniform_depth(4.0),
        name="straight",
    )


# --------------------------------------------------------------------------
# DepthField
# --------------------------------------------------------------------------
def test_uniform_depth_is_constant_everywhere():
    depth = DepthField.uniform_depth(3.5)
    assert depth(0.0, 0.0) == pytest.approx(3.5)
    assert depth(1e4, -1e4) == pytest.approx(3.5)


def test_uniform_depth_broadcasts():
    depth = DepthField.uniform_depth(3.5)
    values = depth(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(values, 3.5)


def test_depth_field_requires_a_depth_or_points():
    with pytest.raises(ValueError, match="uniform depth or survey points"):
        DepthField()


def test_survey_depth_interpolates_between_soundings():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    depths = np.array([2.0, 4.0, 2.0, 4.0])
    field = DepthField(points=points, depths=depths)
    assert field(5.0, 5.0) == pytest.approx(3.0, abs=1e-9)


def test_survey_depth_falls_back_to_nearest_outside_the_hull():
    """Extrapolated bathymetry is worse than none, so clamp to nearest."""
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    depths = np.array([2.0, 4.0, 2.0, 4.0])
    field = DepthField(points=points, depths=depths)
    far = field(500.0, 500.0)
    assert np.isfinite(far)
    assert 2.0 <= far <= 4.0


def test_survey_depth_respects_the_minimum():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    depths = np.array([0.6, 0.6, 0.6, 0.6])
    field = DepthField(points=points, depths=depths, minimum=1.0)
    assert field(5.0, 5.0) == pytest.approx(1.0)


def test_survey_depth_rejects_non_positive_soundings():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="positive"):
        DepthField(points=points, depths=np.array([1.0, -1.0, 2.0]))


def test_survey_depth_rejects_mismatched_shapes():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="one entry per point"):
        DepthField(points=points, depths=np.array([1.0, 2.0]))


def test_depth_field_builds_a_matching_shallow_model():
    field = DepthField.uniform_depth(3.0)
    model = field.shallow_model(0.0, 0.0)
    assert model.depth == pytest.approx(3.0)
    assert model.enabled


# --------------------------------------------------------------------------
# CurrentField
# --------------------------------------------------------------------------
def test_still_water_has_no_current():
    current = CurrentField.still()
    assert current.is_still
    np.testing.assert_allclose(current(0.0, 0.0), [0.0, 0.0])


def test_uniform_current_returns_its_vector():
    current = CurrentField(velocity=(-0.3, 0.1))
    np.testing.assert_allclose(current(100.0, 50.0), [-0.3, 0.1])
    assert not current.is_still


def test_current_velocity_3d_has_zero_vertical_component():
    current = CurrentField(velocity=(-0.3, 0.1))
    np.testing.assert_allclose(current.velocity_3d(0.0, 0.0), [-0.3, 0.1, 0.0])


def test_callable_current_is_evaluated_at_the_position():
    current = CurrentField(function=lambda x, y: (0.001 * x, 0.0))
    np.testing.assert_allclose(current(2000.0, 0.0), [2.0, 0.0])


def test_uniform_current_must_be_a_two_vector():
    with pytest.raises(ValueError, match="2-vector"):
        CurrentField(velocity=(1.0, 2.0, 3.0))


def test_current_does_not_alias_its_stored_vector():
    """A caller mutating the result must not corrupt the field."""
    current = CurrentField(velocity=(-0.3, 0.1))
    got = current(0.0, 0.0)
    got[0] = 99.0
    np.testing.assert_allclose(current(0.0, 0.0), [-0.3, 0.1])


# --------------------------------------------------------------------------
# Course geometry
# --------------------------------------------------------------------------
def test_course_length_is_the_polyline_length(straight_course):
    assert straight_course.length == pytest.approx(1000.0)


def test_course_rejects_a_degenerate_centreline():
    with pytest.raises(ValueError, match="at least two points"):
        Course(centreline=np.array([[0.0, 0.0]]), half_width=10.0,
               depth=DepthField.uniform_depth(3.0))


def test_course_rejects_repeated_points():
    with pytest.raises(ValueError, match="repeated points"):
        Course(centreline=np.array([[0.0, 0.0], [0.0, 0.0]]),
               half_width=10.0, depth=DepthField.uniform_depth(3.0))


def test_course_rejects_non_positive_width():
    with pytest.raises(ValueError, match="half_width must be positive"):
        Course(centreline=np.array([[0.0, 0.0], [10.0, 0.0]]),
               half_width=0.0, depth=DepthField.uniform_depth(3.0))


def test_position_at_walks_the_centreline(straight_course):
    np.testing.assert_allclose(
        straight_course.position_at(np.array(250.0)), [250.0, 0.0], atol=1e-9)


def test_heading_of_a_straight_course_is_zero(straight_course):
    assert straight_course.heading_at(np.array(500.0)) == pytest.approx(
        0.0, abs=1e-9)


def test_offset_position_displaces_to_port(straight_course):
    """Port is +y for a course heading along +x."""
    offset = straight_course.offset_position(np.array(500.0), np.array(10.0))
    np.testing.assert_allclose(offset, [500.0, 10.0], atol=1e-6)


def test_offset_position_is_perpendicular_to_the_centreline():
    course = Course(centreline=np.array([[0.0, 0.0], [100.0, 100.0]]),
                    half_width=30.0, depth=DepthField.uniform_depth(3.0))
    station = np.array(70.0)
    centre = course.position_at(station)
    offset = course.offset_position(station, np.array(12.0))
    step = offset - centre
    assert np.hypot(*step) == pytest.approx(12.0, abs=1e-6)
    heading = course.heading_at(station)
    along = np.array([np.cos(heading), np.sin(heading)])
    assert abs(step @ along) < 1e-6


def test_nearest_station_recovers_the_offset_station(straight_course):
    for station in (0.0, 123.0, 500.0, 999.0):
        point = straight_course.offset_position(np.array(station),
                                                np.array(7.0))
        assert straight_course.nearest_station(*point) == pytest.approx(
            station, abs=1e-6)


def test_nearest_station_projects_onto_segments_not_just_vertices():
    """A coarse polyline must still give the true perpendicular foot."""
    course = Course(centreline=np.array([[0.0, 0.0], [1000.0, 0.0]]),
                    half_width=30.0, depth=DepthField.uniform_depth(3.0))
    assert course.nearest_station(500.0, 5.0) == pytest.approx(500.0, abs=1e-9)


def test_is_inside_respects_the_channel_width(straight_course):
    assert straight_course.is_inside(500.0, 0.0)
    assert straight_course.is_inside(500.0, 29.0)
    assert not straight_course.is_inside(500.0, 31.0)


def test_half_width_may_vary_along_the_course():
    course = Course(
        centreline=np.array([[0.0, 0.0], [500.0, 0.0], [1000.0, 0.0]]),
        half_width=np.array([40.0, 10.0, 40.0]),
        depth=DepthField.uniform_depth(3.0),
    )
    assert course.half_width_at(500.0) == pytest.approx(10.0)
    assert not course.is_inside(500.0, 20.0)
    assert course.is_inside(0.0, 20.0)


def test_depth_profile_samples_the_centreline(straight_course):
    station, depth = straight_course.depth_profile(11)
    assert station[0] == pytest.approx(0.0)
    assert station[-1] == pytest.approx(straight_course.length)
    np.testing.assert_allclose(depth, 4.0)


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------
def test_tangent_plane_origin_maps_to_zero():
    east, north = local_tangent_plane(42.3601, -71.0942, (42.3601, -71.0942))
    assert float(east) == pytest.approx(0.0, abs=1e-6)
    assert float(north) == pytest.approx(0.0, abs=1e-6)


def test_tangent_plane_north_offset_matches_arc_length():
    """0.01 deg of latitude is about 1112 m anywhere on Earth."""
    _, north = local_tangent_plane(42.3701, -71.0942, (42.3601, -71.0942))
    assert float(north) == pytest.approx(1112.0, rel=0.01)


def test_tangent_plane_east_offset_shrinks_with_latitude():
    east, _ = local_tangent_plane(42.3601, -71.0842, (42.3601, -71.0942))
    expected = 1112.0 * np.cos(np.radians(42.3601))
    assert float(east) == pytest.approx(expected, rel=0.01)


# --------------------------------------------------------------------------
# The Charles sketch, and its guard
# --------------------------------------------------------------------------
def test_charles_sketch_is_marked_as_not_survey_data():
    course = charles_river_sketch()
    assert course.is_survey is False
    assert course.depth.is_survey is False
    assert "SKETCH" in course.name


def test_require_survey_refuses_the_sketch():
    """A sketch reproduces the shape of the problem and none of its values.

    Once results reach a table that distinction is invisible, so quoting a
    routing number has to go through this guard.
    """
    course = charles_river_sketch()
    with pytest.raises(ValueError, match="sketch, not survey data"):
        course.require_survey()


def test_require_survey_passes_for_data_marked_as_surveyed():
    course = Course(centreline=np.array([[0.0, 0.0], [100.0, 0.0]]),
                    half_width=20.0, depth=DepthField.uniform_depth(3.0),
                    is_survey=True)
    course.require_survey()


def test_charles_sketch_depths_stay_in_a_plausible_band():
    course = charles_river_sketch()
    _, depth = course.depth_profile(200)
    assert depth.min() > 1.5
    assert depth.max() < 8.5


def test_charles_sketch_narrows_at_the_bridges():
    course = charles_river_sketch()
    assert course.half_width.min() < 30.0
    assert course.half_width.max() > 40.0


def test_charles_sketch_has_a_downstream_current():
    course = charles_river_sketch()
    assert not course.current.is_still
