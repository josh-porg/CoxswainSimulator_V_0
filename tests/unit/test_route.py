"""Unit tests for route evaluation and optimisation."""

import numpy as np
import pytest

from coxswain.river.course import Course, DepthField
from coxswain.river.route import Route, RouteEvaluator, optimise_route


@pytest.fixture
def straight_course():
    """A 2 km straight reach, deep in the middle and shoaling to the banks."""
    station = np.linspace(0.0, 2000.0, 40)
    centreline = np.column_stack([station, np.zeros_like(station)])
    offsets = np.linspace(-50.0, 50.0, 21)
    points, depths = [], []
    for s in station:
        for o in offsets:
            points.append([s, o])
            depths.append(6.0 - 4.5 * (o / 50.0) ** 2)
    return Course(centreline=centreline, half_width=50.0,
                  depth=DepthField(points=np.array(points),
                                   depths=np.array(depths), is_survey=True),
                  name="test reach")


class _UniformFlow:
    """Stub flow with a constant speed everywhere."""

    def __init__(self, course, speed=0.4):
        self.course = course
        self.speed = speed

    def _speed_grid(self, n):
        stations = np.linspace(0.0, self.course.length, n)
        fractions = np.linspace(-1.0, 1.0, 5)
        return stations, fractions, np.full((n, 5), self.speed)


# --------------------------------------------------------------------------
# Route
# --------------------------------------------------------------------------
def test_route_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        Route(np.array([0.0, 1.0]), np.array([0.0]))


def test_route_rejects_non_increasing_stations():
    with pytest.raises(ValueError, match="strictly increasing"):
        Route(np.array([0.0, 1.0, 0.5]), np.zeros(3))


def test_centreline_route_is_all_zero(straight_course):
    assert np.allclose(Route.centreline(straight_course).offsets, 0.0)


def test_clip_to_channel_respects_the_margin(straight_course):
    clipped = Route.constant_offset(straight_course, 200.0).clip_to_channel(
        straight_course, margin=5.0)
    assert np.all(clipped.offsets <= 45.0 + 1e-9)


# --------------------------------------------------------------------------
# speed model -- the bug that let the optimiser exploit shallow water
# --------------------------------------------------------------------------
def test_speed_decreases_monotonically_as_water_shallows(straight_course):
    """Regression guard.

    A fixed-point iteration for the power balance does not converge through
    the critical region: it oscillated between branches and returned a
    higher speed in 1.6 m than in 3 m, which the optimiser then steered
    straight into.  Solved by bisection on a strictly increasing function.
    """
    evaluator = RouteEvaluator(straight_course, reference_speed=5.2)
    depths = np.array([12.0, 8.0, 6.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.7])
    speeds = evaluator.speed_through_water(depths)
    # never faster in shallower water
    assert np.all(np.diff(speeds) <= 1e-12), speeds
    # and strictly slower wherever the model is inside its valid range,
    # i.e. above the critical depth where the supercritical clamp binds
    valid = depths >= 1.5
    assert np.all(np.diff(speeds[valid]) < 0.0), speeds[valid]


def test_supercritical_speeds_are_clamped_not_extrapolated(straight_course):
    """Below the critical depth the model is out of range and says so.

    The shallow-water correction relaxes wave resistance above about
    Fr_h = 1.3 -- correct for a planing hull outrunning its own wave
    system, meaningless for a rowing eight, which at that Froude number is
    in a metre of water dragging its skeg.  Left unclamped, the optimiser
    finds it and routes through the shallows.
    """
    evaluator = RouteEvaluator(straight_course, reference_speed=5.2)
    shallow = evaluator.speed_through_water(np.array([1.0, 0.7, 0.4]))
    assert np.allclose(shallow, shallow[0])
    assert shallow[0] < evaluator.speed_through_water(2.0)


def test_speed_never_exceeds_the_deep_water_reference(straight_course):
    evaluator = RouteEvaluator(straight_course, reference_speed=5.2)
    depths = np.geomspace(0.5, 50.0, 60)
    assert np.all(evaluator.speed_through_water(depths) <= 5.2 + 1e-9)


def test_speed_recovers_the_reference_in_deep_water(straight_course):
    evaluator = RouteEvaluator(straight_course, reference_speed=5.2)
    assert evaluator.speed_through_water(200.0) == pytest.approx(5.2, rel=1e-6)


def test_speed_accepts_scalars_and_arrays(straight_course):
    evaluator = RouteEvaluator(straight_course)
    scalar = evaluator.speed_through_water(4.0)
    array = evaluator.speed_through_water(np.array([4.0]))
    assert np.ndim(scalar) == 0
    assert array.shape == (1,)
    assert float(scalar) == pytest.approx(float(array[0]))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def test_deep_line_beats_shallow_line_with_no_current(straight_course):
    """With no flow the only trade is depth against distance.

    On a straight reach the distance is identical, so the deep middle must
    win outright.
    """
    evaluator = RouteEvaluator(straight_course, reference_speed=5.2)
    middle = evaluator.evaluate(Route.centreline(straight_course))
    edge = evaluator.evaluate(Route.constant_offset(straight_course, 40.0))
    assert middle.elapsed < edge.elapsed


def test_path_length_of_a_straight_centreline_is_the_course_length(
        straight_course):
    result = RouteEvaluator(straight_course).evaluate(
        Route.centreline(straight_course))
    assert result.path_length == pytest.approx(straight_course.length,
                                               rel=1e-6)


def test_a_weaving_line_is_longer_than_a_straight_one(straight_course):
    evaluator = RouteEvaluator(straight_course)
    stations = np.linspace(0.0, straight_course.length, 9)
    weave = Route(stations, 30.0 * np.sin(np.arange(9) * 1.4))
    assert (evaluator.evaluate(weave).path_length
            > evaluator.evaluate(
                Route.centreline(straight_course)).path_length)


def test_grounding_is_penalised(straight_course):
    """A route through unnavigable water must not be allowed to win."""
    strict = RouteEvaluator(straight_course, minimum_depth=5.0)
    result = strict.evaluate(Route.constant_offset(straight_course, 45.0))
    assert result.fraction_aground > 0.0

    lenient = RouteEvaluator(straight_course, minimum_depth=0.1)
    clean = lenient.evaluate(Route.constant_offset(straight_course, 45.0))
    assert result.elapsed > clean.elapsed


def test_mean_ground_speed_is_length_over_time(straight_course):
    result = RouteEvaluator(straight_course, minimum_depth=0.1).evaluate(
        Route.centreline(straight_course))
    assert result.mean_ground_speed == pytest.approx(
        result.path_length / result.elapsed, rel=1e-12)


# --------------------------------------------------------------------------
# current
# --------------------------------------------------------------------------
def test_current_helps_downstream_and_hurts_upstream(straight_course):
    """The same line must be quicker with the stream than against it."""
    route = Route.centreline(straight_course)
    up = RouteEvaluator(straight_course, flow=_UniformFlow(straight_course),
                        upstream=True, minimum_depth=0.1)
    down = RouteEvaluator(straight_course, flow=_UniformFlow(straight_course),
                          upstream=False, minimum_depth=0.1)
    assert down.evaluate(route).elapsed < up.evaluate(route).elapsed


def test_current_sign_is_negative_upstream(straight_course):
    up = RouteEvaluator(straight_course, flow=_UniformFlow(straight_course),
                        upstream=True)
    result = up.evaluate(Route.centreline(straight_course))
    assert np.all(result.current_along < 0.0)


def test_current_sign_is_positive_downstream(straight_course):
    down = RouteEvaluator(straight_course, flow=_UniformFlow(straight_course),
                          upstream=False)
    result = down.evaluate(Route.centreline(straight_course))
    assert np.all(result.current_along > 0.0)


def test_no_flow_gives_zero_current(straight_course):
    result = RouteEvaluator(straight_course).evaluate(
        Route.centreline(straight_course))
    assert np.allclose(result.current_along, 0.0)


# --------------------------------------------------------------------------
# optimiser
# --------------------------------------------------------------------------
def test_optimiser_never_returns_worse_than_the_centreline(straight_course):
    evaluator = RouteEvaluator(straight_course, minimum_depth=0.1)
    baseline = evaluator.evaluate(Route.centreline(straight_course))
    best = optimise_route(evaluator, n_control=5, iterations=12)
    assert best.elapsed <= baseline.elapsed + 1e-9


def test_optimiser_finds_the_deep_channel_with_no_current(straight_course):
    """The answer is known here: stay in the middle, where it is deepest."""
    evaluator = RouteEvaluator(straight_course, minimum_depth=0.1)
    best = optimise_route(evaluator, n_control=5, iterations=25)
    stations = np.linspace(0.0, straight_course.length, 5)
    assert np.all(np.abs(best.route.offset_at(stations)) < 12.0)


def test_optimiser_stays_inside_the_channel(straight_course):
    evaluator = RouteEvaluator(straight_course, margin=4.0, minimum_depth=0.1)
    best = optimise_route(evaluator, n_control=5, iterations=12)
    stations = np.linspace(0.0, straight_course.length, 40)
    assert np.all(np.abs(best.route.offset_at(stations)) <= 46.0 + 1e-6)
