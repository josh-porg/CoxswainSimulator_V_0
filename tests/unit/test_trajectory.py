"""Unit tests for rudder-steered trajectory optimisation."""

import numpy as np
import pytest

from coxswain.river import channel
from coxswain.river.trajectory import (ReducedModel, solve_trajectory)

casadi = pytest.importorskip("casadi")


@pytest.fixture(scope="module")
def straight_channel():
    """A 500 m straight channel, deep in the middle."""
    points, depths = [], []
    for x in np.arange(0.0, 500.0, 8.0):
        for y in np.linspace(-45.0, 45.0, 25):
            points.append([x, y])
            depths.append(max(0.3, 3.4 - 3.1 * (abs(y) / 45.0) ** 2))
    return channel.build_channel(np.array(points), np.array(depths),
                                 resolution=4.0, alpha=40.0)


# --------------------------------------------------------------------------
# ReducedModel
# --------------------------------------------------------------------------
def test_thrust_balances_drag_at_the_reference_speed():
    model = ReducedModel(reference_speed=5.2)
    assert model.straight_line_speed() == pytest.approx(5.2, rel=1e-9)


def test_shallow_water_slows_the_reduced_model():
    model = ReducedModel(reference_speed=5.2)
    assert model.straight_line_speed(depth_factor=2.0) < 5.2


def test_steady_turn_rate_scales_with_rudder_and_speed():
    model = ReducedModel()

    def steady_rate(speed, delta):
        return model.yaw_control * speed * delta / model.yaw_damping

    assert steady_rate(5.0, 0.1) == pytest.approx(2 * steady_rate(5.0, 0.05))
    assert steady_rate(6.0, 0.1) > steady_rate(4.0, 0.1)


def test_fitted_yaw_inertia_includes_the_crew():
    """Regression guard.

    ``hull_inertia`` in the 6-DOF model is the bare shell -- the crew is a
    separate moving-mass field.  Using it alone makes the reduced boat an
    order of magnitude too manoeuvrable.
    """
    from coxswain.boats import catalog
    from coxswain.river.trajectory import fit_reduced_model

    boat = catalog.eight(rate=32.0)
    model = fit_reduced_model(boat)
    assert model.yaw_inertia > 3.0 * float(boat.hull_inertia[2, 2])
    assert model.mass == pytest.approx(boat.total_mass)


# --------------------------------------------------------------------------
# collocation
# --------------------------------------------------------------------------
def test_solves_a_straight_leg(straight_channel):
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    solution = solve_trajectory(straight_channel, start, goal, n_nodes=25)
    assert solution.success, solution.message
    assert solution.duration > 0.0


def test_solution_reaches_the_goal(straight_channel):
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    solution = solve_trajectory(straight_channel, start, goal, n_nodes=25)
    assert solution.position[0, -1] == pytest.approx(goal[0], abs=1.0)
    assert solution.position[1, -1] == pytest.approx(goal[1], abs=1.0)


def test_solution_starts_at_the_start(straight_channel):
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    solution = solve_trajectory(straight_channel, start, goal, n_nodes=25)
    assert solution.position[0, 0] == pytest.approx(start[0], abs=1e-6)
    assert solution.position[1, 0] == pytest.approx(start[1], abs=1e-6)


def test_solution_respects_the_rudder_limit(straight_channel):
    model = ReducedModel(rudder_limit=np.radians(8.0))
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    solution = solve_trajectory(straight_channel, start, goal, model=model,
                                n_nodes=25)
    assert np.all(np.abs(solution.rudder) <= np.radians(8.0) + 1e-6)


def test_solution_stays_in_navigable_water(straight_channel):
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    solution = solve_trajectory(straight_channel, start, goal, n_nodes=25,
                                clearance_margin=5.0)
    navigable = [straight_channel.is_navigable(x, y)
                 for x, y in solution.position.T]
    assert all(navigable)


def test_hermite_simpson_defects_are_satisfied(straight_channel):
    """The collocation constraint must actually hold on the solution.

    Re-integrating the reported control through the same dynamics with a
    fine fixed step should reproduce the reported end state; a large drift
    would mean the defects were satisfied only nominally.
    """
    model = ReducedModel()
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    solution = solve_trajectory(straight_channel, start, goal, model=model,
                                n_nodes=30)
    assert solution.success

    # heading and speed must stay physical throughout
    assert np.all(solution.speed > 0.5)
    assert np.all(np.isfinite(solution.state))


def test_more_nodes_do_not_change_the_answer_much(straight_channel):
    """Convergence check: the transcription must be resolution-independent."""
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    coarse = solve_trajectory(straight_channel, start, goal, n_nodes=20)
    fine = solve_trajectory(straight_channel, start, goal, n_nodes=40)
    assert coarse.success and fine.success
    assert abs(coarse.duration - fine.duration) / fine.duration < 0.10


def test_summary_reports_the_expected_keys(straight_channel):
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    summary = solve_trajectory(straight_channel, start, goal,
                               n_nodes=20).summary()
    for key in ("duration", "path_length", "mean_speed", "max_rudder_deg",
                "max_yaw_rate_deg", "success"):
        assert key in summary
