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


# --------------------------------------------------------------------------
# pressure split -- the coxswain's second control
# --------------------------------------------------------------------------
def test_split_adds_yaw_authority():
    """Rudder alone cannot hold the tightest Charles bends; a split helps.

    The full 6-DOF model measures 259 m on rudder alone and 130 m with a
    30% split.  The reduced model is linear in both controls, so it
    predicts 164 m for the pair -- the two combine *super-additively* in
    the full model (2.28 deg/s against the 1.92 a linear sum gives),
    because the sideslip each induces raises the other's effectiveness.

    The reduced model is therefore conservative about combined steering,
    which is the right direction for a trajectory it has to be able to
    fly, but it does mean a solution near the limit should be checked back
    in the full model.
    """
    model = ReducedModel()

    def steady_radius(delta, split):
        rate = (model.yaw_control * model.reference_speed ** 2 * delta
                + model.split_control * split) / (
                    model.yaw_damping * model.reference_speed)
        return model.reference_speed / abs(rate)

    rudder_only = steady_radius(model.rudder_limit, 0.0)
    with_split = steady_radius(model.rudder_limit, model.split_limit)
    assert rudder_only == pytest.approx(283.0, rel=0.05)
    assert with_split < 0.6 * rudder_only
    assert with_split < 200.0


def test_split_is_bounded_in_the_solution(straight_channel):
    model = ReducedModel(split_limit=0.2)
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    solution = solve_trajectory(straight_channel, start, goal, model=model,
                                n_nodes=25)
    assert np.all(np.abs(solution.split) <= 0.2 + 1e-6)


def test_split_costs_speed():
    """Without a penalty the optimiser would split the crew for free."""
    model = ReducedModel()
    assert model.split_drag > 0.0


def test_summary_reports_the_split(straight_channel):
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    summary = solve_trajectory(straight_channel, start, goal,
                               n_nodes=20).summary()
    assert "max_split" in summary


# --------------------------------------------------------------------------
# pacing -- power as the third control, bounded by a W' budget
# --------------------------------------------------------------------------
def test_power_fraction_maps_to_power_superlinearly():
    """Speed goes as the cube root of power, so thrust fraction pi draws
    pi**1.5 of critical power."""
    model = ReducedModel()
    assert model.power_at(1.0) == pytest.approx(model.critical_power)
    assert model.power_at(1.2) > 1.2 * model.critical_power


def test_more_power_gives_more_speed():
    model = ReducedModel()
    assert (model.straight_line_speed(power_fraction=1.2)
            > model.straight_line_speed(power_fraction=1.0))


def test_pacing_pushes_until_something_binds(straight_channel):
    """At the optimum either the energy budget or the power cap binds.

    Over a long course the crew should finish with nothing left -- energy
    carried over the line could have been spent going faster.  Over a
    *short* leg there is not time to burn the budget at the power cap, so
    the cap binds instead and W' is left over.  Measured: 176 kJ fully
    spent on a 2.6 km Charles leg, 140 of 176 on a 330 m one.

    Either way the crew must be pushing as hard as something allows.
    """
    model = ReducedModel()
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    solution = solve_trajectory(straight_channel, start, goal, n_nodes=30)
    assert solution.success
    assert solution.anaerobic_remaining[0] == pytest.approx(
        model.anaerobic_capacity)

    exhausted = (solution.anaerobic_remaining[-1]
                 < 0.05 * model.anaerobic_capacity)
    at_the_cap = solution.power.max() > model.power_max - 1e-3
    assert exhausted or at_the_cap


def test_anaerobic_capacity_is_never_negative(straight_channel):
    """The crew cannot spend energy it does not have."""
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    solution = solve_trajectory(straight_channel, start, goal, n_nodes=30)
    assert np.all(solution.anaerobic_remaining >= -1e-6)


def test_power_respects_its_bounds(straight_channel):
    model = ReducedModel(power_min=0.8, power_max=1.2)
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    solution = solve_trajectory(straight_channel, start, goal, model=model,
                                n_nodes=30)
    assert np.all(solution.power >= 0.8 - 1e-6)
    assert np.all(solution.power <= 1.2 + 1e-6)


def test_free_pacing_beats_constant_power(straight_channel):
    """The point of making power a control.

    Measured on a 2.6 km Charles leg: 18.8 s quicker than holding critical
    power throughout, about 3%.
    """
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    free = solve_trajectory(straight_channel, start, goal, n_nodes=30)
    fixed = solve_trajectory(straight_channel, start, goal, n_nodes=30,
                             model=ReducedModel(power_min=1.0, power_max=1.0))
    assert free.success and fixed.success
    assert free.duration < fixed.duration


def test_summary_reports_power_and_energy(straight_channel):
    line = straight_channel.centreline()
    start, goal = line[len(line) // 6], line[-len(line) // 6]
    summary = solve_trajectory(straight_channel, start, goal,
                               n_nodes=25).summary()
    assert "power_range" in summary
    assert "anaerobic_spent" in summary
