"""Unit tests for the stroke-resolved CasADi model."""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.core.rigid_body import assemble_mass_matrix
from coxswain.river.strokemodel import (HydroCoefficients, StrokeAggregates,
                                        StrokePeriodicFit,
                                        StrokeResolvedModel,
                                        planar_mass_matrix)

casadi = pytest.importorskip("casadi")


@pytest.fixture(scope="module")
def boat():
    return catalog.eight(rate=32.0)


@pytest.fixture(scope="module")
def model(boat):
    return StrokeResolvedModel(boat)


# --------------------------------------------------------------------------
# Fourier fits
# --------------------------------------------------------------------------
def test_fit_reproduces_a_pure_harmonic():
    period = 2.0
    t = np.linspace(0.0, period, 128, endpoint=False)
    samples = 3.0 + 2.0 * np.cos(2 * np.pi * t / period) \
        - 1.5 * np.sin(4 * np.pi * t / period)
    fit = StrokePeriodicFit.fit(samples, period, n_harmonics=4)
    np.testing.assert_allclose(fit(t), samples, atol=1e-10)


def test_fit_mean_is_the_dc_term():
    period = 1.5
    samples = 7.0 + np.sin(2 * np.pi * np.arange(64) / 64)
    fit = StrokePeriodicFit.fit(samples, period, n_harmonics=3)
    assert fit.mean == pytest.approx(7.0, abs=1e-12)


def test_fit_is_periodic():
    fit = StrokePeriodicFit.fit(np.random.default_rng(0).normal(size=64),
                                2.0, n_harmonics=5)
    assert float(fit(0.3)) == pytest.approx(float(fit(2.3)), abs=1e-10)


def test_derivative_matches_finite_differences():
    """The analytic derivative must not drift from its parent fit."""
    period = 1.875
    rng = np.random.default_rng(1)
    fit = StrokePeriodicFit.fit(rng.normal(size=128), period, n_harmonics=6)
    rate = fit.derivative()
    step = 1e-6
    for t in np.linspace(0.0, period, 11):
        numerical = (float(fit(t + step)) - float(fit(t - step))) / (2 * step)
        assert float(rate(t)) == pytest.approx(numerical, abs=1e-4)


def test_derivative_of_a_constant_is_zero():
    fit = StrokePeriodicFit.fit(np.full(32, 4.0), 1.0, n_harmonics=3)
    assert float(fit.derivative()(0.4)) == pytest.approx(0.0, abs=1e-12)


def test_casadi_and_numpy_evaluation_agree():
    import casadi as ca

    fit = StrokePeriodicFit.fit(np.random.default_rng(2).normal(size=64),
                                2.0, n_harmonics=5)
    t = ca.MX.sym("t")
    f = ca.Function("f", [t], [fit.casadi(t)])
    for value in (0.0, 0.37, 1.1, 1.99):
        assert float(f(value)) == pytest.approx(float(fit(value)), abs=1e-12)


# --------------------------------------------------------------------------
# aggregates
# --------------------------------------------------------------------------
def test_aggregates_reproduce_the_numpy_crew_field(boat, model):
    """The fit must not diverge from the model it was fitted from."""
    aggregates = model.aggregates
    for t in np.linspace(0.0, boat.timing.period, 17, endpoint=False):
        mass, position, _, _ = boat.crew_field(t)
        assert float(aggregates.first_moment(t)) == pytest.approx(
            float(np.sum(mass * position[:, 0])), abs=0.05)
        assert float(aggregates.yaw_inertia(t)) == pytest.approx(
            float(np.sum(mass * (position[:, 0] ** 2
                                 + position[:, 1] ** 2))), abs=0.5)


def test_crew_yaw_inertia_is_large_and_nearly_steady(model):
    """The crew dominate the yaw inertia but barely change it.

    ~7800 kg m2 against 1915 for the bare hull, varying only 6% over the
    stroke -- which is why the crew's yaw reaction is a small correction
    while their contribution to the inertia is not.
    """
    aggregates = model.aggregates
    times = np.linspace(0.0, aggregates.period, 128, endpoint=False)
    inertia = np.asarray(aggregates.yaw_inertia(times))
    assert inertia.mean() > 5000.0
    assert (inertia.max() - inertia.min()) / inertia.mean() < 0.12


def test_thrust_is_zero_on_the_recovery(model, boat):
    """Blades out of the water: no thrust, and hence no split authority."""
    aggregates = model.aggregates
    mid_recovery = boat.timing.period * (
        boat.timing.drive_fraction + 1.0) / 2.0
    peak_drive = boat.timing.period * boat.timing.drive_fraction * 0.5
    assert abs(float(aggregates.thrust(mid_recovery))) < 0.3 * abs(
        float(aggregates.thrust(peak_drive)))


def test_split_authority_follows_the_thrust(model, boat):
    """The split is a scaling of oar force, so it vanishes with it."""
    aggregates = model.aggregates
    mid_recovery = boat.timing.period * (
        boat.timing.drive_fraction + 1.0) / 2.0
    peak_drive = boat.timing.period * boat.timing.drive_fraction * 0.5
    assert abs(float(aggregates.yaw_per_split(mid_recovery))) < 0.3 * abs(
        float(aggregates.yaw_per_split(peak_drive)))


# --------------------------------------------------------------------------
# mass matrix -- transcription of the tested 3-D form
# --------------------------------------------------------------------------
def test_planar_mass_matrix_is_symmetric():
    matrix = planar_mass_matrix(855.0, 9700.0, 120.0, -45.0)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)


def test_planar_mass_matrix_matches_the_three_dimensional_one():
    """Regression guard on the transcription.

    The planar blocks must equal the corresponding entries of
    :func:`assemble_mass_matrix`, which is checked against the paper.
    """
    mass = np.array([40.0, 60.0])
    position = np.array([[1.5, 0.4, 0.0], [-2.0, -0.3, 0.0]])
    total, inertia_zz = 855.0, 1915.0

    full = assemble_mass_matrix(total, np.diag([10.0, 500.0, inertia_zz]),
                                mass, position)
    first = (mass[:, None] * position).sum(axis=0)
    crew_zz = float(np.sum(mass * (position[:, 0] ** 2
                                   + position[:, 1] ** 2)))
    planar = planar_mass_matrix(total, inertia_zz + crew_zz,
                                first[0], first[1])

    assert planar[0, 0] == pytest.approx(full[0, 0])
    assert planar[0, 2] == pytest.approx(full[0, 5])
    assert planar[1, 2] == pytest.approx(full[1, 5])
    assert planar[2, 0] == pytest.approx(full[5, 0])
    assert planar[2, 2] == pytest.approx(full[5, 5])


def test_planar_mass_matrix_is_positive_definite():
    matrix = planar_mass_matrix(855.0, 9700.0, 200.0, 90.0)
    assert np.linalg.eigvalsh(matrix).min() > 0.0


# --------------------------------------------------------------------------
# hydrodynamic coefficients
# --------------------------------------------------------------------------
def test_weathervane_is_stabilising(boat):
    """Positive N per (u v): crabbing to port turns the bow to port,
    which reduces the sideslip.  Without this the model spins up."""
    hydro = HydroCoefficients.from_boat(boat)
    assert hydro.yaw_from_sway > 0.0


def test_yaw_damping_opposes_rotation(boat):
    hydro = HydroCoefficients.from_boat(boat)
    assert hydro.yaw_from_yaw < 0.0


def test_sway_damping_opposes_sideslip(boat):
    hydro = HydroCoefficients.from_boat(boat)
    assert hydro.sway_from_sway_linear < 0.0
    assert hydro.sway_from_sway_quadratic < 0.0


def test_weathervane_dominates_the_rudder(boat):
    """Why an eight turns badly.

    Ignoring sideslip, full rudder implies about 3.5 deg/s; the measured
    figure is 1.1.  The difference is the skeg weathervaning against the
    turn, and it is the largest single term in the yaw balance.
    """
    hydro = HydroCoefficients.from_boat(boat)
    speed = 5.2
    naive = abs(hydro.yaw_from_rudder * speed ** 2 * np.radians(12.0)
                / (hydro.yaw_from_yaw * speed))
    assert np.degrees(naive) > 2.5


# --------------------------------------------------------------------------
# the assembled dynamics
# --------------------------------------------------------------------------
def test_dynamics_builds_a_callable_function(model):
    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 176000.0])
    value = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()
    assert value.shape == (7,)
    assert np.all(np.isfinite(value))


def test_straight_and_level_produces_no_turn(model):
    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 176000.0])
    value = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()
    assert value[2] == pytest.approx(0.0, abs=1e-12)   # psi_dot
    assert value[4] == pytest.approx(0.0, abs=1e-9)    # v_dot
    assert value[5] == pytest.approx(0.0, abs=1e-9)    # r_dot


def test_position_derivative_is_the_rotated_velocity(model):
    function = model.function()
    psi = 0.6
    state = np.array([0.0, 0.0, psi, 5.0, 0.3, 0.0, 176000.0])
    value = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()
    assert value[0] == pytest.approx(5.0 * np.cos(psi) - 0.3 * np.sin(psi))
    assert value[1] == pytest.approx(5.0 * np.sin(psi) + 0.3 * np.cos(psi))


def test_rudder_turns_the_boat(model):
    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 176000.0])
    straight = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()
    turning = np.array(
        function(state, [np.radians(12.0), 0.0, 1.0], 0.2)).ravel()
    assert abs(turning[5]) > abs(straight[5])


def test_split_turns_the_boat_only_on_the_drive(model, boat):
    """The physical point of the whole module.

    A pressure split makes a yaw moment during the drive and none on the
    recovery, because the blades are out of the water.
    """
    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 176000.0])
    drive = boat.timing.period * boat.timing.drive_fraction * 0.5
    recovery = boat.timing.period * (boat.timing.drive_fraction + 1.0) / 2.0

    on_drive = np.array(function(state, [0.0, 0.30, 1.0], drive)).ravel()
    on_recovery = np.array(function(state, [0.0, 0.30, 1.0], recovery)).ravel()
    assert abs(on_drive[5]) > 3.0 * abs(on_recovery[5])


def test_surge_oscillation_matches_the_full_model(model, boat):
    """The reason for resolving the stroke at all.

    Integrating the stroke model reproduces the 6-DOF surge swing to
    better than 5%; the stroke-averaged model has no swing at all.
    """
    from coxswain.sim.simulator import RowingSimulator

    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 176000.0])
    dt, steps = 0.004, 2000
    surge = np.empty(steps)
    t = 0.0
    for i in range(steps):
        surge[i] = state[3]
        step = np.array(function(state, [0.0, 0.0, 1.0], t)).ravel()
        state = state + dt * step
        t += dt
    tail = surge[int(0.6 * steps):]

    reference = RowingSimulator(boat).run(duration=8.0, dt=0.006,
                                          surge_speed=5.2)
    window = reference.last_cycles(2)
    expected = float(np.ptp(reference.surge_speed[window]))
    assert float(np.ptp(tail)) == pytest.approx(expected, rel=0.05)
