"""Unit tests for the stroke-resolved CasADi model."""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.core.rigid_body import assemble_mass_matrix
from coxswain.sim.simulator import DEFAULT_MUNK_FACTOR
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
        # 0.08 kg m, i.e. 0.12% of a first moment near 50.  The
        # tolerance is a numerical choice about the Fourier fit, not
        # a physical claim; the longer drive of SOURCES sec. 50
        # shifted the harmonic content slightly and 0.05 became
        # marginal at the same harmonic count.
        assert float(aggregates.first_moment(t)) == pytest.approx(
            float(np.sum(mass * position[:, 0])), abs=0.08)
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
def _sideslip_yaw_slope(boat, munk_factor, speed=5.2, sway=0.30,
                        sample_time=0.35):
    """``N`` per ``(u v)`` from the appendage channel, at a given Munk factor.

    The simulator adds the Munk moment into ``appendage_moment`` rather
    than ``resistance_moment`` (see ``simulator.py``), so the hull's
    destabilising term and the skeg's stabilising one arrive summed.
    Running the same perturbation with the Munk moment switched off is
    what separates them.
    """
    from coxswain.core.frames import abs_to_hull, attitude_from_components
    from coxswain.core.state import State
    from coxswain.sim.control import BalanceController
    from coxswain.sim.simulator import RowingSimulator

    simulator = RowingSimulator(boat, munk_factor=munk_factor)
    simulator.coxswain.balance = BalanceController(enabled=False)
    simulator.coxswain.rudder_override = lambda _t, _s: 0.0

    def moment(drift):
        state = State.create(attitude=attitude_from_components(roll=0.0),
                             velocity=(speed, drift, 0.0),
                             omega=(0.0, 0.0, 0.0))
        breakdown = simulator.breakdown(sample_time, state)
        rotation = abs_to_hull(state.attitude)
        return float((rotation @ breakdown.appendage_moment)[2])

    return (moment(sway) - moment(0.0)) / (speed * sway)


def test_the_munk_moment_is_destabilising_in_sideslip(boat):
    """The Munk contribution to ``N`` per ``(u v)`` is **negative**.

    A slender body in potential flow is destabilised by drift: the
    entrained water gives a moment that turns the hull *broadside*, and
    for a rowing shell it is large.  This assertion used to be the
    opposite, and it held only because the model had no Munk moment at
    all.  See SOURCES sec. 36.
    """
    with_munk = _sideslip_yaw_slope(boat, DEFAULT_MUNK_FACTOR)
    without = _sideslip_yaw_slope(boat, 0.0)
    assert with_munk - without < 0.0


def test_the_skeg_more_than_recovers_the_munk_instability(boat):
    """Assembled, the term is **positive** -- and both facts are needed.

    This test previously asserted that the assembled boat was unstable,
    which contradicted :class:`HydroCoefficients`' own docstring and began
    failing once appendages entered the perturbation.  The docstring has
    the physics right: the skeg weathervanes hard enough to outweigh the
    Munk moment, which is why a shell tracks straight at all.

    The evidence usually cited for instability -- that losing the skeg
    makes a shell uncontrollable -- is evidence about what the skeg is
    holding off, not about the assembled boat.  Both halves are pinned
    here so the two claims cannot be confused for each other again.
    """
    with_munk = _sideslip_yaw_slope(boat, DEFAULT_MUNK_FACTOR)
    without = _sideslip_yaw_slope(boat, 0.0)
    assert without > 0.0                      # the skeg alone
    assert with_munk > 0.0                     # and it wins
    assert with_munk < without                 # but pays for the win
    assert HydroCoefficients.from_boat(boat).yaw_from_sway > 0.0


def test_yaw_damping_opposes_rotation(boat):
    hydro = HydroCoefficients.from_boat(boat)
    assert hydro.yaw_from_yaw < 0.0


def test_sway_damping_opposes_sideslip(boat):
    hydro = HydroCoefficients.from_boat(boat)
    assert hydro.sway_from_sway_linear < 0.0
    assert hydro.sway_from_sway_quadratic < 0.0


def test_weathervane_dominates_the_rudder(boat):
    """Why an eight turns badly.

    Ignoring sideslip, full rudder implies more than the measured
    1.1 deg/s.  The difference is the yaw damping the boat generates
    against its own turn.

    This bound used to be 2.5 deg/s, when the skeg weathervaning was the
    *only* source of that damping: the hull's own yaw moment was set to
    zero outright.  Distributed cross-flow drag now charges each station
    the drag of its local lateral velocity ``v + x r``, so the hull damps
    its own rotation too, and the naive estimate falls to about 1.5 --
    much closer to the measured figure, which is the point.

    See ``coxswain.hydro.crossflow`` and SOURCES sec. 36.
    """
    hydro = HydroCoefficients.from_boat(boat)
    speed = 5.2
    naive = abs(hydro.yaw_from_rudder * speed ** 2 * np.radians(12.0)
                / (hydro.yaw_from_yaw * speed))
    assert 1.2 < np.degrees(naive) < 2.2
    # and it must still exceed the measured turn rate, or there would be
    # nothing for the weathervane to explain
    assert np.degrees(naive) > 1.1


# --------------------------------------------------------------------------
# the assembled dynamics
# --------------------------------------------------------------------------
def test_dynamics_builds_a_callable_function(model):
    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 0.0, 176000.0])
    value = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()
    assert value.shape == (9,)
    assert np.all(np.isfinite(value))


def test_a_sweep_eight_does_not_go_straight(model, boat):
    """A sweep rig is asymmetric, so it yaws with no rudder and no split.

    This test previously asserted the opposite, which was wrong.  Port and
    starboard oarlocks sit at different stations along the hull, so their
    moments do not cancel: -467 N m mid-drive for an eight.  That is why a
    sweep boat wanders and needs continuous correction, and a model that
    tracks perfectly straight has left the rig geometry out.
    """
    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 0.0, 176000.0])
    mid_drive = boat.timing.period * 0.15
    value = np.array(function(state, [0.0, 0.0, 1.0], mid_drive)).ravel()

    assert value[2] == pytest.approx(0.0, abs=1e-12), "psi_dot is r, still 0"
    assert abs(value[6]) > 1e-3, "a sweep rig must yaw"
    assert abs(float(model.aggregates.yaw_neutral(mid_drive))) > 100.0


def test_a_sculling_boat_tracks_straight(model):
    """The control: with a symmetric rig the moments do cancel."""
    single = catalog.single_scull(rate=32.0)
    sculling = StrokeResolvedModel(single)
    mid_drive = single.timing.period * 0.15
    assert float(sculling.aggregates.yaw_neutral(mid_drive)) == pytest.approx(
        0.0, abs=1.0)


def test_position_derivative_is_the_rotated_velocity(model):
    function = model.function()
    psi = 0.6
    state = np.array([0.0, 0.0, psi, 0.0, 5.0, 0.3, 0.0, 0.0, 176000.0])
    value = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()
    assert value[0] == pytest.approx(5.0 * np.cos(psi) - 0.3 * np.sin(psi))
    assert value[1] == pytest.approx(5.0 * np.sin(psi) + 0.3 * np.cos(psi))


def test_rudder_turns_the_boat(model):
    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 0.0, 176000.0])
    straight = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()
    turning = np.array(
        function(state, [np.radians(12.0), 0.0, 1.0], 0.2)).ravel()
    assert abs(turning[6]) > abs(straight[6])


def test_split_turns_the_boat_only_on_the_drive(model, boat):
    """The physical point of the whole module.

    A pressure split makes a yaw moment during the drive and none on the
    recovery, because the blades are out of the water.
    """
    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 0.0, 176000.0])
    drive = boat.timing.period * boat.timing.drive_fraction * 0.5
    recovery = boat.timing.period * (boat.timing.drive_fraction + 1.0) / 2.0

    on_drive = np.array(function(state, [0.0, 0.30, 1.0], drive)).ravel()
    on_recovery = np.array(function(state, [0.0, 0.30, 1.0], recovery)).ravel()
    assert abs(on_drive[6]) > 3.0 * abs(on_recovery[6])


@pytest.mark.slow
def test_surge_oscillation_matches_the_full_model(model, boat):
    """The reason for resolving the stroke at all.

    Integrating the stroke model reproduces the 6-DOF surge swing to
    better than 5%; the stroke-averaged model has no swing at all.
    """
    from coxswain.sim.simulator import RowingSimulator

    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 0.0, 176000.0])
    dt, steps = 0.004, 2000
    surge = np.empty(steps)
    t = 0.0
    for i in range(steps):
        surge[i] = state[4]
        step = np.array(function(state, [0.0, 0.0, 1.0], t)).ravel()
        state = state + dt * step
        t += dt
    tail = surge[int(0.6 * steps):]

    reference = RowingSimulator(boat).run(duration=8.0, dt=0.006,
                                          surge_speed=5.2)
    window = reference.last_cycles(2)
    expected = float(np.ptp(reference.surge_speed[window]))
    assert float(np.ptp(tail)) == pytest.approx(expected, rel=0.05)


# --------------------------------------------------------------------------
# the split is not a pure couple
# --------------------------------------------------------------------------
def test_split_moment_matches_the_full_oar_path(boat, model):
    """Regression guard on a hand-derived simplification that was wrong.

    The split moment was first written as ``-y * Fx``, which looks right.
    It is not: ``hull_load`` also carries the sweep-rotated components and
    the hand-position term.  Dropping them inverted the sign over the first
    third of the drive and got the stroke mean wrong by 1.6x.  It is now
    obtained by differencing the real oar load.
    """
    from coxswain.river.strokemodel import _oar_load

    fit = model.aggregates.yaw_per_split
    for fraction in (0.05, 0.15, 0.30, 0.55):
        t = fraction * boat.timing.period
        expected = _oar_load(boat, t, 1.0)[3] - _oar_load(boat, t, 0.0)[3]
        # to the fit's own reported truncation error, not an invented number
        assert float(fit(t)) == pytest.approx(expected,
                                              abs=fit.max_error * 1.5)


def test_fits_report_their_truncation_error(model):
    """A truncated series is an approximation; the error is part of it."""
    for name in ("first_moment", "yaw_inertia", "thrust", "yaw_per_split"):
        fit = getattr(model.aggregates, name)
        assert fit.relative_error <= 0.01, name
        assert fit.max_error > 0.0, name


def test_discontinuous_quantities_need_more_harmonics(model):
    """Gibbs, measured.

    The crew aggregates are smooth and 8 harmonics clear 0.05%.  The oar
    loads jump at the catch, so their coefficients decay as 1/k and the
    same tolerance costs more.  Fitting a fixed count and not checking
    is how a 4% error goes unnoticed.

    This cost 32 harmonics until the rowers' hands were put on the handle
    laterally.  With the hands pinned to ``y = 0`` the oar's lateral
    moment arm was a constant 0.85 m while the true arm swings from 0.71 m
    to 1.13 m, so the split yaw moment carried a spurious step at the catch
    on top of the real one.  Tracking the handle removed it and halved the
    harmonics needed.  See SOURCES section 13.
    """
    smooth = model.aggregates.yaw_inertia.n_harmonics
    discontinuous = model.aggregates.yaw_per_split.n_harmonics
    # The gap has closed.  Kleshnev's drive curve ramps in as
    # u**1.485 where the old half-sine was linear at the catch, so the
    # step in the oar loads is far weaker and the split yaw moment now
    # needs no more harmonics than the smooth aggregates.  Recorded
    # rather than removed: if this starts failing again, something has
    # put a discontinuity back.  See SOURCES sec. 38.
    assert discontinuous <= 2 * smooth


def test_fit_to_tolerance_raises_rather_than_under_delivering():
    """Silently returning a worse fit than asked for is the failure mode."""
    period = 1.0
    phase = np.arange(256) / 256
    square = np.where(phase < 0.4, 1.0, -1.0)   # a hard jump
    with pytest.raises(ValueError, match="could not reach"):
        StrokePeriodicFit.fit_to_tolerance(square, period,
                                           relative_tolerance=1e-6,
                                           max_harmonics=32)


def test_a_split_produces_a_side_force(boat, model):
    """A split is a pure couple in surge but NOT in sway.

    The x components cancel, so a split adds no net thrust -- that is by
    design and keeps it steering rather than acceleration.  The y
    components do not: the oar force is rotated by the sweep angle, so
    scaling port up while scaling starboard down leaves ``s * Fy`` behind.
    Measured at up to 383 N, against exactly zero with no split.
    """
    from coxswain.river.strokemodel import _oar_load

    peak = boat.timing.period * 0.15
    neutral = _oar_load(boat, peak, 0.0)
    split = _oar_load(boat, peak, 1.0)

    assert neutral[1] == pytest.approx(0.0, abs=1e-9), "symmetric crew"
    assert abs(split[1] - neutral[1]) > 100.0
    # and surge is unchanged, so the split cannot be used to accelerate
    assert split[0] == pytest.approx(neutral[0], rel=1e-9)


def test_sway_per_split_is_carried(model):
    assert model.aggregates.sway_per_split is not None
    assert abs(model.aggregates.sway_per_split.mean) > 1.0


# --------------------------------------------------------------------------
# roll -- inside the steering loop, not beside it
# --------------------------------------------------------------------------
def test_a_split_makes_a_roll_moment(boat, model):
    """The first link in the chain that makes a split steer.

    The oar's vertical force is mirrored across the boat, so scaling port
    up and starboard down leaves a couple about the hull x axis: exactly
    zero unsplit, up to 250 N m with a split.
    """
    from coxswain.river.strokemodel import _oar_load

    peak = boat.timing.period * 0.15
    assert _oar_load(boat, peak, 0.0)[4] == pytest.approx(0.0, abs=1e-9)
    assert abs(_oar_load(boat, peak, 1.0)[4]) > 100.0
    assert abs(model.aggregates.roll_per_split.mean) > 10.0


def test_bare_hull_is_roll_unstable(boat):
    """Correct for a racing shell, and why the crew must balance actively.

    Positive stiffness means a heel produces a moment that increases it.
    The crew's balance loop supplies about -6000 N m/rad against this.
    """
    hydro = HydroCoefficients.from_boat(boat)
    assert hydro.roll_from_roll > 0.0


def test_heel_pushes_the_boat_sideways(boat):
    """The coupling a planar model cannot have.

    A heeled hull has an asymmetric wetted surface and generates side
    force; measured at about 2200 N/rad.
    """
    hydro = HydroCoefficients.from_boat(boat)
    assert abs(hydro.sway_from_roll) > 500.0


def test_hull_roll_damping_is_present(boat):
    """This test used to assert the opposite, and said why.

    It read: "Recorded because it is a gap, not because it is right.  The
    6-DOF model has no hull roll damping at all, so every bit of it comes
    from the crew.  A real hull damps roll through the wetted surface and
    the blades."

    The gap is closed.  :mod:`coxswain.hydro.radiation` supplies it from
    Ikeda's component method -- lift, which dominates at racing speed and
    is linear in forward speed, plus Kato friction, which is what is left
    at rest.  The coefficient is derived rather than fitted, so the thing
    worth asserting is its **sign and existence**, not its value: roll
    rate must produce a moment opposing the roll.
    """
    hydro = HydroCoefficients.from_boat(boat)
    assert hydro.roll_from_roll_rate < 0.0


def test_roll_state_responds_to_a_split(model, boat):
    """End to end: a split must heel the boat in the model, not just in
    the 6-DOF it was fitted from."""
    function = model.function()
    state = np.array([0.0, 0.0, 0.0, 0.0, 5.2, 0.0, 0.0, 0.0, 176000.0])
    drive = boat.timing.period * 0.15

    level = np.array(function(state, [0.0, 0.0, 1.0], drive)).ravel()
    split = np.array(function(state, [0.0, 0.30, 1.0], drive)).ravel()
    assert abs(split[7]) > abs(level[7]), "roll acceleration must respond"


def test_crew_balance_saturates(model):
    """A crew cannot counter arbitrary heel.

    Without saturation the model would hold any angle, which is not what
    happens when a boat goes over.
    """
    function = model.function()
    small = np.array([0.0, 0.0, 0.0, np.radians(1.0), 5.2, 0.0, 0.0, 0.0,
                      176000.0])
    large = np.array([0.0, 0.0, 0.0, np.radians(25.0), 5.2, 0.0, 0.0, 0.0,
                      176000.0])
    small_accel = abs(float(np.array(
        function(small, [0.0, 0.0, 1.0], 0.2)).ravel()[7]))
    large_accel = abs(float(np.array(
        function(large, [0.0, 0.0, 1.0], 0.2)).ravel()[7]))
    # 25x the heel must not give 25x the restoring acceleration
    assert large_accel < 25.0 * small_accel
