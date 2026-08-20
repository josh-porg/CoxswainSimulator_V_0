"""Unit tests for the full six-degree-of-freedom symbolic model."""

import numpy as np
import pytest
from scipy.optimize import brentq

from coxswain.boats import catalog
from coxswain.river.hullsurrogate import TABULATED, HullSurrogate
from coxswain.river.sixdof import CrewTensorFit, SixDofModel

casadi = pytest.importorskip("casadi")


@pytest.fixture(scope="module")
def boat():
    return catalog.eight(rate=32.0)


@pytest.fixture(scope="module")
def surrogate(boat):
    return HullSurrogate.from_boat(boat, n_heave=17, n_pitch=9, n_roll=9)


@pytest.fixture(scope="module")
def model(boat, surrogate):
    return SixDofModel(boat, surrogate=surrogate)


# --------------------------------------------------------------------------
# the hull surrogate -- a bounded smoothing, so the bound must be measured
# --------------------------------------------------------------------------
def test_surrogate_reproduces_the_exact_mesh_off_node(surrogate):
    """On-node agreement is meaningless: a spline is exact at its knots.

    This samples at random interior points instead.
    """
    worst = surrogate.validate(n_samples=40)
    assert max(worst.values()) < 0.05, worst


def test_surrogate_covers_every_quantity_the_dynamics_needs(surrogate):
    values = surrogate(0.0, 0.0, 0.0)
    for name in TABULATED:
        assert name in values
        assert np.isfinite(values[name])


def test_surrogate_volume_decreases_as_the_boat_rises(surrogate):
    assert surrogate(-0.10, 0.0, 0.0)["volume"] > \
        surrogate(0.02, 0.0, 0.0)["volume"]


def test_heel_moves_the_centre_of_buoyancy_sideways(surrogate):
    """The restoring lever: without it there is no roll hydrostatics."""
    port = surrogate(-0.02, 0.0, np.radians(5.0))["buoyancy_y"]
    level = surrogate(-0.02, 0.0, 0.0)["buoyancy_y"]
    starboard = surrogate(-0.02, 0.0, np.radians(-5.0))["buoyancy_y"]
    assert level == pytest.approx(0.0, abs=1e-3)
    assert port * starboard < 0.0, "must be antisymmetric in heel"


def test_pitch_moves_the_centre_of_buoyancy_fore_and_aft(surrogate):
    bow_down = surrogate(-0.02, np.radians(2.0), 0.0)["buoyancy_x"]
    level = surrogate(-0.02, 0.0, 0.0)["buoyancy_x"]
    assert abs(bow_down - level) > 1e-4


def test_surrogate_is_differentiable(surrogate):
    import casadi as ca

    z, theta, phi = ca.MX.sym("z"), ca.MX.sym("t"), ca.MX.sym("p")
    values = surrogate.casadi(z, theta, phi)
    stacked = ca.vertcat(*[values[name] for name in TABULATED])
    jacobian = ca.Function("J", [z, theta, phi],
                           [ca.jacobian(stacked, ca.vertcat(z, theta, phi))])
    value = np.array(jacobian(-0.02, 0.01, 0.02))
    assert np.all(np.isfinite(value))
    assert np.abs(value).max() > 0.0


# --------------------------------------------------------------------------
# the crew tensor
# --------------------------------------------------------------------------
def test_crew_tensor_matches_the_numpy_crew_field(boat):
    import casadi as ca

    crew = CrewTensorFit.from_boat(boat, n_samples=256)
    for t in np.linspace(0.0, boat.timing.period, 9, endpoint=False):
        mass, position, _, acceleration = boat.crew_field(t)
        expected = (mass[:, None] * position).sum(axis=0)
        got = np.array(ca.DM(crew.moment_hull(t, ca))).ravel()
        np.testing.assert_allclose(got, expected, atol=0.5)

        cross = (mass[:, None] * np.cross(position, acceleration)).sum(axis=0)
        got_cross = np.array(ca.DM(crew.cross_accel_hull(t, ca))).ravel()
        np.testing.assert_allclose(got_cross, cross, atol=25.0)


def test_crew_cross_acceleration_is_not_the_product_of_first_moments(boat):
    """The term that was missing, and why it is easy to miss.

    ``sum m (r x a)`` is not ``(sum m r) x (sum m a)``: the second
    multiplies by mass twice.  For yaw it cancels for a symmetric crew;
    for pitch it does not, because the crew sit above the centre of mass.
    """
    mass, position, _, acceleration = boat.crew_field(0.3)
    true_cross = (mass[:, None] * np.cross(position, acceleration)).sum(axis=0)
    naive = np.cross((mass[:, None] * position).sum(axis=0),
                     (mass[:, None] * acceleration).sum(axis=0))
    assert abs(true_cross[1]) > 100.0, "pitch component must be substantial"
    assert not np.allclose(true_cross, naive, rtol=0.1)


def test_a_sweep_crew_yaws_itself(boat):
    """A sweep crew is not port-starboard symmetric, so this does not cancel.

    The planar model assumed it did, on the reasoning that ``sum m y xddot``
    vanishes for a mirrored crew.  That reasoning fails here: with both
    hands constrained to one handle, a sweep rower's two arms follow
    genuinely different paths, so their accelerations do not mirror.  The
    yaw component measures 69 N m against 103 for pitch -- the same order,
    not negligible.
    """
    mass, position, _, acceleration = boat.crew_field(0.3)
    cross = (mass[:, None] * np.cross(position, acceleration)).sum(axis=0)
    assert abs(cross[2]) > 10.0


def test_a_sculling_crew_does_not_yaw_itself():
    """The control: a sculler's arms are mirrored, so it does cancel."""
    scull = catalog.single_scull(rate=32.0)
    mass, position, _, acceleration = scull.crew_field(0.3)
    cross = (mass[:, None] * np.cross(position, acceleration)).sum(axis=0)
    assert abs(cross[2]) < 1e-6 * max(abs(cross[1]), 1.0)


# --------------------------------------------------------------------------
# the assembled model
# --------------------------------------------------------------------------
def test_model_produces_thirteen_finite_derivatives(model):
    function = model.function()
    state = np.zeros(13)
    state[6] = 5.2
    state[12] = model.anaerobic_capacity
    value = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()
    assert value.shape == (13,)
    assert np.all(np.isfinite(value))


def test_position_derivative_is_the_absolute_velocity(model):
    function = model.function()
    state = np.zeros(13)
    state[6:9] = [5.0, 0.3, -0.1]
    state[12] = model.anaerobic_capacity
    value = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()
    np.testing.assert_allclose(value[0:3], [5.0, 0.3, -0.1], atol=1e-12)


def test_a_sculling_rig_produces_no_lateral_forcing():
    """Control test for the sweep asymmetry.

    A symmetric rig must give exactly zero sway, roll and yaw acceleration;
    a sweep rig must not.  This is what shows the eight's asymmetry is the
    rig and not a bug.
    """
    scull = catalog.single_scull(rate=32.0)
    sculling = SixDofModel(
        scull, surrogate=HullSurrogate.from_boat(scull, n_heave=13,
                                                 n_pitch=7, n_roll=7))
    function = sculling.function()
    state = np.zeros(13)
    state[6] = 4.5
    state[12] = sculling.anaerobic_capacity
    value = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()

    assert value[7] == pytest.approx(0.0, abs=1e-9), "sway"
    assert value[9] == pytest.approx(0.0, abs=1e-9), "roll rate"
    assert value[11] == pytest.approx(0.0, abs=1e-9), "yaw rate"


def test_a_sweep_rig_does_produce_lateral_forcing(model):
    function = model.function()
    state = np.zeros(13)
    state[6] = 5.2
    state[12] = model.anaerobic_capacity
    value = np.array(function(state, [0.0, 0.0, 1.0], 0.2)).ravel()
    assert abs(value[11]) > 1e-3, "a sweep rig must yaw"


def test_buoyancy_balances_weight_at_the_float_point(model, boat):
    weight = boat.total_mass * 9.80665
    float_z = brentq(
        lambda z: boat.water.density * 9.80665
        * model.surrogate(z, 0.0, 0.0)["volume"] - weight, -0.16, 0.06)
    assert -0.16 < float_z < 0.06
    lift = (boat.water.density * 9.80665
            * model.surrogate(float_z, 0.0, 0.0)["volume"])
    assert lift == pytest.approx(weight, rel=1e-6)


def test_model_is_differentiable_in_state_and_control(model):
    """The reason for all of it: IPOPT needs exact derivatives."""
    import casadi as ca

    state = ca.MX.sym("state", 13)
    control = ca.MX.sym("control", 3)
    derivative = model.derivative(state, control, 0.2)
    jacobian = ca.Function("J", [state, control],
                           [ca.jacobian(derivative,
                                        ca.vertcat(state, control))])
    point = np.zeros(13)
    point[6] = 5.2
    point[12] = model.anaerobic_capacity
    value = np.array(jacobian(point, [0.05, 0.1, 1.0]))
    assert np.all(np.isfinite(value))
    assert np.abs(value).max() > 0.0
