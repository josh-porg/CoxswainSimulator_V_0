"""Non-dimensionalisation of the trajectory NLP.

This should have been in place before the first solve.  Without it the
transcription spanned six orders of magnitude and IPOPT hit its iteration
cap; with it the same problem converges in half the iterations and half
the time.  These tests exist so it cannot quietly come undone.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.river.scaling import ProblemScaling


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("casadi")
    from coxswain.river.hullsurrogate import HullSurrogate
    from coxswain.river.sixdof import SixDofModel

    boat = catalog.eight(rate=32.0)
    return SixDofModel(boat, surrogate=HullSurrogate.from_boat(
        boat, n_heave=13, n_pitch=7, n_roll=9))


@pytest.fixture
def sample(model):
    state = np.zeros(model.n_states)
    state[0], state[1] = 15.0, 8.0
    state[2] = 0.014
    state[3] = np.radians(1.0)
    state[5] = 1.2
    state[6], state[7] = 4.2, 1.8
    state[9] = 0.3
    state[12] = model.anaerobic_capacity
    control = np.array([np.radians(5.0), 0.05, 1.0])
    return state, control


# --------------------------------------------------------------------------
# the problem really is badly scaled without help
# --------------------------------------------------------------------------
def test_the_physical_problem_spans_many_orders_of_magnitude(sample):
    """The motivation, measured rather than asserted.

    Position in metres against roll in radians against the crew's reserve
    in joules.  An interior-point method inherits this conditioning
    directly.
    """
    state, control = sample
    unscaled = ProblemScaling.unscaled(len(state), len(control))
    assert unscaled.scaled_spread(state, control) > 1e5


def test_scaling_brings_everything_to_order_one(model, sample):
    state, control = sample
    scaling = ProblemScaling.for_six_dof(model, leg_length=20.0, speed=4.6)
    assert scaling.scaled_spread(state, control) < 100.0

    scaled = scaling.to_scaled_state(state)
    assert np.abs(scaled).max() <= 1.5


def test_spread_reports_the_unscaled_problem(model):
    """Named carefully: it measures what is being divided out, so it stays
    large.  Confusing it with the post-scaling spread would make the
    scaling look like it had failed."""
    scaling = ProblemScaling.for_six_dof(model, leg_length=20.0, speed=4.6)
    assert scaling.spread > 1e4


# --------------------------------------------------------------------------
# it is a change of units, not of model
# --------------------------------------------------------------------------
def test_scaling_round_trips(model, sample):
    state, control = sample
    scaling = ProblemScaling.for_six_dof(model, leg_length=20.0, speed=4.6)
    np.testing.assert_allclose(
        scaling.from_scaled_state(scaling.to_scaled_state(state)), state,
        rtol=1e-12)
    np.testing.assert_allclose(
        scaling.from_scaled_control(scaling.to_scaled_control(control)),
        control, rtol=1e-12)


def test_unscaled_scaling_is_the_identity(sample):
    state, control = sample
    scaling = ProblemScaling.unscaled(len(state), len(control))
    np.testing.assert_allclose(scaling.to_scaled_state(state), state)
    np.testing.assert_allclose(scaling.to_scaled_control(control), control)


def test_scaled_dynamics_describe_the_same_trajectory(model, sample):
    """The physics must be untouched.

    ``d(x/s)/dt = f(x, u)/s`` -- so unscaling the scaled derivative has to
    reproduce the original one exactly, not approximately.
    """
    import casadi as ca

    state, control = sample
    scaling = ProblemScaling.for_six_dof(model, leg_length=20.0, speed=4.6)
    raw = model.function()
    scaled = scaling.scaled_dynamics(raw, ca)

    physical = np.array(raw(state, control, 0.3)).ravel()
    through = np.array(scaled(ca.DM(scaling.to_scaled_state(state)),
                              ca.DM(scaling.to_scaled_control(control)),
                              0.3)).ravel()
    np.testing.assert_allclose(through * scaling.state, physical, rtol=1e-9)


def test_position_scale_follows_the_leg_not_the_origin(model):
    """Scaling by the absolute coordinate would scale by wherever the
    tangent-plane origin happens to sit, which is arbitrary and can be
    kilometres."""
    short = ProblemScaling.for_six_dof(model, leg_length=20.0)
    long = ProblemScaling.for_six_dof(model, leg_length=800.0)
    assert long.state[0] > short.state[0]
    np.testing.assert_allclose(short.state[6:9], long.state[6:9])


def test_every_scale_is_strictly_positive(model):
    """A zero scale would divide by zero; a negative one would silently
    flip the sign of a bound."""
    scaling = ProblemScaling.for_six_dof(model, leg_length=20.0)
    assert np.all(scaling.state > 0.0)
    assert np.all(scaling.control > 0.0)


def test_attitude_scales_come_from_the_surrogate_range(model):
    """So the optimiser's O(1) box is the range the hull was actually
    sampled over, which is also where the bounds are."""
    scaling = ProblemScaling.for_six_dof(model, leg_length=20.0)
    assert scaling.state[3] == pytest.approx(
        float(np.abs(model.surrogate.roll).max()))
