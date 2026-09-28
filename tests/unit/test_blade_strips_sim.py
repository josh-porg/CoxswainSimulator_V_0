"""Strip integration on the dynamic oar: off by default, the centre-point law in the limit."""
import numpy as np
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.core.state import STATE_SIZE, State
from coxswain.sim.dynamic_oar import DynamicOarSimulator, _match_key


@pytest.fixture(scope="module")
def single():
    return physics.resolve("research").apply(catalog.single_scull(rate=32.0))


def _mid_drive(sim, surge=4.6, angle=np.radians(5.0), rate=-2.8):
    n = sim.n_oar_states
    y = sim.augmented_initial_state(surge)
    y[STATE_SIZE:STATE_SIZE + n] = angle
    y[STATE_SIZE + n:STATE_SIZE + 2 * n] = rate
    return y


def test_off_by_default(single):
    assert DynamicOarSimulator(single, peak_torque=100.0).blade_span is None


def test_a_bad_span_is_refused(single):
    with pytest.raises(ValueError, match="blade_span"):
        DynamicOarSimulator(single, peak_torque=100.0, blade_span=0.0)


@pytest.mark.parametrize("law", ["slip", "liftdrag"])
def test_a_vanishing_span_is_the_centre_point_law(single, law):
    base = DynamicOarSimulator(single, peak_torque=100.0, blade_law=law)
    thin = DynamicOarSimulator(single, peak_torque=100.0, blade_law=law, blade_span=1e-5)
    y = _mid_drive(base)
    np.testing.assert_allclose(thin.derivative(0.1, y), base.derivative(0.1, y),
                               rtol=1e-6, atol=1e-8)


def test_a_real_span_loads_further_out_mid_drive(single):
    sim = DynamicOarSimulator(single, peak_torque=100.0, blade_span=0.43)
    oar = sim._oars[0]
    assert float(oar.blade.outboard) == pytest.approx(float(oar.outboard))
    y = _mid_drive(sim)
    state = State.from_vector(y[:STATE_SIZE])
    lock = single.rig.seats[0].oarlocks[0]
    f_n, f_t, arm = sim._blade_loads_arm(0, np.radians(5.0), -2.8, state, lock)
    centre = DynamicOarSimulator(single, peak_torque=100.0)._blade_loads_arm(
        0, np.radians(5.0), -2.8, state, lock)
    assert f_t == 0.0
    assert arm > float(oar.outboard)
    assert f_n > centre[0] > 0.0


def test_matched_torques_do_not_share_a_cache_entry_across_spans(single):
    assert (_match_key(single, 300.0, "sweep", "slip")
            == _match_key(single, 300.0, "sweep", "slip", "none", None))
    assert (_match_key(single, 300.0, "sweep", "slip")
            != _match_key(single, 300.0, "sweep", "slip", "none", 0.43))
