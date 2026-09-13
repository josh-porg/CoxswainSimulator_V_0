r"""[CR06]'s release rule on the dynamic oar: the blade never brakes.

[CR06] takes the blade out when its normal velocity returns to zero, where its
force is exactly zero.  The dynamic oar's default keeps it in until the finish
angle, and the tier 1 law ``-sign(slip) C2 slip^2`` brakes whenever the slip
turns non-driving before then.  ``release="slip"`` is the study; the default
is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.core.state import STATE_SIZE, State
from coxswain.sim.dynamic_oar import DynamicOarSimulator


@pytest.fixture(scope="module")
def eight():
    from coxswain.boats import catalog

    return catalog.eight(rate=32.0)


def _state(sim, surge, angle, rate):
    n = sim.n_oar_states
    y = sim.augmented_initial_state(surge)
    y[STATE_SIZE:STATE_SIZE + n] = angle
    y[STATE_SIZE + n:STATE_SIZE + 2 * n] = rate
    return y


#: Late drive, the oar slowed, the boat fast: slip = l phi_dot + v cos(phi) > 0,
#: which the tier 1 law turns into a braking load.
BRAKING = dict(surge=5.0, angle=np.radians(-30.0), rate=-0.5)
#: Mid-drive, the blade driving: slip < 0.
DRIVING = dict(surge=4.0, angle=np.radians(10.0), rate=-2.0)


def test_the_default_release_is_the_finish_angle(eight):
    assert DynamicOarSimulator(eight, peak_torque=600.0).release == "angle"


def test_an_unknown_release_rule_is_refused(eight):
    with pytest.raises(ValueError, match="release rule"):
        DynamicOarSimulator(eight, peak_torque=600.0, release="whenever")


@pytest.mark.parametrize("law", ["slip", "liftdrag"])
def test_the_states_are_what_they_claim(eight, law):
    """Guard the fixtures: the braking state really brakes under the default."""
    sim = DynamicOarSimulator(eight, peak_torque=600.0, blade_law=law)
    lock = eight.rig.seats[sim._seats[0]].oarlocks[0]
    for case, driving in ((BRAKING, False), (DRIVING, True)):
        y = _state(sim, **case)
        state = State.from_vector(y[:STATE_SIZE])
        f_n, _f_t = sim._blade_loads(0, case["angle"], case["rate"], state, lock)
        speed = sim._lock_speed_on_normal(state, lock, case["angle"])
        slip = float(sim._oars[0].blade.slip_velocity(case["angle"], case["rate"],
                                                     speed))
        assert (slip < 0.0) == driving, (case, slip)
        assert f_n != 0.0          # the default loads the blade either way


@pytest.mark.parametrize("law", ["slip", "liftdrag"])
def test_a_released_blade_puts_nothing_on_the_hull(eight, law):
    sim = DynamicOarSimulator(eight, peak_torque=600.0, blade_law=law,
                              release="slip")
    y = _state(sim, **BRAKING)
    n = sim.n_oar_states
    sim._oar_state = (y[STATE_SIZE:STATE_SIZE + n],
                      y[STATE_SIZE + n:STATE_SIZE + 2 * n])
    force, moment = sim._oar_loads(0.1, State.from_vector(y[:STATE_SIZE]))
    sim._oar_state = None
    assert np.all(force == 0.0) and np.all(moment == 0.0)


@pytest.mark.parametrize("law", ["slip", "liftdrag"])
def test_a_released_blade_does_not_turn_the_oar(eight, law):
    """``phi_ddot = (-tau - I' phi_dot^2 / 2) / I`` -- no blade term."""
    sim = DynamicOarSimulator(eight, peak_torque=600.0, blade_law=law,
                              release="slip")
    y = _state(sim, **BRAKING)
    n = sim.n_oar_states
    got = sim.derivative(0.1, y)[STATE_SIZE + n]
    state = State.from_vector(y[:STATE_SIZE])
    oar = sim._oars[0]
    torque = sim._torque(0, BRAKING["angle"], state, 0.1)
    moment, slope = oar.inertia_at(BRAKING["angle"])
    expected = (-torque - 0.5 * slope * BRAKING["rate"] ** 2) / moment
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("law", ["slip", "liftdrag"])
def test_a_driving_blade_is_loaded_exactly_as_before(eight, law):
    angle_rule = DynamicOarSimulator(eight, peak_torque=600.0, blade_law=law)
    slip_rule = DynamicOarSimulator(eight, peak_torque=600.0, blade_law=law,
                                    release="slip")
    y = _state(angle_rule, **DRIVING)
    n = angle_rule.n_oar_states
    oar_state = (y[STATE_SIZE:STATE_SIZE + n], y[STATE_SIZE + n:STATE_SIZE + 2 * n])
    state = State.from_vector(y[:STATE_SIZE])
    loads = []
    for sim in (angle_rule, slip_rule):
        sim._oar_state = oar_state
        loads.append(sim._oar_loads(0.1, state))
        sim._oar_state = None
    np.testing.assert_allclose(loads[1][0], loads[0][0], rtol=1e-13, atol=1e-9)
    np.testing.assert_allclose(loads[1][1], loads[0][1], rtol=1e-13, atol=1e-9)
    assert slip_rule.derivative(0.1, y)[STATE_SIZE + n] == pytest.approx(
        angle_rule.derivative(0.1, y)[STATE_SIZE + n], rel=1e-12)


@pytest.mark.slow
def test_under_the_release_rule_nothing_starts_the_oar_at_the_catch():
    """Pins a finding, not a feature.

    The oar is reset to rest at the catch and the pull shape is zero there.
    Under the default rule the drive is started by the WATER: the boat carries
    the stationary blade, and the non-driving slip loads it towards the finish
    (-829 N on the eight at 380 W, with the handle torque at 0.0).  [CR06]'s
    rule removes that load, and nothing is left to move the oar -- it sits at
    the catch for the whole stroke.  A real oar enters the drive already
    moving; handing it in that way needs the rower (phase 4.3).
    """
    from coxswain.boats import catalog
    from coxswain.sim.oarloop import torque_shape

    boat = catalog.build("8+", rate=28.0)
    assert torque_shape(boat)(float(boat.oar_sweep.catch_angle)) == 0.0
    torque = DynamicOarSimulator.peak_torque_for_power(boat, 380.0)

    default = DynamicOarSimulator(boat, peak_torque=torque)
    released = DynamicOarSimulator(boat, peak_torque=torque, release="slip")
    n = default.n_oar_states
    lock = boat.rig.seats[default._seats[0]].oarlocks[0]
    catch = float(boat.oar_sweep.catch_angle)

    y = default.augmented_initial_state(5.6)
    state = State.from_vector(y[:STATE_SIZE])
    assert default._torque(0, catch, state, 0.0) == 0.0
    f_catch, _f_t = default._blade_loads(0, catch, 0.0, state, lock)
    assert f_catch < -100.0       # the water drives the parked blade

    run = released.run_strokes(1, surge_speed=5.6)
    angles = run.last_states[STATE_SIZE]
    rates = run.last_states[STATE_SIZE + n]
    assert np.all(angles == catch) and np.all(rates == 0.0)
    assert not run.strokes[0].finished
