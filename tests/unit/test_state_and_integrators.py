"""Unit tests for the state vector, the integrators and the controllers."""

import numpy as np
import pytest

from coxswain.core import integrators
from coxswain.core.frames import PITCH, ROLL, YAW, hull_to_abs
from coxswain.core.state import SLICES, STATE_SIZE, State, pack, unpack
from coxswain.sim.control import BalanceController, Coxswain, HeadingController


# ==========================================================================
# State
# ==========================================================================
def test_state_layout_is_twelve_elements():
    assert STATE_SIZE == 12
    assert SLICES["position"] == slice(0, 3)
    assert SLICES["attitude"] == slice(3, 6)
    assert SLICES["velocity"] == slice(6, 9)
    assert SLICES["omega"] == slice(9, 12)


def test_state_round_trips_through_a_vector():
    y = np.arange(12.0)
    np.testing.assert_allclose(State.from_vector(y).to_vector(), y)


def test_state_rejects_a_wrong_length_vector():
    with pytest.raises(ValueError, match="shape"):
        State.from_vector(np.zeros(11))


def test_named_attitude_accessors_use_the_documented_order():
    state = State.create(attitude=[0.1, 0.2, 0.3])
    assert state.roll == pytest.approx(0.1)
    assert state.pitch == pytest.approx(0.2)
    assert state.yaw == pytest.approx(0.3)
    assert state.attitude[ROLL] == pytest.approx(0.1)
    assert state.attitude[PITCH] == pytest.approx(0.2)
    assert state.attitude[YAW] == pytest.approx(0.3)


def test_velocity_hull_is_the_absolute_velocity_rotated_back():
    attitude = np.array([0.1, -0.2, 0.7])
    velocity = np.array([5.0, 0.3, -0.1])
    state = State.create(attitude=attitude, velocity=velocity)
    expected = hull_to_abs(attitude).T @ velocity
    np.testing.assert_allclose(state.velocity_hull, expected, atol=1e-12)


def test_surge_speed_equals_the_hull_x_velocity():
    state = State.create(attitude=[0.0, 0.0, np.pi / 2],
                         velocity=[0.0, 5.0, 0.0])
    assert state.surge_speed == pytest.approx(5.0, abs=1e-12)


def test_speed_is_frame_independent():
    velocity = np.array([4.0, 3.0, 0.0])
    plain = State.create(velocity=velocity)
    turned = State.create(attitude=[0.2, -0.1, 1.3], velocity=velocity)
    assert plain.speed == pytest.approx(turned.speed)
    assert plain.speed == pytest.approx(5.0)


def test_sideslip_is_zero_for_pure_forward_motion():
    assert State.create(velocity=[5.0, 0.0, 0.0]).sideslip == pytest.approx(0.0)


def test_sideslip_is_positive_for_motion_to_port():
    state = State.create(velocity=[5.0, 0.5, 0.0])
    assert state.sideslip > 0.0


def test_sideslip_is_zero_at_rest():
    assert State.create().sideslip == pytest.approx(0.0)


def test_omega_hull_matches_the_rotation():
    attitude = np.array([0.15, -0.25, 0.9])
    omega_body = np.array([0.1, -0.05, 0.4])
    state = State.create(attitude=attitude,
                         omega=hull_to_abs(attitude) @ omega_body)
    np.testing.assert_allclose(state.omega_hull, omega_body, atol=1e-12)


def test_replace_produces_a_modified_copy():
    state = State.create(velocity=[1.0, 0.0, 0.0])
    changed = state.replace(velocity=[2.0, 0.0, 0.0])
    assert state.velocity[0] == pytest.approx(1.0)
    assert changed.velocity[0] == pytest.approx(2.0)


def test_zeros_is_all_zero():
    np.testing.assert_allclose(State.zeros().to_vector(), np.zeros(12))


def test_pack_and_unpack_are_inverses():
    parts = (np.array([1.0, 2, 3]), np.array([0.1, 0.2, 0.3]),
             np.array([4.0, 5, 6]), np.array([0.4, 0.5, 0.6]))
    for original, recovered in zip(parts, unpack(pack(*parts))):
        np.testing.assert_allclose(recovered, original)


# ==========================================================================
# Integrators
# ==========================================================================
def test_rk4_solves_exponential_decay_accurately():
    times, states = integrators.rk4(lambda t, y: -2.0 * y, (0.0, 2.0),
                                    np.array([1.0]), 0.01)
    np.testing.assert_allclose(states[0], np.exp(-2.0 * times), rtol=1e-8)


def test_rk4_is_fourth_order():
    """Halving the step must cut the error by roughly sixteen."""
    def error(dt):
        _, states = integrators.rk4(lambda t, y: y, (0.0, 1.0),
                                    np.array([1.0]), dt)
        return abs(states[0, -1] - np.e)

    assert error(0.1) / error(0.05) == pytest.approx(16.0, rel=0.25)


def test_rk4_conserves_energy_of_a_harmonic_oscillator():
    def derivative(t, y):
        return np.array([y[1], -y[0]])

    _, states = integrators.rk4(derivative, (0.0, 50.0),
                                np.array([1.0, 0.0]), 0.01)
    energy = 0.5 * (states[0] ** 2 + states[1] ** 2)
    assert np.ptp(energy) < 1e-6


def test_rk4_does_not_mutate_the_initial_condition():
    """The legacy integrator wrote back into the caller's array."""
    y0 = np.array([1.0, 2.0, 3.0])
    original = y0.copy()
    integrators.rk4(lambda t, y: y, (0.0, 0.5), y0, 0.1)
    np.testing.assert_allclose(y0, original)


def test_rk4_starts_at_the_initial_condition():
    times, states = integrators.rk4(lambda t, y: y, (0.0, 1.0),
                                    np.array([3.0]), 0.1)
    assert times[0] == pytest.approx(0.0)
    assert states[0, 0] == pytest.approx(3.0)


def test_rk4_lands_exactly_on_the_end_time():
    times, _ = integrators.rk4(lambda t, y: y, (0.0, 1.0), np.array([1.0]),
                               0.03)
    assert times[-1] == pytest.approx(1.0)


def test_rk4_rejects_a_non_positive_step():
    with pytest.raises(ValueError, match="dt must be positive"):
        integrators.rk4(lambda t, y: y, (0.0, 1.0), np.array([1.0]), 0.0)


def test_rk4_is_deterministic():
    args = (lambda t, y: -y, (0.0, 1.0), np.array([1.0]), 0.01)
    _, first = integrators.rk4(*args)
    _, second = integrators.rk4(*args)
    np.testing.assert_array_equal(first, second)


def test_adaptive_matches_rk4_on_a_smooth_problem():
    def derivative(t, y):
        return np.array([y[1], -4.0 * y[0]])

    y0 = np.array([1.0, 0.0])
    grid = np.linspace(0.0, 5.0, 101)
    _, adaptive = integrators.adaptive(derivative, (0.0, 5.0), y0,
                                       t_eval=grid, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(adaptive[0], np.cos(2.0 * grid), atol=1e-6)


def test_estimate_step_resolves_the_faster_mode():
    assert integrators.estimate_step(2.0, heave_period=0.5) < 0.5 / 50
    assert integrators.estimate_step(0.3, heave_period=5.0) < 0.3 / 50


# ==========================================================================
# Controllers
# ==========================================================================
def test_balance_controller_opposes_roll():
    controller = BalanceController()
    assert controller.moment(0.05, 0.0) < 0.0
    assert controller.moment(-0.05, 0.0) > 0.0


def test_balance_controller_opposes_roll_rate():
    controller = BalanceController(stiffness=0.0, damping=1000.0)
    assert controller.moment(0.0, 0.1) < 0.0


def test_balance_controller_saturates():
    controller = BalanceController(max_moment=500.0)
    assert controller.moment(10.0, 0.0) == pytest.approx(-500.0)
    assert controller.moment(-10.0, 0.0) == pytest.approx(500.0)


def test_balance_controller_can_be_disabled():
    assert BalanceController(enabled=False).moment(0.3, 0.5) == 0.0


def test_balance_controller_is_zero_at_equilibrium():
    assert BalanceController().moment(0.0, 0.0) == pytest.approx(0.0)


def test_heading_controller_applies_rudder_towards_the_target():
    """Positive rudder yaws to starboard, so a port-side error needs +rudder."""
    controller = HeadingController(target=0.0)
    drifted_to_port = State.create(attitude=[0.0, 0.0, 0.1])
    assert controller.deflection(0.0, drifted_to_port) > 0.0


def test_heading_controller_reverses_for_the_other_side():
    controller = HeadingController(target=0.0)
    drifted = State.create(attitude=[0.0, 0.0, -0.1])
    assert controller.deflection(0.0, drifted) < 0.0


def test_heading_controller_is_neutral_on_target():
    controller = HeadingController(target=0.3)
    on_course = State.create(attitude=[0.0, 0.0, 0.3])
    assert controller.deflection(0.0, on_course) == pytest.approx(0.0)


def test_heading_controller_damps_yaw_rate():
    controller = HeadingController(target=0.0, gain=0.0, rate_gain=1.0)
    turning = State.create(omega=[0.0, 0.0, 0.2])
    assert controller.deflection(0.0, turning) > 0.0


def test_heading_controller_saturates_at_the_deflection_limit():
    controller = HeadingController(target=0.0, max_deflection=0.4)
    hard_over = State.create(attitude=[0.0, 0.0, 3.0])
    assert abs(controller.deflection(0.0, hard_over)) == pytest.approx(0.4)


def test_heading_controller_wraps_the_error():
    """A target just across the +-pi cut must not command a full turn."""
    controller = HeadingController(target=np.pi - 0.05)
    just_past = State.create(attitude=[0.0, 0.0, -np.pi + 0.05])
    assert abs(controller.deflection(0.0, just_past)) < 0.4


def test_heading_controller_accepts_a_time_varying_target():
    controller = HeadingController(target=lambda t: 0.1 * t)
    assert controller.target_heading(5.0) == pytest.approx(0.5)


def test_heading_controller_can_be_disabled():
    controller = HeadingController(target=0.0, enabled=False)
    assert controller.deflection(0.0, State.create(attitude=[0, 0, 1.0])) == 0.0


def test_coxswain_bundles_both_loops():
    coxswain = Coxswain()
    state = State.create(attitude=[0.05, 0.0, 0.1])
    assert coxswain.roll_moment(state) < 0.0
    assert coxswain.rudder(0.0, state) > 0.0


def test_coxswain_override_takes_precedence():
    coxswain = Coxswain(rudder_override=lambda t, s: 0.123)
    assert coxswain.rudder(0.0, State.zeros()) == pytest.approx(0.123)
