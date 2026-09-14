r"""Blade added mass on the dynamic oar, solved together with the hull.

Checked against its own definition (Patton's AR-2 plate), against the default
derivative it must reduce to, and against the structure a combined mass
matrix must have.  Off by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.core.state import STATE_SIZE, State
from coxswain.crew.blade_added_mass import (BIG_BLADE_WIDTH, PATTON_AR2,
                                            patton_added_mass)
from coxswain.sim.dynamic_oar import DynamicOarSimulator


@pytest.fixture(scope="module")
def eight():
    from coxswain.boats import catalog

    return catalog.eight(rate=32.0)


def _driving(sim, surge=4.85, angle=np.radians(10.0), rate=-2.5):
    n = sim.n_oar_states
    y = sim.augmented_initial_state(surge)
    y[STATE_SIZE:STATE_SIZE + n] = angle
    y[STATE_SIZE + n:STATE_SIZE + 2 * n] = rate
    return y


def test_pattons_added_mass_for_a_big_blade():
    sweep = patton_added_mass(0.52, BIG_BLADE_WIDTH["sweep"], 999.1)
    scull = patton_added_mass(0.43, BIG_BLADE_WIDTH["scull"], 999.1)
    assert sweep == pytest.approx(PATTON_AR2 * np.pi * 999.1 / 4 * 0.52 * 0.25 ** 2)
    assert sweep == pytest.approx(21.4, abs=0.1)
    assert scull == pytest.approx(13.2, abs=0.1)


def test_pattons_coefficient_is_refused_off_aspect_ratio_two():
    with pytest.raises(ValueError, match="aspect-ratio-2"):
        patton_added_mass(0.52, 0.10, 999.1)


def test_added_mass_is_off_by_default_and_refused_where_it_is_not_solved(eight):
    assert DynamicOarSimulator(eight, peak_torque=600.0).blade_added_mass == "none"
    with pytest.raises(ValueError, match="blade added mass"):
        DynamicOarSimulator(eight, peak_torque=600.0, blade_added_mass="grift")
    with pytest.raises(ValueError, match="clock crew"):
        DynamicOarSimulator(eight, peak_torque=600.0, blade_added_mass="patton",
                            crew="follows")
    with pytest.raises(ValueError, match="clock crew"):
        DynamicOarSimulator(eight, peak_torque=600.0, blade_added_mass="patton",
                            release="slip")


def test_with_no_added_mass_the_coupled_solve_is_the_default_derivative(eight):
    """The assembly, checked by switching its one new ingredient off."""
    default = DynamicOarSimulator(eight, peak_torque=600.0)
    coupled = DynamicOarSimulator(eight, peak_torque=600.0,
                                  blade_added_mass="patton")
    coupled._blade_mass = [0.0] * coupled.n_oar_states
    y = _driving(default)
    np.testing.assert_allclose(coupled.derivative(0.1, y),
                               default.derivative(0.1, y),
                               rtol=1e-9, atol=1e-9)


def test_the_combined_mass_matrix_is_symmetric_positive_definite(eight):
    sim = DynamicOarSimulator(eight, peak_torque=600.0, blade_added_mass="patton")
    y = _driving(sim)
    n = sim.n_oar_states
    state = State.from_vector(y[:STATE_SIZE])
    system, _rhs = sim._coupled_system(0.1, state, y[STATE_SIZE:STATE_SIZE + n],
                                       y[STATE_SIZE + n:STATE_SIZE + 2 * n])
    assert system.shape == (6 + n, 6 + n)
    np.testing.assert_allclose(system, system.T, atol=1e-9)
    np.linalg.cholesky(system)                     # raises if not SPD

    # the oar diagonal: the seat's own inertia plus the blade's m l^2
    oar = sim._oars[0]
    lock = eight.rig.seats[sim._seats[0]].oarlocks[0]
    inertia, _balance = sim._seat_balance(
        oar, np.radians(10.0), -2.5, sim._torque(0, np.radians(10.0), state, 0.1),
        state, [lock], 0)
    expected = inertia + sim._blade_mass[0] * oar.outboard ** 2
    assert system[6, 6] == pytest.approx(expected, rel=1e-12)
    assert sim._blade_mass[0] == pytest.approx(21.4, abs=0.2)


def test_a_held_oar_does_not_move(eight):
    sim = DynamicOarSimulator(eight, peak_torque=600.0, blade_added_mass="patton")
    n = sim.n_oar_states
    y = _driving(sim, angle=sim._oars[0].finish_angle - 0.01)
    rates = sim.derivative(0.1, y)[STATE_SIZE + n:STATE_SIZE + 2 * n]
    assert np.all(rates == 0.0)


def _in_air(sim, tau=0.05, surge=5.0):
    """A state with every oar on the sweep, blade out, ``tau`` into the stroke."""
    n = sim.n_oar_states
    angle, rate = sim._sweep_pose(tau)
    y = sim.augmented_initial_state(surge)
    y[STATE_SIZE:STATE_SIZE + n] = angle
    y[STATE_SIZE + n:STATE_SIZE + 2 * n] = rate
    sim._in_air = np.ones(n, dtype=bool)
    sim._stroke_start = 0.0
    return y


def test_on_the_sweep_catch_an_oar_in_the_air_follows_the_sweep(eight):
    """No balance and no added mass before entry: an identity row carrying the
    sweep's own acceleration, and nothing added to the hull block."""
    sim = DynamicOarSimulator(eight, peak_torque=600.0, catch="sweep",
                              blade_added_mass="patton")
    tau = 0.05
    y = _in_air(sim, tau)
    n = sim.n_oar_states
    state = State.from_vector(y[:STATE_SIZE])
    system, rhs = sim._coupled_system(tau, state, y[STATE_SIZE:STATE_SIZE + n],
                                      y[STATE_SIZE + n:STATE_SIZE + 2 * n])
    np.testing.assert_array_equal(system[6:, 6:], np.eye(n))
    np.testing.assert_array_equal(system[:6, 6:], 0.0)
    expected = sim._sweep_motion(tau)[1]
    np.testing.assert_allclose(rhs[6:], expected, rtol=0, atol=1e-12)
    rates = sim.derivative(tau, y)[STATE_SIZE + n:STATE_SIZE + 2 * n]
    np.testing.assert_allclose(rates, expected, rtol=1e-12)


def test_on_the_sweep_catch_the_coupled_solve_reduces_to_the_default(eight):
    """With the added mass zeroed, the coupled solve is the sweep catch's own
    derivative, in the air and in the water."""
    default = DynamicOarSimulator(eight, peak_torque=600.0, catch="sweep")
    coupled = DynamicOarSimulator(eight, peak_torque=600.0, catch="sweep",
                                  blade_added_mass="patton")
    coupled._blade_mass = [0.0] * coupled.n_oar_states
    y = _in_air(default)
    _in_air(coupled)
    np.testing.assert_allclose(coupled.derivative(0.05, y),
                               default.derivative(0.05, y),
                               rtol=1e-9, atol=1e-9)
    n = default.n_oar_states
    default._in_air = np.zeros(n, dtype=bool)
    coupled._in_air = np.zeros(n, dtype=bool)
    y = _driving(default)
    np.testing.assert_allclose(coupled.derivative(0.1, y),
                               default.derivative(0.1, y),
                               rtol=1e-9, atol=1e-9)


def test_added_mass_changes_the_oar_mid_drive(eight):
    """Present, not a rounding error: the blade's water is a large inertia."""
    default = DynamicOarSimulator(eight, peak_torque=600.0)
    coupled = DynamicOarSimulator(eight, peak_torque=600.0,
                                  blade_added_mass="patton")
    y = _driving(default)
    n = default.n_oar_states
    without = default.derivative(0.1, y)[STATE_SIZE + n]
    with_mass = coupled.derivative(0.1, y)[STATE_SIZE + n]
    assert abs(with_mass - without) > 0.05 * abs(without), (without, with_mass)
