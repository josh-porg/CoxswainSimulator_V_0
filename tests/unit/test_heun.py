r"""Heun at 60 Hz is the same boat as RK4 at 100 Hz, and half the cost.

The trainer's physics is the derivative, evaluated; RK4 evaluates it
four times a step, Heun twice.  These hold the low tiers' choice to the
reference over a race-length stretch of the eight the trainer actually
runs -- per-seat timing scatter, tables on -- and pin which tiers make
it, so a later tier edit cannot quietly hand the reference method to a
laptop or the cheap one to the studies.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.core.integrators import heun_step, rk4_step
from coxswain.sim.control import Coxswain
from coxswain.sim.simulator import RowingSimulator


def _run(scheme: str, rate: float, seconds: float = 24.0):
    boat = catalog.eight(rate=32.0)
    boat.tabulate_crew = True
    boat.phase_offsets = np.linspace(-0.02, 0.02, boat.n_seats)
    sim = RowingSimulator(boat, coxswain=Coxswain(), fast=True)
    sim.scheme = scheme
    y = sim.initial_state(surge_speed=4.6)
    dt = 1.0 / rate
    t = 0.0
    speeds = []
    for _ in range(int(seconds * rate)):
        y = sim.step(y, t, dt)
        t += dt
        speeds.append(float(np.hypot(y[6], y[7])))
    return y, float(np.mean(speeds[-int(5 * rate):]))


@pytest.mark.slow
def test_heun_at_sixty_is_rk4_at_a_hundred_to_a_few_centimetres():
    y_ref, v_ref = _run("rk4", 100.0)
    y_heun, v_heun = _run("heun", 60.0)
    assert np.all(np.isfinite(y_heun))
    distance = float(np.hypot(y_ref[0], y_ref[1]))
    assert distance > 100.0                       # it actually went somewhere
    assert abs(float(np.hypot(y_heun[0], y_heun[1])) - distance) < 0.05
    # A 5 s window of mean speed lands on a different stroke phase
    # under the two step sizes; 0.005 m/s on 6.2 is 0.08%, and the
    # 60 s A/B this rests on measured 0.0002.
    assert abs(v_heun - v_ref) < 5.0e-3
    assert abs(float(y_heun[3] - y_ref[3])) < np.radians(0.05)   # roll
    assert abs(float(y_heun[5] - y_ref[5])) < np.radians(0.05)   # heading


def test_heun_is_a_second_order_step():
    """On y' = y it must match the trapezoid rule exactly, and beat
    Euler: the error over one unit of time is O(dt^2)."""
    f = lambda t, y: y
    for dt, bound in ((0.1, 2e-3), (0.01, 2e-5)):
        y = np.array([1.0])
        t = 0.0
        while t < 1.0 - 1e-12:
            y = heun_step(f, t, y, dt)
            t += dt
        assert abs(float(y[0]) - np.e) < bound * np.e, (dt, y)


def test_the_simulator_defaults_to_rk4_and_the_studies_never_see_heun():
    boat = catalog.coxed_four(rate=30.0)
    sim = RowingSimulator(boat)
    assert sim.scheme == "rk4"
    y = sim.initial_state(surge_speed=4.0)
    # the default step IS rk4_step, to the bit
    direct = rk4_step(sim.derivative, 0.0, y, 0.01)
    assert np.array_equal(sim.step(y, 0.0, 0.01), direct)


def test_the_tiers_choose_heun_below_high_and_rk4_at_high():
    from coxswain.viz.menu import tier_settings

    for key in ("ultra", "minimal", "standard"):
        assert tier_settings(key).physics_scheme == "heun", key
    assert tier_settings("high").physics_scheme == "rk4"


def test_the_flag_overrides_the_tier():
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    with open(os.path.join(root, "scripts", "fpv.py"), encoding="utf-8") as f:
        text = f.read()
    assert '"--physics-scheme"' in text
    assert "simulator.scheme = (args.physics_scheme" in text
