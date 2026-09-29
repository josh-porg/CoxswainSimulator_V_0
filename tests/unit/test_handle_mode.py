"""Phase 4.3 rung 1: hands on the handle. The mechanics, not the athletes' numbers."""
import numpy as np
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.core.state import STATE_SIZE
from coxswain.sim.control import Coxswain
from coxswain.sim.dynamic_oar import DynamicOarSimulator


@pytest.fixture(scope="module")
def run():
    boat = physics.resolve("research").apply(catalog.single_scull(rate=30.0))
    boat.power_scales = np.ones(boat.n_seats)
    sim = DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="handle",
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    return sim, sim.run_strokes(4, surge_speed=4.2)


def test_the_hands_stay_on_the_handle(run):
    """Every step of the last stroke, every oar is exactly on the hands' sweep."""
    sim, result = run
    n = sim.n_oar_states
    start = result.last_time[0]
    for k in range(0, result.last_time.size, 7):
        angle, rate = sim._sweep_pose(result.last_time[k] - start)
        assert result.last_states[STATE_SIZE:STATE_SIZE + n, k] == pytest.approx(angle, abs=1e-12)
        assert result.last_states[STATE_SIZE + n:STATE_SIZE + 2 * n, k] == pytest.approx(rate, abs=1e-12)


def test_the_blade_enters_by_eq_16_every_stroke_and_is_out_on_the_recovery(run):
    sim, result = run
    period = sim.boat.timing.period
    strokes = {int(when // period) for _slot, when, _a, _r in sim.entries}
    assert strokes == {0, 1, 2, 3}
    air = sim._air_mask
    tau = result.last_time - result.last_time[0]
    recovery = tau > 0.75 * period
    assert air[:, recovery].all()                       # blade out late in the stroke
    assert (~air).any()                                 # and in for part of the drive


def test_power_is_the_constraint_power_and_positive(run):
    sim, result = run
    stroke = result.strokes[-1]
    assert stroke.handle_power > 50.0
    assert stroke.entry_work == 0.0                     # counted in the constraint power
    power = sim._handle_power(result.last_time, result.last_states)
    per_rower = np.trapezoid(power, result.last_time) / sim.boat.timing.period
    assert stroke.handle_power == pytest.approx(per_rower, rel=1e-9)


def test_the_mode_needs_the_sweep_entry_and_the_angle_release():
    boat = physics.resolve("research").apply(catalog.single_scull(rate=30.0))
    with pytest.raises(ValueError, match="catch='sweep'"):
        DynamicOarSimulator(boat, peak_torque=0.0, catch="rest", crew="handle")
    with pytest.raises(ValueError):
        DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="handle", release="slip")
    with pytest.raises(ValueError, match="clock crew"):
        DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="handle",
                            blade_law="liftdrag", blade_added_mass="patton")
