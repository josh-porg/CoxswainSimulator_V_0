r"""[CR06]'s catch on the dynamic oar: the blade enters at zero normal velocity.

[CR06] section 2.5 starts the drive when the blade's velocity through the
water normal to it is zero (their eq. 16), and its oar follows the body until
then.  ``catch="rest"`` -- the default, untouched -- resets the oar to the
catch angle at rest with the blade loaded at once, which lets the water start
the drive (TRACKING).  ``catch="sweep"`` puts the oar on the prescribed sweep
with its blade out until eq. 16 holds, then hands it to the torque drive with
the angle and rate the sweep gave it.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.core.state import STATE_SIZE, State
from coxswain.sim.dynamic_oar import DynamicOarSimulator


@pytest.fixture(scope="module")
def eight():
    from coxswain.boats import catalog

    return catalog.build("8+", rate=28.0)


@pytest.fixture(scope="module")
def torque(eight):
    return DynamicOarSimulator.peak_torque_for_power(eight, 380.0)


def test_the_default_catch_is_at_rest(eight, torque):
    assert DynamicOarSimulator(eight, peak_torque=torque).catch == "rest"


def test_an_unknown_catch_rule_is_refused(eight, torque):
    with pytest.raises(ValueError, match="catch rule"):
        DynamicOarSimulator(eight, peak_torque=torque, catch="early")


def test_the_sweep_catch_refuses_the_following_crew(eight, torque):
    with pytest.raises(ValueError, match="sweep catch"):
        DynamicOarSimulator(eight, peak_torque=torque, catch="sweep",
                            crew="follows")


def test_the_sweep_catch_takes_blade_added_mass(eight, torque):
    """Refused until 2026-09-14: the parked catch set 63 N s of water moving
    per blade at entry, which the sweep catch removes by construction."""
    sim = DynamicOarSimulator(eight, peak_torque=torque, catch="sweep",
                              blade_added_mass="patton")
    assert sim.catch == "sweep" and sim.blade_added_mass == "patton"


def _on_the_sweep(sim, tau, surge=5.0):
    n = sim.n_oar_states
    angle, rate = sim._sweep_pose(tau)
    y = sim.augmented_initial_state(surge)
    y[STATE_SIZE:STATE_SIZE + n] = angle
    y[STATE_SIZE + n:STATE_SIZE + 2 * n] = rate
    sim._in_air = np.ones(n, dtype=bool)
    sim._stroke_start = 0.0
    return y


def test_an_oar_on_the_sweep_puts_nothing_on_the_hull(eight, torque):
    sim = DynamicOarSimulator(eight, peak_torque=torque, catch="sweep")
    n = sim.n_oar_states
    y = _on_the_sweep(sim, 0.05)
    sim._oar_state = (y[STATE_SIZE:STATE_SIZE + n],
                      y[STATE_SIZE + n:STATE_SIZE + 2 * n])
    force, moment = sim._oar_loads(0.05, State.from_vector(y[:STATE_SIZE]))
    sim._oar_state = None
    assert np.all(force == 0.0) and np.all(moment == 0.0)


def test_an_oar_on_the_sweep_moves_with_it(eight, torque):
    sim = DynamicOarSimulator(eight, peak_torque=torque, catch="sweep")
    tau = 0.07
    y = _on_the_sweep(sim, tau)
    hull, angles, rates = sim._split(y)
    angle_rate, rate_rate = sim._oar_rates(tau, hull, angles, rates)
    sweep, timing = eight.oar_sweep, eight.timing
    h = 1e-6
    expected_accel = (float(sweep.rate(tau + h, timing))
                      - float(sweep.rate(tau - h, timing))) / (2 * h)
    assert np.allclose(angle_rate, float(sweep.rate(tau, timing)), rtol=0,
                       atol=1e-12)
    assert np.allclose(rate_rate, expected_accel, rtol=1e-3)


@pytest.mark.slow
def test_the_blade_enters_past_the_catch_already_moving(eight, torque):
    """Measured before this was built, on the prescribed sweep: 4.7 degrees
    past the catch on the eight at rate 28, at -1.15 rad/s."""
    sim = DynamicOarSimulator(eight, peak_torque=torque, catch="sweep")
    period = float(eight.timing.period)
    sim.run_strokes(2, surge_speed=5.2)
    second = [e for e in sim.entries if e[1] > period]
    assert sorted(e[0] for e in second) == list(range(sim.n_oar_states))
    catch = float(eight.oar_sweep.catch_angle)
    for _slot, _t, angle, rate in second:
        assert catch - np.radians(10.0) < angle < catch
        assert rate < 0.0


@pytest.mark.slow
def test_the_energy_the_sweep_carries_in_is_counted_as_rower_work(eight,
                                                                  torque):
    """Measured before it was counted: 71 J a seat, 8.9% of the handle power
    on the eight at rate 28.  Left out, the sweep catch and the rest catch
    would be compared at unequal power."""
    sim = DynamicOarSimulator(eight, peak_torque=torque, catch="sweep")
    period = float(eight.timing.period)
    run = sim.run_strokes(2, surge_speed=5.8)
    rowers = len(sim._seats)
    for record in run.strokes:
        start = record.index * period
        expected = sum(sim._entry_energy(slot, angle, rate)
                       for slot, when, angle, rate in sim.entries
                       if start - 1e-9 <= when < start + period - 1e-9)
        assert record.entry_work == pytest.approx(expected / rowers,
                                                  rel=1e-12)
        assert record.entry_work > 20.0
    rest = DynamicOarSimulator(eight, peak_torque=torque).run_strokes(
        1, surge_speed=5.8)
    assert rest.strokes[0].entry_work == 0.0


@pytest.mark.slow
def test_under_both_cr06_rules_the_drive_now_runs(eight, torque):
    """The case ``tests/test_release_rule.py`` pins as stuck under the rest
    catch: with the blade entering already moving, the release rule's drive
    runs to its finish and the boat keeps its speed."""
    sim = DynamicOarSimulator(eight, peak_torque=torque, catch="sweep",
                              release="slip")
    run = sim.run_strokes(3, surge_speed=5.2)
    last = run.strokes[-1]
    assert last.finished
    assert last.mean_speed > 4.0
