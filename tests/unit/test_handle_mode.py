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
    with pytest.raises(ValueError, match="clock crew or the hands"):
        DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="follows",
                            blade_law="liftdrag", blade_added_mass="patton")


def _handle_sim(mass):
    boat = physics.resolve("research").apply(catalog.single_scull(rate=30.0))
    boat.power_scales = np.ones(boat.n_seats)
    return DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="handle",
                               blade_law="liftdrag", blade_added_mass=mass,
                               coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)


@pytest.fixture(scope="module")
def massive():
    sim = _handle_sim("patton")
    return sim, sim.run_strokes(3, surge_speed=4.2)


def test_added_mass_with_zero_mass_is_the_plain_handle_run():
    """The coupled solve, with the entrained mass set to zero, is the uncoupled one."""
    plain = _handle_sim("none").run_strokes(2, surge_speed=4.2)
    sim = _handle_sim("patton")
    sim._blade_mass = [0.0] * sim.n_oar_states
    coupled = sim.run_strokes(2, surge_speed=4.2)
    for a, b in zip(plain.strokes, coupled.strokes):
        assert b.mean_speed == pytest.approx(a.mean_speed, rel=1e-9)
        assert b.handle_power == pytest.approx(a.handle_power, rel=1e-7)


def test_added_mass_keeps_the_hands_on_the_handle_and_is_consistent(massive):
    """The coupled solve's blade normal acceleration, g . X_h + l phi_ddot + c, is the time
    derivative of the blade's normal velocity w_n along the run. With the blade in and out at
    w_n = 0 ([CR06] eq. 16) the entrained water's net work, -m [w_n^2 / 2], then vanishes; it
    may still move energy between the handle and the hull."""
    sim, result = massive
    n = sim.n_oar_states
    t, y = result.last_time, result.last_states
    start = t[0]
    for k in range(0, t.size, 9):
        angle, rate = sim._sweep_pose(t[k] - start)
        assert y[STATE_SIZE:STATE_SIZE + n, k] == pytest.approx(angle, abs=1e-12)
    from coxswain.core.rigid_body import solve_accelerations
    from coxswain.core.state import State
    lock = sim.boat.rig.seats[sim._seats[0]].oarlocks[0]
    wet = np.flatnonzero(~sim._air_mask[0])
    w_n, w_dot = [], []
    for k in wet:
        state = State.from_vector(y[:STATE_SIZE, k])
        angle = float(y[STATE_SIZE, k])
        rate, accel = sim._sweep_motion(t[k] - start)
        saved = sim._in_air
        sim._in_air = sim._air_mask[:, k].copy()
        try:
            system, rhs = sim._coupled_system(t[k], state, y[STATE_SIZE:STATE_SIZE + n, k],
                                              y[STATE_SIZE + n:STATE_SIZE + 2 * n, k])
        finally:
            sim._in_air = saved
        hull = solve_accelerations(system, rhs)[:6]
        mass, arm, g, c, wn, _h = sim._blade_mass_terms(
            0, lock, angle, rate, state, state.rot_hull_to_abs,
            np.asarray(state.omega, float), np.asarray(state.omega_hull, float))
        w_n.append(wn)
        w_dot.append(float(g @ hull) + arm * accel + c)
    w_n, w_dot, tw = np.array(w_n), np.array(w_dot), t[wet]
    numeric = np.gradient(w_n, tw)
    inner = slice(3, -3)
    scale = np.max(np.abs(w_dot))
    assert np.max(np.abs(numeric[inner] - w_dot[inner])) < 0.03 * scale
    # in and out at w_n ~ 0, to within one step of the switch (m w_n < 1 N s here)
    assert abs(w_n[0]) < 0.1 and abs(w_n[-1]) < 0.1
    net = np.trapezoid(w_dot * w_n, tw)
    assert abs(net) < 0.02 * np.trapezoid(np.abs(w_dot * w_n), tw)
