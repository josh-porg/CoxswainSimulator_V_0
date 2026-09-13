r"""Phase 4.1: the prescribed crew follows the dynamic oar.

The baseline this closes, measured on a settled stroke at 380 W before it was
built: the clock crew's hands sat up to 0.19 m off the dynamic handle on the
eight and 0.74 m on the single, whose oar ran 48 degrees behind the body
mid-drive.

These check the construction against definitions that already exist -- the
rig's handle, the oar's inertia, the stroke table -- and that the clock crew
is untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.core.state import STATE_SIZE
from coxswain.crew.follow import FollowingCrew
from coxswain.sim.dynamic_oar import DynamicOarSimulator


@pytest.fixture(scope="module")
def single():
    from coxswain.boats import catalog

    return catalog.single_scull(rate=30.0)


@pytest.fixture(scope="module")
def eight():
    from coxswain.boats import catalog

    return catalog.eight(rate=32.0)


def _handle(lock, angle):
    side = int(lock.side)
    axis = np.array([np.sin(angle), side * np.cos(angle), 0.0])
    return np.asarray(lock.position, dtype=float) \
        - float(lock.oar.inboard) * axis


def _interior(follower, count=7):
    """Grid nodes well clear of the rate floor at both ends."""
    nodes = np.flatnonzero(~follower.floored)
    return nodes[np.linspace(len(nodes) // 8, len(nodes) - len(nodes) // 8,
                             count).astype(int)]


# ---------------------------------------------------------------------------
# the construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["single", "eight"])
def test_the_hands_are_on_the_handle_at_every_angle(name, request):
    boat = request.getfixturevalue(name)
    seat = next(i for i, s in enumerate(boat.rig.seats) if s.oarlocks)
    follower = FollowingCrew(boat, seat=seat)
    lock = boat.rig.seats[seat].oarlocks[0]
    anchor = float(boat.crew[seat].rower.station.x_ankle)
    catch = float(boat.oar_sweep.catch_angle)
    finish = float(boat.oar_sweep.finish_angle)
    worst = 0.0
    for angle in np.linspace(finish, catch, 90):
        hand = follower.hand(angle) + np.array([anchor, 0.0, 0.0])
        gap = (_handle(lock, angle) - hand)[[0, 2]]
        worst = max(worst, float(np.linalg.norm(gap)))
    assert worst < 0.002, worst


def test_the_body_has_one_kinetic_energy(single):
    """Seen from the hull, ``sum m |v|^2 / 2``; seen from the oar,
    ``I phi_dot^2 / 2``.  The same number, because the inertia is built from
    the velocities the hull is given."""
    sim = DynamicOarSimulator(single, peak_torque=400.0, crew="follows")
    follower = sim._followers[0]
    mass = follower.masses
    for node in _interior(follower):
        angle = float(follower.angle[node])
        rate = -2.3
        _p, velocity, _a = follower.drive_state(angle, rate, 4.0)
        hull_side = 0.5 * float((mass * np.sum(velocity ** 2, axis=1)).sum())
        oar_side = 0.5 * follower.inertia(angle) * rate ** 2
        assert hull_side == pytest.approx(oar_side, rel=1e-9)
        assert sim._oars[0].inertia(angle) == pytest.approx(
            follower.inertia(angle) + sim._oars[0].inertia.oar_inertia,
            rel=1e-12)


def test_the_power_books_close_outside_the_floor(single):
    """``sum m v.a = d/dt (I phi_dot^2 / 2)`` wherever the sweep rate is
    above the floor.  Inside it the velocity is capped and they part; that is
    the regularisation's known cost, not tested away."""
    sim = DynamicOarSimulator(single, peak_torque=400.0, crew="follows")
    follower, oar = sim._followers[0], sim._oars[0]
    mass = follower.masses
    rate, acc = -2.3, 4.0
    for node in _interior(follower):
        angle = float(follower.angle[node])
        _p, velocity, accel = follower.drive_state(angle, rate, acc)
        hull_side = float((mass * np.sum(velocity * accel, axis=1)).sum())
        moment, slope = oar.inertia_at(angle)
        body = moment - oar.inertia.oar_inertia
        oar_side = 0.5 * slope * rate ** 3 + body * rate * acc
        assert hull_side == pytest.approx(oar_side, rel=0.02), angle


def test_the_recovery_starts_at_the_finish_and_reaches_the_catch(single):
    follower = FollowingCrew(single, seat=0)
    finish = float(single.oar_sweep.finish_angle)
    drive_took = 1.11
    at_finish, _v, _a = follower.drive_state(finish, -1.0, 0.0)
    starting, _v, _a = follower.recovery_state(drive_took, drive_took)
    np.testing.assert_allclose(starting, at_finish, atol=1e-9)
    arriving, _v, _a = follower.recovery_state(follower.period, drive_took)
    catch_pose, _v, _a = follower.table.at(0.0)
    np.testing.assert_allclose(arriving, catch_pose, atol=1e-9)


def test_a_body_at_rest_at_the_catch_is_at_rest(single):
    follower = FollowingCrew(single, seat=0)
    catch = float(single.oar_sweep.catch_angle)
    _p, velocity, _a = follower.drive_state(catch, 0.0, 0.0)
    assert np.all(velocity == 0.0)


# ---------------------------------------------------------------------------
# the wiring
# ---------------------------------------------------------------------------
def test_the_default_crew_is_still_the_clock(eight):
    clock = DynamicOarSimulator(eight, peak_torque=600.0)
    assert clock.crew == "clock"
    n = clock.n_oar_states
    assert clock.augmented_initial_state(5.0).shape == (STATE_SIZE + 2 * n,)
    assert all(f is None for f in clock._followers)


def test_an_unknown_crew_mode_is_refused(eight):
    with pytest.raises(ValueError, match="crew mode"):
        DynamicOarSimulator(eight, peak_torque=600.0, crew="muscles")


def _following_mid_drive(sim, surge=4.5, angle=np.radians(10.0), rate=-2.0):
    n = sim.n_oar_states
    y = sim.augmented_initial_state(surge)
    assert y.shape == (STATE_SIZE + 3 * n,)
    y[STATE_SIZE:STATE_SIZE + n] = angle
    y[STATE_SIZE + n:STATE_SIZE + 2 * n] = rate
    return y


def test_the_drive_clock_runs_only_while_the_blade_is_in(single):
    sim = DynamicOarSimulator(single, peak_torque=400.0, crew="follows")
    n = sim.n_oar_states
    y = _following_mid_drive(sim)
    assert np.all(sim.derivative(0.3, y)[STATE_SIZE + 2 * n:] == 1.0)
    y[STATE_SIZE:STATE_SIZE + n] = sim._oars[0].finish_angle - 0.01
    y[STATE_SIZE + 2 * n:] = 1.05
    assert np.all(sim.derivative(1.5, y)[STATE_SIZE + 2 * n:] == 0.0)


def test_the_oar_balance_is_the_clock_crews_arithmetic(single):
    """Same balance, same order of operations; only the inertia it is handed
    differs."""
    sim = DynamicOarSimulator(single, peak_torque=400.0, crew="follows")
    n = sim.n_oar_states
    y = _following_mid_drive(sim)
    got = sim.derivative(0.1, y)
    _a, expected = sim._oar_rates(0.1, y[:STATE_SIZE],
                                  y[STATE_SIZE:STATE_SIZE + n],
                                  y[STATE_SIZE + n:STATE_SIZE + 2 * n])
    assert got[STATE_SIZE + n:STATE_SIZE + 2 * n].tolist() == expected.tolist()


def test_the_hull_feels_the_following_body(single):
    """Not the clock body: at the same hull state and time, mid-drive, the
    surge acceleration differs."""
    follows = DynamicOarSimulator(single, peak_torque=400.0, crew="follows")
    clock = DynamicOarSimulator(single, peak_torque=400.0)
    y = _following_mid_drive(follows)
    n = clock.n_oar_states
    surge_follows = follows.derivative(0.1, y)[6]
    surge_clock = clock.derivative(0.1, y[:STATE_SIZE + 2 * n])[6]
    assert abs(surge_follows - surge_clock) > 0.05, (surge_follows,
                                                     surge_clock)


def test_outside_a_derivative_the_field_is_the_prescribed_one(single):
    sim = DynamicOarSimulator(single, peak_torque=400.0, crew="follows")
    _m, following, _v, _a = sim.crew_field(0.4)
    _m, prescribed, _v, _a = single.crew_field(0.4)
    np.testing.assert_array_equal(following, prescribed)


# ---------------------------------------------------------------------------
# momentum -- the defect the first settled run found
# ---------------------------------------------------------------------------
def _crew_momentum_books(sim, strokes=4, surge=5.3):
    """``(integral of sum m a_x dt, change in sum m v_x)`` over the last
    stroke.  The hull feels the crew only through ``sum m a``, so the two
    must agree or the hull is handed momentum the crew never had."""
    run = sim.run_strokes(strokes, surge_speed=surge)
    times, states = run.last_time, run.last_states
    n = sim.n_oar_states
    sim._stroke_start = float(times[0])
    momentum, force = [], []
    for k in range(times.size):
        t = float(times[k])
        y = states[:, k]
        angles = y[STATE_SIZE:STATE_SIZE + n]
        rates = y[STATE_SIZE + n:STATE_SIZE + 2 * n]
        if sim.crew == "follows":
            _r, acc = sim._oar_rates(t, y[:STATE_SIZE], angles, rates)
            sim._crew_state = (angles, rates, acc, y[STATE_SIZE + 2 * n:], t)
        mass, _p, velocity, accel = sim.crew_field(t)
        sim._crew_state = None
        momentum.append(float((mass * velocity[:, 0]).sum()))
        force.append(float((mass * accel[:, 0]).sum()))
    return float(np.trapezoid(force, times)), momentum[-1] - momentum[0]


@pytest.mark.slow
def test_the_clock_crew_hands_the_hull_no_momentum_of_its_own():
    from coxswain.boats import catalog

    boat = catalog.build("8+", rate=28.0)
    sim = DynamicOarSimulator(boat, peak_torque=DynamicOarSimulator
                              .peak_torque_for_power(boat, 380.0))
    impulse, change = _crew_momentum_books(sim)
    assert abs(impulse - change) < 0.5, (impulse, change)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "phase 4.1 defect, TRACKING: the following crew's velocity jumps at the "
    "finish (the oar is stopped dead there, and the body with it) and at the "
    "catch, and inside the rate floor its acceleration is not dv/dt. "
    "Measured on the eight at 380 W: +361 N s per stroke against a momentum "
    "change of +108; the settled speed it produced, 5.06 m/s, is an artefact"))
def test_the_following_crew_hands_the_hull_no_momentum_of_its_own():
    from coxswain.boats import catalog

    boat = catalog.build("8+", rate=28.0)
    sim = DynamicOarSimulator(boat, peak_torque=DynamicOarSimulator
                              .peak_torque_for_power(boat, 380.0),
                              crew="follows")
    impulse, change = _crew_momentum_books(sim)
    assert abs(impulse - change) < 0.5, (impulse, change)
