r"""The dynamic oar on the full 6-DOF hull.

The reduced model (``tests/test_oarloop.py``) passed phase 2's gate and said
what it could not show: the crew's mass did not move, so there was no
intracycle surge swing -- the thing that destroyed the efficiency-only
wiring.  This is the same oar physics on the full simulator, with the
prescribed crew surging on its own clock and every other line of the force
assembly the one ``shipped`` runs.

The fast tests check the pieces against definitions that already exist.
The slow ones measure the question the reduced model left open.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.core.state import STATE_SIZE, State
from coxswain.sim.dynamic_oar import DynamicOarSimulator


@pytest.fixture(scope="module")
def single():
    from coxswain.boats import catalog

    return catalog.single_scull(rate=30.0)


@pytest.fixture(scope="module")
def eight():
    from coxswain.boats import catalog

    return catalog.eight(rate=32.0)


def _straight(surge: float, yaw_rate: float = 0.0) -> State:
    y = np.zeros(STATE_SIZE)
    y[6] = surge
    y[11] = yaw_rate
    return State.from_vector(y)


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------
def test_shipped_still_uses_the_prescribed_oar_block(eight):
    """The seam is a seam. The shipped simulator's own ``_oar_loads`` is
    the inline block moved verbatim; the golden trajectory holds it to the
    bit (``tests/test_stepwise.py``). Here: the research override is not
    what the base class runs."""
    from coxswain.sim.simulator import RowingSimulator

    assert (RowingSimulator._oar_loads
            is not DynamicOarSimulator._oar_loads)


def test_straight_running_slip_is_the_blade_models_own(eight):
    """``u_lock . n / cos(phi)`` is just the surge when nothing turns."""
    sim = DynamicOarSimulator(eight, peak_torque=600.0)
    state = _straight(4.85)
    for seat in eight.rig.seats:
        for lock in seat.oarlocks:
            for angle in np.radians((50.0, 10.0, -30.0)):
                assert sim._lock_speed_on_normal(state, lock, angle) == \
                    pytest.approx(4.85, rel=1e-12)


def test_a_turning_boat_loads_its_two_sides_differently(single):
    """Under yaw the outside blade meets faster water than the inside one."""
    sim = DynamicOarSimulator(single, peak_torque=400.0)
    state = _straight(4.0, yaw_rate=0.2)
    locks = single.rig.seats[0].oarlocks
    speeds = {int(lock.side): sim._lock_speed_on_normal(state, lock,
                                                        np.radians(10.0))
              for lock in locks}
    assert len(speeds) == 2
    assert abs(speeds[+1] - speeds[-1]) > 0.05, speeds


def test_blade_load_acts_at_the_blade_with_no_gearing(eight):
    """The Newtonian statement, checked against a hand calculation.

    For the hull-plus-crew system the blade force is the external load, so
    the hull gets ``sum F_n n`` at ``r_lock + l a`` -- not ``gearing`` times
    it. The reduced model's level check caught exactly that factor.
    """
    sim = DynamicOarSimulator(eight, peak_torque=600.0)
    n = sim.n_oar_states
    # A clearly DRIVING blade: slip = l phi_dot + v cos(phi) is about -1 m/s.
    # It was -2.0 rad/s, which with the blade force at the tip sat at -0.34
    # and with it at the blade centre (l = 2.30 m) sits at +0.18 -- a
    # braking blade, which is not what this test is about.
    angle, rate = np.radians(10.0), -2.5
    sim._oar_state = (np.full(n, angle), np.full(n, rate))
    state = _straight(4.85)
    try:
        force, moment = sim._oar_loads(0.0, state)
    finally:
        sim._oar_state = None

    expected_f, expected_m = np.zeros(3), np.zeros(3)
    for slot, seat in enumerate(sim._seats):
        oar = sim._oars[slot]
        for lock in eight.rig.seats[seat].oarlocks:
            side = int(lock.side)
            fn = float(oar.blade.normal_force(angle, rate, 4.85))
            normal = np.array([np.cos(angle), -side * np.sin(angle), 0.0])
            axis = np.array([np.sin(angle), side * np.cos(angle), 0.0])
            point = np.asarray(lock.position) + oar.outboard * axis
            expected_f += fn * normal
            expected_m += np.cross(point, fn * normal)

    assert np.allclose(force, expected_f, rtol=1e-12, atol=1e-9)
    assert np.allclose(moment, expected_m, rtol=1e-12, atol=1e-9)
    # A balanced eight's lateral loads cancel; the thrust does not.
    assert abs(force[1]) < 1e-6 * abs(force[0])
    assert force[0] > 0.0


def test_a_finished_oar_puts_no_load_on_the_hull(eight):
    sim = DynamicOarSimulator(eight, peak_torque=600.0)
    n = sim.n_oar_states
    finish = sim._oars[0].finish_angle
    sim._oar_state = (np.full(n, finish - 1e-3), np.full(n, -2.0))
    try:
        force, moment = sim._oar_loads(0.0, _straight(4.85))
    finally:
        sim._oar_state = None
    assert np.allclose(force, 0.0) and np.allclose(moment, 0.0)


def test_refuses_what_it_does_not_model(eight):
    from coxswain.boats import catalog

    staggered = catalog.eight(rate=32.0)
    offsets = np.zeros(staggered.n_seats)
    offsets[3] = 0.02
    staggered.phase_offsets = offsets
    with pytest.raises(ValueError, match="synchronised"):
        DynamicOarSimulator(staggered, peak_torque=600.0)

    with pytest.raises(ValueError, match="blade_contact"):
        DynamicOarSimulator(eight, peak_torque=600.0,
                            blade_contact=object())


def test_power_is_the_closed_form_work(single):
    """Work per drive is ``peak * integral(shape dphi)``, whatever the speed.

    Measured by integrating ``|tau phi_dot|`` through a real stroke and
    compared with the closed form. A sculler, so the count of two oars per
    rower is exercised too.
    """
    torque = DynamicOarSimulator.peak_torque_for_power(single, 380.0)
    run = DynamicOarSimulator(single, peak_torque=torque).run_strokes(
        1, surge_speed=4.3)
    assert run.strokes[0].finished
    assert run.strokes[0].handle_power == pytest.approx(380.0, rel=0.01)
    # And the stroke carries a measured blade efficiency, not a placeholder.
    level = run.strokes[0].blade_efficiency
    assert level is not None and 0.0 < level < 1.0, level
    # And the last stroke's full augmented state is kept, consistent with the
    # speed trace kept beside it -- it is what the blade-path figure draws.
    n = DynamicOarSimulator(single, peak_torque=torque).n_oar_states
    assert run.last_states.shape == (STATE_SIZE + 2 * n,
                                     run.last_time.size)
    assert np.allclose(np.hypot(run.last_states[6], run.last_states[7]),
                       run.last_speed)


def test_blade_efficiency_is_measured_on_the_states_the_boat_had(single):
    """Hand-checked against the blade model's own definition.

    Two samples: one mid-drive, one with the oar past its finish. The first
    must contribute ``1 - |slip|/|blade speed|`` weighted by its blade load;
    the second must contribute nothing, because its blade is out.
    """
    sim = DynamicOarSimulator(single, peak_torque=400.0)
    n = sim.n_oar_states
    oar = sim._oars[0]
    states = np.zeros((STATE_SIZE + 2 * n, 2))
    states[6, :] = 4.3
    angle, rate = np.radians(10.0), -2.2
    states[STATE_SIZE:STATE_SIZE + n, 0] = angle
    states[STATE_SIZE + n:, 0] = rate
    states[STATE_SIZE:STATE_SIZE + n, 1] = oar.finish_angle - 1e-3
    states[STATE_SIZE + n:, 1] = -9.0

    got = sim._stroke_blade_efficiency(np.array([0.0, 0.01]), states)
    # Straight running: both locks see the surge, so both have the same
    # efficiency and the force weighting cannot move it.
    expected = float(oar.blade.efficiency(angle, rate, 4.3))
    assert got == pytest.approx(expected, rel=1e-12)
    assert 0.0 < got < 1.0


def test_no_loaded_blade_means_no_measurement(single):
    sim = DynamicOarSimulator(single, peak_torque=400.0)
    n = sim.n_oar_states
    states = np.zeros((STATE_SIZE + 2 * n, 1))
    states[STATE_SIZE:STATE_SIZE + n, 0] = sim._oars[0].finish_angle - 1e-3
    assert sim._stroke_blade_efficiency(np.array([0.0]), states) is None


# ---------------------------------------------------------------------------
# what the reduced model could not show
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_it_does_not_collapse_where_the_efficiency_wiring_did():
    """Phase 1's failure, re-run with the force model in place.

    The efficiency-only wiring took the eight at rate 28 to 0.63 m/s from
    either side -- positive feedback between the surge dip and blade
    efficiency, with no restoring term. Same boat, same rate, the crew's
    surge swing present, started from both sides of the answer: the two
    runs must meet, at a racing speed.
    """
    from coxswain.boats import catalog

    boat = catalog.eight(rate=28.0)
    torque = DynamicOarSimulator.peak_torque_for_power(boat, 283.0)
    finals = []
    for start in (3.4, 6.5):
        run = DynamicOarSimulator(boat, peak_torque=torque).run_strokes(
            18, surge_speed=start)
        assert all(s.finished for s in run.strokes[-4:])
        assert run.strokes[-1].surge_swing > 0.3, "the swing must be present"
        finals.append(run.settled_speed())
    assert min(finals) > 3.0, finals
    assert abs(finals[0] - finals[1]) < 0.01 * np.mean(finals), finals


@pytest.mark.slow
def test_the_eight_and_four_reach_published_pace_at_380_watts():
    """The reduced model's result, on the full hull."""
    from coxswain.boats import catalog

    for name, low, high in (("8+", 5.0, 5.6), ("4+", 4.5, 5.1)):
        boat = catalog.build(name, rate=32.0)
        torque = DynamicOarSimulator.peak_torque_for_power(boat, 380.0)
        run = DynamicOarSimulator(boat, peak_torque=torque).run_strokes(
            16, surge_speed=low)
        assert run.drift() < 0.005, (name, run.drift())
        assert run.settled_power() == pytest.approx(380.0, rel=0.01)
        assert low <= run.settled_speed() <= high, (name,
                                                    run.settled_speed())


@pytest.mark.slow
def test_the_gate_holds_on_the_full_hull():
    """Phase 2's gate, scored on the full hull the way the baseline was.

    Where the fitted eta-against-v line reaches zero, as a multiple of
    mean speed. Measured 2026-09-12 on the eight at rate 28:

        baseline, prescribed force     0.020  (through the origin)
        the floor the target sets      0.15
        reduced model, no crew swing   8.9
        full 6-DOF hull                1.81

    Re-measured 2026-09-13 with the blade force at the blade centre (2.30 m,
    not the tip at 2.56): reduced model 6.4, full hull 1.57, eta/v spread
    37%; eta 0.518 at 2.90 m/s to 0.660 at 5.39, swing 80% to 47%.

    It passes, and by twelve times the floor -- but it is a much weaker
    pass than the reduced model promised, and the difference is recorded
    rather than smoothed over. On the full hull eta still rises with
    speed (0.557 at 2.97 m/s to 0.691 at 5.48), where the reduced model's
    was nearly flat. The crew's surge swing is back, and it is largest
    exactly where the boat is slowest (78% of mean speed at 2.97 m/s, 46%
    at 5.48).

    Thresholds sit well below the measurement so the test is about the
    gate, not about the third decimal: the crossing must clear the floor
    with room to spare, and eta/v must be nowhere near constant -- it was
    flat to 2.7% on the baseline and is 39% here.
    """
    from coxswain.boats import catalog
    from coxswain.sim.oarloop import drag_curve

    boat = catalog.eight(rate=28.0)
    drag = drag_curve(boat)
    speeds, etas = [], []
    for torque in (200.0, 450.0, 900.0):
        run = DynamicOarSimulator(boat, peak_torque=torque).run_strokes(
            16, surge_speed=4.5)
        assert run.drift() < 0.01, (torque, run.drift())
        speed = run.settled_speed()
        speeds.append(speed)
        etas.append(drag(speed) * speed / (run.settled_power() * 8))

    speeds, etas = np.array(speeds), np.array(etas)
    slope, intercept = np.polyfit(speeds, etas, 1)
    crossing = abs(-intercept / slope) / speeds.mean()
    assert crossing > 0.5, (slope, intercept, crossing)

    ratio = etas / speeds
    assert np.ptp(ratio) / ratio.mean() > 0.2, ratio


@pytest.mark.slow
def test_the_residual_rise_in_eta_is_half_swing_and_half_blade():
    """Why eta still rises with speed on the full hull -- measured, not guessed.

    The first guess was that the swing starves the blade at low speed, as
    it did the efficiency-only wiring. It was measured and is backwards:
    blade efficiency at instantaneous speed is 1.09x its mean-speed value
    when slow and 0.87x when fast.

    What it actually is, on the eight at rate 28:

    * **the drag channel** -- drag power is steeply nonlinear in speed, so
      a swinging boat spends more than R(v_mean) v_mean, the numerator
      eta uses. 16% more at 3 m/s, 6% at 5.5. Charging for it halves the
      rise in eta, 24% to 13%.
    * **the blade itself** -- a slower boat lets the same pull slip more.
      That part is real physics, recorded in SOURCES sec. 7, and exactly
      the speed dependence the prescribed model could not have.

    The drag half rides on the prescribed crew motion, which is defect
    two and phase 4's job, not phase 2's.
    """
    from coxswain.boats import catalog
    from coxswain.sim.oarloop import drag_curve

    boat = catalog.eight(rate=28.0)
    drag = drag_curve(boat)
    rows = []
    for torque in (200.0, 900.0):
        run = DynamicOarSimulator(boat, peak_torque=torque).run_strokes(
            14, surge_speed=4.5)
        power = run.settled_power() * 8
        speed = run.settled_speed()
        swing_cost = run.drag_power_ratio(drag)
        naive = drag(speed) * speed / power
        rows.append((swing_cost, naive, naive * swing_cost))

    (cost_slow, naive_slow, true_slow), (cost_fast, naive_fast, true_fast) = rows
    # The swing costs more on the slow boat, and always costs something.
    assert cost_slow > cost_fast > 1.0, (cost_slow, cost_fast)
    # Charging for it removes part of the rise, not all of it: what is left
    # is the blade slipping more on a slower boat.
    naive_rise = naive_fast - naive_slow
    true_rise = true_fast - true_slow
    assert naive_rise > true_rise > 0.0, (naive_rise, true_rise)


# ---------------------------------------------------------------------------
# one body, several oars
# ---------------------------------------------------------------------------
def _mid_drive(sim, surge=4.0, angle=np.radians(10.0), rate=-2.0):
    y = sim.augmented_initial_state(surge)
    n = sim.n_oar_states
    y[STATE_SIZE:STATE_SIZE + n] = angle
    y[STATE_SIZE + n:] = rate
    return y, State.from_vector(y[:STATE_SIZE]), angle, rate


def test_a_sweep_seat_keeps_the_single_oar_balance(eight):
    """Bit-identical to the unit's own balance -- the sculler fix does not
    touch a seat with one oar."""
    sim = DynamicOarSimulator(eight, peak_torque=600.0)
    y, state, angle, rate = _mid_drive(sim, surge=4.85)
    n = sim.n_oar_states
    got = sim.derivative(0.1, y)[STATE_SIZE + n]
    lock = eight.rig.seats[sim._seats[0]].oarlocks[0]
    torque = sim._torque(0, angle, state, 0.1)
    expected = float(sim._oars[0].acceleration(
        angle, rate, -torque, sim._lock_speed_on_normal(state, lock, angle)))
    assert got == expected


def test_a_sculler_is_one_body_with_two_oars(single):
    """``(I_crew + 2 I_oar) phi_ddot = -2 tau + sum blade - (1/2) I' w^2``.

    Not the mean of two oars each carrying the whole rower, which counted
    the body twice.
    """
    sim = DynamicOarSimulator(single, peak_torque=400.0)
    y, state, angle, rate = _mid_drive(sim)
    n = sim.n_oar_states
    got = sim.derivative(0.1, y)[STATE_SIZE + n]

    oar = sim._oars[0]
    locks = single.rig.seats[0].oarlocks
    torque = sim._torque(0, angle, state, 0.1)
    moment, slope = oar.inertia_at(angle)
    speeds = [sim._lock_speed_on_normal(state, lock, angle) for lock in locks]
    blade = sum(float(oar.blade_torque(angle, rate, v)) for v in speeds)
    one_body = ((-2 * torque + blade - 0.5 * slope * rate ** 2)
                / (moment + oar.inertia.oar_inertia))
    assert got == pytest.approx(one_body, rel=1e-12)

    per_oar_mean = float(np.mean([oar.acceleration(angle, rate, -torque, v)
                                  for v in speeds]))
    assert abs(got - per_oar_mean) > 0.01 * abs(per_oar_mean)


def test_a_scullers_drive_closes_its_energy_books(single):
    """Handle work = work against the water + the seat's kinetic energy.

    Integrated over one drive at a fixed 4 m/s with the seat balance. The
    per-oar form this replaced left 2.2% of the handle work unaccounted for.
    """
    sim = DynamicOarSimulator(single, peak_torque=368.0)
    oar = sim._oars[0]
    locks = single.rig.seats[0].oarlocks
    state = _straight(4.0)
    n_locks = len(locks)

    angle, rate, dt = float(oar.catch_angle), 0.0, 0.0005
    handle = water = 0.0
    while angle > oar.finish_angle:
        torque = sim._torque(0, angle, state, 0.0)
        acc = sim._seat_acceleration(oar, angle, rate, torque, state, locks)
        blade = sum(float(oar.blade_torque(
            angle, rate, sim._lock_speed_on_normal(state, lock, angle)))
            for lock in locks)
        handle += n_locks * abs(torque * rate) * dt
        water += blade * rate * dt
        angle += dt * rate
        rate += dt * acc
    inertia = oar.inertia_at(angle)[0] + (n_locks - 1) * oar.inertia.oar_inertia
    kinetic = 0.5 * inertia * rate ** 2
    residual = handle + water - kinetic
    assert abs(residual) < 0.002 * handle, (handle, water, kinetic, residual)


# ---------------------------------------------------------------------------
# tier 2: lift and drag on the angle of attack
# ---------------------------------------------------------------------------
def test_an_unknown_blade_law_is_refused(eight):
    with pytest.raises(ValueError, match="blade law"):
        DynamicOarSimulator(eight, peak_torque=600.0, blade_law="magic")


def test_the_default_is_still_tier_one(eight):
    sim = DynamicOarSimulator(eight, peak_torque=600.0)
    assert sim.blade_law == "slip"
    assert sim._liftdrag == []


def test_tier_two_puts_both_load_components_on_the_hull(eight):
    """Normal AND tangential, at the blade -- checked by hand."""
    from coxswain.crew.liftdrag import LiftDragBlade

    sim = DynamicOarSimulator(eight, peak_torque=600.0, blade_law="liftdrag")
    n = sim.n_oar_states
    # -2.5 rad/s puts the attack angle near 33 degrees, where the tangential
    # load is tens of newtons; at -2.0 the normal flow sat near zero once the
    # blade force moved from the tip to the blade centre, and it was ~1 N.
    angle, rate = np.radians(25.0), -2.5
    sim._oar_state = (np.full(n, angle), np.full(n, rate))
    state = _straight(4.85)
    try:
        force, moment = sim._oar_loads(0.0, state)
    finally:
        sim._oar_state = None

    expected_f, expected_m = np.zeros(3), np.zeros(3)
    tangential_seen = 0.0
    for slot, seat in enumerate(sim._seats):
        lock = eight.rig.seats[seat].oarlocks[0]
        oar = sim._oars[slot]
        blade = LiftDragBlade.big_blade(outboard=oar.outboard,
                                        area=lock.oar.blade_area,
                                        density=eight.water.density)
        side = int(lock.side)
        f_n, f_t = blade.loads(angle, rate, np.array([4.85, 0.0]), side)
        tangential_seen = max(tangential_seen, abs(f_t))
        normal = np.array([np.cos(angle), -side * np.sin(angle), 0.0])
        axis = np.array([np.sin(angle), side * np.cos(angle), 0.0])
        load = f_n * normal + f_t * axis
        point = np.asarray(lock.position) + oar.outboard * axis
        expected_f += load
        expected_m += np.cross(point, load)
    assert np.allclose(force, expected_f, rtol=1e-12, atol=1e-9)
    assert np.allclose(moment, expected_m, rtol=1e-12, atol=1e-9)
    assert tangential_seen > 1.0, "a blade at 25 degrees has flow along it"


def test_tier_two_turns_the_oar_with_its_normal_load_only(eight):
    """``I phi_ddot + (1/2) I' w^2 = -tau + l F_n`` for a sweep seat."""
    sim = DynamicOarSimulator(eight, peak_torque=600.0, blade_law="liftdrag")
    y, state, angle, rate = _mid_drive(sim, surge=4.85)
    n = sim.n_oar_states
    got = sim.derivative(0.1, y)[STATE_SIZE + n]

    oar = sim._oars[0]
    lock = eight.rig.seats[sim._seats[0]].oarlocks[0]
    torque = sim._torque(0, angle, state, 0.1)
    f_n, _f_t = sim._blade_loads(0, angle, rate, state, lock)
    moment, slope = oar.inertia_at(angle)
    expected = (-torque + oar.outboard * f_n - 0.5 * slope * rate ** 2) / moment
    assert got == pytest.approx(expected, rel=1e-12)


def test_tier_two_efficiency_uses_tier_ones_definition(single):
    """``1 - |normal slip| / |blade speed|``, weighted by normal load --
    so the two tiers are scored against one band."""
    sim = DynamicOarSimulator(single, peak_torque=400.0, blade_law="liftdrag")
    n = sim.n_oar_states
    states = np.zeros((STATE_SIZE + 2 * n, 1))
    states[6, 0] = 4.3
    angle, rate = np.radians(10.0), -2.2
    states[STATE_SIZE:STATE_SIZE + n, 0] = angle
    states[STATE_SIZE + n:, 0] = rate
    got = sim._stroke_blade_efficiency(np.array([0.0]), states)

    blade = sim._liftdrag[0]
    oar = sim._oars[0]
    locks = single.rig.seats[0].oarlocks
    weighted = total = 0.0
    for lock in locks:
        u = np.array([4.3, 0.0])
        f_n, _ = blade.loads(angle, rate, u, int(lock.side))
        w_n, _ = blade.relative_velocity(angle, rate, u, int(lock.side))
        eff = min(max(1.0 - abs(w_n) / abs(oar.outboard * rate), 0.0), 1.0)
        weighted += abs(f_n) * eff
        total += abs(f_n)
    assert got == pytest.approx(weighted / total, rel=1e-12)


def test_the_figure_draws_both_tier_two_load_components(single):
    """It used to refuse a tier 2 run rather than draw the wrong load. It now
    draws the simulator own loads: normal and tangential, the tangential one
    not zero, and a tier 1 trace still carries none."""
    from coxswain.core.state import State as _State
    from coxswain.viz.bladepath import dynamic_trace

    sim = DynamicOarSimulator(single, peak_torque=400.0, blade_law="liftdrag")
    run = sim.run_strokes(1, surge_speed=4.0)
    trace = dynamic_trace(sim, run, "tier 2")
    assert trace.has_tangential
    assert trace.tangential.shape == trace.load.shape
    assert np.abs(trace.tangential).max() > 1.0
    assert np.allclose(np.linalg.norm(trace.tangential_direction, axis=1), 1.0)

    # The first drive sample, checked against the simulator itself.
    n = sim.n_oar_states
    oar = sim._oars[0]
    states = run.last_states
    k = int(np.flatnonzero(states[STATE_SIZE] > oar.finish_angle)[0])
    lock = single.rig.seats[0].oarlocks[0]
    f_n, f_t = sim._blade_loads(0, float(states[STATE_SIZE, k]),
                                float(states[STATE_SIZE + n, k]),
                                _State.from_vector(states[:STATE_SIZE, k]),
                                lock)
    assert trace.load[0] == pytest.approx(f_n, rel=1e-12, abs=1e-9)
    assert trace.tangential[0] == pytest.approx(f_t, rel=1e-12, abs=1e-9)

    tier_one = DynamicOarSimulator(single, peak_torque=400.0)
    plain = dynamic_trace(tier_one, tier_one.run_strokes(1, surge_speed=4.0),
                          "tier 1")
    assert not plain.has_tangential
