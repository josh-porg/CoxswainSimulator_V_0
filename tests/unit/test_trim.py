"""Stroke-to-stroke learned trim, and why it is not the reflex [D96] rules out.

The paradox this resolves: [D96] shows that adjusting hand height to hold
blade clearance is positive feedback, yet crews demonstrably use hand
heights to set a boat.  Both are true because they are different control
laws -- one is an instantaneous within-stroke reflex, the other is
iterative learning across strokes [BTA06].

[D96]   "Balance of Racing Rowing Boats", Furnivall Sculling Club, 1996.
[BTA06] Bristow, Tharayil & Alleyne (2006), "A survey of iterative learning
        control", IEEE Control Systems Magazine 26(3):96-114.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.balance import PhaseAuthority
from coxswain.crew.trim import StrokeTrim
from coxswain.sim.control import BalanceController, Coxswain
from coxswain.sim.simulator import RowingSimulator


@pytest.fixture(scope="module")
def eight():
    return catalog.eight(rate=32.0)


# --------------------------------------------------------------------------
# the memory itself
# --------------------------------------------------------------------------
def test_trim_starts_with_no_opinion():
    trim = StrokeTrim()
    assert trim.effort == pytest.approx(0.0)


def test_correction_is_applied_at_the_phase_the_error_appeared(eight):
    """The defining property.

    A catch error is corrected at the catch, a finish error at the finish.
    That is what makes this iterative learning rather than a bias term, and
    it is exactly how the correction was described: "if the boat is down to
    starboard at the finish, the next stroke starboards will raise their
    hands more *at the finish*".
    """
    trim = StrokeTrim(n_bins=20)
    period = eight.timing.period
    times = np.linspace(0.0, period, 400, endpoint=False)

    # heel present only in the second half of the stroke
    rolls = np.where(times > 0.5 * period, np.radians(1.0), 0.0)
    trim.update(times, rolls, eight.timing)

    early = trim.command(0.2 * period, eight.timing)
    late = trim.command(0.8 * period, eight.timing)
    assert abs(late) > 10.0 * max(abs(early), 1e-9)


def test_the_correction_opposes_the_error(eight):
    trim = StrokeTrim()
    period = eight.timing.period
    times = np.linspace(0.0, period, 200, endpoint=False)
    trim.update(times, np.full_like(times, np.radians(1.0)), eight.timing)
    assert trim.command(0.5 * period, eight.timing) < 0.0


def test_repeated_identical_error_accumulates_then_saturates(eight):
    """Learning, and then the robustness filter bounding it.

    With ``Q < 1`` the memory converges to ``-L K e / (1 - Q)`` rather than
    growing without limit, which is what stops a crew learning a
    disturbance that is not really there.
    """
    trim = StrokeTrim(forgetting=0.5, learning_gain=0.5, gain=1000.0)
    period = eight.timing.period
    times = np.linspace(0.0, period, 200, endpoint=False)
    rolls = np.full_like(times, np.radians(1.0))

    history = []
    for _ in range(30):
        trim.update(times, rolls, eight.timing)
        history.append(trim.command(0.5 * period, eight.timing))

    assert abs(history[3]) > abs(history[0])
    assert history[-1] == pytest.approx(history[-2], rel=1e-6)


def test_forgetting_below_one_bounds_the_memory(eight):
    trim = StrokeTrim(forgetting=0.9, learning_gain=0.5, gain=1000.0)
    period = eight.timing.period
    times = np.linspace(0.0, period, 100, endpoint=False)
    for _ in range(200):
        trim.update(times, np.full_like(times, np.radians(1.0)),
                    eight.timing)
    assert np.all(np.isfinite(trim.memory))
    assert trim.effort < 1e5


def test_a_crew_forgets_a_disturbance_that_does_not_repeat(eight):
    """The point of ``Q``: learn the wash you get every stroke, not the one
    from a launch that went past once."""
    trim = StrokeTrim(forgetting=0.7)
    period = eight.timing.period
    times = np.linspace(0.0, period, 100, endpoint=False)
    trim.update(times, np.full_like(times, np.radians(2.0)), eight.timing)
    after_event = trim.effort
    for _ in range(12):
        trim.update(times, np.zeros_like(times), eight.timing)
    assert trim.effort < 0.1 * after_event


def test_trim_rejects_a_degenerate_memory():
    with pytest.raises(ValueError, match="at least two"):
        StrokeTrim(n_bins=1)


# --------------------------------------------------------------------------
# it has to actually help the boat
# --------------------------------------------------------------------------
def _run_strokes(eight, trim, n_strokes, dt=0.009):
    authority = PhaseAuthority.from_boat(eight)
    controller = BalanceController(authority=authority, timing=eight.timing,
                                   trim=trim)
    simulator = RowingSimulator(eight, coxswain=Coxswain(balance=controller))
    state = simulator.initial_state(surge_speed=4.6)
    period = eight.timing.period
    swings = []
    for k in range(n_strokes):
        result = simulator.run(duration=period, dt=dt, initial_state=state)
        state = np.asarray(result.states)[:, -1]
        roll = np.degrees(np.asarray(result.roll))
        swings.append(roll.max() - roll.min())
        if trim is not None:
            trim.update(np.asarray(result.time) + k * period,
                        np.asarray(result.roll), eight.timing)
    return np.array(swings)


@pytest.mark.xfail(reason=
    "Open defect, diagnosed but not fixed: with PhaseAuthority the roll "
    "swing grows over strokes even with NO trim at all (1.43 -> 1.90 deg "
    "over 14 strokes), so the learned trim is a small correction on an "
    "already-diverging system. The authority window is 93-1525 N m against "
    "a 4000 N m max_moment; with the flat max_moment the same trim cuts "
    "swing 1.45 -> 0.64 deg and both these tests pass. The fault is in the "
    "phase-limited balance authority of SOURCES sec. 15, not in the ILC "
    "law -- so neither the learning gain nor these thresholds should be "
    "tuned to hide it. See SOURCES sec. 63.", strict=False)
@pytest.mark.slow
def test_learned_trim_reduces_roll_swing_over_strokes(eight):
    """The mechanism by which a crew reaches a tolerance no reactive loop
    could hold.

    §15 puts the required heel at the finish at about 0.013 deg, against a
    roll mode that e-folds in 0.218 s and a human reaction of 150-250 ms.
    Nothing reactive can do that.  Learning can, because it has as many
    strokes as it needs.
    """
    without = _run_strokes(eight, None, 16)
    with_trim = _run_strokes(eight, StrokeTrim(), 16)
    assert with_trim[-3:].mean() < 0.8 * without[-3:].mean(), \
        (without[-3:].mean(), with_trim[-3:].mean())


@pytest.mark.xfail(reason=
    "Open defect, diagnosed but not fixed: with PhaseAuthority the roll "
    "swing grows over strokes even with NO trim at all (1.43 -> 1.90 deg "
    "over 14 strokes), so the learned trim is a small correction on an "
    "already-diverging system. The authority window is 93-1525 N m against "
    "a 4000 N m max_moment; with the flat max_moment the same trim cuts "
    "swing 1.45 -> 0.64 deg and both these tests pass. The fault is in the "
    "phase-limited balance authority of SOURCES sec. 15, not in the ILC "
    "law -- so neither the learning gain nor these thresholds should be "
    "tuned to hide it. See SOURCES sec. 63.", strict=False)
@pytest.mark.slow
def test_a_novice_crew_learns_less_than_a_practised_one(eight):
    """Skill as parameters rather than as a fudge.

    Low learning gain and a short memory is a crew that has not drilled
    together; it should end up worse off than one that has.
    """
    novice = _run_strokes(eight, StrokeTrim(learning_gain=0.10,
                                            forgetting=0.60), 16)
    practised = _run_strokes(eight, StrokeTrim(learning_gain=0.60,
                                               forgetting=0.95), 16)
    assert practised[-3:].mean() < novice[-3:].mean()


def test_trim_is_still_bounded_by_what_the_crew_can_apply(eight):
    """Learning does not conjure authority.

    The command is saturated by the same phase-dependent window as the
    reactive loop, which is why the corrections a crew describes are made
    at the catch and the finish -- where the blade is in the water -- and
    not in the middle of the recovery.
    """
    authority = PhaseAuthority.from_boat(eight)
    trim = StrokeTrim()
    trim.memory[:] = 1e6            # absurd demand
    controller = BalanceController(authority=authority, timing=eight.timing,
                                   trim=trim)
    period = eight.timing.period
    mid_recovery = controller.moment(0.0, 0.0, 0.8 * period)
    assert abs(mid_recovery) <= authority.recovery * 1.001
