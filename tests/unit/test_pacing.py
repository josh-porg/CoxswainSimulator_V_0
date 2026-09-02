"""Course-specific pacing, and the two bugs that made it lie.

The headline claim -- that a variable power schedule beats a flat one on a
course with varying current -- is easy to produce accidentally, because
schedules that overspend the anaerobic reserve are fast.  The first
version of this module reported a 16 s saving that was a crew 15 kJ
overdrawn.  Most of what is pinned here is therefore about *feasibility*,
not about speed.

The still-water case is the load-bearing test: with no current the
optimality condition collapses to constant power, so any amplitude the
optimiser finds there is an artefact.
"""

import numpy as np
import pytest

from coxswain.crew.exertion import ROWER_ANAEROBIC_WORK, ROWER_CRITICAL_POWER
from coxswain.crew.pacing import CoursePacing, CourseSegment

#: A quadratic resistance calibrated so a masters eight sits near
#: 4.23 m/s at about 313 W per rower at the gate.  Analytic on purpose:
#: these tests are about the pacing algebra, not the hull.
DRAG = 0.80 * 313.0 * 8 / 4.23 ** 3


def resistance(speed):
    return DRAG * speed * speed


def pacing(currents, length=1000.0, **kwargs):
    segments = [CourseSegment(length=length, current=c) for c in currents]
    return CoursePacing(segments, resistance, **kwargs)


# --------------------------------------------------------------------------
# the still-water collapse
# --------------------------------------------------------------------------
def test_still_water_wants_constant_power():
    """With no current every ``k`` is 1 and the optimum is flat.

    This is the classical result and the check that no course dependence
    has been smuggled in.  An optimiser that finds a saving here is
    exploiting the solver, not the river.
    """
    plan, amplitude = pacing([0.0] * 5).optimise()
    assert amplitude == pytest.approx(0.0, abs=1e-9)
    assert plan.powers.std() == pytest.approx(0.0, abs=1e-9)


def test_uniform_current_also_wants_constant_power():
    """A current that never changes is not a course feature."""
    _plan, amplitude = pacing([-0.3] * 5).optimise()
    assert amplitude == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# the direction of the effect
# --------------------------------------------------------------------------
def test_power_rises_where_the_current_is_most_adverse():
    """Push hardest where you are slowest -- the counter-intuitive half.

    Extra watts in slow water buy more *seconds* than the same watts in
    fast water, even though they buy fewer metres per second.
    """
    currents = [-0.15, -0.45, -0.20, -0.55, -0.25]
    model = pacing(currents)
    plan, amplitude = model.optimise(span=150.0, samples=61)
    assert amplitude > 1.0
    hardest = int(np.argmin(currents))        # most adverse
    easiest = int(np.argmax(currents))        # least adverse
    assert plan.powers[hardest] == plan.powers.max()
    assert plan.powers[easiest] == plan.powers.min()


def test_power_ordering_follows_the_current_ordering():
    """The schedule is monotone in the current, segment by segment."""
    currents = [-0.10, -0.30, -0.50, -0.70]
    plan, _amplitude = pacing(currents).optimise(span=150.0, samples=61)
    assert np.all(np.diff(plan.powers) > 0.0)


def test_a_helping_current_is_eased_not_pushed():
    """Sign matters: positive current means the river is on your side."""
    plan, _ = pacing([0.4, -0.4]).optimise(span=150.0, samples=61)
    assert plan.powers[0] < plan.powers[1]


# --------------------------------------------------------------------------
# feasibility -- where the first version went wrong
# --------------------------------------------------------------------------
def test_the_optimum_never_goes_into_anaerobic_deficit():
    """The reserve may be emptied.  It may not be overdrawn.

    Pinned because the fast-looking schedules are exactly the illegal
    ones, so a search that does not enforce this will select for it.
    """
    plan, _ = pacing([-0.15, -0.45, -0.20, -0.55, -0.25]).optimise(
        span=200.0, samples=61)
    assert plan.reserve.min() > -1.0


def test_the_optimum_actually_spends_the_reserve():
    """Finishing with the reserve intact is not a raced boat."""
    plan, _ = pacing([-0.15, -0.45, -0.20, -0.55, -0.25]).optimise(
        span=200.0, samples=61)
    assert plan.reserve.min() < 1.0


def test_the_answer_does_not_depend_on_the_search_grid():
    """A physical optimum cannot move when the sampling changes.

    It did, before the bisection replaced a Newton step that oscillated
    across the root: coarser grids found *better* times because they
    landed on schedules the balance solver had failed to make legal.
    """
    model = pacing([-0.15, -0.45, -0.20, -0.55, -0.25])
    amplitudes = [model.optimise(span=span, samples=n)[1]
                  for span, n in ((60.0, 41), (120.0, 41), (200.0, 41),
                                  (320.0, 21))]
    assert max(amplitudes) - min(amplitudes) < 0.2


def test_a_spent_reserve_does_not_refill_faster_than_a_full_one():
    """Recovery is measured from a floor of zero, not from the deficit.

    Refilling from a negative balance gave a bigger gap and so a bigger
    refill -- a crew that had blown up recovering faster than one that had
    not.  That is what made the reserve non-monotone in power.
    """
    model = pacing([0.0, 0.0])
    overdrawn = model.evaluate([ROWER_CRITICAL_POWER + 400.0,
                                ROWER_CRITICAL_POWER - 50.0])
    gained = overdrawn.reserve[2] - overdrawn.reserve[1]
    assert gained <= ROWER_ANAEROBIC_WORK
    assert gained > 0.0


# --------------------------------------------------------------------------
# the flat baseline
# --------------------------------------------------------------------------
def test_flat_power_is_the_two_parameter_result():
    """``P = CP + W'/T``, solved self-consistently with the speed."""
    model = pacing([0.0] * 5)
    power = model.flat_power()
    plan = model.evaluate(np.full(5, power))
    assert power == pytest.approx(
        ROWER_CRITICAL_POWER + ROWER_ANAEROBIC_WORK / plan.total_time,
        rel=1e-3)


def test_flat_power_sits_just_above_critical_power():
    """A nineteen-minute race is rowed about 3% over CP, not 30%.

    Worth pinning because it bounds every tactical claim built on top:
    the reserve buys a handful of surges, not a harder race.
    """
    power = pacing([0.0] * 5).flat_power()
    assert 1.01 < power / ROWER_CRITICAL_POWER < 1.06


def test_more_power_is_never_slower():
    model = pacing([-0.3] * 3)
    slow = model.evaluate([300.0] * 3).total_time
    fast = model.evaluate([340.0] * 3).total_time
    assert fast < slow


def test_an_adverse_current_costs_time():
    still = pacing([0.0] * 3).evaluate([313.0] * 3).total_time
    against = pacing([-0.4] * 3).evaluate([313.0] * 3).total_time
    assert against > still


def test_evaluate_rejects_a_wrong_length_schedule():
    with pytest.raises(ValueError, match="expected 3 powers"):
        pacing([0.0] * 3).evaluate([300.0, 310.0])
