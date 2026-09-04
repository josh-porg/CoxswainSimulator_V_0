"""Does the head race belong to the time-trial literature or not?

The pacing literature splits on whether position or the clock decides.
These tests pin the quantities that settle it, so the conclusion in
SOURCES sec. 88 cannot drift without something failing.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "scripts"))

from coxswain.crew.exertion import (ROWER_ANAEROBIC_WORK,  # noqa: E402
                                    ROWER_CRITICAL_POWER)
from coxswain.crew.pacing import CoursePacing, CourseSegment  # noqa: E402

strategy = pytest.importorskip("pacing_strategy")

K = 0.8 * 313.0 * 8 / 4.23 ** 3


def resistance(v):
    return K * v * v


def model(n=12, length=402.0):
    return CoursePacing([CourseSegment(length) for _ in range(n)],
                        resistance)


# --------------------------------------------------------------------------
def test_the_reserve_is_worth_far_less_over_twenty_minutes():
    """The number that decides which literature applies.

    Tactical pacing is spending W'.  Over a 2 km final it is a real
    fraction of what a crew spends; over twenty minutes it is not, and
    that is why 2 km advice does not carry across.
    """
    rows = dict((label, watts) for label, _s, watts, _p
                in strategy.reserve_worth([("2k", 370.0),
                                           ("head", 1230.0)]))
    assert rows["2k"] / rows["head"] > 3.0
    assert rows["head"] / ROWER_CRITICAL_POWER < 0.05


def test_reserve_worth_is_just_w_prime_over_time():
    label, seconds, watts, percent = strategy.reserve_worth(
        [("x", 600.0)])[0]
    assert watts == pytest.approx(ROWER_ANAEROBIC_WORK / 600.0)
    assert percent == pytest.approx(
        100.0 * watts / ROWER_CRITICAL_POWER)


def test_chasing_costs_time():
    """Surging early and paying later is slower at equal work."""
    course = model()
    baseline = course.flat_power()
    lost, _plan = strategy.chase_cost(course, baseline, 0.10, 3)
    assert lost > 0.0


def test_chasing_harder_costs_more():
    course = model()
    baseline = course.flat_power()
    gentle, _a = strategy.chase_cost(course, baseline, 0.05, 3)
    hard, _b = strategy.chase_cost(course, baseline, 0.15, 3)
    assert hard > gentle


def test_not_chasing_costs_nothing():
    course = model()
    baseline = course.flat_power()
    lost, _plan = strategy.chase_cost(course, baseline, 0.0, 3)
    assert lost == pytest.approx(0.0, abs=1e-9)


def test_the_chase_is_priced_at_equal_work():
    """Otherwise it would be measuring effort, not distribution."""
    course = model()
    baseline = course.flat_power()
    reference = course.evaluate(np.full(len(course.segments), baseline))
    _lost, plan = strategy.chase_cost(course, baseline, 0.10, 3)
    assert float(np.average(plan.powers,
                            weights=reference.durations)) == \
        pytest.approx(baseline, rel=1e-9)


def test_chasing_is_expensive_but_not_ruinous():
    """The claim in the write-up: seconds, not minutes.

    If this ever fails it means either the model or the conclusion has
    moved, and the write-up says the wrong thing either way.
    """
    course = model()
    baseline = course.flat_power()
    lost, _plan = strategy.chase_cost(course, baseline, 0.10, 3)
    assert 0.1 < lost < 30.0
