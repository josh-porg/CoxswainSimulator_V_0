"""Pacing evenness, against Xia's empirical result and against theory.

Xia (2025) analysed 179 elite women's eight Final A races and found that
**winning crews pace more evenly than losing ones** -- first place showed a
balanced distribution across the four 500 m segments while sixth place
fluctuated far more.  He reports it as a pattern and never as a time.
These tests pin the machinery that prices it.
"""

import numpy as np
import pytest

from coxswain.crew.pacing import CoursePacing, CourseSegment

K = 0.8 * 313.0 * 8 / 4.23 ** 3


def resistance(v):
    return K * v * v


def uniform(n=4, length=500.0):
    """A lane: no current, no wind, deep water.  Xia's condition."""
    return CoursePacing([CourseSegment(length) for _ in range(n)],
                        resistance)


# --------------------------------------------------------------------------
# the speed coefficient
# --------------------------------------------------------------------------
def test_speed_coefficient_averages_to_one():
    """It is normalised by the boat's own race mean, so it must."""
    model = uniform()
    plan = model.evaluate(np.array([300.0, 340.0, 310.0, 330.0]))
    assert float(np.average(plan.speed_coefficient(),
                            weights=plan.durations)) == pytest.approx(
        1.0, rel=1e-9)


def test_flat_pacing_has_zero_scatter():
    model = uniform()
    plan = model.evaluate(np.full(4, model.flat_power()))
    assert plan.pacing_scatter() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(plan.speed_coefficient(), 1.0, atol=1e-12)


def test_the_coefficient_is_blind_to_conditions():
    """Xia's whole reason for using it: it strips the day out.

    The same power schedule in a headwind is slower everywhere, so raw
    splits change -- but the *distribution* does not, and the coefficient
    must report that.
    """
    powers = np.array([300.0, 340.0, 310.0, 330.0])
    calm = uniform().evaluate(powers)
    blown = CoursePacing([CourseSegment(500.0, headwind=4.0)
                          for _ in range(4)], resistance).evaluate(powers)
    assert blown.total_time > calm.total_time      # genuinely slower
    np.testing.assert_allclose(blown.speed_coefficient(),
                               calm.speed_coefficient(), rtol=0.02)


# --------------------------------------------------------------------------
# what scatter costs
# --------------------------------------------------------------------------
def test_scatter_costs_time():
    model = uniform()
    lost, achieved = model.price_scatter(0.10, samples=200)
    assert lost > 0.0
    assert achieved > 0.0


def test_zero_scatter_costs_nothing():
    lost, achieved = uniform().price_scatter(0.0, samples=20)
    assert lost == pytest.approx(0.0, abs=1e-9)
    assert achieved == pytest.approx(0.0, abs=1e-12)


def test_the_cost_is_quadratic_in_the_scatter():
    """Convexity, so doubling the spread roughly quadruples the loss."""
    model = uniform()
    small, _ = model.price_scatter(0.05, samples=400)
    large, _ = model.price_scatter(0.10, samples=400)
    assert large / small == pytest.approx(4.0, rel=0.15)


def test_the_cost_matches_the_analytic_convexity_estimate():
    """The strong form: it is not merely quadratic, it is the RIGHT one.

    For ``t = L/v`` with ``v ~ P^e`` the second-order loss from spreading
    ``P`` at fixed mean is ``0.5 T e(e+1) var(P)/P^2``.  The draws here are
    mean-centred over ``n`` segments, so their realised variance is
    ``(n-1)/n`` of nominal -- with four segments, three quarters.  Missing
    that factor is an easy way to conclude the model is 25% wrong when it
    is exact.
    """
    model = uniform(n=4)
    baseline = model.flat_power()
    duration = model.evaluate(np.full(4, baseline)).total_time
    exponent = 1.0 / 3.0                      # v ~ P^(1/3) for v^2 drag

    for spread in (0.05, 0.10, 0.15):
        lost, _ = model.price_scatter(spread, samples=600)
        realised = spread ** 2 * 3.0 / 4.0
        predicted = 0.5 * duration * exponent * (exponent + 1.0) * realised
        assert lost == pytest.approx(predicted, rel=0.10)


def test_pricing_holds_total_work_fixed():
    """The comparison must be at equal work, not equal reserve.

    Balancing to spend the reserve exactly lets a segment below CP
    *recover* W', and with a 300 s recovery constant against 120 s
    segments that recovery is large enough to leave scattered schedules
    finishing with kilojoules unspent -- which would price a solver
    artefact as a pacing cost.
    """
    model = uniform()
    baseline = model.flat_power()
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 1.0, size=4)
    noise -= noise.mean()
    powers = baseline * (1.0 + 0.10 * noise)
    plan = model.evaluate(powers)
    reference = model.evaluate(np.full(4, baseline))
    assert float(np.average(powers, weights=reference.durations)) == \
        pytest.approx(baseline, rel=1e-9)
    assert plan.total_time > reference.total_time


# --------------------------------------------------------------------------
# Xia's finding, as a claim about this model
# --------------------------------------------------------------------------
def test_even_pacing_wins_a_lane_race():
    """In uniform conditions the optimum is flat -- Xia's 1st-place pattern.

    His winning crews paced evenly and his sixth-place crews did not.  In
    a lane there is no current, depth or shelter gradient to exploit, so
    the model must agree that evenness is optimal rather than merely
    tidy.
    """
    model = uniform()
    _plan, amplitude = model.optimise()
    assert amplitude == pytest.approx(0.0, abs=1e-9)


def test_a_varying_course_is_the_exception_that_proves_it():
    """On a river evenness is NOT optimal, and that is not a contradiction.

    Xia's lanes are uniform; the Charles is not.  The same physics that
    makes flat pacing optimal in a lane makes it suboptimal where the
    water varies, so a head race and a 2 km final call for different
    schedules and the model should say so.
    """
    river = CoursePacing([CourseSegment(500.0, depth=d)
                          for d in (12.0, 2.2, 12.0, 2.2)], resistance)
    _plan, amplitude = river.optimise(span=200.0, samples=61)
    assert amplitude > 0.0


# --------------------------------------------------------------------------
# the executable bound -- how the literature enters the model
# --------------------------------------------------------------------------
def test_flat_pacing_has_zero_power_scatter_by_construction():
    model = uniform()
    plan = model.evaluate(np.full(4, model.flat_power()))
    assert plan.power_scatter() == pytest.approx(0.0, abs=1e-12)


def test_on_a_varying_course_the_speed_coefficient_measures_the_COURSE():
    """Why the executable bound is on power and not on Xia's measure.

    In a buoyed lane, speed varies only because the crew made it vary, so
    Xia's speed coefficient is a clean measure of pacing choice.  On a
    river it is not: a crew holding one power the whole way still produces
    a large speed scatter, because the water changes underneath them.
    Bounding that would reject flat pacing as unexecutable.
    """
    river = CoursePacing([CourseSegment(500.0, depth=d)
                          for d in (12.0, 2.2, 12.0, 2.2)], resistance)
    flat = river.evaluate(np.full(4, river.flat_power()))
    assert flat.power_scatter() == pytest.approx(0.0, abs=1e-12)
    assert flat.pacing_scatter() > 0.02


def test_the_power_bound_restricts_the_optimum():
    river = CoursePacing([CourseSegment(500.0, depth=d)
                          for d in (12.0, 2.2, 12.0, 2.2)], resistance)
    free, _a = river.optimise(span=200.0, samples=61)
    tight, _b = river.optimise(span=200.0, samples=61,
                               max_power_scatter=0.002)
    assert tight.power_scatter() <= 0.002 + 1e-9
    assert tight.power_scatter() < free.power_scatter()
    assert tight.total_time >= free.total_time


def test_a_generous_bound_does_not_bite():
    river = CoursePacing([CourseSegment(500.0, depth=d)
                          for d in (12.0, 2.2, 12.0, 2.2)], resistance)
    free, _a = river.optimise(span=200.0, samples=61)
    loose, _b = river.optimise(span=200.0, samples=61,
                               max_power_scatter=1.0)
    assert loose.total_time == pytest.approx(free.total_time, rel=1e-9)


def test_whether_the_optimum_is_more_even_than_flat_depends_on_the_course():
    """Xia's "winners pace evenly" does NOT generalise to a river.

    This test was written to assert that the optimum is always at least
    as even as flat pacing, on the strength of Xia's result and of the
    real Charles, where optimising takes the speed coefficient from 0.0890
    down to 0.0855.  It is false in general, and the counterexample is
    this synthetic alternating-depth river, where the optimum eases hard
    in the shallows and pushes in the deep and so **amplifies** the speed
    variation.

    Both behaviours are the same rule -- spend where a watt buys the most
    seconds -- reading different courses.  Whether that ends up looking
    even is a property of the water, not a target.  Which is exactly why
    the executable bound belongs on power, where the crew's choice lives.
    """
    river = CoursePacing([CourseSegment(500.0, depth=d)
                          for d in (12.0, 2.2, 12.0, 2.2)], resistance)
    flat = river.evaluate(np.full(4, river.flat_power()))
    best, _a = river.optimise(span=200.0, samples=61)
    assert best.total_time < flat.total_time          # it is faster
    assert best.pacing_scatter() > flat.pacing_scatter()   # and less even
