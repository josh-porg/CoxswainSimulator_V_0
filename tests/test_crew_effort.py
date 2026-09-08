r"""The crew's skill, and what they have left to give.

Two things arrive together here because they are the same question from
opposite ends: how consistent a crew is stroke to stroke, and how long
they can hold what the coxswain asks for.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.crew.exertion import (ROWER_CRITICAL_POWER, WPrimeBalance,
                                    mean_handle_power, optimal_pace)
from coxswain.crew.variability import (ELITE, JUNIOR, SKILL_ANCHORS,
                                       for_skill, skill_label)


def test_the_measured_skill_points_are_not_interpolated_away():
    """Elite and junior are measured [K-VAR].  A slider set to either
    must produce the measured number exactly, not whatever a smooth
    curve happens to give at that point -- otherwise the calibration is
    quietly replaced by the interpolation."""
    assert for_skill(0.75).power_sigma == pytest.approx(ELITE.power_sigma)
    assert for_skill(0.75).timing_sigma == pytest.approx(ELITE.timing_sigma)
    assert for_skill(0.35).power_sigma == pytest.approx(JUNIOR.power_sigma)
    assert for_skill(0.35).timing_sigma == pytest.approx(JUNIOR.timing_sigma)


def test_skill_runs_the_right_way_and_ideal_is_perfect():
    """More skill is less scatter, monotonically, and the top of the
    slider is the uniformity every result in this project was computed
    with before variability existed."""
    sigmas = [for_skill(s).power_sigma for s in np.linspace(0.0, 1.0, 21)]
    assert all(b <= a + 1e-12 for a, b in zip(sigmas, sigmas[1:]))
    assert for_skill(1.0).power_sigma == 0.0
    assert for_skill(1.0).timing_sigma == 0.0
    assert skill_label(1.0) == "ideal"
    assert skill_label(0.0) == "novice"


def test_timing_degrades_faster_than_power():
    """Across the measured presets the timing-to-power ratio rises from
    0.35 to 0.59.  A less experienced crew is disproportionately worse
    at going together than at pulling evenly, and tying timing to power
    by one ratio would flatten that."""
    ratio_elite = ELITE.timing_sigma / ELITE.power_sigma
    ratio_junior = JUNIOR.timing_sigma / JUNIOR.power_sigma
    assert ratio_junior > ratio_elite


def test_race_pace_spends_the_reserve_exactly_at_the_line():
    """``P = CP + W'/T`` is the whole point of the two-parameter model:
    row it and you cross the line with nothing left and nothing wasted."""
    reserve = WPrimeBalance()
    duration = 360.0
    power = optimal_pace(duration)
    remaining, t, dt = reserve.capacity, 0.0, 0.5
    while remaining > 0.0 and t < 10 * duration:
        remaining = reserve.step(remaining, power, dt)
        t += dt
    assert t == pytest.approx(duration, rel=0.02)


def test_a_harder_call_cannot_be_held_as_long():
    """What a coxswain is actually trading."""
    reserve = WPrimeBalance()
    base = optimal_pace(360.0)

    def lasts(power):
        remaining, t, dt = reserve.capacity, 0.0, 0.5
        while remaining > 0.0 and t < 4000.0:
            remaining = reserve.step(remaining, power, dt)
            t += dt
        return t

    assert lasts(base * 1.30) < lasts(base * 1.10) < lasts(base)
    # Below critical power there is no reserve to spend at all.
    assert lasts(ROWER_CRITICAL_POWER * 0.95) >= 4000.0


def test_the_stepwise_reserve_agrees_with_the_batch_one():
    """The real-time seam must not be a second, differing model -- the
    same trap the simulator's own step()/run() split has to avoid."""
    reserve = WPrimeBalance()
    power, dt, n = 340.0, 0.25, 400
    batch = reserve.integrate(np.full(n, power), dt)[-1]
    remaining = reserve.capacity
    for _ in range(n):
        remaining = reserve.step(remaining, power, dt)
    assert remaining == pytest.approx(float(batch), rel=1e-9, abs=1e-6)


def test_the_catalog_rows_above_what_anyone_can_hold():
    """Worth pinning, because it is the reason race pace had to be
    computed rather than assumed: the default force scale is a power no
    crew sustains, and nothing noticed until the reserve was tracked."""
    from coxswain.boats import catalog

    boat = catalog.eight(rate=32, rower_mass=75, rower_stature=1.83,
                         coxswain_mass=55)
    watts = mean_handle_power(boat, samples=180)
    assert watts > ROWER_CRITICAL_POWER * 1.3, watts
    # And a crew asked for it empties in a couple of minutes.
    assert WPrimeBalance().endurance(watts) < 180.0
