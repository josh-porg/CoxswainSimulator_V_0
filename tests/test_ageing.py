r"""Age-scaled physiology, and the reserve coming from the crew.

The trainer used a literature CP of 302.7 W and W' of 11.4 kJ for every
crew.  Those are means for young male athletes; this project's own
Women's Veteran four pulls about 126 W, so the reserve model believed
they were rowing at 42% of their aerobic ceiling and they could never
tire -- "FADING" could not fire, and the optimal pace was answering a
question about somebody else.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.crew.ageing import (AGE_REFERENCE, W_PRIME_FLOOR,
                                  critical_power_from_erg, crew_physiology,
                                  w_prime_factor, w_prime_for_age)
from coxswain.crew.exertion import ROWER_ANAEROBIC_WORK, ROWER_CRITICAL_POWER


def test_age_does_not_touch_power_only_the_reserve():
    """The erg score IS the measurement; age must not rescale it.

    Two crews with the same ergs and different ages get the same CP to
    within the small W'/t term, and differ in W'.
    """
    watts, seconds = [130.0] * 4, [1340.0] * 4
    young_cp, young_w = crew_physiology(watts, seconds, [30] * 4)
    old_cp, old_w = crew_physiology(watts, seconds, [65] * 4)
    assert old_w < young_w * 0.75
    # CP moves only by the reserve term, which is a few watts
    assert abs(old_cp - young_cp) < 5.0
    assert young_cp < 130.0, "CP sits just under the 5k power"


def test_the_factor_holds_at_and_below_the_reference_age():
    assert w_prime_factor(AGE_REFERENCE) == 1.0
    assert w_prime_factor(22) == 1.0, "not MORE than the reference"
    assert w_prime_factor(None) == 1.0
    assert w_prime_factor("") == 1.0
    assert w_prime_factor(0) == 1.0, "a nonsense age ages nothing"
    assert w_prime_factor(150) == 1.0


def test_the_factor_falls_with_age_and_stops_falling():
    assert w_prime_factor(40) == pytest.approx(0.90)
    assert w_prime_factor(60) == pytest.approx(0.70)
    ages = [30, 40, 50, 60, 70, 80, 100]
    factors = [w_prime_factor(a) for a in ages]
    assert factors == sorted(factors, reverse=True)
    assert min(factors) >= W_PRIME_FLOOR
    assert w_prime_for_age(60) == pytest.approx(ROWER_ANAEROBIC_WORK * 0.70)


def test_critical_power_is_the_two_parameter_model_read_backwards():
    """``P = CP + W'/t``, so ``CP = P - W'/t``.  Not a new assumption."""
    w_prime = 11400.0
    for watts, seconds in ((130.0, 1340.0), (250.0, 1150.0), (300.0, 1200.0)):
        cp = critical_power_from_erg(watts, seconds, w_prime)
        assert cp == pytest.approx(watts - w_prime / seconds)
        # the model round-trips: holding CP + W'/t for t gives the erg
        assert cp + w_prime / seconds == pytest.approx(watts)
    # a short piece cannot drive CP negative
    assert critical_power_from_erg(200.0, 30.0, w_prime) == pytest.approx(100.0)


def test_a_crew_with_no_numbers_behaves_exactly_as_before():
    cp, w_prime = crew_physiology(None, None, None)
    assert cp == pytest.approx(ROWER_CRITICAL_POWER)
    assert w_prime == pytest.approx(ROWER_ANAEROBIC_WORK)
    # half a lineup filled in is still usable
    cp, w_prime = crew_physiology([130.0, None], [1340.0, None], [62, None])
    assert cp < ROWER_CRITICAL_POWER
    assert 0.0 < w_prime < ROWER_ANAEROBIC_WORK


def test_the_projects_own_four_gets_its_own_reserve():
    """The number that motivated this: 302.7 W for a crew pulling 126."""
    from coxswain.viz.menu import build_boat
    from coxswain.viz.rigview import PRESETS

    lineup = PRESETS["HOCR 4+"]()
    boat, _made = build_boat("4+", 30.0, lineup=lineup)
    cp, w_prime = boat.crew_physiology
    mean_erg = float(np.mean([r.watts for r in lineup.rowers]))
    assert 110.0 < cp < mean_erg, (cp, mean_erg)
    assert cp < 0.5 * ROWER_CRITICAL_POWER, "the literature CP was 2.4x theirs"
    # Veteran 60+: the reserve is well down on the young reference
    assert w_prime == pytest.approx(ROWER_ANAEROBIC_WORK * w_prime_factor(62))


def test_the_trainer_uses_the_crews_reserve_when_it_has_one():
    import os

    text = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts", "fpv.py"),
        encoding="utf-8").read()
    assert 'getattr(boat, "crew_physiology", (None, None))' in text
    assert "WPrimeBalance(critical_power=_cp, capacity=_wprime)" in text
    assert "if _cp else WPrimeBalance()" in text, "literature pair otherwise"
