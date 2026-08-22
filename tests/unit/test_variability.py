"""Stroke-to-stroke variation in crew power and timing.

[K-VAR] Kleshnev, "Rowing Science: New Analysis of Variability of Rower's
        Technique", parts 1-3, row2k.  Elite sculler force variation 2.3%
        against a junior's 5.1%; work per stroke 1.3% against 4.7%.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.variability import (CLUB, ELITE, JUNIOR, CrewVariability)


@pytest.fixture
def eight():
    return catalog.eight(rate=32.0)


# --------------------------------------------------------------------------
# the presets carry the published numbers
# --------------------------------------------------------------------------
def test_the_presets_bracket_the_measured_range():
    """[K-VAR] measured 2.3% for an elite sculler and 5.1% for a junior."""
    assert ELITE.power_sigma == pytest.approx(0.023)
    assert JUNIOR.power_sigma == pytest.approx(0.051)
    assert ELITE.power_sigma < CLUB.power_sigma < JUNIOR.power_sigma


def test_negative_scatter_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        CrewVariability(power_sigma=-0.1)


# --------------------------------------------------------------------------
# the draw
# --------------------------------------------------------------------------
def test_a_draw_is_centred_on_the_nominal_crew():
    variability = CrewVariability(power_sigma=0.03, timing_sigma=0.01,
                                  seed=3)
    powers = np.concatenate([variability.draw(8)[0] for _ in range(400)])
    timings = np.concatenate([variability.draw(8)[1] for _ in range(400)])
    assert powers.mean() == pytest.approx(1.0, abs=0.01)
    assert timings.mean() == pytest.approx(0.0, abs=0.002)


def test_the_scatter_matches_what_was_asked_for():
    variability = CrewVariability(power_sigma=0.04, timing_sigma=0.02,
                                  seed=11)
    powers = np.concatenate([variability.draw(8)[0] for _ in range(600)])
    timings = np.concatenate([variability.draw(8)[1] for _ in range(600)])
    assert powers.std() == pytest.approx(0.04, rel=0.15)
    assert timings.std() == pytest.approx(0.02, rel=0.15)


def test_a_rower_never_pulls_negative():
    """An unbounded Gaussian eventually would."""
    variability = CrewVariability(power_sigma=0.9, seed=5)
    for _ in range(200):
        assert np.all(variability.draw(8)[0] >= 0.0)


def test_draws_are_reproducible_from_the_seed():
    a = CrewVariability(seed=42).draw(8)
    b = CrewVariability(seed=42).draw(8)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_reset_rewinds_the_stream():
    variability = CrewVariability(seed=7)
    first = variability.draw(8)[0]
    variability.draw(8)
    variability.reset()
    np.testing.assert_allclose(variability.draw(8)[0], first)


def test_persistent_bias_is_distinct_from_scatter():
    """A rower who is *consistently* strong is a different problem from one
    who is inconsistent: bias can be rigged or seated around, scatter
    cannot."""
    variability = CrewVariability(power_sigma=0.0, power_bias_sigma=0.05,
                                  timing_sigma=0.0, seed=2)
    first = variability.draw(8)[0]
    second = variability.draw(8)[0]
    np.testing.assert_allclose(first, second)
    assert first.std() > 0.01


def test_zero_scatter_gives_a_perfect_crew():
    variability = CrewVariability(power_sigma=0.0, timing_sigma=0.0)
    power, timing = variability.draw(8)
    np.testing.assert_allclose(power, np.ones(8))
    np.testing.assert_allclose(timing, np.zeros(8))


# --------------------------------------------------------------------------
# it reaches the boat
# --------------------------------------------------------------------------
def test_apply_sets_both_channels_on_the_boat(eight):
    variability = CrewVariability(power_sigma=0.05, timing_sigma=0.02,
                                  seed=1)
    variability.apply(eight)
    assert not np.allclose(eight.power_scales, np.ones(eight.n_seats))
    assert not np.allclose(eight.phase_offsets, np.zeros(eight.n_seats))


def test_timing_is_converted_to_stroke_fractions(eight):
    """``phase_offsets`` is a fraction of a stroke; the draw is in seconds.

    Getting this wrong by a factor of the period would be a factor of two
    error at rate 32 and would not obviously look wrong.
    """
    variability = CrewVariability(power_sigma=0.0, timing_sigma=0.0,
                                  timing_bias_sigma=0.05, seed=4)
    _, timing = variability.draw(eight.n_seats)
    variability.reset()
    variability.apply(eight)
    np.testing.assert_allclose(eight.phase_offsets,
                               timing / eight.timing.period, rtol=1e-9)


def test_a_scattered_crew_produces_a_yaw_disturbance(eight):
    """Power imbalance between the sides is a standing steering bias --
    which is exactly the disturbance the steering study corrects."""
    from coxswain.core.state import State
    from coxswain.sim.simulator import RowingSimulator

    simulator = RowingSimulator(eight)
    state = State.from_vector(simulator.initial_state(surge_speed=4.6))
    baseline = float(np.asarray(simulator.breakdown(0.2, state).oar_moment)[2])

    sides = np.array([s.oarlocks[0].side for s in eight.rig.seats],
                     dtype=float)
    eight.power_scales = np.where(sides > 0, 1.05, 0.95)
    scattered = float(np.asarray(
        RowingSimulator(eight).breakdown(0.2, state).oar_moment)[2])

    assert abs(scattered - baseline) > 10.0


def test_power_scales_reject_a_bad_shape(eight):
    with pytest.raises(ValueError, match="one entry per seat"):
        eight.power_scales = np.ones(3)


def test_power_scales_reject_negative_values(eight):
    with pytest.raises(ValueError, match="non-negative"):
        eight.power_scales = -np.ones(eight.n_seats)
