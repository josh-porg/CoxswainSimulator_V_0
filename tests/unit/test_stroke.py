"""Unit tests for stroke timing and Fourier profile representation."""

import numpy as np
import pytest

from coxswain.crew.stroke import DEFAULT_HARMONICS, FourierProfile, StrokeTiming


# --------------------------------------------------------------------------
# timing -- Formaggia et al. section 5
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rate,expected_drive", [
    (24.0, 0.800),      # the fit's reference point: tau_a = 0.8 at r = 24
    (30.0, 0.756875),
    (32.0, 0.745),
    (40.0, 0.710),
])
def test_drive_duration_matches_the_published_fit(rate, expected_drive):
    """tau_a = 0.00015625 (r-24)^2 - 0.008125 (r-24) + 0.8"""
    assert StrokeTiming(rate).drive_duration == pytest.approx(expected_drive,
                                                              abs=1e-6)


@pytest.mark.parametrize("rate", [18.0, 24.0, 30.0, 36.0, 40.0])
def test_period_is_sixty_over_rate(rate):
    assert StrokeTiming(rate).period == pytest.approx(60.0 / rate)


@pytest.mark.parametrize("rate", [18.0, 24.0, 30.0, 36.0, 40.0])
def test_drive_and_recovery_sum_to_the_period(rate):
    timing = StrokeTiming(rate)
    assert (timing.drive_duration + timing.recovery_duration
            == pytest.approx(timing.period))


def test_recovery_is_longer_than_the_drive_at_low_rates():
    timing = StrokeTiming(20.0)
    assert timing.ratio > 2.0


def test_drive_and_recovery_approach_parity_at_racing_rates():
    """The paper's stated qualitative behaviour: the split evens out."""
    assert StrokeTiming(40.0).ratio == pytest.approx(1.11, abs=0.05)
    assert StrokeTiming(20.0).ratio > StrokeTiming(40.0).ratio


def test_ratio_decreases_monotonically_with_rate():
    rates = np.arange(18.0, 44.0, 2.0)
    ratios = [StrokeTiming(r).ratio for r in rates]
    assert all(b < a for a, b in zip(ratios, ratios[1:]))


@pytest.mark.parametrize("rate", [20.0, 30.0, 40.0])
def test_drive_fraction_is_between_zero_and_one(rate):
    assert 0.0 < StrokeTiming(rate).drive_fraction < 1.0


def test_rejects_non_positive_rate():
    with pytest.raises(ValueError, match="stroke rate must be positive"):
        StrokeTiming(0.0)


def test_rejects_a_cadence_where_the_fit_is_invalid():
    """At very high rates the empirical tau_a exceeds the stroke period."""
    with pytest.raises(ValueError, match="not.*valid at this cadence"):
        StrokeTiming(90.0)


# --------------------------------------------------------------------------
# phase
# --------------------------------------------------------------------------
def test_phase_wraps_over_the_period():
    timing = StrokeTiming(30.0)
    assert timing.phase(0.0) == pytest.approx(0.0)
    assert timing.phase(timing.period) == pytest.approx(0.0)
    assert timing.phase(timing.period / 2) == pytest.approx(0.5)
    assert timing.phase(2.5 * timing.period) == pytest.approx(0.5)


def test_is_drive_covers_exactly_the_drive_fraction():
    timing = StrokeTiming(30.0)
    times = np.linspace(0.0, timing.period, 10001, endpoint=False)
    fraction = timing.is_drive(times).mean()
    assert fraction == pytest.approx(timing.drive_fraction, abs=1e-3)


def test_is_drive_is_true_at_the_catch_and_false_late_in_recovery():
    timing = StrokeTiming(30.0)
    assert timing.is_drive(0.0)
    assert not timing.is_drive(0.95 * timing.period)


# --------------------------------------------------------------------------
# FourierProfile
# --------------------------------------------------------------------------
def test_constant_profile_has_zero_derivatives():
    profile = FourierProfile.constant(2.5, period=2.0)
    jet = profile(np.array(0.37))
    assert jet.value == pytest.approx(2.5)
    assert jet.first == pytest.approx(0.0)
    assert jet.second == pytest.approx(0.0)


def test_single_harmonic_reproduces_a_cosine_exactly():
    profile = FourierProfile([0.0, 1.0], [0.0, 0.0], period=2.0)
    t = np.linspace(0.0, 4.0, 51)
    omega = np.pi  # 2 pi / 2.0
    jet = profile(t)
    np.testing.assert_allclose(jet.value, np.cos(omega * t), atol=1e-12)
    np.testing.assert_allclose(jet.first, -omega * np.sin(omega * t), atol=1e-12)
    np.testing.assert_allclose(jet.second, -omega ** 2 * np.cos(omega * t),
                               atol=1e-12)


def test_profile_derivatives_agree_with_finite_differences():
    timing = StrokeTiming(30.0)
    profile = FourierProfile.from_catch_finish(0.2, 1.4, timing)
    h = 1e-6
    for t in (0.1, 0.4, 0.9, 1.6):
        centre = profile(np.array(t))
        plus, minus = profile(np.array(t + h)), profile(np.array(t - h))
        assert centre.first == pytest.approx(
            float(plus.value - minus.value) / (2 * h), abs=1e-4)
        assert centre.second == pytest.approx(
            float(plus.value - 2 * centre.value + minus.value) / h ** 2,
            abs=1e-2)


def test_profile_is_exactly_periodic():
    timing = StrokeTiming(30.0)
    profile = FourierProfile.from_catch_finish(-0.5, 0.8, timing)
    for t in (0.0, 0.31, 0.77, 1.42):
        a, b = profile(np.array(t)), profile(np.array(t + timing.period))
        assert a.value == pytest.approx(b.value, abs=1e-12)
        assert a.first == pytest.approx(b.first, abs=1e-12)
        assert a.second == pytest.approx(b.second, abs=1e-12)


def test_profile_has_no_acceleration_discontinuity_at_the_catch():
    """The specific legacy defect: a step in acceleration at phase 0.

    The legacy model switched the driving frequency between drive and
    recovery, leaving position and velocity continuous but jumping the
    acceleration by 3.4 m/s^2.  A Fourier series cannot do that.
    """
    timing = StrokeTiming(30.0)
    profile = FourierProfile.from_catch_finish(0.0, 1.0, timing)
    eps = 1e-6
    before = profile(np.array(timing.period - eps)).second
    after = profile(np.array(eps)).second
    assert float(abs(after - before)) < 1e-4


def test_profile_reaches_close_to_the_catch_and_finish_values():
    """Fourier truncation rounds the extremes slightly; it must stay close."""
    timing = StrokeTiming(30.0)
    profile = FourierProfile.from_catch_finish(0.0, 1.0, timing)
    assert profile(np.array(0.0)).value == pytest.approx(0.0, abs=0.05)
    assert profile(np.array(timing.drive_duration)).value == pytest.approx(
        1.0, abs=0.05)


def test_profile_spends_the_drive_fraction_moving_catch_to_finish():
    """The extremum near the end of the drive marks the finish."""
    timing = StrokeTiming(30.0)
    profile = FourierProfile.from_catch_finish(0.0, 1.0, timing)
    phases = np.linspace(0.0, 1.0, 2001, endpoint=False)
    values = profile.value_at_phase(phases)
    assert phases[np.argmax(values)] == pytest.approx(timing.drive_fraction,
                                                      abs=0.03)
    # the minimum sits at the catch, i.e. phase 0 modulo 1
    catch_phase = phases[np.argmin(values)]
    assert min(catch_phase, 1.0 - catch_phase) < 0.03


def test_fit_samples_round_trips_a_band_limited_signal():
    period = 2.0
    n = 256
    phase = np.arange(n) / n
    samples = 0.3 + 1.2 * np.cos(2 * np.pi * phase) - 0.7 * np.sin(
        4 * np.pi * phase)

    profile = FourierProfile.fit_samples(samples, period, n_harmonics=4)
    np.testing.assert_allclose(profile.value_at_phase(phase), samples,
                               atol=1e-10)


def test_fit_samples_recovers_known_coefficients():
    period, n = 1.5, 128
    phase = np.arange(n) / n
    samples = 2.0 + 0.5 * np.cos(2 * np.pi * phase) + 0.25 * np.sin(
        6 * np.pi * phase)
    profile = FourierProfile.fit_samples(samples, period, n_harmonics=3)

    assert profile.cos_coefficients[0] == pytest.approx(2.0, abs=1e-12)
    assert profile.cos_coefficients[1] == pytest.approx(0.5, abs=1e-12)
    assert profile.sin_coefficients[3] == pytest.approx(0.25, abs=1e-12)


def test_fit_samples_rejects_too_few_samples():
    with pytest.raises(ValueError, match="need at least"):
        FourierProfile.fit_samples(np.zeros(5), 2.0, n_harmonics=8)


def test_n_harmonics_reports_the_truncation_level():
    timing = StrokeTiming(30.0)
    profile = FourierProfile.from_catch_finish(0.0, 1.0, timing, n_harmonics=6)
    assert profile.n_harmonics == 6


def test_more_harmonics_track_the_idealised_shape_more_closely():
    timing = StrokeTiming(30.0)
    coarse = FourierProfile.from_catch_finish(0.0, 1.0, timing, n_harmonics=2)
    fine = FourierProfile.from_catch_finish(0.0, 1.0, timing, n_harmonics=16)
    assert abs(fine(np.array(0.0)).value) < abs(coarse(np.array(0.0)).value)


def test_mismatched_coefficient_arrays_are_rejected():
    with pytest.raises(ValueError, match="must match"):
        FourierProfile([0.0, 1.0], [0.0], period=2.0)


def test_non_positive_period_is_rejected():
    with pytest.raises(ValueError, match="period must be positive"):
        FourierProfile([0.0], [0.0], period=0.0)


def test_with_period_preserves_shape_but_retimes():
    timing = StrokeTiming(30.0)
    original = FourierProfile.from_catch_finish(0.0, 1.0, timing)
    retimed = original.with_period(1.0)

    np.testing.assert_allclose(retimed.cos_coefficients,
                               original.cos_coefficients)
    phases = np.linspace(0.0, 1.0, 33)
    np.testing.assert_allclose(retimed.value_at_phase(phases),
                               original.value_at_phase(phases), atol=1e-12)
    # same shape, faster: derivative scales with the period ratio
    ratio = original.period / retimed.period
    assert retimed(np.array(0.3)).first == pytest.approx(
        ratio * original(np.array(0.3 * ratio)).first, rel=1e-9)


def test_default_harmonics_constant_is_sane():
    assert 4 <= DEFAULT_HARMONICS <= 32


def test_profile_accepts_scalar_and_array_time():
    timing = StrokeTiming(30.0)
    profile = FourierProfile.from_catch_finish(0.0, 1.0, timing)
    scalar = profile(np.array(0.4))
    vector = profile(np.array([0.4, 0.9]))
    assert scalar.value == pytest.approx(vector.value[0])
    assert scalar.first == pytest.approx(vector.first[0])
