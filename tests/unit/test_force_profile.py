"""The force profile against the review literature.

Warmenhoven et al. (2018) survey fifty years of force-profile research and
report two things this model can be held to: where the peak sits, and how
it moves with stroke rate.  The second turned out to be a defect -- the
model moved it the wrong way -- and these tests pin both the defect's fix
and the fact that the fix is opt-in.
"""

import numpy as np
import pytest

from coxswain.crew.oarlock import (
    DRIVE_SHAPE,
    DRIVE_SHAPE_MEAN,
    MCBRIDE_SHIFT_PER_SPM,
    OarForceProfile,
)
from coxswain.crew.stroke import StrokeTiming


def cycle_peak(profile, rate):
    """Where the peak falls as a fraction of the whole STROKE CYCLE.

    McBride's measurement is quoted in cycle terms, and the distinction
    matters: the drive is a growing fraction of the cycle as rate rises,
    so a peak fixed within the drive still moves within the cycle.
    """
    timing = StrokeTiming(float(rate))
    return profile.peak_position(timing) * timing.drive_fraction


# --------------------------------------------------------------------------
# the default is unchanged
# --------------------------------------------------------------------------
def test_default_profile_still_peaks_where_kleshnev_put_it():
    """Nothing in the catalogue's calibration may move by default."""
    profile = OarForceProfile()
    expected = DRIVE_SHAPE[0] / sum(DRIVE_SHAPE)
    assert profile.peak_position(StrokeTiming(32.0)) == pytest.approx(
        expected, rel=1e-9)


def test_default_profile_shape_is_bit_for_bit_the_old_curve():
    timing = StrokeTiming(32.0)
    t = np.linspace(0.0, timing.period, 200)
    a, b = DRIVE_SHAPE
    u = np.clip(np.mod(t, timing.period) / timing.drive_duration, 0.0, 1.0)
    peak = a / (a + b)
    curve = u ** a * (1.0 - u) ** b / (peak ** a * (1.0 - peak) ** b)
    expected = np.where(np.mod(t, timing.period) <= timing.drive_duration,
                        np.maximum(curve, 0.0), 0.0)
    np.testing.assert_allclose(OarForceProfile().magnitude(t, timing),
                               expected, atol=1e-12)


def test_default_mean_to_peak_matches_the_recorded_constant():
    """``DRIVE_SHAPE_MEAN`` is the MPFR of the default curve."""
    assert OarForceProfile().mean_to_peak(StrokeTiming(32.0)) == \
        pytest.approx(DRIVE_SHAPE_MEAN, rel=1e-3)


def test_default_profile_is_rate_independent_within_the_drive():
    profile = OarForceProfile()
    positions = [profile.peak_position(StrokeTiming(r))
                 for r in (20.0, 28.0, 36.0)]
    assert max(positions) - min(positions) < 1e-12


# --------------------------------------------------------------------------
# the defect the review exposed
# --------------------------------------------------------------------------
def test_without_the_shift_the_peak_moves_the_WRONG_way_with_rate():
    """The model's peak arrives LATER in the cycle at race pace.

    McBride measured it arriving earlier.  This test exists to state the
    defect rather than to bless it: it pins the sign so that anyone who
    fixes the drive-fraction fit is told they have changed this too.
    """
    profile = OarForceProfile()
    drift = (cycle_peak(profile, 20.0) - cycle_peak(profile, 36.0)) * 100.0
    assert drift < 0.0
    assert drift == pytest.approx(-4.63, abs=0.1)


def test_the_calibrated_shift_reproduces_mcbride():
    """3.4% of the cycle earlier, 20 spm to race pace."""
    profile = OarForceProfile(shift_per_spm=MCBRIDE_SHIFT_PER_SPM)
    drift = (cycle_peak(profile, 20.0) - cycle_peak(profile, 36.0)) * 100.0
    assert drift == pytest.approx(3.4, abs=0.05)


def test_the_shift_is_opt_in():
    """Every speed calibration in the catalogue predates this."""
    assert OarForceProfile().shift_per_spm == 0.0
    assert OarForceProfile().peak_shift == 0.0


def test_the_shifted_peak_moves_monotonically_earlier_with_rate():
    profile = OarForceProfile(shift_per_spm=MCBRIDE_SHIFT_PER_SPM)
    positions = [profile.peak_position(StrokeTiming(r))
                 for r in (20.0, 26.0, 32.0, 36.0)]
    assert np.all(np.diff(positions) < 0.0)


# --------------------------------------------------------------------------
# properties of the shape itself
# --------------------------------------------------------------------------
def test_front_loading_lowers_the_mean_to_peak_ratio():
    """A consequence of the parameterisation, and labelled as one.

    Holding ``a + b`` fixed while the mode moves forward skews the curve,
    which lowers MPFR.  That is arithmetic, not a measured claim about
    rowers -- the review reports elite rowers with *higher* MPFR, and
    nothing here predicts that from the peak position alone.
    """
    timing = StrokeTiming(36.0)
    plain = OarForceProfile().mean_to_peak(timing)
    shifted = OarForceProfile(
        shift_per_spm=MCBRIDE_SHIFT_PER_SPM).mean_to_peak(timing)
    assert shifted < plain


def test_mean_to_peak_is_a_ratio_in_the_unit_interval():
    for rate in (20.0, 32.0, 40.0):
        value = OarForceProfile().mean_to_peak(StrokeTiming(rate))
        assert 0.0 < value < 1.0


def test_the_profile_is_still_zero_through_the_recovery():
    timing = StrokeTiming(32.0)
    profile = OarForceProfile(shift_per_spm=MCBRIDE_SHIFT_PER_SPM)
    recovery = np.linspace(timing.drive_duration * 1.01, timing.period, 40)
    np.testing.assert_allclose(profile.magnitude(recovery, timing), 0.0,
                               atol=1e-12)


def test_the_peak_of_the_shifted_curve_is_where_it_says_it_is():
    """``peak_position`` must describe the curve, not merely parameterise it."""
    timing = StrokeTiming(36.0)
    profile = OarForceProfile(shift_per_spm=MCBRIDE_SHIFT_PER_SPM)
    u = np.linspace(1e-4, 1.0 - 1e-4, 4000)
    curve = profile.magnitude(u * timing.drive_duration, timing)
    assert u[int(np.argmax(curve))] == pytest.approx(
        profile.peak_position(timing), abs=2e-3)


def test_an_explicit_peak_shift_works_without_a_rate_term():
    profile = OarForceProfile(peak_shift=0.10)
    assert profile.peak_position(StrokeTiming(32.0)) == pytest.approx(
        DRIVE_SHAPE[0] / sum(DRIVE_SHAPE) - 0.10, rel=1e-9)
