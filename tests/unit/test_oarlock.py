"""Unit tests for oar forces and their transmission to the hull.

The key quantitative claim checked here is Formaggia eq. (12)/(14a): an
ideal lever delivers only ``r_h / L`` of the oarlock force to the hull,
about 0.31 for a sweep rig.  The legacy code applied the full oarlock
force, overstating thrust by more than three times.
"""

import numpy as np
import pytest

from coxswain.boats.rig import SCULLING_OAR, SWEEP_OAR, Oar
from coxswain.crew.anthropometry import PORT, STARBOARD
from coxswain.crew.oarlock import (
    OarAngleSweep,
    OarForceProfile,
    hull_load,
    oar_force,
)
from coxswain.crew.stroke import StrokeTiming


@pytest.fixture
def timing():
    return StrokeTiming(32.0)


# --------------------------------------------------------------------------
# force profile, Formaggia eq. (15)
# --------------------------------------------------------------------------
def test_force_is_zero_at_the_catch(timing):
    assert OarForceProfile().magnitude(0.0, timing) == pytest.approx(0.0)


def test_force_is_zero_at_the_finish(timing):
    magnitude = OarForceProfile().magnitude(timing.drive_duration, timing)
    assert magnitude == pytest.approx(0.0, abs=1e-12)


def test_force_peaks_at_forty_percent_of_the_drive(timing):
    """Kleshnev: peak force at 40% of the drive length, not the middle.

    The profile used to be a symmetric half-sine, which peaks at 50%.
    Real force curves are front-loaded.
    """
    profile = OarForceProfile()
    fractions = np.linspace(0.0, 1.0, 2001)
    curve = np.array([profile.magnitude(f * timing.drive_duration, timing)
                      for f in fractions])
    assert fractions[curve.argmax()] == pytest.approx(0.40, abs=0.01)
    assert curve.max() == pytest.approx(1.0, rel=1e-6)


def test_force_has_decayed_to_kleshnevs_value_by_sixty_percent(timing):
    """The second constraint the curve is fitted to.

    Kleshnev reports the force already down to 74% of peak at 60% of the
    drive, where a half-sine is still at 95%.  Fitting both the peak
    position and this decay is what fixes the exponents.
    """
    profile = OarForceProfile()
    at_sixty = profile.magnitude(0.60 * timing.drive_duration, timing)
    assert at_sixty == pytest.approx(0.74, abs=0.01)
    assert at_sixty < np.sin(np.pi * 0.60)      # strictly ahead of a half-sine


def test_half_sine_remains_available_for_comparison(timing):
    """Every catalogue speed calibration predates the change."""
    profile = OarForceProfile(shape="half_sine")
    for fraction in (0.25, 0.5, 0.75):
        t = fraction * timing.drive_duration
        assert profile.magnitude(t, timing) == pytest.approx(
            np.sin(np.pi * fraction), rel=1e-9)


def test_no_force_during_the_recovery(timing):
    profile = OarForceProfile()
    for fraction in (0.55, 0.75, 0.95):
        t = fraction * timing.period
        if t > timing.drive_duration:
            assert profile.magnitude(t, timing) == pytest.approx(0.0)


def test_force_profile_is_periodic(timing):
    profile = OarForceProfile()
    for t in (0.2, 0.6, 1.1):
        assert profile.magnitude(t, timing) == pytest.approx(
            profile.magnitude(t + timing.period, timing))


def test_force_profile_is_continuous_at_the_finish(timing):
    """It vanishes at both ends, so no impulsive load on the hull."""
    profile = OarForceProfile()
    eps = 1e-7
    before = profile.magnitude(timing.drive_duration - eps, timing)
    after = profile.magnitude(timing.drive_duration + eps, timing)
    assert abs(float(before) - float(after)) < 1e-5


def test_default_peaks_match_the_published_values():
    profile = OarForceProfile()
    assert profile.max_x == pytest.approx(1200.0)
    assert profile.max_z == pytest.approx(200.0)


# --------------------------------------------------------------------------
# oar sweep angle
# --------------------------------------------------------------------------
def test_blade_starts_bow_ward_and_finishes_stern_ward(timing):
    sweep = OarAngleSweep()
    assert float(sweep(0.0, timing)) > 0.0
    assert float(sweep(timing.drive_duration * 0.999, timing)) < 0.0


def test_sweep_angle_is_monotone_through_the_drive(timing):
    sweep = OarAngleSweep()
    times = np.linspace(0.0, timing.drive_duration * 0.999, 40)
    angles = np.array([float(sweep(t, timing)) for t in times])
    assert (np.diff(angles) < 0).all()


def test_sweep_returns_to_the_catch_angle_by_the_next_stroke(timing):
    sweep = OarAngleSweep()
    assert float(sweep(0.0, timing)) == pytest.approx(
        float(sweep(timing.period, timing)), abs=1e-9)


def test_total_sweep_is_a_realistic_arc():
    sweep = OarAngleSweep()
    assert np.radians(70.0) < sweep.total_sweep < np.radians(110.0)


# --------------------------------------------------------------------------
# force decomposition
# --------------------------------------------------------------------------
def test_thrust_tapers_towards_the_catch(timing):
    """f_x = |F| cos(phi), so the propulsive share is smallest at the ends.

    Deriving f_y from f_x tan(phi) instead makes the *total* load largest
    at the catch, which is backwards.
    """
    early = oar_force(0.08 * timing.drive_duration, timing, PORT)
    middle = oar_force(0.5 * timing.drive_duration, timing, PORT)
    assert middle[0] > early[0]


def test_lateral_force_reverses_with_the_rigged_side(timing):
    t = 0.3 * timing.drive_duration
    port = oar_force(t, timing, PORT)
    starboard = oar_force(t, timing, STARBOARD)
    assert port[1] == pytest.approx(-starboard[1], rel=1e-12)
    assert port[0] == pytest.approx(starboard[0], rel=1e-12)
    assert port[2] == pytest.approx(starboard[2], rel=1e-12)


def test_a_sculler_generates_no_net_lateral_force(timing):
    """Two oars on opposite sides cancel, recovering the paper's planar model."""
    t = 0.4 * timing.drive_duration
    total = oar_force(t, timing, PORT) + oar_force(t, timing, STARBOARD)
    assert total[1] == pytest.approx(0.0, abs=1e-12)


def test_horizontal_magnitude_follows_the_half_sine(timing):
    profile = OarForceProfile()
    t = 0.35 * timing.drive_duration
    force = oar_force(t, timing, PORT, profile)
    magnitude = np.hypot(force[0], force[1])
    expected = profile.max_x * float(profile.magnitude(t, timing))
    assert magnitude == pytest.approx(expected, rel=1e-12)


def test_vertical_force_uses_the_vertical_peak(timing):
    """Sampled at the peak of the drive, which is now 40% not 50%."""
    profile = OarForceProfile(max_x=1000.0, max_z=150.0)
    t = 0.40 * timing.drive_duration
    force = oar_force(t, timing, PORT, profile)
    assert force[2] == pytest.approx(150.0, rel=1e-3)


def test_no_force_at_all_during_the_recovery(timing):
    t = 0.9 * timing.period
    np.testing.assert_allclose(oar_force(t, timing, PORT), np.zeros(3),
                               atol=1e-12)


# --------------------------------------------------------------------------
# the oar as a lever, Formaggia eq. (12)
# --------------------------------------------------------------------------
def test_sweep_gearing_matches_standard_rigging():
    assert SWEEP_OAR.gearing == pytest.approx(1.14 / 3.70, rel=1e-12)
    assert 0.28 < SWEEP_OAR.gearing < 0.34


def test_sculling_gearing():
    assert SCULLING_OAR.gearing == pytest.approx(0.88 / 2.88, rel=1e-12)


def test_outboard_plus_inboard_is_the_oar_length():
    assert SWEEP_OAR.outboard + SWEEP_OAR.inboard == pytest.approx(
        SWEEP_OAR.length)


def test_effective_gearing_includes_blade_efficiency():
    oar = Oar(length=3.7, inboard=1.14, blade_efficiency=0.8)
    assert oar.effective_gearing == pytest.approx(0.8 * oar.gearing)


def test_oar_rejects_an_inboard_outside_its_length():
    with pytest.raises(ValueError, match="inboard"):
        Oar(length=3.0, inboard=3.5)
    with pytest.raises(ValueError, match="inboard"):
        Oar(length=3.0, inboard=0.0)


def test_oar_rejects_an_impossible_blade_efficiency():
    with pytest.raises(ValueError, match="blade_efficiency"):
        Oar(length=3.0, inboard=1.0, blade_efficiency=1.5)


def test_only_the_geared_fraction_reaches_the_hull():
    """The legacy code applied the full oarlock force: 3.2x too much."""
    applied = np.array([1000.0, 0.0, 0.0])
    force, _ = hull_load(applied, np.array([0.0, 0.85, 0.38]),
                         np.array([-0.5, -0.2, 0.6]), SWEEP_OAR.gearing)
    assert force[0] == pytest.approx(1000.0 * SWEEP_OAR.gearing, rel=1e-12)
    assert force[0] < 0.35 * 1000.0


def test_hull_force_is_parallel_to_the_applied_force():
    applied = np.array([800.0, 300.0, 120.0])
    force, _ = hull_load(applied, np.zeros(3), np.zeros(3), 0.31)
    np.testing.assert_allclose(force, 0.31 * applied, rtol=1e-12)


def test_hull_moment_uses_the_papers_lever_arm():
    """(x_o - x_h + (r_h/L) x_h) x F_o, per eq. (14b)."""
    applied = np.array([900.0, 100.0, 50.0])
    oarlock = np.array([0.3, 0.85, 0.38])
    hand = np.array([-0.4, -0.15, 0.62])
    gearing = 0.31

    _, moment = hull_load(applied, oarlock, hand, gearing)
    expected_lever = oarlock - hand + gearing * hand
    np.testing.assert_allclose(moment, np.cross(expected_lever, applied),
                               rtol=1e-12)


def test_zero_force_gives_zero_load():
    force, moment = hull_load(np.zeros(3), np.array([0.3, 0.85, 0.38]),
                              np.array([-0.4, 0.0, 0.6]), 0.31)
    np.testing.assert_allclose(force, np.zeros(3))
    np.testing.assert_allclose(moment, np.zeros(3))


def test_an_outboard_oarlock_yaws_the_boat():
    """A longitudinal load at a lateral offset makes a yaw moment."""
    applied = np.array([900.0, 0.0, 0.0])
    _, moment = hull_load(applied, np.array([0.0, 0.85, 0.38]), np.zeros(3),
                          0.31)
    assert abs(moment[2]) > 0.0
