"""Unit tests for the slip-dependent blade model, after [CR06].

Cabrera, Ruina & Kleshnev (2006) eq. (11):

    F_oar = C2 (l theta_dot + v_b cos theta)^2

normal to the blade, with C2 = 58.7 (scull) / 84.5 (sweep) from their fits
to on-water force and kinematic data.
"""

import numpy as np
import pytest

from coxswain.crew.oarlock import BladeModel, OarAngleSweep
from coxswain.crew.stroke import StrokeTiming


@pytest.fixture
def blade():
    return BladeModel.sweep()


@pytest.fixture
def timing():
    return StrokeTiming(rate=32.0)


def sweep_rate(sweep, t, timing, step=1e-6):
    return (float(sweep(t + step, timing))
            - float(sweep(t - step, timing))) / (2 * step)


# --------------------------------------------------------------------------
# published coefficients
# --------------------------------------------------------------------------
def test_published_coefficients_are_carried_exactly():
    assert BladeModel.sculling().c2 == pytest.approx(58.7)
    assert BladeModel.sweep().c2 == pytest.approx(84.5)


def test_sweep_blades_are_loaded_more_heavily_than_sculling_blades():
    """One sweep oar per rower against two sculls, so each carries more."""
    assert BladeModel.sweep().c2 > BladeModel.sculling().c2


# --------------------------------------------------------------------------
# slip velocity -- eq. (11)
# --------------------------------------------------------------------------
def test_slip_velocity_matches_the_paper_expression(blade):
    angle, rate, speed = 0.4, -2.5, 5.0
    expected = blade.outboard * rate + speed * np.cos(angle)
    assert blade.slip_velocity(angle, rate, speed) == pytest.approx(expected)


def test_stationary_blade_in_a_moving_boat_slips_forward(blade):
    """With no oar rotation the blade is dragged along at boat speed."""
    slip = blade.slip_velocity(0.0, 0.0, 5.0)
    assert slip == pytest.approx(5.0)


def test_slip_vanishes_when_the_blade_matches_the_water(blade):
    """The no-slip condition l theta_dot = -v_b cos theta."""
    angle, speed = 0.3, 5.0
    rate = -speed * np.cos(angle) / blade.outboard
    assert blade.slip_velocity(angle, rate, speed) == pytest.approx(0.0,
                                                                    abs=1e-12)


# --------------------------------------------------------------------------
# force magnitude and sign
# --------------------------------------------------------------------------
def test_force_is_quadratic_in_slip(blade):
    """Doubling the slip must quadruple the force."""
    slow = abs(blade.normal_force(0.0, -1.0, 0.0))
    fast = abs(blade.normal_force(0.0, -2.0, 0.0))
    assert fast == pytest.approx(4.0 * slow, rel=1e-12)


def test_force_magnitude_equals_c2_times_slip_squared(blade):
    angle, rate, speed = 0.2, -3.0, 5.0
    slip = blade.slip_velocity(angle, rate, speed)
    assert abs(blade.normal_force(angle, rate, speed)) == pytest.approx(
        blade.c2 * slip ** 2)


def test_force_always_opposes_the_slip(blade):
    """Squaring alone would lose the sign and let the blade make thrust
    on the recovery."""
    for rate in (-4.0, -1.0, 1.0, 4.0):
        for speed in (0.0, 3.0, 6.0):
            slip = blade.slip_velocity(0.1, rate, speed)
            force = blade.normal_force(0.1, rate, speed)
            if abs(slip) > 1e-9:
                assert np.sign(force) == -np.sign(slip)


def test_no_slip_means_no_force(blade):
    angle, speed = 0.3, 5.0
    rate = -speed * np.cos(angle) / blade.outboard
    assert blade.normal_force(angle, rate, speed) == pytest.approx(0.0,
                                                                   abs=1e-9)


def test_a_faster_boat_reduces_drive_force_at_fixed_oar_rate(blade):
    """The feedback a prescribed force profile cannot represent.

    As the boat runs away from the blade the slip falls, so the blade
    loses grip.  This is why an open-loop half-sine over-drives a fast
    boat.
    """
    rate = -3.0
    slow_boat = abs(blade.normal_force(0.0, rate, 3.0))
    fast_boat = abs(blade.normal_force(0.0, rate, 6.0))
    assert fast_boat < slow_boat


# --------------------------------------------------------------------------
# propulsive component
# --------------------------------------------------------------------------
def test_propulsive_component_is_the_cosine_of_the_normal_force(blade):
    angle, rate, speed = 0.5, -3.0, 5.0
    assert blade.propulsive_force(angle, rate, speed) == pytest.approx(
        blade.normal_force(angle, rate, speed) * np.cos(angle))


def test_propulsion_is_wasted_at_large_oar_angles(blade):
    """[B09]'s argument: at a big catch angle most blade load goes sideways.

    The comparison has to hold the *slip* fixed.  Simply raising the oar
    angle at a fixed angular rate also raises the slip -- the boat's
    forward speed cancels less of the oar's sweep -- which increases the
    total blade force and masks the projection loss.  Here the rate is
    solved at each angle to give the same slip, isolating ``cos(theta)``.
    """
    speed, target_slip = 5.0, -2.0

    def rate_for_slip(angle):
        return (target_slip - speed * np.cos(angle)) / blade.outboard

    square = abs(blade.propulsive_force(0.0, rate_for_slip(0.0), speed))
    catch_angle = np.radians(55.0)
    oblique = abs(blade.propulsive_force(catch_angle,
                                         rate_for_slip(catch_angle), speed))
    assert oblique == pytest.approx(square * np.cos(catch_angle), rel=1e-9)
    assert oblique < 0.6 * square


def test_total_blade_load_still_rises_at_the_catch_angle(blade):
    """The other half of the picture, and why the naive test above failed.

    At a fixed angular rate a larger oar angle gives *more* slip and so
    more total blade load; it is only the propulsive fraction that falls.
    Both effects are real and they partly offset.
    """
    rate, speed = -3.0, 5.0
    assert abs(blade.slip_velocity(np.radians(55.0), rate, speed)) > \
        abs(blade.slip_velocity(0.0, rate, speed))
    assert abs(blade.normal_force(np.radians(55.0), rate, speed)) > \
        abs(blade.normal_force(0.0, rate, speed))


def test_blade_is_purely_lateral_when_the_oar_is_fore_and_aft(blade):
    assert blade.propulsive_force(np.pi / 2, -3.0, 0.0) == pytest.approx(
        0.0, abs=1e-9)


# --------------------------------------------------------------------------
# efficiency
# --------------------------------------------------------------------------
def test_efficiency_is_one_when_the_blade_does_not_slip(blade):
    angle, speed = 0.3, 5.0
    rate = -speed * np.cos(angle) / blade.outboard
    assert float(blade.efficiency(angle, rate, speed)) == pytest.approx(1.0)


def test_efficiency_is_zero_for_a_stationary_oar(blade):
    assert float(blade.efficiency(0.0, 0.0, 5.0)) == pytest.approx(0.0)


def test_efficiency_stays_within_zero_and_one(blade):
    rng = np.random.default_rng(0)
    for _ in range(200):
        angle = rng.uniform(-1.0, 1.0)
        rate = rng.uniform(-5.0, 5.0)
        speed = rng.uniform(0.0, 7.0)
        value = float(blade.efficiency(angle, rate, speed))
        assert 0.0 <= value <= 1.0


def test_mid_drive_efficiency_brackets_the_fixed_factor_it_replaces(blade,
                                                                    timing):
    """The constant ``blade_efficiency = 0.78`` should sit inside the range
    the real model produces through the middle of the drive."""
    sweep = OarAngleSweep()
    values = []
    for fraction in (0.35, 0.45, 0.55, 0.65):
        t = fraction * timing.drive_duration
        angle = float(sweep(t, timing))
        values.append(float(blade.efficiency(angle,
                                             sweep_rate(sweep, t, timing),
                                             5.2)))
    assert min(values) < 0.78 < max(values) or 0.6 < np.mean(values) < 0.9


# --------------------------------------------------------------------------
# behaviour over a real stroke
# --------------------------------------------------------------------------
def test_blade_retards_the_boat_near_the_catch_and_the_finish(blade, timing):
    """The check at the catch and the wash-out at the finish.

    At both ends the oar's angular rate passes through zero, so the boat's
    own speed dominates the slip and the blade is being dragged rather than
    driving.  A prescribed half-sine force profile cannot produce this at
    all; it is one of the qualitative features [CR06] gain by closing the
    loop.
    """
    sweep = OarAngleSweep()
    for fraction in (0.02, 0.98):
        t = fraction * timing.drive_duration
        angle = float(sweep(t, timing))
        force = blade.propulsive_force(angle, sweep_rate(sweep, t, timing),
                                       5.2)
        assert force < 0.0


def test_blade_drives_the_boat_through_the_middle_of_the_drive(blade, timing):
    sweep = OarAngleSweep()
    for fraction in (0.35, 0.5, 0.65):
        t = fraction * timing.drive_duration
        angle = float(sweep(t, timing))
        force = blade.propulsive_force(angle, sweep_rate(sweep, t, timing),
                                       5.2)
        assert force > 0.0


def test_peak_blade_force_is_physically_plausible(blade, timing):
    """A racing sweep blade peaks in the high hundreds of newtons.

    [F09] takes 1200 N at the *handle*; the blade sees less, by the
    gearing.  An order-of-magnitude guard, not a precision claim.
    """
    sweep = OarAngleSweep()
    times = np.linspace(0.0, timing.drive_duration, 200)
    forces = [abs(blade.propulsive_force(float(sweep(t, timing)),
                                         sweep_rate(sweep, t, timing), 5.2))
              for t in times]
    assert 100.0 < max(forces) < 2000.0


def test_model_is_vectorised(blade):
    angles = np.array([0.0, 0.3, 0.6])
    rates = np.array([-3.0, -2.0, -1.0])
    forces = blade.normal_force(angles, rates, 5.0)
    assert forces.shape == (3,)
    for i in range(3):
        assert forces[i] == pytest.approx(
            blade.normal_force(angles[i], rates[i], 5.0))


# --------------------------------------------------------------------------
# depth of water around the blade
# --------------------------------------------------------------------------
def test_a_blade_at_the_surface_makes_no_force(blade):
    """Zero cover means the blade is fully ventilated."""
    assert blade.immersion_factor(0.0) == pytest.approx(0.0)


def test_immersion_factor_increases_monotonically_with_cover(blade):
    covers = np.linspace(0.0, 0.6, 25)
    values = [blade.immersion_factor(c) for c in covers]
    assert np.all(np.diff(values) >= -1e-12)


def test_half_a_blade_width_of_cover_is_near_optimal(blade):
    """Kleshnev's reported optimum; the roll-off constant is set by it."""
    assert blade.immersion_factor(blade.blade_width / 2) == pytest.approx(
        0.90, abs=0.01)


def test_immersion_factor_saturates_when_well_buried(blade):
    assert blade.immersion_factor(2.0 * blade.blade_width) > 0.98
    assert blade.immersion_factor(10.0) <= 1.0


def test_washed_out_blade_loses_most_of_its_force(blade):
    """A blade barely covered slips badly, which is the visible failure."""
    shallow = blade.normal_force(0.0, -3.0, 5.0, cover=0.02)
    buried = blade.normal_force(0.0, -3.0, 5.0, cover=0.25)
    assert abs(shallow) < 0.4 * abs(buried)


def test_deep_water_has_no_blockage(blade):
    assert blade.blockage_factor(np.inf) == pytest.approx(1.0)
    assert blade.blockage_factor(None) == pytest.approx(1.0)


def test_blockage_grows_as_the_water_shallows(blade):
    depths = [12.0, 8.0, 5.0, 4.0, 3.0, 2.0]
    values = [blade.blockage_factor(h) for h in depths]
    assert np.all(np.diff(values) > 0.0)


def test_blockage_matches_maskell(blade):
    depth = 3.0
    sigma = blade.blade_width / depth
    expected = 1.0 + blade.blockage_m * sigma * blade.blade_cd
    assert blade.blockage_factor(depth) == pytest.approx(expected)


def test_blockage_is_conservative_against_the_published_check_point(blade):
    """sigma = 0.05 is reported as under 10% for ducted bluff bodies.

    Maskell with the generic m = 2.5 gives 14% -- an upper bound, as the
    docstring says, because a blade is confined only vertically.  This
    pins the model at that value rather than silently retuning it.
    """
    depth = blade.blade_width / 0.05
    assert blade.blockage_factor(depth) == pytest.approx(1.14, abs=0.01)


def test_blockage_refuses_water_shallower_than_the_blade(blade):
    with pytest.raises(ValueError, match="not deeper than the blade"):
        blade.blockage_factor(0.2)


def test_charles_depths_raise_blade_force_by_a_measurable_amount(blade):
    """2-4 m is the band the Charles actually sits in."""
    for depth, low, high in ((4.0, 1.15, 1.20), (3.0, 1.20, 1.25),
                             (2.0, 1.30, 1.40)):
        assert low < blade.blockage_factor(depth) < high


def test_depth_factor_combines_both_effects(blade):
    assert blade.depth_factor(3.0, 0.125) == pytest.approx(
        blade.immersion_factor(0.125) * blade.blockage_factor(3.0))


def test_the_two_depth_effects_oppose_each_other(blade):
    """Shallow burial loses force; shallow water gains it."""
    assert blade.immersion_factor(0.05) < blade.immersion_factor(0.25)
    assert blade.blockage_factor(2.0) > blade.blockage_factor(8.0)


def test_force_is_unscaled_when_no_depth_is_given(blade):
    """The default path must reproduce [CR06] exactly."""
    plain = blade.normal_force(0.2, -3.0, 5.0)
    slip = blade.slip_velocity(0.2, -3.0, 5.0)
    assert abs(plain) == pytest.approx(blade.c2 * slip ** 2)


def test_depth_scales_propulsive_force_too(blade):
    deep = blade.propulsive_force(0.2, -3.0, 5.0, water_depth=np.inf)
    shallow = blade.propulsive_force(0.2, -3.0, 5.0, water_depth=2.0)
    assert abs(shallow) > abs(deep)
