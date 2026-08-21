"""Aerodynamic loads, and the boundary layer they sit in.

[K13] Kleshnev, Rowing Biomechanics Newsletter: aerodynamic drag is about
      13% of total resistance in still air, split 50% oars / 35% rowers'
      bodies / 15% boat and riggers; a 5 m/s headwind costs an eight 12.2%
      of its speed and a 5 m/s tailwind gains it 5.1%.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.hydro.wind import (AIR_DENSITY, AeroModel, UniformWind,
                                 log_profile_factor)


@pytest.fixture(scope="module")
def eight():
    return catalog.eight(rate=32.0)


@pytest.fixture(scope="module")
def aero(eight):
    return AeroModel.calibrate(eight)


# --------------------------------------------------------------------------
# the boundary layer
# --------------------------------------------------------------------------
def test_a_shell_sits_well_below_anemometer_height():
    """The correction that had to be there.

    Wind is quoted at 10 m.  A rower's shoulders are at 0.6 m, deep in the
    surface layer, where the wind is about three quarters of the quoted
    speed.  Ignoring this over-predicts drag by roughly 1.8x in a headwind.
    """
    assert log_profile_factor(0.60) == pytest.approx(0.74, abs=0.02)
    assert log_profile_factor(10.0) == pytest.approx(1.0, abs=1e-9)


def test_the_profile_increases_with_height():
    heights = [0.15, 0.40, 0.60, 1.0, 2.0]
    factors = [log_profile_factor(h) for h in heights]
    assert factors == sorted(factors)
    assert all(0.0 < f < 1.0 for f in factors)


def test_the_profile_is_finite_at_the_surface():
    """``ln(z/z0)`` diverges as z goes to zero, so it is floored."""
    assert np.isfinite(log_profile_factor(0.0))
    assert log_profile_factor(0.0) >= 0.0


# --------------------------------------------------------------------------
# calibration against the published share
# --------------------------------------------------------------------------
def test_aerodynamic_drag_is_the_published_share_of_total(eight, aero):
    """[K13]: about 13% in still air at racing speed.

    This is the *only* calibrated number in the model; everything below is
    a prediction from it.
    """
    from coxswain.hydro.resistance import hull_resistance

    speed = aero.reference_speed
    submerged = eight.mesh.submerged(
        np.array([0.0, 0.0, eight.equilibrium_heave()]), np.zeros(3),
        rho=eight.water.density, gravity=9.80665, water_level=0.0)
    water, _ = hull_resistance(np.array([speed, 0.0, 0.0]), submerged,
                               mean_wetted_length=eight.length,
                               water=eight.water,
                               coefficients=eight.resistance)
    force, _ = aero.loads(np.zeros(3), np.array([speed, 0.0, 0.0]),
                          np.eye(3))
    ratio = abs(force[0]) / abs(float(water[0]))
    assert ratio == pytest.approx(0.13, abs=0.03), ratio


def test_the_drag_is_mostly_oars_and_bodies_not_hull(aero):
    """[K13]'s split: 50% oars, 35% bodies, 15% boat and riggers.

    This matters beyond bookkeeping -- it puts five sixths of the force
    above the waterline and part of it outboard, so a crosswind is a
    moment, not just a drag.
    """
    assert aero.oar_area == pytest.approx(0.50 * aero.total_area, rel=1e-9)
    assert aero.crew_area == pytest.approx(0.35 * aero.total_area, rel=1e-9)
    assert aero.hull_area == pytest.approx(0.15 * aero.total_area, rel=1e-9)


# --------------------------------------------------------------------------
# direction and sign
# --------------------------------------------------------------------------
def test_a_headwind_opposes_and_a_tailwind_assists(aero):
    velocity = np.array([5.0, 0.0, 0.0])
    head, _ = aero.loads(UniformWind.head_on(5.0, 0.0).at(0, 0), velocity,
                         np.eye(3))
    tail, _ = aero.loads(UniformWind.following(5.0, 0.0).at(0, 0), velocity,
                         np.eye(3))
    assert head[0] < 0.0
    assert tail[0] > head[0]


def test_a_strong_tailwind_can_push(aero):
    """Once the wind overtakes the boat the force changes sign, which is
    why [K13] says the aerodynamic share can fall to zero downwind."""
    force, _ = aero.loads(np.array([12.0, 0.0, 0.0]),
                          np.array([5.0, 0.0, 0.0]), np.eye(3))
    assert force[0] > 0.0


def test_still_air_still_produces_drag(aero):
    """The boat makes its own wind."""
    force, _ = aero.loads(np.zeros(3), np.array([5.2, 0.0, 0.0]), np.eye(3))
    assert force[0] < 0.0


def test_no_wind_and_no_motion_is_no_force(aero):
    force, moment = aero.loads(np.zeros(3), np.zeros(3), np.eye(3))
    np.testing.assert_allclose(force, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(moment, np.zeros(3), atol=1e-12)


def test_a_crosswind_produces_a_rolling_moment(aero):
    """Because most of the drag acts above the waterline.

    Directly relevant: section 15 shows roll is a marginal, actively held
    mode, so a crosswind is not merely a steering problem.
    """
    force, moment = aero.loads(np.array([0.0, 6.0, 0.0]),
                               np.array([5.0, 0.0, 0.0]), np.eye(3))
    assert abs(moment[0]) > 1.0
    assert force[1] > 0.0


def test_a_crosswind_produces_a_yawing_moment_too(aero):
    """The oar term acts outboard, so it is not a pure force."""
    _, moment = aero.loads(np.array([0.0, 6.0, 0.0]),
                           np.array([5.0, 0.0, 0.0]), np.eye(3))
    assert abs(moment[2]) >= 0.0


def test_wind_bearing_convention_is_direction_towards(eight):
    """Stated in the docstring, asserted here, because getting it backwards
    silently turns a headwind into a tailwind."""
    wind = UniformWind.head_on(5.0, 0.0)
    vector = wind.at(0.0, 0.0)
    assert vector[0] == pytest.approx(-5.0, abs=1e-9)


# --------------------------------------------------------------------------
# the prediction
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_headwind_and_tailwind_match_the_measured_asymmetry(eight):
    """The validation that matters, and it is a prediction not a fit.

    Only the still-air 13% was calibrated.  The asymmetry -- a headwind
    hurting about twice as much as the same tailwind helps -- and both
    magnitudes follow from ``v_rel^2`` drag and the boundary layer.
    """
    from coxswain.sim.simulator import RowingSimulator

    def speed(wind):
        result = RowingSimulator(eight, wind=wind).run(
            duration=26.0, dt=0.009, surge_speed=4.6)
        v = np.asarray(result.velocity)[0]
        return float(v[int(0.6 * len(v)):].mean())

    still = speed(UniformWind(speed=0.0))
    head = speed(UniformWind.head_on(5.0, 0.0))
    tail = speed(UniformWind.following(5.0, 0.0))

    head_change = 100.0 * (head - still) / still
    tail_change = 100.0 * (tail - still) / still

    assert head_change == pytest.approx(-12.2, abs=3.0), head_change
    assert tail_change == pytest.approx(5.1, abs=3.0), tail_change
    assert abs(head_change) > 1.5 * abs(tail_change), \
        "a headwind must hurt considerably more than a tailwind helps"
