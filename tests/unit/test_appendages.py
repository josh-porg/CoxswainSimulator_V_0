"""Unit tests for the skeg and rudder.

The yaw-rate sign is pinned hard here: getting it backwards turns the skeg
from a damper into an amplifier, which made the boat spin up without bound
the first time the 6-DOF model ran.
"""

import numpy as np
import pytest

from coxswain.hydro.appendages import (
    RUDDER_EIGHT,
    SKEG_EIGHT,
    LiftingSurface,
    surface_load,
)
from coxswain.hydro.resistance import FRESH_WATER

FORWARD = np.array([5.0, 0.0, 0.0])


def flow_at_sideslip(speed: float, sideslip_deg: float) -> np.ndarray:
    angle = np.radians(sideslip_deg)
    return speed * np.array([np.cos(angle), np.sin(angle), 0.0])


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_area_and_aspect_ratio():
    surface = LiftingSurface(span=0.2, chord=0.1, position=np.zeros(3))
    assert surface.area == pytest.approx(0.02)
    assert surface.aspect_ratio == pytest.approx(0.2 ** 2 / 0.02)


def test_skeg_is_a_very_low_aspect_ratio_surface():
    assert 0.4 < SKEG_EIGHT.aspect_ratio < 1.0


def test_polhamus_slope_is_well_below_two_pi_at_low_aspect_ratio():
    """The whole reason for the correction: 2 pi would be badly wrong here."""
    assert SKEG_EIGHT.lift_curve_slope < 0.25 * 2 * np.pi


def test_polhamus_slope_approaches_two_pi_at_high_aspect_ratio():
    slender = LiftingSurface(span=8.0, chord=0.2, position=np.zeros(3))
    assert slender.lift_curve_slope == pytest.approx(2 * np.pi, rel=0.15)


def test_lift_curve_slope_increases_with_aspect_ratio():
    slopes = [
        LiftingSurface(span=s, chord=0.2, position=np.zeros(3)).lift_curve_slope
        for s in (0.1, 0.4, 1.0, 3.0)
    ]
    assert all(b > a for a, b in zip(slopes, slopes[1:]))


def test_sweep_reduces_the_lift_curve_slope():
    plain = LiftingSurface(span=0.5, chord=0.2, position=np.zeros(3))
    swept = LiftingSurface(span=0.5, chord=0.2, position=np.zeros(3),
                           sweep=np.radians(30.0))
    assert swept.lift_curve_slope < plain.lift_curve_slope


@pytest.mark.parametrize("kwargs,message", [
    ({"span": 0.0, "chord": 0.1}, "span and chord must be positive"),
    ({"span": 0.1, "chord": -0.1}, "span and chord must be positive"),
])
def test_surface_validates_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LiftingSurface(position=np.zeros(3), **kwargs)


def test_surface_rejects_a_bad_position():
    with pytest.raises(ValueError, match="length-3"):
        LiftingSurface(span=0.1, chord=0.1, position=np.zeros(2))


# --------------------------------------------------------------------------
# directional stability -- the skeg
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sideslip_deg", [2.0, 5.0, -3.0])
def test_skeg_turns_the_bow_into_the_flow(sideslip_deg):
    """Weathercock stability: the yaw moment shares the sign of sideslip.

    Sideslip to port means the boat is tracking to port of its heading;
    the fin swings the nose that way to line the hull up with its path.
    """
    _, moment = surface_load(SKEG_EIGHT, flow_at_sideslip(5.0, sideslip_deg),
                             0.0, 0.0, FRESH_WATER)
    assert moment[2] * sideslip_deg > 0.0


def test_skeg_side_force_opposes_sideslip():
    force, _ = surface_load(SKEG_EIGHT, flow_at_sideslip(5.0, 5.0), 0.0, 0.0,
                            FRESH_WATER)
    assert force[1] < 0.0


@pytest.mark.parametrize("yaw_rate", [0.1, 0.3, -0.2])
def test_skeg_damps_yaw_rate(yaw_rate):
    """The sign that made the first 6-DOF run spin up without bound."""
    _, moment = surface_load(SKEG_EIGHT, FORWARD, yaw_rate, 0.0, FRESH_WATER)
    assert moment[2] * yaw_rate < 0.0, "skeg must oppose yaw rate, not feed it"


def test_yaw_damping_grows_with_distance_aft():
    near = LiftingSurface(span=0.129, chord=0.202,
                          position=np.array([-2.0, 0.0, -0.18]))
    far = LiftingSurface(span=0.129, chord=0.202,
                         position=np.array([-6.0, 0.0, -0.18]))
    _, near_moment = surface_load(near, FORWARD, 0.2, 0.0, FRESH_WATER)
    _, far_moment = surface_load(far, FORWARD, 0.2, 0.0, FRESH_WATER)
    assert abs(far_moment[2]) > abs(near_moment[2])


def test_a_forward_fin_destabilises_the_weathercock_response():
    """Position sets *static* stability, not rate damping.

    A fin ahead of the centre of mass turns the bow *away* from the flow,
    which is why a skeg goes at the stern.
    """
    canard = LiftingSurface(span=0.129, chord=0.202,
                            position=np.array([+6.0, 0.0, -0.18]))
    _, moment = surface_load(canard, flow_at_sideslip(5.0, 5.0), 0.0, 0.0,
                             FRESH_WATER)
    assert moment[2] < 0.0


@pytest.mark.parametrize("x_ac", [-6.0, -2.0, 2.0, 6.0])
def test_every_fin_damps_yaw_rate_regardless_of_position(x_ac):
    """Rate damping comes from the fin's own motion through the water, so
    its sign does not depend on which side of the centre of mass it sits."""
    fin = LiftingSurface(span=0.129, chord=0.202,
                         position=np.array([x_ac, 0.0, -0.18]))
    _, moment = surface_load(fin, FORWARD, 0.2, 0.0, FRESH_WATER)
    assert moment[2] < 0.0


# --------------------------------------------------------------------------
# the rudder
# --------------------------------------------------------------------------
def test_positive_rudder_yaws_to_starboard():
    """The convention HeadingController relies on."""
    _, moment = surface_load(RUDDER_EIGHT, FORWARD, 0.0, np.radians(10.0),
                             FRESH_WATER)
    assert moment[2] < 0.0


def test_rudder_response_is_antisymmetric():
    _, port = surface_load(RUDDER_EIGHT, FORWARD, 0.0, np.radians(-10.0),
                           FRESH_WATER)
    _, starboard = surface_load(RUDDER_EIGHT, FORWARD, 0.0, np.radians(10.0),
                                FRESH_WATER)
    assert port[2] == pytest.approx(-starboard[2], rel=1e-12)


def test_rudder_moment_grows_with_deflection():
    moments = [
        abs(surface_load(RUDDER_EIGHT, FORWARD, 0.0, np.radians(d),
                         FRESH_WATER)[1][2])
        for d in (2.0, 6.0, 12.0)
    ]
    assert all(b > a for a, b in zip(moments, moments[1:]))


def test_rudder_deflection_is_limited():
    at_limit = surface_load(RUDDER_EIGHT, FORWARD, 0.0,
                            RUDDER_EIGHT.max_deflection, FRESH_WATER)[1][2]
    beyond = surface_load(RUDDER_EIGHT, FORWARD, 0.0,
                          3.0 * RUDDER_EIGHT.max_deflection, FRESH_WATER)[1][2]
    assert beyond == pytest.approx(at_limit, rel=1e-12)


def test_a_fixed_surface_ignores_deflection():
    _, undeflected = surface_load(SKEG_EIGHT, FORWARD, 0.0, 0.0, FRESH_WATER)
    _, deflected = surface_load(SKEG_EIGHT, FORWARD, 0.0, np.radians(15.0),
                                FRESH_WATER)
    np.testing.assert_allclose(undeflected, deflected, atol=1e-14)


def test_rudder_authority_grows_with_speed():
    slow = abs(surface_load(RUDDER_EIGHT, np.array([2.0, 0.0, 0.0]), 0.0,
                            np.radians(10.0), FRESH_WATER)[1][2])
    fast = abs(surface_load(RUDDER_EIGHT, np.array([6.0, 0.0, 0.0]), 0.0,
                            np.radians(10.0), FRESH_WATER)[1][2])
    assert fast == pytest.approx(9.0 * slow, rel=1e-9)


# --------------------------------------------------------------------------
# drag
# --------------------------------------------------------------------------
def test_lifting_surfaces_produce_induced_drag():
    force, _ = surface_load(SKEG_EIGHT, flow_at_sideslip(5.0, 6.0), 0.0, 0.0,
                            FRESH_WATER)
    assert force[0] < 0.0


def test_no_induced_drag_at_zero_incidence():
    force, _ = surface_load(SKEG_EIGHT, FORWARD, 0.0, 0.0, FRESH_WATER)
    assert force[0] == pytest.approx(0.0, abs=1e-12)


def test_induced_drag_is_quadratic_in_lift():
    """``C_D_i = C_L^2 / (pi AR e)`` -- quadratic in *lift*, which is the
    physics.  It used to be asserted as quadratic in *incidence*, which was
    the same statement only while lift was linear in incidence.  Once the
    Whicker-Fehlner cross-flow term went in, doubling the angle stopped
    exactly doubling the lift, and this test failed for a correct model.
    Written against lift, it holds either way."""
    from coxswain.hydro.appendages import lift_coefficient_at
    small, _ = surface_load(SKEG_EIGHT, flow_at_sideslip(5.0, 2.0), 0.0, 0.0,
                            FRESH_WATER)
    large, _ = surface_load(SKEG_EIGHT, flow_at_sideslip(5.0, 4.0), 0.0, 0.0,
                            FRESH_WATER)
    ratio = (lift_coefficient_at(SKEG_EIGHT, np.radians(4.0))
             / lift_coefficient_at(SKEG_EIGHT, np.radians(2.0))) ** 2
    assert large[0] == pytest.approx(ratio * small[0], rel=0.02)
    # and it is still very nearly quadratic in incidence at these angles
    assert 3.9 < large[0] / small[0] < 4.4


def test_induced_drag_is_small_compared_with_hull_resistance():
    force, _ = surface_load(SKEG_EIGHT, flow_at_sideslip(5.5, 3.0), 0.0, 0.0,
                            FRESH_WATER)
    assert abs(force[0]) < 20.0


# --------------------------------------------------------------------------
# degenerate cases
# --------------------------------------------------------------------------
def test_no_load_at_rest():
    force, moment = surface_load(SKEG_EIGHT, np.zeros(3), 0.0, 0.0,
                                 FRESH_WATER)
    np.testing.assert_allclose(force, np.zeros(3))
    np.testing.assert_allclose(moment, np.zeros(3))


def test_no_load_when_not_submerged():
    force, moment = surface_load(SKEG_EIGHT, FORWARD, 0.2, 0.1, FRESH_WATER,
                                 submerged=False)
    np.testing.assert_allclose(force, np.zeros(3))
    np.testing.assert_allclose(moment, np.zeros(3))


def test_moment_is_the_cross_product_of_position_and_force():
    force, moment = surface_load(SKEG_EIGHT, flow_at_sideslip(5.0, 4.0), 0.1,
                                 0.0, FRESH_WATER)
    np.testing.assert_allclose(moment, np.cross(SKEG_EIGHT.position, force),
                               atol=1e-12)


def test_a_deep_surface_also_produces_a_roll_moment():
    """Side force below the hull rolls the boat as well as yawing it."""
    force, moment = surface_load(SKEG_EIGHT, flow_at_sideslip(5.0, 5.0), 0.0,
                                 0.0, FRESH_WATER)
    assert abs(moment[0]) > 0.0
    assert np.sign(moment[0]) == np.sign(-SKEG_EIGHT.position[2] * force[1])


# --------------------------------------------------------------------------
# non-linear lift (Whicker-Fehlner)
# --------------------------------------------------------------------------
def test_small_angles_still_follow_the_lift_curve_slope():
    """The whole point of adding the non-linear term is that it changes
    nothing in the linear regime, so every stability derivative already
    validated stays put."""
    from coxswain.hydro.appendages import RUDDER_EIGHT, lift_coefficient_at
    # the cross-flow term is O(angle) relative to the linear one, so the
    # departure shrinks with the angle rather than sitting at some fixed
    # tolerance -- check that it does, which is the real claim.
    previous = None
    for degrees in (0.5, 0.25, 0.1, 0.05):
        angle = np.radians(degrees)
        linear = RUDDER_EIGHT.lift_curve_slope * angle
        departure = abs(lift_coefficient_at(RUDDER_EIGHT, angle) / linear - 1.0)
        assert departure < 0.01
        if previous is not None:
            assert departure < previous
        previous = departure
    assert previous < 1e-3


def test_lift_is_odd_in_the_angle():
    from coxswain.hydro.appendages import RUDDER_EIGHT, lift_coefficient_at
    for degrees in (3.0, 12.0, 25.0, 40.0):
        angle = np.radians(degrees)
        assert lift_coefficient_at(RUDDER_EIGHT, -angle) == pytest.approx(
            -lift_coefficient_at(RUDDER_EIGHT, angle))


def test_a_working_rudder_makes_more_lift_than_the_linear_law():
    """At 25 degrees the side-edge vortices are worth about 15%, and the
    gain grows with deflection -- which is what pulls the turn-rate
    response back towards proportional."""
    from coxswain.hydro.appendages import RUDDER_EIGHT, lift_coefficient_at
    gains = []
    for degrees in (8.0, 25.0):
        angle = np.radians(degrees)
        linear = RUDDER_EIGHT.lift_curve_slope * angle
        gains.append(lift_coefficient_at(RUDDER_EIGHT, angle) / linear - 1.0)
    assert 0.05 < gains[0] < 0.13
    assert 0.10 < gains[1] < 0.22
    assert gains[1] > gains[0]


def test_the_surface_stalls_instead_of_lifting_for_ever():
    """A low-aspect-ratio plate holds on to about 40 degrees and then
    gives up.  The old linear law would have gone on for ever, which
    matters for a skeg during a broach, not just for the rudder."""
    from coxswain.hydro.appendages import SKEG_EIGHT, lift_coefficient_at
    angles = np.radians(np.arange(5.0, 90.0, 1.0))
    lift = np.array([lift_coefficient_at(SKEG_EIGHT, a) for a in angles])
    peak = np.degrees(angles[int(np.argmax(lift))])
    # flat-plate data for aspect ratio near 0.6 peaks around 45-50 degrees
    assert 38.0 < peak < 55.0
    assert lift_coefficient_at(SKEG_EIGHT, np.radians(80.0)) < lift.max()


def test_zero_crossflow_coefficient_recovers_the_old_linear_surface():
    """The switch that lets the change be isolated in any later test."""
    import dataclasses
    from coxswain.hydro.appendages import RUDDER_EIGHT, lift_coefficient_at
    linearised = dataclasses.replace(RUDDER_EIGHT, crossflow_coefficient=0.0)
    angle = np.radians(20.0)
    expected = linearised.lift_curve_slope * np.sin(angle) * np.cos(angle)
    assert lift_coefficient_at(linearised, angle) == pytest.approx(expected)
