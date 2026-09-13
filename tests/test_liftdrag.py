r"""Tier 2 blade: lift and drag resolved against the angle of attack.

The coefficients are Caplan & Gardner's flume fits via a secondary source
([CG06a] in SOURCES), and are provisional until the primary is read.  These
tests check the model's *construction* -- that it reduces to tier 1's law
where it must, that its signs are right in every quadrant, and that it says
where its numbers came from.  They do not, and cannot, check the constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.crew.liftdrag import LiftDragBlade


@pytest.fixture
def blade():
    return LiftDragBlade.big_blade(outboard=2.56)


def _q(blade):
    return 0.5 * blade.density * blade.area


def test_the_constants_are_the_recorded_ones_and_say_so():
    big = LiftDragBlade.big_blade(outboard=2.5)
    macon = LiftDragBlade.macon(outboard=2.5)
    assert (big.lift_amplitude, big.drag_amplitude) == (1.25, 2.07)
    assert (macon.lift_amplitude, macon.drag_amplitude) == (1.24, 1.90)
    # A result built on these must not be able to pass for a primary one.
    assert "CG06a" in LiftDragBlade.PROVENANCE
    assert "provisional" in LiftDragBlade.PROVENANCE.lower()


def test_pure_normal_flow_is_tier_one_law(blade):
    """At alpha = 90 degrees: no tangential load, normal ``1/2 rho A A_d w^2``."""
    for w_n in (-3.0, -0.8, 1.4):
        f_n, f_t = blade.loads(0.0, w_n / blade.outboard,
                               np.zeros(2), side=+1)
        assert f_n == pytest.approx(
            -np.sign(w_n) * _q(blade) * blade.drag_amplitude * w_n ** 2,
            rel=1e-12)
        assert f_t == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("w_n", [-2.3, -0.4, 0.7, 2.9])
@pytest.mark.parametrize("w_a", [-1.8, -0.2, 0.5, 2.2])
def test_resolved_coefficients_match_explicit_lift_and_drag(blade, w_n, w_a):
    """The easy thing to get backwards is the lift's direction.

    Build drag along -w and lift perpendicular to w, oriented to push the
    plate against its normal motion, and project both onto the blade's axes.
    """
    speed = np.hypot(w_n, w_a)
    alpha = np.arctan2(abs(w_n), abs(w_a))
    q = _q(blade) * speed ** 2
    lift = q * blade.lift_amplitude * np.sin(2 * alpha)
    drag = q * blade.drag_amplitude * np.sin(alpha) ** 2
    w_hat = np.array([w_n, w_a]) / speed
    perp = np.sign(w_n) * np.sign(w_a) * np.array([-w_a, w_n]) / speed
    explicit = -drag * w_hat + lift * perp

    c_n, c_t = blade.coefficients(alpha)
    resolved = np.array([-np.sign(w_n) * q * c_n, -np.sign(w_a) * q * c_t])
    assert np.allclose(explicit, resolved, rtol=1e-10, atol=1e-9)
    assert abs((lift * perp) @ w_hat) < 1e-9
    assert np.sign(explicit[0]) == -np.sign(w_n)


def test_the_tangential_sign_is_a_prediction(blade):
    """For the Big Blade ``A_d - 2 A_l < 0``, so the tangential load points
    AGAINST the drag direction along the shaft. Pinned because it is the
    first thing to check against Grift's measured tangential traces."""
    assert blade.drag_amplitude - 2 * blade.lift_amplitude < 0
    alpha = np.radians(40.0)
    _c_n, c_t = blade.coefficients(alpha)
    assert c_t < 0.0


def test_a_blade_at_rest_in_still_water_carries_nothing(blade):
    assert blade.loads(0.3, 0.0, np.zeros(2), side=-1) == (0.0, 0.0)


def test_the_relative_velocity_is_resolved_on_the_blades_own_side():
    """A starboard oar's axes are the port oar's mirror image."""
    blade = LiftDragBlade.big_blade(outboard=2.5)
    u = np.array([4.0, 0.0])
    port = blade.relative_velocity(np.radians(20.0), -1.5, u, +1)
    starboard = blade.relative_velocity(np.radians(20.0), -1.5, u, -1)
    # Surge is along the centreline, so both sides see the same components.
    assert port == pytest.approx(starboard)
    # Sway is not: it projects with opposite sign onto the two sides' axes.
    sway = np.array([0.0, 0.5])
    p = blade.relative_velocity(np.radians(20.0), -1.5, sway, +1)
    s = blade.relative_velocity(np.radians(20.0), -1.5, sway, -1)
    assert p[1] == pytest.approx(-s[1])
