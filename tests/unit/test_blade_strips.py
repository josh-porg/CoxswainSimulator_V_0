"""Blade loads integrated across the span: the same laws, checked against their exact limits."""
import numpy as np
import pytest

from coxswain.crew.blade_strips import liftdrag_strips, slip_strips
from coxswain.crew.liftdrag import LiftDragBlade
from coxswain.crew.oarlock import BladeModel

SPAN = 0.43
CENTRE = 1.795


@pytest.fixture
def slip_blade():
    return BladeModel(c2=140.88, outboard=CENTRE)


def test_with_no_rotation_the_strips_are_the_centre_point_law(slip_blade):
    """Uniform slip along the span: every strip sees the centre's slip."""
    angle, v = np.radians(20.0), 4.6
    f, r_cp = slip_strips(slip_blade, angle, 0.0, v, SPAN)
    assert f == pytest.approx(float(slip_blade.normal_force(angle, 0.0, v)), rel=1e-12)
    assert r_cp == pytest.approx(CENTRE, rel=1e-12)


def test_pure_rotation_matches_the_closed_form(slip_blade):
    """v = 0: F = (C2/b) w^2 (r2^3 - r1^3) / 3, r_cp = (3/4)(r2^4 - r1^4)/(r2^3 - r1^3)."""
    rate = -2.0
    r1, r2 = CENTRE - SPAN / 2, CENTRE + SPAN / 2
    f, r_cp = slip_strips(slip_blade, 0.0, rate, 0.0, SPAN, strips=2001)
    assert f == pytest.approx(slip_blade.c2 / SPAN * rate ** 2 * (r2 ** 3 - r1 ** 3) / 3, rel=1e-6)
    assert r_cp == pytest.approx(0.75 * (r2 ** 4 - r1 ** 4) / (r2 ** 3 - r1 ** 3), rel=1e-6)
    # which exceeds the centre-point read by exactly 1 + b^2 / (12 l^2)
    centre = float(slip_blade.normal_force(0.0, rate, 0.0))
    assert f / centre == pytest.approx(1 + SPAN ** 2 / (12 * CENTRE ** 2), rel=1e-6)


def test_a_small_centre_slip_gives_a_larger_load_further_out(slip_blade):
    """Mid-drive: the rotation and the boat's motion nearly cancel at the centre, so the
    outer strips carry the load (SOURCES sec. 162: centre of pressure 1.86-1.93 m)."""
    angle, v = np.radians(5.0), 4.6
    rate = -(v * np.cos(angle) + 0.5) / CENTRE              # 0.5 m/s driving slip at the centre
    f, r_cp = slip_strips(slip_blade, angle, rate, v, SPAN)
    centre = float(slip_blade.normal_force(angle, rate, v))
    assert f > centre > 0.0
    assert r_cp > CENTRE + 0.03


def test_a_slip_that_changes_sign_along_the_blade_nets_out(slip_blade):
    """Zero slip at the centre: inner strips brake and outer strips drive, equally, so the
    force nets to zero where the centre-point law also reads zero."""
    v, angle = 4.0, 0.0
    rate = -v / CENTRE
    f, _ = slip_strips(slip_blade, angle, rate, v, SPAN, strips=4001)
    assert float(slip_blade.normal_force(angle, rate, v)) == pytest.approx(0.0, abs=1e-9)
    assert abs(f) < 1e-6 * slip_blade.c2                   # symmetric about the centre: nets to zero


def test_strip_count_converges(slip_blade):
    args = (np.radians(10.0), -2.4, 4.5, SPAN)
    fine, _ = slip_strips(slip_blade, *args, strips=4001)
    coarse, _ = slip_strips(slip_blade, *args)
    assert coarse == pytest.approx(fine, rel=2e-3)


def test_the_geometry_is_refused_when_impossible(slip_blade):
    with pytest.raises(ValueError):
        slip_strips(slip_blade, 0.0, -2.0, 4.0, 0.0)
    with pytest.raises(ValueError):
        slip_strips(slip_blade, 0.0, -2.0, 4.0, 2 * CENTRE)
    with pytest.raises(ValueError):
        slip_strips(slip_blade, 0.0, -2.0, 4.0, SPAN, strips=2)


def test_tier_two_strips_reduce_to_its_centre_point_loads_without_rotation():
    blade = LiftDragBlade.big_blade(outboard=CENTRE, area=0.083)
    angle, lock = np.radians(30.0), (4.4, 0.0)
    f_n, f_t, r_cp = liftdrag_strips(blade, angle, 0.0, lock, 1, SPAN)
    c_n, c_t = blade.loads(angle, 0.0, lock, 1)
    assert f_n == pytest.approx(c_n, rel=1e-12)
    assert f_t == pytest.approx(c_t, rel=1e-12)
    assert r_cp == pytest.approx(CENTRE, rel=1e-12)


def test_tier_two_strips_converge_to_the_centre_point_as_the_span_shrinks():
    blade = LiftDragBlade.big_blade(outboard=CENTRE, area=0.083)
    angle, rate, lock = np.radians(15.0), -2.3, (4.6, 0.0)
    c_n, c_t = blade.loads(angle, rate, lock, 1)
    f_n, f_t, _ = liftdrag_strips(blade, angle, rate, lock, 1, 1e-4)
    assert f_n == pytest.approx(c_n, rel=1e-6)
    assert f_t == pytest.approx(c_t, rel=1e-6)
