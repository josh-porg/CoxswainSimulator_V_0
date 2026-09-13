"""Segment moments of inertia, from de Leva (1996) Table 4 radii of gyration.

Groundwork for phase 4.3's torque-driven chain.  Checked against the table's
own definition, ``I = m (r l)^2``, and against the parallel-axis theorem for
the segments the model lumps.
"""

from __future__ import annotations

import pytest

from coxswain.crew.anthropometry import (
    DE_LEVA_MALE, DE_LEVA_RADII, REFERENCE_STATURE, RowerAnthropometry)


def test_a_segment_moment_is_the_tables_own_definition():
    rower = RowerAnthropometry(mass=73.0, stature=REFERENCE_STATURE["male"])
    thigh = rower.by_name("thigh_port")
    spec = DE_LEVA_MALE["thigh"]
    mass = spec.mass_fraction * 73.0
    length = spec.length_fraction * REFERENCE_STATURE["male"]
    for axis, percent in enumerate(DE_LEVA_RADII["male"]["thigh"]):
        assert thigh.inertia[axis] == pytest.approx(
            mass * (percent / 100.0 * length) ** 2, rel=1e-12)
    # on de Leva's own reference subject: 10.34 kg, 0.4222 m, r = 32.9 %
    assert thigh.inertia[0] == pytest.approx(0.1995, abs=5e-4)


def test_moments_scale_with_mass_and_the_square_of_stature():
    small = RowerAnthropometry(mass=70.0, stature=1.70)
    large = RowerAnthropometry(mass=91.0, stature=1.87)
    ratio = (91.0 / 70.0) * (1.87 / 1.70) ** 2
    for a, b in zip(small.segments, large.segments):
        for axis in range(3):
            assert b.inertia[axis] == pytest.approx(ratio * a.inertia[axis],
                                                    rel=1e-12)


@pytest.mark.parametrize("lumped,proximal,distal", [
    ("forearm_hand_port", "forearm", "hand"),
    ("shank_foot_port", "shank", "foot"),
])
def test_a_lumped_segment_obeys_the_parallel_axis_theorem(lumped, proximal,
                                                          distal):
    rower = RowerAnthropometry()
    seg = rower.by_name(lumped)
    parts = []
    for name in (proximal, distal):
        spec = rower._table[name]
        mass = spec.mass_fraction * rower.mass
        length = spec.length_fraction * rower.stature
        radii = DE_LEVA_RADII["male"][name]
        parts.append((mass, length, spec.com_fraction,
                      [mass * (r / 100.0 * length) ** 2 for r in radii]))
    (m_p, l_p, c_p, i_p), (m_d, l_d, c_d, i_d) = parts
    centre = seg.com_fraction * seg.length
    offset = (m_p * (c_p * l_p - centre) ** 2
              + m_d * (l_p + c_d * l_d - centre) ** 2)
    assert seg.inertia[0] == pytest.approx(i_p[0] + i_d[0] + offset, rel=1e-12)
    assert seg.inertia[1] == pytest.approx(i_p[1] + i_d[1] + offset, rel=1e-12)
    # both CMs on the long axis: no offset about it
    assert seg.inertia[2] == pytest.approx(i_p[2] + i_d[2], rel=1e-12)
    assert offset > 0.0


@pytest.mark.parametrize("sex", ["male", "female"])
def test_every_segment_has_positive_moments(sex):
    rower = RowerAnthropometry(sex=sex)
    for segment in rower.segments:
        assert all(value > 0.0 for value in segment.inertia), segment.name


def test_both_sexes_carry_radii_for_every_segment_the_model_uses():
    assert set(DE_LEVA_RADII["male"]) == set(DE_LEVA_RADII["female"]) \
        == set(DE_LEVA_MALE)
