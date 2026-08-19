"""Unit tests for the de Leva (1996) body-segment parameter tables."""

import numpy as np
import pytest

from coxswain.crew import anthropometry
from coxswain.crew.anthropometry import (
    CENTRELINE,
    DE_LEVA_FEMALE,
    DE_LEVA_MALE,
    N_SEGMENTS,
    PORT,
    REFERENCE_MASS,
    REFERENCE_STATURE,
    STARBOARD,
    RowerAnthropometry,
)


# --------------------------------------------------------------------------
# the raw table
# --------------------------------------------------------------------------
@pytest.mark.parametrize("table", [DE_LEVA_MALE, DE_LEVA_FEMALE])
def test_whole_body_mass_fractions_sum_to_one(table):
    """head + trunk(3) + 2 x (arm, forearm, hand, thigh, shank, foot) = 1."""
    total = sum(
        spec.mass_fraction * (2 if spec.paired else 1)
        for spec in table.values()
    )
    assert total == pytest.approx(1.0, abs=2e-4)


@pytest.mark.parametrize("table", [DE_LEVA_MALE, DE_LEVA_FEMALE])
def test_every_fraction_is_physical(table):
    for name, spec in table.items():
        assert 0.0 < spec.mass_fraction < 0.5, name
        assert 0.0 < spec.length_fraction < 0.5, name
        assert 0.0 < spec.com_fraction < 1.0, name


@pytest.mark.parametrize("table,stature,expected_mm", [
    (DE_LEVA_MALE, REFERENCE_STATURE["male"], 603.3),
    (DE_LEVA_FEMALE, REFERENCE_STATURE["female"], 614.8),
])
def test_trunk_subsegments_stack_to_the_whole_trunk_length(table, stature,
                                                           expected_mm):
    """de Leva's whole-trunk row (CERV->MIDH) must equal LPT + MPT + UPT.

    This is the consistency check that forced the UPT endpoints to be the
    CERV->XYPH pair rather than SUPRA->XYPH.
    """
    stacked = sum(table[name].length_fraction
                  for name in ("lower_trunk", "mid_trunk", "upper_trunk"))
    assert stacked * stature * 1000.0 == pytest.approx(expected_mm, abs=0.5)


def test_male_head_mass_fraction_matches_the_published_value():
    assert DE_LEVA_MALE["head"].mass_fraction == pytest.approx(0.0694)


def test_male_thigh_mass_fraction_matches_the_published_value():
    assert DE_LEVA_MALE["thigh"].mass_fraction == pytest.approx(0.1416)


def test_reference_subject_constants():
    assert REFERENCE_MASS["male"] == pytest.approx(73.0)
    assert REFERENCE_STATURE["male"] == pytest.approx(1.741)
    assert REFERENCE_MASS["female"] == pytest.approx(61.9)
    assert REFERENCE_STATURE["female"] == pytest.approx(1.735)


# --------------------------------------------------------------------------
# the 12-segment model
# --------------------------------------------------------------------------
def test_model_has_exactly_twelve_segments():
    """Formaggia et al. section 4.2 specify p = 12."""
    assert len(RowerAnthropometry().segments) == N_SEGMENTS == 12


@pytest.mark.parametrize("sex", ["male", "female"])
@pytest.mark.parametrize("mass", [60.0, 85.0, 106.0])
def test_segment_masses_sum_to_body_mass(sex, mass):
    anthro = RowerAnthropometry(mass=mass, stature=1.85, sex=sex)
    assert anthro.total_segment_mass == pytest.approx(mass, rel=2e-4)


def test_segment_masses_scale_linearly_with_body_mass():
    light = RowerAnthropometry(mass=60.0).segment_masses
    heavy = RowerAnthropometry(mass=120.0).segment_masses
    np.testing.assert_allclose(heavy, 2.0 * light, rtol=1e-12)


def test_segment_lengths_scale_linearly_with_stature():
    short = RowerAnthropometry(stature=1.70)
    tall = RowerAnthropometry(stature=1.90)
    ratio = 1.90 / 1.70
    for name in ("thigh", "shank", "upper_arm"):
        assert tall.length(name) == pytest.approx(ratio * short.length(name),
                                                  rel=1e-12)


def test_paired_segments_come_in_port_and_starboard():
    segments = RowerAnthropometry().segments
    sides = [s.side for s in segments]
    assert sides.count(CENTRELINE) == 4
    assert sides.count(PORT) == 4
    assert sides.count(STARBOARD) == 4


def test_port_and_starboard_segments_have_equal_mass():
    anthro = RowerAnthropometry()
    for stem in ("upper_arm", "forearm_hand", "thigh", "shank_foot"):
        port = anthro.by_name(f"{stem}_port")
        starboard = anthro.by_name(f"{stem}_starboard")
        assert port.mass == pytest.approx(starboard.mass)
        assert port.length == pytest.approx(starboard.length)


def test_lumped_forearm_hand_conserves_mass_and_length():
    anthro = RowerAnthropometry(mass=85.0, stature=1.88)
    lumped = anthro.by_name("forearm_hand_port")
    expected_mass = 85.0 * (DE_LEVA_MALE["forearm"].mass_fraction
                            + DE_LEVA_MALE["hand"].mass_fraction)
    expected_length = anthro.length("forearm") + anthro.length("hand")
    assert lumped.mass == pytest.approx(expected_mass, rel=1e-12)
    assert lumped.length == pytest.approx(expected_length, rel=1e-12)


def test_lumped_centre_of_mass_lies_between_the_two_component_centres():
    anthro = RowerAnthropometry(mass=85.0, stature=1.88)
    lumped = anthro.by_name("shank_foot_port")
    shank_com = DE_LEVA_MALE["shank"].com_fraction * anthro.length("shank")
    foot_com = anthro.length("shank") + (DE_LEVA_MALE["foot"].com_fraction
                                         * anthro.length("foot"))
    lumped_com = lumped.com_fraction * lumped.length
    assert shank_com < lumped_com < foot_com


def test_lumped_centre_of_mass_is_the_mass_weighted_mean():
    anthro = RowerAnthropometry(mass=85.0, stature=1.88)
    lumped = anthro.by_name("forearm_hand_port")

    m_f = 85.0 * DE_LEVA_MALE["forearm"].mass_fraction
    m_h = 85.0 * DE_LEVA_MALE["hand"].mass_fraction
    d_f = DE_LEVA_MALE["forearm"].com_fraction * anthro.length("forearm")
    d_h = anthro.length("forearm") + (DE_LEVA_MALE["hand"].com_fraction
                                      * anthro.length("hand"))
    expected = (m_f * d_f + m_h * d_h) / (m_f + m_h)

    assert lumped.com_fraction * lumped.length == pytest.approx(expected,
                                                                rel=1e-12)


def test_trunk_stack_reproduces_the_whole_trunk_centre_of_mass():
    """Stacking LPT/MPT/UPT must put the aggregate trunk CM where de Leva's
    whole-trunk row puts it: 293 mm above the hip for the male reference."""
    anthro = RowerAnthropometry(mass=REFERENCE_MASS["male"],
                                stature=REFERENCE_STATURE["male"])
    heights, masses = [], []
    running = 0.0
    for name in ("lower_trunk", "mid_trunk", "upper_trunk"):
        segment = anthro.by_name(name)
        heights.append(running + segment.length * (1.0 - segment.com_fraction))
        masses.append(segment.mass)
        running += segment.length

    aggregate = np.average(heights, weights=masses)
    assert aggregate * 1000.0 == pytest.approx(293.3, abs=6.0)


def test_by_name_raises_for_an_unknown_segment():
    with pytest.raises(KeyError, match="no segment named"):
        RowerAnthropometry().by_name("tail")


def test_segment_masses_are_ordered_consistently_with_segments():
    anthro = RowerAnthropometry()
    np.testing.assert_allclose(
        anthro.segment_masses, [s.mass for s in anthro.segments]
    )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs,message", [
    ({"mass": 0.0}, "body mass must be positive"),
    ({"mass": -5.0}, "body mass must be positive"),
    ({"stature": 0.0}, "stature must be positive"),
    ({"sex": "other"}, "sex must be one of"),
])
def test_constructor_validates_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RowerAnthropometry(**kwargs)


def test_male_and_female_tables_differ():
    male = RowerAnthropometry(mass=75.0, stature=1.80, sex="male")
    female = RowerAnthropometry(mass=75.0, stature=1.80, sex="female")
    assert male.by_name("thigh_port").mass != pytest.approx(
        female.by_name("thigh_port").mass)


def test_module_exports_are_importable():
    for name in anthropometry.__all__:
        assert hasattr(anthropometry, name), name
