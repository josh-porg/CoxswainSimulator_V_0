"""Unit tests for the de Leva body-segment inertial parameters."""

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
# the source table
# --------------------------------------------------------------------------
@pytest.mark.parametrize("table,label", [(DE_LEVA_MALE, "male"),
                                         (DE_LEVA_FEMALE, "female")])
def test_whole_body_mass_fractions_sum_to_one(table, label):
    """Head + trunk + 2 x (each paired segment) must be the whole body."""
    total = 0.0
    for spec in table.values():
        total += spec.mass_fraction * (2 if spec.paired else 1)
    assert total == pytest.approx(1.0, abs=5e-4), (
        f"{label} mass fractions sum to {total:.5f}"
    )


@pytest.mark.parametrize("table,expected", [(DE_LEVA_MALE, 0.4346),
                                            (DE_LEVA_FEMALE, 0.4257)])
def test_trunk_thirds_sum_to_the_whole_trunk_mass(table, expected):
    """de Leva's whole-trunk row is 43.46% (male) / 42.57% (female)."""
    thirds = sum(table[n].mass_fraction
                 for n in ("upper_trunk", "mid_trunk", "lower_trunk"))
    assert thirds == pytest.approx(expected, abs=5e-4)


def test_trunk_thirds_stack_to_the_whole_trunk_length():
    """LPT + MPT + UPT must equal de Leva's CERV->MIDH trunk length.

    This is why the upper trunk uses the CERV->XYPH endpoints rather than
    SUPRA->XYPH: only then do the three sub-segments form one rigid link
    that the kinematic chain can rotate as a unit.
    """
    stature = REFERENCE_STATURE["male"]
    total_mm = sum(DE_LEVA_MALE[n].length_fraction * stature * 1000.0
                   for n in ("lower_trunk", "mid_trunk", "upper_trunk"))
    assert total_mm == pytest.approx(603.3, abs=0.5)


def test_female_trunk_thirds_stack_to_the_whole_trunk_length():
    stature = REFERENCE_STATURE["female"]
    total_mm = sum(DE_LEVA_FEMALE[n].length_fraction * stature * 1000.0
                   for n in ("lower_trunk", "mid_trunk", "upper_trunk"))
    assert total_mm == pytest.approx(614.8, abs=0.5)


def test_all_fractions_are_physically_sensible():
    for table in (DE_LEVA_MALE, DE_LEVA_FEMALE):
        for name, spec in table.items():
            assert 0.0 < spec.mass_fraction < 0.5, name
            assert 0.0 < spec.length_fraction < 0.5, name
            assert 0.0 < spec.com_fraction < 1.0, name


# --------------------------------------------------------------------------
# the scaled 12-segment model
# --------------------------------------------------------------------------
@pytest.fixture
def rower():
    return RowerAnthropometry(mass=85.0, stature=1.88, sex="male")


def test_model_has_exactly_twelve_segments(rower):
    assert len(rower.segments) == N_SEGMENTS == 12


def test_segment_masses_sum_to_body_mass(rower):
    assert rower.total_segment_mass == pytest.approx(85.0, rel=1e-3)


@pytest.mark.parametrize("mass", [55.0, 68.0, 85.0, 106.0])
def test_segment_masses_sum_to_body_mass_at_any_mass(mass):
    anthro = RowerAnthropometry(mass=mass, stature=1.85)
    assert anthro.total_segment_mass == pytest.approx(mass, rel=1e-3)


@pytest.mark.parametrize("sex", ["male", "female"])
def test_both_sexes_conserve_mass(sex):
    anthro = RowerAnthropometry(mass=70.0, stature=1.75, sex=sex)
    assert anthro.total_segment_mass == pytest.approx(70.0, rel=1e-3)


def test_segment_names_are_unique(rower):
    names = [s.name for s in rower.segments]
    assert len(set(names)) == len(names)


def test_paired_segments_have_matching_masses(rower):
    for base in ("upper_arm", "forearm_hand", "thigh", "shank_foot"):
        port = rower.by_name(f"{base}_port")
        starboard = rower.by_name(f"{base}_starboard")
        assert port.mass == pytest.approx(starboard.mass)
        assert port.length == pytest.approx(starboard.length)
        assert port.side == PORT
        assert starboard.side == STARBOARD


def test_centreline_segments_are_marked_as_such(rower):
    for name in ("head", "upper_trunk", "mid_trunk", "lower_trunk"):
        assert rower.by_name(name).side == CENTRELINE


def test_masses_scale_linearly_with_body_mass():
    light = RowerAnthropometry(mass=60.0, stature=1.80)
    heavy = RowerAnthropometry(mass=120.0, stature=1.80)
    np.testing.assert_allclose(heavy.segment_masses,
                               2.0 * light.segment_masses, rtol=1e-12)


def test_lengths_scale_linearly_with_stature():
    short = RowerAnthropometry(mass=80.0, stature=1.60)
    tall = RowerAnthropometry(mass=80.0, stature=1.92)
    ratio = 1.92 / 1.60
    assert tall.length("thigh") == pytest.approx(ratio * short.length("thigh"))


def test_lengths_are_independent_of_mass():
    light = RowerAnthropometry(mass=60.0, stature=1.85)
    heavy = RowerAnthropometry(mass=110.0, stature=1.85)
    assert light.length("shank") == pytest.approx(heavy.length("shank"))


def test_reference_athlete_reproduces_table_lengths():
    """At the reference stature, lengths must equal de Leva Table 4 in mm."""
    anthro = RowerAnthropometry(mass=REFERENCE_MASS["male"],
                                stature=REFERENCE_STATURE["male"])
    expected_mm = {"thigh": 422.2, "shank": 434.0, "upper_arm": 281.7,
                   "forearm": 268.9, "hand": 86.2, "head": 203.3,
                   "mid_trunk": 215.5, "lower_trunk": 145.7}
    for name, mm in expected_mm.items():
        assert anthro.length(name) * 1000.0 == pytest.approx(mm, abs=0.2), name


# --------------------------------------------------------------------------
# lumping
# --------------------------------------------------------------------------
def test_lumped_forearm_hand_conserves_mass(rower):
    lumped = rower.by_name("forearm_hand_port").mass
    separate = (DE_LEVA_MALE["forearm"].mass_fraction
                + DE_LEVA_MALE["hand"].mass_fraction) * 85.0
    assert lumped == pytest.approx(separate, rel=1e-12)


def test_lumped_shank_foot_conserves_mass(rower):
    lumped = rower.by_name("shank_foot_port").mass
    separate = (DE_LEVA_MALE["shank"].mass_fraction
                + DE_LEVA_MALE["foot"].mass_fraction) * 85.0
    assert lumped == pytest.approx(separate, rel=1e-12)


def test_lumped_segment_length_is_the_sum_of_its_parts(rower):
    lumped = rower.by_name("shank_foot_port").length
    assert lumped == pytest.approx(rower.length("shank") + rower.length("foot"))


def test_lumped_centre_of_mass_lies_between_the_two_parts(rower):
    """The joined CM must sit between the two component CMs."""
    segment = rower.by_name("forearm_hand_port")
    forearm_len = rower.length("forearm")
    proximal_cm = DE_LEVA_MALE["forearm"].com_fraction * forearm_len
    distal_cm = forearm_len + DE_LEVA_MALE["hand"].com_fraction * rower.length("hand")
    lumped_cm = segment.com_fraction * segment.length
    assert proximal_cm < lumped_cm < distal_cm


def test_trunk_stack_reproduces_the_whole_trunk_centre_of_mass(rower):
    """Aggregate CM of the three trunk masses must match de Leva's whole-trunk row.

    de Leva gives the whole trunk (CERV->MIDH, 603.3 mm male) with its CM at
    51.38% from the cervicale, i.e. 293.3 mm above the hip.  Stacking the
    three sub-segments must land in the same place.
    """
    lower = rower.length("lower_trunk")
    mid = rower.length("mid_trunk")
    upper = rower.length("upper_trunk")

    heights = {
        "lower_trunk": lower * (1 - rower.by_name("lower_trunk").com_fraction),
        "mid_trunk": lower + mid * (1 - rower.by_name("mid_trunk").com_fraction),
        "upper_trunk": lower + mid
                       + upper * (1 - rower.by_name("upper_trunk").com_fraction),
    }
    masses = {n: rower.by_name(n).mass for n in heights}
    aggregate = (sum(masses[n] * heights[n] for n in heights)
                 / sum(masses.values()))

    trunk_length = lower + mid + upper
    expected = trunk_length * (1.0 - 0.5138)
    assert aggregate == pytest.approx(expected, rel=0.02)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs,match", [
    ({"mass": 0.0}, "body mass must be positive"),
    ({"mass": -5.0}, "body mass must be positive"),
    ({"stature": 0.0}, "stature must be positive"),
    ({"sex": "unknown"}, "sex must be one of"),
])
def test_invalid_construction_is_rejected(kwargs, match):
    base = {"mass": 80.0, "stature": 1.85, "sex": "male"}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        RowerAnthropometry(**base)


def test_unknown_segment_name_raises(rower):
    with pytest.raises(KeyError, match="no segment named"):
        rower.by_name("tail")
