"""Unit tests for the published stroke-kinematics datasets.

The point of these tests is provenance and internal consistency: the
numbers are transcribed from papers, so what can be checked is that the
transcription is self-consistent, that the derived link angles follow from
the stated definitions, and that the implied geometry is physical.
"""

import numpy as np
import pytest

from coxswain.crew import stroke_data
from coxswain.crew.stroke_data import (
    CAPLAN_GARDNER_2010,
    DATASETS,
    KEYFRAME_NAMES,
    StrokeKinematicsDataset,
    default_dataset,
)


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------
def test_default_dataset_is_caplan_gardner():
    assert default_dataset() is CAPLAN_GARDNER_2010


def test_every_dataset_is_registered_under_its_own_name():
    for name, dataset in DATASETS.items():
        assert dataset.name == name


def test_every_dataset_carries_a_citation():
    for dataset in DATASETS.values():
        assert len(dataset.citation) > 30, dataset.name


@pytest.mark.parametrize("dataset", DATASETS.values(), ids=lambda d: d.name)
def test_every_dataset_has_one_value_per_keyframe(dataset):
    for field in ("shank", "knee", "hip", "trunk"):
        assert len(getattr(dataset, field)) == len(KEYFRAME_NAMES)


def test_mismatched_keyframe_count_is_rejected():
    with pytest.raises(ValueError, match="one per keyframe"):
        StrokeKinematicsDataset(
            name="broken", citation="x" * 40, rate=30.0,
            shank=(1.0, 2.0), knee=(1.0, 2.0, 3.0, 4.0),
            hip=(1.0, 2.0, 3.0, 4.0), trunk=(1.0, 2.0, 3.0, 4.0),
        )


# --------------------------------------------------------------------------
# derived link angles follow from the stated definitions
# --------------------------------------------------------------------------
def test_shank_is_stored_bow_referenced():
    """Caplan's 91.6 deg at the catch is a vertical shin.

    They measure from the ground on the stern side, so the bow-referenced
    link angle is 180 - 91.6 = 88.4 deg -- near vertical either way, which
    is the sanity check that the conversion has not been applied twice.
    """
    assert CAPLAN_GARDNER_2010.shank[0] == pytest.approx(88.4, abs=0.05)
    assert CAPLAN_GARDNER_2010.shank[2] == pytest.approx(11.5, abs=0.05)


def test_thigh_follows_from_the_interior_knee_angle():
    dataset = CAPLAN_GARDNER_2010
    expected = np.asarray(dataset.knee) - 180.0 + np.asarray(dataset.shank)
    np.testing.assert_allclose(dataset.thigh, expected, atol=1e-12)


def test_trunk_link_converts_from_vertical_to_bow_referenced():
    dataset = CAPLAN_GARDNER_2010
    np.testing.assert_allclose(dataset.trunk_link,
                               90.0 - np.asarray(dataset.trunk), atol=1e-12)


def test_trunk_leans_towards_the_stern_at_the_catch():
    """Negative measured trunk angle = forward lean = link angle past 90 deg."""
    assert CAPLAN_GARDNER_2010.trunk[0] < 0.0
    assert CAPLAN_GARDNER_2010.trunk_link[0] > 90.0


def test_trunk_lays_back_towards_the_bow_at_the_finish():
    assert CAPLAN_GARDNER_2010.trunk[2] > 0.0
    assert CAPLAN_GARDNER_2010.trunk_link[2] < 90.0


# --------------------------------------------------------------------------
# internal consistency: the four angle sets describe ONE linkage
# --------------------------------------------------------------------------
@pytest.mark.parametrize("index,label", [(1, "mid_drive"), (2, "finish"),
                                         (3, "mid_recovery")])
def test_hip_angle_is_reproduced_from_the_other_three(index, label):
    """Shank + knee + trunk determine the hip angle; check against measurement.

    Agreement to better than a degree at three of the four keyframes is
    what justifies driving the model from this dataset at all.
    """
    dataset = CAPLAN_GARDNER_2010
    assert dataset.hip_reconstructed[index] == pytest.approx(
        dataset.hip[index], abs=1.0), label


def test_hip_angle_at_the_catch_agrees_within_the_measurement_scatter():
    """The catch is the one keyframe that does not agree closely.

    The residual is 17.4 deg against a +-13.5 deg standard deviation on the
    catch knee angle, so it is inside the scatter -- but it is real, and
    pinning it here means a future change to the dataset cannot quietly
    make it worse.
    """
    dataset = CAPLAN_GARDNER_2010
    residual = abs(dataset.hip_reconstructed[0] - dataset.hip[0])
    assert 15.0 < residual < 20.0


def test_implied_seat_travel_matches_published_slide_excursion():
    """0.60-0.70 m is what on-water crews use of a ~0.75 m slide."""
    dataset = CAPLAN_GARDNER_2010
    offsets = dataset.hip_offset(shank_length=0.469, thigh_length=0.456)
    travel = offsets.max() - offsets.min()
    assert 0.58 <= travel <= 0.70


def test_catch_and_finish_agree_on_the_seat_height():
    """The level-track constraint, recovered from the data.

    A seat runs on a rail, so the hip height above the ankle must be the
    same at every instant.  The catch and finish keyframes agree on it to
    about a millimetre, which is what the model's seat-height calibration
    relies on.
    """
    dataset = CAPLAN_GARDNER_2010
    heights = dataset.hip_height(shank_length=0.469, thigh_length=0.456)
    assert abs(heights[0] - heights[2]) < 0.005


def test_mid_stroke_keyframes_imply_a_higher_hip():
    """The known departure, pinned.

    Taken literally the mid-drive and mid-recovery angles put the hip
    ~7 cm higher than the catch and finish do.  A seat cannot do that, so
    the model defaults to the level-seat constraint instead; this test
    documents the size of the discrepancy being set aside.
    """
    dataset = CAPLAN_GARDNER_2010
    heights = dataset.hip_height(shank_length=0.469, thigh_length=0.456)
    assert 0.05 < (heights[1] - heights[0]) < 0.09


# --------------------------------------------------------------------------
# keyframe phases
# --------------------------------------------------------------------------
def test_keyframe_phases_bracket_the_drive():
    """Caplan define the mid-points as 50% of each phase by time."""
    drive = 0.4
    phases = CAPLAN_GARDNER_2010.keyframe_phases(drive)
    assert phases[0] == pytest.approx(0.0)
    assert phases[1] == pytest.approx(0.5 * drive)
    assert phases[2] == pytest.approx(drive)
    assert phases[3] == pytest.approx(drive + 0.5 * (1.0 - drive))


@pytest.mark.parametrize("drive", [0.3, 0.4, 0.45, 0.5])
def test_keyframe_phases_are_increasing_and_inside_one_stroke(drive):
    phases = CAPLAN_GARDNER_2010.keyframe_phases(drive)
    assert np.all(np.diff(phases) > 0)
    assert phases[0] >= 0.0
    assert phases[-1] < 1.0


# --------------------------------------------------------------------------
# cross-source agreement
# --------------------------------------------------------------------------
def test_kleshnev_catch_knee_angle_is_close_to_caplans():
    """45.4 deg (elite, on water) vs 41.0 deg (university, ergometer)."""
    kleshnev = stroke_data.KLESHNEV_ELITE.knee[0]
    caplan = CAPLAN_GARDNER_2010.knee[0]
    assert abs(kleshnev - caplan) < 10.0


def test_the_olympic_dataset_reports_a_much_larger_layback():
    """Why it is recorded but not used as the default driver."""
    olympic = stroke_data.OLYMPIC_VS_TRADITIONAL_2025.trunk[2]
    caplan = CAPLAN_GARDNER_2010.trunk[2]
    assert olympic > caplan + 25.0


def test_raising_the_stretcher_rotates_the_rower_backwards():
    """Caplan's headline result, preserved across the three positions."""
    standard = CAPLAN_GARDNER_2010.trunk[0]
    raised = stroke_data.CAPLAN_GARDNER_2010_RAISED_2.trunk[0]
    assert raised > standard, "forward lean should reduce as stretchers rise"


def test_raising_the_stretcher_steepens_the_shin_at_the_catch():
    standard = CAPLAN_GARDNER_2010.shank[0]
    raised = stroke_data.CAPLAN_GARDNER_2010_RAISED_2.shank[0]
    assert raised < standard
