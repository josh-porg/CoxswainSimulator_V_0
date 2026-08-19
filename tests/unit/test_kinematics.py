"""Unit tests for the joint-driven rower kinematics.

Three things are pinned here:

* the chain reproduces the *measured* joint angles it is driven by;
* derived quantities (slide travel, seat height, segment speeds) match
  independently published values;
* the segment accelerations are continuous, which is what the legacy
  piecewise model was not.
"""

import numpy as np
import pytest

from coxswain.crew import stroke_data
from coxswain.crew.anthropometry import RowerAnthropometry
from coxswain.crew.kinematics import (
    DEFAULT_ARM_POSTURE,
    SEGMENT_ORDER,
    THIGH_MODES,
    JointDrivenRower,
    RowerStation,
)
from coxswain.crew.stroke import StrokeTiming

RATE = 32.0


@pytest.fixture
def timing():
    return StrokeTiming(rate=RATE)


@pytest.fixture
def anthro():
    return RowerAnthropometry(mass=85.0, stature=1.88)


@pytest.fixture
def rower(anthro, timing):
    return JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing)


@pytest.fixture
def keyframe_times(timing):
    phases = stroke_data.CAPLAN_GARDNER_2010.keyframe_phases(
        timing.drive_fraction)
    return phases * timing.period


# --------------------------------------------------------------------------
# fidelity to the driving data
# --------------------------------------------------------------------------
def test_shank_angle_reproduces_the_measured_values(rower, keyframe_times):
    dataset = stroke_data.CAPLAN_GARDNER_2010
    for index, t in enumerate(keyframe_times):
        actual = np.degrees(rower.joint_angles.shank(t).value)
        assert actual == pytest.approx(dataset.shank[index], abs=1.0)


def test_trunk_angle_reproduces_the_measured_values(rower, keyframe_times):
    dataset = stroke_data.CAPLAN_GARDNER_2010
    for index, t in enumerate(keyframe_times):
        actual = np.degrees(rower.joint_angles.trunk(t).value)
        assert actual == pytest.approx(dataset.trunk_link[index], abs=1.0)


def test_thigh_angle_matches_measurement_at_catch_and_finish(rower,
                                                             keyframe_times):
    """The level-seat constraint must agree with the data where the data
    are self-consistent -- at the two ends of the leg's range."""
    dataset = stroke_data.CAPLAN_GARDNER_2010
    for index in (0, 2):
        actual = np.degrees(rower.thigh_angle(keyframe_times[index]))
        assert actual == pytest.approx(dataset.thigh[index], abs=1.5)


def test_measured_thigh_mode_reproduces_all_four_keyframes(anthro, timing,
                                                           keyframe_times):
    rower = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing,
                             thigh_mode="measured")
    dataset = stroke_data.CAPLAN_GARDNER_2010
    for index, t in enumerate(keyframe_times):
        actual = np.degrees(rower.thigh_angle(t))
        assert actual == pytest.approx(dataset.thigh[index], abs=1.5)


# --------------------------------------------------------------------------
# derived quantities vs independent measurements
# --------------------------------------------------------------------------
def test_slide_travel_matches_published_values(rower):
    """On-water crews use 0.60-0.70 m of a ~0.75 m slide."""
    assert 0.58 <= rower.slide_travel() <= 0.72


def test_seat_height_is_calibrated_from_the_data(rower):
    """Stretcher-to-seat differential is 150 +- 50 mm (Sayer) / 172 +- 15 mm.

    Measured from the heel; the hip joint sits above the seat and the
    ankle joint above the heel, so the joint-to-joint value is smaller.
    """
    assert 0.08 <= rower.station.seat_height <= 0.20


def test_the_seat_stays_on_its_rail(rower):
    """In level-seat mode the hip height must not vary at all."""
    assert rower.seat_height_variation() < 1e-9


def test_measured_mode_lifts_the_hip_off_the_rail(anthro, timing):
    """Documents why level_seat is the default.

    Taking the measured knee angle literally implies the hip joint rises
    several centimetres mid-stroke, which a seat on a rail cannot do.
    """
    rower = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing,
                             thigh_mode="measured")
    assert 0.03 < rower.seat_height_variation() < 0.12


def test_peak_seat_velocity_is_reached_early_in_the_drive(rower, timing):
    """Rowers reach peak slide speed about a third of the way down.

    This is what makes the crew reaction on the hull reverse sign inside
    the drive, and hence why the boat runs fastest on the recovery.
    """
    times = np.linspace(0.0, timing.drive_duration, 200)
    chain = rower._chain(times)
    speed = np.abs(chain["hip"][0].first)
    peak_fraction = times[int(np.argmax(speed))] / timing.drive_duration
    assert 0.2 < peak_fraction < 0.5


def test_crew_centre_of_mass_travels_bow_ward_during_the_drive(rower, timing):
    catch = rower.centre_of_mass(0.0)[0]
    finish = rower.centre_of_mass(timing.drive_duration)[0]
    assert finish > catch


def test_centre_of_mass_excursion_is_physically_plausible(rower, timing):
    times = np.linspace(0.0, timing.period, 300, endpoint=False)
    x = np.array([rower.centre_of_mass(t)[0] for t in times])
    assert 0.6 <= (x.max() - x.min()) <= 1.0


def test_knee_stays_above_the_hip_at_the_catch(rower):
    joints = rower.joint_positions(0.0)
    assert joints["knee"][2] > joints["hip"][2]


def test_legs_are_nearly_straight_at_the_finish(rower, timing):
    """Knee interior angle at the finish is 171 +- 6.5 deg (Caplan Table II)."""
    t = timing.drive_duration
    shank = np.degrees(rower.joint_angles.shank(t).value)
    thigh = np.degrees(rower.thigh_angle(t))
    knee_interior = 180.0 - shank + thigh
    assert knee_interior == pytest.approx(171.2, abs=8.0)


def test_hands_stay_sternward_of_the_shoulders(rower, timing):
    """A rower's hands are in front of them, and they face the stern."""
    times = np.linspace(0.0, timing.period, 60, endpoint=False)
    chain = rower._chain(times)
    assert np.all(chain["hand"][0].value < chain["shoulder"][0].value)


def test_handle_travel_matches_a_sweep_oar_arc(rower):
    """Inboard ~1.15 m swinging -60 to +35 deg gives ~1.6 m of x travel."""
    assert 1.3 <= rower.handle_travel() <= 1.9


# --------------------------------------------------------------------------
# smoothness -- the legacy failure mode
# --------------------------------------------------------------------------
def test_segment_accelerations_are_continuous_across_the_catch(rower, timing):
    """The legacy piecewise model stepped 3.4 m/s^2 at the catch.

    Sampling either side of the catch must show a step no larger than the
    smooth variation between neighbouring samples elsewhere.
    """
    eps = 1e-6
    _, _, before = rower.segment_state(timing.period - eps)
    _, _, after = rower.segment_state(0.0 + eps)
    assert np.abs(after - before).max() < 1e-3


def test_segment_accelerations_are_continuous_across_the_finish(rower, timing):
    eps = 1e-6
    _, _, before = rower.segment_state(timing.drive_duration - eps)
    _, _, after = rower.segment_state(timing.drive_duration + eps)
    assert np.abs(after - before).max() < 1e-3


def test_segment_state_is_exactly_periodic(rower, timing):
    for name, index in (("position", 0), ("velocity", 1), ("acceleration", 2)):
        start = rower.segment_state(0.0)[index]
        after = rower.segment_state(timing.period)[index]
        np.testing.assert_allclose(start, after, atol=1e-9,
                                   err_msg=f"{name} is not periodic")


def test_velocities_match_finite_differenced_positions(rower, timing):
    step = 1e-6
    for t in (0.1, 0.35, 0.8, 1.4):
        forward = rower.segment_state(t + step)[0]
        backward = rower.segment_state(t - step)[0]
        numerical = (forward - backward) / (2 * step)
        analytic = rower.segment_state(t)[1]
        np.testing.assert_allclose(analytic, numerical, atol=1e-4)


def test_accelerations_match_finite_differenced_velocities(rower, timing):
    step = 1e-6
    for t in (0.1, 0.35, 0.8, 1.4):
        forward = rower.segment_state(t + step)[1]
        backward = rower.segment_state(t - step)[1]
        numerical = (forward - backward) / (2 * step)
        analytic = rower.segment_state(t)[2]
        np.testing.assert_allclose(analytic, numerical, atol=1e-4)


def test_segment_speeds_stay_within_human_limits(rower, timing):
    times = np.linspace(0.0, timing.period, 300, endpoint=False)
    peak_speed = peak_accel = 0.0
    for t in times:
        _, velocity, acceleration = rower.segment_state(t)
        peak_speed = max(peak_speed, np.abs(velocity).max())
        peak_accel = max(peak_accel, np.abs(acceleration).max())
    assert peak_speed < 5.0, "no body segment moves faster than 5 m/s"
    assert peak_accel < 60.0, "no body segment exceeds ~6 g"


# --------------------------------------------------------------------------
# shape, ordering and batching
# --------------------------------------------------------------------------
def test_segment_state_shapes(rower):
    for array in rower.segment_state(0.3):
        assert array.shape == (12, 3)


def test_segment_masses_align_with_segment_order(rower, anthro):
    expected = np.array([anthro.by_name(n).mass for n in SEGMENT_ORDER])
    np.testing.assert_allclose(rower.segment_masses, expected)


def test_lateral_offsets_are_mirrored(rower):
    position, _, _ = rower.segment_state(0.4)
    lookup = dict(zip(SEGMENT_ORDER, position))
    for base in ("upper_arm", "forearm_hand", "thigh", "shank_foot"):
        assert lookup[f"{base}_port"][1] == pytest.approx(
            -lookup[f"{base}_starboard"][1])


def test_centreline_segments_have_no_lateral_offset(rower):
    position, _, _ = rower.segment_state(0.4)
    lookup = dict(zip(SEGMENT_ORDER, position))
    for name in ("head", "upper_trunk", "mid_trunk", "lower_trunk"):
        assert lookup[name][1] == 0.0


def test_lateral_motion_is_not_modelled(rower):
    """The chain is planar: y velocity and acceleration are identically zero."""
    _, velocity, acceleration = rower.segment_state(0.4)
    np.testing.assert_allclose(velocity[:, 1], 0.0)
    np.testing.assert_allclose(acceleration[:, 1], 0.0)


def test_batched_evaluation_matches_per_seat_evaluation(anthro, timing):
    """The seat-sharing optimisation must be exact, not approximate."""
    offsets = np.array([0.0, -1.4, -2.8])
    leader = JointDrivenRower(anthro, RowerStation(x_ankle=3.0), timing)
    batched = leader.segment_state(0.37, x_offsets=offsets)

    for index, offset in enumerate(offsets):
        alone = JointDrivenRower(
            anthro, RowerStation(x_ankle=3.0 + offset), timing)
        expected = alone.segment_state(0.37)
        block = slice(index * 12, (index + 1) * 12)
        for got, want in zip(batched, expected):
            np.testing.assert_allclose(got[block], want, atol=1e-12)


def test_identical_rowers_share_a_kinematics_signature(anthro, timing):
    a = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing)
    b = JointDrivenRower(anthro, RowerStation(x_ankle=-2.5), timing)
    assert a.kinematics_signature() == b.kinematics_signature()


def test_phase_offset_breaks_the_signature(anthro, timing):
    a = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing)
    b = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing,
                         phase_offset=0.05)
    assert a.kinematics_signature() != b.kinematics_signature()


def test_different_athletes_break_the_signature(timing):
    a = JointDrivenRower(RowerAnthropometry(mass=85.0, stature=1.88),
                         RowerStation(x_ankle=0.0), timing)
    b = JointDrivenRower(RowerAnthropometry(mass=70.0, stature=1.88),
                         RowerStation(x_ankle=0.0), timing)
    assert a.kinematics_signature() != b.kinematics_signature()


def test_station_offset_shifts_only_the_longitudinal_position(anthro, timing):
    bow = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing)
    stern = JointDrivenRower(anthro, RowerStation(x_ankle=-3.0), timing)
    bow_pos, bow_vel, _ = bow.segment_state(0.3)
    stern_pos, stern_vel, _ = stern.segment_state(0.3)

    np.testing.assert_allclose(stern_pos[:, 0], bow_pos[:, 0] - 3.0, atol=1e-12)
    np.testing.assert_allclose(stern_pos[:, 1:], bow_pos[:, 1:], atol=1e-12)
    np.testing.assert_allclose(stern_vel, bow_vel, atol=1e-12)


# --------------------------------------------------------------------------
# phase offsets
# --------------------------------------------------------------------------
def test_phase_offset_delays_the_stroke(anthro, timing):
    synced = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing)
    offset = 0.1
    late = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing,
                            phase_offset=offset)
    shifted = late.segment_state(offset * timing.period)[0]
    np.testing.assert_allclose(shifted, synced.segment_state(0.0)[0],
                               atol=1e-6)


def test_zero_phase_offset_is_a_no_op(anthro, timing):
    a = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing)
    b = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing,
                         phase_offset=0.0)
    np.testing.assert_allclose(a.segment_state(0.6)[0],
                               b.segment_state(0.6)[0], atol=1e-15)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_unknown_thigh_mode_is_rejected(anthro, timing):
    with pytest.raises(ValueError, match="thigh_mode must be one of"):
        JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing,
                         thigh_mode="guess")


def test_thigh_modes_constant_is_accurate():
    assert THIGH_MODES == ("level_seat", "measured")


def test_unreachable_seat_height_is_rejected(anthro, timing):
    with pytest.raises(ValueError, match="level-seat constraint is unreachable"):
        JointDrivenRower(anthro, RowerStation(x_ankle=0.0, seat_height=0.95),
                         timing)


def test_impossible_arm_reach_is_rejected(anthro, timing):
    posture = tuple((f, 1.6, elev) for f, _, elev in DEFAULT_ARM_POSTURE)
    with pytest.raises(ValueError, match="demands a reach"):
        JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing,
                         arm_posture=posture)


def test_arm_keyframes_must_stay_within_one_stroke(anthro, timing):
    posture = ((0.0, 0.9, 184.0), (5.0, 0.5, 176.0))
    with pytest.raises(ValueError, match="within one stroke"):
        JointDrivenRower(anthro, RowerStation(x_ankle=0.0), timing,
                         arm_posture=posture)
