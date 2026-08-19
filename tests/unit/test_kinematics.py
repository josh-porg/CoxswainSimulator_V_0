"""Unit tests for the joint-driven rower linkage.

Covers three things:
  * the kinematic chain is geometrically correct and hits realistic
    rowing positions (calibration);
  * the level-seat constraint holds exactly;
  * segment velocities and accelerations are exact derivatives of the
    positions, and are continuous everywhere -- the defect that put a
    1.9 kN step force on the hull at every catch.
"""

import numpy as np
import pytest

from coxswain.crew.anthropometry import RowerAnthropometry
from coxswain.crew.kinematics import (
    DEFAULT_JOINT_ANGLES,
    SEGMENT_ORDER,
    JointAngles,
    JointDrivenRower,
    RowerStation,
)
from coxswain.crew.stroke import StrokeTiming


@pytest.fixture
def timing():
    return StrokeTiming(rate=32.0)


@pytest.fixture
def rower(timing):
    return JointDrivenRower(
        RowerAnthropometry(mass=85.0, stature=1.88),
        RowerStation(x_ankle=0.0),
        JointAngles.from_catch_finish(timing),
    )


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------
def test_segment_order_has_twelve_entries():
    assert len(SEGMENT_ORDER) == 12
    assert len(set(SEGMENT_ORDER)) == 12


def test_segment_state_shapes(rower):
    position, velocity, acceleration = rower.segment_state(0.3)
    for array in (position, velocity, acceleration):
        assert array.shape == (12, 3)


def test_segment_masses_match_the_anthropometry(rower):
    assert rower.total_mass == pytest.approx(85.0, rel=2e-4)
    assert rower.segment_masses.shape == (12,)


def test_thigh_is_not_an_independent_driver():
    """It follows from the level-track constraint, so must not be listed."""
    assert "thigh" not in DEFAULT_JOINT_ANGLES
    assert set(DEFAULT_JOINT_ANGLES) == {"shank", "trunk", "upper_arm",
                                         "forearm"}


def test_link_lengths_come_from_the_anthropometry(rower):
    anthro = rower.anthropometry
    assert rower.shank_length == pytest.approx(anthro.length("shank"))
    assert rower.thigh_length == pytest.approx(anthro.length("thigh"))
    assert rower.trunk_length == pytest.approx(
        anthro.length("lower_trunk") + anthro.length("mid_trunk")
        + anthro.length("upper_trunk"))


# --------------------------------------------------------------------------
# the level-seat constraint
# --------------------------------------------------------------------------
def test_seat_height_is_constant_through_the_stroke(rower, timing):
    """The seat runs on a level track: hip z must not vary at all."""
    times = np.linspace(0.0, timing.period, 200, endpoint=False)
    hip_z = np.array([rower.joint_positions(t)["hip"][2] for t in times])
    assert np.ptp(hip_z) < 1e-9


def test_seat_sits_at_the_configured_height(rower):
    hip = rower.joint_positions(0.0)["hip"]
    station = rower.station
    assert hip[2] == pytest.approx(station.z_ankle + station.seat_height,
                                   abs=1e-12)


def test_unreachable_leg_geometry_is_rejected(timing):
    """A seat the thigh cannot reach at full leg extension must fail loudly."""
    with pytest.raises(ValueError, match="level-seat constraint"):
        JointDrivenRower(
            RowerAnthropometry(mass=85.0, stature=1.88),
            RowerStation(x_ankle=0.0, seat_height=0.95),
            JointAngles.from_catch_finish(timing),
        )


def test_reachable_but_marginal_geometry_is_accepted(timing):
    """The guard must not be so tight that plausible rigs are refused."""
    JointDrivenRower(
        RowerAnthropometry(mass=85.0, stature=1.88),
        RowerStation(x_ankle=0.0, seat_height=0.22),
        JointAngles.from_catch_finish(timing),
    )


def test_thigh_angle_stays_in_the_expected_band(rower, timing):
    times = np.linspace(0.0, timing.period, 200, endpoint=False)
    thigh = np.degrees(rower.thigh_angle(times))
    assert -60.0 < thigh.min() < -35.0
    assert -5.0 < thigh.max() < 10.0


# --------------------------------------------------------------------------
# calibration against real rowing geometry
# --------------------------------------------------------------------------
def test_slide_travel_is_realistic(rower, timing):
    """Rowers use 0.60-0.70 m of a ~0.75 m slide."""
    assert 0.60 <= rower.slide_travel(timing) <= 0.70


def test_trunk_sweep_is_realistic(rower, timing):
    """Total trunk rotation catch-to-finish is 50-70 deg."""
    times = np.linspace(0.0, timing.period, 400, endpoint=False)
    trunk = np.degrees(rower.joint_angles.trunk(times).value)
    assert 50.0 <= np.ptp(trunk) <= 70.0


def test_body_is_compressed_at_the_catch_and_extended_at_the_finish(rower,
                                                                    timing):
    catch = rower.joint_positions(0.0)
    finish = rower.joint_positions(timing.drive_duration)
    # the seat moves towards the bow during the drive
    assert finish["hip"][0] > catch["hip"][0]
    # the knees are up at the catch and down at the finish
    assert catch["knee"][2] > finish["knee"][2] + 0.25


def test_knees_are_above_the_seat_at_the_catch(rower):
    catch = rower.joint_positions(0.0)
    assert catch["knee"][2] > catch["hip"][2] + 0.20


def test_hands_move_towards_the_bow_during_the_drive(rower, timing):
    """The rower faces the stern, so drawing to the body is bow-ward."""
    catch = rower.joint_positions(0.0)["hand"][0]
    finish = rower.joint_positions(timing.drive_duration)["hand"][0]
    assert finish > catch


def test_shoulders_are_stern_of_the_hips_at_the_catch(rower):
    """Body angle: the rower is folded forward over the knees."""
    catch = rower.joint_positions(0.0)
    assert catch["shoulder"][0] < catch["hip"][0]


def test_shoulders_are_bow_of_the_hips_at_the_finish(rower, timing):
    """Layback."""
    finish = rower.joint_positions(timing.drive_duration)
    assert finish["shoulder"][0] > finish["hip"][0]


def test_crew_centre_of_mass_moves_bow_ward_during_the_drive(rower, timing):
    """This reaction is what makes the hull run fastest on the recovery."""
    catch = rower.centre_of_mass(0.0)
    finish = rower.centre_of_mass(timing.drive_duration)
    assert finish[0] > catch[0]


def test_ankle_is_fixed_in_the_hull_frame(rower, timing):
    times = np.linspace(0.0, timing.period, 50, endpoint=False)
    ankles = np.array([rower.joint_positions(t)["ankle"] for t in times])
    assert np.ptp(ankles, axis=0).max() < 1e-12


# --------------------------------------------------------------------------
# derivative exactness
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t", [0.05, 0.31, 0.72, 1.28, 1.61])
def test_segment_velocity_is_the_derivative_of_position(rower, t):
    h = 1e-6
    plus, _, _ = rower.segment_state(t + h)
    minus, _, _ = rower.segment_state(t - h)
    _, velocity, _ = rower.segment_state(t)
    np.testing.assert_allclose(velocity, (plus - minus) / (2 * h), atol=1e-5)


@pytest.mark.parametrize("t", [0.05, 0.31, 0.72, 1.28, 1.61])
def test_segment_acceleration_is_the_derivative_of_velocity(rower, t):
    h = 1e-5
    _, plus, _ = rower.segment_state(t + h)
    _, minus, _ = rower.segment_state(t - h)
    _, _, acceleration = rower.segment_state(t)
    np.testing.assert_allclose(acceleration, (plus - minus) / (2 * h),
                               atol=1e-4)


# --------------------------------------------------------------------------
# smoothness -- the legacy catch discontinuity
# --------------------------------------------------------------------------
def test_segment_acceleration_is_continuous_across_the_catch(rower, timing):
    """Legacy behaviour jumped 3.41 m/s^2 here; a Fourier chain cannot."""
    eps = 1e-6
    _, _, before = rower.segment_state(timing.period - eps)
    _, _, after = rower.segment_state(eps)
    assert np.abs(after - before).max() < 1e-3


def test_segment_acceleration_is_continuous_across_the_finish(rower, timing):
    eps = 1e-6
    _, _, before = rower.segment_state(timing.drive_duration - eps)
    _, _, after = rower.segment_state(timing.drive_duration + eps)
    assert np.abs(after - before).max() < 1e-3


def test_segment_acceleration_has_no_steps_anywhere_in_the_cycle(rower,
                                                                 timing):
    times = np.linspace(0.0, 2 * timing.period, 2001)
    acceleration = np.array([rower.segment_state(t)[2] for t in times])
    steps = np.abs(np.diff(acceleration, axis=0)).max()
    assert steps < 0.5


def test_segment_accelerations_stay_physically_bounded(rower, timing):
    times = np.linspace(0.0, timing.period, 400, endpoint=False)
    peak = max(np.abs(rower.segment_state(t)[2]).max() for t in times)
    assert peak < 30.0, f"segment acceleration {peak:.1f} m/s^2 is implausible"


# --------------------------------------------------------------------------
# periodicity
# --------------------------------------------------------------------------
def test_kinematics_are_exactly_periodic(rower, timing):
    for t in (0.0, 0.4, 1.1):
        a = rower.segment_state(t)
        b = rower.segment_state(t + timing.period)
        for first, second in zip(a, b):
            np.testing.assert_allclose(first, second, atol=1e-12)


def test_crew_returns_to_its_starting_position_each_stroke(rower, timing):
    start = rower.centre_of_mass(0.0)
    later = rower.centre_of_mass(5 * timing.period)
    np.testing.assert_allclose(start, later, atol=1e-12)


# --------------------------------------------------------------------------
# lateral layout
# --------------------------------------------------------------------------
def test_centreline_segments_have_no_lateral_offset(rower):
    position, _, _ = rower.segment_state(0.3)
    for index, name in enumerate(SEGMENT_ORDER):
        if name in ("head", "upper_trunk", "mid_trunk", "lower_trunk"):
            assert position[index, 1] == pytest.approx(0.0)


def test_paired_segments_are_mirrored_about_the_centreline(rower):
    position, _, _ = rower.segment_state(0.3)
    index = {name: i for i, name in enumerate(SEGMENT_ORDER)}
    for stem in ("upper_arm", "forearm_hand", "thigh", "shank_foot"):
        port = position[index[f"{stem}_port"]]
        starboard = position[index[f"{stem}_starboard"]]
        assert port[1] == pytest.approx(-starboard[1])
        assert port[0] == pytest.approx(starboard[0])
        assert port[2] == pytest.approx(starboard[2])


def test_lateral_offsets_do_not_move(rower, timing):
    """The chain is planar, so y velocity and acceleration are exactly zero."""
    for t in np.linspace(0.0, timing.period, 20, endpoint=False):
        _, velocity, acceleration = rower.segment_state(t)
        np.testing.assert_allclose(velocity[:, 1], 0.0, atol=1e-15)
        np.testing.assert_allclose(acceleration[:, 1], 0.0, atol=1e-15)


# --------------------------------------------------------------------------
# station placement and phase offset
# --------------------------------------------------------------------------
def test_station_translates_the_whole_chain(timing):
    anthro = RowerAnthropometry(mass=85.0, stature=1.88)
    angles = JointAngles.from_catch_finish(timing)
    bow = JointDrivenRower(anthro, RowerStation(x_ankle=0.0), angles)
    stern = JointDrivenRower(anthro, RowerStation(x_ankle=-3.0), angles)

    bow_position, _, _ = bow.segment_state(0.4)
    stern_position, _, _ = stern.segment_state(0.4)
    np.testing.assert_allclose(stern_position[:, 0], bow_position[:, 0] - 3.0,
                               atol=1e-12)
    np.testing.assert_allclose(stern_position[:, 2], bow_position[:, 2],
                               atol=1e-12)


def test_phase_offset_shifts_the_stroke_in_time(timing):
    anthro = RowerAnthropometry(mass=85.0, stature=1.88)
    station = RowerStation(x_ankle=0.0)
    synced = JointDrivenRower(anthro, station,
                              JointAngles.from_catch_finish(timing))
    offset = 0.1
    late = JointDrivenRower(
        anthro, station,
        JointAngles.from_catch_finish(timing, phase_offset=offset))

    t = 0.55
    reference, _, _ = synced.segment_state(t - offset * timing.period)
    shifted, _, _ = late.segment_state(t)
    np.testing.assert_allclose(shifted, reference, atol=1e-10)


def test_zero_phase_offset_is_a_no_op(timing):
    a = JointAngles.from_catch_finish(timing)
    b = JointAngles.from_catch_finish(timing, phase_offset=0.0)
    np.testing.assert_allclose(a.trunk.cos_coefficients,
                               b.trunk.cos_coefficients)


def test_joint_positions_returns_every_joint(rower):
    joints = rower.joint_positions(0.2)
    assert set(joints) == {"ankle", "knee", "hip", "shoulder", "elbow", "hand"}
    for value in joints.values():
        assert value.shape == (3,)
