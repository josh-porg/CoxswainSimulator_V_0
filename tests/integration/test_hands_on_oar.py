"""The rowers must actually be holding their oars.

Cabrera, Ruina & Kleshnev (2006) eq. (8) put the constraint plainly:

    "Since the rower always has a grip on the oar handle, the fore-aft
    positions of the rower's hand and the oar handle relative to the foot
    stretcher are the same."

Before this was enforced, the model's rowers' hands sat up to 0.46 m from
their own handles -- they were rowing thin air, and the arm segment masses
were moving on an invented trajectory rather than the one the rig dictates.

These tests also pin the two pieces of real rigging and biomechanics that
had to be added before the constraint could be satisfied at all:

* a sweep rower's two hands grip **different points** along one handle;
* a sweep rower **rotates their trunk** to follow the handle across the
  body, while a sculler does not.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.anthropometry import PORT, STARBOARD
from coxswain.crew.kinematics import (
    MAX_SHOULDER_PROTRACTION,
    MAX_TRUNK_ROTATION,
)
from coxswain.crew.oarlock import handle_position

#: The hand track is a Fourier fit of the handle path, so it reproduces it
#: to truncation error rather than exactly.  40 mm on a 1.14 m inboard is
#: under 4% of the moment arm.
TOLERANCE = 0.040


def _hand_errors(boat, seat_index=0, n_samples=240):
    """Distance from each hand to the grip it is supposed to be holding."""
    seat = boat.rig.seats[seat_index]
    rower = boat.crew[seat_index].rower
    targets = boat._hand_targets(seat)
    times = np.linspace(0.0, boat.timing.period, n_samples, endpoint=False)

    errors = {}
    for side, target in targets.items():
        track = rower._hand_tracks[side]
        got = np.array([track.position(t) for t in times])
        want = np.array([target(t) for t in times])
        errors[side] = np.linalg.norm(got - want, axis=1)
    return errors


# --------------------------------------------------------------------------
# the constraint itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["8+", "4+", "1x"])
def test_every_hand_stays_on_its_grip(name):
    boat = catalog.build(name, rate=32.0)
    for seat_index in range(boat.n_seats):
        for side, error in _hand_errors(boat, seat_index).items():
            assert error.max() < TOLERANCE, (
                f"{name} seat {seat_index} {side}: hand leaves the handle by "
                f"{error.max() * 1000:.0f} mm"
            )


@pytest.mark.parametrize("name", ["8+", "1x"])
def test_mean_hand_error_is_a_few_millimetres(name):
    boat = catalog.build(name, rate=32.0)
    for side, error in _hand_errors(boat).items():
        assert error.mean() < 0.010


def test_the_constraint_holds_across_stroke_rates():
    for rate in (20.0, 26.0, 32.0, 38.0):
        boat = catalog.eight(rate=rate)
        for side, error in _hand_errors(boat).items():
            assert error.max() < TOLERANCE, f"rate {rate}"


def test_hand_velocity_follows_the_handle():
    """Position agreement is not enough; the derivative must track too.

    The arm segment masses are accelerated along this track, so a hand
    that is in the right place but moving wrongly still injects the wrong
    force into the hull.
    """
    boat = catalog.eight(rate=32.0)
    seat = boat.rig.seats[0]
    target = boat._hand_targets(seat)[STARBOARD]
    track = boat.crew[0].rower._hand_tracks[STARBOARD]

    step = 1e-4
    for t in np.linspace(0.05, boat.timing.period - 0.05, 40):
        numerical = (np.asarray(target(t + step))
                     - np.asarray(target(t - step))) / (2 * step)
        analytic = np.array([component(t).first
                             for component in track.components])
        # the handle path has a corner at the catch and finish, so compare
        # in the mean rather than pointwise
        assert np.linalg.norm(analytic - numerical) < 1.5


# --------------------------------------------------------------------------
# sweep versus sculling
# --------------------------------------------------------------------------
def test_a_sweep_rower_holds_one_oar_with_two_hands():
    boat = catalog.eight(rate=32.0)
    seat = boat.rig.seats[0]
    assert not seat.is_sculling
    targets = boat._hand_targets(seat)
    assert set(targets) == {PORT, STARBOARD}

    lock = seat.oarlocks[0]
    t = 0.3 * boat.timing.period
    inside = np.asarray(targets[lock.side](t))
    outside = np.asarray(targets[-lock.side](t))

    # both on the same shaft, separated by the grip spacing
    separation = np.linalg.norm(inside - outside)
    assert separation == pytest.approx(lock.oar.grip_separation, abs=1e-9)


def test_the_outside_hand_takes_the_end_of_the_handle():
    """The outside hand is further from the oarlock than the inside hand."""
    boat = catalog.eight(rate=32.0)
    seat = boat.rig.seats[0]
    lock = seat.oarlocks[0]
    targets = boat._hand_targets(seat)

    t = 0.3 * boat.timing.period
    inside = np.linalg.norm(np.asarray(targets[lock.side](t)) - lock.position)
    outside = np.linalg.norm(np.asarray(targets[-lock.side](t))
                             - lock.position)
    assert outside > inside
    assert outside == pytest.approx(lock.oar.inboard, abs=1e-9)


def test_a_sculler_takes_one_handle_per_hand():
    boat = catalog.single_scull(rate=30.0)
    seat = boat.rig.seats[0]
    assert seat.is_sculling
    targets = boat._hand_targets(seat)

    t = 0.3 * boat.timing.period
    for lock in seat.oarlocks:
        reach = np.linalg.norm(np.asarray(targets[lock.side](t))
                               - lock.position)
        assert reach == pytest.approx(lock.oar.inboard, abs=1e-9)


def test_sweep_oars_have_a_grip_separation_and_sculling_oars_do_not():
    from coxswain.boats.rig import SCULLING_OAR, SWEEP_OAR

    assert SWEEP_OAR.grip_separation > 0.2
    assert SCULLING_OAR.grip_separation == 0.0


def test_a_grip_separation_wider_than_the_inboard_is_rejected():
    from coxswain.boats.rig import Oar

    with pytest.raises(ValueError, match="grip_separation"):
        Oar(length=3.7, inboard=1.14, grip_separation=1.5)


# --------------------------------------------------------------------------
# trunk rotation
# --------------------------------------------------------------------------
def test_a_sweep_rower_rotates_their_trunk():
    """Following the handle across the body requires turning the shoulders.

    Without this degree of freedom the outside shoulder finishes 0.82 m
    from a 0.70 m arm.
    """
    boat = catalog.eight(rate=32.0)
    rotation = np.degrees(boat.crew[0].rower.trunk_rotation)
    assert np.ptp(rotation) > 5.0, "sweep rowers must turn their shoulders"


def test_a_sculler_rows_square():
    """Sculling is symmetric, so no rotation should be needed."""
    boat = catalog.single_scull(rate=30.0)
    rotation = boat.crew[0].rower.trunk_rotation
    np.testing.assert_allclose(rotation, 0.0, atol=1e-9)


def test_trunk_rotation_stays_anatomically_plausible():
    boat = catalog.eight(rate=32.0)
    rotation = np.abs(boat.crew[0].rower.trunk_rotation)
    assert rotation.max() <= MAX_TRUNK_ROTATION + 1e-12


def test_port_and_starboard_rigged_seats_rotate_oppositely():
    """Adjacent seats in an alternating rig turn towards their own rigger."""
    boat = catalog.eight(rate=32.0)
    stroke_side = boat.rig.seats[0].oarlocks[0].side
    second_side = boat.rig.seats[1].oarlocks[0].side
    assert stroke_side == -second_side

    first = boat.crew[0].rower.trunk_rotation
    second = boat.crew[1].rower.trunk_rotation
    assert np.sign(first.mean()) == -np.sign(second.mean())


# --------------------------------------------------------------------------
# reach
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["8+", "4+", "1x"])
def test_no_arm_is_asked_to_stretch_past_its_length(name):
    boat = catalog.build(name, rate=32.0)
    for member in boat.crew:
        for side, margin in member.rower._arm_reach_margin.items():
            assert margin <= 1.0 + 1e-9, (
                f"{name}: {side} arm at {margin:.3f} of full extension"
            )


def test_an_impossible_rig_is_rejected_with_a_useful_message():
    """Tripling the oar inboard puts the handle out of any rower's reach.

    The check has to fire loudly: a rower who cannot hold their oar is a
    modelling error, not a configuration to silently approximate.
    """
    from coxswain.boats.boat import Boat
    from coxswain.boats.rig import Oar, build_sweep_rig
    from coxswain.crew.stroke import StrokeTiming
    from coxswain.hydro.hull import parametric_offsets

    reference = catalog.eight(rate=32.0)
    absurd = Oar(length=8.0, inboard=3.5, grip_separation=0.3)
    rig = build_sweep_rig(n_seats=8, spacing=1.22, stern_station=-4.30,
                          span=0.85, oarlock_height=0.38, oar=absurd,
                          coxswain_position=[-5.6, 0.0, 0.15],
                          coxswain_mass=55.0)

    with pytest.raises(ValueError, match="cannot reach the oar handle"):
        Boat(name="unreachable", offsets=reference.offsets, rig=rig,
             hull_mass=reference.hull_mass,
             hull_inertia=reference.hull_inertia,
             timing=StrokeTiming(32.0))


def test_shoulder_protraction_is_bounded():
    assert 0.02 <= MAX_SHOULDER_PROTRACTION <= 0.08


# --------------------------------------------------------------------------
# the arms still carry mass sensibly
# --------------------------------------------------------------------------
def test_arm_segments_move_laterally_in_a_sweep_boat():
    """The planar chain cannot do this; the constrained arms must.

    A sweep rower's hands cross the centreline, so the arm masses have
    genuine lateral motion and contribute a real roll and yaw couple.
    """
    boat = catalog.eight(rate=32.0)
    from coxswain.crew.kinematics import SEGMENT_ORDER

    index = SEGMENT_ORDER.index("forearm_hand_starboard")
    times = np.linspace(0.0, boat.timing.period, 60, endpoint=False)
    lateral = np.array([boat.crew[0].rower.segment_state(t)[0][index, 1]
                        for t in times])
    assert np.ptp(lateral) > 0.10, "sweep arms must sweep across the boat"


def test_crew_segment_masses_are_unchanged_by_the_constraint():
    """Constraining the arms must not create or destroy mass."""
    boat = catalog.eight(rate=32.0)
    for member in boat.crew:
        expected = member.rower.anthropometry.mass
        assert member.rower.total_mass == pytest.approx(expected, rel=1e-9)


def test_segment_accelerations_stay_finite_and_bounded():
    boat = catalog.eight(rate=32.0)
    times = np.linspace(0.0, boat.timing.period, 200, endpoint=False)
    peak = 0.0
    for t in times:
        _, _, acceleration = boat.crew[0].rower.segment_state(t)
        assert np.isfinite(acceleration).all()
        peak = max(peak, np.abs(acceleration).max())
    assert peak < 80.0, f"peak segment acceleration {peak:.1f} m/s^2"
