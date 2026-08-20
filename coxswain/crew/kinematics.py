"""Rower body-segment kinematics from a joint-driven serial linkage.

Formaggia et al. obtained ``x_ij``, ``x_dot_ij`` and ``x_ddot_ij`` -- the
hull-frame position, velocity and acceleration of each of the 12 body
segments -- by reconstructing motion-capture trajectories with fitted
analytic functions.  This module does the same thing with *published*
motion-capture angles (:mod:`coxswain.crew.stroke_data`) in place of the
unavailable raw trajectories: the rower is treated as an actuated serial
kinematic chain whose **joint angles** are prescribed from measurement,
and every segment's motion follows by forward kinematics.

That is also the approach taken by Serveto et al., *A three-dimensional
model of the boat-oars-rower system using ADAMS and LifeMOD*, Proc. IMechE
Part P 224(1) (2010) 75-83, which drives an articulated multibody rower
from joint coordinates.

Why forward kinematics rather than inverse
------------------------------------------
Driving the *seat position* and solving inverse kinematics for the knee
introduces a ``sqrt`` that goes singular at full leg extension -- exactly
the configuration a rower passes through at every finish, where segment
accelerations would blow up.  Driving joint angles is unconditionally well
posed: pure composition of sines and cosines, no branch anywhere in the
workspace.  The one inverse-kinematics solve that does occur, for the arm
posture, is done **once at construction** on four keyframes, never inside
a derivative evaluation.

Exact derivatives
-----------------
The whole chain is evaluated in :class:`~coxswain.core.taylor.Jet2`
arithmetic, so velocities and accelerations are differentiated
automatically rather than by hand.  Combined with the smooth
:class:`~coxswain.crew.stroke.FourierProfile` drivers, segment
accelerations are continuous everywhere -- no impulsive force at the
catch.

Frame
-----
Positions are in the **hull frame**, measured from the hull centre of mass
``G_h``.  The chain lies in the ``x``-``z`` plane; port/starboard limbs are
displaced in ``y`` by a fixed half-span.  Link angles are measured from the
``+x`` (bow) axis, positive towards ``+z`` (up): ``0`` points at the bow,
``90 deg`` straight up, ``180 deg`` at the stern.  The rower faces the
stern, so the drive sweeps the hands from about ``180 deg`` towards the
bow, and the seat travels *towards the bow* as the legs extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from ..core.taylor import Jet2, constant
from .anthropometry import CENTRELINE, PORT, STARBOARD, RowerAnthropometry
from .stroke import FourierProfile, FourierTrack, StrokeTiming
from .stroke_data import StrokeKinematicsDataset, default_dataset

__all__ = [
    "RowerStation",
    "JointAngles",
    "JointDrivenRower",
    "SEGMENT_ORDER",
    "THIGH_MODES",
    "DEFAULT_ARM_POSTURE",
    "KEYFRAME_HARMONICS",
    "ARM_TRACK_HARMONICS",
    "MAX_SHOULDER_PROTRACTION",
    "MAX_TRUNK_ROTATION",
]

#: Order in which segment quantities are returned.  Matches
#: :attr:`RowerAnthropometry.segments`.
SEGMENT_ORDER = (
    "head", "upper_trunk", "mid_trunk", "lower_trunk",
    "upper_arm_port", "upper_arm_starboard",
    "forearm_hand_port", "forearm_hand_starboard",
    "thigh_port", "thigh_starboard",
    "shank_foot_port", "shank_foot_starboard",
)

#: Harmonics retained when smoothing a four-keyframe dataset.  Deliberately
#: low: with four measured instants per stroke, more harmonics would fit
#: spline artefacts rather than rower motion, and segment accelerations
#: feed straight into the hull forces.
KEYFRAME_HARMONICS = 3

#: Harmonics kept when fitting the hand and elbow position tracks.  Higher
#: than the joint-angle drivers because the handle path is a hard
#: constraint rather than an interpolation of four keyframes, and losing
#: its shape would let the hands drift off the oar.
ARM_TRACK_HARMONICS = 10

#: Shoulder-to-handle distance, as a fraction of arm length, at which the
#: elbow solve stops trying to bend the arm.
#:
#: An extension of 1.0 is simply a straight arm, which is what a rower
#: actually has through the middle of the drive -- it is not an error.  The
#: clamp exists only so the elbow has a defined position rather than
#: sitting exactly on the straight-arm singularity; the *reachability*
#: check is separate, and compares against the arm length itself.
MAX_ARM_EXTENSION = 0.9995

#: Largest trunk rotation about the vertical, in radians.
#:
#: A sweep rower turns their shoulders towards the rigger so the hands can
#: follow the handle as it sweeps across the body; a sculler barely rotates
#: at all.  Reported ranges for sweep trunk rotation are 25-40 deg, so 40
#: is taken as the anatomical limit.  Purely planar shoulders leave a sweep
#: rower's outside shoulder 0.82 m from a 0.70 m arm, i.e. unable to hold
#: their own oar.
MAX_TRUNK_ROTATION = np.radians(40.0)

#: Arm extension, as a fraction of arm length, that the trunk rotation
#: solve aims to stay under.  Below the hard :data:`MAX_ARM_EXTENSION` so
#: the elbow solve keeps a usable margin from the straight-arm singularity.
COMFORTABLE_ARM_EXTENSION = 0.95

#: How far the shoulder joint can travel forward of its skeletal position,
#: in metres, by protracting the scapula.
#:
#: Mid-drive a rower's arms are genuinely straight, so the required reach
#: sits within a percent of full arm length; at the catch they reach
#: further still by rolling the shoulder girdle forward.  Scapular
#: protraction gives 4-6 cm of that reach and is what lets a sculler hold
#: both handles through the middle of the drive.  Without it the model
#: declares a perfectly ordinary rig unreachable.
MAX_SHOULDER_PROTRACTION = 0.06

#: Height of the shoulder joint centre below the cervicale, as a fraction
#: of the hip-to-cervicale trunk length.
#:
#: de Leva's trunk segment ends at the cervicale; the shoulder joint centre
#: is the landmark his *upper arm* segment starts from, and it lies below
#: and lateral to it.  For a 1.88 m athlete the drop is 5-6 cm on a 0.65 m
#: trunk.  It matters because the trunk rotates about the hip, so the
#: shoulder's fore-aft excursion scales with this length -- and the trunk
#: carries 43% of body mass.
SHOULDER_DROP_FRACTION = 0.085

#: How the thigh link angle is obtained.
#:
#: ``"level_seat"``
#:     Derive it from the shank angle by holding the hip height above the
#:     ankle constant.  The seat physically cannot leave its rail, so this
#:     is exact for the seat itself.  Default.
#: ``"measured"``
#:     Take it from the dataset's measured knee angle.  Faithful to the
#:     source, but the implied hip-joint height then varies by about 7 cm
#:     over the stroke -- more than the pelvis really moves, and most
#:     likely an artefact of averaging angles across subjects at fixed time
#:     points.  Selectable because it is what the data literally say.
THIGH_MODES = ("level_seat", "measured")

#: Hand position relative to the shoulder joint, as
#: ``(drive_fraction, extension, elevation_deg)`` at each arm keyframe.
#:
#: ``drive_fraction`` locates the keyframe: values in ``[0, 1]`` are that
#: fraction of the way through the drive, and values above 1 continue into
#: the recovery on the same scale (so ``1.0`` is the finish).
#: ``extension`` is the shoulder-to-hand distance as a **fraction of total
#: arm length**, so it is stature-independent and can never demand a reach
#: the arm does not have.  ``elevation_deg`` is the direction of the hand
#: from the shoulder, from the bow axis towards ``+z``; values near
#: 180 deg point at the stern, which is where a rower's hands are.
#:
#: Caplan & Gardner measured only the legs and trunk, so the arms are the
#: one part of the chain not driven directly by published angles.  These
#: postures encode the standard sequencing of the stroke -- legs, then
#: back, then arms -- so the arms stay nearly straight through the first
#: half of the drive and the draw is concentrated in the last third.  A
#: uniform draw across the whole drive, which is what four evenly spaced
#: keyframes would give, understates the peak hand speed by about half.
#: The postures are converted to link angles by a one-off inverse
#: kinematics solve at construction time, never inside a derivative
#: evaluation.
DEFAULT_ARM_POSTURE: Sequence[Tuple[float, float, float]] = (
    (0.00, 0.97, 184.0),   # catch: arms straight, reaching for the stern
    (0.50, 0.96, 182.0),   # mid-drive: still straight, legs doing the work
    (0.80, 0.78, 179.0),   # late drive: arms break, draw begins
    (1.00, 0.46, 176.0),   # finish: handle in to the lower ribs
    (1.55, 0.86, 182.0),   # mid-recovery: hands away again
)


@dataclass(frozen=True)
class RowerStation:
    """Where one rower is anchored in the hull frame.

    ``x_ankle`` places the footboard longitudinally; the whole chain grows
    from there.  Half-spans are lateral (``y``) offsets applied to the
    paired limbs.  ``seat_height`` is the height of the hip joint above the
    ankle joint; when ``None`` it is calibrated from the driving dataset's
    catch and finish keyframes, where the measured shank and knee angles
    independently agree on it to about a millimetre.
    """

    x_ankle: float
    z_ankle: float = -0.05
    seat_height: float = None
    foot_half_span: float = 0.16
    hip_half_span: float = 0.10
    shoulder_half_span: float = 0.20


@dataclass
class JointAngles:
    """Fourier-smoothed joint-angle drivers for one rower.

    ``shank`` and ``trunk`` come from measured data.  ``thigh`` is present
    only in ``"measured"`` mode; otherwise it is derived at evaluation time
    from the level-seat constraint.  The two arm profiles come from the
    keyframe inverse-kinematics solve.
    """

    shank: FourierProfile
    trunk: FourierProfile
    upper_arm: FourierProfile = None
    forearm: FourierProfile = None
    thigh: FourierProfile = None

    @classmethod
    def from_dataset(cls, dataset: StrokeKinematicsDataset,
                     timing: StrokeTiming,
                     n_harmonics: int = KEYFRAME_HARMONICS,
                     phase_offset: float = 0.0,
                     include_thigh: bool = False) -> "JointAngles":
        """Build the leg and trunk drivers from a measured dataset.

        ``phase_offset`` shifts this rower's stroke in normalised phase.  It
        is zero for a perfectly synchronised crew; a small non-zero value
        models imperfect timing.
        """
        phases = dataset.keyframe_phases(timing.drive_fraction)

        def build(values_deg):
            profile = FourierProfile.from_keyframes(
                phases, np.radians(np.asarray(values_deg, dtype=float)),
                timing, n_harmonics,
            )
            return _shift_phase(profile, phase_offset)

        built = {"shank": build(dataset.shank),
                 "trunk": build(dataset.trunk_link)}
        if include_thigh:
            built["thigh"] = build(dataset.thigh)
        return cls(**built)


def _shift_phase(profile: FourierProfile, phase_offset: float) -> FourierProfile:
    """Rotate a Fourier profile by a fraction of a period."""
    if phase_offset == 0.0:
        return profile
    cos_c = np.array(profile.cos_coefficients, dtype=float)
    sin_c = np.array(profile.sin_coefficients, dtype=float)
    shifted_cos = np.zeros_like(cos_c)
    shifted_sin = np.zeros_like(sin_c)
    shifted_cos[0] = cos_c[0]
    for k in range(1, len(cos_c)):
        delta = 2.0 * np.pi * k * phase_offset
        c, s = np.cos(delta), np.sin(delta)
        shifted_cos[k] = cos_c[k] * c - sin_c[k] * s
        shifted_sin[k] = sin_c[k] * c + cos_c[k] * s
    return FourierProfile(shifted_cos, shifted_sin, profile.period)


class JointDrivenRower:
    """One rower, as an actuated planar linkage anchored at the footboard.

    The chain runs::

        ankle (fixed) -> knee -> hip/seat -> shoulder -> elbow -> hand

    with the trunk a single rigid link from hip to shoulder, along which
    the three de Leva trunk masses are distributed.
    """

    def __init__(self, anthropometry: RowerAnthropometry,
                 station: RowerStation, timing: StrokeTiming,
                 dataset: StrokeKinematicsDataset = None,
                 thigh_mode: str = "level_seat",
                 arm_posture=None,
                 hand_targets: Dict[int, object] = None,
                 phase_offset: float = 0.0,
                 n_harmonics: int = KEYFRAME_HARMONICS):
        if thigh_mode not in THIGH_MODES:
            raise ValueError(
                f"thigh_mode must be one of {THIGH_MODES}, got {thigh_mode!r}"
            )

        self.anthropometry = anthropometry
        self.timing = timing
        self.dataset = default_dataset() if dataset is None else dataset
        self.thigh_mode = thigh_mode
        self.phase_offset = float(phase_offset)
        self.n_harmonics = int(n_harmonics)

        self._segments = {s.name: s for s in anthropometry.segments}
        self._masses = np.array(
            [self._segments[name].mass for name in SEGMENT_ORDER]
        )

        # Link lengths straight from the anthropometry table.
        self.shank_length = anthropometry.length("shank")
        self.thigh_length = anthropometry.length("thigh")
        # de Leva's trunk runs hip (MIDH) to *cervicale*, but the link the
        # arms hang from ends at the shoulder joint centre, which sits
        # below and lateral to the cervicale.  Using the full trunk length
        # as the hip-to-shoulder arm overstates the shoulder's excursion by
        # about 8%, and the trunk is 43% of body mass, so it feeds straight
        # into the crew centre-of-mass travel that sets the hull's speed
        # fluctuation.
        self.trunk_stack_length = (anthropometry.length("lower_trunk")
                                   + anthropometry.length("mid_trunk")
                                   + anthropometry.length("upper_trunk"))
        self.trunk_length = (self.trunk_stack_length
                             * (1.0 - SHOULDER_DROP_FRACTION))
        self.upper_arm_length = anthropometry.length("upper_arm")
        self.forearm_length = (anthropometry.length("forearm")
                               + anthropometry.length("hand"))

        if station.seat_height is None:
            station = RowerStation(
                x_ankle=station.x_ankle, z_ankle=station.z_ankle,
                seat_height=self._calibrate_seat_height(),
                foot_half_span=station.foot_half_span,
                hip_half_span=station.hip_half_span,
                shoulder_half_span=station.shoulder_half_span,
            )
        self.station = station

        self.joint_angles = JointAngles.from_dataset(
            self.dataset, timing, n_harmonics=n_harmonics,
            phase_offset=phase_offset,
            include_thigh=(thigh_mode == "measured"),
        )

        self.hand_targets = hand_targets
        self._reference_side = (None if hand_targets is None
                                else sorted(hand_targets)[0])
        self._hand_tracks = {}
        self._elbow_tracks = {}
        self._arm_reach_margin = {}
        if hand_targets is None:
            arm_posture = (DEFAULT_ARM_POSTURE if arm_posture is None
                           else arm_posture)
            self._attach_arm_drivers(arm_posture, n_harmonics, phase_offset)
        else:
            self._attach_constrained_arms(hand_targets, phase_offset)
        self._validate_leg_reach()

    # -- calibration ------------------------------------------------------
    def _calibrate_seat_height(self) -> float:
        """Hip height above the ankle, from the dataset's catch and finish.

        Those two keyframes are the ones where the measured shank and knee
        angles agree on a single hip height (to about 1 mm for the default
        dataset), because the legs are at the extremes of their range and
        the seat is momentarily still.
        """
        heights = self.dataset.hip_height(self.shank_length,
                                          self.thigh_length)
        return float(np.mean([heights[0], heights[2]]))

    def _attach_arm_drivers(self, arm_posture, n_harmonics, phase_offset):
        """Solve arm inverse kinematics at the keyframes, then smooth."""
        arm_length = self.upper_arm_length + self.forearm_length
        drive = self.timing.drive_fraction

        phases, upper, fore = [], [], []
        for drive_fraction, extension, elevation in arm_posture:
            phases.append(drive_fraction * drive)
            reach = extension * arm_length
            bearing = np.radians(elevation)
            label = f"drive fraction {drive_fraction:.2f}"
            upper_angle, fore_angle = self._solve_arm_angles(
                reach * np.cos(bearing), reach * np.sin(bearing), label
            )
            upper.append(upper_angle)
            fore.append(fore_angle)

        phases = np.asarray(phases)
        if np.any(phases >= 1.0):
            raise ValueError(
                "arm keyframe phases must stay within one stroke; got "
                f"{phases.max():.3f} of a period"
            )

        # One more harmonic than the leg/trunk drivers: the arm draw is
        # deliberately concentrated in the last third of the drive, and
        # three harmonics cannot resolve that without rounding it away.
        arm_harmonics = n_harmonics + 1

        def build(values_deg):
            # Unwrap before fitting.  Link angles come out of atan2 on the
            # principal branch, so a sequence that physically sweeps through
            # +-180 deg would otherwise be interpolated the long way round,
            # producing a spurious full rotation and enormous accelerations.
            radians = np.unwrap(np.radians(values_deg))
            profile = FourierProfile.from_keyframes(
                phases, radians, self.timing, arm_harmonics
            )
            return _shift_phase(profile, phase_offset)

        self.joint_angles.upper_arm = build(upper)
        self.joint_angles.forearm = build(fore)

    def _solve_arm_angles(self, dx: float, dz: float, label: str):
        """Two-link IK for the arm, in degrees, elbow trailing downwards.

        Solved once per keyframe at construction, so the singularity at
        full extension is a construction-time error with a clear message
        rather than a ``nan`` inside an integration step.
        """
        reach = np.hypot(dx, dz)
        upper, fore = self.upper_arm_length, self.forearm_length
        if reach > 0.999 * (upper + fore):
            raise ValueError(
                f"arm posture {label!r} demands a reach of {reach:.3f} m but "
                f"the arm is only {upper + fore:.3f} m long"
            )
        if reach < abs(upper - fore) * 1.001:
            raise ValueError(
                f"arm posture {label!r} folds the arm past its inner limit"
            )

        base = np.arctan2(dz, dx)
        # law of cosines for the shoulder-side interior angle
        cos_offset = np.clip(
            (reach ** 2 + upper ** 2 - fore ** 2) / (2.0 * reach * upper),
            -1.0, 1.0,
        )
        offset = np.arccos(cos_offset)

        # Two solutions exist; take the one with the elbow trailing DOWN and
        # behind the shoulder-hand line, which is what a rower does at the
        # finish.  The hands are sternward (base near pi), so rotating
        # *further* positive swings the elbow below the shoulder.
        upper_angle = base + offset
        elbow_x = upper * np.cos(upper_angle)
        elbow_z = upper * np.sin(upper_angle)
        fore_angle = np.arctan2(dz - elbow_z, dx - elbow_x)
        return np.degrees(upper_angle), np.degrees(fore_angle)

    # -- arms constrained to the oar handle -------------------------------
    def shoulder_position(self, t, side: int = CENTRELINE,
                          rotation=0.0) -> np.ndarray:
        """Shoulder joint in the hull frame, from the legs and trunk only.

        Independent of the arms, so it can drive the arm inverse kinematics
        without circularity.
        """
        axis_x, axis_z = self.trunk_axis_point(t)
        shoulder_x, shoulder_z = axis_x, axis_z
        return self._shoulder_from_axis(shoulder_x.value, shoulder_z.value,
                                        side, rotation)

    def _shoulder_from_axis(self, axis_x, axis_z, side, rotation):
        """Place one shoulder given the trunk axis point and a trunk rotation.

        ``rotation`` turns the shoulder line about the vertical through the
        trunk axis; positive carries the port shoulder towards the bow.  At
        zero rotation this reduces to the planar model, a shoulder squarely
        abeam of the trunk axis.
        """
        offset = side * self.station.shoulder_half_span
        axis_x, axis_z, rotation = np.broadcast_arrays(
            np.asarray(axis_x, dtype=float),
            np.asarray(axis_z, dtype=float),
            np.asarray(rotation, dtype=float),
        )
        return np.stack([
            axis_x + offset * np.sin(rotation),
            offset * np.cos(rotation) * np.ones_like(axis_x),
            axis_z,
        ], axis=-1)

    def trunk_axis_point(self, t):
        """The shoulder-height point on the trunk centreline, as ``(x, z)``.

        Independent of both the arms and the trunk rotation.
        """
        t = np.asarray(t, dtype=float)
        zero = np.zeros_like(t)
        angles = self.joint_angles

        ankle_x = constant(self.station.x_ankle + zero)
        ankle_z = constant(self.station.z_ankle + zero)
        shank = angles.shank(t)
        knee_x = ankle_x + self.shank_length * shank.cos()
        knee_z = ankle_z + self.shank_length * shank.sin()
        thigh_sin, thigh_cos = self.thigh_sincos(t)
        hip_x = knee_x + self.thigh_length * thigh_cos
        hip_z = knee_z + self.thigh_length * thigh_sin
        trunk = angles.trunk(t)
        return (hip_x + self.trunk_length * trunk.cos(),
                hip_z + self.trunk_length * trunk.sin())

    def _attach_constrained_arms(self, hand_targets, phase_offset,
                                 n_samples: int = 256):
        """Put the hands ON the oar handle, and solve the body to suit.

        A rower holds the oar; the handle position is fixed by the rig and
        the oar's sweep angle, so the hands are not free to be prescribed.
        Driving them from an invented posture instead left them up to
        0.46 m off the handle for a sweep eight -- the crew were rowing
        thin air.

        Two things are solved for, once, at construction:

        1. the **trunk rotation** about the vertical at each instant, the
           smallest that brings both shoulders within arm's reach of the
           handle.  A sweep rower must turn towards the rigger to follow
           the handle across the body; without this degree of freedom the
           outside shoulder ends up 0.82 m from a 0.70 m arm.
        2. the **elbow** position, by a two-link solve from the rotated
           shoulder to the handle.

        Hands and elbows are then Fourier-fitted as *position* tracks, so
        the hands-on-handle constraint holds by construction and the
        velocities and accelerations stay exact.
        """
        period = self.timing.period
        times = np.arange(n_samples) / n_samples * period
        axis_x, axis_z = self.trunk_axis_point(times)
        axis_x, axis_z = axis_x.value, axis_z.value

        sides = sorted(hand_targets)
        handles = {side: np.asarray([np.asarray(hand_targets[side](t),
                                                dtype=float)
                                     for t in times])
                   for side in sides}

        self.trunk_rotation = self._solve_trunk_rotation(
            axis_x, axis_z, handles, sides)

        shift = int(round(phase_offset * n_samples)) if phase_offset else 0
        self._hand_tracks = {}
        self._elbow_tracks = {}
        self._arm_reach_margin = {}

        for side in sides:
            shoulders = self._shoulder_from_axis(axis_x, axis_z, side,
                                                 self.trunk_rotation)
            elbows, margin = self._solve_elbows(shoulders, handles[side], side)
            order = np.roll(np.arange(n_samples), shift)
            self._hand_tracks[side] = FourierTrack.fit_samples(
                handles[side][order], period, ARM_TRACK_HARMONICS)
            self._elbow_tracks[side] = FourierTrack.fit_samples(
                elbows[order], period, ARM_TRACK_HARMONICS)
            self._arm_reach_margin[side] = margin

        self._shoulder_tracks = {
            side: FourierTrack.fit_samples(
                self._shoulder_from_axis(axis_x, axis_z, side,
                                         self.trunk_rotation)[
                    np.roll(np.arange(n_samples), shift)],
                period, ARM_TRACK_HARMONICS)
            for side in (PORT, STARBOARD)
        }

    def _solve_trunk_rotation(self, axis_x, axis_z, handles, sides,
                              n_trial: int = 81):
        """Smallest trunk rotation bringing every hand within reach.

        Swept rather than iterated: the objective is cheap, the domain is
        one bounded angle, and a sweep cannot land on a local minimum.  The
        result is the rotation of least magnitude whose worst-case arm
        extension is acceptable; if none qualifies, the one that minimises
        that extension, leaving :meth:`_solve_elbows` to raise with a
        message naming the geometry at fault.
        """
        arm_length = self.upper_arm_length + self.forearm_length
        trials = np.linspace(-MAX_TRUNK_ROTATION, MAX_TRUNK_ROTATION, n_trial)

        n_samples = len(axis_x)
        best = np.zeros(n_samples)
        for index in range(n_samples):
            worst = np.zeros(n_trial)
            for side in sides:
                # each arm reaches its OWN grip: for a sweep rower the two
                # hands sit at different points along the same handle
                shoulder = self._shoulder_from_axis(
                    axis_x[index], axis_z[index], side, trials)
                reach = np.linalg.norm(
                    handles[side][index] - shoulder, axis=-1)
                worst = np.maximum(worst, reach / arm_length)

            acceptable = np.flatnonzero(worst <= COMFORTABLE_ARM_EXTENSION)
            if acceptable.size:
                best[index] = trials[acceptable[
                    np.argmin(np.abs(trials[acceptable]))]]
            else:
                best[index] = trials[int(np.argmin(worst))]
        return best

    def _solve_elbows(self, shoulders, hands, side):
        """Two-link elbow solve for a whole stroke at once.

        Returns ``(elbow_positions, worst_reach_fraction)``.  The elbow is
        placed below and outboard of the shoulder-to-handle line, which is
        where a rower's elbow goes.
        """
        upper, fore = self.upper_arm_length, self.forearm_length
        reach_vector = hands - shoulders
        reach = np.linalg.norm(reach_vector, axis=1)
        arm_length = upper + fore

        # Where the arm alone is not long enough, let the shoulder girdle
        # protract towards the handle -- the mechanism a real rower uses.
        unit = reach_vector / np.maximum(reach, 1e-12)[:, None]
        protraction = np.clip(
            reach - MAX_ARM_EXTENSION * arm_length, 0.0,
            MAX_SHOULDER_PROTRACTION)
        shoulders = shoulders + protraction[:, None] * unit
        reach_vector = hands - shoulders
        reach = np.linalg.norm(reach_vector, axis=1)

        worst = float(reach.max() / arm_length)
        if worst > 1.0:
            index = int(np.argmax(reach))
            raise ValueError(
                f"the rower cannot reach the oar handle: at stroke phase "
                f"{index / len(reach):.2f} the shoulder is {reach[index]:.3f} m "
                f"from the handle but the arm is only {arm_length:.3f} m "
                f"long, even with {MAX_SHOULDER_PROTRACTION * 100:.0f} cm of "
                f"shoulder protraction and up to "
                f"{np.degrees(MAX_TRUNK_ROTATION):.0f} deg of trunk rotation. "
                f"Check the rig span, the oar inboard, or the seat station "
                f"against the athlete's stature."
            )

        safe_reach = np.minimum(reach, MAX_ARM_EXTENSION * arm_length)
        unit = reach_vector / reach[:, None]

        # Angle between the reach line and the upper arm (law of cosines).
        cos_offset = np.clip(
            (safe_reach ** 2 + upper ** 2 - fore ** 2)
            / (2.0 * safe_reach * upper), -1.0, 1.0)
        offset = np.arccos(cos_offset)

        # Swing the elbow out of the reach line, downwards and outboard.
        # Build an in-plane perpendicular: the component of -z orthogonal to
        # the reach direction, nudged outboard so a straight arm still has a
        # well-defined elbow plane.
        outboard = np.zeros_like(unit)
        outboard[:, 1] = side if side != CENTRELINE else 1.0
        drop = np.array([0.0, 0.0, -1.0]) + 0.35 * outboard
        perpendicular = drop - (drop * unit).sum(axis=1)[:, None] * unit
        norm = np.linalg.norm(perpendicular, axis=1)
        perpendicular = perpendicular / np.maximum(norm, 1e-9)[:, None]

        elbows = shoulders + upper * (
            np.cos(offset)[:, None] * unit
            + np.sin(offset)[:, None] * perpendicular)
        return elbows, worst

    def _validate_leg_reach(self, n_samples: int = 256) -> None:
        """Check the level-track constraint stays solvable over a stroke."""
        if self.thigh_mode != "level_seat":
            return
        period = self.timing.period
        times = np.linspace(0.0, period, n_samples, endpoint=False)
        shank_angle = self.joint_angles.shank(times).value
        sine = ((self.station.seat_height
                 - self.shank_length * np.sin(shank_angle))
                / self.thigh_length)
        worst = int(np.argmax(np.abs(sine)))
        if np.abs(sine[worst]) > 0.97:
            raise ValueError(
                "level-seat constraint is unreachable: at shank angle "
                f"{np.degrees(shank_angle[worst]):.1f} deg the thigh would "
                f"need sin(angle) = {sine[worst]:.3f}. Reduce the shank "
                "angle range, raise seat_height, or lengthen the thigh."
            )

    # -- mass -------------------------------------------------------------
    @property
    def segment_masses(self) -> np.ndarray:
        """Shape ``(12,)``, ordered by :data:`SEGMENT_ORDER`."""
        return self._masses

    @property
    def total_mass(self) -> float:
        return float(self._masses.sum())

    # -- kinematics -------------------------------------------------------
    def thigh_sincos(self, t):
        """``(sin, cos)`` of the thigh link angle, as :class:`Jet2` pairs.

        In ``level_seat`` mode this comes from the constant-hip-height
        constraint; working with the sine and cosine directly avoids an
        ``arcsin``, and hence avoids putting a branch cut into a quantity
        that gets differentiated twice.
        """
        if self.thigh_mode == "measured":
            angle = self.joint_angles.thigh(t)
            return angle.sin(), angle.cos()

        shank = self.joint_angles.shank(t)
        sine = (constant(self.station.seat_height)
                - self.shank_length * shank.sin()) / self.thigh_length
        cosine = (1.0 - sine * sine).sqrt()  # hip is always bow-ward of knee
        return sine, cosine

    def thigh_angle(self, t) -> np.ndarray:
        """Thigh link angle in radians, for inspection and plots."""
        sine, cosine = self.thigh_sincos(t)
        return np.arctan2(sine.value, cosine.value)

    def _chain(self, t):
        """Evaluate every joint centre as a pair of ``Jet2`` (x, z)."""
        angles = self.joint_angles

        def link(base_x, base_z, length, angle_jet):
            return (base_x + length * angle_jet.cos(),
                    base_z + length * angle_jet.sin())

        t = np.asarray(t, dtype=float)
        zero = np.zeros_like(t)

        ankle_x = constant(self.station.x_ankle + zero)
        ankle_z = constant(self.station.z_ankle + zero)

        knee_x, knee_z = link(ankle_x, ankle_z, self.shank_length,
                              angles.shank(t))
        thigh_sin, thigh_cos = self.thigh_sincos(t)
        hip_x = knee_x + self.thigh_length * thigh_cos
        hip_z = knee_z + self.thigh_length * thigh_sin
        shoulder_x, shoulder_z = link(hip_x, hip_z, self.trunk_length,
                                      angles.trunk(t))

        chain = {
            "ankle": (ankle_x, ankle_z),
            "knee": (knee_x, knee_z),
            "hip": (hip_x, hip_z),
            "shoulder": (shoulder_x, shoulder_z),
        }

        if self.hand_targets is None:
            elbow_x, elbow_z = link(shoulder_x, shoulder_z,
                                    self.upper_arm_length,
                                    angles.upper_arm(t))
            hand_x, hand_z = link(elbow_x, elbow_z, self.forearm_length,
                                  angles.forearm(t))
            chain["elbow"] = (elbow_x, elbow_z)
            chain["hand"] = (hand_x, hand_z)
        else:
            # Arms constrained to the oar: the hand track IS the handle.
            # Represented in the chain by the reference side so the
            # single-plane accessors keep working; per-side tracks are
            # used directly by segment_state.
            side = self._reference_side
            hand = self._hand_tracks[side](t)
            elbow = self._elbow_tracks[side](t)
            chain["elbow"] = (elbow[0], elbow[2])
            chain["hand"] = (hand[0], hand[2])

        return chain

    def joint_positions(self, t) -> Dict[str, np.ndarray]:
        """Joint-centre positions in the hull frame, for inspection/plots."""
        chain = self._chain(t)
        return {name: np.array([x.value, 0.0, z.value])
                for name, (x, z) in chain.items()}

    def skeleton(self, t) -> Dict[str, np.ndarray]:
        """Full three-dimensional joint positions, both sides resolved.

        :meth:`joint_positions` returns the *planar* chain, which collapses
        both arms onto the centreline -- fine for the fore-aft dynamics,
        wrong for anything that looks at the rower.  This returns every
        joint with its true lateral position, including the trunk rotation
        and the two separately-solved arms, which is what a sweep rower
        actually looks like: two hands on one handle, off to one side.
        """
        t_array = np.asarray(t, dtype=float)
        chain = self._chain(t_array)
        axis_x, axis_z = chain["shoulder"]
        rotation = self._rotation_at(t_array)

        joints: Dict[str, np.ndarray] = {}
        hip_x, hip_z = chain["hip"]
        joints["hip"] = np.array([hip_x.value, 0.0, hip_z.value])

        for side, suffix in ((PORT, "port"), (STARBOARD, "starboard")):
            joints[f"ankle_{suffix}"] = np.array([
                self.station.x_ankle, side * self.station.foot_half_span,
                self.station.z_ankle])
            knee_x, knee_z = chain["knee"]
            joints[f"knee_{suffix}"] = np.array([
                knee_x.value,
                side * 0.5 * (self.station.foot_half_span
                              + self.station.hip_half_span),
                knee_z.value])

            shoulder = self._shoulder_from_axis(axis_x.value, axis_z.value,
                                                side, rotation)
            joints[f"shoulder_{suffix}"] = np.asarray(shoulder)

            if self.hand_targets is None:
                # unconstrained arms are planar; mirror them for drawing
                elbow_x, elbow_z = chain["elbow"]
                hand_x, hand_z = chain["hand"]
                lateral = side * self.station.shoulder_half_span
                joints[f"elbow_{suffix}"] = np.array(
                    [elbow_x.value, lateral, elbow_z.value])
                joints[f"hand_{suffix}"] = np.array(
                    [hand_x.value, lateral, hand_z.value])
            else:
                track_side = (side if side in self._hand_tracks
                              else self._reference_side)
                joints[f"elbow_{suffix}"] = np.asarray(
                    self._elbow_tracks[track_side].position(t_array))
                joints[f"hand_{suffix}"] = np.asarray(
                    self._hand_tracks[track_side].position(t_array))

        centre = 0.5 * (joints["shoulder_port"] + joints["shoulder_starboard"])
        joints["neck"] = centre
        head_rise = (self.trunk_stack_length - self.trunk_length
                     + self._segments["head"].length
                     * self._segments["head"].com_fraction)
        direction = centre - joints["hip"]
        norm = np.linalg.norm(direction)
        joints["head"] = centre + head_rise * direction / max(norm, 1e-9)
        return joints

    def _rotation_at(self, t):
        """Trunk rotation at time ``t``, interpolated from the solve grid."""
        rotation = getattr(self, "trunk_rotation", None)
        if rotation is None:
            return np.zeros_like(np.asarray(t, dtype=float))
        period = self.timing.period
        grid = np.arange(len(rotation)) / len(rotation) * period
        phase = np.mod(np.asarray(t, dtype=float), period)
        return np.interp(phase, grid, rotation, period=period)

    #: Bones to draw, as pairs of :meth:`skeleton` keys.
    BONES = (
        ("ankle_port", "knee_port"), ("knee_port", "hip"),
        ("ankle_starboard", "knee_starboard"), ("knee_starboard", "hip"),
        ("hip", "shoulder_port"), ("hip", "shoulder_starboard"),
        ("shoulder_port", "shoulder_starboard"),
        ("neck", "head"),
        ("shoulder_port", "elbow_port"), ("elbow_port", "hand_port"),
        ("shoulder_starboard", "elbow_starboard"),
        ("elbow_starboard", "hand_starboard"),
    )

    def hand_path(self, n_samples: int = 200) -> np.ndarray:
        """Handle path over one stroke, shape ``(n, 2)`` as ``(x, z)``."""
        times = np.linspace(0.0, self.timing.period, n_samples, endpoint=False)
        chain = self._chain(times)
        return np.column_stack([chain["hand"][0].value,
                                chain["hand"][1].value])

    def kinematics_signature(self):
        """Key identifying rowers whose segment motion is identical.

        Two rowers sharing this signature move the same way relative to
        their own footboard, so the whole chain need only be evaluated
        once and the result shifted by each seat's ``x_ankle``.  A
        standard crew is homogeneous, which turns nine chain evaluations
        per derivative call into one -- the single largest cost in the
        simulation loop.

        Deliberately excludes ``station.x_ankle``: that is precisely the
        difference the sharing is designed to factor out.

        Keyed on *values* rather than object identity, so two separately
        constructed but identical athletes still share.  Any field that
        could change the motion must appear here; when in doubt, adding a
        field costs a little speed, omitting one costs correctness.
        """
        station = self.station
        anthro = self.anthropometry
        return (
            anthro.mass, anthro.stature, anthro.sex,
            self.dataset.name, self.timing.rate, self.thigh_mode,
            self.phase_offset, self.n_harmonics,
            station.z_ankle, station.seat_height, station.foot_half_span,
            station.hip_half_span, station.shoulder_half_span,
        )

    def segment_state(self, t, x_offsets=None):
        """Position, velocity and acceleration of all 12 segment masses.

        Returns three ``(12, 3)`` arrays in the hull frame, ordered by
        :data:`SEGMENT_ORDER`.  Velocity and acceleration are relative to
        the hull -- the transport terms are added by
        :func:`coxswain.core.rigid_body.moving_mass_reaction`.

        If ``x_offsets`` is given, the chain is evaluated once and returned
        for every offset at once, with shapes ``(len(x_offsets) * 12, 3)``
        ordered seat-major.  The offsets are longitudinal shifts *relative
        to this rower's own* ``x_ankle``.
        """
        if x_offsets is not None:
            return self._segment_state_batched(t, x_offsets)
        chain = self._chain(t)
        seg = self._segments
        anthro = self.anthropometry

        def along(start, end, fraction):
            sx, sz = start
            ex, ez = end
            return (sx + (ex - sx) * fraction, sz + (ez - sz) * fraction)

        ankle, knee = chain["ankle"], chain["knee"]
        hip, shoulder = chain["hip"], chain["shoulder"]
        elbow, hand = chain["elbow"], chain["hand"]

        lower_len = anthro.length("lower_trunk")
        mid_len = anthro.length("mid_trunk")
        upper_len = anthro.length("upper_trunk")
        lower_com = lower_len * (1.0 - seg["lower_trunk"].com_fraction)
        mid_com = lower_len + mid_len * (1.0 - seg["mid_trunk"].com_fraction)
        upper_com = (lower_len + mid_len
                     + upper_len * (1.0 - seg["upper_trunk"].com_fraction))

        # The three trunk masses are distributed along the full anatomical
        # trunk; the kinematic link is shorter (it ends at the shoulder
        # joint), so fractions are taken against the anatomical length and
        # rescaled onto the link.
        trunk_len = self.trunk_length
        stack_len = self.trunk_stack_length
        head_offset = (stack_len
                       + seg["head"].length * seg["head"].com_fraction)

        arm_places = {}
        places = {
            "head": (along(hip, shoulder, head_offset / trunk_len), CENTRELINE),
            "upper_trunk": (along(hip, shoulder, upper_com / trunk_len),
                            CENTRELINE),
            "mid_trunk": (along(hip, shoulder, mid_com / trunk_len), CENTRELINE),
            "lower_trunk": (along(hip, shoulder, lower_com / trunk_len),
                            CENTRELINE),
        }
        del stack_len
        for side, suffix in ((PORT, "port"), (STARBOARD, "starboard")):
            if self.hand_targets is not None:
                # Each arm reaches its own target: for sculling that is its
                # own handle, for sweep both arms reach the single handle.
                track_side = (side if side in self._hand_tracks
                              else self._reference_side)
                side_hand = self._hand_tracks[track_side](t)
                side_elbow = self._elbow_tracks[track_side](t)
                side_shoulder = (shoulder[0], shoulder[1])
                arm_places[suffix] = (side_shoulder, side_elbow, side_hand)
            places[f"upper_arm_{suffix}"] = (
                along(shoulder, elbow, seg[f"upper_arm_{suffix}"].com_fraction),
                side)
            places[f"forearm_hand_{suffix}"] = (
                along(elbow, hand, seg[f"forearm_hand_{suffix}"].com_fraction),
                side)
            places[f"thigh_{suffix}"] = (
                along(knee, hip, seg[f"thigh_{suffix}"].com_fraction), side)
            places[f"shank_foot_{suffix}"] = (
                along(ankle, knee, seg[f"shank_foot_{suffix}"].com_fraction),
                side)

        n = len(SEGMENT_ORDER)
        position = np.zeros((n, 3))
        velocity = np.zeros((n, 3))
        acceleration = np.zeros((n, 3))

        for index, name in enumerate(SEGMENT_ORDER):
            (x_jet, z_jet), side = places[name]
            position[index] = (x_jet.value, self._lateral(name, side),
                               z_jet.value)
            velocity[index] = (x_jet.first, 0.0, z_jet.first)
            acceleration[index] = (x_jet.second, 0.0, z_jet.second)

        if arm_places:
            self._place_arm_segments(position, velocity, acceleration,
                                     arm_places, seg)

        return position, velocity, acceleration

    def _place_arm_segments(self, position, velocity, acceleration,
                            arm_places, seg):
        """Overwrite the arm rows with the true 3-D constrained tracks.

        The planar chain cannot represent an arm reaching a handle that is
        off the centreline, which is every sweep stroke.  These four
        segments therefore carry genuine lateral motion, and with it a real
        crew contribution to the roll and yaw moments.
        """
        index_of = {name: i for i, name in enumerate(SEGMENT_ORDER)}
        for suffix, (shoulder_xz, elbow, hand) in arm_places.items():
            side = PORT if suffix == "port" else STARBOARD
            shoulder = (shoulder_xz[0],
                        constant(np.zeros_like(shoulder_xz[0].value)
                                 + side * self.station.shoulder_half_span),
                        shoulder_xz[1])

            for name, start, end in (
                    (f"upper_arm_{suffix}", shoulder, elbow),
                    (f"forearm_hand_{suffix}", elbow, hand)):
                fraction = seg[name].com_fraction
                row = index_of[name]
                for axis in range(3):
                    blended = start[axis] + (end[axis] - start[axis]) * fraction
                    position[row, axis] = blended.value
                    velocity[row, axis] = blended.first
                    acceleration[row, axis] = blended.second

    def _segment_state_batched(self, t, x_offsets):
        """Share one chain evaluation across seats that move identically.

        Velocities and accelerations are identical for every seat -- only
        the constant longitudinal offset differs -- so they are broadcast
        rather than recomputed.
        """
        position, velocity, acceleration = self.segment_state(t)
        offsets = np.asarray(x_offsets, dtype=float)
        n_seats = len(offsets)

        tiled = np.tile(position, (n_seats, 1))
        tiled[:, 0] += np.repeat(offsets, position.shape[0])
        return (tiled,
                np.tile(velocity, (n_seats, 1)),
                np.tile(acceleration, (n_seats, 1)))

    def _lateral(self, name: str, side: int) -> float:
        """Fixed ``y`` offset of a segment.

        The chain is planar, so lateral motion is not modelled; roll
        moments come from asymmetric oar forces rather than from the crew
        swaying.  The offsets still matter because they set the crew's
        contribution to the roll inertia.
        """
        if side == CENTRELINE:
            return 0.0
        if name.startswith("shank_foot"):
            return side * self.station.foot_half_span
        if name.startswith("thigh"):
            return side * 0.5 * (self.station.foot_half_span
                                 + self.station.hip_half_span)
        return side * self.station.shoulder_half_span

    # -- derived quantities used for calibration and tests ----------------
    def centre_of_mass(self, t) -> np.ndarray:
        """Crew centre of mass in the hull frame at time ``t``."""
        position, _, _ = self.segment_state(t)
        return (self._masses[:, None] * position).sum(axis=0) / self.total_mass

    def _sample(self, key: str, index: int, n_samples: int) -> np.ndarray:
        times = np.linspace(0.0, self.timing.period, n_samples, endpoint=False)
        chain = self._chain(times)
        return chain[key][index].value

    def slide_travel(self, n_samples: int = 400) -> float:
        """Peak-to-peak longitudinal travel of the seat (hip joint)."""
        hip_x = self._sample("hip", 0, n_samples)
        return float(hip_x.max() - hip_x.min())

    def seat_height_variation(self, n_samples: int = 400) -> float:
        """Peak-to-peak vertical movement of the hip joint."""
        hip_z = self._sample("hip", 1, n_samples)
        return float(hip_z.max() - hip_z.min())

    def handle_travel(self, n_samples: int = 400) -> float:
        """Peak-to-peak longitudinal travel of the hands."""
        hand_x = self._sample("hand", 0, n_samples)
        return float(hand_x.max() - hand_x.min())
