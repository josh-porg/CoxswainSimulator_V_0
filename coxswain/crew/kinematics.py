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
from .stroke import FourierProfile, StrokeTiming
from .stroke_data import StrokeKinematicsDataset, default_dataset

__all__ = [
    "RowerStation",
    "JointAngles",
    "JointDrivenRower",
    "SEGMENT_ORDER",
    "THIGH_MODES",
    "DEFAULT_ARM_POSTURE",
    "KEYFRAME_HARMONICS",
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
                 arm_posture: Dict[str, Tuple[float, float]] = None,
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
        self.trunk_length = (anthropometry.length("lower_trunk")
                             + anthropometry.length("mid_trunk")
                             + anthropometry.length("upper_trunk"))
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

        arm_posture = DEFAULT_ARM_POSTURE if arm_posture is None else arm_posture
        self._attach_arm_drivers(arm_posture, n_harmonics, phase_offset)
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
        elbow_x, elbow_z = link(shoulder_x, shoulder_z, self.upper_arm_length,
                                angles.upper_arm(t))
        hand_x, hand_z = link(elbow_x, elbow_z, self.forearm_length,
                              angles.forearm(t))

        return {
            "ankle": (ankle_x, ankle_z),
            "knee": (knee_x, knee_z),
            "hip": (hip_x, hip_z),
            "shoulder": (shoulder_x, shoulder_z),
            "elbow": (elbow_x, elbow_z),
            "hand": (hand_x, hand_z),
        }

    def joint_positions(self, t) -> Dict[str, np.ndarray]:
        """Joint-centre positions in the hull frame, for inspection/plots."""
        chain = self._chain(t)
        return {name: np.array([x.value, 0.0, z.value])
                for name, (x, z) in chain.items()}

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

        trunk_len = self.trunk_length
        head_offset = trunk_len + seg["head"].length * seg["head"].com_fraction

        places = {
            "head": (along(hip, shoulder, head_offset / trunk_len), CENTRELINE),
            "upper_trunk": (along(hip, shoulder, upper_com / trunk_len),
                            CENTRELINE),
            "mid_trunk": (along(hip, shoulder, mid_com / trunk_len), CENTRELINE),
            "lower_trunk": (along(hip, shoulder, lower_com / trunk_len),
                            CENTRELINE),
        }
        for side, suffix in ((PORT, "port"), (STARBOARD, "starboard")):
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

        return position, velocity, acceleration

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
