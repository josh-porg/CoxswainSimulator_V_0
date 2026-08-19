"""Rower body-segment kinematics from a joint-driven serial linkage.

Formaggia et al. obtained ``x_ij``, ``x_dot_ij`` and ``x_ddot_ij`` -- the
hull-frame position, velocity and acceleration of each of the 12 body
segments -- by reconstructing motion-capture trajectories with fitted
analytic functions.  Without that mocap data, this module takes the other
route found in the literature (Serveto et al., *A three-dimensional model
of the boat-oars-rower system using ADAMS and LifeMOD*, Proc. IMechE Part
P, 2010): treat the rower as an actuated serial kinematic chain and
prescribe the **joint angles**, then obtain every segment's motion by
forward kinematics.

Why forward kinematics rather than inverse
------------------------------------------
Driving the *seat position* and solving inverse kinematics for the knee
introduces a ``sqrt`` that goes singular at full leg extension -- exactly
the configuration a rower passes through at every finish, where segment
accelerations would blow up.  Driving the joint angles directly is
unconditionally well posed: it is pure composition of sines and cosines,
with no branch and no singularity anywhere in the workspace.

Exact derivatives
-----------------
The whole chain is evaluated in :class:`~coxswain.core.taylor.Jet2`
arithmetic, so velocities and accelerations are differentiated
automatically rather than by hand.  Combined with the smooth
:class:`~coxswain.crew.stroke.FourierProfile` joint drivers, the resulting
segment accelerations are continuous everywhere -- no impulsive force at
the catch.

Frame
-----
All positions are in the **hull frame**, measured from the hull centre of
mass ``G_h``.  The chain lies in the ``x``-``z`` plane; port/starboard
limbs are displaced in ``y`` by a fixed half-span.  Link angles are
measured from the ``+x`` (bow) axis, positive towards ``+z`` (up), so
``0`` points at the bow, ``90 deg`` straight up and ``180 deg`` at the
stern.  The rower faces the stern, so the drive sweeps the hands from
about ``180 deg`` towards the bow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..core.taylor import Jet2, constant
from .anthropometry import CENTRELINE, PORT, STARBOARD, RowerAnthropometry
from .stroke import DEFAULT_HARMONICS, FourierProfile, StrokeTiming

__all__ = [
    "RowerStation",
    "JointAngles",
    "DEFAULT_JOINT_ANGLES",
    "DEFAULT_JOINT_PHASES",
    "JointDrivenRower",
    "SEGMENT_ORDER",
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

#: Catch and finish angles for each driven joint, in degrees.
#:
#: These are link *direction* angles (from the +x axis towards +z), not
#: relative joint angles, which makes them directly readable off a video
#: frame.  Values are calibrated so the derived seat travel, trunk sweep and
#: crew centre-of-mass excursion match published on-water measurements for a
#: sweep eight -- see :func:`JointDrivenRower.slide_travel` and the
#: calibration assertions in ``tests/unit/test_kinematics.py``.
#:
#: The thigh is **not** listed: it is not free.  The seat runs on a level
#: track, so the hip height above the ankle is fixed and the thigh angle
#: follows from the shank angle by
#: ``L_thigh sin(a_thigh) = seat_height - L_shank sin(a_shank)``.
DEFAULT_JOINT_ANGLES: Dict[str, Tuple[float, float]] = {
    # link            (catch, finish) in degrees
    "shank":          (95.0, 14.0),
    "trunk":          (118.0, 62.0),
    "upper_arm":      (191.0, 236.0),
    "forearm":        (186.0, 132.0),
}

#: Per-joint phase lag, as a fraction of the stroke period.
#:
#: Rowers do not move every joint at once: the drive sequences legs, then
#: back, then arms, and the recovery reverses it.  Driving all joints with
#: a common phase concentrates the whole crew's momentum change into one
#: burst and roughly doubles the hull's speed fluctuation -- an eight
#: comes out swinging 2.9 m/s peak to peak instead of the measured
#: 1.2-1.5 m/s.  Staggering the joints spreads the same total momentum
#: transfer over more of the cycle, which is a large part of what good
#: sequencing buys a crew.
DEFAULT_JOINT_PHASES: Dict[str, float] = {
    "shank": 0.00,       # legs lead
    "trunk": 0.10,       # the back opens after the legs
    "upper_arm": 0.18,   # arms break last
    "forearm": 0.20,
}


@dataclass(frozen=True)
class RowerStation:
    """Where one rower is anchored in the hull frame.

    ``x_ankle`` places the footboard longitudinally; the whole chain grows
    from there.  Half-spans are lateral (``y``) offsets applied to the
    paired limbs.  ``seat_height`` is the height of the hip joint above the
    ankle joint and is held constant -- the seat runs on a level track,
    which is what removes the thigh angle as an independent driver.
    """

    x_ankle: float
    z_ankle: float = -0.05
    seat_height: float = 0.13
    foot_half_span: float = 0.16
    hip_half_span: float = 0.10
    shoulder_half_span: float = 0.20


@dataclass
class JointAngles:
    """Fourier-smoothed joint-angle drivers for one rower.

    The thigh is absent by design; see :data:`DEFAULT_JOINT_ANGLES`.
    """

    shank: FourierProfile
    trunk: FourierProfile
    upper_arm: FourierProfile
    forearm: FourierProfile

    @classmethod
    def from_catch_finish(cls, timing: StrokeTiming,
                          angles: Dict[str, Tuple[float, float]] = None,
                          n_harmonics: int = DEFAULT_HARMONICS,
                          phase_offset: float = 0.0,
                          joint_phases: Dict[str, float] = None
                          ) -> "JointAngles":
        """Build drivers from catch/finish angle pairs given in degrees.

        ``phase_offset`` shifts this rower's whole stroke in normalised
        phase.  It is zero for a synchronised crew; a small non-zero value
        models imperfect timing, which is one of the things a
        coxswain-facing simulator eventually wants to quantify.

        ``joint_phases`` staggers the joints *within* the stroke -- the
        legs-back-arms sequence.  See :data:`DEFAULT_JOINT_PHASES`.
        """
        angles = dict(DEFAULT_JOINT_ANGLES if angles is None else angles)
        phases = dict(DEFAULT_JOINT_PHASES if joint_phases is None
                      else joint_phases)
        built = {}
        for name, (catch, finish) in angles.items():
            profile = FourierProfile.from_catch_finish(
                np.radians(catch), np.radians(finish), timing, n_harmonics
            )
            built[name] = _shift_phase(
                profile, phase_offset + phases.get(name, 0.0))
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
        # f(t - offset*T): cos and sin coefficients mix
        shifted_cos[k] = cos_c[k] * c - sin_c[k] * s
        shifted_sin[k] = sin_c[k] * c + cos_c[k] * s
    return FourierProfile(shifted_cos, shifted_sin, profile.period)


class JointDrivenRower:
    """One rower, as an actuated planar linkage anchored at the footboard.

    The chain runs::

        ankle (fixed) -> knee -> hip/seat -> shoulder -> elbow -> hand

    with the trunk treated as a single rigid link from hip to shoulder,
    along which the three de Leva trunk masses are distributed.
    """

    def __init__(self, anthropometry: RowerAnthropometry,
                 station: RowerStation, joint_angles: JointAngles):
        self.anthropometry = anthropometry
        self.station = station
        self.joint_angles = joint_angles

        self._segments = {s.name: s for s in anthropometry.segments}
        self._masses = np.array(
            [self._segments[name].mass for name in SEGMENT_ORDER]
        )

        # Link lengths taken straight from the anthropometry table.
        self.shank_length = anthropometry.length("shank")
        self.thigh_length = anthropometry.length("thigh")
        self.trunk_length = (anthropometry.length("lower_trunk")
                             + anthropometry.length("mid_trunk")
                             + anthropometry.length("upper_trunk"))
        self.upper_arm_length = anthropometry.length("upper_arm")
        self.forearm_length = (anthropometry.length("forearm")
                               + anthropometry.length("hand"))

        self._validate_leg_reach()

    def _validate_leg_reach(self, n_samples: int = 128) -> None:
        """Check the level-track constraint stays solvable over a stroke.

        ``sin(a_thigh) = (seat_height - L_shank sin(a_shank)) / L_thigh``
        has no solution if the shank angle demands more vertical drop than
        the thigh can supply.  Catch it here, with a message naming the
        offending angle, rather than as a ``nan`` deep inside a derivative
        evaluation.
        """
        period = self.joint_angles.shank.period
        times = np.linspace(0.0, period, n_samples, endpoint=False)
        shank_angle = self.joint_angles.shank(times).value
        sine = ((self.station.seat_height
                 - self.shank_length * np.sin(shank_angle))
                / self.thigh_length)
        worst = np.argmax(np.abs(sine))
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
        """``(sin, cos)`` of the thigh angle from the level-seat constraint.

        Returned as :class:`Jet2` pairs.  Working with the sine and cosine
        directly avoids an ``arcsin`` -- and hence avoids introducing a
        branch cut into a quantity that is differentiated twice.
        """
        shank = self.joint_angles.shank(t)
        sine = (constant(self.station.seat_height)
                - self.shank_length * shank.sin()) / self.thigh_length
        cosine = (1.0 - sine * sine).sqrt()  # hip is always bow-ward of knee
        return sine, cosine

    def thigh_angle(self, t) -> np.ndarray:
        """Derived thigh link angle in radians, for inspection and plots."""
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

    def segment_state(self, t):
        """Position, velocity and acceleration of all 12 segment masses.

        Returns three ``(12, 3)`` arrays in the hull frame, ordered by
        :data:`SEGMENT_ORDER`.  Velocity and acceleration are relative to
        the hull -- the transport terms are added by
        :func:`coxswain.core.rigid_body.moving_mass_reaction`.
        """
        chain = self._chain(t)
        seg = self._segments
        anthro = self.anthropometry

        def along(start, end, fraction):
            """Point a given fraction of the way from ``start`` to ``end``."""
            sx, sz = start
            ex, ez = end
            return (sx + (ex - sx) * fraction, sz + (ez - sz) * fraction)

        ankle, knee = chain["ankle"], chain["knee"]
        hip, shoulder = chain["hip"], chain["shoulder"]
        elbow, hand = chain["elbow"], chain["hand"]

        # Trunk sub-masses: heights above the hip along the hip->shoulder link.
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

    def slide_travel(self, timing: StrokeTiming, n_samples: int = 400) -> float:
        """Peak-to-peak longitudinal travel of the seat (hip joint)."""
        times = np.linspace(0.0, timing.period, n_samples, endpoint=False)
        hip_x = np.array([self.joint_positions(t)["hip"][0] for t in times])
        return float(hip_x.max() - hip_x.min())

    def handle_travel(self, timing: StrokeTiming, n_samples: int = 400) -> float:
        """Peak-to-peak longitudinal travel of the hands."""
        times = np.linspace(0.0, timing.period, n_samples, endpoint=False)
        hand_x = np.array([self.joint_positions(t)["hand"][0] for t in times])
        return float(hand_x.max() - hand_x.min())
