"""A complete boat: hull, rig, crew and appendages.

:class:`Boat` is the single object the simulator consumes, and the seam
along which everything is swappable.  An eight and a coxed four differ
only in the :class:`Boat` handed to
:class:`~coxswain.sim.simulator.RowingSimulator`; nothing in the dynamics
knows which is which.

Composition
-----------
``Boat`` owns:

* a :class:`~coxswain.hydro.hull.HullMesh` -- the shape, from an offsets
  table, which is what makes different hulls genuinely different rather
  than a change of coefficient;
* a :class:`~coxswain.boats.rig.Rig` -- seats, oarlocks and oars;
* one :class:`~coxswain.crew.kinematics.JointDrivenRower` per seat;
* a list of :class:`~coxswain.hydro.appendages.LiftingSurface`.

Mass bookkeeping follows the paper: ``G_h`` is the *hull* centre of mass
and the hull inertia is taken about it, while the total mass ``M_t``
entering the translational equation includes the crew.  The crew's
offsets contribute to the mass matrix through
:func:`coxswain.core.rigid_body.assemble_mass_matrix`, so they must not
also be folded into the hull inertia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from ..crew.anthropometry import PORT, STARBOARD, RowerAnthropometry
from ..crew.kinematics import JointDrivenRower, RowerStation
from ..crew.oarlock import OarAngleSweep, OarForceProfile
from ..crew.stroke import StrokeTiming
from ..crew.stroke_data import StrokeKinematicsDataset, default_dataset
from ..hydro.appendages import LiftingSurface
from ..hydro.hull import HullMesh, HullOffsets
from ..hydro.resistance import FRESH_WATER, ResistanceCoefficients, WaterProperties
from ..hydro.shallow import ShallowWaterModel
from .rig import Rig

__all__ = ["Boat", "CrewMember"]


@dataclass
class CrewMember:
    """One athlete: their anthropometry, their seat, and their kinematics."""

    rower: JointDrivenRower
    seat_index: int

    @property
    def mass(self) -> float:
        return self.rower.total_mass


class Boat:
    """Hull + rig + crew + appendages, ready to simulate."""

    def __init__(self, name: str, offsets: HullOffsets, rig: Rig,
                 hull_mass: float, hull_inertia: np.ndarray,
                 timing: StrokeTiming,
                 anthropometry: Sequence[RowerAnthropometry] = None,
                 appendages: Sequence[LiftingSurface] = (),
                 water: WaterProperties = FRESH_WATER,
                 resistance: ResistanceCoefficients = None,
                 force_profile: OarForceProfile = None,
                 oar_sweep: OarAngleSweep = None,
                 n_girth: int = 16,
                 crew_phase_offsets: Sequence[float] = None,
                 default_anthropometry: RowerAnthropometry = None,
                 stroke_dataset: StrokeKinematicsDataset = None,
                 shallow: ShallowWaterModel = None,
                 blade_model=None):
        self.name = name
        self.offsets = offsets
        self.mesh = HullMesh(offsets, n_girth=n_girth)
        self.rig = rig
        self.hull_mass = float(hull_mass)
        self.hull_inertia = np.asarray(hull_inertia, dtype=float)
        self.timing = timing
        self.appendages = tuple(appendages)
        self.water = water
        self.resistance = resistance or ResistanceCoefficients()
        self.force_profile = force_profile or OarForceProfile()
        self.oar_sweep = oar_sweep or OarAngleSweep()
        self.stroke_dataset = stroke_dataset or default_dataset()
        #: Finite-depth correction; deep water unless a depth is given.
        self.shallow = shallow or ShallowWaterModel()
        #: Optional :class:`~coxswain.crew.oarlock.BladeModel`.
        #: When set, the simulator computes blade efficiency
        #: from slip and water depth each step instead of
        #: using the oar's fixed ``blade_efficiency``.
        self.blade_model = blade_model

        if self.hull_inertia.shape != (3, 3):
            raise ValueError("hull_inertia must be a 3x3 tensor")
        if np.linalg.eigvalsh(self.hull_inertia).min() <= 0:
            raise ValueError("hull_inertia must be positive definite")

        self.crew = self._build_crew(anthropometry, crew_phase_offsets,
                                     default_anthropometry)
        self._validate()

    # -- construction ----------------------------------------------------
    def _build_crew(self, anthropometry, phase_offsets,
                    default_anthropometry) -> Tuple[CrewMember, ...]:
        n_seats = self.rig.n_seats
        if anthropometry is None:
            template = default_anthropometry or RowerAnthropometry(
                mass=85.0, stature=1.88)
            anthropometry = [template] * n_seats
        if len(anthropometry) != n_seats:
            raise ValueError(
                f"{self.rig.n_seats} seats but {len(anthropometry)} rowers"
            )
        offsets = ([0.0] * n_seats if phase_offsets is None
                   else list(phase_offsets))
        if len(offsets) != n_seats:
            raise ValueError("one phase offset per seat is required")

        crew: List[CrewMember] = []
        for index, (athlete, seat, offset) in enumerate(
                zip(anthropometry, self.rig.seats, offsets)):
            station = RowerStation(x_ankle=seat.station_x)
            crew.append(CrewMember(
                rower=JointDrivenRower(
                    athlete, station, self.timing,
                    dataset=self.stroke_dataset, phase_offset=offset,
                    hand_targets=self._hand_targets(seat),
                ),
                seat_index=index,
            ))
        return tuple(crew)

    def _hand_targets(self, seat):
        """Where this rower's hands must be, as ``{side: callable(t)}``.

        A rower holds the oar, so the hands are not free -- they sit on the
        handle, whose position follows from the rig geometry and the oar's
        sweep angle.  This is what closes the loop between the crew
        kinematics and the rig.

        A **sculler** has an oarlock each side and each hand takes its own
        handle.  A **sweep** rower has one oar held in both hands, so both
        arms are given the same target, off the centreline -- which is what
        makes a sweep crew's arm motion genuinely asymmetric, and the
        source of the crew's own contribution to the roll and yaw couple.
        """
        from ..crew.oarlock import handle_position

        def target_for(lock, grip_offset=0.0):
            def target(t):
                return handle_position(t, self.timing, lock, self.oar_sweep,
                                       grip_offset)
            return target

        if seat.is_sculling:
            return {lock.side: target_for(lock) for lock in seat.oarlocks}

        lock = seat.oarlocks[0]
        # The outside hand -- the one away from the rigger -- takes the end
        # of the handle; the inside hand sits a grip separation closer to
        # the oarlock.  Without that spread the outside arm cannot reach.
        return {
            -lock.side: target_for(lock, 0.0),
            lock.side: target_for(lock, lock.oar.grip_separation),
        }

    def _validate(self) -> None:
        capacity = self.offsets.design_displacement(self.water.density)
        if self.total_mass > capacity:
            raise ValueError(
                f"{self.name}: all-up mass {self.total_mass:.0f} kg exceeds "
                f"the hull's design displacement {capacity:.0f} kg; the boat "
                "would swamp"
            )

    # -- mass properties -------------------------------------------------
    @property
    def crew_mass(self) -> float:
        return float(sum(member.mass for member in self.crew))

    @property
    def coxswain_mass(self) -> float:
        return float(self.rig.coxswain_mass)

    @property
    def total_mass(self) -> float:
        """``M_t`` of the paper: hull + crew + coxswain."""
        return self.hull_mass + self.crew_mass + self.coxswain_mass

    @property
    def length(self) -> float:
        return self.offsets.length

    @property
    def n_seats(self) -> int:
        return self.rig.n_seats

    # -- crew state ------------------------------------------------------
    def _crew_groups(self):
        """Group seats whose rowers move identically, for batch evaluation.

        Built once and reused: a homogeneous crew collapses to a single
        group, so a derivative evaluation costs one kinematic chain rather
        than one per seat.  Each entry is
        ``(representative_rower, seat_indices, x_offsets)``.
        """
        if getattr(self, "_crew_group_cache", None) is not None:
            return self._crew_group_cache

        groups = {}
        for member in self.crew:
            groups.setdefault(member.rower.kinematics_signature(),
                              []).append(member)

        built = []
        for members in groups.values():
            leader = members[0].rower
            offsets = np.array([m.rower.station.x_ankle
                                - leader.station.x_ankle for m in members])
            indices = np.array([m.seat_index for m in members])
            built.append((leader, indices, offsets))

        self._crew_group_cache = built
        return built

    def crew_field(self, t: float):
        """Stacked segment masses, positions, velocities, accelerations.

        Returns ``(mass, position, velocity, acceleration)`` with shapes
        ``(n, )`` and ``(n, 3)``, all in the hull frame, where ``n`` is 12
        per rower plus one for a coxswain if fitted.  The coxswain is a
        fixed mass, as the paper suggests.

        Rows are ordered seat-major within each kinematics group, then the
        coxswain.  Nothing downstream depends on the row order -- the mass
        matrix and the reaction sums are both permutation invariant -- but
        ``crew_field_by_seat`` is available when the caller does care.
        """
        masses, positions, velocities, accelerations = [], [], [], []
        for leader, indices, offsets in self._crew_groups():
            position, velocity, acceleration = leader.segment_state(
                t, x_offsets=offsets)
            positions.append(position)
            velocities.append(velocity)
            accelerations.append(acceleration)
            masses.append(np.tile(leader.segment_masses, len(indices)))

        if self.rig.has_coxswain and self.rig.coxswain_mass > 0:
            masses.append(np.array([self.rig.coxswain_mass]))
            positions.append(np.asarray(self.rig.coxswain_position,
                                        dtype=float).reshape(1, 3))
            velocities.append(np.zeros((1, 3)))
            accelerations.append(np.zeros((1, 3)))

        return (np.concatenate(masses), np.vstack(positions),
                np.vstack(velocities), np.vstack(accelerations))

    def hand_positions(self, t: float) -> np.ndarray:
        """Hand (oar handle) position for every seat, shape ``(n_seats, 3)``.

        Batched over kinematics groups for the same reason as
        :meth:`crew_field`: the oar moment needs a hand position per seat,
        and evaluating the chain once per seat made this the single most
        expensive part of a derivative call.

        The rower's joint chain is **sagittal** -- it has no lateral degree
        of freedom, so it puts the hand on the centreline -- and the
        batching shifts group members in ``x`` only.  Taken together those
        used to pin every rower's hands to ``y = 0``, which is wrong twice
        over: a sweep handle sweeps a wide lateral arc (``+0.19`` m at the
        catch to ``-0.28`` m through mid-drive on this rig), and port and
        starboard rowers mirror each other.

        The hands are on the handle, so the handle is where they are. The
        joint chain still sets ``x`` and ``z``, which it agrees with the
        rig geometry on exactly; only the lateral component is taken from
        the oar. For a sculler the two handles mirror and the mean is on
        the centreline, which is the right answer there too.
        """
        from ..crew.oarlock import handle_position

        positions = np.zeros((self.n_seats, 3))
        for leader, indices, offsets in self._crew_groups():
            hand = leader.joint_positions(t)["hand"]
            positions[indices] = hand
            positions[indices, 0] += offsets
        for index, seat in enumerate(self.rig.seats):
            if not seat.oarlocks:
                continue
            lateral = np.mean([
                float(handle_position(t, self.timing, lock,
                                      self.oar_sweep)[1])
                for lock in seat.oarlocks])
            positions[index, 1] = lateral
        return positions

    def crew_field_by_seat(self, t: float):
        """Per-seat segment states, as a list indexed by seat.

        Slower than :meth:`crew_field`; for plotting and inspection, where
        knowing which rower a segment belongs to matters.
        """
        return [member.rower.segment_state(t) for member in self.crew]

    def crew_centre_of_mass(self, t: float) -> np.ndarray:
        mass, position, _, _ = self.crew_field(t)
        return (mass[:, None] * position).sum(axis=0) / mass.sum()

    def equilibrium_heave(self, t: float = 0.0) -> float:
        """Static float height with the crew where they are at time ``t``."""
        return self.mesh.equilibrium_heave(self.total_mass,
                                           rho=self.water.density)

    def trim_attitude(self, t: float = 0.0, tolerance: float = 1e-6,
                      max_iterations: int = 80):
        """Solve for the heave and pitch that put the boat in static trim.

        Returns ``(heave, pitch)``.  Newton iteration on the two-equation
        residual (net vertical force, net pitch moment) using a numerical
        Jacobian -- the residual is cheap and only two-dimensional.
        """
        from ..core.frames import attitude_from_components

        gravity = 9.81
        mass, position, _, _ = self.crew_field(t)
        weight_total = self.total_mass * gravity

        def residual(unknowns):
            heave, pitch = unknowns
            attitude = attitude_from_components(pitch=pitch)
            props = self.mesh.submerged(np.array([0.0, 0.0, heave]), attitude,
                                        rho=self.water.density,
                                        gravity=gravity)
            from ..core.frames import hull_to_abs
            rot = hull_to_abs(attitude)
            crew_abs = position @ rot.T
            weight_moment = np.cross(
                crew_abs, np.tile([0.0, 0.0, -gravity], (len(mass), 1))
                * mass[:, None]).sum(axis=0)
            return np.array([
                props.buoyancy_force[2] - weight_total,
                props.buoyancy_moment[1] + weight_moment[1],
            ])

        guess = np.array([self.mesh.equilibrium_heave(
            self.total_mass, rho=self.water.density), 0.0])
        step = np.array([1e-5, 1e-6])

        for _ in range(max_iterations):
            value = residual(guess)
            if np.abs(value).max() < tolerance * max(1.0, weight_total):
                break
            jacobian = np.empty((2, 2))
            for column in range(2):
                probe = guess.copy()
                probe[column] += step[column]
                jacobian[:, column] = (residual(probe) - value) / step[column]
            try:
                guess = guess - np.linalg.solve(jacobian, value)
            except np.linalg.LinAlgError:  # pragma: no cover - degenerate
                break
        return float(guess[0]), float(guess[1])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Boat({self.name!r}, {self.n_seats} seats, "
                f"{self.total_mass:.0f} kg, {self.length:.1f} m)")
