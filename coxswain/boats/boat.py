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

    #: Optional tabulated wave resistance, ``N`` against speed in m/s,
    #: from :meth:`coxswain.hydro.michell.MichellWave.tabulate`.  When
    #: present it REPLACES the constant wave coefficient entirely, so the
    #: hull's wave drag comes from its own offsets rather than from a
    #: number.  Left ``None`` the old coefficient applies, which keeps
    #: every previously published figure reproducible.
    wave_table = None

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
                 blade_model=None,
                 recovery_arrival: float = 1.0,
                 uniform_traverse: float = 0.0,
                 drive_lag: float = 0.0):
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
        #: Retiming of the recovery traverse ("slow into the front").
        #: Applied identically to the crew's joint drivers and to the oar
        #: sweep, so the hands stay on the handle; see
        #: :func:`coxswain.crew.stroke.recovery_warp` and SOURCES sec. 25.
        self.recovery_arrival = float(recovery_arrival)
        if self.recovery_arrival != 1.0:
            import dataclasses as _dc
            self.oar_sweep = _dc.replace(
                self.oar_sweep, recovery_arrival=self.recovery_arrival)
        #: Experimental reparameterisation of the crew's traverse towards
        #: constant rate.  **Off by default and not recommended above
        #: ~0.5.**  It does reduce the surge fluctuation, but not honestly:
        #: Blend towards a constant-rate crew traverse, 0 to 1.
        #:
        #: The hull's speed fluctuation is set almost entirely by the peak
        #: of the crew's centre-of-mass velocity, and real crews sit close
        #: to the constant-rate floor while a four-keyframe interpolant is
        #: humped.  This was previously abandoned because it truncated the
        #: stroke; that was a normalisation bug in the warp composition,
        #: not a property of the method.  See SOURCES sec. 40.
        self.uniform_traverse = float(uniform_traverse)
        #: Retiming of the drive so the crew reaches peak speed
        #: later in it; see
        #: :func:`coxswain.crew.stroke.drive_timing_warp` and
        #: SOURCES sec. 40.  Applied to the oar sweep as well as
        #: to the joint angles, which is what keeps the hands on
        #: the handle.
        self.drive_lag = float(drive_lag)
        self.phase_warp = None
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

        if self.drive_lag != 0.0:
            from ..crew.stroke import drive_timing_warp
            import dataclasses as _dc
            knots = np.linspace(0.0, 1.0, 512, endpoint=False)
            images = drive_timing_warp(knots, self.timing.drive_fraction,
                                       self.drive_lag)
            self.phase_warp = (tuple(knots.tolist()), tuple(images.tolist()))
            self.oar_sweep = _dc.replace(self.oar_sweep,
                                         warp_knots=self.phase_warp)

        if self.uniform_traverse > 0.0:
            self.phase_warp = self._traverse_warp(
                anthropometry, default_anthropometry)
            import dataclasses as _dc
            self.oar_sweep = _dc.replace(self.oar_sweep,
                                         warp_knots=self.phase_warp)

        self.crew = self._build_crew(anthropometry, crew_phase_offsets,
                                     default_anthropometry)
        self._validate()

    def _traverse_warp(self, anthropometry, default_anthropometry):
        """Warp table making this crew traverse at near-constant rate.

        Built from a probe rower -- same dataset, timing and stature, no
        hand constraint -- whose centre-of-mass path gives the arc length
        to reparameterise by.  The probe's legs and trunk carry ~85% of
        the moving mass, and the arms follow the handle regardless, so the
        probe is an adequate stand-in for the constrained crew.
        """
        from ..crew.stroke import uniform_traverse_warp

        template = (anthropometry[0] if anthropometry
                    else default_anthropometry or RowerAnthropometry(
                        mass=85.0, stature=1.88))
        phases = np.linspace(0.0, 1.0, 512, endpoint=False)
        period = self.timing.period
        drive = self.timing.drive_fraction

        def centre_of_mass(warp):
            probe = JointDrivenRower(
                template, RowerStation(x_ankle=0.0), self.timing,
                dataset=self.stroke_dataset, phase_warp=warp)
            mass = probe.segment_masses
            return np.array([
                float((mass * probe.segment_state(t)[0][:, 0]).sum()
                      / mass.sum())
                for t in phases * period])

        # Iterate.  The warp is derived from the centre of mass but applied
        # to joint *angles*, and the centre of mass is a nonlinear function
        # of those, so one pass leaves the traverse well short of uniform --
        # measured 1.82 m/s of velocity swing against a constant-rate bound
        # of 1.33 for the same travel.  Re-deriving the warp from the
        # already-warped crew converges in a few passes.
        warp = None
        for _ in range(self._TRAVERSE_PASSES):
            com = centre_of_mass(warp)
            step = uniform_traverse_warp(phases, com, drive,
                                         blend=self.uniform_traverse)
            if warp is None:
                composed = step
            else:
                composed = np.interp(step, phases, np.asarray(warp[1]))
            # Keep the composition monotone, then **renormalise each
            # phase onto its own interval**.
            #
            # The previous version clipped to [0, 1) and took a running
            # maximum.  That enforces monotonicity but destroys the thing
            # a reparameterisation is for: a running maximum flattens any
            # decreasing stretch into a plateau, and the clip pins
            # everything past an overshoot at the endpoint, so the warp
            # stops spanning its interval.  Postures near the catch and
            # the finish then never get sampled at all, and crew
            # centre-of-mass travel fell from 0.744 m to 0.499 m as the
            # blend went to 0.9 -- which looks like a fix, because
            # shrinking the crew's motion does reduce the hull's speed
            # fluctuation, and is not one.
            #
            # A time reparameterisation traverses the *same path* at
            # different rates, so travel is preserved exactly.  That
            # requires the warp to be onto: the drive must still span
            # [0, drive] and the recovery [drive, 1].  Renormalising each
            # phase separately keeps the catch, the finish and the next
            # catch fixed, which is also what leaves the force profile's
            # clock alone.
            composed = np.maximum.accumulate(composed)
            head = phases < drive
            tail = ~head
            for mask, lo, hi in ((head, 0.0, drive), (tail, drive, 1.0)):
                block = composed[mask]
                span = block[-1] - block[0]
                if span <= 1e-12:
                    composed[mask] = np.linspace(lo, hi, mask.sum(),
                                                 endpoint=False)
                    continue
                composed[mask] = lo + (block - block[0]) * (hi - lo) / span
            composed = np.clip(composed, 0.0, 1.0 - 1e-9)
            warp = (tuple(phases.tolist()), tuple(composed.tolist()))
        return warp

    #: Fixed-point passes for the traverse warp.  Three is enough: the
    #: residual peakiness falls by about an order of magnitude a pass.
    _TRAVERSE_PASSES = 3

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
                    recovery_arrival=self.recovery_arrival,
                    phase_warp=self.phase_warp,
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
    @property
    def phase_offsets(self) -> np.ndarray:
        """Per-seat stroke-phase offset, as a fraction of one stroke.

        Zero for every seat means a perfectly synchronised crew, which is
        what every earlier version of this model assumed.  That is not a
        small idealisation: section 15 shows roll is an unstable mode held
        by a few percent of the drive's authority through the recovery,
        and port/starboard timing asymmetry is one of the main things that
        disturbs it.  Setting these to zero sets the disturbance to zero.

        Positive means *late* -- that rower reaches a given point in the
        stroke later than the reference.  Ordered by seat index.
        """
        offsets = getattr(self, "_phase_offsets", None)
        if offsets is None:
            offsets = np.zeros(self.n_seats)
            self._phase_offsets = offsets
        return offsets

    @property
    def power_scales(self) -> np.ndarray:
        """Per-seat multiplier on handle force, one entry per seat.

        Ones everywhere is a crew of identical rowers all pulling their
        nominal load.  Real crews are not that: rowers differ from each
        other, and each rower differs from stroke to stroke.

        This is separate from the coxswain's *commanded* pressure split,
        which is a control input.  This is what the crew actually does,
        including the part nobody asked for.
        """
        scales = getattr(self, "_power_scales", None)
        if scales is None:
            scales = np.ones(self.n_seats)
            self._power_scales = scales
        return scales

    @power_scales.setter
    def power_scales(self, values) -> None:
        values = np.asarray(values, dtype=float).ravel()
        if values.shape != (self.n_seats,):
            raise ValueError(
                f"power_scales must have one entry per seat "
                f"({self.n_seats}), got {values.shape}")
        if np.any(values < 0.0):
            raise ValueError("power_scales must be non-negative")
        self._power_scales = values

    @phase_offsets.setter
    def phase_offsets(self, values) -> None:
        values = np.asarray(values, dtype=float).ravel()
        if values.shape != (self.n_seats,):
            raise ValueError(
                f"phase_offsets must have one entry per seat "
                f"({self.n_seats}), got {values.shape}")
        self._phase_offsets = values
        # the grouping keys on the offsets, so it has to be rebuilt
        self._crew_group_cache = None

    def _crew_groups(self):
        """Group seats whose rowers move identically, for batch evaluation.

        Built once and reused: a homogeneous, synchronised crew collapses
        to a single group, so a derivative evaluation costs one kinematic
        chain rather than one per seat.  Each entry is
        ``(representative_rower, seat_indices, x_offsets, phase_offset)``.

        The phase offset is part of the grouping key: two rowers with
        identical anthropometry but different timing are not doing the same
        thing and cannot share an evaluation.  A crew with all-distinct
        offsets therefore costs one chain per seat, which is the honest
        price of modelling them as individuals.
        """
        if getattr(self, "_crew_group_cache", None) is not None:
            return self._crew_group_cache

        phases = self.phase_offsets
        groups = {}
        for member in self.crew:
            key = (member.rower.kinematics_signature(),
                   round(float(phases[member.seat_index]), 12))
            groups.setdefault(key, []).append(member)

        built = []
        for (_, phase), members in groups.items():
            leader = members[0].rower
            offsets = np.array([m.rower.station.x_ankle
                                - leader.station.x_ankle for m in members])
            indices = np.array([m.seat_index for m in members])
            built.append((leader, indices, offsets, float(phase)))

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
        period = self.timing.period
        for leader, indices, offsets, phase in self._crew_groups():
            position, velocity, acceleration = leader.segment_state(
                t - phase * period, x_offsets=offsets)
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
        period = self.timing.period
        for leader, indices, offsets, phase in self._crew_groups():
            hand = leader.joint_positions(t - phase * period)["hand"]
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
