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
from ..core.frames import cross3

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


#: Whether hulls carry a Michell wave table by default.  See
#: :attr:`Boat.wave_table`.
USE_MICHELL = True

#: One table per hull shape, because the integral is most of a second
#: and every boat of a class has the same offsets.  Keyed on the
#: offsets' contents rather than their identity, so two separately
#: built boats of the same class share the work.
_TABLE_CACHE = {}


class StrokeTable:
    """One rower's segment kinematics over one stroke, tabulated.

    Why this exists
    ---------------
    The derivative needs every segment's position, velocity and
    acceleration, and the hand, at the current stroke time.  Solving the
    joint chain for them costs about 2.3 ms per evaluation -- Taylor jets
    through a Fourier-driven linkage, in Python -- and RK4 at 100 Hz asks
    for it 6.6 times a frame.  That was 15 ms of every frame, on a fast
    desktop, before a triangle was drawn, and it did not change with the
    graphics setting because it is not graphics.

    The chain depends on stroke time and on nothing else: not the hull's
    state, not the water, not the wind.  It is periodic in the stroke.
    So it is solved ONCE at ``samples`` phases and interpolated, which is
    the same idea the renderer already uses for the drawn crew, applied
    where it actually costs something.

    What it does not change
    -----------------------
    The numbers.  At 480 samples a stroke is 4 ms apart at rate 30, and
    linear interpolation between exact samples is out by about
    ``h^2 |a| / 8`` in position -- a tenth of a millimetre -- and by a
    similar fraction of the acceleration's own curvature.  Both are far
    below anything the model claims.  ``tests/unit/test_stroke_table.py``
    holds it to that against the direct solve.

    The per-seat phase offsets are not baked in: they only shift the
    lookup time, so a crew whose timing drifts stroke to stroke reads the
    same table at different phases and the table stays valid.
    """

    def __init__(self, rower, period: float, samples: int = 480):
        self.period = float(period)
        self.samples = int(samples)
        taus = np.arange(self.samples) * self.period / self.samples
        position, velocity, acceleration, hand = [], [], [], []
        for tau in taus:
            # One chain solve per sample.  Asking for the hand through
            # joint_positions solved the same chain a second time and
            # doubled the build.
            p, v, a, h = rower.segment_state(tau, with_hand=True)
            position.append(p)
            velocity.append(v)
            acceleration.append(a)
            hand.append(h)
        # Positions are stored RELATIVE to this rower's own footboard,
        # so the table can be shared between rowers who move identically
        # but sit at different stations; the caller anchors it.  The
        # first shared version stored absolute x and put a whole seat's
        # spacing -- 8.5 m -- into the second group's positions.
        # Velocity and acceleration carry no origin and need nothing.
        anchor = float(rower.station.x_ankle)
        self.position = np.asarray(position, dtype=float)      # (N, 12, 3)
        self.position[:, :, 0] -= anchor
        self.velocity = np.asarray(velocity, dtype=float)
        self.acceleration = np.asarray(acceleration, dtype=float)
        self.hand = np.asarray(hand, dtype=float)              # (N, 3)
        self.hand[:, 0] -= anchor

    def _weights(self, t: float):
        u = (float(t) / self.period) % 1.0 * self.samples
        i = int(u)
        f = u - i
        return i % self.samples, (i + 1) % self.samples, f

    def at(self, t: float):
        """``(position, velocity, acceleration)``, each ``(12, 3)``."""
        i, j, f = self._weights(t)
        w = 1.0 - f
        return (w * self.position[i] + f * self.position[j],
                w * self.velocity[i] + f * self.velocity[j],
                w * self.acceleration[i] + f * self.acceleration[j])

    def hand_at(self, t: float) -> np.ndarray:
        i, j, f = self._weights(t)
        return (1.0 - f) * self.hand[i] + f * self.hand[j]


def _michell_table(offsets):
    """The hull's own wave resistance curve, built once per hull shape."""
    import numpy as np

    from ..hydro.michell import MichellWave, elliptical_offsets

    key = None
    try:
        stations = np.asarray(offsets.station, dtype=float)
        half = np.asarray(offsets.half_beam, dtype=float)
        key = (stations.tobytes(), half.tobytes(), stations.shape,
               half.shape)
    except Exception:                                    # pragma: no cover
        key = None
    if key is not None and key in _TABLE_CACHE:
        return _TABLE_CACHE[key]

    x, z, beam = elliptical_offsets(offsets, stations=641, levels=81)
    table = MichellWave(station=x, level=z, half_beam=beam).tabulate()
    if key is not None:
        _TABLE_CACHE[key] = table
    return table


class Boat:
    """Hull + rig + crew + appendages, ready to simulate."""

    #: Tabulated wave resistance, ``N`` against speed in m/s, from
    #: :meth:`coxswain.hydro.michell.MichellWave.tabulate`.  It REPLACES
    #: the constant wave coefficient entirely, so wave drag comes from
    #: the hull's own offsets rather than from a number.
    #:
    #: **On by default since the Holt comparison.**  Against forty-seven
    #: instrumented 2000 m races [H20] it is better on all four boat
    #: classes and on both independent tests: mean speed error falls from
    #: 8.7% to 4.2%, and the surge swing -- which nothing was ever tuned
    #: to -- moves toward the measured value in every class.
    #:
    #: It also settles a worry.  Michell makes the boats faster, which
    #: looked like an overshoot until the measurement showed the model
    #: had been 5 to 13 percent too SLOW.  The speed it adds is a
    #: correction, not an excess.
    #:
    #: Set :data:`coxswain.boats.boat.USE_MICHELL` to ``False`` to get
    #: the old constant coefficient back and reproduce figures published
    #: before this changed.
    _wave_table = None

    @property
    def wave_table(self):
        if not USE_MICHELL:
            return None
        if self._wave_table is None:
            self._wave_table = _michell_table(self.offsets)
        return self._wave_table

    @wave_table.setter
    def wave_table(self, value):
        self._wave_table = value

    #: Read the crew kinematics and the oar force from stroke tables
    #: rather than solving them on every derivative evaluation.  OFF by
    #: default: the studies and the golden trajectory are the exact
    #: chain, bit for bit, and a tolerance-level change to them is a
    #: model change in disguise.  The real-time trainer turns it on,
    #: because there a frame is the budget and the tables are held to
    #: the chain by ``tests/unit/test_stroke_table.py``.
    tabulate_crew: bool = False
    #: Samples per stroke in the table; see :class:`StrokeTable`.
    table_samples: int = 480

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

    def _oar_table(self):
        """The oarlock force and the sweep rate over one stroke, tabulated.

        Same reasoning as :class:`StrokeTable`.  ``oar_force`` is a
        function of stroke time, the force profile and the sweep -- a
        shape factor through a Fourier-driven angle -- and the profile
        shows it evaluated eight times per derivative, 6.6 derivatives
        a frame, through ``_ramp`` and ``magnitude`` in Python.  It is
        periodic; it is solved once.

        Only the port side is stored: the starboard force is the port
        force with ``f_y`` reversed, which is exactly how ``oar_force``
        builds it (``side * |F| sin(phi)``).  Power scales, the split
        gain and the blade's length fraction are applied by the caller
        afterwards, as before, so nothing that varies stroke to stroke
        is baked in.
        """
        from ..crew.oarlock import oar_force

        cached = self.__dict__.get("_oar_cache")
        key = (id(self.force_profile), id(self.oar_sweep),
               round(float(self.timing.period), 12),
               round(float(self.timing.drive_fraction), 12))
        if cached is not None and cached[0] == key:
            return cached[1]

        class _OarTable:
            def __init__(inner, force, rate, period, samples):
                inner.force, inner.rate = force, rate
                inner.period, inner.samples = period, samples

            def _weights(inner, t):
                u = (float(t) / inner.period) % 1.0 * inner.samples
                i = int(u)
                return i % inner.samples, (i + 1) % inner.samples, u - i

            def force_at(inner, t, side):
                i, j, f = inner._weights(t)
                out = (1.0 - f) * inner.force[i] + f * inner.force[j]
                if side < 0:
                    out = out * np.array([1.0, -1.0, 1.0])
                return out

            def rate_at(inner, t):
                i, j, f = inner._weights(t)
                return (1.0 - f) * inner.rate[i] + f * inner.rate[j]

        samples = int(self.table_samples)
        period = float(self.timing.period)
        force = np.zeros((samples, 3))
        rate = np.zeros(samples)
        for k in range(samples):
            tau = period * k / samples
            force[k] = oar_force(tau, self.timing, +1, self.force_profile,
                                 self.oar_sweep)
            rate[k] = float(self.oar_sweep.rate(tau, self.timing))
        built = _OarTable(force, rate, period, samples)
        self.__dict__["_oar_cache"] = (key, built)
        return built

    def oar_force_at(self, t: float, side: int, exact: bool = False):
        """``oar_force`` for one oarlock, from the table unless ``exact``."""
        if self.tabulate_crew and not exact:
            return self._oar_table().force_at(t, side)
        from ..crew.oarlock import oar_force
        return oar_force(t, self.timing, side, self.force_profile,
                         self.oar_sweep)

    def oar_rate_at(self, t: float, exact: bool = False) -> float:
        """``oar_sweep.rate`` at ``t``, from the table unless ``exact``."""
        if self.tabulate_crew and not exact:
            return float(self._oar_table().rate_at(t))
        return float(self.oar_sweep.rate(t, self.timing))

    def warm_crew_tables(self) -> float:
        """Build every table this crew will need, now.  Returns seconds.

        Left to first use, the build lands on the opening frames of the
        outing -- about half a second per distinct rower -- and reads as
        the game stalling on the start line.  Under the loading screen
        it is just loading.
        """
        import time

        started = time.perf_counter()
        # Every rower, not only the current group leaders.  A crew in
        # perfect time is ONE group with one leader; the moment per-seat
        # timing scatter is applied it is eight groups with eight
        # different leaders, none of which has a table -- and the first
        # version warmed the one and then rebuilt all eight on the
        # opening frames, which is precisely the stall this exists to
        # prevent.  Rowers that share a chain share a table through the
        # cache key, so this costs nothing extra for a matched crew.
        for member in self.crew:
            self._stroke_table(member.rower)
        self._lateral_table()
        self._oar_table()
        return time.perf_counter() - started

    def _stroke_table(self, rower) -> "StrokeTable":
        """This rower's table, built on first use and kept.

        Keyed on the rower and on the timing, so a boat re-rated to a
        different stroke rate rebuilds rather than reading a stale
        period.  Rowers are not mutated after construction -- the rig
        editor and the rate menu build a new boat -- so identity is a
        safe key.
        """
        tables = self.__dict__.setdefault("_stroke_tables", {})
        # Keyed on what the rower DOES and what the timing IS, not on
        # which objects they are.  The catalogue builds one rower object
        # per seat even for a matched crew, and _crew_groups already
        # decides "moves identically" by kinematics_signature -- so the
        # same signature shares one table, and a matched eight builds
        # one, not eight.  A re-rate builds a new StrokeTiming, and an
        # equal one must hit rather than rebuild.
        key = (rower.kinematics_signature(),
               round(float(self.timing.period), 12),
               round(float(self.timing.drive_fraction), 12))
        table = tables.get(key)
        if table is None:
            table = StrokeTable(rower, self.timing.period,
                                self.table_samples)
            tables[key] = table
        return table

    def _tabulated(self, leader, t: float, offsets):
        """The batched segment state, read from the table.

        The same broadcast as
        :meth:`~coxswain.crew.kinematics.JointDrivenRower._segment_state_batched`:
        one chain, shifted along ``x`` for each seat in the group.
        """
        position, velocity, acceleration = self._stroke_table(leader).at(t)
        offsets = np.asarray(offsets, dtype=float)
        n = len(offsets)
        tiled = np.tile(position, (n, 1))
        # Re-anchor at THIS leader's footboard, then shift per seat.
        tiled[:, 0] += (float(leader.station.x_ankle)
                        + np.repeat(offsets, position.shape[0]))
        return (tiled, np.tile(velocity, (n, 1)),
                np.tile(acceleration, (n, 1)))

    def crew_field(self, t: float, exact: bool = False):
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
            if self.tabulate_crew and not exact:
                position, velocity, acceleration = self._tabulated(
                    leader, t - phase * period, offsets)
            else:
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

    def hand_positions(self, t: float, exact: bool = False) -> np.ndarray:
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
        use_table = self.tabulate_crew and not exact
        for leader, indices, offsets, phase in self._crew_groups():
            if use_table:
                hand = self._stroke_table(leader).hand_at(t - phase * period)
                hand = hand + np.array([float(leader.station.x_ankle),
                                        0.0, 0.0])
            else:
                hand = leader.joint_positions(t - phase * period)["hand"]
            positions[indices] = hand
            positions[indices, 0] += offsets
        if use_table:
            positions[:, 1] = self._lateral_table().at(t)
            return positions
        for index, seat in enumerate(self.rig.seats):
            if not seat.oarlocks:
                continue
            lateral = np.mean([
                float(handle_position(t, self.timing, lock,
                                      self.oar_sweep)[1])
                for lock in seat.oarlocks])
            positions[index, 1] = lateral
        return positions

    def _lateral_table(self):
        """The handles' lateral sweep per seat over one stroke, tabulated.

        Same reasoning as :class:`StrokeTable`: ``handle_position`` is a
        function of stroke time and the rig alone, and it was being
        evaluated for every oarlock on every derivative call.
        """
        from ..crew.oarlock import handle_position

        cached = self.__dict__.get("_lateral_cache")
        key = (id(self.timing), id(self.oar_sweep), float(self.timing.period))
        if cached is not None and cached[0] == key:
            return cached[1]

        class _Lateral:
            def __init__(inner, table, period, samples):
                inner.table, inner.period, inner.samples = table, period, samples

            def at(inner, t):
                u = (float(t) / inner.period) % 1.0 * inner.samples
                i = int(u)
                f = u - i
                i %= inner.samples
                return ((1.0 - f) * inner.table[i]
                        + f * inner.table[(i + 1) % inner.samples])

        samples = int(self.table_samples)
        period = float(self.timing.period)
        table = np.zeros((samples, self.n_seats))
        for k in range(samples):
            tau = period * k / samples
            for index, seat in enumerate(self.rig.seats):
                if not seat.oarlocks:
                    continue
                table[k, index] = np.mean([
                    float(handle_position(tau, self.timing, lock,
                                          self.oar_sweep)[1])
                    for lock in seat.oarlocks])
        built = _Lateral(table, period, samples)
        self.__dict__["_lateral_cache"] = (key, built)
        return built

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
            weight_moment = cross3(
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
