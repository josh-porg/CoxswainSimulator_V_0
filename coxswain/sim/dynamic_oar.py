r"""The full 6-DOF hull with the oar angle as a state -- research only.

What this adds to the reduced model
-----------------------------------
:mod:`coxswain.sim.oarloop` answered phase 2's gate on two equations: hull
surge and one oar.  It said, in its own docstring, what it could not show.
**The crew's mass did not move**, so there was no intracycle surge swing --
and that swing is exactly what destroyed the efficiency-only wiring, because
the hull's minimum coincides with peak oar force.

This puts the swing back.  The hull is the full simulator: six degrees of
freedom, the prescribed crew moving on its own clock with its inertial
reactions, hydrostatics, cross-flow and radiation damping, appendages.  Only
one thing is replaced -- how the oars load the hull -- through the
:meth:`~coxswain.sim.simulator.RowingSimulator._oar_loads` seam, so every
other line of the force assembly is the one ``shipped`` runs.

The augmented state
-------------------
``[12 hull states, phi_1 .. phi_n, phi_dot_1 .. phi_dot_n]``, one angle per
seat.  A sculler's two oars share their seat's angle, exactly as the
reduced model had it, so this reduces to the validated reduced model in
straight running.  With ``crew="follows"`` one more state per seat is
appended, ``d_1 .. d_n``: how long the drive has lasted, which stops at the
finish so the recovery can be retimed from it.

The blade force on each oar is normal to the shaft and opposes the slip, and
the slip is taken against the **water-relative velocity of the oarlock**, not
the boat's surge: ``u_lock = v_hull + omega x r_lock``, projected on the
blade normal.  In straight running that is ``u cos(phi)``, identical to
:meth:`~coxswain.crew.oarlock.BladeModel.slip_velocity`; under yaw or sway
the two sides differ, which is how a turning boat's blades feel it.

For the hull-plus-crew system the blade force is the only external
horizontal force the oars supply, so it is applied **at the blade** with no
gearing factor -- the lever sets how hard the rower pulls, not how much of
the load reaches the boat.  (Applying ``gearing`` here too is the bug the
reduced model's level check caught; not repeated.)

Limits, stated before they are measured
---------------------------------------
* **By default the crew is still prescribed in time** (``crew="clock"``),
  so the hands follow the old sweep while the oar follows its own dynamics:
  up to 0.19 m off the handle on the eight at 380 W, 0.74 m on the single.
  ``crew="follows"`` is phase 4.1 -- the body slaved to the oar angle through
  the drive, hands on the handle by construction; see
  :mod:`coxswain.crew.follow`.  It is a study, and **not
  momentum-consistent**: its velocity jumps at the finish and catch are
  handed to the hull as impulses, but inside the rate floor it still hands
  the hull +148 N s a stroke on the eight (TRACKING).  Do not read speeds
  from it.  That cannot be fixed kinematically; phase 4.3 enforces the
  handle by a constraint force instead.
* **Synchronised crews only.**  Oar states are reset to the catch at each
  stroke boundary, and with per-seat phase offsets those boundaries differ by
  seat.  Refused rather than approximated.
* **Recovery holds the oar.**  Once an oar reaches the finish angle its blade
  is out: no force, and the state is held until the next catch resets it.
  The switch is not smooth; RK4 steps across it with a small local error.
  A measured oar is still sweeping at about half its peak rate at release
  and turns round in the air ([CR06] Fig. 3); moving it there needs the
  rower, which is phase 4.3.
* **The release rule** is ``release="angle"`` by default: the blade is in
  until the finish angle, and the tier 1 law brakes whenever slip turns
  non-driving before then.  ``release="slip"`` is [CR06]'s rule -- load only
  while the normal velocity is driving -- as a study.  It has no memory, so
  a blade that stops driving mid-drive can grip again; [CR06] closes the
  drive at the first return to zero.  **Under the rest catch it cannot
  run:** the oar is reset to rest at the catch and the pull shape is zero
  there, so under the default rule the drive is started by the water loading
  the parked blade (-829 N on the eight at 380 W, handle torque 0.0) -- and
  under this rule nothing starts it at all; the oar sits at the catch.
* **The catch rule** is ``catch="rest"`` by default, as just described.
  ``catch="sweep"`` is [CR06]'s entry (their section 2.5, eq. 16): the oar
  follows the prescribed sweep with its blade out, as their oar follows the
  body, until the blade's normal velocity through the water is zero, and is
  a torque-driven state from there with the sweep's angle and rate.  On the
  eight at rate 28 that is about 4.7 degrees past the catch at -1.15 rad/s.
  With ``release="slip"`` both of [CR06]'s transitions run.  A study; clock
  crew only, and not with blade added mass.
* ``blade_contact`` (the lost stroke length of an unset boat) is not wired
  here, and is refused rather than silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core import integrators
from ..core.frames import euler_rates
from ..core.rigid_body import solve_accelerations
from ..core.state import STATE_SIZE, State
from .results import SimulationResult
from .simulator import RowingSimulator

__all__ = ["DynamicOarSimulator", "StrokeRecord", "DynamicRun",
           "simulator_for"]


@dataclass(frozen=True)
class StrokeRecord:
    """What one stroke of the dynamic model did."""

    index: int
    mean_speed: float          # m/s, ground frame, over the whole stroke
    drive_duration: float      # s; the period if the drive never finished
    finished: bool
    handle_power: float        # W per rower, mean over the stroke
    surge_swing: float         # (max - min) / mean hull speed over the stroke
    #: Force-weighted blade efficiency over the drive, MEASURED on the run:
    #: see :meth:`DynamicOarSimulator._stroke_blade_efficiency`.  ``None``
    #: when no blade carried load in the stroke.
    blade_efficiency: Optional[float] = None
    #: J per rower, sweep catch only: the kinetic energy the prescribed sweep
    #: handed the oars and the rower's reflected inertia by the moment each
    #: blade entered.  The rower's work, as [CR06]'s body supplies it, and
    #: already inside ``handle_power``.  Zero under the rest catch.
    entry_work: float = 0.0


@dataclass(frozen=True)
class DynamicRun:
    strokes: List[StrokeRecord]
    period: float
    #: Time and ground speed over the LAST stroke, kept so the power a surge
    #: swing costs can be measured without integrating again.
    last_time: Optional[np.ndarray] = None
    last_speed: Optional[np.ndarray] = None
    #: The full augmented state over the LAST stroke, ``(12 + 2n, samples)``:
    #: hull states, then oar angles, then oar rates.  Kept so a figure of the
    #: blade's path through the water can be drawn from what the boat
    #: actually did, not from the prescribed sweep.
    last_states: Optional[np.ndarray] = None

    def settled_speed(self, last: int = 4) -> float:
        return float(np.mean([s.mean_speed for s in self.strokes[-last:]]))

    def settled_power(self, last: int = 4) -> float:
        return float(np.mean([s.handle_power for s in self.strokes[-last:]]))

    def settled_blade_efficiency(self, last: int = 4) -> Optional[float]:
        """Mean measured blade efficiency over the last strokes, or ``None``."""
        values = [s.blade_efficiency for s in self.strokes[-last:]
                  if s.blade_efficiency is not None]
        return float(np.mean(values)) if values else None

    def drift(self, last: int = 4) -> float:
        """Relative change in stroke speed across the last strokes."""
        tail = [s.mean_speed for s in self.strokes[-last:]]
        return abs(tail[-1] - tail[0]) / max(float(np.mean(tail)), 1e-9)

    def drag_power_ratio(self, drag) -> float:
        """Drag power averaged over the last stroke, against drag power at
        that stroke's mean speed.

        Above one because drag power is steeply nonlinear in speed: a boat
        that surges spends more on drag than one held at its mean speed.
        That excess is what a swing costs -- Hofmijster's velocity
        efficiency is its reciprocal -- and it is what separates the loss
        to the swing from the blade's own loss in ``eta(v)``.  Measured on
        the eight at rate 28 it is 1.16 at 3 m/s and 1.06 at 5.5.

        ``drag`` is a callable of speed, e.g.
        :func:`~coxswain.sim.oarloop.drag_curve`.
        """
        if self.last_time is None or self.last_speed is None:
            raise ValueError("this run did not keep its last stroke")
        span = float(self.last_time[-1] - self.last_time[0])
        mean = float(np.trapezoid(self.last_speed, self.last_time)) / span
        power = np.array([drag(float(v)) * float(v) for v in self.last_speed])
        averaged = float(np.trapezoid(power, self.last_time)) / span
        return averaged / max(drag(mean) * mean, 1e-12)


class DynamicOarSimulator(RowingSimulator):
    """:class:`RowingSimulator` with a dynamic oar angle per seat.

    ``peak_torque`` is the handle torque at the peak of the rower's
    front-loaded pull, N m about the pin, before each seat's
    ``power_scales`` entry.  It is the model's natural input; to drive at a
    stated wattage, solve for it (the reduced model's
    :func:`~coxswain.sim.oarloop.settle_at_power` gives a starting point).
    """

    #: The blade laws this simulator can run.  ``"slip"`` is tier 1 --
    #: [CR06] Model 1, a normal load from the normal slip -- and is the
    #: default, with its arithmetic untouched.  ``"liftdrag"`` is tier 2 --
    #: :class:`~coxswain.crew.liftdrag.LiftDragBlade`, lift and drag on the
    #: angle of attack, on provisional coefficients.
    BLADE_LAWS = ("slip", "liftdrag")

    #: When the blade leaves the water.  ``"angle"``: at the finish angle,
    #: as before -- and until then the tier 1 law brakes whenever the slip
    #: turns non-driving.  ``"slip"``: [CR06]'s rule, the blade carries load
    #: only while its normal velocity is driving, so it never brakes.  The oar
    #: is held at the finish angle under both.
    RELEASE_RULES = ("angle", "slip")

    #: Blade added mass.  ``"none"``: as before.  ``"patton"``: a constant
    #: added mass per blade, Patton's AR-2 plate in unbounded fluid -- an
    #: upper bound -- solved together with the hull; see
    #: :mod:`coxswain.crew.blade_added_mass`.  A study.
    ADDED_MASS_MODELS = ("none", "patton")

    #: How the crew moves.  ``"clock"``: prescribed in time, as before --
    #: the hands follow the old sweep while the oar follows its own
    #: dynamics.  ``"follows"``: phase 4.1, the body slaved to the oar angle
    #: through the drive and retimed through the recovery; see
    #: :mod:`coxswain.crew.follow`.
    CREW_MODES = ("clock", "follows")

    #: How the oar enters the drive.  ``"rest"``: reset to the catch angle
    #: at rest with the blade loaded at once -- as before, which lets the
    #: water start the drive (TRACKING).  ``"sweep"``: [CR06] section 2.5 --
    #: the oar follows the prescribed sweep with its blade out, as [CR06]'s
    #: oar follows the body, until the blade's normal velocity through the
    #: water reaches zero (their eq. 16); from that step it is a
    #: torque-driven state carrying the angle and rate the sweep gave it.
    #: A study.
    CATCH_RULES = ("rest", "sweep")

    def __init__(self, boat, peak_torque: float, blade_law: str = "slip",
                 crew: str = "clock", release: str = "angle",
                 blade_added_mass: str = "none", catch: str = "rest",
                 **kwargs):
        if blade_law not in self.BLADE_LAWS:
            raise ValueError("unknown blade law %r; this simulator runs %s"
                             % (blade_law, ", ".join(self.BLADE_LAWS)))
        if crew not in self.CREW_MODES:
            raise ValueError("unknown crew mode %r; this simulator runs %s"
                             % (crew, ", ".join(self.CREW_MODES)))
        if release not in self.RELEASE_RULES:
            raise ValueError("unknown release rule %r; this simulator runs %s"
                             % (release, ", ".join(self.RELEASE_RULES)))
        if blade_added_mass not in self.ADDED_MASS_MODELS:
            raise ValueError("unknown blade added mass %r; this simulator runs %s"
                             % (blade_added_mass,
                                ", ".join(self.ADDED_MASS_MODELS)))
        if blade_added_mass != "none" and (crew != "clock"
                                           or release != "angle"):
            raise ValueError(
                "blade added mass is solved with the clock crew and the finish-"
                "angle release only; crew=%r, release=%r" % (crew, release))
        if catch not in self.CATCH_RULES:
            raise ValueError("unknown catch rule %r; this simulator runs %s"
                             % (catch, ", ".join(self.CATCH_RULES)))
        if catch != "rest" and crew != "clock":
            raise ValueError(
                "the sweep catch is built for the clock crew; crew=%r"
                % (crew,))
        self.catch = catch
        self.blade_added_mass = blade_added_mass
        self.blade_law = blade_law
        self.crew = crew
        self.release = release
        super().__init__(boat, **kwargs)
        offsets = np.asarray(boat.phase_offsets, dtype=float)
        if offsets.size and np.ptp(offsets) > 1e-12:
            raise ValueError(
                "the dynamic oar resets every seat at one catch, so it needs "
                "a synchronised crew; this boat has per-seat phase offsets")
        if self.blade_contact is not None:
            raise ValueError(
                "blade_contact is not wired into the dynamic oar yet; "
                "refusing rather than silently dropping it")

        from ..crew.oardynamics import InertiaProfile, OarDynamics
        from .oarloop import torque_shape

        self.peak_torque = float(peak_torque)
        self._shape = torque_shape(boat)

        # One balance per seat with oars.  The reflected inertia is the
        # expensive part (a joint-chain solve per sample), and seats whose
        # rowers move identically share it.
        self._seats, self._oars, self._followers = [], [], []
        shared = {}
        for index, seat in enumerate(boat.rig.seats):
            if not seat.oarlocks:
                continue
            rower = boat.crew[index].rower
            key = rower.kinematics_signature()
            if key not in shared:
                if crew == "follows":
                    from ..crew.follow import FollowingCrew

                    # The oar's inertia must be built from the same body
                    # velocities the hull will feel, or the two halves of
                    # the model disagree about the body's kinetic energy.
                    follower = FollowingCrew(boat, seat=index)
                    shared[key] = (follower.profile(
                        float(seat.oarlocks[0].oar.inertia_about_lock)),
                        follower)
                else:
                    shared[key] = (InertiaProfile.of(boat, seat=index), None)
            self._seats.append(index)
            self._oars.append(OarDynamics.from_boat(
                boat, inertia=shared[key][0], seat=index))
            self._followers.append(shared[key][1])
        self.n_oar_states = len(self._seats)
        self._oar_state = None
        #: Sweep catch only: which oars are still on the sweep with their
        #: blade out; the same mask for every step of the stroke just run,
        #: ``(n_oars, steps + 1)``, so the post-run tallies can skip those
        #: steps; and every entry as ``(slot, t, angle, rate)``.
        self._in_air = None
        self._air_mask = None
        self.entries = []
        # Following crew only: per oar, the angle, rate and acceleration and
        # how long the drive has lasted, set for the one hull derivative
        # that needs them; and the time the current stroke's catch fell at.
        self._crew_state = None
        self._stroke_start = 0.0
        #: Following crew only: every velocity jump handed to the hull, as
        #: ``(t, sum m dv)`` -- the crew's momentum change relative to the
        #: hull, hull frame -- so the books can be checked.
        self._crew_jumps = []
        #: Oar states per seat: angle and rate, plus the elapsed drive time
        #: when the crew follows the oar.
        self._per_oar = 3 if crew == "follows" else 2
        #: Added mass per blade, kg, per seat slot; zero unless studied.
        self._blade_mass = [0.0] * self.n_oar_states
        if blade_added_mass == "patton":
            from ..crew.blade_added_mass import (BIG_BLADE_WIDTH,
                                                 patton_added_mass)

            kind = "sweep" if bool(boat.rig.is_sweep) else "scull"
            for slot, seat in enumerate(self._seats):
                oar = boat.rig.seats[seat].oarlocks[0].oar
                if float(oar.blade_length) <= 0.0:
                    raise ValueError(
                        "blade added mass needs the oar's blade_length; "
                        "this oar has none")
                self._blade_mass[slot] = patton_added_mass(
                    oar.blade_length, BIG_BLADE_WIDTH[kind],
                    boat.water.density)

        # Tier 2: one lift-drag blade per seat, sized from that seat's own
        # oar -- sweep and sculling blades differ in area, 0.110 m^2 against
        # 0.083 -- and the water it rows in.
        self._liftdrag = []
        if self.blade_law == "liftdrag":
            from ..crew.liftdrag import LiftDragBlade

            for slot, seat in enumerate(self._seats):
                lock = boat.rig.seats[seat].oarlocks[0]
                self._liftdrag.append(LiftDragBlade.big_blade(
                    outboard=float(self._oars[slot].outboard),
                    area=float(lock.oar.blade_area),
                    density=float(boat.water.density)))

    @staticmethod
    def peak_torque_for_power(boat, watts: float) -> float:
        """The peak handle torque that delivers ``watts`` per rower.

        **Closed form, not a search.**  The pull is a function of oar
        angle, so the work done over one drive is
        ``peak * integral(shape(phi) dphi)`` over the rigged arc, whatever
        the boat's speed and however long the drive takes -- the path is
        fixed even though the timing is not.  So per-rower power is that
        work, times the oars each rower drives, over the stroke period.

        Exact while the sweep is monotone through the drive, which it is:
        the handle torque always drives ``phi`` towards the finish and the
        blade resistance cannot reverse it.  ``run_strokes`` measures the
        power independently, and the test holds the two together.
        """
        from .oarloop import torque_shape

        shape = torque_shape(boat)
        catch = float(boat.oar_sweep.catch_angle)
        finish = float(boat.oar_sweep.finish_angle)
        phi = np.linspace(finish, catch, 4001)
        per_radian = float(np.trapezoid([shape(p) for p in phi], phi))
        locks = sum(len(seat.oarlocks) for seat in boat.rig.seats)
        rowers = sum(1 for seat in boat.rig.seats if seat.oarlocks)
        return (float(watts) * float(boat.timing.period) * rowers
                / (locks * per_radian))

    #: Matched torques for the sweep catch, keyed by :func:`_match_key`.
    _MATCHED = {}

    @classmethod
    def torque_for_power(cls, boat, watts: float, catch: str = "rest",
                         blade_law: str = "slip", start: float = None,
                         strokes: int = 12, iterations: int = 3) -> float:
        """The peak handle torque at which each rower does ``watts``.

        Under the rest catch this is :meth:`peak_torque_for_power`, exactly:
        the pull is a function of angle over a fixed arc, so its work does
        not depend on the run.

        Under the sweep catch it cannot be closed-form.  The rower's work
        includes the kinetic energy the sweep carries into the water
        (:meth:`_entry_energy`) -- 8.9% of handle power on the eight at rate
        28 -- and that goes as the square of the speed the blade enters at,
        which the run decides.  So the torque is found by settling the boat,
        driven straight: start from the closed form, settle, rescale by
        ``watts / measured handle power``, ``iterations`` times.  The result
        is cached for the boat's configuration, because a trajectory fit
        builds many simulators on one boat.
        """
        closed = cls.peak_torque_for_power(boat, watts)
        if catch not in cls.CATCH_RULES:
            raise ValueError("unknown catch rule %r; this simulator runs %s"
                             % (catch, ", ".join(cls.CATCH_RULES)))
        if catch == "rest":
            return closed
        key = _match_key(boat, watts, catch, blade_law)
        if key in cls._MATCHED:
            return cls._MATCHED[key]
        from .control import Coxswain

        speed = 4.5 if start is None else float(start)
        torque = closed
        for _ in range(int(iterations)):
            sim = cls(boat, peak_torque=torque, catch=catch,
                      blade_law=blade_law,
                      coxswain=Coxswain(rudder_override=lambda t, s: 0.0),
                      fast=True)
            run = sim.run_strokes(int(strokes), surge_speed=speed)
            speed = run.settled_speed()
            torque *= float(watts) / run.settled_power()
        cls._MATCHED[key] = torque
        return torque

    # -- state -----------------------------------------------------------
    def augmented_initial_state(self, surge_speed: float = 4.0) -> np.ndarray:
        hull = self.initial_state(surge_speed=surge_speed)
        return np.concatenate([hull, self._catch_angles(),
                               np.zeros((self._per_oar - 1)
                                        * self.n_oar_states)])

    def _catch_angles(self) -> np.ndarray:
        return np.array([oar.catch_angle for oar in self._oars], dtype=float)

    def _split(self, y: np.ndarray):
        n = self.n_oar_states
        return (y[:STATE_SIZE], y[STATE_SIZE:STATE_SIZE + n],
                y[STATE_SIZE + n:STATE_SIZE + 2 * n])

    # -- kinematics at the oarlock ------------------------------------------
    def _water_velocity_hull(self, state: State) -> np.ndarray:
        """Hull velocity through the water, hull frame -- as ``breakdown``."""
        from ..core.frames import abs_to_hull

        velocity = state.velocity_hull
        if self.course is not None:
            current = self.course.current_at(state.position[0],
                                             state.position[1])
            if np.any(current):
                velocity = abs_to_hull(state.attitude) @ (
                    state.velocity - current)
        return np.asarray(velocity, dtype=float)

    def _lock_speed_on_normal(self, state, lock, angle) -> float:
        """Water-relative oarlock velocity projected on the blade normal.

        Returned as the *equivalent surge* ``v`` such that
        ``v cos(phi)`` is that projection, so the blade model's own slip
        definition can be reused unchanged.  Drive angles lie within
        [-35, 55] degrees, so ``cos(phi) >= 0.57`` and the division is
        well conditioned.
        """
        side = int(lock.side)
        normal = np.array([np.cos(angle), -side * np.sin(angle), 0.0])
        velocity = self._water_velocity_hull(state) + np.cross(
            np.asarray(state.omega_hull, dtype=float),
            np.asarray(lock.position, dtype=float))
        return float(velocity @ normal) / max(float(np.cos(angle)), 1e-6)

    def _lock_velocity(self, state, lock) -> np.ndarray:
        """The oarlock's water-relative velocity, hull frame."""
        return self._water_velocity_hull(state) + np.cross(
            np.asarray(state.omega_hull, dtype=float),
            np.asarray(lock.position, dtype=float))

    def _blade_loads(self, slot, angle, rate, state, lock):
        """``(F_n, F_t)``: the blade's load normal to it and along the shaft.

        Tier 1 has no tangential load, by construction.  Tier 2 resolves the
        whole velocity of the blade through the water, so it has both.
        """
        if self.blade_law == "slip":
            speed = self._lock_speed_on_normal(state, lock, angle)
            blade = self._oars[slot].blade
            if self.release == "slip" and float(
                    blade.slip_velocity(angle, rate, speed)) >= 0.0:
                return 0.0, 0.0                   # not driving: out, [CR06]
            return (float(blade.normal_force(angle, rate, speed)), 0.0)
        velocity = self._lock_velocity(state, lock)
        blade = self._liftdrag[slot]
        if self.release == "slip":
            w_n, _w_a = blade.relative_velocity(angle, rate, velocity[:2],
                                                int(lock.side))
            if w_n >= 0.0:
                return 0.0, 0.0                   # not driving: out, [CR06]
        return blade.loads(angle, rate, velocity[:2], int(lock.side))

    def _torque(self, seat_slot: int, angle: float, state: State,
                t: float) -> float:
        """Handle torque for one seat at time ``t``.

        ``t`` is required.  It used to be absent and the coxswain's split
        was read at t = 0 -- harmless while every split was a constant,
        wrong the moment a cox changes the call mid-piece, which is what a
        steering controller does.
        """
        seat = self._seats[seat_slot]
        gain = float(np.asarray(self.boat.power_scales, dtype=float)[seat])
        split = self.coxswain.split(t, state)
        locks = self.boat.rig.seats[seat].oarlocks
        if split != 0.0:
            gain *= float(np.mean([self.coxswain.side_gain(split, int(k.side))
                                   for k in locks]))
        return self.peak_torque * gain * self._shape(angle)

    # -- the seam ------------------------------------------------------------
    def _oar_loads(self, t: float, state: State):
        """Blade forces, applied at the blades, from the current oar states."""
        force = np.zeros(3)
        moment = np.zeros(3)
        if self._oar_state is None:
            return force, moment
        angles, rates = self._oar_state
        for slot, seat in enumerate(self._seats):
            oar, angle, rate = self._oars[slot], float(angles[slot]), \
                float(rates[slot])
            if angle <= oar.finish_angle:
                continue                          # blade out: recovery
            if self._in_air is not None and self._in_air[slot]:
                continue                          # on the sweep, blade out
            for lock in self.boat.rig.seats[seat].oarlocks:
                side = int(lock.side)
                normal = np.array([np.cos(angle), -side * np.sin(angle), 0.0])
                axis = np.array([np.sin(angle), side * np.cos(angle), 0.0])
                if self.blade_law == "slip" and self.release == "angle":
                    speed = self._lock_speed_on_normal(state, lock, angle)
                    normal_force = float(oar.blade.normal_force(angle, rate,
                                                                speed))
                    load = normal_force * normal
                else:
                    f_n, f_t = self._blade_loads(slot, angle, rate, state,
                                                 lock)
                    # The tangential load acts along the shaft, so it moves
                    # the boat without turning the oar.
                    load = f_n * normal + f_t * axis
                point = np.asarray(lock.position, dtype=float) \
                    + oar.outboard * axis
                force += load
                moment += np.cross(point, load)
        return force, moment

    # -- dynamics ------------------------------------------------------------
    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        hull, angles, rates = self._split(y)
        if self.crew == "follows":
            return self._following_derivative(t, y, hull, angles, rates)
        if self.blade_added_mass != "none":
            return self._coupled_derivative(t, hull, angles, rates)
        self._oar_state = (angles, rates)
        try:
            hull_rate = super().derivative(t, hull)
        finally:
            self._oar_state = None

        angle_rate, rate_rate = self._oar_rates(t, hull, angles, rates)
        return np.concatenate([hull_rate, angle_rate, rate_rate])

    def _following_derivative(self, t, y, hull, angles, rates):
        """Phase 4.1: the oar first, then the hull, whose crew follows it.

        The other way round from the clock crew, and it has to be: the
        body's acceleration now carries ``phi_ddot``, and ``phi_ddot`` needs
        only the hull's state, never its acceleration, so there is no
        algebraic loop.  One consequence, recorded rather than hidden: the
        coxswain's split is read before the hull's own derivative instead of
        after, so a steering law that updates its call inside that
        derivative is heard one evaluation later.
        """
        n = self.n_oar_states
        elapsed = y[STATE_SIZE + 2 * n:STATE_SIZE + 3 * n]
        angle_rate, rate_rate = self._oar_rates(t, hull, angles, rates)
        self._oar_state = (angles, rates)
        self._crew_state = (angles, rates, rate_rate, elapsed, float(t))
        try:
            hull_rate = super().derivative(t, hull)
        finally:
            self._oar_state = None
            self._crew_state = None
        # The drive clock runs while the blade is in and stops at the
        # finish, so through the recovery it holds how long the drive took.
        live = np.array([float(angles[k]) > self._oars[k].finish_angle
                         for k in range(n)], dtype=float)
        return np.concatenate([hull_rate, angle_rate, rate_rate, live])

    def _oar_rates(self, t: float, hull: np.ndarray, angles, rates):
        """``(phi_dot, phi_ddot)`` for every seat, from the hull's state."""
        state = State.from_vector(hull)
        angle_rate = np.zeros(self.n_oar_states)
        rate_rate = np.zeros(self.n_oar_states)
        for slot, seat in enumerate(self._seats):
            oar, angle, rate = self._oars[slot], float(angles[slot]), \
                float(rates[slot])
            if angle <= oar.finish_angle:
                continue                          # held until the catch
            if self._in_air is not None and self._in_air[slot]:
                # Sweep catch: on the prescribed sweep until the blade enters.
                angle_rate[slot], rate_rate[slot] = self._sweep_motion(
                    t - self._stroke_start)
                continue
            torque = self._torque(slot, angle, state, t)
            locks = self.boat.rig.seats[seat].oarlocks
            angle_rate[slot] = rate
            if (len(locks) == 1 and self.blade_law == "slip"
                    and self.release == "angle"):
                # A sweep seat: one rower, one oar -- exactly the balance the
                # unit was validated on, arithmetic unchanged.
                rate_rate[slot] = float(oar.acceleration(
                    angle, rate, -torque,
                    self._lock_speed_on_normal(state, locks[0], angle)))
            else:
                rate_rate[slot] = self._seat_acceleration(
                    oar, angle, rate, torque, state, locks, slot)
        return angle_rate, rate_rate

    # -- the sweep catch ----------------------------------------------------
    def _sweep_pose(self, tau: float):
        """``(angle, rate)`` of the prescribed sweep at stroke time ``tau``."""
        sweep, timing = self.boat.oar_sweep, self.boat.timing
        return (float(sweep(tau, timing)), float(sweep.rate(tau, timing)))

    def _sweep_motion(self, tau: float):
        """``(rate, acceleration)`` of the prescribed sweep at ``tau``.

        The sweep has an analytic rate but no acceleration; a central
        difference of the rate, one-sided at the catch, is what the
        integrator needs between the steps that put the oar back on the
        sweep exactly.
        """
        sweep, timing = self.boat.oar_sweep, self.boat.timing
        step = 1e-4 * float(timing.period)
        low = max(float(tau) - step, 0.0)
        high = float(tau) + step
        rate = float(sweep.rate(tau, timing))
        accel = ((float(sweep.rate(high, timing))
                  - float(sweep.rate(low, timing))) / (high - low))
        return rate, accel

    def _entry_energy(self, slot: int, angle: float, rate: float) -> float:
        """Kinetic energy a seat's oars and body carry into the water, J.

        ``(1/2) I phi_dot^2`` with the seat's own balance inertia -- the
        body's reflected inertia and every oar the rower swings.  The torque
        drive never did that work; the sweep did, and in [CR06] the body
        does.  Measured on the eight at rate 28 it is 71 J a seat, 33 W a
        rower, 8.9% of the handle power, so leaving it out of the tally
        would compare the sweep catch against the rest catch at unequal
        power.
        """
        oar = self._oars[slot]
        moment, _slope = oar.inertia_at(angle)
        locks = len(self.boat.rig.seats[self._seats[slot]].oarlocks)
        oar_inertia = float(getattr(oar.inertia, "oar_inertia", 0.0))
        return 0.5 * (moment + (locks - 1) * oar_inertia) * rate * rate

    def _normal_velocity(self, slot: int, angle: float, rate: float,
                         state: State) -> float:
        """The blade's velocity through the water along its normal.

        Negative drives.  The same quantity each blade law's release test
        uses, so entry and release are one condition read two ways.
        """
        lock = self.boat.rig.seats[self._seats[slot]].oarlocks[0]
        if self.blade_law == "slip":
            speed = self._lock_speed_on_normal(state, lock, angle)
            return float(self._oars[slot].blade.slip_velocity(angle, rate,
                                                              speed))
        w_n, _w_a = self._liftdrag[slot].relative_velocity(
            angle, rate, self._lock_velocity(state, lock)[:2],
            int(lock.side))
        return float(w_n)

    # -- the following crew's jumps -------------------------------------------
    def _following(self, t, y, stroke_start, work):
        """Run ``work()`` with the crew field following the state ``y``."""
        n = self.n_oar_states
        saved = (self._crew_state, self._stroke_start)
        self._stroke_start = float(stroke_start)
        self._crew_state = (y[STATE_SIZE:STATE_SIZE + n],
                            y[STATE_SIZE + n:STATE_SIZE + 2 * n],
                            np.zeros(n),
                            y[STATE_SIZE + 2 * n:STATE_SIZE + 3 * n],
                            float(t))
        try:
            return work()
        finally:
            self._crew_state, self._stroke_start = saved

    def _hand_jump_to_hull(self, t, before, start_before, after,
                           start_after):
        """``after`` with the hull's velocity changed by the crew's jump.

        The crew's velocity relative to the hull goes from what ``before``
        gives to what ``after`` gives, instantaneously.  Hull plus crew
        conserves momentum and angular momentum across it:
        ``M [dv; d omega] = -[sum m dv_rel; sum m r x dv_rel]``, with ``M``
        the same mass matrix the derivative solves against.  Energy is not
        conserved -- a body stopped dead is an inelastic event, and the
        energy it loses is the rower's to absorb.
        """
        from ..core.frames import hull_to_abs

        mass, position, v0, _a = self._following(
            t, before, start_before, lambda: self.crew_field(t))
        state = State.from_vector(after[:STATE_SIZE])
        _m, _p, v1, _a = self._following(
            t, after, start_after, lambda: self.crew_field(t))
        matrix = self._following(
            t, after, start_after, lambda: self.mass_matrix(t, state))

        jump_hull = v1 - v0
        rot = hull_to_abs(state.attitude)
        jump = jump_hull @ rot.T
        arm = position @ rot.T
        impulse = -(mass[:, None] * jump).sum(axis=0)
        moment = -(mass[:, None] * np.cross(arm, jump)).sum(axis=0)
        change = np.linalg.solve(matrix, np.concatenate([impulse, moment]))

        out = np.array(after, dtype=float)
        out[6:9] += change[0:3]
        out[9:12] += change[3:6]
        self._crew_jumps.append(
            (float(t), (mass[:, None] * jump_hull).sum(axis=0)))
        return out

    def _integrate_stroke(self, t_span, y0, dt):
        """One stroke, ``(times, states)`` as :func:`integrators.rk4` gives.

        The clock crew is exactly that call.  The following crew is stepped
        by hand so a finish can be caught on the step it happens: the body
        is taken as still driving at the finish angle with the rate the oar
        arrived at, against the retimed recovery the state now selects, and
        the difference is handed to the hull.
        """
        if self.catch == "sweep":
            return self._integrate_sweep_catch(t_span, y0, dt)
        self._air_mask = None
        if self.crew != "follows":
            return integrators.rk4(self.derivative, t_span, y0, dt)
        t_start, t_end = float(t_span[0]), float(t_span[1])
        n_steps = int(np.ceil((t_end - t_start) / dt))
        n = self.n_oar_states
        finishes = np.array([oar.finish_angle for oar in self._oars])
        times = np.empty(n_steps + 1)
        states = np.empty((len(y0), n_steps + 1))
        t, y = t_start, np.array(y0, dtype=float)
        times[0], states[:, 0] = t, y
        start = self._stroke_start
        for i in range(n_steps):
            step = min(dt, t_end - t)
            live = y[STATE_SIZE:STATE_SIZE + n] > finishes
            y = integrators.rk4_step(self.derivative, t, y, step)
            t += step
            crossed = live & (y[STATE_SIZE:STATE_SIZE + n] <= finishes)
            if crossed.any():
                driving = np.array(y, dtype=float)
                driving[STATE_SIZE:STATE_SIZE + n][crossed] = \
                    finishes[crossed] + 1e-9
                y = self._hand_jump_to_hull(t, driving, start, y, start)
            times[i + 1], states[:, i + 1] = t, y
        return times, states

    def _integrate_sweep_catch(self, t_span, y0, dt):
        """One stroke under the sweep catch, stepped by hand.

        The same grid :func:`~coxswain.core.integrators.rk4` builds.  After
        every step an oar still in the air is put back exactly on the
        prescribed sweep -- the integrator only carries it between steps --
        and it is handed to the torque drive on the step its blade's normal
        velocity through the water reaches zero, [CR06] eq. 16.  The oar
        then carries the sweep's angle and rate at that instant.
        """
        t_start, t_end = float(t_span[0]), float(t_span[1])
        n_steps = int(np.ceil((t_end - t_start) / dt))
        n = self.n_oar_states
        times = np.empty(n_steps + 1)
        states = np.empty((len(y0), n_steps + 1))
        air = np.zeros((n, n_steps + 1), dtype=bool)
        t, y = t_start, np.array(y0, dtype=float)
        if self._in_air is None:
            self._in_air = np.zeros(n, dtype=bool)
        times[0], states[:, 0], air[:, 0] = t, y, self._in_air
        for i in range(n_steps):
            step = min(dt, t_end - t)
            y = integrators.rk4_step(self.derivative, t, y, step)
            t += step
            if self._in_air.any():
                angle, rate = self._sweep_pose(t - self._stroke_start)
                state = State.from_vector(y[:STATE_SIZE])
                for slot in np.flatnonzero(self._in_air):
                    y[STATE_SIZE + slot] = angle
                    y[STATE_SIZE + n + slot] = rate
                    if self._normal_velocity(slot, angle, rate, state) <= 0.0:
                        self._in_air[slot] = False
                        self.entries.append((int(slot), float(t),
                                             float(angle), float(rate)))
            times[i + 1], states[:, i + 1] = t, y
            air[:, i + 1] = self._in_air
        self._air_mask = air
        return times, states

    def _catch(self, t0, y, index):
        """Reset every oar to the catch; hand the crew's jump to the hull."""
        n = self.n_oar_states
        before = np.array(y, dtype=float)
        after = np.array(y, dtype=float)
        after[STATE_SIZE:STATE_SIZE + n] = self._catch_angles()
        after[STATE_SIZE + n:] = 0.0
        if index == 0:
            self.entries = []
        # The sweep is at the catch angle with zero rate at tau = 0, so the
        # reset state is already on it.
        self._in_air = (np.ones(n, dtype=bool) if self.catch == "sweep"
                        else None)
        if self.crew == "follows" and index > 0:
            after = self._hand_jump_to_hull(t0, before, self._stroke_start,
                                            after, t0)
        return after

    # -- the crew ------------------------------------------------------------
    def crew_field(self, t: float, exact: bool = False):
        """The crew's masses and motion -- following the oar when asked.

        The clock crew, and any call made outside a derivative (a
        measurement, a plot), get the base class's prescribed field.
        """
        if self.crew != "follows" or self._crew_state is None or exact:
            return super().crew_field(t, exact=exact)
        angles, rates, accelerations, elapsed, now = self._crew_state
        boat = self.boat
        since_catch = now - self._stroke_start
        period = float(boat.timing.period)
        phases = np.asarray(boat.phase_offsets, dtype=float)
        slot_of = {seat: slot for slot, seat in enumerate(self._seats)}

        masses, positions, velocities, accels = [], [], [], []
        for member in boat.crew:
            rower = member.rower
            slot = slot_of.get(member.seat_index)
            if slot is None:
                # a rower with no oar has nothing to follow
                position, velocity, accel = boat._stroke_table(rower).at(
                    now - float(phases[member.seat_index]) * period)
                position = np.array(position, dtype=float)
            else:
                follower = self._followers[slot]
                angle = float(angles[slot])
                if angle > self._oars[slot].finish_angle:
                    position, velocity, accel = follower.drive_state(
                        angle, float(rates[slot]), float(accelerations[slot]))
                else:
                    position, velocity, accel = follower.recovery_state(
                        since_catch, float(elapsed[slot]))
            position[:, 0] += float(rower.station.x_ankle)
            masses.append(np.asarray(rower.segment_masses, dtype=float))
            positions.append(position)
            velocities.append(velocity)
            accels.append(accel)

        rig = boat.rig
        if rig.has_coxswain and rig.coxswain_mass > 0:
            masses.append(np.array([rig.coxswain_mass]))
            positions.append(np.asarray(rig.coxswain_position,
                                        dtype=float).reshape(1, 3))
            velocities.append(np.zeros((1, 3)))
            accels.append(np.zeros((1, 3)))

        return (np.concatenate(masses), np.vstack(positions),
                np.vstack(velocities), np.vstack(accels))

    def _seat_acceleration(self, oar, angle, rate, torque, state,
                           locks, slot: Optional[int] = None) -> float:
        inertia, rhs = self._seat_balance(oar, angle, rate, torque, state,
                                          locks, slot)
        return float(rhs / inertia)

    def _seat_balance(self, oar, angle, rate, torque, state,
                      locks, slot: Optional[int] = None):
        """``phi_ddot`` for a seat whose one rower swings several oars.

        One body, one balance::

            (I_crew + n I_oar) phi_ddot + (1/2)(dI/dphi) phi_dot^2
                = -n tau + sum_i l F_n,i

        It used to be the MEAN of each oar's own balance, and each of those
        carried the rower's whole reflected inertia -- so a sculler's body
        was counted once per oar.  Measured on a single over one drive, the
        per-oar form left 2.2% of the handle work unaccounted for, 16.5 J of
        760; this closes the books to 0.2 J.  It shortens the drive by about
        8%, not by half: blade resistance goes as slip squared, so a faster
        sweep brakes itself and the drive length is set by the blade, not
        the inertia.
        """
        moment, slope = oar.inertia_at(angle)
        oar_inertia = float(getattr(oar.inertia, "oar_inertia", 0.0))
        seat_inertia = moment + (len(locks) - 1) * oar_inertia
        if self.blade_law == "slip" and self.release == "angle":
            blade = sum(float(oar.blade_torque(
                angle, rate, self._lock_speed_on_normal(state, lock, angle)))
                for lock in locks)
        else:
            if slot is None:
                slot = self._oars.index(oar)
            # Only the normal load turns the oar; the tangential load acts
            # along the shaft, through the pin.
            blade = sum(oar.outboard * self._blade_loads(
                slot, angle, rate, state, lock)[0] for lock in locks)
        return seat_inertia, (-len(locks) * torque + blade
                              - 0.5 * slope * rate ** 2)

    # -- blade added mass: hull and oars solved together -----------------------
    def _coupled_system(self, t: float, state: State, angles, rates):
        """``(A, b)`` for ``A [G_ddot; omega_dot; phi_ddot] = b``, absolute frame.

        Per blade in the water, with normal ``n``, centre ``r`` and added mass
        ``m``, the blade's normal acceleration is ``g . X_h + l phi_ddot + c``
        with ``g = [n; r x n]``, and its added-mass force is
        ``-m (w_n_dot n + w_n n_dot)`` -- the rate of change of the entrained
        momentum ``m w_n n``.  Moving the acceleration terms to the left::

            (M + m g g^T) X_h + m l g phi_ddot          = f - m (c g + w_n h)
            m l g^T X_h + (I_seat + m l^2) phi_ddot     = rhs - m l c

        ``h = [n_dot; r x n_dot]``.  The matrix is the old diagonal plus a sum
        of ``m v v^T``, so it stays symmetric positive definite.  A held oar
        (past its finish) keeps ``phi_ddot = 0`` through an identity row.
        """
        n = self.n_oar_states
        self._oar_state = (angles, rates)
        try:
            matrix = self.mass_matrix(t, state)
            forces = self.breakdown(t, state)
        finally:
            self._oar_state = None

        system = np.zeros((6 + n, 6 + n))
        rhs = np.zeros(6 + n)
        system[:6, :6] = matrix
        rhs[:6] = forces.generalised()
        rot = state.rot_hull_to_abs
        omega_abs = np.asarray(state.omega, dtype=float)
        omega_hull = np.asarray(state.omega_hull, dtype=float)

        for slot, seat in enumerate(self._seats):
            row = 6 + slot
            oar, angle, rate = self._oars[slot], float(angles[slot]), \
                float(rates[slot])
            if angle <= oar.finish_angle:
                system[row, row] = 1.0            # held until the catch
                continue
            if self._in_air is not None and self._in_air[slot]:
                # Sweep catch, blade out: the oar follows the prescribed
                # sweep and sets no water moving, so no balance and no added
                # mass -- only the sweep's own acceleration.
                system[row, row] = 1.0
                rhs[row] = self._sweep_motion(t - self._stroke_start)[1]
                continue
            torque = self._torque(slot, angle, state, t)
            locks = self.boat.rig.seats[seat].oarlocks
            inertia, balance = self._seat_balance(oar, angle, rate, torque,
                                                  state, locks, slot)
            system[row, row] += inertia
            rhs[row] += balance

            mass, arm = float(self._blade_mass[slot]), float(oar.outboard)
            for lock in locks:
                side = int(lock.side)
                normal = np.array([np.cos(angle), -side * np.sin(angle), 0.0])
                axis = np.array([np.sin(angle), side * np.cos(angle), 0.0])
                point = np.asarray(lock.position, dtype=float) + arm * axis
                normal_abs, point_abs = rot @ normal, rot @ point
                g = np.concatenate([normal_abs, np.cross(point_abs, normal_abs)])

                water = self._lock_velocity(state, lock) + arm * rate * normal
                w_n = float(water @ normal)
                normal_dot = -rate * axis + np.cross(omega_hull, normal)
                c = (float(normal_abs @ np.cross(omega_abs, np.cross(
                    omega_abs, point_abs))) + float(water @ normal_dot))
                normal_dot_abs = rot @ normal_dot
                h = np.concatenate([normal_dot_abs,
                                    np.cross(point_abs, normal_dot_abs)])

                system[:6, :6] += mass * np.outer(g, g)
                system[:6, row] += mass * arm * g
                system[row, :6] += mass * arm * g
                system[row, row] += mass * arm * arm
                rhs[:6] -= mass * (c * g + w_n * h)
                rhs[row] -= mass * arm * c
        return system, rhs

    def _coupled_derivative(self, t: float, hull, angles, rates):
        state = State.from_vector(hull)
        system, rhs = self._coupled_system(t, state, angles, rates)
        accel = solve_accelerations(system, rhs)
        n = self.n_oar_states
        angle_rate = np.array([float(rates[k]) if float(angles[k])
                               > self._oars[k].finish_angle else 0.0
                               for k in range(n)])
        return np.concatenate([state.velocity,
                               euler_rates(state.attitude, state.omega),
                               accel[0:3], accel[3:6], angle_rate,
                               accel[6:6 + n]])

    # -- measurement ---------------------------------------------------------
    def _stroke_blade_efficiency(self, times, states) -> Optional[float]:
        """Force-weighted blade efficiency over one stroke, from the run.

        The same quantity the scorecard's level target has always scored
        -- ``1 - |slip| / |blade speed|``, weighted by blade load, from
        :meth:`~coxswain.crew.oarlock.BladeModel.efficiency` -- but fed the
        states the boat actually had, not a schedule.

        The prescribed version evaluates a PRESCRIBED sweep at ONE speed and
        weights by the prescribed force curve.  Every one of those three is
        replaced here: the oar angle and rate are the integrated states, the
        water speed is the oarlock's instantaneous water-relative speed on
        the blade normal -- so the crew's surge swing is inside it -- and
        the weight is the blade force the water actually put on the blade.
        That is what makes it a measurement of this physics and not a
        restatement of the old one.

        Uniform weights in time, which is what a fixed-step run gives.  An
        oar past its finish angle has its blade out and is excluded, as it
        is from the loads.
        """
        n = self.n_oar_states
        angles = states[STATE_SIZE:STATE_SIZE + n]
        rates = states[STATE_SIZE + n:STATE_SIZE + 2 * n]
        air = self._air_mask
        weighted, total = 0.0, 0.0
        for k in range(np.asarray(times).size):
            state = State.from_vector(states[:STATE_SIZE, k])
            for slot, seat in enumerate(self._seats):
                oar = self._oars[slot]
                angle, rate = float(angles[slot, k]), float(rates[slot, k])
                if angle <= oar.finish_angle:
                    continue
                if air is not None and air[slot, k]:
                    continue                      # on the sweep, blade out
                for lock in self.boat.rig.seats[seat].oarlocks:
                    if self.blade_law == "slip":
                        speed = self._lock_speed_on_normal(state, lock, angle)
                        load = abs(float(oar.blade.normal_force(angle, rate,
                                                                speed)))
                        if self.release == "slip" and float(
                                oar.blade.slip_velocity(angle, rate,
                                                        speed)) >= 0.0:
                            load = 0.0            # released: carries nothing
                        if load <= 0.0:
                            continue
                        efficiency = float(oar.blade.efficiency(
                            angle, rate, speed))
                    else:
                        # Same definition as tier 1, so the two tiers score
                        # against one band: 1 - |normal slip|/|blade speed|,
                        # weighted by the normal load.
                        blade = self._liftdrag[slot]
                        velocity = self._lock_velocity(state, lock)[:2]
                        f_n, _f_t = self._blade_loads(slot, angle, rate,
                                                      state, lock)
                        load = abs(float(f_n))
                        if load <= 0.0:
                            continue
                        w_n, _w_a = blade.relative_velocity(
                            angle, rate, velocity, int(lock.side))
                        sweep = abs(oar.outboard * rate)
                        efficiency = (float(np.clip(1.0 - abs(w_n) / sweep,
                                                    0.0, 1.0))
                                      if sweep > 1e-9 else 0.0)
                    weighted += load * efficiency
                    total += load
        return weighted / total if total > 0.0 else None

    # -- running ---------------------------------------------------------------
    def run(self, duration: float, initial_state: np.ndarray = None,
            dt: float = None, method: str = "rk4",
            surge_speed: float = 4.5, on_stroke=None) -> SimulationResult:
        """Integrate for ``duration`` seconds, as the base class does.

        The contract every consumer already uses -- ``steer``,
        ``fit_reduced_model``, the report's settles -- so none of them has
        to know the oar is a state.  ``initial_state`` may be the twelve
        hull states (a boat placed on a path and pointed down it) or the
        full augmented state; the oars are reset to the catch at the start
        of every stroke either way.  Returns an ordinary
        :class:`~coxswain.sim.results.SimulationResult` of the hull states.

        Fixed-step only: the oar resets at every catch and the blade
        switches out at the finish angle, and an adaptive step would step
        across both without knowing.
        """
        if method != "rk4":
            raise ValueError(
                "the dynamic oar needs the fixed-step integrator: it resets "
                "at every catch and switches at the finish angle; got %r"
                % (method,))
        period = float(self.boat.timing.period)
        if dt is None:
            dt = integrators.estimate_step(period)
        n = self.n_oar_states

        if initial_state is None:
            hull = self.initial_state(surge_speed=surge_speed)
        else:
            given = np.asarray(initial_state, dtype=float)
            if given.shape == (STATE_SIZE,):
                hull = given
            elif given.shape == (STATE_SIZE + self._per_oar * n,):
                hull = given[:STATE_SIZE]
            else:
                raise ValueError(
                    "initial state must be the %d hull states or the %d "
                    "augmented ones, got shape %s"
                    % (STATE_SIZE, STATE_SIZE + self._per_oar * n,
                       given.shape))
        y = np.concatenate([hull, self._catch_angles(),
                            np.zeros((self._per_oar - 1) * n)])

        pieces_t, pieces_y = [], []
        t0, stroke = 0.0, 0
        self._crew_jumps = []
        while t0 < duration - 1e-9:
            if on_stroke is not None:
                on_stroke(stroke, self.boat)
            y = self._catch(t0, y, stroke)
            self._stroke_start = t0
            t1 = min(t0 + period, float(duration))
            times, states = self._integrate_stroke((t0, t1), y, dt)
            keep = slice(0, -1) if t1 < duration else slice(None)
            pieces_t.append(times[keep])
            pieces_y.append(states[:STATE_SIZE, keep])
            t0, y = float(times[-1]), states[:, -1]
            stroke += 1

        return SimulationResult(time=np.concatenate(pieces_t),
                                states=np.concatenate(pieces_y, axis=1),
                                boat=self.boat)

    def run_strokes(self, strokes: int, surge_speed: float = 4.0,
                    dt: Optional[float] = None) -> DynamicRun:
        """Integrate stroke by stroke, resetting every oar at each catch."""
        period = float(self.boat.timing.period)
        if dt is None:
            dt = integrators.estimate_step(period)
        n = self.n_oar_states
        y = self.augmented_initial_state(surge_speed)
        t0 = 0.0
        records = []
        rowers = max(len(self._seats), 1)

        self._crew_jumps = []
        for index in range(int(strokes)):
            y = self._catch(t0, y, index)
            self._stroke_start = t0
            times, states = self._integrate_stroke((t0, t0 + period), y, dt)

            speed = np.hypot(states[6], states[7])
            angles = states[STATE_SIZE:STATE_SIZE + n]
            rates = states[STATE_SIZE + n:STATE_SIZE + 2 * n]

            finishes = np.array([oar.finish_angle for oar in self._oars])
            done = np.all(angles <= finishes[:, None] + 1e-12, axis=0)
            finished = bool(done.any())
            drive = float(times[int(np.argmax(done))] - t0) if finished \
                else period

            power = np.zeros_like(times)
            for slot, seat in enumerate(self._seats):
                n_locks = len(self.boat.rig.seats[seat].oarlocks)
                for k in range(times.size):
                    angle = float(angles[slot, k])
                    if angle <= finishes[slot]:
                        continue
                    if self._air_mask is not None and self._air_mask[slot, k]:
                        continue                  # the sweep, not the rower
                    torque = self._torque(slot, angle,
                                          State.from_vector(
                                              states[:STATE_SIZE, k]),
                                          float(times[k]))
                    power[k] += n_locks * abs(torque * float(rates[slot, k]))
            per_rower = float(np.trapezoid(power, times)) / period / rowers
            # Sweep catch: the energy the sweep carried into the water is
            # the rower's work too, or the comparison is at unequal power.
            entry_work = sum(
                self._entry_energy(slot, angle, rate)
                for slot, when, angle, rate in self.entries
                if t0 - 1e-9 <= when < t0 + period - 1e-9) / rowers
            per_rower += entry_work / period

            mean = float(speed.mean())
            records.append(StrokeRecord(
                index=index, mean_speed=mean, drive_duration=drive,
                finished=finished, handle_power=per_rower,
                surge_swing=float(np.ptp(speed)) / max(mean, 1e-9),
                blade_efficiency=self._stroke_blade_efficiency(times,
                                                               states),
                entry_work=entry_work))
            t0 = float(times[-1])
            y = states[:, -1]

        return DynamicRun(strokes=records, period=period,
                          last_time=np.asarray(times, dtype=float).copy(),
                          last_speed=np.asarray(speed, dtype=float).copy(),
                          last_states=np.asarray(states, dtype=float).copy())


def simulator_for(boat, **kwargs):
    """The simulator a boat's physics needs.

    A boat stamped with a profile whose oar angle is a dynamic state gets a
    :class:`DynamicOarSimulator`; anything else gets the ordinary
    :class:`~coxswain.sim.simulator.RowingSimulator`, unchanged.  ``kwargs``
    go to the constructor either way.

    A dynamic-oar boat must say how hard its crew pulls, as
    ``boat.handle_watts`` -- handle power per rower, the analogue of
    ``power_scales`` -- and is refused without it.  There is deliberately no
    default: a default wattage would be a number nobody chose, and every
    speed downstream of it would inherit that.
    """
    from .. import physics

    stamp = getattr(boat, "physics_profile", None)
    if stamp is None or not physics.resolve(stamp).uses_dynamic_oar:
        return RowingSimulator(boat, **kwargs)
    watts = getattr(boat, "handle_watts", None)
    if watts is None:
        raise ValueError(
            "boat is stamped %r, whose oar angle is a dynamic state, and "
            "states no crew power: set boat.handle_watts (W per rower) "
            "first -- there is no default" % (stamp,))
    # The profile names the catch; an explicit argument overrides it.
    catch = kwargs.pop("catch", physics.resolve(stamp).catch)
    torque = DynamicOarSimulator.torque_for_power(
        boat, float(watts), catch=catch,
        blade_law=kwargs.get("blade_law", "slip"))
    return DynamicOarSimulator(boat, peak_torque=torque, catch=catch,
                               **kwargs)


def _match_key(boat, watts, catch, blade_law) -> tuple:
    """Everything a matched sweep-catch torque depends on, as a cache key.

    Two boats share a torque only if they would row the same stroke: same
    hull shape, hull and crew mass, timing, per-seat power, rig geometry,
    physics stamp, wave model and water depth.  The hull shape was missing
    until 2026-09-13, so two boats differing only in their offsets -- a hull
    study -- shared one matched torque.
    """
    offsets = getattr(boat, "offsets", None)
    hull = ()
    if offsets is not None:
        hull = tuple(np.asarray(getattr(offsets, field), dtype=float).tobytes()
                     for field in ("station", "beam", "depth")
                     if getattr(offsets, field, None) is not None)
    rig = []
    for seat in boat.rig.seats:
        for lock in seat.oarlocks:
            oar = lock.oar
            rig.append((int(lock.side),
                        tuple(np.round(np.asarray(lock.position, float), 9)),
                        float(oar.inboard), float(oar.outboard),
                        float(getattr(oar, "blade_length", 0.0))))
    shallow = getattr(boat, "shallow", None)
    return (str(boat.name), hull, float(boat.timing.period),
            float(boat.timing.drive_fraction), round(float(boat.total_mass), 9),
            tuple(np.round(np.asarray(boat.power_scales, float), 12)),
            tuple(rig), getattr(boat, "physics_profile", None),
            type(getattr(boat, "wave_table", None)).__name__,
            float(getattr(shallow, "depth", float("inf"))),
            float(watts), str(catch), str(blade_law))
