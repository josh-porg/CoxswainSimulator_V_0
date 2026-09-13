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
straight running.

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
* **The crew is still prescribed in time**, so the hands follow the old
  sweep while the oar follows its own dynamics.  The crew's surge reaction is
  real and present; the hand-on-handle constraint is not enforced.  That is
  the inconsistency the forward-dynamic rower (phase 4) removes.
* **Synchronised crews only.**  Oar states are reset to the catch at each
  stroke boundary, and with per-seat phase offsets those boundaries differ by
  seat.  Refused rather than approximated.
* **Recovery holds the oar.**  Once an oar reaches the finish angle its blade
  is out: no force, and the state is held until the next catch resets it.
  The switch is not smooth; RK4 steps across it with a small local error.
* ``blade_contact`` (the lost stroke length of an unset boat) is not wired
  here, and is refused rather than silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core import integrators
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

    def __init__(self, boat, peak_torque: float, **kwargs):
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
        self._seats, self._oars = [], []
        shared = {}
        for index, seat in enumerate(boat.rig.seats):
            if not seat.oarlocks:
                continue
            rower = boat.crew[index].rower
            key = rower.kinematics_signature()
            if key not in shared:
                shared[key] = InertiaProfile.of(boat, seat=index)
            self._seats.append(index)
            self._oars.append(OarDynamics.from_boat(
                boat, inertia=shared[key], seat=index))
        self.n_oar_states = len(self._seats)
        self._oar_state = None

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

    # -- state -----------------------------------------------------------
    def augmented_initial_state(self, surge_speed: float = 4.0) -> np.ndarray:
        hull = self.initial_state(surge_speed=surge_speed)
        return np.concatenate([hull, self._catch_angles(),
                               np.zeros(self.n_oar_states)])

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
            for lock in self.boat.rig.seats[seat].oarlocks:
                side = int(lock.side)
                speed = self._lock_speed_on_normal(state, lock, angle)
                normal_force = float(oar.blade.normal_force(angle, rate,
                                                            speed))
                normal = np.array([np.cos(angle), -side * np.sin(angle), 0.0])
                axis = np.array([np.sin(angle), side * np.cos(angle), 0.0])
                load = normal_force * normal
                point = np.asarray(lock.position, dtype=float) \
                    + oar.outboard * axis
                force += load
                moment += np.cross(point, load)
        return force, moment

    # -- dynamics ------------------------------------------------------------
    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        hull, angles, rates = self._split(np.asarray(y, dtype=float))
        self._oar_state = (angles, rates)
        try:
            hull_rate = super().derivative(t, hull)
        finally:
            self._oar_state = None

        state = State.from_vector(hull)
        angle_rate = np.zeros(self.n_oar_states)
        rate_rate = np.zeros(self.n_oar_states)
        for slot, seat in enumerate(self._seats):
            oar, angle, rate = self._oars[slot], float(angles[slot]), \
                float(rates[slot])
            if angle <= oar.finish_angle:
                continue                          # held until the catch
            torque = self._torque(slot, angle, state, t)
            locks = self.boat.rig.seats[seat].oarlocks
            angle_rate[slot] = rate
            if len(locks) == 1:
                # A sweep seat: one rower, one oar -- exactly the balance the
                # unit was validated on, arithmetic unchanged.
                rate_rate[slot] = float(oar.acceleration(
                    angle, rate, -torque,
                    self._lock_speed_on_normal(state, locks[0], angle)))
            else:
                rate_rate[slot] = self._seat_acceleration(
                    oar, angle, rate, torque, state, locks)
        return np.concatenate([hull_rate, angle_rate, rate_rate])

    def _seat_acceleration(self, oar, angle, rate, torque, state,
                           locks) -> float:
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
        blade = sum(float(oar.blade_torque(
            angle, rate, self._lock_speed_on_normal(state, lock, angle)))
            for lock in locks)
        return float((-len(locks) * torque + blade
                      - 0.5 * slope * rate ** 2) / seat_inertia)

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
        weighted, total = 0.0, 0.0
        for k in range(np.asarray(times).size):
            state = State.from_vector(states[:STATE_SIZE, k])
            for slot, seat in enumerate(self._seats):
                oar = self._oars[slot]
                angle, rate = float(angles[slot, k]), float(rates[slot, k])
                if angle <= oar.finish_angle:
                    continue
                for lock in self.boat.rig.seats[seat].oarlocks:
                    speed = self._lock_speed_on_normal(state, lock, angle)
                    load = abs(float(oar.blade.normal_force(angle, rate,
                                                            speed)))
                    if load <= 0.0:
                        continue
                    weighted += load * float(oar.blade.efficiency(
                        angle, rate, speed))
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
            elif given.shape == (STATE_SIZE + 2 * n,):
                hull = given[:STATE_SIZE]
            else:
                raise ValueError(
                    "initial state must be the %d hull states or the %d "
                    "augmented ones, got shape %s"
                    % (STATE_SIZE, STATE_SIZE + 2 * n, given.shape))
        y = np.concatenate([hull, self._catch_angles(), np.zeros(n)])

        pieces_t, pieces_y = [], []
        t0, stroke = 0.0, 0
        while t0 < duration - 1e-9:
            if on_stroke is not None:
                on_stroke(stroke, self.boat)
            y = np.array(y, dtype=float)
            y[STATE_SIZE:STATE_SIZE + n] = self._catch_angles()
            y[STATE_SIZE + n:] = 0.0
            t1 = min(t0 + period, float(duration))
            times, states = integrators.rk4(self.derivative, (t0, t1), y, dt)
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

        for index in range(int(strokes)):
            y = np.array(y, dtype=float)
            y[STATE_SIZE:STATE_SIZE + n] = self._catch_angles()
            y[STATE_SIZE + n:] = 0.0
            times, states = integrators.rk4(self.derivative,
                                            (t0, t0 + period), y, dt)

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
                    torque = self._torque(slot, angle,
                                          State.from_vector(
                                              states[:STATE_SIZE, k]),
                                          float(times[k]))
                    power[k] += n_locks * abs(torque * float(rates[slot, k]))
            per_rower = float(np.trapezoid(power, times)) / period / rowers

            mean = float(speed.mean())
            records.append(StrokeRecord(
                index=index, mean_speed=mean, drive_duration=drive,
                finished=finished, handle_power=per_rower,
                surge_swing=float(np.ptp(speed)) / max(mean, 1e-9),
                blade_efficiency=self._stroke_blade_efficiency(times,
                                                               states)))
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
    torque = DynamicOarSimulator.peak_torque_for_power(boat, float(watts))
    return DynamicOarSimulator(boat, peak_torque=torque, **kwargs)
