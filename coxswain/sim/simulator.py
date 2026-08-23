"""The 6-DOF simulator.

Assembles Formaggia et al. eq. (14), generalised to six degrees of
freedom, and integrates it.  Every term is active: unlike the legacy
implementation, no moment is commented out.

Right-hand side
---------------
Force (absolute frame)::

    F = crew reaction (transport terms)
      + (r_h/L) sum F_o          oar thrust through the lever
      + buoyancy                  from the wetted-surface integral
      + M_t g                     gravity on hull and crew
      + hull resistance
      + appendage forces

Moment about ``G_h`` (absolute frame)::

    M = crew reaction moment
      + sum (x_o - x_h + (r_h/L) x_h) x F_o
      + hydrostatic moment
      + crew weight moment        sum m_ij r_ij x g
      + appendage moments
      - omega x (I omega)         gyroscopic

Left-hand side is the coupled mass matrix from
:func:`coxswain.core.rigid_body.assemble_mass_matrix`.

Frames
------
The dynamics are in the absolute frame, as the paper's are.  Hydrodynamic
forces are naturally hull-frame and are rotated once, on the way out.
The one thing this module must never do is add a hull-frame vector to an
absolute-frame one; every local variable therefore carries a ``_hull`` or
``_abs`` suffix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..boats.boat import Boat
from ..core import integrators
from ..core.frames import (abs_to_hull, cross3, euler_rates, hull_to_abs,
                           rotate_inertia_to_abs)
from ..core.rigid_body import (
    MovingMassField,
    assemble_mass_matrix,
    gyroscopic_moment,
    moving_mass_reaction,
    solve_accelerations,
)
from ..core.state import STATE_SIZE, State
from ..crew.balance import BalanceRig
from ..crew.oarlock import hull_load, oar_force
from ..hydro.appendages import surface_load
from ..hydro.resistance import hull_resistance
from .control import Coxswain
from .results import SimulationResult

__all__ = ["RowingSimulator", "ForceBreakdown", "GRAVITY"]

GRAVITY = 9.81


@dataclass
class ForceBreakdown:
    """Per-source forces and moments, for diagnosis and for the tests."""

    crew_force: np.ndarray
    crew_moment: np.ndarray
    oar_force: np.ndarray
    oar_moment: np.ndarray
    buoyancy_force: np.ndarray
    buoyancy_moment: np.ndarray
    gravity_force: np.ndarray
    gravity_moment: np.ndarray
    resistance_force: np.ndarray
    resistance_moment: np.ndarray
    appendage_force: np.ndarray
    appendage_moment: np.ndarray
    gyroscopic: np.ndarray
    resistance_detail: dict

    def total_force(self) -> np.ndarray:
        return (self.crew_force + self.oar_force + self.buoyancy_force
                + self.gravity_force + self.resistance_force
                + self.appendage_force)

    def total_moment(self) -> np.ndarray:
        return (self.crew_moment + self.oar_moment + self.buoyancy_moment
                + self.gravity_moment + self.resistance_moment
                + self.appendage_moment + self.gyroscopic)

    def generalised(self) -> np.ndarray:
        return np.concatenate([self.total_force(), self.total_moment()])


class RowingSimulator:
    """Integrates a :class:`~coxswain.boats.boat.Boat` in six degrees of
    freedom.

    Parameters
    ----------
    boat:
        The boat to simulate.
    rudder:
        Callable ``(t, state) -> deflection`` in radians, or ``None`` for a
        centred rudder.  This is the hook a steering controller -- and
        later a trajectory optimiser -- plugs into.
    water_level:
        Still-water surface height in the absolute frame.
    """

    def __init__(self, boat: Boat, coxswain: Optional[Coxswain] = None,
                 rudder: Optional[Callable[[float, State], float]] = None,
                 water_level: float = 0.0, gravity: float = GRAVITY,
                 course=None, wind=None, aero=None, blade_contact=None,
                 added_mass=True, munk_factor: float = 0.35):
        self.boat = boat
        #: Geometry linking the crew's balance effort to the hull load.
        #: Sweep rigs make this more than a roll couple; see
        #: :mod:`coxswain.crew.balance`.
        self.balance_rig = BalanceRig.from_boat(boat)
        self.coxswain = Coxswain() if coxswain is None else coxswain
        if rudder is not None:
            self.coxswain.rudder_override = rudder
        self.water_level = float(water_level)
        self.gravity = float(gravity)
        #: Hull added mass.  Entrained water, by strip theory; in sway and
        #: yaw it is comparable to the boat itself, so leaving it out makes
        #: the boat several times too easy to turn.  Pass
        #: ``added_mass=False`` to recover the old behaviour, or an
        #: :class:`~coxswain.hydro.addedmass.AddedMass` to override it.
        #: Strength of the added-mass Munk moment, 0 to 1.  Off by
        #: default and deliberately so; see
        #: :meth:`coxswain.hydro.addedmass.AddedMass.coriolis`.
        self.munk_factor = float(munk_factor)
        from ..hydro.crossflow import CrossFlowHull
        #: Station table for the distributed cross-flow integral.
        self._cross_flow = CrossFlowHull(boat.offsets)
        if added_mass is True:
            from ..hydro.addedmass import AddedMass
            self._added_mass = AddedMass.from_offsets(
                boat.offsets, rho=boat.water.density)
        elif added_mass in (False, None):
            self._added_mass = None
        else:
            self._added_mass = added_mass
        #: Optional :class:`~coxswain.river.course.Course`.  When set, the
        #: water depth and current are looked up at the boat's position
        #: every step instead of being uniform, which is what makes a river
        #: different from a buoyed course.
        self.course = course
        #: True wind field, absolute frame, or ``None`` for still air.
        self.wind = wind
        #: Aerodynamic force model.  Calibrated from the boat when a wind
        #: field is supplied, so that aero drag is the published fraction
        #: of total resistance in still air (see coxswain.hydro.wind).
        if aero is None and wind is not None:
            from ..hydro.wind import AeroModel
            aero = AeroModel.calibrate(boat)
        self.aero = aero
        #: Blades touching the water on the recovery.  ``None`` keeps the
        #: old assumption that they are always clear -- which is the intent
        #: of good rowing, but not what always happens, and the difference
        #: is what gives roll error a cost in seconds.
        self.blade_contact = blade_contact
        self._shallow_cache = (None, None)
        self._crew_cache = (None, None)
        self._hand_cache = (None, None)

    def _blade_efficiency(self, t: float, state: State, blade) -> float:
        """Instantaneous blade efficiency, from slip and water depth.

        Replaces the fixed ``blade_efficiency`` factor with the value
        implied by [CR06]'s slip model at this oar angle, oar rate and boat
        speed, scaled by the depth of water around the blade.

        Off the drive the blade is out of the water and the oarlock force
        is zero anyway, so the value there is immaterial; it is clamped to
        keep the product well behaved.
        """
        boat = self.boat
        angle = float(boat.oar_sweep(t, boat.timing))
        rate = float(boat.oar_sweep.rate(t, boat.timing))
        speed = float(state.velocity_hull[0])

        efficiency = float(blade.efficiency(angle, rate, speed))

        depth = None
        if self.course is not None:
            depth = float(self.course.depth_at(state.position[0],
                                               state.position[1]))
        elif np.isfinite(boat.shallow.depth):
            depth = float(boat.shallow.depth)

        # Blockage only -- deliberately *not* ``depth_factor``, which also
        # carries ``immersion_factor``.  The constant this replaces is a
        # measured *total* blade efficiency at nominal cover, so it already
        # absorbs the immersion loss; multiplying it back in charges for it
        # twice.  That double count is what cost 14% of boat speed in the
        # first attempt (SOURCES section 7).  A caller who wants to study
        # blade depth passes ``blade_cover`` explicitly and then owns the
        # bookkeeping.
        if depth is not None and depth > blade.blade_width:
            efficiency *= blade.blockage_factor(depth)
        cover = getattr(self.boat, "blade_cover", None)
        if cover is not None:
            efficiency *= blade.immersion_factor(cover)

        return float(np.clip(efficiency, 0.0, 1.0))

    def _shallow_at(self, position: np.ndarray):
        """Shallow-water model for the depth under the boat right now.

        Cached on depth quantised to 1 cm: the correction varies smoothly
        with depth, a shell moves a few metres per step, and rebuilding the
        model every evaluation would dominate the derivative cost for no
        change in the answer.
        """
        from ..hydro.shallow import ShallowWaterModel

        depth = float(self.course.depth_at(position[0], position[1]))
        key = round(depth, 2)
        cached_key, cached_model = self._shallow_cache
        if cached_key == key:
            return cached_model

        template = self.boat.shallow
        model = ShallowWaterModel(
            depth=key,
            max_amplification=template.max_amplification,
            subcritical_limit=template.subcritical_limit,
            supercritical_relax=template.supercritical_relax,
            gravity=template.gravity,
        )
        self._shallow_cache = (key, model)
        return model

    def crew_field(self, t: float):
        """Cached crew evaluation.

        The derivative needs the crew field twice -- once to build the
        mass matrix and once for the reaction forces -- and evaluating 97
        segment jets is the second most expensive thing per step after the
        hull integral.  Both calls are at the same ``t``, so a
        single-entry cache halves the cost.
        """
        key, value = self._crew_cache
        if key is not None and key == t:
            return value
        value = self.boat.crew_field(t)
        self._crew_cache = (t, value)
        return value

    def hand_positions(self, t: float) -> np.ndarray:
        """Cached per-seat hand positions, for the oar moment arms."""
        key, value = self._hand_cache
        if key is not None and key == t:
            return value
        value = self.boat.hand_positions(t)
        self._hand_cache = (t, value)
        return value

    # -- force assembly ---------------------------------------------------
    def breakdown(self, t: float, state: State) -> ForceBreakdown:
        """Every force and moment acting, in the absolute frame."""
        boat = self.boat
        rot = hull_to_abs(state.attitude)
        gravity_abs = np.array([0.0, 0.0, -self.gravity])

        # -- crew: prescribed motion in the hull frame -------------------
        mass, position_hull, velocity_hull, acceleration_hull = \
            self.crew_field(t)
        field_abs = MovingMassField(
            mass=mass, position=position_hull, velocity=velocity_hull,
            acceleration=acceleration_hull,
        ).to_abs(rot)
        crew_force, crew_moment = moving_mass_reaction(field_abs, state.omega)

        # -- gravity: on the whole system, moment from the crew offsets --
        gravity_force = boat.total_mass * gravity_abs
        gravity_moment = cross3(field_abs.position,
                                mass[:, None] * gravity_abs).sum(axis=0)

        # -- oars ---------------------------------------------------------
        oar_force_hull = np.zeros(3)
        oar_moment_hull = np.zeros(3)
        hands = self.hand_positions(t)
        blade = getattr(boat, "blade_model", None)
        gearing_scale = 1.0
        if blade is not None:
            # The rower pulls what they pull; how much of it moves the boat
            # depends on how much the blade slips.  Replacing the fixed
            # blade_efficiency with the instantaneous value makes that
            # depend on boat speed and on the depth of water around the
            # blade, which is the whole point of carrying a depth field.
            gearing_scale = self._blade_efficiency(t, state, blade)

        # The coxswain's second control: pressure on one side.  Applied
        # symmetrically so it is a pure yaw couple and adds no net thrust.
        split = self.coxswain.split(t, state)

        phases = boat.phase_offsets
        period = boat.timing.period
        for seat_index, seat in enumerate(boat.rig.seats):
            hand_hull = hands[seat_index]
            # A rower who is late is late in their oar as well as their
            # body.  Evaluating the oar at the boat's time while the body
            # runs on its own would decouple the hands from the handle,
            # which is the constraint the whole crew model rests on.
            seat_time = t - float(phases[seat_index]) * period
            for lock in seat.oarlocks:
                applied = oar_force(seat_time, boat.timing, lock.side,
                                    boat.force_profile, boat.oar_sweep)
                if split != 0.0:
                    applied = applied * self.coxswain.side_gain(split,
                                                                lock.side)
                # What the crew actually produce, as distinct from what the
                # coxswain asked for.  Individual differences and
                # stroke-to-stroke scatter both live here.
                applied = applied * float(boat.power_scales[seat_index])
                gearing = (lock.oar.gearing * gearing_scale
                           if blade is not None
                           else lock.oar.effective_gearing)
                force, moment = hull_load(applied, lock.position, hand_hull,
                                          gearing)
                oar_force_hull += force
                oar_moment_hull += moment

        # -- hydrostatics -------------------------------------------------
        submerged = boat.mesh.submerged(
            state.position, state.attitude, rho=boat.water.density,
            gravity=self.gravity, water_level=self.water_level,
        )

        # -- water-relative motion ----------------------------------------
        # Hydrodynamic forces depend on motion through the *water*; the
        # crew's inertial reactions and the trajectory are in the ground
        # frame.  With a current those are different vectors, and mixing
        # them is the same class of error as mixing hull and absolute
        # frames.
        shallow = boat.shallow
        velocity_hull = state.velocity_hull
        if self.course is not None:
            current_abs = self.course.current_at(state.position[0],
                                                 state.position[1])
            if np.any(current_abs):
                velocity_hull = abs_to_hull(state.attitude) @ (
                    state.velocity - current_abs)
            shallow = self._shallow_at(state.position)

        # -- resistance and appendages (hull frame) -----------------------
        resistance_hull, detail = hull_resistance(
            velocity_hull, submerged, boat.length, boat.water,
            boat.resistance, shallow,
        )
        # Distributed cross-flow drag.
        #
        # The lateral force above is computed from the uniform sideslip,
        # and the hull's yaw moment used to be set to zero outright -- all
        # yaw damping came from the skeg and rudder.  That is survivable
        # only while the hull has no Munk moment either; with the Munk
        # moment present and nothing opposing it, an eight yawed 3 degrees
        # off course reaches 285 degrees within ten strokes.
        #
        # Charging each strip the drag of its *local* lateral velocity
        # ``v + x r`` recovers both the sway force -- identically, at zero
        # yaw rate -- and the yaw damping, which is what actually holds a
        # hull straight.  See :mod:`coxswain.hydro.crossflow`.
        resistance_moment_hull = np.zeros(3)
        immersion = 1.0
        design_area = self._cross_flow.lateral_area
        if design_area > 0.0:
            immersion = float(submerged.lateral_area) / design_area
        sway_force, yaw_moment = self._cross_flow.load(
            velocity_hull[1], float(state.omega_hull[2]),
            boat.water.density, boat.resistance.cross_flow_lateral,
            immersion,
        )
        # replace the lumped lateral term rather than adding to it
        resistance_hull[1] = sway_force
        resistance_moment_hull[2] = yaw_moment

        deflection = float(self.coxswain.rudder(t, state))
        appendage_force_hull = np.zeros(3)
        appendage_moment_hull = np.zeros(3)
        yaw_rate_hull = float(state.omega_hull[2])
        for surface in boat.appendages:
            force, moment = surface_load(
                surface, velocity_hull, yaw_rate_hull, deflection,
                boat.water,
            )
            appendage_force_hull += force
            appendage_moment_hull += moment

        # -- wind ----------------------------------------------------------
        # Five-sixths of aerodynamic drag is crew and oars, not hull, so it
        # acts above the waterline and outboard: a crosswind is a roll and
        # yaw moment as well as a drag.  See coxswain.hydro.wind.
        if self.aero is not None and self.wind is not None:
            true_wind = self.wind.at(float(state.position[0]),
                                     float(state.position[1]), t)
            wind_force, wind_moment = self.aero.loads(
                true_wind, state.velocity, rot)
            appendage_force_hull = appendage_force_hull + wind_force
            appendage_moment_hull = appendage_moment_hull + wind_moment

        # -- blades touching the water, on the recovery only ---------------
        # During the drive the blade is meant to be in the water and the oar
        # model already accounts for it.  On the recovery it is a fault, and
        # an expensive one: at two degrees of heel eight blades drag more
        # than the hull does.  The same contact is a powerful stabiliser --
        # about twenty-five times the crew's own recovery authority -- which
        # is the trade a crew makes when the boat is unset.
        if self.blade_contact is not None:
            phase = boat.timing.phase(t)
            if phase >= boat.timing.drive_fraction:
                contact_drag, contact_roll = self.blade_contact.loads(
                    float(state.roll), float(state.velocity_hull[0]))
                appendage_force_hull = appendage_force_hull + np.array(
                    [contact_drag, 0.0, 0.0])
                appendage_moment_hull = appendage_moment_hull + np.array(
                    [contact_roll, 0.0, 0.0])

        # -- crew balance, applied through the riggers --------------------
        # Not a pure couple.  The crew balance by changing handle height,
        # which loads each oar as a lever and puts a vertical force at its
        # rigger.  Handle-height trim is equal and opposite across the
        # boat, so the net *force* does cancel -- but in a conventionally
        # rigged sweep boat the two sides' riggers sit a seat apart
        # longitudinally, so the same forces carry a pitch couple of about
        # 0.72 of the roll couple.  See coxswain.crew.balance.
        balance_force, balance_moment = self.balance_rig.loads(
            self.coxswain.roll_moment(state, t))
        appendage_force_hull = appendage_force_hull + np.array(balance_force)
        appendage_moment_hull = appendage_moment_hull + np.array(
            balance_moment)

        # -- gyroscopic ---------------------------------------------------
        inertia_abs = rotate_inertia_to_abs(boat.hull_inertia, state.attitude)
        gyro = gyroscopic_moment(inertia_abs, state.omega)

        # Added-mass Coriolis, the Munk moment among it.  Grouped with the
        # appendages because both are hull-frame hydrodynamic loads and
        # because they are the two things that decide whether the boat
        # holds a line: the Munk moment pushes it broadside, the skeg and
        # rudder push back.
        munk_force = np.zeros(3)
        munk_moment = np.zeros(3)
        if self._added_mass is not None:
            load = self._added_mass.coriolis(rot.T @ state.velocity,
                                             rot.T @ state.omega,
                                             munk_factor=self.munk_factor)
            munk_force, munk_moment = load[0:3], load[3:6]

        return ForceBreakdown(
            crew_force=crew_force,
            crew_moment=crew_moment,
            oar_force=rot @ oar_force_hull,
            oar_moment=rot @ oar_moment_hull,
            buoyancy_force=submerged.buoyancy_force,
            buoyancy_moment=submerged.buoyancy_moment,
            gravity_force=gravity_force,
            gravity_moment=gravity_moment,
            resistance_force=rot @ resistance_hull,
            resistance_moment=rot @ resistance_moment_hull,
            appendage_force=rot @ (appendage_force_hull + munk_force),
            appendage_moment=rot @ (appendage_moment_hull + munk_moment),
            gyroscopic=gyro,
            resistance_detail=detail,
        )

    def mass_matrix(self, t: float, state: State) -> np.ndarray:
        boat = self.boat
        rot = hull_to_abs(state.attitude)
        mass, position_hull, _, _ = self.crew_field(t)
        return assemble_mass_matrix(
            total_mass=boat.total_mass,
            inertia_abs=rotate_inertia_to_abs(boat.hull_inertia,
                                              state.attitude),
            mass=mass,
            position=position_hull @ rot.T,
            added_mass_abs=self._added_mass_abs(rot),
        )

    def _added_mass_abs(self, rot: np.ndarray) -> np.ndarray:
        """Hull added mass, rotated from the hull frame to the absolute one.

        The matrix is body-fixed -- it is a property of the hull's shape --
        so each 3x3 block transforms as ``R A R^T``.
        """
        if self._added_mass is None:
            return None
        a = self._added_mass.matrix
        out = np.zeros((6, 6))
        out[0:3, 0:3] = rot @ a[0:3, 0:3] @ rot.T
        out[3:6, 3:6] = rot @ a[3:6, 3:6] @ rot.T
        block = rot @ a[0:3, 3:6] @ rot.T
        out[0:3, 3:6] = block
        out[3:6, 0:3] = block.T
        return out

    # -- dynamics ---------------------------------------------------------
    def derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        """State derivative, in the form ``solve_ivp`` expects."""
        state = State.from_vector(y)

        matrix = self.mass_matrix(t, state)
        forces = self.breakdown(t, state)
        acceleration = solve_accelerations(matrix, forces.generalised())

        return np.concatenate([
            state.velocity,
            euler_rates(state.attitude, state.omega),
            acceleration[0:3],
            acceleration[3:6],
        ])

    # -- running ----------------------------------------------------------
    def initial_state(self, surge_speed: float = 0.0,
                      trim: bool = True) -> np.ndarray:
        """A sensible starting state: in static trim, moving at ``surge_speed``.

        Starting out of trim excites a large heave/pitch transient that
        takes several strokes to decay and contaminates any short run.
        """
        if trim:
            heave, pitch = self.boat.trim_attitude(0.0)
        else:
            heave, pitch = 0.0, 0.0
        return np.concatenate([
            [0.0, 0.0, heave],
            [0.0, pitch, 0.0],
            [surge_speed, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

    def run(self, duration: float, initial_state: np.ndarray = None,
            dt: float = None, method: str = "rk4",
            surge_speed: float = 4.5) -> SimulationResult:
        """Integrate for ``duration`` seconds.

        ``method`` is ``"rk4"`` (fixed step, reproducible) or
        ``"adaptive"`` (``solve_ivp``).
        """
        if initial_state is None:
            initial_state = self.initial_state(surge_speed=surge_speed)
        initial_state = np.asarray(initial_state, dtype=float)
        if initial_state.shape != (STATE_SIZE,):
            raise ValueError(
                f"initial state must have shape ({STATE_SIZE},), "
                f"got {initial_state.shape}"
            )

        if dt is None:
            dt = integrators.estimate_step(self.boat.timing.period)

        if method == "rk4":
            times, states = integrators.rk4(self.derivative, (0.0, duration),
                                            initial_state, dt)
        elif method == "adaptive":
            times, states = integrators.adaptive(
                self.derivative, (0.0, duration), initial_state,
                max_step=dt * 4, t_eval=np.arange(0.0, duration, dt),
            )
        else:
            raise ValueError(f"unknown method {method!r}")

        return SimulationResult(time=times, states=states, boat=self.boat)
