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
from ..core.frames import (cross3, euler_rates, hull_to_abs,
                           rotate_inertia_to_abs)
from ..core.rigid_body import (
    MovingMassField,
    assemble_mass_matrix,
    gyroscopic_moment,
    moving_mass_reaction,
    solve_accelerations,
)
from ..core.state import STATE_SIZE, State
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
                 water_level: float = 0.0, gravity: float = GRAVITY):
        self.boat = boat
        self.coxswain = Coxswain() if coxswain is None else coxswain
        if rudder is not None:
            self.coxswain.rudder_override = rudder
        self.water_level = float(water_level)
        self.gravity = float(gravity)
        self._crew_cache = (None, None)
        self._hand_cache = (None, None)

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
        for seat_index, seat in enumerate(boat.rig.seats):
            hand_hull = hands[seat_index]
            for lock in seat.oarlocks:
                applied = oar_force(t, boat.timing, lock.side,
                                    boat.force_profile, boat.oar_sweep)
                force, moment = hull_load(applied, lock.position, hand_hull,
                                          lock.oar.effective_gearing)
                oar_force_hull += force
                oar_moment_hull += moment

        # -- hydrostatics -------------------------------------------------
        submerged = boat.mesh.submerged(
            state.position, state.attitude, rho=boat.water.density,
            gravity=self.gravity, water_level=self.water_level,
        )

        # -- resistance and appendages (hull frame) -----------------------
        resistance_hull, detail = hull_resistance(
            state.velocity_hull, submerged, boat.length, boat.water,
            boat.resistance,
        )
        # resistance acts at the centre of the wetted volume; the offset
        # from G_h is small and its moment is dominated by the appendages
        resistance_moment_hull = np.zeros(3)

        deflection = float(self.coxswain.rudder(t, state))
        appendage_force_hull = np.zeros(3)
        appendage_moment_hull = np.zeros(3)
        yaw_rate_hull = float(state.omega_hull[2])
        for surface in boat.appendages:
            force, moment = surface_load(
                surface, state.velocity_hull, yaw_rate_hull, deflection,
                boat.water,
            )
            appendage_force_hull += force
            appendage_moment_hull += moment

        # -- crew balance couple (see sim.control) ------------------------
        # a pure couple about the hull x axis: handle-height trim is equal
        # and opposite across the boat, so it adds no net force
        appendage_moment_hull = appendage_moment_hull + np.array(
            [self.coxswain.roll_moment(state), 0.0, 0.0])

        # -- gyroscopic ---------------------------------------------------
        inertia_abs = rotate_inertia_to_abs(boat.hull_inertia, state.attitude)
        gyro = gyroscopic_moment(inertia_abs, state.omega)

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
            appendage_force=rot @ appendage_force_hull,
            appendage_moment=rot @ appendage_moment_hull,
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
        )

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
