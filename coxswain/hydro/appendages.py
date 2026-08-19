"""Skeg and rudder: the surfaces that give a shell directional stability.

The paper's model is planar and has neither.  They matter here for two
reasons: without a skeg the yaw mode is neutrally stable and any
disturbance integrates away, and the whole point of a coxswain-facing
simulator on a winding river is steering.

Both are treated as low-aspect-ratio lifting surfaces.  Lift-curve slope
comes from the Polhamus/Helmbold formula, which is the standard
correction for surfaces of aspect ratio below about 2 (a shell's fin is
around 0.6, far outside the range where thin-aerofoil ``2 pi`` applies)::

    C_L_alpha = 2 pi AR / (2 + sqrt(AR^2 (1 + tan^2 Lambda) + 4))

Induced drag follows the usual ``C_L^2 / (pi AR e)``.

Local flow angle
----------------
A surface at ``x_ac`` ahead of (positive) or behind (negative) the centre
of mass sees a sideslip modified by the yaw rate,

    beta_local = beta + r x_ac / V

which is what supplies the yaw damping: a fin *aft* of the centre of mass
opposes yaw rate.  This carries over directly from aircraft
directional-stability practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resistance import FRESH_WATER, WaterProperties

__all__ = ["LiftingSurface", "SKEG_EIGHT", "RUDDER_EIGHT", "surface_load"]


@dataclass(frozen=True)
class LiftingSurface:
    """A fin, skeg or rudder.

    Attributes
    ----------
    span, chord:
        Geometric span (depth below the hull) and mean chord, in metres.
    position:
        ``(3,)`` hull-frame position of the aerodynamic centre, relative to
        ``G_h``.  ``x`` negative places it aft, which is where a skeg goes.
    sweep:
        Leading-edge sweep angle in radians.
    oswald:
        Oswald efficiency factor for the induced-drag term.
    controllable:
        ``True`` for a rudder, whose deflection is a control input.
    max_deflection:
        Deflection limit in radians, applied to controllable surfaces.
    """

    span: float
    chord: float
    position: np.ndarray
    sweep: float = 0.0
    oswald: float = 0.8
    controllable: bool = False
    max_deflection: float = np.radians(25.0)
    flap_effectiveness: float = 1.0

    def __post_init__(self) -> None:
        if self.span <= 0 or self.chord <= 0:
            raise ValueError("span and chord must be positive")
        if np.asarray(self.position).shape != (3,):
            raise ValueError("position must be a length-3 vector")

    @property
    def area(self) -> float:
        return self.span * self.chord

    @property
    def aspect_ratio(self) -> float:
        return self.span ** 2 / self.area

    @property
    def lift_curve_slope(self) -> float:
        """``C_L_alpha`` per radian, from the Polhamus/Helmbold formula."""
        aspect = self.aspect_ratio
        tan_sweep = np.tan(self.sweep)
        return (2.0 * np.pi * aspect
                / (2.0 + np.sqrt(aspect ** 2 * (1.0 + tan_sweep ** 2) + 4.0)))


#: Fin fitted to a typical eight: 129 mm deep, 202 mm chord, well aft.
SKEG_EIGHT = LiftingSurface(
    span=0.129, chord=0.202, position=np.array([-6.0, 0.0, -0.18]),
    sweep=np.radians(5.0),
)

#: Coxswain's rudder, mounted just aft of the skeg.
RUDDER_EIGHT = LiftingSurface(
    span=0.090, chord=0.120, position=np.array([-6.6, 0.0, -0.16]),
    controllable=True, flap_effectiveness=1.0,
)


def surface_load(surface: LiftingSurface, velocity_hull: np.ndarray,
                 yaw_rate: float, deflection: float = 0.0,
                 water: WaterProperties = FRESH_WATER,
                 submerged: bool = True):
    """Force and moment from one lifting surface, in the hull frame.

    Parameters
    ----------
    velocity_hull:
        Hull-frame translational velocity ``(u, v, w)``.
    yaw_rate:
        Hull-frame yaw rate ``r``, rad/s.
    deflection:
        Rudder angle in radians; ignored unless the surface is
        ``controllable``.  Positive deflection yaws the bow to
        **starboard** (negative yaw), which is the convention
        :class:`~coxswain.sim.control.HeadingController` assumes.

    Returns
    -------
    ``(force, moment)`` -- ``(3,)`` hull-frame arrays, the moment taken
    about ``G_h``.
    """
    if not submerged:
        return np.zeros(3), np.zeros(3)

    u, v = float(velocity_hull[0]), float(velocity_hull[1])
    speed = float(np.hypot(u, v))
    if speed < 1e-6:
        return np.zeros(3), np.zeros(3)

    x_ac = float(surface.position[0])
    sideslip = np.arctan2(v, u)
    # Rotation carries the surface sideways at omega x r = (0, r x_ac, 0),
    # so the flow it sees is deflected by an extra r x_ac / V.  For a
    # surface aft of the centre of mass (x_ac < 0) that opposes the yaw
    # rate, which is where yaw damping comes from -- getting this sign
    # backwards turns the skeg into a yaw *amplifier*.
    local_angle = sideslip + yaw_rate * x_ac / speed

    if surface.controllable:
        limited = np.clip(deflection, -surface.max_deflection,
                          surface.max_deflection)
        local_angle = local_angle - surface.flap_effectiveness * limited

    lift_coefficient = surface.lift_curve_slope * local_angle
    dynamic_pressure = 0.5 * water.density * speed ** 2
    area = surface.area

    side_force = -lift_coefficient * dynamic_pressure * area
    induced_drag = (lift_coefficient ** 2
                    / (np.pi * surface.aspect_ratio * surface.oswald)
                    * dynamic_pressure * area)

    force = np.array([-induced_drag, side_force, 0.0])
    moment = np.cross(np.asarray(surface.position, dtype=float), force)
    return force, moment
