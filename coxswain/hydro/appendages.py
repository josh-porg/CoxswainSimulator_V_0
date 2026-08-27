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

Why the lift is not linear
--------------------------
A linear ``C_L = C_L_alpha alpha`` is only the first term.  Low-aspect-ratio
surfaces carry a second, **non-linear lift** from the vortices rolling up
off their side edges, and at low aspect ratio it is not a correction --
it is a large fraction of the total once the surface is working hard.
The standard treatment for exactly this case, a low-aspect-ratio
all-movable control surface in water, is Whicker and Fehlner (1958),
which is what ship rudders are designed to.  The lift splits into a
potential part and a cross-flow part::

    C_L = C_L_alpha sin(a) cos(a) + C_Dc sin(a)|sin(a)| cos(a)

``C_Dc`` is the cross-flow drag coefficient of the section, about 0.85 for
a rectangular plate (Hoerner).  Three things follow, and all of them
matter here:

* At small angles ``sin a cos a -> a`` and the second term vanishes as
  ``a^2``, so **every linearised stability derivative is unchanged**.
  The skeg in normal running sees under a degree and does not notice.
* At a rudder's working angles it is worth having.  Net of the ``cos``
  the potential term picks up, the rudder gains 9% of its lift at 8
  degrees and 15% at 25 -- so the *gain itself* grows with deflection,
  which is what bends the response back towards linear.
* ``sin a cos a`` peaks at 45 degrees and falls away after, so **stall is
  built in** rather than bolted on.  A low-aspect-ratio plate really does
  hold on until about 40 degrees, which is why a shell keeps steering at
  deflections that would have stalled a high-aspect-ratio foil long
  before.

The old linear model was missing all of this.  Its rudder moment came out
exactly proportional to deflection -- 17.70 N m per degree, from 2 degrees
right up to the 25 degree stop -- while the hull's cross-flow yaw damping
grows as the *square* of yaw rate.  A linear driver against a quadratic
damper gives a turn rate going as roughly the square root of helm, so five
times the rudder bought well under three times the turn rate, which is
what a coxswain reports.  Giving the rudder the small-angle law and then
asking it to work at 25 degrees is one half of that; see
:mod:`coxswain.hydro.crossflow` for the other half.

References
----------
[WF58] Whicker, L.F. and Fehlner, L.F. (1958). *Free-Stream
       Characteristics of a Family of Low-Aspect-Ratio, All-Movable
       Control Surfaces for Application to Ship Design.* DTMB Report 933.
[MT07] Molland, A.F. and Turnock, S.R. (2007). *Marine Rudders and
       Control Surfaces.* Butterworth-Heinemann. Chapter 3 gives the
       linear-plus-cross-flow form used here.
[H65]  Hoerner, S.F. (1965). *Fluid-Dynamic Drag*, and *Fluid-Dynamic
       Lift* (1975), for the cross-flow drag coefficient of plates.

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

__all__ = ["LiftingSurface", "SKEG_EIGHT", "RUDDER_EIGHT", "surface_load",
           "lift_coefficient_at", "flap_effectiveness_ratio",
           "MAX_RUDDER_DEFLECTION",
           "FLAP_VISCOUS_FACTOR"]


#: Largest rudder deflection the coxswain can call for, radians.
#:
#: **45 degrees, reported from the boat.**  This was 25 degrees in the
#: appendage model and 12 degrees in the router's reduced model -- two
#: different wrong answers to the same question, in four different files,
#: which is how a constant behaves when nothing owns it.  Both understated
#: it, and the 12 degree figure was the reason the route optimiser thought
#: an eight needed a pressure split for 134 strokes of a 4.8 km race: full
#: helm there bought only 1.05 deg/s, so everything the course demanded
#: above the median fell to the crew instead of the rudder.
#:
#: A rowing rudder is a small flap on a fixed fin with a wire linkage and
#: no aerodynamic balance, so it can be pulled to angles an aircraft
#: control surface never sees.  What it *does* at that angle is a separate
#: question the lift model has to answer -- see
#: :func:`flap_effectiveness_ratio` and the Whicker-Fehlner term in
#: :func:`lift_coefficient_at`, which together decide whether the last ten
#: degrees of helm buy anything or just drag.
MAX_RUDDER_DEFLECTION = np.radians(45.0)


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
    #: Tip chord over root chord.  ``1.0`` is rectangular and is the
    #: default, so surfaces written before this existed are unaffected;
    #: ``0.0`` is a triangle running to a point.
    #:
    #: This is not cosmetic.  A rowing fin is a swept delta, and a triangle
    #: encloses **half** the area of the rectangle around it, so reading its
    #: planform as ``span x chord`` overstates the area two-fold and
    #: understates the aspect ratio by the same factor.  Aspect ratio is
    #: what sets the lift-curve slope, and at the low values a fin lives at
    #: the slope is very sensitive to it -- getting the shape wrong was
    #: worth more than every dimension on the fin put together.
    #:
    #: When ``taper_ratio`` is given, ``chord`` is the **root** chord.
    taper_ratio: float = 1.0
    oswald: float = 0.8
    controllable: bool = False
    max_deflection: float = MAX_RUDDER_DEFLECTION
    #: Explicit control effectiveness ``dalpha/ddelta``.  Leave at ``None``
    #: to have it computed from :attr:`flap_chord_ratio`, which is what
    #: should normally happen.  ``1.0`` means an all-moving surface and is
    #: only correct when the whole thing pivots.
    flap_effectiveness: Optional[float] = None
    #: Flap chord as a fraction of the surface's total chord.  ``None`` or
    #: ``1.0`` is an all-moving surface; a rudder hinged behind a fixed fin
    #: is the fin-plus-flap chord it sits on.
    flap_chord_ratio: Optional[float] = None
    #: Flapped span as a fraction of the surface's span, for a flap that
    #: does not run the full depth of the fin.
    flap_span_ratio: float = 1.0
    #: Cross-flow drag coefficient of the section, feeding the non-linear
    #: lift term.  0.85 is Hoerner's figure for a rectangular plate and is
    #: in the middle of the range Whicker and Fehlner fit to their
    #: low-aspect-ratio control surfaces.  Set to 0.0 to recover the old
    #: purely linear surface.
    crossflow_coefficient: float = 0.85

    def __post_init__(self) -> None:
        if self.span <= 0 or self.chord <= 0:
            raise ValueError("span and chord must be positive")
        if np.asarray(self.position).shape != (3,):
            raise ValueError("position must be a length-3 vector")

    @property
    def mean_chord(self) -> float:
        """Mean geometric chord of a trapezoidal planform."""
        return self.chord * 0.5 * (1.0 + float(self.taper_ratio))

    @property
    def area(self) -> float:
        return self.span * self.mean_chord

    @property
    def aspect_ratio(self) -> float:
        return self.span ** 2 / self.area

    @property
    def control_effectiveness(self) -> float:
        """``dalpha/ddelta``: incidence change per unit deflection.

        Deflecting a flap re-cambers the section instead of pitching it
        bodily, so the surface behaves as though its incidence had moved
        by only a fraction of the flap angle.  That fraction is what
        :func:`flap_effectiveness_ratio` gives, discounted for viscosity
        and for any part of the span the flap does not cover.
        """
        if self.flap_effectiveness is not None:
            return float(self.flap_effectiveness)
        if self.flap_chord_ratio is None:
            return 1.0
        return (flap_effectiveness_ratio(self.flap_chord_ratio)
                * FLAP_VISCOUS_FACTOR * float(self.flap_span_ratio))

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


def flap_effectiveness_ratio(chord_ratio: float) -> float:
    """Control effectiveness ``tau`` of a plain flap, from thin-aerofoil theory.

    ``tau = dalpha/ddelta`` -- the change in the **whole surface's**
    effective angle of attack per unit flap deflection::

        tau = 1 - (theta - sin theta) / pi,    theta = arccos(2 c_f/c - 1)

    This is the standard preliminary-design result and it is tabulated in
    every stability and control text; the values below are the ones those
    tables carry:

    =========  =====
    c_f / c    tau
    =========  =====
    0.10       0.44
    0.20       0.55
    0.25       0.61
    0.30       0.66
    0.40       0.75
    0.50       0.82
    1.00       1.00
    =========  =====

    The ``c_f/c = 1`` limit is an **all-moving** surface, and that is the
    only case in which ``tau = 1``.  A hinged flap can never reach it: a
    quarter-chord flap moves the surface's effective incidence by only 61%
    of its own deflection, because deflecting a flap re-cambers the section
    rather than pitching it bodily.

    Thin-aerofoil theory is inviscid and known to run high -- measured
    effectiveness falls below it as the deflection grows and the flap
    boundary layer thickens.  See :data:`FLAP_VISCOUS_FACTOR`.

    References
    ----------
    [P49]  Perkins, C.D. and Hage, R.E. (1949). *Airplane Performance,
           Stability and Control.* Wiley.
    [E96]  Etkin, B. and Reid, L.D. (1996). *Dynamics of Flight: Stability
           and Control*, 3rd ed. Wiley, sec. 2.5.
    [N98]  Nelson, R.C. (1998). *Flight Stability and Automatic Control*,
           2nd ed. McGraw-Hill -- carries the tabulated curve above.
    [R85]  Roskam, J. (1985). *Airplane Design Part VI*, for the empirical
           corrections to the inviscid value.
    """
    ratio = float(np.clip(chord_ratio, 1e-6, 1.0))
    theta = np.arccos(2.0 * ratio - 1.0)
    return float(1.0 - (theta - np.sin(theta)) / np.pi)


#: Viscous knock-down on the inviscid ``tau``.
#:
#: Thin-aerofoil theory assumes attached flow over the flap.  Measured
#: plain-flap effectiveness runs below it, by roughly 10-20% at the
#: moderate deflections a control surface actually works at, and the gap
#: widens as the flap approaches its own stall.  0.85 is the usual
#: preliminary-design allowance.
FLAP_VISCOUS_FACTOR = 0.85


def lift_coefficient_at(surface: LiftingSurface, angle):
    """Lift coefficient of ``surface`` at angle of attack ``angle``.

    Whicker-Fehlner: a potential term plus a cross-flow term, both carrying
    the ``cos`` that makes the surface stall near 45 degrees instead of
    growing without limit.  Reduces to ``C_L_alpha * angle`` as the angle
    goes to zero, so no linearised derivative anywhere in the model moves.
    """
    angle = np.asarray(angle, dtype=float)
    sin_a, cos_a = np.sin(angle), np.cos(angle)
    potential = surface.lift_curve_slope * sin_a * cos_a
    crossflow = surface.crossflow_coefficient * sin_a * np.abs(sin_a) * cos_a
    return potential + crossflow


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
        local_angle = local_angle - surface.control_effectiveness * limited

    lift_coefficient = lift_coefficient_at(surface, local_angle)
    dynamic_pressure = 0.5 * water.density * speed ** 2
    area = surface.area

    side_force = -lift_coefficient * dynamic_pressure * area
    induced_drag = (lift_coefficient ** 2
                    / (np.pi * surface.aspect_ratio * surface.oswald)
                    * dynamic_pressure * area)

    force = np.array([-induced_drag, side_force, 0.0])
    moment = np.cross(np.asarray(surface.position, dtype=float), force)
    return force, moment
