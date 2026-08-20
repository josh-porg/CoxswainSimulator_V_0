"""The hydrodynamics, symbolically, without linearising anything.

An earlier version of :mod:`coxswain.river.strokemodel` reduced the whole
hull and appendage response to four linear coefficients, fitted by
perturbing the full model at one operating point.  That was expedience,
not necessity.  CasADi handles the real expressions: the ITTC friction
line, the Polhamus lift slope, ``atan2`` for angle of attack, the
quadratic cross-flow terms.  All of it differentiates.

The linearisation was also probably wrong in a way that mattered.  The
coefficients were fitted at ``v = 0.3 m/s`` in *straight running* and then
used in a split-driven turn, where the sideslip and the flow over the skeg
are not the same thing at all.

What is ported
--------------
:func:`hull_resistance` -- Formaggia's decomposition, term for term:
shape drag on the transverse area, viscous drag on the wetted area through
the ITTC 1957 correlation line, wave drag on the waterplane with the
shallow-water factor, plus the quadratic cross-flow terms in sway and
heave.

:func:`surface_load` -- one lifting surface: sideslip plus the yaw-rate
deflection ``r x_ac / V``, the flap term for a controllable surface, a
linear lift slope and the induced drag that follows from it.

Both mirror the numpy functions of the same name line for line, and
``tests/unit/test_hydro_casadi.py`` checks them against those numpy
originals over a grid of states rather than at a single point.  The
numpy versions stay the reference implementation; these exist so an
optimiser can differentiate them, not to replace them.

Smoothing
---------
Two places need care because the numpy originals are not differentiable:

``sign(u)`` and ``v |v|``
    Fine as written -- ``fabs`` is differentiable away from zero and the
    boat never sits at exactly zero speed.  A small floor keeps the
    Reynolds number and the speed division finite.

the speed guard
    numpy returns zero below ``1e-6``; here the speed is floored instead,
    which is continuous rather than a branch.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SPEED_FLOOR",
    "friction_coefficient",
    "shallow_water_factor",
    "hull_resistance",
    "surface_load",
    "appendage_loads",
]

#: Speeds below this are floored rather than branched on.  A racing shell
#: at 5 m/s is nine orders of magnitude above it; it exists only to keep
#: the Reynolds number and the ``v / speed`` divisions finite if an
#: optimiser probes a stopped boat.
SPEED_FLOOR = 1e-3


def friction_coefficient(reynolds, friction_zero: float = 0.075):
    """ITTC 1957 model-ship correlation line, ``C_f0 / (log10(Re) - 2)^2``.

    Base ten, as the correlation is defined -- the legacy code read the
    paper's "log" as natural and understated ``C_f`` about eightfold.  The
    denominator is floored so the expression stays finite if an optimiser
    probes a near-zero Reynolds number.
    """
    import casadi as ca

    log_re = ca.log10(ca.fmax(reynolds, 10.0))
    return friction_zero / ca.fmax(log_re - 2.0, 0.1) ** 2


def shallow_water_factor(speed, depth, gravity: float = 9.80665,
                         max_amplification: float = 3.0,
                         subcritical_limit: float = 0.92,
                         supercritical_relax: float = 1.6):
    """Wave-resistance multiplier for finite depth.

    A smooth stand-in for
    :class:`~coxswain.hydro.shallow.ShallowWaterModel`, which is piecewise
    around the critical depth Froude number.  The shape is a Lorentzian in
    ``Fr_h`` peaking at 1: it rises like the real curve through the
    subcritical range, caps at ``max_amplification`` rather than diverging,
    and relaxes above critical.

    The cap is a modelling choice and not a measurement -- the same caveat
    the piecewise model carries.  It matters here because the Charles at
    2-4 m puts an eight close to the peak.
    """
    import casadi as ca

    froude = speed / ca.sqrt(gravity * ca.fmax(depth, 0.2))
    width = 1.0 - subcritical_limit
    peak = max_amplification - 1.0
    return 1.0 + peak / (1.0 + ((froude - 1.0) / width) ** 2)


def hull_resistance(u, v, w, wetted_area, transverse_area, plan_area,
                    lateral_area, mean_wetted_length, depth=None,
                    density: float = 1000.0,
                    kinematic_viscosity: float = 1.0e-6,
                    shape: float = 0.01, wave: float = 0.02,
                    friction_zero: float = 0.075, form_factor: float = 1.0,
                    cross_flow_lateral: float = 1.0,
                    cross_flow_vertical: float = 1.0):
    """Hull resistance in the hull frame, as a CasADi expression.

    Mirrors :func:`coxswain.hydro.resistance.hull_resistance` term for
    term.  ``depth`` of ``None`` is deep water.
    """
    import casadi as ca

    speed_x = ca.fmax(ca.fabs(u), SPEED_FLOOR)
    dynamic_pressure = 0.5 * density * u * u

    reynolds = speed_x * mean_wetted_length / kinematic_viscosity
    c_f = friction_coefficient(reynolds, friction_zero)

    shape_drag = dynamic_pressure * transverse_area * shape
    viscous_drag = dynamic_pressure * wetted_area * c_f * form_factor
    factor = 1.0 if depth is None else shallow_water_factor(speed_x, depth)
    wave_drag = dynamic_pressure * plan_area * wave * factor

    longitudinal = shape_drag + viscous_drag + wave_drag
    # tanh gives the sign of u without a branch, and is exact away from
    # zero at this scale
    force_x = -ca.tanh(u / SPEED_FLOOR) * longitudinal

    force_y = -0.5 * density * cross_flow_lateral * lateral_area \
        * v * ca.fabs(v)
    force_z = -0.5 * density * cross_flow_vertical * plan_area \
        * w * ca.fabs(w)
    return ca.vertcat(force_x, force_y, force_z)


def surface_load(surface, u, v, yaw_rate, deflection=0.0,
                 density: float = 1000.0):
    """Force and moment from one lifting surface, as CasADi expressions.

    Mirrors :func:`coxswain.hydro.appendages.surface_load`, including the
    sign of the yaw-rate term: rotation carries the surface sideways at
    ``r x_ac``, so the flow it sees is deflected by ``r x_ac / V``.  For a
    surface aft of the centre of mass that opposes the yaw rate, which is
    where yaw damping comes from -- and getting it backwards turns the skeg
    into a yaw amplifier.
    """
    import casadi as ca

    speed = ca.fmax(ca.sqrt(u * u + v * v), SPEED_FLOOR)
    x_ac = float(surface.position[0])

    sideslip = ca.atan2(v, u)
    local_angle = sideslip + yaw_rate * x_ac / speed

    if surface.controllable:
        limited = ca.fmax(ca.fmin(deflection, surface.max_deflection),
                          -surface.max_deflection)
        local_angle = local_angle - surface.flap_effectiveness * limited

    lift_coefficient = surface.lift_curve_slope * local_angle
    dynamic_pressure = 0.5 * density * speed ** 2
    area = float(surface.area)

    side_force = -lift_coefficient * dynamic_pressure * area
    induced_drag = (lift_coefficient ** 2
                    / (np.pi * surface.aspect_ratio * surface.oswald)
                    * dynamic_pressure * area)

    force = ca.vertcat(-induced_drag, side_force, 0.0)
    position = ca.DM([float(surface.position[0]), float(surface.position[1]),
                      float(surface.position[2])])
    moment = ca.cross(position, force)
    return force, moment


def appendage_loads(surfaces, u, v, yaw_rate, deflection=0.0,
                    density: float = 1000.0):
    """Summed force and moment from every appendage."""
    import casadi as ca

    force = ca.DM.zeros(3)
    moment = ca.DM.zeros(3)
    for surface in surfaces:
        one_force, one_moment = surface_load(surface, u, v, yaw_rate,
                                             deflection, density)
        force = force + one_force
        moment = moment + one_moment
    return force, moment
