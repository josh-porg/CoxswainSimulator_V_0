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


class WaveSurface:
    """A depth-aware wave table as a differentiable CasADi function.

    Research profiles give boats a
    :class:`~coxswain.hydro.finite_depth_michell.FiniteDepthWaveTable`:
    Michell's integral in deep water and Sretenskii's in finite depth.  An
    optimiser cannot call a Python table inside a CasADi graph, so the
    table is sampled once onto a grid in speed and log depth and fitted
    with a cubic B-spline, which CasADi differentiates.  Deep water is a
    one-dimensional B-spline in speed.

    Two surfaces, for the same reason the table interpolates two ways.
    Below ``Fr_h`` 0.8 the hull's humps sit at fixed speed, so resistance is
    fitted against speed and log depth.  Through and above critical the
    peak sits at fixed ``Fr_h``, so ``R / U^2`` is fitted against ``Fr_h``
    and log depth on nodes dense through 1.  A surface in speed alone
    missed the peak by 17-26% in 1.5 m of water.  The two are blended
    smoothly over ``Fr_h`` 0.8-0.9.

    Inputs outside the grids are clamped to them: slower than the first
    speed node reads that node's value scaled by ``u^2``, deeper than the
    last depth node reads the last row, and ``Fr_h`` above the last node
    reads the last node's coefficient.
    """

    #: ``Fr_h`` nodes for the critical surface; dense where the peak is.
    FROUDE = np.unique(np.concatenate([
        np.linspace(0.30, 0.80, 21),
        np.linspace(0.80, 1.20, 81),
        np.linspace(1.20, 3.00, 37),
    ]))

    def __init__(self, speeds, depths, shallow_values, deep_values,
                 froude=None, froude_values=None):
        import casadi as ca

        self.speeds = np.asarray(speeds, dtype=float)
        self.depths = np.asarray(depths, dtype=float)
        self._shallow = ca.interpolant(
            "wave_depth", "bspline",
            [self.speeds, np.log(self.depths)],
            np.asarray(shallow_values, dtype=float).ravel(order="F"))
        self._deep = ca.interpolant(
            "wave_deep", "bspline", [self.speeds],
            np.asarray(deep_values, dtype=float))
        self.froude = None
        if froude is not None:
            self.froude = np.asarray(froude, dtype=float)
            self._critical = ca.interpolant(
                "wave_froude", "bspline",
                [self.froude, np.log(self.depths)],
                np.asarray(froude_values, dtype=float).ravel(order="F"))

    @classmethod
    def from_table(cls, table, speeds=None, depths=None, froude=None):
        """Sample a table carrying ``at_depth`` onto the grids."""
        if speeds is None:
            speeds = np.arange(0.5, 8.0 + 1e-9, 0.025)
        if depths is None:
            depths = np.geomspace(0.3, 45.0, 80)
        if froude is None:
            froude = cls.FROUDE
        speeds = np.asarray(speeds, dtype=float)
        depths = np.asarray(depths, dtype=float)
        froude = np.asarray(froude, dtype=float)
        shallow = np.array([[float(table.at_depth(u, h)) for h in depths]
                            for u in speeds])
        deep = np.array([float(table(u)) for u in speeds])
        critical = np.empty((len(froude), len(depths)))
        for j, h in enumerate(depths):
            wave_speed2 = 9.80665 * h
            for i, fr in enumerate(froude):
                u = fr * np.sqrt(wave_speed2)
                critical[i, j] = (float(table.at_depth(u, h))
                                  / (fr * fr * wave_speed2))
        return cls(speeds, depths, shallow, deep, froude, critical)

    def __call__(self, speed, depth=None):
        """Wave resistance, N, as a CasADi expression."""
        import casadi as ca

        low, high = float(self.speeds[0]), float(self.speeds[-1])
        clamped = ca.fmin(ca.fmax(speed, low), high)
        # below the grid, resistance goes as u^2 from the first node
        scale = ca.if_else(speed < low, (speed / low) ** 2, 1.0)
        if depth is None:
            return self._deep(clamped) * scale
        bounded = ca.fmin(ca.fmax(depth, float(self.depths[0])),
                          float(self.depths[-1]))
        log_depth = ca.log(bounded)
        by_speed = self._shallow(ca.vertcat(clamped, log_depth)) * scale
        if self.froude is None:
            return by_speed
        froude = speed / ca.sqrt(9.80665 * bounded)
        weight = ca.fmin(ca.fmax((froude - 0.8) / 0.1, 0.0), 1.0)
        weight = weight * weight * (3.0 - 2.0 * weight)
        coefficient = self._critical(ca.vertcat(
            ca.fmin(ca.fmax(froude, float(self.froude[0])),
                    float(self.froude[-1])), log_depth))
        by_froude = coefficient * speed * speed
        return (1.0 - weight) * by_speed + weight * by_froude


def wave_function_for(boat):
    """The boat's :class:`WaveSurface`, or ``None`` if its table is plain.

    A boat whose wave table knows depth -- which research profiles give it
    -- gets a surface built once and kept on the table, so every model and
    every solve on that hull shares it.  Any other boat gets ``None``, and
    :func:`hull_resistance` keeps its constant wave coefficient and smoothed
    shallow-water factor exactly as before.
    """
    table = getattr(boat, "wave_table", None)
    if getattr(table, "at_depth", None) is None:
        return None
    surface = getattr(table, "_casadi_surface", None)
    if surface is None:
        surface = WaveSurface.from_table(table)
        table._casadi_surface = surface
    return surface


def hull_resistance(u, v, w, wetted_area, transverse_area, plan_area,
                    lateral_area, mean_wetted_length, depth=None,
                    density: float = 1000.0,
                    kinematic_viscosity: float = 1.0e-6,
                    shape: float = 0.01, wave: float = 0.02,
                    friction_zero: float = 0.075, form_factor: float = 1.0,
                    cross_flow_lateral: float = 1.0,
                    cross_flow_vertical: float = 1.0,
                    wave_function: WaveSurface = None):
    """Hull resistance in the hull frame, as a CasADi expression.

    Mirrors :func:`coxswain.hydro.resistance.hull_resistance` term for
    term.  ``depth`` of ``None`` is deep water.

    ``wave_function``, when given, is the boat's own wave resistance
    against speed and depth (:class:`WaveSurface`, which research profiles
    use) and replaces the constant wave coefficient and the smoothed
    shallow-water factor together.  Without it nothing changes.
    """
    import casadi as ca

    speed_x = ca.fmax(ca.fabs(u), SPEED_FLOOR)
    dynamic_pressure = 0.5 * density * u * u

    reynolds = speed_x * mean_wetted_length / kinematic_viscosity
    c_f = friction_coefficient(reynolds, friction_zero)

    shape_drag = dynamic_pressure * transverse_area * shape
    viscous_drag = dynamic_pressure * wetted_area * c_f * form_factor
    if wave_function is not None:
        wave_drag = wave_function(speed_x, depth)
    else:
        factor = (1.0 if depth is None
                  else shallow_water_factor(speed_x, depth))
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


def lift_coefficient_at(surface, angle):
    """Whicker-Fehlner lift coefficient, symbolically.

    Mirrors :func:`coxswain.hydro.appendages.lift_coefficient_at` term for
    term: a potential term plus a cross-flow term, both carrying the
    ``cos`` that stalls the surface near 45 degrees instead of letting it
    grow without limit.

    **This used to be ``lift_curve_slope * angle``**, the small-angle
    limit, and the two models therefore disagreed about the boat by up to
    6.5% of rudder force -- the linear form reading *low* around 15
    degrees, because the cross-flow term it omitted adds more than the
    ``cos`` it also omitted takes away.  The optimiser was solving for a
    boat with a weaker rudder than the simulator would give it, which
    matters precisely where ``SOURCES.md`` sec. 10 says steering is
    marginal: against the tightest bends.  It survived because the test
    comparing the two paths could not run at all (see below), so nothing
    was checking.

    Differentiable enough for the NLP: ``sin|sin|`` has derivative
    ``2|sin|cos``, which is continuous through zero, so the kink in
    ``fabs`` does not reach the gradient.
    """
    import casadi as ca

    sin_a, cos_a = ca.sin(angle), ca.cos(angle)
    potential = surface.lift_curve_slope * sin_a * cos_a
    crossflow = (surface.crossflow_coefficient * sin_a * ca.fabs(sin_a)
                 * cos_a)
    return potential + crossflow


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
        # ``control_effectiveness``, not the raw ``flap_effectiveness``
        # field: the field is None for a surface that derives its
        # effectiveness from the flap chord ratio instead of stating it,
        # and the NumPy path has always gone through the property.  Using
        # the field here made the two implementations disagree wherever it
        # was set, and raise TypeError wherever it was not.
        local_angle = local_angle - surface.control_effectiveness * limited

    lift_coefficient = lift_coefficient_at(surface, local_angle)
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
