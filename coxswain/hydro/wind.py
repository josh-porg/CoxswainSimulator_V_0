"""Aerodynamic loads: the largest force this model was missing.

Shells are extraordinarily wind-sensitive, and the sensitivity is
asymmetric in a way that matters for racing: [K13] reports that a 5 m/s
headwind makes an eight **12.2% slower** while the same tailwind makes it
only **5.1% faster**.  Any model used to plan a race on a river with a
prevailing wind has to carry that.

Where the drag actually is
--------------------------
Not on the hull.  [K13] puts aerodynamic drag at about **13% of total
resistance** in still air, and splits it:

| component | share of aerodynamic drag |
|---|---|
| oars | 50% |
| rowers' bodies | 35% |
| boat and riggers | 15% |

So five-sixths of it is crew and oars — things that move relative to the
boat, sweep through the air, and present a changing area through the
stroke.  A single lumped hull windage term would put the force in the
wrong place as well as getting its magnitude wrong: the oars' share acts
at the riggers, a metre outboard, so a crosswind on the oars is a **roll
and yaw moment**, not just drag.  That matters here because §15 established
that roll is a marginal, actively-held mode.

These proportions are for still air.  [K13] notes they can rise up to
fourfold in a headwind and fall to zero in a sufficient tailwind, which is
simply the ``v_rel^2`` scaling and comes out of the model rather than
being imposed.

What is modelled
----------------
The three components above, each with its own reference area and its own
point of application, driven by the **relative** wind — true wind minus
boat velocity — resolved into the hull frame.  A uniform, steady wind
field is the default; :class:`WindField` is deliberately an interface so
that a spatially varying or gusting field can be substituted without
touching the force model.  Rivers are not uniform: a bend puts the wind
across the boat that was behind it, and the Charles has bridges and banks
that shelter parts of the reach.

Calibration
-----------
The absolute level is set by matching [K13]'s 13% figure at racing speed,
so the one free parameter is fixed by published data rather than chosen.
The headwind/tailwind asymmetry is then a *prediction*, not a fit, and is
checked against [K13]'s measured 12.2% / 5.1%.

References
----------
[K13] Kleshnev, V. Rowing Biomechanics Newsletter, on wind effects and the
      composition of aerodynamic drag.  Summarised figures: aerodynamic
      drag ~13% of total resistance, split 50% oars / 35% bodies / 15%
      boat and riggers; 5 m/s headwind costs an eight 12.2% of speed, a
      5 m/s tailwind gains 5.1%.
[H21] "Rowing Against the Wind: An Analysis of the Impact of Variable
      Wind Conditions", Harvard,
      https://dash.harvard.edu/server/api/core/bitstreams/4f46c026-cb8b-4f50-aa82-1a63b2baa8a5/content
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WindField", "UniformWind", "AeroModel", "AIR_DENSITY",
           "log_profile_factor", "WATER_ROUGHNESS", "ANEMOMETER_HEIGHT"]

#: Density of air at 15 C and sea level, kg/m^3.
AIR_DENSITY = 1.225



#: Roughness length for open water, metres.  Standard value for the
#: logarithmic wind profile over a smooth water surface; it rises with
#: wave height, so this is the calm-water end.
WATER_ROUGHNESS = 2.0e-4

#: Height at which meteorological wind speeds are quoted, metres.  WMO
#: standard anemometer height.
ANEMOMETER_HEIGHT = 10.0


def log_profile_factor(height: float,
                       reference_height: float = ANEMOMETER_HEIGHT,
                       roughness: float = WATER_ROUGHNESS) -> float:
    """Fraction of the quoted wind speed actually felt at ``height``.

    ``u(z)/u(z_ref) = ln(z/z0) / ln(z_ref/z0)`` -- the neutral logarithmic
    boundary layer.

    This is not a refinement, it is a correction that has to be there.
    Wind speeds are quoted at 10 m; a rower's shoulders are at about 0.6 m
    and an oar shaft lower still, deep inside the surface layer where the
    wind is substantially slower.  Over calm water the factor at 0.6 m is
    about **0.74**, so driving the drag model with the quoted speed
    over-predicts the force by roughly ``1/0.74^2 = 1.8`` in a headwind.

    Taking it as unity was the single largest error in the first version
    of this model: it made a 5 m/s headwind cost an eight 17.9% of its
    speed against the 12.2% [K13] measures.
    """
    height = max(float(height), 2.0 * roughness)
    return float(np.log(height / roughness)
                 / np.log(reference_height / roughness))


class WindField:
    """True wind as a function of position and time, in the absolute frame.

    Subclass to provide a spatially varying or unsteady field; the force
    model only ever calls :meth:`at`.
    """

    def at(self, x, y, t=0.0):
        raise NotImplementedError


@dataclass(frozen=True)
class UniformWind(WindField):
    """Steady, spatially uniform wind.

    ``speed`` in m/s and ``bearing`` in radians measured as the direction
    the wind blows *towards*, in the same absolute frame as the boat --
    note this is the opposite of the meteorological convention, which
    names the direction wind comes *from*.  Stated explicitly because
    getting it backwards silently turns a headwind into a tailwind.
    """

    speed: float = 0.0
    bearing: float = 0.0

    def at(self, x, y, t=0.0):
        return np.array([self.speed * np.cos(self.bearing),
                         self.speed * np.sin(self.bearing), 0.0])

    @classmethod
    def head_on(cls, speed: float, heading: float) -> "UniformWind":
        """A pure headwind of ``speed`` for a boat on ``heading``."""
        return cls(speed=speed, bearing=heading + np.pi)

    @classmethod
    def following(cls, speed: float, heading: float) -> "UniformWind":
        return cls(speed=speed, bearing=heading)


@dataclass
class AeroModel:
    """Aerodynamic loads on hull, crew and oars.

    Each component carries its own drag area ``C_d A``; the split between
    them follows [K13] and the total is calibrated so that aerodynamic
    drag is :attr:`still_air_fraction` of total resistance at
    :attr:`reference_speed`.
    """

    #: Effective drag areas, m^2.  Set by :meth:`calibrate`.
    hull_area: float = 0.10
    crew_area: float = 0.23
    oar_area: float = 0.33

    #: Height above the waterline at which each component's force acts.
    #: These now do double duty: they set the moment arm *and* which part
    #: of the wind boundary layer the component sits in.
    hull_height: float = 0.15
    crew_height: float = 0.60
    oar_height: float = 0.40

    #: Apply the logarithmic wind profile.  Quoted wind speeds are at 10 m;
    #: a shell is not.  See :func:`log_profile_factor`.
    use_log_profile: bool = True
    reference_height: float = ANEMOMETER_HEIGHT
    roughness: float = WATER_ROUGHNESS

    #: Lateral offset at which the oar force acts, metres.  A crosswind on
    #: the oars is therefore a roll and yaw moment, not merely drag.
    oar_offset: float = 0.85

    density: float = AIR_DENSITY

    #: [K13]: aerodynamic drag is about 13% of total resistance in still
    #: air at racing speed.
    still_air_fraction: float = 0.13
    reference_speed: float = 5.2

    @property
    def total_area(self) -> float:
        return self.hull_area + self.crew_area + self.oar_area

    @classmethod
    def calibrate(cls, boat, reference_speed: float = 5.2,
                  still_air_fraction: float = 0.13, **kwargs):
        """Fix the drag areas from the boat's own hydrodynamic resistance.

        The total is whatever makes aerodynamic drag the published
        fraction of resistance at racing speed; the split between hull,
        crew and oars is [K13]'s 15/35/50.
        """
        import numpy as np

        from .resistance import hull_resistance

        submerged = boat.mesh.submerged(
            np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
            rho=boat.water.density, gravity=9.80665, water_level=0.0)
        force, _ = hull_resistance(
            np.array([reference_speed, 0.0, 0.0]), submerged,
            mean_wetted_length=boat.length, water=boat.water,
            coefficients=boat.resistance)
        water = float(abs(np.asarray(force)[0]))

        target = still_air_fraction * water
        area = 2.0 * target / (AIR_DENSITY * reference_speed ** 2)
        return cls(hull_area=0.15 * area, crew_area=0.35 * area,
                   oar_area=0.50 * area,
                   still_air_fraction=still_air_fraction,
                   reference_speed=reference_speed, **kwargs)

    # -- loads ------------------------------------------------------------
    def relative_wind_hull(self, wind_abs, velocity_abs, rotation):
        """Apparent wind in the hull frame.

        The boat's own motion is most of it: at 5 m/s a shell generates
        more apparent wind than most days provide.
        """
        relative = np.asarray(wind_abs, dtype=float) \
            - np.asarray(velocity_abs, dtype=float)
        return np.asarray(rotation, dtype=float).T @ relative

    def loads(self, wind_abs, velocity_abs, rotation):
        """``(force, moment)`` in the hull frame.

        Each component is a bluff body: force along the apparent wind,
        magnitude ``0.5 rho C_d A |v| v``.  Using ``|v| v`` rather than
        ``v^2`` keeps the sign right in a tailwind that overtakes the
        boat, which is exactly the case [K13] says drives the aerodynamic
        share to zero.
        """
        apparent = self.relative_wind_hull(wind_abs, velocity_abs, rotation)
        speed = float(np.linalg.norm(apparent[:2]))
        if speed == 0.0:
            return np.zeros(3), np.zeros(3)

        force = np.zeros(3)
        moment = np.zeros(3)
        boat_velocity_hull = np.asarray(rotation, dtype=float).T             @ np.asarray(velocity_abs, dtype=float)
        true_hull = np.asarray(rotation, dtype=float).T             @ np.asarray(wind_abs, dtype=float)

        for area, height, offset in (
                (self.hull_area, self.hull_height, 0.0),
                (self.crew_area, self.crew_height, 0.0),
                (self.oar_area, self.oar_height, self.oar_offset)):
            # Each component sits at its own height in the boundary layer,
            # so each feels a different fraction of the quoted wind.  The
            # boat's own motion is not attenuated -- it is not wind.
            shear = log_profile_factor(height, self.reference_height,
                                       self.roughness)                 if self.use_log_profile else 1.0
            local = shear * true_hull - boat_velocity_hull
            local_speed = float(np.hypot(local[0], local[1]))
            component = 0.5 * self.density * area * local_speed * local
            component[2] = 0.0
            force += component
            arm = np.array([0.0, offset, height])
            moment += np.cross(arm, component)
        return force, moment
