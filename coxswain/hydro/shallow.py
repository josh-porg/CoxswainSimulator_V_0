"""Shallow-water correction to wave-making resistance.

Why this matters here
---------------------
The long-term target of this project is trajectory optimisation on the
Charles, which is a river of a few metres' depth, not an Olympic lake.
Day, Campbell, Clelland, Doctors and Cichowicz, "Realistic evaluation of
hull performance for rowing shells, canoes, and kayaks in unsteady flow",
*Journal of Sports Sciences* **29**(10) (2011) 1059-1069, put the scale of
the effect plainly:

    "On a rowing lake with depth of 3.0 m, the critical speed is around
    5.4 m/s; many elite rowers will be travelling at this speed at some
    point in their stroke cycle."

and, for a men's pair at rate 35 whose speed swings by nearly half its
mean,

    "in water 3.0 m deep, the depth Froude Number would vary from 0.65 to
    1.09."

So a racing shell on a shallow course does not merely sit near critical
-- it sweeps *through* critical twice every stroke.  Any route choice that
trades distance against depth needs this term.

The governing parameter
-----------------------
The depth Froude number ``Fr_h = U / sqrt(g h)``.  Day et al. give the
regimes:

* ``Fr_h <= 0.5`` -- "results are similar to deep water";
* approaching ``Fr_h = 1`` -- "wavelengths, wave heights, and wave
  resistance all increase";
* ``Fr_h > 1`` (supercritical) -- "the transverse components of the wave
  pattern disappear and the wave resistance may be reduced compared with
  the critical value".

The model
---------
Subcritically this uses Schlichting's construction: a hull at speed ``U``
in depth ``h`` makes the same transverse wavelength as a hull at a higher
speed ``U_inf`` in deep water, so it suffers that deeper hull's wave
resistance.  Matching wavelengths through the finite-depth dispersion
relation ``U^2 = (g lambda / 2 pi) tanh(2 pi h / lambda)`` gives

    U = U_inf * sqrt( tanh( g h / U_inf^2 ) )

which is solved for ``U_inf``.  Since the wave term goes as ``U^2``, the
resistance amplification is ``(U_inf / U)^2``.

Schlichting's construction diverges at ``Fr_h = 1`` -- the matched deep
water speed runs away to infinity -- which is the known limit of the
method, so the factor is capped and then relaxed supercritically.  The
cap is a modelling choice, not a measurement, and is exposed as a
parameter; see :data:`DEFAULT_MAX_AMPLIFICATION` for what is and is not
justified.

Blockage is deliberately absent.  Schlichting's second term, the speed
loss from return flow round the hull, scales with ``sqrt(A_m) / h``.  A
racing eight has a midship submerged section of roughly 0.05 m^2, so
``sqrt(A_m) / h`` is about 0.07 in 3 m of water; the correction is
negligible at that blockage and is not modelled.  A shell in a narrow
lock would be a different matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEEP_WATER_FROUDE",
    "DEFAULT_MAX_AMPLIFICATION",
    "ShallowWaterModel",
    "depth_froude",
    "critical_speed",
    "matched_deep_water_speed",
    "wave_resistance_factor",
]

#: Depth Froude number below which Day et al. report deep-water behaviour.
DEEP_WATER_FROUDE = 0.5

#: Ceiling on the wave-resistance amplification near critical speed.
#:
#: Schlichting's matched-speed construction has no finite limit at
#: ``Fr_h = 1``, so something must bound it.  Finite-depth thin-ship
#: calculations for slender hulls put the near-critical peak at a few
#: times the deep-water value, and 3.0 is taken as a representative
#: figure.  It is **not** measured for a rowing shell: pinning it down
#: needs the towing-tank programme Day et al. describe.  Treat any result
#: that depends sensitively on behaviour at ``0.9 < Fr_h < 1.1`` as
#: indicative.
DEFAULT_MAX_AMPLIFICATION = 3.0

GRAVITY = 9.81


def depth_froude(speed, depth, gravity: float = GRAVITY):
    """``Fr_h = U / sqrt(g h)``.  Infinite depth gives zero."""
    speed = np.abs(np.asarray(speed, dtype=float))
    depth = np.asarray(depth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        froude = np.where(np.isfinite(depth) & (depth > 0),
                          speed / np.sqrt(gravity * np.maximum(depth, 1e-12)),
                          0.0)
    return froude


def critical_speed(depth, gravity: float = GRAVITY):
    """Speed at which ``Fr_h = 1``: the shallow-water wave speed."""
    return np.sqrt(gravity * np.asarray(depth, dtype=float))


def matched_deep_water_speed(speed, depth, gravity: float = GRAVITY,
                             tolerance: float = 1e-10,
                             max_iterations: int = 60):
    """Deep-water speed making the same transverse wavelength.

    Solves ``U = U_inf sqrt(tanh(g h / U_inf^2))`` for ``U_inf >= U``.

    The right-hand side is monotone increasing in ``U_inf``, so plain
    bisection converges unconditionally.  Only meaningful subcritically;
    above ``Fr_h = 1`` no solution exists and the caller must not use it.
    """
    speed = np.asarray(speed, dtype=float)
    depth = np.asarray(depth, dtype=float)

    def shallow_equivalent(deep):
        return deep * np.sqrt(np.tanh(gravity * depth / deep ** 2))

    low = np.maximum(speed, 1e-9)
    high = np.maximum(4.0 * low, 1.0)
    # grow the bracket until it straddles the target
    for _ in range(max_iterations):
        if np.all(shallow_equivalent(high) >= speed):
            break
        high = high * 2.0

    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        too_slow = shallow_equivalent(middle) < speed
        low = np.where(too_slow, middle, low)
        high = np.where(too_slow, high, middle)
        if np.all(high - low < tolerance * np.maximum(high, 1.0)):
            break
    return 0.5 * (low + high)


def wave_resistance_factor(speed, depth, gravity: float = GRAVITY,
                           max_amplification: float =
                           DEFAULT_MAX_AMPLIFICATION,
                           subcritical_limit: float = 0.92,
                           supercritical_relax: float = 1.6):
    """Multiplier on deep-water wave resistance for finite depth.

    Returns 1.0 in deep water, rises towards ``max_amplification`` as the
    depth Froude number approaches 1, and relaxes back towards 1 well
    above it.

    Parameters
    ----------
    subcritical_limit:
        Depth Froude number beyond which Schlichting's construction is
        abandoned and the factor is blended to the cap.  Its matched
        speed diverges at 1.0, so it cannot be used up to critical.
    supercritical_relax:
        Depth Froude number by which the supercritical factor has decayed
        back to 1.  Day et al. note the transverse wave system disappears
        above critical, so the resistance falls away from its peak.
    """
    froude = depth_froude(speed, depth, gravity)
    speed = np.abs(np.asarray(speed, dtype=float))
    factor = np.ones_like(froude)

    # -- subcritical: Schlichting's matched deep-water speed -------------
    schlichting = (froude > DEEP_WATER_FROUDE) & (froude <= subcritical_limit)
    if np.any(schlichting):
        matched = matched_deep_water_speed(
            np.where(schlichting, speed, 1.0),
            np.where(schlichting, np.asarray(depth, dtype=float), 1e9),
            gravity)
        ratio = np.where(schlichting,
                         (matched / np.maximum(speed, 1e-12)) ** 2, 1.0)
        factor = np.where(schlichting, ratio, factor)

    # value at the handover, used to blend smoothly to the cap
    handover = _schlichting_factor_at(subcritical_limit, gravity)

    # -- near-critical: blend from the handover value to the cap ---------
    near = (froude > subcritical_limit) & (froude <= 1.0)
    if np.any(near):
        progress = (froude - subcritical_limit) / (1.0 - subcritical_limit)
        blended = handover + (max_amplification - handover) * _smoothstep(
            progress)
        factor = np.where(near, blended, factor)

    # -- supercritical: relax back towards deep-water ---------------------
    above = froude > 1.0
    if np.any(above):
        progress = np.clip((froude - 1.0) / (supercritical_relax - 1.0),
                           0.0, 1.0)
        decayed = max_amplification + (1.0 - max_amplification) * _smoothstep(
            progress)
        factor = np.where(above, decayed, factor)

    return np.clip(factor, 1.0, max_amplification)


def _smoothstep(x):
    """C1 ramp from 0 to 1 on ``[0, 1]``; keeps the factor differentiable."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _schlichting_factor_at(froude: float, gravity: float = GRAVITY) -> float:
    """Schlichting amplification at a given depth Froude number.

    Scale free: the factor depends only on ``Fr_h``, so any consistent
    speed and depth give the same answer.
    """
    depth = 1.0
    speed = froude * np.sqrt(gravity * depth)
    matched = matched_deep_water_speed(np.array(speed), np.array(depth),
                                       gravity)
    return float((matched / speed) ** 2)


@dataclass(frozen=True)
class ShallowWaterModel:
    """Configuration for the shallow-water correction.

    ``depth`` of ``inf`` disables it entirely, which is the default so
    that nothing changes for open-water work until a depth is supplied.
    """

    depth: float = float("inf")
    max_amplification: float = DEFAULT_MAX_AMPLIFICATION
    subcritical_limit: float = 0.92
    supercritical_relax: float = 1.6
    gravity: float = GRAVITY

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("water depth must be positive")
        if self.max_amplification < 1.0:
            raise ValueError("max_amplification must be at least 1")
        if not 0.5 < self.subcritical_limit < 1.0:
            raise ValueError("subcritical_limit must lie in (0.5, 1)")
        if self.supercritical_relax <= 1.0:
            raise ValueError("supercritical_relax must exceed 1")

    @property
    def enabled(self) -> bool:
        return np.isfinite(self.depth)

    def froude(self, speed):
        return depth_froude(speed, self.depth, self.gravity)

    @property
    def critical_speed(self) -> float:
        return float(critical_speed(self.depth, self.gravity))

    def factor(self, speed):
        """Wave-resistance multiplier at this speed."""
        if not self.enabled:
            return np.ones_like(np.asarray(speed, dtype=float))
        return wave_resistance_factor(
            speed, self.depth, self.gravity, self.max_amplification,
            self.subcritical_limit, self.supercritical_relax)
