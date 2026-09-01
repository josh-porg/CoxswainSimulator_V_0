"""Hull resistance.

Longitudinal resistance follows Formaggia et al. section 6.1 exactly::

    R_shape = 1/2 rho |Gamma_X| C_dX  V_X^2        C_dX = 0.01
    R_vis   = 1/2 rho |Gamma|   C_f   V_X^2        C_f from ITTC-1957
    R_wave  = 1/2 rho |Gamma_Z| C_dw  V_X^2        C_dw = 0.02

with the ITTC 1957 model-ship correlation line

    C_f = C_f0 / (log10(Re) - 2)^2       C_f0 = 0.075
    Re  = V_X L_M / nu

``L_M`` is the mean wetted length, taken as the hull length as the paper
suggests.

Two things the legacy implementation got wrong here
---------------------------------------------------
1. It used ``np.log`` where the ITTC correlation line is defined with
   ``log10``.  The paper writes "log", but its reference [13] is the ITTC
   1957 line, which is unambiguously base 10.  Natural log understates
   ``C_f`` by a factor of about eight.
2. ``mu_water = 0.00982 Pa s`` is roughly ten times the real value
   (~1.0e-3 at 20 C), so ``nu`` was ten times too large.  The same file
   hard-coded ``1e-6`` for the skeg's Reynolds number, contradicting
   itself.  The two errors happened to cancel to within about 10%, which
   is why the drag looked plausible.

Lateral and vertical resistance
-------------------------------
The paper is planar and needs neither.  Six degrees of freedom do: a
shell's large lateral area makes sideslip heavily damped, and heave needs
damping or it rings at its undamped natural frequency forever.  Both are
modelled as quadratic cross-flow drag on the corresponding projected
area, which is standard practice for slender bodies and is the same form
as the longitudinal terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hull import SubmergedProperties
from .shallow import ShallowWaterModel

__all__ = [
    "WaterProperties",
    "FRESH_WATER",
    "SEA_WATER",
    "ResistanceCoefficients",
    "PAPER_LITERAL",
    "friction_coefficient",
    "hull_resistance",
    "DEEP_WATER",
]


@dataclass(frozen=True)
class WaterProperties:
    """Density and kinematic viscosity of the water being rowed on."""

    density: float           # kg/m^3
    kinematic_viscosity: float  # m^2/s
    name: str = ""

    def __post_init__(self) -> None:
        if self.density <= 0 or self.kinematic_viscosity <= 0:
            raise ValueError("water properties must be positive")


#: Fresh water at 15 C -- the Charles River, and the default.
FRESH_WATER = WaterProperties(density=999.1, kinematic_viscosity=1.139e-6,
                              name="fresh water at 15 C")

#: Sea water at 15 C, for coastal and open-water rowing.
SEA_WATER = WaterProperties(density=1025.9, kinematic_viscosity=1.188e-6,
                            name="sea water at 15 C")


@dataclass(frozen=True)
class ResistanceCoefficients:
    """Dimensionless resistance coefficients.

    ``shape`` and ``friction_zero`` are the paper's stated values.  The
    wave term needs care -- see :data:`PAPER_LITERAL` and the note below.

    ``cross_flow_*`` have no counterpart in the paper; see the module
    docstring.
    """

    shape: float = 0.01           # C_dX, on the transverse area
    friction_zero: float = 0.075  # C_f0, the ITTC-1957 constant
    wave: float = 0.00084         # C_dw, on the reference area below
    wave_reference: str = "wetted"  # "wetted" | "plan" | "transverse"
    cross_flow_lateral: float = 0.60
    cross_flow_vertical: float = 0.80
    form_factor: float = 1.0      # (1 + k), multiplies the friction term

    def __post_init__(self) -> None:
        allowed = {"wetted", "plan", "transverse"}
        if self.wave_reference not in allowed:
            raise ValueError(
                f"wave_reference must be one of {sorted(allowed)}, "
                f"got {self.wave_reference!r}"
            )

    def wave_area(self, properties: SubmergedProperties) -> float:
        return {
            "wetted": properties.wetted_area,
            "plan": properties.plan_area,
            "transverse": properties.transverse_area,
        }[self.wave_reference]


#: The paper's stated wave model, verbatim: ``C_dw = 0.02`` on the
#: waterplane area.  Provided so the published model can be reproduced
#: exactly, but **it is not the default**, because taken literally it is
#: not physical: it predicts 2.4 kN of wave drag on an eight at 5.5 m/s
#: and 314 N on the paper's own single scull at 4.5 m/s, against measured
#: *total* resistances of roughly 450 N and 90 N respectively.
#:
#: The paper also writes its area measures as ``integral of q dsigma``,
#: weighted by submersion depth, which would make them volumes -- so the
#: intended reference area for ``C_dw`` is genuinely ambiguous in the
#: source.  The default instead references the wave coefficient to the
#: wetted surface, as ITTC practice does for ``C_F``, with the magnitude
#: calibrated so an eight's total resistance matches published towing
#: data.  See ``docs/validation.md``.
PAPER_LITERAL = ResistanceCoefficients(wave=0.02, wave_reference="plan")

#: Wave coefficient reduced to what the rowing literature supports.
#:
#: The default ``0.00084`` was calibrated against a stated "measured total
#: resistance of roughly 450 N for an eight at 5.5 m/s".  Two independent
#: sources now disagree with the consequence of that calibration, which
#: makes wave drag **28% of total** at 6 m/s:
#:
#: * Pulman, *The Physics of Rowing*, eq. 13: a VIII at racing speed sits
#:   at Froude 0.35, which his figure 8 puts at a **local minimum** of the
#:   wave-drag curve -- "although wave drag is certainly present, it is
#:   not significant for a racing boat".  A hull 30 times longer than it
#:   is wide is shaped precisely to achieve that.
#: * Buckmann and Harris (2014) coast-down an 8+ and get a drag constant
#:   of 10.5 kg/m, 95% interval 9.6-11.4, against this model's 16.1.
#:
#: Their number cannot be taken at face value either: at 6 m/s it implies
#: 378 N total, which is **below** the ITTC skin friction (354 N) plus air
#: drag (67 N) computed from the model's own geometry -- and that geometry
#: reproduces the measured displacement to 0.1% and sits inside the
#: published wetted-area band.  A measurement below the friction floor is
#: measuring something other than steady drag; a coast-down neglects added
#: mass and starts at the instant the crew stops moving, both of which
#: bias the deceleration low.
#:
#: So this set takes the direction both sources agree on without adopting
#: either magnitude: wave drag reduced to about a tenth of total at racing
#: Froude number, which puts the model just above its own physical floor
#: rather than a third above it.  ``scripts/validate_drag.py`` is the test.
#:
#: **The structural fix is not this.**  A constant coefficient makes wave
#: drag a fixed fraction of a ``v^2`` law, so it can never have the hollow
#: Pulman's figure 8 shows and the hull is designed around.  Getting that
#: right needs Michell's integral over a distributed source -- a real
#: piece of work, and the two-point bow/stern interference factor is not a
#: shortcut to it: taken literally it oscillates a hundredfold with speed
#: and puts a masters eight on a hump.
LOW_WAVE = ResistanceCoefficients(wave=0.00022, wave_reference="wetted")


def friction_coefficient(reynolds: float,
                         friction_zero: float = 0.075) -> float:
    """ITTC 1957 model-ship correlation line.

    ``C_f = C_f0 / (log10(Re) - 2)^2``.  Below ``Re = 1e5`` the line is not
    meaningful; it is clamped there rather than allowed to blow up as
    ``log10(Re) -> 2``.
    """
    reynolds = max(float(reynolds), 1.0e5)
    return friction_zero / (np.log10(reynolds) - 2.0) ** 2


#: The default: unbounded depth, so the shallow-water term is inert.
DEEP_WATER = ShallowWaterModel()


def hull_resistance(velocity_hull: np.ndarray,
                    properties: SubmergedProperties,
                    mean_wetted_length: float,
                    water: WaterProperties = FRESH_WATER,
                    coefficients: ResistanceCoefficients = None,
                    shallow: ShallowWaterModel = None,
                    wave_table=None):
    """Resistance force on the hull, in the hull frame.

    Parameters
    ----------
    velocity_hull:
        Translational velocity resolved in the hull frame, ``(u, v, w)``.
    properties:
        Current submerged-surface measures.
    mean_wetted_length:
        ``L_M`` for the Reynolds number.
    shallow:
        Finite-depth correction; ``None`` means deep water.  Only the wave
        term is scaled -- Day et al. note that the boundary-layer changes
        behind the viscous term "are less likely to be sensitive to water
        depth".

    Returns
    -------
    ``(force, breakdown)`` where ``force`` is a ``(3,)`` hull-frame array
    and ``breakdown`` is a dict of the individual longitudinal components,
    which the validation tests check term by term.
    """
    coefficients = coefficients or ResistanceCoefficients()
    u, v, w = (float(velocity_hull[0]), float(velocity_hull[1]),
               float(velocity_hull[2]))

    dynamic_pressure = 0.5 * water.density * u * u
    reynolds = abs(u) * mean_wetted_length / water.kinematic_viscosity
    c_f = friction_coefficient(reynolds, coefficients.friction_zero)

    shape = dynamic_pressure * properties.transverse_area * coefficients.shape
    viscous = (dynamic_pressure * properties.wetted_area * c_f
               * coefficients.form_factor)
    shallow = shallow or DEEP_WATER
    depth_factor = float(shallow.factor(u))
    if wave_table is not None:
        # Michell's integral, precomputed against speed.  This replaces the
        # constant coefficient entirely: there is no fitted wave number
        # left, only the hull's own offsets.  The shallow-water factor
        # still multiplies it, because finite depth changes the wave
        # system and Michell as written here assumes deep water.
        wave = float(np.abs(wave_table(abs(u)))) * depth_factor
    else:
        wave = (dynamic_pressure * coefficients.wave_area(properties)
                * coefficients.wave * depth_factor)

    total_longitudinal = shape + viscous + wave
    # resistance always opposes the motion
    force_x = -np.sign(u) * total_longitudinal

    force_y = (-0.5 * water.density * coefficients.cross_flow_lateral
               * properties.lateral_area * v * abs(v))
    force_z = (-0.5 * water.density * coefficients.cross_flow_vertical
               * properties.plan_area * w * abs(w))

    breakdown = {
        "shape": shape,
        "viscous": viscous,
        "wave": wave,
        "total_longitudinal": total_longitudinal,
        "reynolds": reynolds,
        "friction_coefficient": c_f,
        "depth_factor": depth_factor,
        "depth_froude": float(shallow.froude(u)),
    }
    return np.array([force_x, force_y, force_z]), breakdown
