"""Distributed vertical damping on the hull, and the pitch damping it gives.

The exact defect :mod:`coxswain.hydro.crossflow` fixed for yaw, still
present in the vertical plane.  Heave resistance was a **single lumped
force at the origin**:

    Z = -0.5 rho Cd A_plan w |w|

A force at the origin exerts no moment about the origin, so the hull had
**no pitch damping at all**.  Every bit of what little there was came
from the appendages, and they are small and near the centreline.

What that costs
---------------
A coxed four at rate 30 went unstable.  Not resonance -- the response was
at 0.867 Hz, which is no harmonic of the 0.5 Hz stroke, so it was the
boat's own coupled heave/pitch mode growing on its own.  Pitch rms went
from 0.7 degrees to 5 degrees in sixty seconds and reached **25 degrees**
over a race leg, with the hull riding 1.5 m clear of its own waterline.
A shell pitches a fraction of a degree.

It is rate-dependent and sharp: rates 30 and 32 diverge, 18, 22, 26, 28
and 36 do not.  A resonance that narrow is itself the symptom -- a real
hull's pitch response is broad because it is well damped.

Two things were missing, and they are separate
----------------------------------------------
**The moment.** Local vertical velocity at station ``x`` is ``w - x q``,
so a pitching hull drives its ends through the water vertically even with
no heave at all.  Integrating the sectional load against ``x`` is where
pitch damping comes from.  This is the same Hoerner cross-flow argument
already accepted for sway and yaw, applied to the plane it was left out
of.

**The linear term.** Quadratic damping is proportional to ``w|w|``, which
**vanishes faster than the energy going in** as the amplitude falls.  So
a small oscillation is effectively undamped and grows until the quadratic
term finally catches it -- which is exactly the observed behaviour, a
slow build to a violent limit cycle.  Real heave and pitch damping at
these amplitudes is dominated by **wave radiation**, which is linear in
velocity.  Without a linear term no amount of quadratic damping makes
small motions decay at the right rate.

Consistency with what was there before
--------------------------------------
At zero pitch rate and zero radiation coefficient this reduces
**exactly** to the previous lumped force, because the integral of local
waterline beam along the length is the plan area.  A strict extension,
not a re-tuning.

The radiation coefficient, honestly
-----------------------------------
Sectional wave-radiation damping is :math:`b_{33} = \\rho g^2 \\bar{A}^2 /
\\omega^3` per unit length, where :math:`\\bar{A}` is the amplitude of the
radiated wave per unit heave amplitude [N78]_.  That is frequency
dependent, and a time-domain model cannot carry a frequency-dependent
coefficient without a convolution, so it is evaluated **once at the
hull's own heave natural frequency** -- the standard constant-coefficient
simplification, and the frequency that matters here because that is the
mode going unstable.

:data:`AMPLITUDE_RATIO` is the weak number.  For a wall-sided section at
:math:`\\omega^2 B / 2g` near 0.4, which is where a shell sits, published
values run about 0.4-0.7 [F78]_.  This project has no measurement of it
for a racing shell, so it is a **parameter, not a result**: quote a band,
and see :func:`damping_ratio` for what any choice implies.

References
----------
.. [N78] Newman, J. N. (1977) *Marine Hydrodynamics*, MIT Press, ch. 6 --
   the radiation damping/wave amplitude relation.
.. [F78] Faltinsen, O. M. (1990) *Sea Loads on Ships and Offshore
   Structures*, Cambridge, ch. 3 -- sectional added mass and damping for
   wall-sided and Lewis-form sections.
"""

from __future__ import annotations

import numpy as np

__all__ = ["HeaveFlowHull", "heave_flow_load", "AMPLITUDE_RATIO",
           "damping_ratio"]

GRAVITY = 9.80665

#: Radiated wave amplitude per unit heave amplitude, non-dimensional.
#:
#: See the module docstring: a parameter, not a result.  0.55 is the
#: middle of the published band for a wall-sided section at the reduced
#: frequency a rowing shell oscillates at.
AMPLITUDE_RATIO = 0.55


class HeaveFlowHull:
    """Station table for the vertical strip integral, built once per boat."""

    def __init__(self, offsets, n_strips: int = 41):
        x = np.asarray(offsets.station, dtype=float)
        beam = np.asarray(offsets.beam, dtype=float)
        self.station = np.linspace(float(x[0]), float(x[-1]), int(n_strips))
        self.beam = np.interp(self.station, x, beam)
        #: Integral of waterline beam -- the plan area the lumped force
        #: used, so the two agree at zero pitch rate by construction.
        self.plan_area = float(np.trapezoid(self.beam, self.station))

    def radiation_coefficient(self, frequency: float, rho: float,
                              amplitude_ratio: float = AMPLITUDE_RATIO):
        """Sectional linear damping ``b_33`` per unit length, N s/m^2.

        ``frequency`` is in rad/s.  Returns zero for a non-positive
        frequency rather than dividing by it, so a caller that has no
        estimate simply gets the quadratic model back.
        """
        if frequency <= 0.0 or amplitude_ratio <= 0.0:
            return np.zeros_like(self.beam)
        # b33 = rho g^2 A^2 / omega^3, per unit length, scaled by the
        # local beam relative to the maximum: a fine section radiates
        # less than the midship one.
        peak = max(float(self.beam.max()), 1e-9)
        return (rho * GRAVITY ** 2 * amplitude_ratio ** 2 / frequency ** 3
                * (self.beam / peak))

    def load(self, heave_rate: float, pitch_rate: float, rho: float,
             drag_coefficient: float, immersion: float = 1.0,
             frequency: float = 0.0,
             amplitude_ratio: float = AMPLITUDE_RATIO):
        """``(force_z, moment_y)`` from distributed vertical damping.

        Sign convention matches the rest of the model: pitch rate ``q``
        is positive bow-up, so a station forward of the origin moves
        **down** at ``-x q`` when the bow rises.
        """
        beam = self.beam * float(immersion)
        local = float(heave_rate) - self.station * float(pitch_rate)

        sectional = (-0.5 * rho * float(drag_coefficient) * beam
                     * np.abs(local) * local)
        if frequency > 0.0:
            sectional = sectional - (
                self.radiation_coefficient(frequency, rho, amplitude_ratio)
                * float(immersion) * local)

        force_z = float(np.trapezoid(sectional, self.station))
        # Positive pitch is bow-up, and a downward force forward of the
        # origin pitches the bow down, hence the sign.
        moment_y = float(-np.trapezoid(self.station * sectional,
                                       self.station))
        return force_z, moment_y

    def damping_ratio(self, mass: float, frequency: float, rho: float,
                      amplitude_ratio: float = AMPLITUDE_RATIO) -> float:
        """Fraction of critical damping the **linear** term alone gives.

        The number to quote when asked how much of this is invented.
        """
        if frequency <= 0.0 or mass <= 0.0:
            return 0.0
        total = float(np.trapezoid(
            self.radiation_coefficient(frequency, rho, amplitude_ratio),
            self.station))
        return total / (2.0 * mass * frequency)


def heave_flow_load(offsets, heave_rate: float, pitch_rate: float,
                    rho: float, drag_coefficient: float,
                    immersion: float = 1.0, frequency: float = 0.0,
                    amplitude_ratio: float = AMPLITUDE_RATIO):
    """One-shot form of :meth:`HeaveFlowHull.load`."""
    return HeaveFlowHull(offsets).load(
        heave_rate, pitch_rate, rho, drag_coefficient, immersion,
        frequency, amplitude_ratio)


def damping_ratio(offsets, mass: float, frequency: float, rho: float,
                  amplitude_ratio: float = AMPLITUDE_RATIO) -> float:
    """One-shot form of :meth:`HeaveFlowHull.damping_ratio`."""
    return HeaveFlowHull(offsets).damping_ratio(mass, frequency, rho,
                                                amplitude_ratio)
