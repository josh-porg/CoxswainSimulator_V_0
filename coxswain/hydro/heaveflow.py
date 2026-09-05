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
**The moment**, which is this module.  Local vertical velocity at station
``x`` is ``w - x q``, so a pitching hull drives its ends through the
water vertically even with no heave at all.  Integrating the sectional
load against ``x`` is where pitch damping comes from.  This is the same
Hoerner cross-flow argument already accepted for sway and yaw, applied to
the plane it was left out of.

**The linear term**, which is *not* this module.  Quadratic damping goes
as ``w|w|`` and so **vanishes faster than the energy going in** as the
amplitude falls, leaving small motions effectively undamped.  Real
damping at those amplitudes is wave radiation, which is linear in
velocity, and it lives in :mod:`coxswain.hydro.radiation` -- derived
there from potential flow for all six degrees of freedom at once, rather
than bolted onto the vertical plane here.  It was implemented in this
module first and then moved, because having the same physics in two
places is how this project has previously shipped a bug that could not
be fixed in one edit.

Consistency with what was there before
--------------------------------------
At zero pitch rate and zero radiation coefficient this reduces
**exactly** to the previous lumped force, because the integral of local
waterline beam along the length is the plan area.  A strict extension,
not a re-tuning.

References
----------
Hoerner, S.F. (1965) *Fluid-Dynamic Drag*, ch. 3 -- cross-flow drag.
See :mod:`coxswain.hydro.radiation` for the linear part.
"""

from __future__ import annotations

import numpy as np

__all__ = ["HeaveFlowHull", "heave_flow_load"]


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

    def load(self, heave_rate: float, pitch_rate: float, rho: float,
             drag_coefficient: float, immersion: float = 1.0):
        """``(force_z, moment_y)`` from distributed vertical damping.

        Sign convention matches the rest of the model: pitch rate ``q``
        is positive bow-up, so a station forward of the origin moves
        **down** at ``-x q`` when the bow rises.
        """
        beam = self.beam * float(immersion)
        local = float(heave_rate) - self.station * float(pitch_rate)

        sectional = (-0.5 * rho * float(drag_coefficient) * beam
                     * np.abs(local) * local)
        force_z = float(np.trapezoid(sectional, self.station))
        # Positive pitch is bow-up, and a downward force forward of the
        # origin pitches the bow down, hence the sign.
        moment_y = float(-np.trapezoid(self.station * sectional,
                                       self.station))
        return force_z, moment_y

def heave_flow_load(offsets, heave_rate: float, pitch_rate: float,
                    rho: float, drag_coefficient: float,
                    immersion: float = 1.0):
    """One-shot form of :meth:`HeaveFlowHull.load`."""
    return HeaveFlowHull(offsets).load(heave_rate, pitch_rate, rho,
                                       drag_coefficient, immersion)
