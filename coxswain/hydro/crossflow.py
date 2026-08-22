"""Distributed cross-flow drag on the hull, and the yaw damping it gives.

A hull moving sideways sheds vortices along its length.  The standard
manoeuvring treatment is Hoerner's cross-flow drag: each transverse strip
is charged the drag of a two-dimensional bluff section at the **local**
lateral velocity, and the sectional loads are integrated for the sway
force and the yaw moment.

Why this matters here
---------------------
The lateral force was previously computed from the uniform sideslip
alone, and the hull's yaw moment was set to zero outright -- every bit of
yaw damping came from the skeg and the rudder.  That is survivable while
the hull has no Munk moment either, because two large opposing terms are
both missing.  Add the Munk moment on its own (see
:mod:`coxswain.hydro.addedmass`) and the eight broaches: 3 degrees off
course becomes 285 degrees inside ten strokes.

The Munk moment is real, and so is what opposes it.  A slender body in
ideal flow *is* directionally unstable -- that is the classical result --
and real hulls are held straight by viscous cross-flow damping plus their
appendages.  Implementing either one alone gives an unbalanced model.

The local lateral velocity at station ``x`` is ``v + x r``, so a yawing
hull drags its ends through the water sideways even with no sideslip at
all.  That is where the yaw damping comes from, and it scales like
``r |r|``.

Consistency with what was there before
--------------------------------------
At zero yaw rate this reduces **exactly** to the previous lumped force:

    Y = -0.5 rho Cd |v| v * int T(x) dx  =  -0.5 rho Cd |v| v * A_lateral

because the integral of local draft along the length is the lateral
area.  So this is a strict extension, not a re-tuning.

References
----------
Hoerner, S.F. (1965) *Fluid-Dynamic Drag*, ch. 3.
Fossen, T.I. (2011) *Handbook of Marine Craft Hydrodynamics and Motion
Control*, Wiley, sec. 6.4.
"""
from __future__ import annotations

import numpy as np

__all__ = ["cross_flow_load", "CrossFlowHull"]


class CrossFlowHull:
    """Station table for the cross-flow integral, built once per boat."""

    def __init__(self, offsets, n_strips: int = 41):
        x = np.asarray(offsets.station, dtype=float)
        depth = np.asarray(offsets.depth, dtype=float)
        # resample uniformly so the trapezoid weights are simple
        self.station = np.linspace(float(x[0]), float(x[-1]), int(n_strips))
        self.draft = np.interp(self.station, x, depth)
        self.lateral_area = float(np.trapezoid(self.draft, self.station))

    def load(self, sway: float, yaw_rate: float, rho: float,
             drag_coefficient: float, immersion: float = 1.0):
        """Sway force and yaw moment from distributed cross-flow drag.

        ``immersion`` scales the draft table to the instantaneous
        waterline; 1.0 is the design condition.
        """
        draft = self.draft * float(immersion)
        local = float(sway) + self.station * float(yaw_rate)
        sectional = (-0.5 * rho * float(drag_coefficient) * draft
                     * np.abs(local) * local)
        force_y = float(np.trapezoid(sectional, self.station))
        moment_z = float(np.trapezoid(self.station * sectional, self.station))
        return force_y, moment_z


def cross_flow_load(offsets, sway: float, yaw_rate: float, rho: float,
                    drag_coefficient: float, immersion: float = 1.0):
    """One-shot form of :meth:`CrossFlowHull.load`."""
    return CrossFlowHull(offsets).load(sway, yaw_rate, rho,
                                       drag_coefficient, immersion)
