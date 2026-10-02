"""The blade going into the water at the catch: its wetted fraction and its entrained water.

A blade does not enter the water whole. BioRow's on-water norms (Kleshnev, "Visualisation of
Catch Factor and Blade Slip", row2k features/6267; ``data/literature/biorow_catch_norms.csv``)
put a clean catch in two parts:

* the blade's velocity relative to the water turns driving about 65 ms after the catch, and only
  then can "the bottom of the blade ... touch the water without backsplash" -- the instant
  [CR06]'s eq. 16 enters the blade at (normal velocity zero);
* it then takes "another 60-70ms or 6cm= 4deg of the oar movement to bury the blade completely"
  (target catch slip 6 deg), buried meaning a vertical oar angle of -3 deg, the blade's top edge
  at the surface (0 deg puts the blade centre at the water line; half a blade width is ~3 deg).

So the wetted fraction of the blade's height, eta, runs 0 -> 1 over ``bury_angle`` of oar travel
after entry (the shape between is linear in oar travel: a steady sink, the simplest reading of
"4 deg"; named and swept). Fully buried is exactly [CG07]'s flume condition (top edge at the
surface), so the quasi-steady load scales by eta and is otherwise the flume's.

The entrained water follows the wetted blade: Patton's AR-2 plate added mass goes as the square
of the plate's smaller dimension, which here is its height, so m_a = kappa * m_Patton * eta^2.
``surface_factor`` kappa discounts the unbounded-fluid value for a blade at the surface: Grift
(2020, ch. 2) measured the added-mass drop in force at 20-40% of the steady force with the top
edge at the surface against 50-70% submerged, a ratio 0.3-0.8 (about 0.5). The force is the rate
of change of the entrained momentum, F_a = -d(m_a w_n)/dt = -m_a w_n' - m_a' w_n: water picked
up by the entering blade costs momentum (von Karman's water-entry form).

Validated against populations (SOURCES sec. 173): with the immersion alone the model predicts
BioRow's catch-to-buried 6 deg (5.3-5.9) and the blade turning driving 44-56 ms after the catch
(norm ~65 ms), and keeps the force width in band. Any added mass on top spikes the handle once
the blade is buried; the population force curves bound ``surface_factor`` below about 0.1.

A research option for the hands-on-the-handle crew; nothing in the shipped game uses it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: BioRow's norm: oar travel from first wetting to fully buried, degrees
BIOROW_BURY_DEG = 4.0
#: Grift's surface / submerged added-mass response, (20-40%) / (50-70%)
GRIFT_SURFACE_RANGE = (0.3, 0.8)


@dataclass(frozen=True)
class EntryImmersion:
    """Wetted fraction and added-mass fraction against oar travel since entry."""

    #: oar travel from first wetting to fully buried, degrees
    bury_deg: float = BIOROW_BURY_DEG
    #: added mass at the surface relative to Patton's unbounded-fluid value
    surface_factor: float = 1.0

    def __post_init__(self):
        if not self.bury_deg > 0.0:
            raise ValueError("bury_deg must be positive, got %r" % (self.bury_deg,))
        if not self.surface_factor >= 0.0:
            raise ValueError("surface_factor must be non-negative, got %r" % (self.surface_factor,))

    @property
    def bury(self) -> float:
        return float(np.radians(self.bury_deg))

    def wetted(self, travel: float) -> float:
        """Wetted fraction of the blade's height after ``travel`` rad of oar since entry."""
        return float(min(max(travel / self.bury, 0.0), 1.0))

    def wetted_slope(self, travel: float) -> float:
        """d(wetted)/d(travel), 1/rad."""
        return 1.0 / self.bury if 0.0 < travel < self.bury else 0.0

    def mass_fraction(self, travel: float) -> float:
        """Added mass as a fraction of Patton's whole-blade value."""
        return self.surface_factor * self.wetted(travel) ** 2

    def mass_fraction_slope(self, travel: float) -> float:
        """d(mass_fraction)/d(travel), 1/rad."""
        return 2.0 * self.surface_factor * self.wetted(travel) * self.wetted_slope(travel)

    def key(self) -> tuple:
        return ("entry_immersion", float(self.bury_deg), float(self.surface_factor))
