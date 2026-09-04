r"""Fetch-limited chop on a walled basin, and what it costs a shell.

The Charles is a river: long, narrow, and banked. Lake Union is a **walled
box**, and that changes the sea state qualitatively rather than by degree.
Crews who race Tail of the Lake describe water far rougher than the wind
alone suggests, and at least one shell has been broken in two there. This
module is about why, and what it costs.

Three mechanisms, and only the first is ordinary
-----------------------------------------------
**Fetch-limited growth.** Wind blowing across open water builds waves until
either the fetch or the duration runs out. On a 2 km lake the fetch runs
out first, and the JONSWAP fetch-limited relations [H73]_ give the height
and period directly from wind speed and fetch:

.. math::

    \frac{gH_s}{U^2} = 0.0016 \left(\frac{gF}{U^2}\right)^{1/2},
    \qquad
    \frac{gT_p}{U} = 0.286 \left(\frac{gF}{U^2}\right)^{1/3}

The consequence for a lake is not that the waves are big. It is that they
are **short**. Ten metres per second over 2 km gives ``H_s`` about 0.23 m
and ``T_p`` about 1.7 s -- a wavelength near 4.5 m, so a 13.4 m four spans
three of them. That is the worst possible ratio for pitching: the hull
cannot ride the wave and cannot ignore it either.

**Reflection.** A shingle beach absorbs; a **vertical bulkhead reflects**.
Lake Union is ringed with seawalls, docks and moored houseboats, and a
reflection coefficient of 0.7-0.9 is normal for such a wall [G92]_.
Incident and reflected trains superpose into a partial standing wave whose
antinodes reach :math:`(1 + K_r)` times the incident height -- close to
**double** within a wavelength or two of the wall.

**Crossing seas.** Reflections off *opposite* shores meet in the middle.
Wave energy is additive, so two trains of height ``H`` give
:math:`H\sqrt2`, and the surface becomes short-crested and confused rather
than a regular swell. This is the "concentrated" chop crews describe, and
it is a property of the basin's shape, not of the weather.

What it costs
-------------
Added resistance in waves goes as the **square** of wave height, which is
why chop is punishing out of proportion to how it looks. For a slender
hull in waves short compared with its length the standard scaling is

.. math::

    R_{aw} \sim \frac{1}{2} \rho g H_s^2 \frac{B^2}{L}

with a coefficient of order one that depends on the hull and the
encounter frequency [F78]_, [SM86]_.

The honest limit of this
------------------------
The largest cost of rough water for a rowing crew is almost certainly
**not** hydrodynamic. It is that eight or four people row worse: catches
get checked, timing goes, the boat will not sit level, and blades come out
half-buried. That is a coupling to the crew, and this module does not
model it -- see :mod:`coxswain.hydro.wake`, which makes the same
disclaimer for the same reason. What is here is the hull's share, and it
should be read as a floor on the true cost.

References
----------
.. [H73] Hasselmann, K. et al. (1973) *Measurements of wind-wave growth
   and swell decay during the Joint North Sea Wave Project (JONSWAP)*,
   Deutsche Hydrographische Zeitschrift A8(12).
.. [G92] Goda, Y. (1992) *Random Seas and Design of Maritime Structures*,
   University of Tokyo Press -- reflection coefficients for vertical and
   composite walls.
.. [F78] Faltinsen, O. M. et al. (1980) *Prediction of resistance and
   propulsion of a ship in a seaway*, 13th ONR Symposium.
.. [SM86] Salvesen, N. (1978) *Added resistance of ships in waves*,
   J. Hydronautics 12(1), 24-34.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FetchLimitedSea", "WalledBasin", "added_resistance"]

GRAVITY = 9.80665
WATER_DENSITY = 1000.0


@dataclass(frozen=True)
class FetchLimitedSea:
    """Significant height and peak period from wind speed and fetch."""

    #: Wind speed at 10 m, m/s.
    wind: float
    #: Fetch, m.
    fetch: float
    #: JONSWAP growth coefficients.  Not tuned here; these are the
    #: published values and they are what makes this a prediction rather
    #: than a curve fit.
    height_coefficient: float = 0.0016
    period_coefficient: float = 0.286

    @property
    def _scaled_fetch(self) -> float:
        return GRAVITY * self.fetch / max(self.wind ** 2, 1e-9)

    @property
    def significant_height(self) -> float:
        """``H_s``, m."""
        if self.wind <= 0.0 or self.fetch <= 0.0:
            return 0.0
        return float(self.height_coefficient * self.wind ** 2 / GRAVITY
                     * np.sqrt(self._scaled_fetch))

    @property
    def peak_period(self) -> float:
        """``T_p``, s."""
        if self.wind <= 0.0 or self.fetch <= 0.0:
            return 0.0
        return float(self.period_coefficient * self.wind / GRAVITY
                     * self._scaled_fetch ** (1.0 / 3.0))

    @property
    def wavelength(self) -> float:
        """Deep-water wavelength at the peak period, m."""
        return float(GRAVITY * self.peak_period ** 2 / (2.0 * np.pi))

    @property
    def steepness(self) -> float:
        """``H_s / L``.  Above about 1/7 a wave breaks."""
        return self.significant_height / max(self.wavelength, 1e-9)


@dataclass(frozen=True)
class WalledBasin:
    """What vertical shores do to a fetch-limited sea.

    ``reflection`` is the amplitude coefficient of the shore.  A shingle
    beach is near 0.1, a rubble mound 0.3-0.5, a **vertical concrete
    bulkhead 0.7-0.9** -- and Lake Union is bulkheads, docks and moored
    hulls almost all the way round.
    """

    sea: FetchLimitedSea
    reflection: float = 0.8
    #: How many wall-facing directions contribute.  One for a single
    #: seawall; two for a narrow basin with walls both sides, which is
    #: what makes a lake confused rather than merely rough.
    walls: int = 2

    def height_near_wall(self, distance) -> np.ndarray:
        """Local ``H_s`` a given distance off a reflecting shore, m.

        Within a wavelength or so of the wall the incident and reflected
        trains are in phase often enough to reach :math:`(1+K_r) H_s`.
        Further out the phase relationship randomises and the two trains
        add as **energy** rather than amplitude, which is the
        :math:`\\sqrt{1+K_r^2}` floor.
        """
        distance = np.maximum(np.asarray(distance, float), 0.0)
        near = np.exp(-distance / max(self.sea.wavelength * 2.0, 1e-9))
        coherent = 1.0 + self.reflection
        incoherent = np.sqrt(1.0 + self.reflection ** 2)
        factor = incoherent + (coherent - incoherent) * near
        return self.sea.significant_height * factor

    @property
    def open_water_height(self) -> float:
        """``H_s`` in mid-basin, where trains from every wall meet.

        Energy adds, so ``n`` independent reflected trains on top of the
        incident one give ``sqrt(1 + n K_r^2)``.  This is the number that
        explains why the middle of a small lake can be rougher than the
        fetch alone predicts.
        """
        return float(self.sea.significant_height
                     * np.sqrt(1.0 + self.walls * self.reflection ** 2))


def added_resistance(height, beam: float, length: float,
                     coefficient: float = 1.0,
                     density: float = WATER_DENSITY) -> np.ndarray:
    """Added resistance in short waves, N.

    .. math::

        R_{aw} = C \\cdot \\tfrac{1}{2} \\rho g H_s^2 B^2 / L

    The **square** is the point: doubling the chop quadruples the cost,
    which is why a lake that looks merely ruffled is so much slower than
    flat water.

    ``coefficient`` is order one and hull-dependent, and this project has
    no measurement of it for a racing shell.  It is therefore a
    **parameter, not a result**: quote a band, not a number, and treat any
    single value as indicative.
    """
    height = np.asarray(height, dtype=float)
    return (coefficient * 0.5 * density * GRAVITY * height ** 2
            * beam ** 2 / max(length, 1e-9))
