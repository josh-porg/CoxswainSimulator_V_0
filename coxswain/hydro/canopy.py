r"""Wind over a rough bank and onto a smooth river.

The aerodynamic force model in :mod:`coxswain.hydro.wind` is calibrated
and validated; what it has never had is a wind *field*.  Until now
:class:`~coxswain.hydro.wind.UniformWind` was the only one, which puts
the same vector everywhere on the reach and makes the Charles a wind
tunnel.  It is not one.  It is a 150 m channel of open water with three
storeys of Cambridge on one bank, Boston on the other, and a line of
mature trees along both -- and a boat rowing 1.5 m above the surface sits
well inside the layer that is still adjusting to all of it.

This module builds that field in the two steps the boundary-layer
literature already separates.

Step one: how rough is the bank
-------------------------------
:func:`raupach_roughness` gives the roughness length ``z0`` and
zero-plane displacement ``d`` of an array of obstacles from just two
numbers -- their height ``h`` and their **frontal area index**
``lambda_f``, the frontal area they present per unit ground area
[R94]_.  Frontal area, not plan area, is what makes the answer depend on
wind direction: a terrace is a wall to a crosswind and almost nothing to
a wind along the street.

The formulation, with Raupach's constants:

.. math::

    \frac{d}{h} &= 1 - \frac{1 - \exp(-\sqrt{2 c_{d1} \lambda})}
                            {\sqrt{2 c_{d1} \lambda}} \\
    \frac{u_*}{U_h} &= \min\left(\sqrt{C_S + C_R \lambda},
                                 \left(\tfrac{u_*}{U}\right)_{max}\right) \\
    \frac{z_0}{h} &= \left(1 - \frac{d}{h}\right)
                     \exp\left(-\kappa \frac{U_h}{u_*} + \psi_h\right)

**Where it is valid.**  Raupach fitted these to ``lambda_f`` up to about
0.2 and they are not to be trusted far beyond it.  ``z0/h`` peaks near
``lambda_f = 0.2`` and falls after, which is real -- densely packed
obstacles shelter each other and the flow skims over the tops -- but the
falling branch is much less well constrained than the rising one.
:func:`macdonald_roughness` is the standard alternative for dense urban
arrays and is provided as a cross-check rather than as a competitor.

Step two: the wind does not arrive adjusted
-------------------------------------------
This is the step that matters for a rowing shell and the one a single
roughness length cannot express.  Air crossing from a rough bank onto
smooth water does not instantly acquire the water's profile: a new
**internal boundary layer** grows from the shoreline, and only inside it
has the flow adjusted to the water.  Above it the air still carries the
bank's profile.

A crew 40 m off the bank at 1.5 m height is deep inside that IBL, in air
that has already adjusted to the water; a crew mid-channel is under a
thicker one.  :func:`internal_boundary_layer` gives its depth from the
standard power law [E58]_, [G90]_, and :func:`sheltered_speed` builds
the profile in two pieces matched at the IBL top.

**The sign of this surprises people, including the author.**  Going from
a rough bank to smooth water, near-surface wind *increases* with fetch:
the retarding surface has fallen away and the air near the water speeds
up.  What a lee bank buys a crew is the short fetch close in, not
distance from the weather.  Sailors have always known this as the lee
shore being soft and the breeze filling in further out; the shelter is
local and it runs out fast.

Comparing that to "the same wind over open water" needs care, and
:func:`open_water_equivalent` is the control that makes it honest: 6 m/s
at 10 m over a suburb is a much windier day aloft than 6 m/s at 10 m over
water, so holding the 10 m reading fixed compares two different days.
The fair control holds the wind at a blending height above both.

The classic experiments behind all of this are Bradley's [B68]_
smooth-to-rough and rough-to-smooth step changes, and :mod:`scripts.canopy`
reproduces the published roughness classes before any of it is pointed at
the Charles.

References
----------
.. [R94] Raupach, M. R. (1994) *Simplified expressions for vegetation
   roughness length and zero-plane displacement as functions of canopy
   height and area index*, Boundary-Layer Meteorology 71, 211-216.
.. [M98] Macdonald, R. W., Griffiths, R. F. and Hall, D. J. (1998) *An
   improved method for the estimation of surface roughness of obstacle
   arrays*, Atmospheric Environment 32(11), 1857-1864.
.. [GO99] Grimmond, C. S. B. and Oke, T. R. (1999) *Aerodynamic
   properties of urban areas derived from analysis of surface form*,
   J. Applied Meteorology 38, 1262-1292.
.. [E58] Elliott, W. P. (1958) *The growth of the atmospheric internal
   boundary layer*, Trans. American Geophysical Union 39, 1048-1054.
.. [B68] Bradley, E. F. (1968) *A micrometeorological study of velocity
   profiles and surface drag in the region modified by a change in
   surface roughness*, Q. J. Royal Met. Society 94, 361-379.
.. [G90] Garratt, J. R. (1990) *The internal boundary layer -- a review*,
   Boundary-Layer Meteorology 50, 171-203.
.. [W92] Wieringa, J. (1992) *Updating the Davenport roughness
   classification*, J. Wind Engineering and Industrial Aerodynamics
   41, 357-368.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["raupach_roughness", "macdonald_roughness",
           "internal_boundary_layer", "sheltered_speed",
           "open_water_equivalent", "regional_speed",
           "DAVENPORT", "Roughness", "BLENDING_HEIGHT"]

KARMAN = 0.40

#: Raupach's constants, [R94]_ table 1.
CD1 = 7.5              # drag coefficient in the displacement expression
C_S = 0.003            # substrate-surface stress coefficient
C_R = 0.3              # roughness-element drag coefficient
USTAR_MAX = 0.3        # cap on u*/U(h); dense arrays stop gaining stress
PSI_H = 0.193          # roughness-sublayer influence function

#: Macdonald's constants for a staggered array, [M98]_.
MACDONALD_A = 4.43
MACDONALD_CD = 1.2
MACDONALD_BETA = 1.0

#: The Davenport classes as revised by Wieringa [W92]_, metres.  These
#: are the numbers any morphometric method has to reproduce before it is
#: allowed near a real site.
DAVENPORT = {
    "sea": 0.0002,
    "smooth": 0.005,
    "open": 0.03,
    "roughly open": 0.10,
    "rough": 0.25,
    "very rough": 0.5,
    "closed": 1.0,
    "chaotic": 2.0,
}


@dataclass(frozen=True)
class Roughness:
    """An aerodynamic surface: roughness length and displacement, m."""

    z0: float
    d: float
    height: float
    frontal_index: float

    def speed_at(self, height: float, friction_velocity: float) -> float:
        """Log-law wind speed at ``height`` above the ground, m/s."""
        z = max(float(height) - self.d, 1.05 * self.z0)
        return friction_velocity / KARMAN * np.log(z / self.z0)


def raupach_roughness(frontal_index: float, height: float) -> Roughness:
    """Roughness length and displacement of an obstacle array, [R94]_.

    ``frontal_index`` is the frontal area of the elements per unit
    ground area; ``height`` their mean height in metres.
    """
    lam = max(float(frontal_index), 1e-6)
    h = float(height)

    root = np.sqrt(2.0 * CD1 * lam)
    d_over_h = 1.0 - (1.0 - np.exp(-root)) / root
    ustar_ratio = min(np.sqrt(C_S + C_R * lam), USTAR_MAX)
    z0_over_h = (1.0 - d_over_h) * np.exp(-KARMAN / ustar_ratio + PSI_H)
    return Roughness(z0=float(z0_over_h * h), d=float(d_over_h * h),
                     height=h, frontal_index=lam)


def macdonald_roughness(plan_index: float, frontal_index: float,
                        height: float) -> Roughness:
    """The morphometric alternative, [M98]_ -- a cross-check, not a rival.

    Needs the *plan* area index as well as the frontal one, so it is only
    usable where both are known.  Included because Raupach's falling
    branch above ``lambda_f = 0.2`` is the weakest part of this module
    and it is worth seeing whether a second method agrees there.
    """
    lam_p = float(np.clip(plan_index, 1e-6, 0.95))
    lam_f = max(float(frontal_index), 1e-6)
    h = float(height)
    d_over_h = 1.0 + MACDONALD_A ** (-lam_p) * (lam_p - 1.0)
    z0_over_h = ((1.0 - d_over_h)
                 * np.exp(-(0.5 * MACDONALD_BETA * MACDONALD_CD / KARMAN ** 2
                            * (1.0 - d_over_h) * lam_f) ** -0.5))
    return Roughness(z0=float(z0_over_h * h), d=float(d_over_h * h),
                     height=h, frontal_index=lam_f)


def internal_boundary_layer(fetch, z0_upwind: float, z0_downwind: float):
    """Depth of the layer adjusted to the new surface, m.

    Elliott's power law as usually written [E58]_, [G90]_::

        delta / z0' = A (x / z0')^0.8

    with ``z0'`` the larger of the two roughness lengths and ``A`` near
    0.75 for a rough-to-smooth change.  Rough-to-smooth transitions --
    which is what a bank onto a river is -- grow their IBL more slowly
    than smooth-to-rough, and the coefficient reflects that.

    The number that matters here: with a 0.5 m bank and 60 m of fetch,
    the adjusted layer is only a few metres deep.  A rower's chest at
    1.5 m is inside it, an anemometer on a boathouse roof is not, and
    that is why the two disagree.
    """
    fetch = np.maximum(np.asarray(fetch, dtype=float), 0.1)
    # The **downwind** roughness sets the scale, not the larger of the
    # two.  This function's own paragraph above says the transition here
    # is rough-to-smooth -- a bank onto water -- and for that direction
    # the layer grows into the new, smoother surface and scales on its
    # roughness.  ``max`` is the smooth-to-rough convention and it was
    # the one implemented.
    #
    # It was not a small error.  With a wooded bank at z0 = 1.5 m it put
    # the internal layer 78 m deep against a blending height of 80,
    # which collapses the two-layer profile onto a single one and makes
    # the sheltered wind come out **higher** the rougher the bank is:
    # 13.3 m/s behind a hedge and 14.1 m/s behind a wood. Shelter ran
    # backwards, and nothing noticed until 470,000 trees were added to
    # the bank and the wind went up. On the downwind scaling the same
    # case gives 12.2 and 9.0 m/s, and the layer is 13 m deep.
    scale = max(float(z0_downwind), 1e-4)
    return 0.75 * scale * (fetch / scale) ** 0.8


#: Roughness of the standard open exposure a forecast wind refers to.
#: A "10 m wind" is not a wind measured 10 m above whatever happens to be
#: underneath -- it is standardised to open short-grass terrain.
Z0_OPEN = 0.03

#: Height above every canopy on the reach where the flow has forgotten
#: the individual surfaces, m.  Urban meteorology's blending height.
BLENDING_HEIGHT = 80.0


def regional_speed(reference_speed: float, reference_height: float = 10.0,
                   blending: float = BLENDING_HEIGHT) -> float:
    """Wind at the blending height from a forecast 10 m wind, m/s.

    Everything downstream is driven from here rather than from the
    forecast directly, and the reason is a bug worth recording: fed a
    32 m canopy, the first version evaluated the log law at 10 m, which
    is *below that canopy's displacement height*.  There is no wind
    profile there -- the expression is not merely inaccurate, it is
    undefined -- and it returned 232 m/s without complaint.

    A forecast 10 m wind refers to a standard open exposure, so that is
    what it is converted from.
    """
    z = max(float(reference_height), 2.0)
    friction = KARMAN * float(reference_speed) / np.log(z / Z0_OPEN)
    return float(friction / KARMAN * np.log(blending / Z0_OPEN))


def sheltered_speed(height, fetch, reference_speed: float,
                    upwind: Roughness, z0_water: float = 2.0e-4,
                    reference_height: float = 10.0,
                    blending: float = BLENDING_HEIGHT):
    """Wind speed over the water at ``height``, ``fetch`` from the bank.

    The profile is built in the two pieces the roughness-change problem
    gives, driven from the blending height so it stays defined over a
    canopy of any height:

    * **Above the IBL** the air is still the bank's, so the bank's
      friction velocity and roughness apply.
    * **Inside** it has adjusted to the water, matched to the bank's
      profile at the IBL top so there is no blending fudge.

    Near-surface wind **increases with fetch**, which reads as wrong for
    a moment and is not: rough-to-smooth means the retarding surface has
    fallen away and the air near the water accelerates.  What a lee bank
    buys a crew is the short fetch close in, not distance in general.
    """
    height = np.maximum(np.asarray(height, dtype=float), 1.05 * z0_water)
    delta = np.maximum(internal_boundary_layer(fetch, upwind.z0, z0_water),
                       2.0 * float(np.max(height)))
    aloft = regional_speed(reference_speed, reference_height, blending)

    # The bank's own profile, anchored at the blending height.  Clamped so
    # a tall canopy cannot push the anchor inside itself.
    span = max(blending - upwind.d, 3.0 * upwind.z0)
    friction = KARMAN * aloft / np.log(span / upwind.z0)

    at_top = friction / KARMAN * np.log(
        np.maximum(delta, 1.05 * upwind.z0) / upwind.z0)
    inside = at_top * (np.log(height / z0_water)
                       / np.log(np.maximum(delta, 2.0 * z0_water) / z0_water))
    above = friction / KARMAN * np.log(
        np.maximum(height, 1.05 * upwind.z0) / upwind.z0)
    return np.where(height <= delta, inside, above)


def open_water_equivalent(height, reference_speed: float,
                          upwind: Roughness = None,
                          z0_water: float = 2.0e-4,
                          reference_height: float = 10.0,
                          blending: float = BLENDING_HEIGHT):
    """The same weather with no bank at all, m/s.

    The control that makes "sheltered" mean something.  Holding the 10 m
    wind fixed between a rough bank and open water compares two different
    days; holding the blending-height wind fixed compares the same day
    over two surfaces, which is the question.
    """
    aloft = regional_speed(reference_speed, reference_height, blending)
    height = np.maximum(np.asarray(height, dtype=float), 1.05 * z0_water)
    return aloft * np.log(height / z0_water) / np.log(blending / z0_water)


#: How much of a tree's silhouette actually blocks the wind.
#:
#: A building is solid and a canopy is not: measured optical porosity of
#: a leafed broadleaf crown runs 0.2-0.5, so roughly half to four fifths
#: of the frontal area is doing something.  0.5 is the middle, and it is
#: a **parameter, not a result** -- this project has no measurement of it
#: for the species on this bank, and a winter crew rows past bare
#: branches at maybe half this.
CANOPY_POROSITY = 0.5


class ShelteredWind:
    """A wind field over the reach, from the banks either side of it.

    Implements the :class:`~coxswain.hydro.wind.WindField` interface, so
    it drops into :class:`~coxswain.hydro.wind.AeroModel` in place of
    :class:`~coxswain.hydro.wind.UniformWind` without the force model
    knowing anything changed.

    Three things vary with position, and all three are the point:

    **The upwind bank.**  Frontal area index is computed over the upwind
    half-sector only, so the same buildings give a different roughness
    for a westerly and a northerly.

    **The fetch.**  Distance from the boat to the upwind shoreline,
    marched over the channel raster.  This is what makes the near bank
    genuinely sheltered and mid-channel much less so.

    **The height.**  Everything is evaluated at the crew's chest, not at
    an anemometer's 10 m, because those are different air.

    Speeds are cached on a grid.  The simulator asks for the wind at every
    derivative evaluation, and re-marching a fetch and re-summing forty
    footprints per call makes the field unusable inside an optimiser.
    """

    #: Side of the cache cell, m.  Wind varies over hundreds of metres
    #: here, so 25 m is finer than the field itself.
    CACHE = 25.0
    #: Stop marching upwind after this far; beyond it the boat is not
    #: sheltered by anything and the open-water profile applies.
    MAX_FETCH = 500.0

    def __init__(self, structures, channel, reference_speed: float,
                 wind_from: float, radius: float = 250.0,
                 height: float = 1.5, reference_height: float = 10.0,
                 trees=None):
        self.structures = structures
        #: Optional :class:`~coxswain.river.structures.TreeStand`.
        #:
        #: Trees were left out of the roughness entirely, which is
        #: defensible on a city bank and wrong on a wooded one.  This
        #: module's own docstring says shelter on the Powerhouse Stretch
        #: is "three storeys of Cambridge **and a line of plane trees on
        #: the bank**", and only the first half was in the sum.  Lake
        #: Union has 470,000 mapped trees on its banks, up to 39 m.
        self.trees = trees
        self.channel = channel
        self.reference_speed = float(reference_speed)
        #: Meteorological bearing the wind comes *from*, degrees.
        self.wind_from = float(wind_from)
        self.radius = float(radius)
        self.height = float(height)
        self.reference_height = float(reference_height)
        # Direction the wind blows *towards*, in the model's maths frame.
        self.bearing = np.radians(90.0 - (self.wind_from + 180.0))
        self._towards = np.array([np.cos(self.bearing), np.sin(self.bearing)])
        self._cache = {}

    # -- geometry ---------------------------------------------------------
    def fetch(self, east: float, north: float) -> float:
        """Metres of open water upwind of a point."""
        step = max(self.channel.resolution, 2.0)
        here = np.array([east, north], dtype=float)
        travelled = 0.0
        while travelled < self.MAX_FETCH:
            here = here - self._towards * step
            travelled += step
            row, column = self.channel.index_of(here[0], here[1])
            if not bool(self.channel.water[row, column]):
                break
        return travelled

    def roughness(self, east: float, north: float) -> Roughness:
        """Raupach roughness of the upwind bank seen from a point.

        Buildings and, where a stand is supplied, trees.  A tree is
        charged its crown as a frontal area -- ``0.6 * height`` wide,
        which is the crown width a broadleaf of a given height reaches --
        and it is a **porous** obstacle, so it is discounted by
        :data:`CANOPY_POROSITY`.  A wall stops the wind; a canopy slows
        it and lets some through.
        """
        index = self.structures.near(east, north, self.radius)
        areas, heights = [], []
        if len(index):
            centres = self.structures.centres[index] - np.array([east, north])
            upwind = np.einsum("ij,j->i", centres, -self._towards) > 0.0
            index = index[upwind] if upwind.any() else index
            for i in index:
                areas.append(self.structures.frontal_width(i, self.bearing)
                             * self.structures.heights[i])
                heights.append(float(self.structures.heights[i]))

        if self.trees is not None and len(self.trees):
            found = self.trees.near(east, north, self.radius)
            if len(found):
                offset = self.trees.points[found] - np.array([east, north])
                upwind = np.einsum("ij,j->i", offset, -self._towards) > 0.0
                found = found[upwind] if upwind.any() else found
                crown = self.trees.heights[found].astype(float)
                areas.extend((CANOPY_POROSITY * 0.6 * crown * crown).tolist())
                heights.extend(crown.tolist())

        if not heights:
            return Roughness(z0=Z0_OPEN, d=0.0, height=1.0,
                             frontal_index=1e-6)
        areas = np.asarray(areas, dtype=float)
        heights = np.asarray(heights, dtype=float)
        ground = 0.5 * np.pi * self.radius ** 2

        # Canopy height weighted by **frontal area**, not by count.
        #
        # Raupach's formulation is for a canopy of uniform height h, and
        # h sets the displacement height -- how far up the profile is
        # pushed.  Averaging a mixed bank by count lets four hundred
        # thousand nine-metre trees outvote the buildings and drag the
        # effective canopy from 14.5 m to 9.3 m, which *lowers* the
        # displacement height and comes out as the wind getting stronger
        # when trees are added.  What sets the displacement is where the
        # drag is, and the drag is the frontal area.
        weight = areas.sum()
        height = (float(np.average(heights, weights=areas)) if weight > 0
                  else float(heights.mean()))
        return raupach_roughness(weight / ground, height)

    # -- the interface ----------------------------------------------------
    def speed_at(self, east: float, north: float) -> float:
        key = (int(east // self.CACHE), int(north // self.CACHE))
        hit = self._cache.get(key)
        if hit is None:
            bank = self.roughness(east, north)
            hit = float(sheltered_speed(self.height, self.fetch(east, north),
                                        self.reference_speed, bank,
                                        reference_height=self.reference_height))
            self._cache[key] = hit
        return hit

    def at(self, x, y, t=0.0):
        return np.array([self.speed_at(float(x), float(y)) * self._towards[0],
                         self.speed_at(float(x), float(y)) * self._towards[1],
                         0.0])
