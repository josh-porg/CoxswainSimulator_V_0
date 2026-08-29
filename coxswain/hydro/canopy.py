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
    scale = max(float(z0_upwind), float(z0_downwind), 1e-4)
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
