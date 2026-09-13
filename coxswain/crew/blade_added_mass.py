r"""Added mass of a rowing blade -- the water a blade drags with it.

A blade accelerating normal to its face accelerates water with it.  For a
plate moving normal to itself, potential flow gives an added mass in that
direction only.  For an aspect-ratio-2 rectangle in unbounded fluid, Patton
(1965), as quoted by Grift, Vijayaragavan, Tummers & Westerweel (2019) eq.
(3.3), [G19]::

    m_h = 0.84 (pi rho / 4) l_a l_b^2          l_a > l_b

The two-dimensional strip value it reduces towards is ``pi rho b^2 / 4`` per
unit span (Brouzet et al. 2018, citing Brennen 1982, [BR18]).

Sized on a Big Blade -- 0.52 x 0.25 m sweep, 0.43 x 0.215 m scull ([C2OAR],
[CR06]) -- this is about 21 kg and 13 kg.  At the blade centre that is about
113 and 42 kg m^2 about the pin: one of the largest inertias in the oar's
balance.

**An upper bound, for three stated reasons.**  The free surface is ignored,
though [G19] shows it changes both steady drag and entrainment.  Entrainment
growth is ignored: [G19] found the effective added mass grows with time
during an acceleration, with a constant matching only its start -- and tested
only to 1.64 m/s^2, where a blade at the catch sees roughly 16.  And the
blade is idealised as a rectangle.

The blade face widths live here, not on :class:`~coxswain.boats.rig.Oar`: the
recovery blade-contact model reads ``Oar.blade_width`` when it exists, and a
research-only number must not reach a path the shipped trainer can run.
"""

from __future__ import annotations

import numpy as np

#: Patton (1965) added-mass coefficient for an aspect-ratio-2 rectangular
#: plate accelerating normal to its face, unbounded inviscid fluid; quoted by
#: [G19] eq. (3.3).
PATTON_AR2 = 0.84

#: Big Blade face widths at the broadest point, metres ([C2OAR]): sweep 25 cm,
#: scull 21.5 cm.  With the rig's blade lengths (0.52 and 0.43 m) both are
#: aspect ratio 2, which is what Patton's coefficient is for.
BIG_BLADE_WIDTH = {"sweep": 0.25, "scull": 0.215}

#: Aspect ratios Patton's AR-2 coefficient is used across.  Outside it the
#: coefficient is not the right number and is refused rather than stretched.
ASPECT_RATIO_RANGE = (1.5, 2.5)


def patton_added_mass(length: float, width: float, density: float) -> float:
    """``0.84 (pi rho / 4) l_a l_b^2``, kilograms, for an AR-2 plate."""
    length, width, density = float(length), float(width), float(density)
    if length <= 0.0 or width <= 0.0 or density <= 0.0:
        raise ValueError("blade length, width and water density must be positive")
    aspect = length / width
    low, high = ASPECT_RATIO_RANGE
    if not low <= aspect <= high:
        raise ValueError(
            "Patton's coefficient is for an aspect-ratio-2 plate; this blade "
            "is %.2f (%.3f x %.3f m)" % (aspect, length, width))
    return PATTON_AR2 * np.pi * density / 4.0 * length * width ** 2
