r"""The water the hull itself pushes about: bow pile-up and drawdown.

:mod:`coxswain.viz.water` draws the **radiated** wave system -- the
Kelvin wedge, which is what the hull leaves behind.  That is a far-field
description and it says nothing about what happens against the boat: the
heap of water standing at the stem, and the trough that runs along the
midbody where the flow has accelerated round the shoulder.  Those are
near-field, they come from the flow *around* the body rather than from
waves radiating away from it, and a coxswain looking over the bow sees
them more clearly than anything else on the water.

Where it comes from
-------------------
Thin-ship theory, the same idealisation :mod:`coxswain.hydro.michell`
uses for wave resistance: a slender hull is replaced by a sheet of
sources on its centreplane whose strength follows the rate at which the
hull is getting wider,

.. math::  \sigma(x, z) = \frac{U}{2\pi} \frac{\partial b}{\partial x}

with ``b`` the half-beam.  Where the hull swells the sheet pushes fluid
out; where it tapers it draws fluid in.

For the surface shape, the low-Froude ("double-body") approximation is
used: the free surface is treated as a rigid wall for the purpose of
computing the flow, which means every source gets an image reflected in
``z = 0``, and the elevation then follows from the linearised Bernoulli
condition,

.. math::  \eta = -\frac{U}{g}\,\phi_x \Big|_{z=0}

This is the standard near-field model and it is the right one here: it
captures stagnation at the stem -- flow slowed, pressure up, water piled
up -- and the acceleration along the midbody that pulls the surface
down.  What it does **not** contain is wave radiation, which is exactly
the part :mod:`coxswain.viz.water` already has from the Kelvin
construction.  The two are complementary and are added.

Why this can run in real time
-----------------------------
The source strengths are proportional to ``U`` and the elevation to
``U * phi_x``, so

.. math::  \eta(x, y; U) = \frac{U^2}{g} F(x, y)

where **F depends only on the shape of the hull**.  So the expensive part
-- a double sum over the centreplane panels for every point on a grid --
is done once, offline, and stored; at run time the surface is one texture
lookup multiplied by ``U^2 / g``.  A crew that lengthens and the boat
that runs faster get a bigger bow wave for free, and correctly, because
the scaling is exact rather than fitted.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

GRAVITY = 9.80665

#: Smoothing applied to the baked field, in grid cells.  See the note at
#: the end of :func:`geometric_field`: this is the resolution limit of
#: thin-ship theory at a fine bow, not a cosmetic blur.
SMOOTH_CELLS = 1.6

__all__ = ["geometric_field", "elevation", "bake"]


def _centreplane(offsets, stations: int = 81, levels: int = 21):
    """``(x, z, half_beam)`` on a regular centreplane grid."""
    from .michell import elliptical_offsets

    x, z, half = elliptical_offsets(offsets, stations=stations, levels=levels)
    return (np.asarray(x, dtype=float), np.asarray(z, dtype=float),
            np.asarray(half, dtype=float))


def geometric_field(offsets, east, north, stations: int = 81,
                    levels: int = 21) -> np.ndarray:
    r"""``F(x, y)`` such that the elevation is ``U^2 / g * F``.

    ``east`` runs along the hull (positive forward, origin amidships) and
    ``north`` across it.  Both are 1-D; the result has shape
    ``(len(north), len(east))``.

    The sum is over centreplane panels and their free-surface images.
    Nothing here depends on speed -- that is the point.
    """
    x, z, half = _centreplane(offsets, stations, levels)
    # **The flow runs from the bow, at +x, toward the stern.**
    #
    # Thin-ship theory is written with the stream along +x, so the source
    # strength is d(half-beam)/dx in *that* sense.  In this project +x is
    # the bow -- the coxswain of a bow-loaded four sits at +4.30 -- so the
    # oncoming flow is along -x and the sign flips.  Getting this wrong
    # puts the pile-up at the stern and a trough at the stem, which is
    # exactly backwards and is what the first run produced.
    slope = -np.gradient(half, x, axis=0)
    dx = float(np.mean(np.diff(x)))
    dz = float(np.mean(np.diff(z)))
    area = abs(dx * dz)
    # Desingularisation, and it has to be generous at the ends.
    #
    # Thin-ship theory assumes the hull is everywhere gently sloped, and
    # at a fine bow it is not: the waterline slope in the last two panels
    # of this four is 15 times the typical value, because that is where
    # the hull stops.  All the source strength therefore lands in a
    # couple of panels, and a 1/r^3 kernel evaluated close to them gives
    # jumps of half the peak between neighbouring grid cells -- visible
    # as spikes, and aliasing differently either side of the hull so that
    # a perfectly symmetric field can look one-sided.
    #
    # The floor is the larger of the panel diagonal and **the hull's own
    # half-beam**: the sheet stands for the hull surface, so no valid
    # field point is closer to it than the surface is.  That is the
    # honest statement of where this model stops resolving.
    core = 1.2 * float(np.hypot(dx, dz))
    surface_beam = np.max(half, axis=1)

    grid_e, grid_n = np.meshgrid(np.asarray(east, dtype=float),
                                 np.asarray(north, dtype=float))
    total = np.zeros(grid_e.shape)

    # phi_x from a source sheet, with the image in z = 0 doubling the
    # contribution of each panel (the rigid-wall approximation).
    for i, station in enumerate(x):
        row = slope[i]
        if not np.any(row):
            continue
        dxs = grid_e - station
        # Nothing may sit closer to this station's sources than the hull
        # surface there.
        keep_out = max(core, float(surface_beam[i]))
        for j, level in enumerate(z):
            strength = float(row[j])
            if strength == 0.0:
                continue
            # Field point on the surface (z = 0); source at depth `level`
            # and its image at -level give the same distance here, so the
            # image simply doubles the term.
            r2 = dxs * dxs + grid_n * grid_n + level * level
            r = np.sqrt(np.maximum(r2, keep_out * keep_out))
            # d/dx of (1/r) is -dx/r^3; the 2 is the image.
            total += 2.0 * strength * area * (-dxs / (r * r * r))
    field = -total / (4.0 * np.pi)

    # Smooth to the resolution the theory actually has.
    #
    # Thin-ship theory needs the hull slope to be small everywhere, and
    # at a fine stem it is 0.264 against a typical 0.018 -- the
    # assumption fails outright in the last panel or two, which is a
    # known singularity of Michell's model at fine ends.  Left raw, the
    # field jumps by half its peak between neighbouring cells there:
    # structure the model is not entitled to, and sharp enough to alias
    # differently on either side of a symmetric hull, which is how a
    # field that is symmetric to machine precision ends up looking as if
    # it leans to starboard.
    #
    # A Gaussian of about a panel width is the honest filter: it keeps
    # the pile-up and the drawdown, which are resolved, and drops the
    # tip detail, which is not.
    try:
        from scipy.ndimage import gaussian_filter

        field = gaussian_filter(field, sigma=SMOOTH_CELLS, mode="nearest")
    except Exception:                             # pragma: no cover
        pass
    return field


def elevation(field: np.ndarray, speed: float) -> np.ndarray:
    """Surface elevation, m, from the baked field at a given speed."""
    return field * (float(speed) ** 2) / GRAVITY


def bake(boat, reach_ahead: float = 6.0, reach_astern: float = 6.0,
         reach_side: float = 4.0, nx: int = 192,
         ny: int = 96) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(east, north, F)`` over a box around the hull.

    Defaults cover a boat length either way and four metres out, which is
    where the near field is worth anything; beyond that the Kelvin
    construction takes over and this has decayed to nothing.
    """
    half = 0.5 * float(boat.length)
    east = np.linspace(-half - reach_astern, half + reach_ahead, nx)
    north = np.linspace(-reach_side, reach_side, ny)
    return east, north, geometric_field(boat.offsets, east, north)
