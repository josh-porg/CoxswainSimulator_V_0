"""Buildings and tree canopy either side of the reach.

The stored DEM is bare earth: 3DEP strips the first returns, so the
roofs and the trees -- the only things on this reach tall enough to
shelter anything -- are exactly what it does not contain.  This module
carries them, from OpenStreetMap footprints fetched by
``tools/extract_charles_structures.py``.

Two consumers, one dataset
--------------------------
:mod:`coxswain.viz.river3d` extrudes the footprints so the coxswain view
shows the skyline a crew actually steers by.  :mod:`coxswain.hydro.canopy`
turns the same footprints into a frontal area index and a roughness
length.  It is worth getting the geometry once and honestly, because the
render makes errors in it immediately visible in a way the wind field
never would.

The heights are mostly inferred
-------------------------------
Of 9463 footprints, 44 carry a surveyed height and 2318 a storey count;
the remaining 7101 are typed guesses.  :attr:`Structures.height_source`
records which is which, so any downstream result can be recomputed
against the measured subset alone.  This matters more for the wind than
for the picture: roughness length scales with element height, so a
systematically wrong height is a systematically wrong shelter.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

__all__ = ["Structures", "charles_structures"]

_CACHE = {}
_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "charles_structures.npz")


class Structures:
    """Footprints in the model's tangent plane, with a coarse grid index.

    The grid is not decoration.  Every wind-field evaluation asks "what
    is within 300 m of here", and a linear scan over 9463 polygons per
    call makes the field unusable inside an optimiser.
    """

    #: Side of the lookup cell, m.  Comfortably larger than the biggest
    #: footprint so a polygon lands in few cells.
    CELL = 100.0

    def __init__(self, polygons, heights, sources, canopy, canopy_heights,
                 trees, tree_heights):
        self.polygons = polygons                  # list of (N, 2) east/north
        self.heights = np.asarray(heights, dtype=float)
        self.height_source = np.asarray(sources, dtype=int)
        self.canopy = canopy
        self.canopy_heights = np.asarray(canopy_heights, dtype=float)
        self.trees = np.asarray(trees, dtype=float).reshape(-1, 2)
        self.tree_heights = np.asarray(tree_heights, dtype=float)
        self.centres = np.array([p.mean(axis=0) for p in polygons]
                                if polygons else np.zeros((0, 2)))
        self._index = self._build_index()

    # -- spatial index ----------------------------------------------------
    def _build_index(self):
        index = {}
        for i, polygon in enumerate(self.polygons):
            low = np.floor(polygon.min(axis=0) / self.CELL).astype(int)
            high = np.floor(polygon.max(axis=0) / self.CELL).astype(int)
            for cx in range(low[0], high[0] + 1):
                for cy in range(low[1], high[1] + 1):
                    index.setdefault((cx, cy), []).append(i)
        return index

    def near(self, east: float, north: float, radius: float):
        """Indices of footprints whose centre is within ``radius``."""
        cells = int(np.ceil(radius / self.CELL))
        cx, cy = int(np.floor(east / self.CELL)), int(np.floor(north
                                                              / self.CELL))
        found = set()
        for i in range(cx - cells, cx + cells + 1):
            for j in range(cy - cells, cy + cells + 1):
                found.update(self._index.get((i, j), ()))
        if not found:
            return np.zeros(0, dtype=int)
        candidates = np.fromiter(found, dtype=int, count=len(found))
        offset = self.centres[candidates] - np.array([east, north])
        return candidates[np.einsum("ij,ij->i", offset, offset)
                          <= radius * radius]

    # -- geometry ---------------------------------------------------------
    def footprint_area(self, index) -> float:
        """Plan area of one footprint by the shoelace formula, m^2."""
        p = self.polygons[int(index)]
        return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], 1))
                               - np.dot(p[:, 1], np.roll(p[:, 0], 1))))

    def frontal_width(self, index, bearing: float) -> float:
        """Width of a footprint seen from ``bearing``, m.

        The extent of the footprint projected onto the axis across the
        wind.  This is the quantity Raupach's frontal area index needs
        and the reason the wind field here is direction-dependent: a
        terrace of houses is a wall to a crosswind and almost nothing to
        a wind along the street.
        """
        p = self.polygons[int(index)]
        across = np.array([-np.sin(bearing), np.cos(bearing)])
        projected = p @ across
        return float(projected.max() - projected.min())

    def canopy_area(self, index) -> float:
        p = self.canopy[int(index)]
        return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], 1))
                               - np.dot(p[:, 1], np.roll(p[:, 0], 1))))


def charles_structures(origin: Tuple[float, float] = None) -> Structures:
    """Load the stored footprints into the model's tangent plane."""
    from .charles import CHARLES_ORIGIN
    from .course import local_tangent_plane

    origin = CHARLES_ORIGIN if origin is None else origin
    if origin in _CACHE:
        return _CACHE[origin]
    if not os.path.exists(_PATH):
        raise FileNotFoundError(
            "%s is missing -- run tools/extract_charles_structures.py, "
            "which fetches it from OpenStreetMap" % _PATH)

    blob = np.load(_PATH)

    def split(xy, offsets):
        if len(xy) == 0:
            return []
        east, north = local_tangent_plane(xy[:, 0], xy[:, 1], origin)
        stacked = np.column_stack([np.asarray(east), np.asarray(north)])
        return [stacked[a:b] for a, b in zip(offsets[:-1], offsets[1:])]

    polygons = split(blob["building_xy"], blob["building_offsets"])
    canopy = split(blob["canopy_xy"], blob["canopy_offsets"])
    tree_xy = blob["tree_xy"]
    if len(tree_xy):
        east, north = local_tangent_plane(tree_xy[:, 0], tree_xy[:, 1], origin)
        trees = np.column_stack([np.asarray(east), np.asarray(north)])
    else:
        trees = np.zeros((0, 2))

    structures = Structures(
        polygons=polygons, heights=blob["building_height"],
        sources=blob["building_height_source"], canopy=canopy,
        canopy_heights=blob["canopy_height"], trees=trees,
        tree_heights=blob["tree_height"])
    _CACHE[origin] = structures
    return structures
