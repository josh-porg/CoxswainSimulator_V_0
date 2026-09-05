"""Buildings and tree canopy either side of the reach.

The stored DEM is bare earth: 3DEP strips the first returns, so the
roofs and the trees -- the only things on this reach tall enough to
shelter anything -- are exactly what it does not contain.  This module
carries them, from OpenStreetMap footprints fetched by
``tools/extract_structures.py``.

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

__all__ = ["Structures", "charles_structures",
           "seattle_structures", "load_structures"]

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

    #: Wall material classes, indexed by :attr:`material`.  Mirrors
    #: ``tools/extract_structures.py``; kept as a class rather than a
    #: colour because what brick looks like is the renderer's business.
    MATERIALS = ("unknown", "brick", "concrete", "glass", "metal", "wood",
                 "stone", "plaster", "tile")
    #: Roof shapes, indexed by :attr:`roof_shape`.
    ROOF_SHAPES = ("flat", "gabled", "hipped", "pyramidal", "skillion",
                   "dome", "round", "mansard", "gambrel", "half-hipped")
    #: Building classes, indexed by :attr:`kind`.
    KINDS = ("other", "house", "apartments", "commercial", "office",
             "industrial", "civic", "boathouse", "houseboat", "roof",
             "garage", "retail")

    def __init__(self, polygons, heights, sources, canopy, canopy_heights,
                 trees, tree_heights, material=None, kind=None, colour=None,
                 roof_shape=None, roof_height=None, names=None,
                 spans=(), span_names=(), water=(), base=None):
        self.polygons = polygons                  # list of (N, 2) east/north
        self.heights = np.asarray(heights, dtype=float)
        self.height_source = np.asarray(sources, dtype=int)
        count = len(polygons)

        def column(values, fill, dtype=int):
            if values is None or len(values) != count:
                return np.full(count, fill, dtype=dtype)
            return np.asarray(values, dtype=dtype)

        #: Index into :attr:`MATERIALS`.  Almost always ``0`` -- only 1%
        #: of Seattle carries ``building:material`` -- which is a fact
        #: about OpenStreetMap, not a placeholder to be filled in.  Any
        #: renderer leaning on this must have an unknown case that looks
        #: deliberate, because that is the common case.
        self.material = column(material, 0, np.int8)
        #: Index into :attr:`KINDS`.  Well populated, because it comes
        #: from ``building=*`` itself rather than an optional extra.
        self.kind = column(kind, 0, np.int8)
        #: Index into :attr:`ROOF_SHAPES`.
        self.roof_shape = column(roof_shape, 0, np.int8)
        #: Roof height in metres above the wall top; ``0`` for flat.
        self.roof_height = column(roof_height, 0.0, float)
        #: Height above ground at which the footprint *starts*, metres.
        #:
        #: Non-zero only for OpenStreetMap ``building:part`` geometry
        #: carrying ``min_height`` -- the massing steps of a tower.  The
        #: Space Needle's saucer starts at 150 m, and extruding it from
        #: the ground like everything else is what rendered the Needle as
        #: a 160 m cylinder.
        self.base = column(base, 0.0, float)
        #: Tagged wall colour as RGB in 0-1, or ``-1`` where untagged.
        if colour is None or len(colour) != count:
            self.colour = np.full((count, 3), -1.0)
        else:
            self.colour = np.asarray(colour, dtype=float).reshape(-1, 3)
        self.names = (np.asarray(names, dtype="<U48") if names is not None
                      and len(names) == count
                      else np.full(count, "", dtype="<U48"))
        #: Named bridge decks as ``(N, 2)`` east/north polylines, and
        #: their names.  Scenery, not gates: a landmark bridge is one a
        #: crew steers *by*, and it carries no piers, arches or rules.
        self.spans = list(spans)
        self.span_names = list(span_names)
        #: Water polygons for scenery only -- every wet thing in the
        #: extract box, not the surveyed racing shoreline.  See
        #: ``tools/extract_structures.py`` for why the two are separate.
        self.water = list(water)
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

    origin = CHARLES_ORIGIN if origin is None else origin
    return load_structures("charles_structures.npz", origin)


def seattle_structures(origin: Tuple[float, float] = None) -> Structures:
    """Lake Union's buildings, canopy and trees.

    Same extractor, same file format, different bounding box -- the
    footprints are stored as latitude and longitude and projected at load
    time, so nothing about the Charles is baked into them.
    """
    from .seattle import SEATTLE_ORIGIN

    origin = SEATTLE_ORIGIN if origin is None else origin
    return load_structures("seattle_structures.npz", origin)


def load_structures(name: str, origin: Tuple[float, float]) -> Structures:
    """Load a stored footprint set into a tangent plane at ``origin``."""
    from .course import local_tangent_plane

    path = os.path.join(os.path.dirname(_PATH), name)
    key = (name, origin)
    if key in _CACHE:
        return _CACHE[key]
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s is missing -- run tools/extract_structures.py "
            "with --out %s, which fetches it from OpenStreetMap"
            % (path, name))

    blob = np.load(path)

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

    def optional(key):
        """Fields added after the first extracts were stored.

        A stored file predating them is not corrupt, it is older, and it
        should keep working -- so a missing field becomes the default
        rather than an exception.
        """
        return blob[key] if key in blob.files else None

    structures = Structures(
        polygons=polygons, heights=blob["building_height"],
        sources=blob["building_height_source"], canopy=canopy,
        canopy_heights=blob["canopy_height"], trees=trees,
        tree_heights=blob["tree_height"],
        material=optional("building_material"),
        kind=optional("building_kind"),
        colour=optional("building_colour"),
        roof_shape=optional("building_roof_shape"),
        roof_height=optional("building_roof_height"),
        names=optional("building_name"),
        base=optional("building_base"),
        spans=split(blob["bridge_xy"], blob["bridge_offsets"])
        if "bridge_xy" in blob.files else [],
        span_names=list(optional("bridge_name")
                        if optional("bridge_name") is not None else []),
        water=split(blob["water_xy"], blob["water_offsets"])
        if "water_xy" in blob.files else [])
    _CACHE[key] = structures
    return structures
