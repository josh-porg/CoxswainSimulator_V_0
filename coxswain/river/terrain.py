"""The land around the river, from the federal elevation survey.

Until now the model's world ended at the water's edge: the bathymetry
says where the river is and how deep, and everything beyond the bank was
a flat shelf invented for the 3-D scene.  This brings in the actual
terrain -- USGS 3DEP, the national lidar-derived elevation model -- for
the rectangle around the racing reach.

Two things want it:

* **The 3-D scene.**  The Cambridge bank is a levee with trees, the
  Boston side carries Storrow Drive on fill, and the land climbs to 50 m
  within the frame.  Flat banks read as a diagram; real ones read as the
  river.
* **The wind, eventually.**  Wind shelter is terrain: the open basin
  against the treed upper reach, and every bridge embankment cutting a
  slot.  A microclimate model starts from exactly this grid.

Provenance
----------
``data/charles_dem.tif`` is an export from the USGS 3DEP Elevation
ImageServer (public domain), bounding box 42.3480-42.3790 N,
71.1000-71.1450 W, 900 x 620 pixels -- about 4 m per pixel, elevations in
metres NAVD88.  The Charles basin pool is held near 0.6 m NAVD88, so
water cells sit near zero and the banks rise from there.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

__all__ = ["Terrain", "charles_terrain", "seattle_terrain",
           "load_terrain", "pool_level_from", "Imagery",
           "seattle_imagery"]

#: Bounding box the stored DEM was exported with, (south, west, north, east).
DEM_BOUNDS = (42.3480, -71.1450, 42.3790, -71.1000)

#: Pool elevation of the Charles basin, metres NAVD88.  Land below this is
#: water (or the odd construction pit the lidar caught).
POOL_LEVEL = 0.6

#: Bounding box of ``seattle_dem.tif``, (south, west, north, east).
#:
#: Deliberately wider than the lake.  Lake Union ends at 47.62 N and the
#: downtown towers stand near 47.60, so a box drawn around the racing
#: water alone would put the skyline -- the thing a crew actually steers
#: by looking down the lake -- outside the terrain entirely.
SEATTLE_DEM_BOUNDS = (47.590, -122.375, 47.670, -122.300)

#: Pool elevation of Lake Union, metres NAVD88.
#:
#: **Measured off the tile, not converted.**  The Ship Canal holds Lake
#: Union between 20 and 22 feet above Puget Sound MLLW, and carrying that
#: through the Seattle tide-station datum into NAVD88 is exactly the kind
#: of two-step arithmetic that silently drops a metre.  Instead this is
#: the 5th percentile of the DEM over the OpenStreetMap lake polygon --
#: 5.10 m, against a median of 5.47 -- so it is the waterline the same
#: two datasets agree on rather than a number from a third source that
#: might be registered differently from either.
SEATTLE_POOL_LEVEL = 5.1

_CACHE = {}


def pool_level_from(elevation, quantile: float = 0.02) -> float:
    """Waterline of the flattest low ground in a tile, metres.

    An impounded lake is the largest dead-flat surface in any urban DEM,
    so its elevation is the low mode of the histogram.  Taking a low
    quantile rather than the minimum keeps the odd dredged pit or lidar
    dropout from dragging the waterline down with it.
    """
    values = np.asarray(elevation, dtype=float)
    values = values[np.isfinite(values) & (values > -50.0)]
    if not len(values):
        return 0.0
    return float(np.quantile(values, quantile))


class Terrain:
    """Elevation over the reach, in the model's tangent plane."""

    def __init__(self, elevation: np.ndarray, east: np.ndarray,
                 north: np.ndarray, pool: float = POOL_LEVEL):
        self.elevation = elevation
        self.east = east
        self.north = north
        #: Waterline for this tile, metres in the tile's own vertical
        #: datum.  Carried per-terrain because it is a fact about the
        #: water body, not about the class: the Charles basin sits at 0.6
        #: m NAVD88 and Lake Union, impounded 20 feet above Puget Sound,
        #: sits near 4.3.  Hard-coding the Charles' value put Lake Union
        #: nearly four metres under its own banks.
        self.pool = float(pool)

    def at(self, east, north, outside=None) -> np.ndarray:
        """Elevation in metres at tangent-plane coordinates.

        Points beyond the DEM's bounding box return ``outside`` (default
        the pool level) rather than the nearest border cell.  Clamping to
        the border silently painted the far ends of the 12 km centreline
        with whatever land happened to sit on the DEM's edge, which looked
        exactly like a bridge deck across the water until it was chased.
        """
        east = np.atleast_1d(np.asarray(east, dtype=float))
        north = np.atleast_1d(np.asarray(north, dtype=float))
        column = np.clip(np.searchsorted(self.east, east) - 1,
                         0, len(self.east) - 1)
        row = np.clip(np.searchsorted(self.north, north) - 1,
                      0, len(self.north) - 1)
        values = self.elevation[row, column].copy()
        beyond = ((east < self.east[0]) | (east > self.east[-1])
                  | (north < self.north[0]) | (north > self.north[-1]))
        values[beyond] = self.pool if outside is None else float(outside)
        return values

    def height_above_water(self, east, north) -> np.ndarray:
        return np.maximum(self.at(east, north) - self.pool, 0.0)


def load_terrain(name: str, bounds, origin, pool=None) -> "Terrain":
    """Load a stored 3DEP tile onto the tangent plane at ``origin``.

    ``pool`` may be a number, or ``None`` to measure the waterline off
    the tile with :func:`pool_level_from`.
    """
    from .course import local_tangent_plane

    key = (name, tuple(origin), pool)
    if key in _CACHE:
        return _CACHE[key]

    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s is missing -- run tools/fetch_dem.py --bounds %s --out %s"
            % (path, " ".join("%g" % b for b in bounds), name))

    fill = POOL_LEVEL if pool is None else float(pool)
    if path.endswith(".npz"):
        # Written by tools/fetch_dem.py: centimetres in int16, plus the
        # extent the service actually served.  That extent is the
        # authority -- ``exportImage`` moves the box to match the pixel
        # aspect and says so only in its JSON reply, and believing the
        # box that was *asked* for put Lake Union 2.2 km from itself.
        blob = np.load(path)
        raw = blob["elevation"]
        grid = raw.astype(float) * float(blob["scale"])
        grid[raw == blob["nodata"]] = fill
        bounds = tuple(float(v) for v in blob["bounds"])
    else:
        from PIL import Image
        grid = np.array(Image.open(path), dtype=float)
        # lidar no-data and water returns come back hugely negative
        grid = np.where(np.isfinite(grid) & (grid > -50.0), grid, fill)
    level = pool_level_from(grid) if pool is None else float(pool)

    south, west, north_edge, east_edge = bounds
    rows, columns = grid.shape
    latitudes = np.linspace(north_edge, south, rows)     # row 0 is north
    longitudes = np.linspace(west, east_edge, columns)

    east_axis, _ = local_tangent_plane(
        np.full(columns, 0.5 * (south + north_edge)), longitudes, origin)
    _, north_axis = local_tangent_plane(
        latitudes, np.full(rows, 0.5 * (west + east_edge)), origin)

    # flip so both axes ascend, which is what searchsorted needs
    terrain = Terrain(elevation=grid[::-1, :], east=np.asarray(east_axis),
                      north=np.asarray(north_axis)[::-1], pool=level)
    _CACHE[key] = terrain
    return terrain


def charles_terrain(origin: Tuple[float, float] = None) -> Terrain:
    """The stored Charles DEM, resampled onto the model's tangent plane."""
    from .charles import CHARLES_ORIGIN

    origin = CHARLES_ORIGIN if origin is None else origin
    return load_terrain("charles_dem.tif", DEM_BOUNDS, origin,
                        pool=POOL_LEVEL)


def seattle_terrain(origin: Tuple[float, float] = None) -> Terrain:
    """Lake Union, Queen Anne, Capitol Hill and downtown.

    Same product and same loader as the Charles -- a different tile and a
    different waterline, and nothing else.  The tile reaches south to the
    downtown towers on purpose: rowing down Lake Union you are looking
    straight at them, so they have to stand on ground the model knows
    about rather than float at zero.
    """
    from .seattle import SEATTLE_ORIGIN

    origin = SEATTLE_ORIGIN if origin is None else origin
    return load_terrain("seattle_dem.npz", SEATTLE_DEM_BOUNDS, origin,
                        pool=SEATTLE_POOL_LEVEL)


class Imagery:
    """An orthophoto and the ground it covers.

    Held as the image plus its bounding box in the tangent plane, so a
    renderer can turn a coordinate into a texture coordinate without
    knowing anything about projections.
    """

    def __init__(self, image, east, north):
        self.image = image
        #: ``(min, max)`` of the covered ground, metres.
        self.east = tuple(float(v) for v in east)
        self.north = tuple(float(v) for v in north)

    def texture_coordinates(self, east, north) -> np.ndarray:
        """``(u, v)`` in 0-1 for tangent-plane coordinates.

        ``v`` counts up from the south edge, which is what VTK wants and
        the opposite of the image's own row order -- the image is flipped
        once at load rather than here, so this stays a pure rescale.
        """
        u = ((np.asarray(east, dtype=float) - self.east[0])
             / max(self.east[1] - self.east[0], 1e-9))
        v = ((np.asarray(north, dtype=float) - self.north[0])
             / max(self.north[1] - self.north[0], 1e-9))
        return np.column_stack([np.clip(u, 0.0, 1.0).ravel(),
                                np.clip(v, 0.0, 1.0).ravel()])


    def sample(self, east, north) -> np.ndarray:
        """Colour of the ground at a point, RGB in 0-1.

        Used to paint building roofs their actual colour.  An orthophoto
        is a picture of roofs seen from directly above, so this is not an
        approximation of the roof colour -- it *is* the roof colour, for
        every one of the fifty thousand buildings in the scene, from a
        source that already had to be fetched for the ground.
        """
        rows, columns = self.image.shape[:2]
        u, v = self.texture_coordinates(east, north).T
        column = np.clip((u * (columns - 1)).astype(int), 0, columns - 1)
        # ``image`` was flipped at load so row 0 is the south edge, which
        # is exactly what ``v`` counts from.
        row = np.clip((v * (rows - 1)).astype(int), 0, rows - 1)
        return self.image[row, column].astype(float) / 255.0


def load_imagery(name: str, origin) -> "Imagery":
    """Load a stored orthophoto onto the tangent plane at ``origin``.

    The bounds come from the ``.json`` sidecar written by
    ``tools/fetch_imagery.py`` -- the extent the service actually served,
    not the one it was asked for.
    """
    from .course import local_tangent_plane

    key = ("imagery", name, tuple(origin))
    if key in _CACHE:
        return _CACHE[key]

    import json

    from PIL import Image
    folder = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")
    path = os.path.join(folder, name)
    sidecar = os.path.splitext(path)[0] + ".json"
    if not (os.path.exists(path) and os.path.exists(sidecar)):
        raise FileNotFoundError(
            "%s (or its .json) is missing -- run tools/fetch_imagery.py"
            % path)
    with open(sidecar) as handle:
        south, west, north, east = json.load(handle)["bounds"]

    # Row 0 of the file is the north edge; VTK's v axis counts up from
    # the south.  Flipping once here keeps every later transform a
    # straight rescale.
    image = np.asarray(Image.open(path).convert("RGB"))[::-1]

    east_axis, _ = local_tangent_plane(
        np.full(2, 0.5 * (south + north)), np.array([west, east]), origin)
    _, north_axis = local_tangent_plane(
        np.array([south, north]), np.full(2, 0.5 * (west + east)), origin)

    imagery = Imagery(image, east_axis, north_axis)
    _CACHE[key] = imagery
    return imagery


def seattle_imagery(origin=None) -> Imagery:
    """NAIP orthoimagery over Lake Union and downtown, public domain."""
    from .seattle import SEATTLE_ORIGIN

    origin = SEATTLE_ORIGIN if origin is None else origin
    return load_imagery("seattle_imagery.jpg", origin)
