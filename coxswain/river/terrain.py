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

__all__ = ["Terrain", "charles_terrain"]

#: Bounding box the stored DEM was exported with, (south, west, north, east).
DEM_BOUNDS = (42.3480, -71.1450, 42.3790, -71.1000)

#: Pool elevation of the Charles basin, metres NAVD88.  Land below this is
#: water (or the odd construction pit the lidar caught).
POOL_LEVEL = 0.6

_CACHE = {}


class Terrain:
    """Elevation over the reach, in the model's tangent plane."""

    def __init__(self, elevation: np.ndarray, east: np.ndarray,
                 north: np.ndarray):
        self.elevation = elevation
        self.east = east
        self.north = north

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
        values[beyond] = POOL_LEVEL if outside is None else float(outside)
        return values

    def height_above_water(self, east, north) -> np.ndarray:
        return np.maximum(self.at(east, north) - POOL_LEVEL, 0.0)


def charles_terrain(origin: Tuple[float, float] = None) -> Terrain:
    """The stored DEM, resampled onto the model's tangent plane."""
    from .charles import CHARLES_ORIGIN
    from .course import local_tangent_plane

    origin = CHARLES_ORIGIN if origin is None else origin
    if origin in _CACHE:
        return _CACHE[origin]

    from PIL import Image
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "charles_dem.tif")
    grid = np.array(Image.open(path), dtype=float)
    # lidar no-data and water returns come back hugely negative
    grid = np.where(np.isfinite(grid) & (grid > -50.0), grid, POOL_LEVEL)

    south, west, north_edge, east_edge = DEM_BOUNDS
    rows, columns = grid.shape
    latitudes = np.linspace(north_edge, south, rows)     # row 0 is north
    longitudes = np.linspace(west, east_edge, columns)

    east_axis, _ = local_tangent_plane(
        np.full(columns, 0.5 * (south + north_edge)), longitudes, origin)
    _, north_axis = local_tangent_plane(
        latitudes, np.full(rows, 0.5 * (west + east_edge)), origin)

    # flip so both axes ascend, which is what searchsorted needs
    terrain = Terrain(elevation=grid[::-1, :], east=np.asarray(east_axis),
                      north=np.asarray(north_axis)[::-1])
    _CACHE[origin] = terrain
    return terrain
