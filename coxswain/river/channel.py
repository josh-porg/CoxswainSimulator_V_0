"""Extracting the navigable channel from digitised depth contours.

What the survey actually is
---------------------------
The CRAB / MIT Sea Grant data is **not** scattered soundings.  It contains
34 distinct depth values spaced 0.3048 m apart -- one-foot isobath
contours, digitised as polyline vertices.  Treating those vertices as
independent depth measurements is wrong in a way that matters:

* the 0.30 m (1 ft) contour is effectively the **shoreline**, and 12% of
  the vertices lie on it;
* a Delaunay interpolation over the vertices fills its own convex hull,
  which spans land wherever the river bends, so asking for the depth in
  the middle of Magazine Beach returns a plausible number;
* nothing in the data says where the water stops.

That is why a coarse centreline could sit 26% of its length in water
shallower than 1.2 m while nominally being "in the channel": the channel
was a fixed 55 m ribbon that had no relationship to the bathymetry.

What this module does
---------------------
1. Reconstructs the **water body** as an alpha shape (concave hull) of the
   contour vertices.  Delaunay triangles whose circumradius exceeds
   ``alpha`` span gaps rather than water and are discarded, so the result
   follows the river instead of bridging its bends.
2. Rasterises that region and interpolates depth inside it only.  Outside
   is land, and reports as such rather than extrapolating.
3. Takes the **navigable** subset -- deeper than a threshold, and
   connected to the main channel, so isolated deep pockets behind a shoal
   do not count.
4. Derives the centreline and the half-width from a distance transform of
   that mask, rather than assuming either.

The centreline is the path of greatest clearance, found by dynamic
programming across the raster.  That is the same machinery the route
optimiser uses, one dimension down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "DEFAULT_ALPHA",
    "attach_current",
    "NAVIGABLE_DEPTH",
    "alpha_shape_mask",
    "ChannelRaster",
    "build_channel",
]

#: Circumradius above which a Delaunay triangle is taken to span land
#: rather than water, in metres.  Contour vertices are digitised every
#: 10-30 m, so a triangle much larger than that is bridging a gap.
DEFAULT_ALPHA = 70.0

#: Water shallower than this is not navigable for a racing shell.  The
#: contour interval is one foot, and 4 ft is the shallowest contour that
#: reliably brackets the channel rather than the bank shelf.
NAVIGABLE_DEPTH = 1.22


def alpha_shape_mask(points: np.ndarray, query: np.ndarray,
                     alpha: float = DEFAULT_ALPHA):
    """Which ``query`` points lie inside the alpha shape of ``points``.

    An alpha shape is a concave hull: the Delaunay triangulation with the
    "fat" triangles removed.  A triangle whose circumcircle is larger than
    ``alpha`` cannot be filled by the sampling density of the input, so it
    is taken to be a gap -- here, land between two bends of the river.

    Returns a boolean array over ``query``.
    """
    from scipy.spatial import Delaunay

    triangulation = Delaunay(points)
    simplices = triangulation.simplices
    corners = points[simplices]

    a = np.linalg.norm(corners[:, 0] - corners[:, 1], axis=1)
    b = np.linalg.norm(corners[:, 1] - corners[:, 2], axis=1)
    c = np.linalg.norm(corners[:, 2] - corners[:, 0], axis=1)
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 1e-12))
    circumradius = a * b * c / (4.0 * area)

    keep = circumradius < alpha
    found = triangulation.find_simplex(query)
    return (found >= 0) & keep[np.clip(found, 0, None)]


@dataclass
class ChannelRaster:
    """The navigable channel, on a regular grid.

    Attributes
    ----------
    east, north:
        1-D grid coordinates in metres.
    water, navigable:
        Boolean rasters, shape ``(len(north), len(east))``.
    depth:
        Depth raster in metres; ``nan`` outside :attr:`water`.
    clearance:
        Distance in metres from each navigable cell to the nearest
        non-navigable one -- the half-width available at that point.
    """

    east: np.ndarray
    north: np.ndarray
    water: np.ndarray
    navigable: np.ndarray
    depth: np.ndarray
    clearance: np.ndarray
    #: Depth-averaged water velocity on the same grid, in m/s, or ``None``
    #: for still water.  Populated by :func:`attach_current`; kept on the
    #: raster rather than passed separately so that :meth:`crop` carries it
    #: along and the trajectory solver cannot silently lose it.
    current_east: np.ndarray = None
    current_north: np.ndarray = None
    #: Distance to the nearest dry cell, in metres.  Distinct from
    #: :attr:`clearance`, which measures to the nearest *non-navigable*
    #: cell.  The boat is bounded by the navigable width; the water is not,
    #: and continuity has to integrate over the whole wetted section.
    water_clearance: np.ndarray = None

    @property
    def resolution(self) -> float:
        return float(self.east[1] - self.east[0])

    @property
    def water_area(self) -> float:
        return float(self.water.sum()) * self.resolution ** 2

    @property
    def navigable_area(self) -> float:
        return float(self.navigable.sum()) * self.resolution ** 2

    def index_of(self, x, y):
        """Nearest grid indices ``(row, column)`` for a position."""
        column = np.clip(np.searchsorted(self.east, x), 0, len(self.east) - 1)
        row = np.clip(np.searchsorted(self.north, y), 0, len(self.north) - 1)
        return row, column

    def is_navigable(self, x, y) -> bool:
        row, column = self.index_of(x, y)
        return bool(self.navigable[row, column])

    def clearance_at(self, x, y) -> float:
        row, column = self.index_of(x, y)
        return float(self.clearance[row, column])

    # -- centreline -------------------------------------------------------
    def centreline(self, smooth: int = 9, clearance_weight: float = 1.0,
                   start=None, end=None) -> np.ndarray:
        """Path of greatest clearance between the two ends of the reach.

        Found as a least-cost path on the navigable mask, with edge cost

            cost = length / clearance ** clearance_weight

        so the path is drawn towards wide water and away from pinch points,
        while still being penalised for wandering.

        A directional sweep will not do here.  Column-by-column dynamic
        programming can only follow a river that is monotone in the sweep
        direction; the Charles doubles back, and the sweep then has to
        cross many rows within one column and fails.  That is the same
        flaw that made the original east-west-binned thalweg wander onto
        the shoals.  A geodesic path respects the actual connectivity of
        the water whichever way it runs.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        navigable = self.navigable
        rows, columns = navigable.shape
        index = -np.ones(navigable.shape, dtype=np.int64)
        cells = np.flatnonzero(navigable.ravel())
        index.ravel()[cells] = np.arange(len(cells))

        clearance = np.maximum(self.clearance, self.resolution)
        weight = 1.0 / clearance ** clearance_weight

        sources, targets, costs = [], [], []
        offsets = ((0, 1), (1, 0), (1, 1), (1, -1))
        for row_step, column_step in offsets:
            source = navigable[max(0, -row_step):rows - max(0, row_step),
                               max(0, -column_step):columns - max(0, column_step)]
            target = navigable[max(0, row_step):rows - max(0, -row_step),
                               max(0, column_step):columns - max(0, -column_step)]
            both = source & target
            if not both.any():
                continue
            a = index[max(0, -row_step):rows - max(0, row_step),
                      max(0, -column_step):columns - max(0, column_step)][both]
            b = index[max(0, row_step):rows - max(0, -row_step),
                      max(0, column_step):columns - max(0, -column_step)][both]
            length = self.resolution * np.hypot(row_step, column_step)
            w_a = weight[max(0, -row_step):rows - max(0, row_step),
                         max(0, -column_step):columns - max(0, column_step)][both]
            w_b = weight[max(0, row_step):rows - max(0, -row_step),
                         max(0, column_step):columns - max(0, -column_step)][both]
            edge = length * 0.5 * (w_a + w_b)
            sources.append(a)
            targets.append(b)
            costs.append(edge)

        source = np.concatenate(sources)
        target = np.concatenate(targets)
        cost = np.concatenate(costs)
        graph = coo_matrix((cost, (source, target)),
                           shape=(len(cells), len(cells))).tocsr()

        cell_rows, cell_columns = np.unravel_index(cells, navigable.shape)
        cell_clearance = self.clearance.ravel()[cells]

        def widest_in_end_column(extreme_column):
            """Middle of the channel mouth at one end of the reach.

            Neither ``argmin`` over the column index nor maximum clearance
            works here.  The first lands on whichever edge cell happens to
            come first; the second is misled because the distance transform
            treats the edge of the surveyed raster as a wall, so every cell
            in the end column reads as narrow and the choice is arbitrary.
            Both put the start of the line against a bank.

            Take the midpoint of the longest unbroken run of navigable
            cells in that column instead -- the centre of the opening.
            """
            band = np.flatnonzero(cell_columns == extreme_column)
            rows = np.sort(cell_rows[band])
            breaks = np.flatnonzero(np.diff(rows) > 1)
            starts = np.concatenate([[0], breaks + 1])
            ends = np.concatenate([breaks, [len(rows) - 1]])
            longest = int(np.argmax(ends - starts))
            middle_row = rows[(starts[longest] + ends[longest]) // 2]
            same_row = band[cell_rows[band] == middle_row]
            return int(same_row[0])

        if start is None:
            start = widest_in_end_column(cell_columns.min())
        if end is None:
            end = widest_in_end_column(cell_columns.max())

        _, predecessors = dijkstra(graph, directed=False, indices=start,
                                   return_predecessors=True)

        path = [end]
        while path[-1] != start:
            previous = predecessors[path[-1]]
            if previous < 0:
                raise ValueError(
                    "no navigable path across the reach; the mask is "
                    "disconnected between its ends"
                )
            path.append(int(previous))
        path = np.array(path[::-1])

        line = np.column_stack([self.east[cell_columns[path]],
                                self.north[cell_rows[path]]])

        if smooth > 1 and len(line) > 2 * smooth:
            kernel = np.ones(smooth) / smooth
            for axis in (0, 1):
                blurred = np.convolve(line[:, axis], kernel, mode="same")
                blurred[:smooth] = line[:smooth, axis]
                blurred[-smooth:] = line[-smooth:, axis]
                line[:, axis] = blurred
        return line

    def crop(self, points: np.ndarray, margin: float = 150.0
             ) -> "ChannelRaster":
        """A sub-raster covering ``points`` plus a margin.

        Trajectory optimisation builds a spline interpolant over whichever
        raster it is handed, and the coefficient count is the cell count.
        Over the full reach that is 728 000 coefficients evaluated at every
        node and midpoint of every solver iteration; cropping to the leg
        being solved cuts it by one to two orders of magnitude.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        east_lo, east_hi = points[:, 0].min() - margin, points[:, 0].max() + margin
        north_lo, north_hi = points[:, 1].min() - margin, points[:, 1].max() + margin

        columns = np.flatnonzero((self.east >= east_lo) & (self.east <= east_hi))
        rows = np.flatnonzero((self.north >= north_lo) & (self.north <= north_hi))
        if len(columns) < 2 or len(rows) < 2:
            raise ValueError("crop region does not overlap the raster")

        window = np.ix_(rows, columns)
        return ChannelRaster(
            east=self.east[columns], north=self.north[rows],
            water=self.water[window], navigable=self.navigable[window],
            depth=self.depth[window], clearance=self.clearance[window],
            current_east=(None if self.current_east is None
                          else self.current_east[window]),
            current_north=(None if self.current_north is None
                           else self.current_north[window]),
            water_clearance=(None if self.water_clearance is None
                             else self.water_clearance[window]),
        )

    @property
    def has_current(self) -> bool:
        return self.current_east is not None

    def current_at(self, x, y):
        """Water velocity ``(east, north)`` at a position, in m/s."""
        if not self.has_current:
            return np.zeros(2)
        row, column = self.index_of(x, y)
        return np.array([self.current_east[row, column],
                         self.current_north[row, column]])

    def half_width_along(self, line: np.ndarray,
                         cap: float = 60.0) -> np.ndarray:
        """Navigable half-width at each point of a centreline."""
        widths = np.array([self.clearance_at(x, y) for x, y in line])
        return np.clip(widths, 1.0, cap)

    def water_half_width_along(self, line: np.ndarray,
                               cap: float = 150.0) -> np.ndarray:
        """Half-width of *water* at each point of a centreline.

        Wider than :meth:`half_width_along` wherever shallow margins
        flank the channel.  Continuity needs this one: integrating the
        cross-sectional area over the navigable width alone understates
        ``A`` and so overstates ``Q/A``.  At the tightest pinch on the
        Charles that was the difference between 2.25 m/s and 0.75 m/s.
        """
        if self.water_clearance is None:
            return self.half_width_along(line, cap=cap)
        widths = np.array([
            float(self.water_clearance[self.index_of(x, y)])
            for x, y in line])
        return np.clip(widths, 1.0, cap)


def attach_current(raster: "ChannelRaster", flow, course,
                   n_centreline: int = 4000) -> "ChannelRaster":
    """Sample a flow field onto the raster grid.

    The trajectory solver needs the current as a gridded, differentiable
    lookup, but a :class:`~coxswain.river.course.CurrentField` is a
    callable that does a nearest-point search against the centreline for
    every query.  Calling it per cell would take minutes.

    Instead: build a KD-tree over a densely resampled centreline, query
    every water cell at once to get station and signed offset, then read
    the flow off its own ``(station, offset-fraction)`` grid.  Vectorised,
    so the whole raster costs about a second.

    Land cells get zero, which is harmless -- the trajectory is
    constrained to navigable water anyway.
    """
    from scipy.spatial import cKDTree

    station = np.linspace(0.0, course.length, n_centreline)
    centre = course.position_at(station)
    heading = course.heading_at(station)
    tree = cKDTree(centre)

    grid_east, grid_north = np.meshgrid(raster.east, raster.north)
    inside = raster.water
    query = np.column_stack([grid_east[inside], grid_north[inside]])
    _, nearest = tree.query(query)

    local_heading = heading[nearest]
    normal = np.column_stack([-np.sin(local_heading), np.cos(local_heading)])
    offset = np.einsum("ij,ij->i", query - centre[nearest], normal)

    half = np.array([max(course.half_width_at(s), 1e-6)
                     for s in station])[nearest]
    fraction = np.clip(offset / half, -1.0, 1.0)

    stations, fractions, speed_grid = flow._speed_grid(120)
    columns = np.clip(np.searchsorted(fractions, fraction) - 1,
                      0, len(fractions) - 2)
    weight = ((fraction - fractions[columns])
              / (fractions[columns + 1] - fractions[columns]))
    query_station = station[nearest]
    low = np.array([np.interp(s, stations, speed_grid[:, c])
                    for s, c in zip(query_station, columns)])
    high = np.array([np.interp(s, stations, speed_grid[:, c + 1])
                     for s, c in zip(query_station, columns)])
    magnitude = low + weight * (high - low)

    # water runs downstream, i.e. towards decreasing station
    east = np.zeros(raster.water.shape)
    north = np.zeros(raster.water.shape)
    east[inside] = -magnitude * np.cos(local_heading)
    north[inside] = -magnitude * np.sin(local_heading)

    return ChannelRaster(
        east=raster.east, north=raster.north, water=raster.water,
        navigable=raster.navigable, depth=raster.depth,
        clearance=raster.clearance, current_east=east, current_north=north,
        water_clearance=raster.water_clearance,
    )


def build_channel(points: np.ndarray, depths: np.ndarray,
                  resolution: float = 4.0, alpha: float = DEFAULT_ALPHA,
                  navigable_depth: float = NAVIGABLE_DEPTH,
                  ) -> ChannelRaster:
    """Rasterise the water body and extract the navigable channel."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.ndimage import (binary_closing, distance_transform_edt, label)

    points = np.asarray(points, dtype=float)
    depths = np.asarray(depths, dtype=float)

    east = np.arange(points[:, 0].min(), points[:, 0].max() + resolution,
                     resolution)
    north = np.arange(points[:, 1].min(), points[:, 1].max() + resolution,
                      resolution)
    grid_east, grid_north = np.meshgrid(east, north)
    query = np.column_stack([grid_east.ravel(), grid_north.ravel()])

    water = alpha_shape_mask(points, query, alpha).reshape(grid_east.shape)
    # close pinholes left where contours crowd together
    water = binary_closing(water, np.ones((3, 3)))

    depth_raster = np.full(grid_east.shape, np.nan)
    interpolator = LinearNDInterpolator(points, depths)
    inside = water.ravel()
    values = interpolator(query[inside])
    filled = depth_raster.ravel()
    filled[inside] = values
    depth_raster = filled.reshape(grid_east.shape)

    navigable = water & (np.nan_to_num(depth_raster, nan=0.0)
                         >= navigable_depth)

    # keep only the largest connected component: deep pockets cut off
    # behind a shoal are not navigable even though they are deep
    labels, count = label(navigable)
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        navigable = labels == int(np.argmax(sizes))

    clearance = distance_transform_edt(navigable) * resolution
    water_clearance = distance_transform_edt(water) * resolution

    return ChannelRaster(east=east, north=north, water=water,
                         navigable=navigable, depth=depth_raster,
                         clearance=clearance,
                         water_clearance=water_clearance)
