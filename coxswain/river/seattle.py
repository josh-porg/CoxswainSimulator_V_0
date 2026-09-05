r"""Lake Union and Portage Bay: the home water, as a testable course.

Tail of the Lake circumnavigates Lake Union over 4000 m, three weeks
before the Head of the Charles and with the same lineups. That makes it
the natural **test of this program on water the crew knows by eye**: if
the line optimiser says something surprising about Lake Union, a coxswain
who rows there every week can say whether it is surprising-and-true or
surprising-and-wrong. The Charles offers no such check.

Why this course exercises different physics
-------------------------------------------
The Charles results are dominated by **depth**. Its racing line sits at a
depth Froude number near one, where a metre of water either way moves the
answer by tens of seconds (SOURCES sec. 66-67, 79).

Lake Union is not like that. It runs about 15 m through the middle, so a
four at 3.9 m/s sees

.. math::

    Fr_h = \frac{3.9}{\sqrt{9.81 \times 15}} = 0.32

which is deeply subcritical -- the shallow-water correction is 1.00 to
three decimals and stays there. **The depth term is switched off**, and
what decides a line is distance and turning.

That is what makes it a good test rather than a lesser one: it exercises
the geometry half of the model in isolation. If the optimiser cannot get a
lap of a lake right with the hard part removed, its Charles answer was
luck.

What is real here and what is not
---------------------------------
**The shoreline is real** -- OpenStreetMap water polygons, extracted by
``tools/extract_seattle_water.py``. For an urban lake carrying a federal
navigation channel these are well surveyed.

**The depth is not.** No soundings are used. :func:`nominal_depth` is a
documented shelf-and-basin profile, and every course built here is
``is_survey=False`` and fails :meth:`Course.require_survey` by design.
NOAA charts the Lake Washington Ship Canal so the data exists; it is
simply not in this repository, and nothing may quietly pretend otherwise.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Tuple

import numpy as np

from .course import Course, CurrentField, DepthField

__all__ = ["SEATTLE_ORIGIN", "water_path", "load_water", "water_mask",
           "nominal_depth", "lake_union_course", "TOTL_LENGTH",
           "load_obstructions", "rowable_mask"]

#: Tangent-plane origin: the middle of Lake Union.
SEATTLE_ORIGIN = (47.6395, -122.3330)

#: Tail of the Lake is a 4 km circuit of Lake Union.  Confirmed
#: independently in SOURCES sec. 93: the crew-matched Charles conversion
#: implies 3966 m from race results alone, against a published 4000 m.
TOTL_LENGTH = 4000.0

#: Maximum depth used by :func:`nominal_depth`, m.  Lake Union's charted
#: basin is about 15 m; the exact value barely matters because ``Fr_h``
#: is 0.32 there and the shallow-water factor is flat.
BASIN_DEPTH = 15.0
#: How far the shelf extends from shore before the basin, m.
SHELF = 90.0

#: Length of the boat, m -- the smoothing scale for a lap.  A hull cannot
#: respond to curvature structure shorter than itself, which is the same
#: argument :meth:`RouteEvaluator._required_yaw` makes for its own window.
BOAT_LENGTH = 13.4


def _resample_closed(line: np.ndarray, count: int) -> np.ndarray:
    """Re-space a closed loop evenly by arc length."""
    loop = np.vstack([line, line[:1]])
    step = np.hypot(*np.diff(loop, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(step)])
    wanted = np.linspace(0.0, along[-1], count, endpoint=False)
    return np.column_stack([np.interp(wanted, along, loop[:, 0]),
                            np.interp(wanted, along, loop[:, 1])])


def _smooth_closed(line: np.ndarray, span: float,
                   spacing: float = None) -> np.ndarray:
    """Boxcar-smooth a closed loop over ``span`` metres of arc.

    Wrapped, not zero-padded: ``mode="same"`` on an open array drags the
    ends toward the origin, and on a lap the two ends are neighbours.
    """
    if spacing is None:
        spacing = float(np.median(np.hypot(*np.diff(line, axis=0).T)))
    width = max(int(round(span / max(spacing, 1e-6))), 3)
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width) / width
    pad = width
    wrapped = np.vstack([line[-pad:], line, line[:pad]])
    return np.column_stack([
        np.convolve(wrapped[:, 0], kernel, mode="same")[pad:-pad],
        np.convolve(wrapped[:, 1], kernel, mode="same")[pad:-pad]])


def water_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(here), "data", "seattle_water.json")


@lru_cache(maxsize=4)
def load_water(path: str = None):
    """``(origin, [(name, points), ...])`` from the extracted polygons."""
    with open(path or water_path(), encoding="utf-8") as handle:
        payload = json.load(handle)
    # **Two points, not four.**  These are shoreline *fragments*, not
    # polygons, and the shortest ones are exactly the segments that close
    # the ring.  The same filter lived in the extractor and was fixed
    # there first; leaving it here re-dropped them on read, so the ring
    # went on failing to close and Lake Union went on filling to
    # 2.72 km2 against a true 2.35.  One bug, two places, and fixing
    # either alone changes nothing.
    pieces = [(piece["name"], np.asarray(piece["points"], dtype=float))
              for piece in payload["pieces"]
              if len(piece["points"]) >= 2]
    return tuple(payload["origin"]), tuple(pieces)


def stitch_rings(fragments, tolerance: float = 5.0):
    """Join open ways into closed rings.

    OpenStreetMap returns a large lake as a **multipolygon relation whose
    members are open ways**, each a piece of the shoreline.  Running
    point-in-polygon on a fragment treats it as if its two ends were
    joined, which fills whatever that chord happens to enclose.

    That is not a subtle failure.  Doing it to Lake Union produced a
    narrow Y-shape of 0.592 km2 against the lake's real 2.1 km2 -- and it
    survived every numeric check in this module, because a wrong mask
    still yields a plausible-looking lap, fetch and dock fraction.  It
    fell over the moment somebody drew a picture of it.
    """
    remaining = [np.asarray(f, dtype=float) for f in fragments
                 if len(f) >= 2]
    rings = []
    while remaining:
        chain = remaining.pop(0)
        extended = True
        while extended and not _closed(chain, tolerance):
            extended = False
            for index, candidate in enumerate(remaining):
                for piece in (candidate, candidate[::-1]):
                    if np.hypot(*(piece[0] - chain[-1])) <= tolerance:
                        chain = np.vstack([chain, piece[1:]])
                    elif np.hypot(*(piece[-1] - chain[0])) <= tolerance:
                        chain = np.vstack([piece[:-1], chain])
                    else:
                        continue
                    remaining.pop(index)
                    extended = True
                    break
                if extended:
                    break
        if len(chain) >= 4:
            rings.append(chain)
    return rings


def _closed(ring: np.ndarray, tolerance: float) -> bool:
    return len(ring) >= 4 and np.hypot(*(ring[-1] - ring[0])) <= tolerance


def _inside(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Ray-crossing point-in-polygon, vectorised over ``points``."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        if ay == by:
            continue
        straddles = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= straddles & (x < crossing)
    return inside


def water_mask(resolution: float = 10.0, names=("Lake Union",
                                                "Portage Bay")):
    """``(east, north, mask)`` for the named water bodies.

    Point-in-polygon rather than the alpha shape
    :func:`coxswain.river.channel.build_channel` uses, because here the
    shoreline is given as an actual polygon.  Reconstructing a boundary
    that is already known would only add error.
    """
    _origin, pieces = load_water()
    wanted = [(name, points) for name, points in pieces
              if not names or name in names]
    if not wanted:
        raise ValueError("no water pieces named %r" % (names,))

    # Stitch first: the fragments are pieces of one shoreline, not
    # polygons in their own right.  Then keep the LARGEST ring only.
    # Stitching a lake's members yields the lake plus whatever smaller
    # loops the relation carries; unioning them all inflated Lake Union
    # from its true 2.13 km2 to 2.74.  Force it closed, because a ring
    # that fails to meet by a few metres still fills wrongly.
    rings = stitch_rings([points for _n, points in wanted])
    if not rings:
        raise ValueError("no ring could be stitched")

    def _area(ring):
        return abs(0.5 * np.sum(ring[:-1, 0] * ring[1:, 1]
                                - ring[1:, 0] * ring[:-1, 1]))

    ring = max(rings, key=_area)
    if np.hypot(*(ring[-1] - ring[0])) > 1e-9:
        ring = np.vstack([ring, ring[:1]])
    wanted = [(names[0] if names else "water", ring)]

    stacked = np.vstack([points for _n, points in wanted])
    east = np.arange(stacked[:, 0].min(), stacked[:, 0].max() + resolution,
                     resolution)
    north = np.arange(stacked[:, 1].min(), stacked[:, 1].max() + resolution,
                      resolution)
    grid_east, grid_north = np.meshgrid(east, north)
    query = np.column_stack([grid_east.ravel(), grid_north.ravel()])

    mask = np.zeros(len(query), dtype=bool)
    for _name, points in wanted:
        mask |= _inside(points, query)
    return east, north, mask.reshape(grid_east.shape)


def obstruction_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(here), "data",
                        "seattle_obstructions.json")


@lru_cache(maxsize=2)
def load_obstructions(path: str = None):
    """Piers, floating homes, marinas and breakwaters, in the local plane.

    **These, not the shoreline, are what keeps a boat off the bank.** Lake
    Union is ringed by docks and carries 141 mapped houseboats; the water
    between them is wet on the map and unrowable in fact, and a line
    optimised against the shoreline alone will happily steer through a
    marina.
    """
    with open(path or obstruction_path(), encoding="utf-8") as handle:
        payload = json.load(handle)
    return tuple((item["kind"], np.asarray(item["points"], dtype=float))
                 for item in payload["obstructions"])


def rowable_mask(resolution: float = 10.0, names=("Lake Union",),
                 margin: float = 8.0):
    """Water with the docks taken out of it.

    ``margin`` is how far off a structure a shell must stay -- a blade
    reaches about 3 m past the rigger, and nobody rows within touching
    distance of a moored boat.
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt

    east, north, water = water_mask(resolution, names=names)
    blocked = np.zeros_like(water)
    columns = len(east)
    for _kind, points in load_obstructions():
        inside = ((points[:, 0] >= east[0]) & (points[:, 0] <= east[-1])
                  & (points[:, 1] >= north[0]) & (points[:, 1] <= north[-1]))
        if not inside.any():
            continue
        cols = np.clip(np.searchsorted(east, points[inside, 0]), 0,
                       columns - 1)
        rows = np.clip(np.searchsorted(north, points[inside, 1]), 0,
                       len(north) - 1)
        blocked[rows, cols] = True
        # A pier is a line of vertices; fill between consecutive ones so a
        # dock is a barrier rather than a row of dots.
        for i in range(len(cols) - 1):
            steps = max(abs(int(cols[i + 1]) - int(cols[i])),
                        abs(int(rows[i + 1]) - int(rows[i])), 1)
            for t in np.linspace(0.0, 1.0, steps + 1):
                r = int(round(rows[i] + t * (rows[i + 1] - rows[i])))
                c = int(round(cols[i] + t * (cols[i + 1] - cols[i])))
                blocked[r, c] = True

    pad = max(int(round(margin / max(resolution, 1e-6))), 1)
    blocked = binary_dilation(blocked, np.ones((2 * pad + 1, 2 * pad + 1)))
    return east, north, water & ~blocked


def fetch_at(point, bearing, resolution: float = 10.0, limit: float = 4000.0,
             mask=None):
    """Metres of open water upwind of ``point``, m.

    ``bearing`` is meteorological -- the direction the wind comes *from*.
    This is the number the chop model needs and the one most easily
    guessed wrong: on a lake the fetch is a strong function of direction,
    because a 3 km basin 1 km wide offers three times the fetch along its
    axis as across it.
    """
    if mask is None:
        east, north, water = water_mask(resolution, names=("Lake Union",))
    else:
        east, north, water = mask
    towards = np.radians(90.0 - (float(bearing) + 180.0))
    step = np.array([np.cos(towards), np.sin(towards)]) * resolution
    here = np.asarray(point, dtype=float)[:2].copy()
    travelled = 0.0
    while travelled < limit:
        here = here - step
        travelled += resolution
        row = int(np.clip(np.searchsorted(north, here[1]), 0,
                          len(north) - 1))
        column = int(np.clip(np.searchsorted(east, here[0]), 0,
                             len(east) - 1))
        if not water[row, column]:
            break
    return travelled


def lake_union_channel(resolution: float = 10.0, margin: float = 8.0):
    """Lake Union as a :class:`~coxswain.river.channel.ChannelRaster`.

    The same object the Charles hands to the 3-D renderer, so Lake Union
    can use that renderer rather than a second one written to avoid it.
    ``navigable`` is the water with the docks removed; ``depth`` is the
    nominal profile and is **not surveyed**.

    This used to be two functions of the same name -- a ``_MaskChannel``
    adapter for the wind model and this one for the renderer -- with the
    second silently shadowing the first, so the wind model would have got
    a ``ChannelRaster`` without anything saying so.  It turned out the
    adapter was never needed: ``ChannelRaster`` already exposes the
    ``east``/``north``/``water``/``navigable``/``resolution``/``index_of``
    that :class:`~coxswain.hydro.canopy.ShelteredWind` marches over, and
    the bathymetry the adapter existed to avoid inventing is invented
    either way by :func:`nominal_depth`, which at least says so.
    """
    from scipy.ndimage import distance_transform_edt

    from .channel import ChannelRaster

    east, north, water = water_mask(resolution, names=("Lake Union",))
    _e, _n, rowable = rowable_mask(resolution, names=("Lake Union",),
                                   margin=margin)
    depth = np.full(water.shape, np.nan)
    reach = distance_transform_edt(water) * resolution
    depth[water] = nominal_depth(reach[water])
    clearance = distance_transform_edt(rowable) * resolution
    return ChannelRaster(east=east, north=north, water=water,
                         navigable=rowable, depth=depth,
                         clearance=clearance)


def nominal_depth(distance_from_shore) -> np.ndarray:
    """Depth from distance to the nearest shore, m.  **Not surveyed.**

    A shelf that reaches :data:`BASIN_DEPTH` over :data:`SHELF` metres,
    then flat.  It exists so a course object has a depth field at all, and
    it is deliberately crude: at ``Fr_h = 0.32`` the shallow-water
    correction is 1.00 regardless, so a better profile would change no
    answer this module is used for.

    If that ever stops being true -- a shallower lake, a faster boat --
    this function is the thing to replace, and the course must stop
    claiming ``is_survey=False`` only after real soundings arrive.
    """
    reach = np.clip(np.asarray(distance_from_shore, float) / SHELF, 0.0, 1.0)
    return 0.5 + (BASIN_DEPTH - 0.5) * np.sqrt(reach)


def lake_union_course(resolution: float = 10.0, offset: float = 50.0,
                      points: int = 400) -> Course:
    """A lap of Lake Union, at a fixed distance off the shore.

    The centreline is the **contour of constant clearance** -- the locus of
    points ``offset`` metres from the nearest shore.  On a closed basin
    that is exactly what a circumnavigation is, and it needs no waypoints
    invented by hand.  An earlier attempt swept angles from the centroid
    and produced an 11.6 km tangle, because Lake Union is 3 km long and
    1 km wide and a centroid sweep is only sensible on a round pond.

    **This is a lap of the lake, not the Tail of the Lake course.**  At
    50 m off the shore the lap measures about 2757 m, and the published
    race is 4000 m, so the real course must take in water beyond the lake
    proper -- most plausibly the ship canal towards Fremont, or Portage
    Bay through the Montlake Cut.  The route is not in this repository and
    is not guessed at here.  What this gives is **real geometry on the
    right water**, which is what the optimiser needs to be tested against.
    """
    from scipy.ndimage import distance_transform_edt

    east, north, mask = rowable_mask(resolution, names=("Lake Union",))
    clearance = distance_transform_edt(mask) * resolution

    # **Smooth the field, not the curve.**  Marching squares walks cell
    # diagonals, so a contour of a raw distance transform is a sawtooth:
    # the 50 m loop came back with 18,288 degrees of total turning for a
    # lap that owes 360, and minimum radii of centimetres.  Smoothing the
    # resulting polyline cannot undo that -- boxcar over a zigzag leaves a
    # smaller zigzag, and at 80 m of smoothing it was still 1030 degrees.
    #
    # Blurring the clearance field first gives a smooth scalar whose level
    # sets are smooth curves, which is the property actually wanted.  The
    # blur is two boat lengths, the scale below which a hull cannot
    # respond to curvature anyway.
    from scipy.ndimage import gaussian_filter
    clearance = gaussian_filter(
        clearance, sigma=2.0 * BOAT_LENGTH / max(resolution, 1e-6))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure()
    try:
        contours = plt.contour(east, north, clearance, levels=[offset])
        loops = [np.asarray(seg, dtype=float)
                 for seg in contours.allsegs[0] if len(seg) > 20]
    finally:
        plt.close(figure)
    if not loops:
        raise ValueError("no closed lap at %.0f m off the shore" % offset)
    line = max(loops, key=lambda p: np.hypot(*np.diff(p, axis=0).T).sum())

    # Start at the northernmost point: the Gasworks end, where the race
    # starts and finishes.  Cutting the loop there makes the closed lap
    # into the open station axis the rest of the package expects.
    begin = int(np.argmax(line[:, 1]))
    line = np.vstack([line[begin:], line[:begin]])

    # A contour comes back closed, so its last point repeats its first,
    # and marching-squares emits coincident vertices where the level
    # grazes a cell edge.  Course rejects both, correctly -- a repeated
    # point has no tangent and every heading derived from it is garbage.
    step = np.hypot(*np.diff(line, axis=0).T)
    line = np.vstack([line[:1], line[1:][step > 1e-6]])
    if np.hypot(*(line[-1] - line[0])) < 1e-6:
        line = line[:-1]

    # **Resample and smooth, or the curvature is all staircase.**  A
    # marching-squares contour steps along cell edges, so its pointwise
    # heading swings by tens of degrees between adjacent vertices at the
    # grid scale.  Fed straight in, the evaluator read a peak yaw of
    # 54 deg/s on a lap whose real turns need about 4, priced the
    # steering accordingly, and returned a lap time of ten days.
    #
    # The Charles centreline never showed this because it arrives
    # pre-smoothed from the channel extraction; a raw contour does not,
    # and nothing downstream is obliged to notice.
    line = _resample_closed(line, points)
    # Three boat lengths, not one.  A one-length window leaves the four
    # corners where Lake Union meets its canal inlets turning at up to
    # 46 deg/s -- real geometry of the 50 m contour, but not a line any
    # crew steers, and it makes the centreline baseline meaningless.
    # A coxswain rounds those corners; so does this.
    line = _smooth_closed(line, span=3.0 * BOAT_LENGTH, spacing=None)

    columns = np.clip(np.searchsorted(east, line[:, 0]), 0, len(east) - 1)
    rows = np.clip(np.searchsorted(north, line[:, 1]), 0, len(north) - 1)
    # The lap sits ``offset`` from shore, so that is its half-width to the
    # outside; inboard there is much more water, and the binding limit is
    # the one that puts a blade on the bank.
    half_width = np.full(len(line), float(offset))

    grid_east, grid_north = np.meshgrid(east, north)
    depths = nominal_depth(clearance[mask])
    samples = np.column_stack([grid_east[mask], grid_north[mask]])

    return Course(
        centreline=line,
        half_width=half_width,
        depth=DepthField(points=samples, depths=depths, is_survey=False),
        current=CurrentField.still(),
        name="Lake Union lap (%.0f m off shore)" % offset,
        is_survey=False,
        notes="shoreline from OpenStreetMap; DEPTH IS NOMINAL, not "
              "surveyed; a lap of the lake, not the Tail of the Lake "
              "race course -- see coxswain.river.seattle",
    )
