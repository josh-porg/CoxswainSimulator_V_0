r"""Head of the Lake: the same maps, the same optimiser, a second course.

    python scripts/render_hotl.py --out out/hotl

Three miles from the College Club dock in north Lake Union, north up the
east shore, the Pocock Turn under the University Bridge, east through
Portage Bay and the Montlake Cut, a buoyed three-point turn in Union Bay,
and a finish north-west at the Conibear Shellhouse.  The course is the
**real one**, traced from the regatta's 2025 buoy map and placed by its
bridge crossings (``tools/trace_hotl_course.py``, SOURCES sec. 118).

Everything here runs through :mod:`coxswain.viz.race_render` and
:func:`~coxswain.river.route.optimise_route`, exactly as the Charles and
Tail of the Lake do.  Nothing about this course is special-cased.

What is different from Tail of the Lake
---------------------------------------
**Four water bodies, not one.**  Lake Union, Portage Bay, the Montlake
Cut and Union Bay, unioned into one channel raster
(:func:`~coxswain.river.seattle.ship_canal_channel`).

**The buoy rule is stated by the regatta.**  "Keeping all yellow buoys
on your starboard side and all orange buoys on your port."  Yellow is a
starboard limit, orange a port limit, each binding over a boat length
either side of the mark -- the same one-sided treatment Tail of the Lake
gets, because a buoy forbids one side and leaves the other alone.

**The corridor is pinched to the Cut.**  The Montlake Cut is 50-55 m
wide between its walls in the OpenStreetMap water polygon, which agrees
with the National Bridge Inventory's 45.7 m of horizontal navigation
clearance at the Montlake Bridge (SOURCES sec. 114).  A crew has 20-25 m
either side of the centreline there and nothing to optimise; the
corridor says so rather than inventing room.

**Depth is charted and surveyed**, not nominal: NOAA ENC soundings and
depth areas over the whole canal, with the USACE 2022 multibeam merged
where the federal channel runs -- which is most of this course.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                             # noqa: E402
from coxswain.river.buoys import one_sided_limits              # noqa: E402
from coxswain.river.buoys import summary as buoy_summary       # noqa: E402
from coxswain.river.course import Course, CurrentField, DepthField  # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,        # noqa: E402
                                  optimise_route)
from coxswain.river.seattle import (SEATTLE_ORIGIN, SHIP_CANAL,  # noqa: E402
                                    bridge_gates, canal_bridges,
                                    load_obstructions,
                                    ship_canal_channel,
                                    surveyed_depth, water_mask)
from coxswain.river.trajectory import ReducedModel              # noqa: E402
from coxswain.viz.race_render import RaceScene, TraceLine, render_all  # noqa: E402

COURSE_PATH = "data/hotl_course.npy"
BUOY_PATH = "data/hotl_buoys.npy"
#: How far either side of a lone buoy the limit binds, m.
BUOY_REACH = 40.0
#: Clearance a shell needs off a mark, m -- blade plus nerves.
BUOY_MARGIN = 6.0
#: The regatta says three miles.
PUBLISHED_LENGTH = 4828.0
#: Cruising speed for a women's veteran coxed four, m/s.
SPEED = 3.9


def hotl_course(resolution: float = 10.0) -> Course:
    """The raced course, as a :class:`Course` the optimiser can use."""
    line = np.load(COURSE_PATH)
    # Resample the traced dashes to a smooth, evenly spaced centreline;
    # the dashes are 40-90 px apart on the map.
    station = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(line, axis=0).T))])
    even = np.arange(0.0, station[-1], 12.0)
    line = np.column_stack([np.interp(even, station, line[:, 0]),
                            np.interp(even, station, line[:, 1])])
    # A light smoothing, endpoint-padded so the ends are not dragged.
    window = 5
    padded = np.vstack([np.repeat(line[:1], window, axis=0), line,
                        np.repeat(line[-1:], window, axis=0)])
    kernel = np.ones(2 * window + 1) / (2 * window + 1)
    line = np.column_stack([
        np.convolve(padded[:, 0], kernel, mode="valid"),
        np.convolve(padded[:, 1], kernel, mode="valid")])

    channel = ship_canal_channel(resolution=resolution)
    rows = np.clip(np.searchsorted(channel.north, line[:, 1]), 0,
                   len(channel.north) - 1)
    columns = np.clip(np.searchsorted(channel.east, line[:, 0]), 0,
                      len(channel.east) - 1)
    # The corridor is the clearance to the nearest dock or wall, with no
    # floor under it -- see the note in render_totl.py on what a floor
    # did there.  0.5 m keeps Course happy where the line grazes a dock.
    half = np.maximum(channel.clearance[rows, columns], 0.5)
    pinched = int((half < 8.0).sum())
    if pinched:
        print("  NOTE: %d of %d stations (%.0f%%) have under 8 m of "
              "clearance; the corridor is pinned to the traced line there."
              % (pinched, len(half), 100.0 * pinched / len(half)))

    # -- buoys as ONE-SIDED limits ------------------------------------
    # Yellow to starboard, orange to port, per the regatta; consecutive
    # marks of one colour are a line.  See coxswain.river.buoys for what
    # happened when they were points.
    port, starboard = half.copy(), half.copy()
    if os.path.exists(BUOY_PATH):
        port, starboard, marks = one_sided_limits(
            line, half, np.load(BUOY_PATH), reach=BUOY_REACH,
            margin=BUOY_MARGIN)
        print("  " + buoy_summary(marks))

    # -- bridges as gates ------------------------------------------------
    # Three of them: I-5, the University Bridge (through the wide arch,
    # by rule) and the Montlake Bridge.  Each pinches the corridor to its
    # navigation opening.
    half, port, starboard, crossings = bridge_gates(line, half, port,
                                                    starboard)
    for bridge, at, _point, along in crossings:
        print("  under the %s at %.0f m: opening %.1f m, corridor +/-%.0f m"
              % (bridge.name, at, bridge.opening,
                 half[int(np.argmin(np.abs(station - at)))]))

    points, depths = surveyed_depth()
    return Course(
        centreline=line,
        half_width=half,
        port_limit=port,
        starboard_limit=starboard,
        depth=DepthField(points=points, depths=depths, is_survey=True),
        current=CurrentField.still(),
        name="Head of the Lake",
        is_survey=False,
        notes="course traced from the 2025 regatta buoy map, placed by "
              "its bridge crossings; shoreline from OpenStreetMap; depth "
              "from NOAA ENC US5SEAGL and USACE eHydro 2022",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/hotl")
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=SPEED)
    args = parser.parse_args(argv)

    course = hotl_course(args.resolution)
    length = course.length
    print("Head of the Lake, traced from the regatta buoy map")
    print("  %.0f m against a published %.0f (%+.1f%%)"
          % (length, PUBLISHED_LENGTH,
             100 * (length / PUBLISHED_LENGTH - 1)))
    print("  corridor half-width: min %.0f, median %.0f m"
          % (course.half_width.min(), np.median(course.half_width)))

    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    evaluator = RouteEvaluator(course, boat=boat,
                               reference_speed=args.speed).with_steering(
        ReducedModel())
    centre = evaluator.evaluate(Route.centreline(course))
    best = optimise_route(evaluator, n_control=11, iterations=40)
    print()
    print("  %-12s %9s %9s %9s %10s %8s" % ("", "score s", "time s",
                                            "length m", "peak yaw", "aground"))
    for label, result in (("as drawn", centre), ("optimised", best)):
        print("  %-12s %9.1f %9.1f %9.0f %10.2f %7.1f%%"
              % (label, result.elapsed, result.elapsed_clean,
                 result.path_length, result.peak_yaw_rate,
                 100.0 * result.fraction_aground))
    print("  SAVED %.1f s of time (%.1f s of score)"
          % (centre.elapsed_clean - best.elapsed_clean,
             centre.elapsed - best.elapsed))

    stations = np.linspace(0.0, length, 500)
    optimised = course.offset_position(stations, best.route.offset_at(stations))

    east, north, water = water_mask(args.resolution, names=SHIP_CANAL)
    try:
        from coxswain.river.structures import seattle_structures
        structures = seattle_structures(SEATTLE_ORIGIN)
    except Exception as error:                       # pragma: no cover
        print("  (no structures: %s)" % str(error)[:60])
        structures = None

    marks = [(course.centreline[0, 0], course.centreline[0, 1], "START"),
             (course.centreline[-1, 0], course.centreline[-1, 1], "FINISH")]
    for bridge in canal_bridges():
        hit = bridge.crossing(course.centreline)
        if hit is not None:
            marks.append((hit[1][0], hit[1][1], bridge.name))
    scene = RaceScene(
        name="Head of the Lake",
        east=east, north=north, water=water,
        lines=[TraceLine(course.centreline, "course as drawn",
                         colour="#7d8f9c", width=1.4, style="--"),
               TraceLine(optimised, "optimised line", colour="#ff9248",
                         width=2.2)],
        obstructions=[p for _k, p in load_obstructions() if len(p) > 1],
        structures=structures,
        depth_at=lambda x, y: float(course.depth_at(x, y)),
        speed=args.speed, boat_length=boat.length,
        marks=marks,
    )
    os.makedirs(args.out, exist_ok=True)
    written = render_all(scene, args.out)
    print()
    for path in written:
        print("  wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
