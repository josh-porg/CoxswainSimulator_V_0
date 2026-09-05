r"""Tail of the Lake: the same maps and views the Charles gets.

    python scripts/render_totl.py --out out/totl

The course is the **real one**, traced from the 2024 regatta map: gold
dashed race line and coloured buoy markers detected by colour, then
georeferenced against the OpenStreetMap shoreline (SOURCES sec. 100).
It measures 3786 m against a published 4000 m -- within 5.3% -- and is
not a contour, a guess, or a length-matched fit.

Everything here runs through :mod:`coxswain.viz.race_render`, the same
renderer the Charles uses, and :func:`~coxswain.river.route.optimise_route`,
the same optimiser. Nothing about this course is special-cased.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                             # noqa: E402
from coxswain.river.course import Course, CurrentField, DepthField  # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,        # noqa: E402
                                  optimise_route)
from coxswain.river.seattle import (SEATTLE_ORIGIN,             # noqa: E402
                                    load_obstructions,
                                    nominal_depth, rowable_mask,
                                    water_mask)
from coxswain.river.trajectory import ReducedModel              # noqa: E402
from coxswain.viz.race_render import RaceScene, TraceLine, render_all  # noqa: E402

COURSE_PATH = "data/totl_course.npy"
BUOY_PATH = "data/totl_buoys.npy"
#: How far either side of a buoy the limit binds, m.
BUOY_REACH = 40.0
#: Clearance a shell needs off a mark, m -- blade plus nerves.
BUOY_MARGIN = 6.0
PUBLISHED_LENGTH = 4000.0


def totl_course(resolution: float = 10.0) -> Course:
    """The raced course, as a :class:`Course` the optimiser can use.

    Half-width comes from the distance transform of the **rowable** water,
    so the corridor the optimiser is allowed to use already has the docks
    taken out of it -- with no floor under it, which is the whole point.
    See the note at the half-width computation below.
    """
    from scipy.ndimage import distance_transform_edt, label

    line = np.load(COURSE_PATH)
    east, north, row = rowable_mask(resolution, names=("Lake Union",))
    marks, _ = label(row)
    main = marks == (np.bincount(marks.ravel())[1:].argmax() + 1)
    clearance = distance_transform_edt(main) * resolution

    columns = np.clip(np.searchsorted(east, line[:, 0]), 0, len(east) - 1)
    rows = np.clip(np.searchsorted(north, line[:, 1]), 0, len(north) - 1)

    # The corridor is the clearance, and **only** the clearance.
    #
    # This used to read ``np.maximum(clearance, 8.0)``, a floor put there
    # to stop the corridor collapsing to nothing where the traced line
    # grazes a dock.  What it actually did was hand the optimiser eight
    # metres of room in exactly the places there was none: 23% of the
    # optimised line came out with under 5 m to the nearest pier, and the
    # 3-D render showed a marina passing through the boat.  A floor on a
    # constraint is not a safety margin, it is the constraint switched
    # off where it binds hardest.
    #
    # Where the traced line itself is inside a structure the clearance is
    # zero and the corridor is zero, which pins the route to the drawn
    # line there rather than inventing water.  That is the honest answer:
    # the map says the race goes here, and the dock survey says there is
    # no room, and this code cannot settle which is wrong.
    # Half a metre, not zero, because ``Course`` requires a positive
    # width -- and not eight, which is what was there.  At 0.5 m the
    # route is pinned to the traced line where the docks bind, which is
    # the honest answer rather than a comfortable one.
    half = np.maximum(clearance[rows, columns], 0.5)
    pinched = int((half < 8.0).sum())
    if pinched:
        print("  NOTE: %d of %d stations (%.0f%%) have under 8 m of "
              "clearance;" % (pinched, len(half), 100.0 * pinched / len(half)))
        print("        the corridor is pinned to the traced line there.")

    # -- buoys as ONE-SIDED limits ------------------------------------
    # "Keep red triangle buoys to port, green to starboard."  A buoy does
    # not narrow the corridor symmetrically: it forbids one side and
    # leaves the other alone.  Without this the optimiser rounds inside
    # the southern turning buoys and "saves" 60.9 s that costs 60 s of
    # penalty each, or a disqualification for two.
    station = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(line, axis=0).T))])
    port = half.copy()
    starboard = half.copy()
    if os.path.exists(BUOY_PATH):
        buoys = np.load(BUOY_PATH)
        heading = np.arctan2(np.gradient(line[:, 1]), np.gradient(line[:, 0]))
        for is_red, bx, by in buoys:
            gap = np.hypot(line[:, 0] - bx, line[:, 1] - by)
            index = int(np.argmin(gap))
            if gap[index] > 200.0:          # not a course buoy
                continue
            normal = np.array([-np.sin(heading[index]),
                               np.cos(heading[index])])
            offset = float(np.dot([bx - line[index, 0], by - line[index, 1]],
                                  normal))
            # Bind over a boat length either side of the mark, not one node.
            near = np.abs(station - station[index]) < BUOY_REACH
            if is_red:                       # keep to port -> stay starboard
                port[near] = np.minimum(port[near], offset - BUOY_MARGIN)
            else:                            # keep to starboard -> stay port
                starboard[near] = np.minimum(starboard[near],
                                             -offset - BUOY_MARGIN)
        port = np.maximum(port, 2.0)
        starboard = np.maximum(starboard, 2.0)

    grid_east, grid_north = np.meshgrid(east, north)
    depths = nominal_depth(distance_transform_edt(main) * resolution)
    samples = np.column_stack([grid_east[main], grid_north[main]])

    return Course(
        centreline=line,
        half_width=half,
        port_limit=port,
        starboard_limit=starboard,
        depth=DepthField(points=samples, depths=depths[main],
                         is_survey=False),
        current=CurrentField.still(),
        name="Tail of the Lake",
        is_survey=False,
        notes="course traced from the 2024 regatta map; shoreline from "
              "OpenStreetMap; DEPTH IS NOMINAL, not surveyed",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/totl")
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=3.9)
    parser.add_argument("--video", action="store_true",
                        help="also write the fly-down video")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args(argv)

    course = totl_course(args.resolution)
    length = course.length
    print("Tail of the Lake, traced from the regatta map")
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
    best = optimise_route(evaluator, n_control=9, iterations=40)
    print()
    print("  %-12s %9s %9s %10s" % ("", "time s", "length m", "peak yaw"))
    print("  %-12s %9.1f %9.0f %10.2f"
          % ("as drawn", centre.elapsed, centre.path_length,
             centre.peak_yaw_rate))
    print("  %-12s %9.1f %9.0f %10.2f"
          % ("optimised", best.elapsed, best.path_length,
             best.peak_yaw_rate))
    print("  SAVED %.1f s" % (centre.elapsed - best.elapsed))

    stations = np.linspace(0.0, length, 400)
    offsets = best.route.offset_at(stations)
    centreline = np.array([course.position_at(s) for s in stations])
    optimised = []
    for index, station in enumerate(stations):
        heading = float(course.heading_at(station))
        normal = np.array([-np.sin(heading), np.cos(heading)])
        optimised.append(centreline[index] + normal * offsets[index])
    optimised = np.asarray(optimised)

    east, north, water = water_mask(args.resolution, names=("Lake Union",))
    try:
        from coxswain.river.structures import seattle_structures
        structures = seattle_structures(SEATTLE_ORIGIN)
    except Exception as error:                       # pragma: no cover
        print("  (no structures: %s)" % str(error)[:60])
        structures = None

    scene = RaceScene(
        name="Tail of the Lake",
        east=east, north=north, water=water,
        lines=[TraceLine(course.centreline, "course as drawn",
                         colour="#7d8f9c", width=1.4, style="--"),
               TraceLine(optimised, "optimised line", colour="#ff9248",
                         width=2.2)],
        obstructions=[p for _k, p in load_obstructions() if len(p) > 1],
        structures=structures,
        depth_at=lambda x, y: float(course.depth_at(x, y)),
        speed=args.speed, boat_length=boat.length,
        marks=[(course.centreline[0, 0], course.centreline[0, 1], "START"),
               (course.centreline[-1, 0], course.centreline[-1, 1],
                "FINISH")],
    )
    os.makedirs(args.out, exist_ok=True)
    written = render_all(scene, args.out)
    if args.video:
        from coxswain.viz.race_render import write_video
        print("  rendering %d frames ..." % args.frames)
        written.append(write_video(
            scene, os.path.join(args.out, "tail_of_the_lake.mp4"),
            frames=args.frames, fps=args.fps))
    print()
    for path in written:
        print("  wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
