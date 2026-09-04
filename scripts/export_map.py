r"""Export the raced course, its fields, and the optimised line.

    python scripts/export_map.py --out out/report/map_data.json

Three things this used to get wrong, all now fixed:

**It showed water nobody races.** :func:`charles_course` is the whole
12.4 km surveyed reach; the Head of the Charles races 4828 m of it.
Everything here now runs on
:func:`~coxswain.river.charles.hocr_race_course`, which is start line to
finish line and ordered bow-first. That correction alone moves the
shallowest depth on the centreline from 2.04 m to **2.41 m** -- the worst
water in the reach is not raced.

**The line was a guess.** It was a greedy depth-follower that ignored what
steering costs and swung 56 m. The line now comes from
:func:`~coxswain.river.route.optimise_route` against a
:class:`~coxswain.river.route.RouteEvaluator` carrying the reduced
steering model -- the same optimiser the rest of the project uses, which
trades distance against depth against the curvature the boat has to buy.

**The layers only existed in plan view.** A river 4.8 km long and 100 m
wide is mostly empty space on a map, and everything that matters varies
*across* the channel. Every field is now also sampled on a
``station x offset`` grid for the straightened view, where the width is
legible.

The yield map
-------------
Also exported: which side to send a crew you are passing. The rulebook
gives the passer the choice and says nothing about which is better; the
river does, because the two banks are not the same depth. Positive means
**port is the worse side for them**, so port is the side to name.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river.charles import (charles_channel,  # noqa: E402
                                    hocr_race_course)
from coxswain.river.route import (Route, RouteEvaluator,  # noqa: E402
                                  optimise_route)
from coxswain.river.trajectory import ReducedModel                # noqa: E402

from course_pacing import build_boat, hull_drag                   # noqa: E402

GRAVITY = 9.80665


def flattened(course, wind, stations, offsets, level=0.0):
    """Depth, current and headwind on a ``station x offset`` grid."""
    depth = np.full((len(offsets), len(stations)), np.nan)
    current = np.full_like(depth, np.nan)
    breeze = np.full_like(depth, np.nan)
    for j, station in enumerate(stations):
        limit = float(course.half_width_at(station))
        heading = float(course.heading_at(station))
        tangent = np.array([np.cos(heading), np.sin(heading)])
        for i, offset in enumerate(offsets):
            if abs(offset) > limit:
                continue
            point = np.atleast_1d(np.asarray(
                course.offset_position(station, offset), float)).ravel()[:2]
            depth[i, j] = float(course.depth_at(point[0], point[1])) + level
            flow = np.asarray(course.current_at(point[0], point[1]))[:2]
            # The race course is bow-first, so its tangent already points
            # the way the crew is going; positive here means adverse.
            current[i, j] = -float(np.dot(flow, tangent))
            if wind is not None:
                speed = float(wind.speed_at(point[0], point[1]))
                breeze[i, j] = -speed * float(np.dot(wind._towards, tangent))

    def clean(grid):
        return [[None if np.isnan(v) else round(float(v), 3) for v in row]
                for row in grid]
    return clean(depth), clean(current), clean(breeze)


def yield_side(course, speed_of, stations, offset=3.5, level=0.0):
    """Speed penalty in percent for sending them to port, not starboard.

    Positive means port hurts them more, so port is the side to name.
    """
    out = []
    for station in stations:
        limit = max(float(course.half_width_at(station)) - 1.0, 0.5)
        reach = min(offset, limit)
        port = np.atleast_1d(np.asarray(
            course.offset_position(station, +reach), float)).ravel()[:2]
        starboard = np.atleast_1d(np.asarray(
            course.offset_position(station, -reach), float)).ravel()[:2]
        fast_port = speed_of(
            float(course.depth_at(port[0], port[1])) + level, 0.0)
        fast_stbd = speed_of(
            float(course.depth_at(starboard[0], starboard[1])) + level, 0.0)
        reference = max(0.5 * (fast_port + fast_stbd), 1e-6)
        out.append(round(100.0 * (fast_stbd - fast_port) / reference, 3))
    return out


def speed_table(drag, shallow, rowers, power,
                depths=None, headwinds=None):
    """``(depth, headwind) -> speed`` as a bilinear interpolator.

    Solving the power balance point by point needed over two million
    ``hull_resistance`` calls to fill the parameter grid.  The relation is
    smooth and two-dimensional, so it is solved once and interpolated --
    the same treatment ``RouteEvaluator.speed_through_water`` and
    ``CoursePacing`` already give it.
    """
    from dataclasses import replace

    from coxswain.hydro.shallow import ShallowWaterModel

    depths = (np.concatenate([np.arange(0.30, 6.0, 0.05),
                              np.arange(6.0, 30.01, 0.5)])
              if depths is None else np.asarray(depths, float))
    headwinds = (np.arange(-10.0, 10.01, 1.0) if headwinds is None
                 else np.asarray(headwinds, float))
    delivered = 0.80 * power * rowers
    template = shallow or ShallowWaterModel()

    probe = np.arange(0.2, 9.01, 0.02)
    hull = np.array([drag(float(v)) for v in probe])

    grid = np.empty((len(depths), len(headwinds)))
    for i, depth in enumerate(depths):
        model = replace(template, depth=float(max(depth, 0.30)))
        factor = np.array([float(model.factor(float(v))) for v in probe])
        resist = hull * factor
        for j, headwind in enumerate(headwinds):
            low, high = 0.2, 9.0
            for _ in range(40):
                mid = 0.5 * (low + high)
                apparent = mid + headwind
                air = 0.5 * 1.225 * 3.22 * apparent * abs(apparent)
                total = float(np.interp(mid, probe, resist)) + air
                if total * mid < delivered:
                    low = mid
                else:
                    high = mid
            grid[i, j] = 0.5 * (low + high)

    def lookup(depth, headwind=0.0):
        row = np.interp(depth, depths, np.arange(len(depths)))
        column = np.interp(headwind, headwinds, np.arange(len(headwinds)))
        i0, j0 = int(np.floor(row)), int(np.floor(column))
        i1 = min(i0 + 1, len(depths) - 1)
        j1 = min(j0 + 1, len(headwinds) - 1)
        fi, fj = row - i0, column - j0
        return float((1 - fi) * (1 - fj) * grid[i0, j0]
                     + fi * (1 - fj) * grid[i1, j0]
                     + (1 - fi) * fj * grid[i0, j1] + fi * fj * grid[i1, j1])
    return lookup


def optimised_line(course, boat, stations, n_control=11, iterations=50):
    """The project's own optimiser, on the raced water, steering charged."""
    evaluator = RouteEvaluator(
        course, boat=boat, reference_speed=3.895).with_steering(
            ReducedModel())
    best = optimise_route(evaluator, n_control=n_control,
                          iterations=iterations)
    return (best.route.offset_at(stations), best.elapsed, best.path_length,
            best.peak_yaw_rate)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/report/map_data.json")
    parser.add_argument("--boat", default="4+", choices=["8+", "4+"])
    parser.add_argument("--stations", type=int, default=120)
    parser.add_argument("--offsets", type=int, default=41)
    parser.add_argument("--levels", type=float, nargs="+",
                        default=[-0.3, 0.0, 0.3])
    parser.add_argument("--directions", type=float, nargs="+",
                        default=[250.0, 340.0, 70.0])
    args = parser.parse_args(argv)

    boat = build_boat(args.boat, 30.0 if args.boat == "4+" else 32.0)
    drag = hull_drag(boat)
    course = hocr_race_course()
    channel = charles_channel()

    stations = np.linspace(0.0, course.length, args.stations)
    widths = np.array([float(course.half_width_at(s)) for s in stations])
    offsets = np.linspace(-float(widths.max()), float(widths.max()),
                          args.offsets)

    data = {
        "boat": args.boat,
        "length": course.length,
        "stations": [round(float(s), 1) for s in stations],
        "offsets": [round(float(o), 2) for o in offsets],
        "half_width": [round(float(w), 1) for w in widths],
        "levels": [float(v) for v in args.levels],
        "directions": [int(v) for v in args.directions],
    }
    centre = np.array([course.position_at(s) for s in stations])
    data["centreline"] = [[round(float(p[0]), 1), round(float(p[1]), 1)]
                          for p in centre]
    data["depth_profile"] = [round(float(course.depth_at(p[0], p[1])), 2)
                             for p in centre]

    print("race course %.0f m, %d stations, %d offsets"
          % (course.length, len(stations), len(offsets)))

    from coxswain.hydro.canopy import ShelteredWind
    from coxswain.river.structures import charles_structures
    structures = charles_structures()

    print("sampling the flattened fields")
    data["flat"] = {}
    for level in args.levels:
        depth, current, _b = flattened(course, None, stations, offsets,
                                       level)
        data["flat"]["depth|%.1f" % level] = depth
        if level == args.levels[0]:
            data["flat"]["current"] = current
    for direction in args.directions:
        field = ShelteredWind(structures, channel, 6.0, direction,
                              height=1.5)
        _d, _c, breeze = flattened(course, field, stations, offsets)
        data["flat"]["wind|%d" % int(direction)] = breeze
        print("  wind %d deg" % int(direction))

    print("running the route optimiser")
    speed_of = speed_table(drag, boat.shallow, boat.n_seats, 307.0)
    line, elapsed, length, yaw = optimised_line(course, boat, stations)
    data["line"] = [round(float(v), 2) for v in line]
    data["line_stats"] = {"elapsed": round(float(elapsed), 1),
                          "length": round(float(length), 1),
                          "peak_yaw": round(float(yaw), 2)}

    reference = RouteEvaluator(
        course, boat=boat, reference_speed=3.895).with_steering(
            ReducedModel()).evaluate(Route.centreline(course))
    data["centre_stats"] = {
        "elapsed": round(float(reference.elapsed), 1),
        "length": round(float(reference.path_length), 1),
        "peak_yaw": round(float(reference.peak_yaw_rate), 2)}

    data["yield_side"] = {}
    for level in args.levels:
        data["yield_side"]["%.1f" % level] = yield_side(
            course, speed_of, stations, level=level)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"))
    print()
    print("wrote %s (%.2f MB)" % (args.out,
                                  os.path.getsize(args.out) / 1e6))
    print("  centreline %.1f s over %.0f m, peak yaw %.2f deg/s"
          % (reference.elapsed, reference.path_length,
             reference.peak_yaw_rate))
    print("  optimised  %.1f s over %.0f m, peak yaw %.2f deg/s"
          % (elapsed, length, yaw))
    print("  SAVED      %.1f s" % (reference.elapsed - elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
