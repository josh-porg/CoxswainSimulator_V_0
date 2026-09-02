r"""Export the course, its fields, and the optimum line, for the map page.

    python scripts/export_map.py --out out/report/map_data.json

The report has tables and static figures. What it does not have is the
thing a coxswain actually wants to interrogate: **the river, with a layer
switch and a slider**, so that "what happens to the line if it is a dry
year" is a drag rather than a rerun.

This produces the data for that. Everything is precomputed here, in
Python, against the real survey; the page only draws.

What "the optimum line" means here
----------------------------------
A **greedy depth-following line**, not the full optimum. At each station
it picks the lateral offset with the fastest achievable speed, subject to
staying in the channel, then smooths the result to something a boat could
actually steer.

That is a deliberate and important restriction. The true optimum trades
depth against **curvature** -- a line that darts sideways for deep water
loses more to steering than the depth is worth, which is what
:mod:`coxswain.river.route` and the MPC work exist to resolve. The greedy
line ignores that entirely, so it is an **upper bound on how much the
line should move**, and the honest way to read it is as *where the good
water is*, not *where to point the boat*. The page says so.

Why precompute a grid rather than solve live
--------------------------------------------
The speed at a point depends on depth through the shallow-water factor,
which is solved by bisection; doing that in JavaScript for every pixel on
every slider drag would be slow and would duplicate physics that is
already tested here. So the parameter space is swept once -- water level
against wind -- and the page interpolates between stored lines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river.charles import (charles_channel,  # noqa: E402
                                    charles_course)

from course_pacing import build_boat, hull_drag                # noqa: E402

GRAVITY = 9.80665


def field_grids(course, channel, step: int = 3):
    """Depth, current and shelter, on a coarsened raster.

    The survey raster is 6 m; every third cell is 18 m, which is finer
    than any of these fields varies and keeps the page under a megabyte.
    """
    water = channel.water[::step, ::step]
    east = channel.east[::step]
    north = channel.north[::step]
    rows, columns = water.shape

    depth = np.full((rows, columns), np.nan)
    current = np.full((rows, columns), np.nan)
    for i in range(rows):
        for j in range(columns):
            if not water[i, j]:
                continue
            x, y = float(east[j]), float(north[i])
            depth[i, j] = float(course.depth_at(x, y))
            velocity = np.asarray(course.current_at(x, y))[:2]
            current[i, j] = float(np.hypot(*velocity))
    return {
        "east": [round(float(v), 1) for v in east],
        "north": [round(float(v), 1) for v in north],
        "water": water.astype(int).tolist(),
        "depth": [[None if np.isnan(v) else round(float(v), 2)
                   for v in row] for row in depth],
        "current": [[None if np.isnan(v) else round(float(v), 4)
                     for v in row] for row in current],
    }


def wind_grid(structures, channel, speed, direction, step: int = 3):
    """Sheltered wind speed at chest height, same grid as the others."""
    from coxswain.hydro.canopy import ShelteredWind

    field = ShelteredWind(structures, channel, speed, direction, height=1.5)
    water = channel.water[::step, ::step]
    east = channel.east[::step]
    north = channel.north[::step]
    out = np.full(water.shape, np.nan)
    for i in range(water.shape[0]):
        for j in range(water.shape[1]):
            if water[i, j]:
                out[i, j] = float(field.speed_at(float(east[j]),
                                                 float(north[i])))
    return [[None if np.isnan(v) else round(float(v), 2) for v in row]
            for row in out]


def speed_table(drag, shallow, rowers, power,
                depths=None, headwinds=None):
    """``(depth, headwind) -> speed`` as a bilinear interpolator.

    Solving the power balance point by point is what an earlier version
    did, and it needed **over two million** ``hull_resistance`` calls to
    fill the parameter grid.  The relation is smooth and two-dimensional,
    so it is solved once on a grid and interpolated -- the same treatment
    ``RouteEvaluator.speed_through_water`` and ``CoursePacing`` already
    give it, for the same reason.
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

    # Cache the hull curve once; it does not depend on depth or wind.
    probe = np.arange(0.2, 9.01, 0.02)
    hull = np.array([drag(float(v)) for v in probe])

    def hull_at(speed):
        return float(np.interp(speed, probe, hull))

    grid = np.empty((len(depths), len(headwinds)))
    for i, depth in enumerate(depths):
        model = replace(template, depth=float(max(depth, 0.30)))
        factor = np.array([float(model.factor(float(v))) for v in probe])

        def excess(speed, _f=factor):
            resist = hull_at(speed) * float(np.interp(speed, probe, _f))
            return resist * speed

        for j, headwind in enumerate(headwinds):
            low, high = 0.2, 9.0
            for _ in range(40):
                mid = 0.5 * (low + high)
                apparent = mid + headwind
                air = 0.5 * 1.225 * 3.22 * apparent * abs(apparent)
                if excess(mid) + air * mid < delivered:
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
        return float((1-fi)*(1-fj)*grid[i0, j0] + fi*(1-fj)*grid[i1, j0]
                     + (1-fi)*fj*grid[i0, j1] + fi*fj*grid[i1, j1])
    return lookup


def greedy_line(course, speed_of, level_offset, wind_field,
                stations, offsets, smooth=9):
    """Best offset at each station, then smoothed to something steerable."""
    best = np.zeros(len(stations))
    for index, station in enumerate(stations):
        limit = max(float(course.half_width_at(station)) - 1.5, 0.5)
        candidates = offsets[np.abs(offsets) <= limit]
        if not len(candidates):
            candidates = np.array([0.0])
        heading = float(course.heading_at(station))
        tangent = np.array([np.cos(heading), np.sin(heading)])
        speeds = []
        for offset in candidates:
            point = course.offset_position(station, offset)
            point = np.atleast_1d(np.asarray(point, float)).ravel()[:2]
            depth = float(course.depth_at(point[0], point[1])) + level_offset
            headwind = 0.0
            if wind_field is not None:
                local = float(wind_field.speed_at(point[0], point[1]))
                headwind = -local * float(np.dot(wind_field._towards,
                                                 tangent))
            speeds.append(speed_of(depth, headwind))
        best[index] = float(candidates[int(np.argmax(speeds))])

    # A boat cannot dart; smooth over roughly a hundred metres.
    kernel = np.ones(smooth) / smooth
    padded = np.pad(best, smooth, mode="edge")
    return np.convolve(padded, kernel, mode="same")[smooth:-smooth]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/report/map_data.json")
    parser.add_argument("--boat", default="4+", choices=["8+", "4+"])
    parser.add_argument("--stations", type=int, default=140)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--levels", type=float, nargs="+",
                        default=[-0.6, -0.3, 0.0, 0.3, 0.6])
    parser.add_argument("--winds", type=float, nargs="+",
                        default=[0.0, 4.0, 8.0])
    parser.add_argument("--directions", type=float, nargs="+",
                        default=[250.0, 340.0, 70.0, 160.0])
    args = parser.parse_args(argv)

    boat = build_boat(args.boat, 30.0 if args.boat == "4+" else 32.0)
    drag = hull_drag(boat)
    course = charles_course()
    channel = charles_channel()
    power = 307.0

    print("exporting fields")
    data = field_grids(course, channel, args.step)
    data["resolution"] = channel.resolution * args.step
    data["boat"] = args.boat
    # The page needs the axes it is sliding along, not just the
    # lines they index.
    data["levels"] = [float(v) for v in args.levels]
    data["winds"] = [float(v) for v in args.winds]
    data["directions"] = [int(v) for v in args.directions]
    data["length"] = course.length

    stations = np.linspace(0.0, course.length, args.stations)
    data["stations"] = [round(float(s), 1) for s in stations]
    centre = np.array([course.position_at(s) for s in stations])
    data["centreline"] = [[round(float(p[0]), 1), round(float(p[1]), 1)]
                          for p in centre]
    data["half_width"] = [round(float(course.half_width_at(s)), 1)
                          for s in stations]
    data["depth_profile"] = [
        round(float(course.depth_at(p[0], p[1])), 2) for p in centre]

    print("exporting wind fields")
    from coxswain.river.structures import charles_structures
    structures = charles_structures()
    data["wind"] = {}
    for direction in args.directions:
        key = "%d" % int(direction)
        data["wind"][key] = wind_grid(structures, channel, 6.0, direction,
                                      args.step)
        print("  %s deg" % key)

    print("tabulating speed against depth and headwind")
    speed_of = speed_table(drag, boat.shallow, boat.n_seats, power)

    print("solving greedy lines over the parameter grid")
    offsets = np.linspace(-45.0, 45.0, 61)
    data["lines"] = {}
    from coxswain.hydro.canopy import ShelteredWind
    for level in args.levels:
        for speed in args.winds:
            for direction in (args.directions if speed > 0
                              else [args.directions[0]]):
                field = (ShelteredWind(structures, channel, speed, direction,
                                       height=1.5) if speed > 0 else None)
                line = greedy_line(course, speed_of, level, field,
                                   stations, offsets)
                key = "%.1f|%.1f|%d" % (level, speed, int(direction))
                data["lines"][key] = [round(float(v), 2) for v in line]
                print("  level %+.1f m, wind %.0f m/s from %d -> "
                      "mean |offset| %.1f m"
                      % (level, speed, int(direction),
                         float(np.mean(np.abs(line)))))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"))
    size = os.path.getsize(args.out) / 1e6
    print()
    print("wrote %s (%.2f MB)" % (args.out, size))
    print("  %d stations, %d lines, %d wind fields"
          % (len(stations), len(data["lines"]), len(data["wind"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
