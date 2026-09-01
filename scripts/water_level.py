r"""Race the same river at different water levels.

    python scripts/water_level.py
    python scripts/water_level.py --levels -0.6 -0.3 0.0 0.2 --statistic p10

The bathymetry this project rests on is a 2016-17 survey referenced to the
basin's normal pool.  The New Charles River Dam holds that pool nearly
constant, which is why a single survey has been usable at all -- but
"nearly" does real work in a dry autumn, and 2026 is being reported as a
low year.  A scenario knob is worth more than an assumption, and the race
is close enough that the estimate will improve before it matters.

A lower level does three separate things, and only the first is obvious
--------------------------------------------------------------------------
**Shallower water is slower.**  The shallow-water wave-resistance term
already in the model handles that, and it is the largest single
environmental number in this project -- 82 s between the shallowest and
deepest plausible course.

**The navigable channel narrows.**  The channel is extracted by
thresholding depth, so dropping the pool moves the navigable boundary
inwards.  That changes what lines are *available*, not just what they
cost, and it is why this sweep re-optimises rather than re-scoring.

**The wave regime moves.**  Critical depth at racing speed is 1.82 m.  A
line that sits comfortably subcritical at normal pool can be pushed
towards the transcritical band by half a metre of drawdown, and there the
linear wake theory this project uses stops describing the physics.

Discharge is a separate lever
-----------------------------
A dry year is usually also a low-flow year, which *helps* a crew rowing
upstream.  Those two pull opposite ways and the model already carries
both: ``--statistic p10`` is the dry-year discharge against ``median``.
Do not vary depth without saying what you have assumed about flow.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river import charles, lines                # noqa: E402
from coxswain.river.charts import CourseGeometry         # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,  # noqa: E402
                                  optimise_route)
from coxswain.river.trajectory import ReducedModel       # noqa: E402

RACE_LENGTH = 4822.0
GRAVITY = 9.80665


def scenario(level, statistic, speed, iterations, month=10):
    """Optimise a line for one water level, and describe what it meets."""
    raster = charles.charles_channel(level_offset=level)
    _, _, race_line, _ = charles.hocr_course(raster)
    course = charles.charles_course(centreline=race_line, month=month,
                                    statistic=statistic, level_offset=level)
    flow = charles.ContinuityFlow(
        course, discharge=charles.monthly_discharge(month, statistic))
    gates = CourseGeometry(channel=raster).gates_on_course()

    evaluator = RouteEvaluator(course, flow=flow, reference_speed=speed,
                               upstream=True, margin=4.0, minimum_depth=1.2,
                               n_samples=900)
    evaluator.with_steering(ReducedModel(), raster=raster, gates=gates)
    evaluator.with_exertion()
    seed = lines.legalise(lines.arch_route(course, raster, gates, 4.0),
                          course, raster, gates, 4.0)
    best = optimise_route(evaluator, n_control=13, iterations=iterations,
                          seed=0, initial=seed)
    route = Route(best.route.stations, best.route.offsets,
                  name="level %+.2f" % level)
    result = evaluator.evaluate(route)

    station = np.linspace(0.0, course.length, 900)
    points = course.offset_position(station, route.offset_at(station))
    depth = np.maximum(np.asarray(course.depth(points[:, 0], points[:, 1]),
                                  dtype=float), 0.15)
    froude = speed / np.sqrt(GRAVITY * depth)
    return dict(level=level, route=route, course=course,
                time=result.elapsed_clean + 60.0 * result.illegal_arches,
                distance=result.path_length,
                navigable=raster.navigable_area,
                min_depth=float(depth.min()),
                median_depth=float(np.median(depth)),
                peak_froude=float(froude.max()),
                near_critical=float(np.mean(froude > 0.9)),
                discharge=charles.monthly_discharge(month, statistic))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=float, nargs="+",
                        default=[-0.6, -0.3, 0.0, 0.2])
    parser.add_argument("--statistic", default="median",
                        choices=("mean", "p10", "median", "p90", "min", "max"))
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--iterations", type=int, default=35)
    args = parser.parse_args(argv)

    speed = RACE_LENGTH / args.race_time
    print("water level scenarios, %s October discharge (%.1f m3/s)"
          % (args.statistic, charles.monthly_discharge(10, args.statistic)))
    print("  critical depth at %.2f m/s is %.2f m"
          % (speed, speed ** 2 / GRAVITY))
    print()

    rows = []
    for level in args.levels:
        print("  optimising at %+.2f m ..." % level)
        rows.append(scenario(level, args.statistic, speed, args.iterations))

    base = next((r for r in rows if abs(r["level"]) < 1e-9), rows[0])
    print()
    print("  %-8s %9s %9s %10s %9s %9s %9s"
          % ("level m", "time s", "vs pool", "navigable", "min dep",
             "med dep", "peak Fh"))
    for row in rows:
        print("  %-8.2f %9.1f %+9.1f %9.0fk %9.2f %9.2f %9.2f"
              % (row["level"], row["time"], row["time"] - base["time"],
                 row["navigable"] / 1000.0, row["min_depth"],
                 row["median_depth"], row["peak_froude"]))
    print()

    span = max(r["time"] for r in rows) - min(r["time"] for r in rows)
    print("  %.1f s across %.1f m of level -- about %.0f s per 100 mm."
          % (span, max(r["level"] for r in rows) - min(r["level"]
                                                       for r in rows),
             span / (10.0 * (max(r["level"] for r in rows)
                             - min(r["level"] for r in rows)))))
    print()

    print("does the LINE move, or only the clock?")
    station = np.linspace(0.0, base["course"].length, 600)
    for row in rows:
        if row is base:
            continue
        shift = float(np.mean(np.abs(row["route"].offset_at(station)
                                     - base["route"].offset_at(station))))
        print("  %+.2f m: line moves %.2f m on average, %.2f m at most"
              % (row["level"], shift,
                 float(np.max(np.abs(row["route"].offset_at(station)
                                     - base["route"].offset_at(station))))))
    print()
    worst = max(rows, key=lambda r: r["peak_froude"])
    print("  the shallowest scenario peaks at Fh = %.2f (%.0f%% of the line"
          % (worst["peak_froude"], 100 * worst["near_critical"]))
    print("  above 0.9).  Critical is 1.0, where the wake stops being")
    print("  described by linear theory and the wave drag peaks hard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
