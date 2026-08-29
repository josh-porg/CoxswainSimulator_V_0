"""Does the wind change the line, or only the clock?

    python scripts/wind_scenarios.py
    python scripts/wind_scenarios.py --speeds 4 8 --directions 250 340 070

Wind is the largest uncontrolled term in this whole project -- 30 to 100
seconds at 6 m/s depending only on which way it blows -- and until the
field varied in space there was nothing to decide about it.  Now that the
sheltered side of the river carries about 80% of the wind the open side
does, there is: **is it worth steering a different line on a windy day,
or is the still-air line still the right one and you simply row it
slower?**

That is the question this answers, by optimising twice.

* ``naive`` -- the line optimised in still air, then scored under the
  wind.  What a crew does if the plan was made the week before.
* ``adapted`` -- the line optimised *with* the wind in the objective.
  What a crew could do if it decided on the morning.

The difference between them is the entire value of knowing the forecast,
and it is worth knowing whether that value is five seconds or fifty
before anybody spends a warm-up rearranging their race plan.

Pacing is the second half
-------------------------
A headwind makes the race longer, and the two-parameter model says the
optimal power is ``CP + W'/T`` -- so a longer race should be rowed at
slightly *lower* power, not higher.  Crews reliably do the opposite.  The
last table prices that mistake.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                        # noqa: E402
from coxswain.crew.exertion import (ROWER_ANAEROBIC_WORK,  # noqa: E402
                                    ROWER_CRITICAL_POWER, optimal_pace)
from coxswain.hydro.canopy import ShelteredWind            # noqa: E402
from coxswain.river import charles, lines                  # noqa: E402
from coxswain.river.charts import CourseGeometry           # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,   # noqa: E402
                                  optimise_route)
from coxswain.river.structures import charles_structures   # noqa: E402
from coxswain.river.trajectory import ReducedModel         # noqa: E402

RACE_LENGTH = 4822.0
#: Measured on this hull at masters speed by scripts/time_budget.py.
SPEED_EXPONENT = 0.498

#: Masters women's crew, per every other script here.
MASTERS = dict(rower_mass=72.0, rower_stature=1.72)


def build(reference_speed, field, course, flow, raster, gates, boat):
    ev = RouteEvaluator(course, flow=flow, reference_speed=reference_speed,
                        upstream=True, margin=4.0, minimum_depth=1.2,
                        n_samples=900)
    ev.with_steering(ReducedModel(), raster=raster, gates=gates)
    ev.with_exertion()
    if field is not None:
        ev.with_wind(field, boat=boat)
    return ev


def score(evaluation):
    return evaluation.elapsed_clean + 60.0 * evaluation.illegal_arches


def separation(a: Route, b: Route, course) -> float:
    """Mean distance between two lines, m -- how far the boat actually moves."""
    station = np.linspace(0.0, course.length, 600)
    return float(np.mean(np.abs(a.offset_at(station) - b.offset_at(station))))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speeds", type=float, nargs="+", default=[4.0, 8.0])
    parser.add_argument("--directions", type=float, nargs="+",
                        default=[250.0, 340.0, 70.0],
                        help="meteorological bearings the wind comes FROM")
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args(argv)

    speed = RACE_LENGTH / args.race_time
    raster = charles.charles_channel()
    _, _, race_line, _ = charles.hocr_course(raster)
    course = charles.charles_course(centreline=race_line, month=10)
    flow = charles.ContinuityFlow(course,
                                  discharge=charles.monthly_discharge(10))
    gates = CourseGeometry(channel=raster).gates_on_course()
    boat = catalog.eight(rate=28.0, **MASTERS)
    structures = charles_structures()

    start = lines.legalise(lines.arch_route(course, raster, gates, margin=4.0),
                           course, raster, gates, margin=4.0)

    print("optimising the still-air line ...")
    clock = time.time()
    calm = build(speed, None, course, flow, raster, gates, boat)
    best = optimise_route(calm, n_control=13, iterations=args.iterations,
                          seed=0, initial=start)
    still = Route(best.route.stations, best.route.offsets, name="still air")
    print("   %.1f s in still air  (%.0f s of wall clock)"
          % (score(calm.evaluate(still)), time.time() - clock))
    print()

    print("  %-18s %9s %9s %9s %8s %8s"
          % ("scenario", "naive", "adapted", "gain", "moves", "vs calm"))
    rows = []
    for wind_speed in args.speeds:
        for direction in args.directions:
            field = ShelteredWind(structures, raster, wind_speed, direction,
                                  height=0.43)
            ev = build(speed, field, course, flow, raster, gates, boat)
            naive = score(ev.evaluate(still))
            found = optimise_route(ev, n_control=13,
                                   iterations=args.iterations, seed=0,
                                   initial=still)
            adapted_route = Route(found.route.stations, found.route.offsets,
                                  name="adapted")
            adapted = score(ev.evaluate(adapted_route))
            moved = separation(still, adapted_route, course)
            calm_time = score(calm.evaluate(still))
            print("  %-18s %8.1fs %8.1fs %+8.1fs %7.1fm %+7.1fs"
                  % ("%.0f m/s from %03d" % (wind_speed, direction),
                     naive, adapted, naive - adapted, moved,
                     naive - calm_time))
            rows.append((wind_speed, direction, naive, adapted, moved))
    print()
    print("  'gain' is what re-optimising the line for the forecast buys.")
    print("  'moves' is how far the adapted line sits from the still-air")
    print("  one on average -- a large gain with a small move means the")
    print("  optimiser found shelter without giving up much river.")
    print()
    pacing(rows, args.race_time)
    return 0


def pacing(rows, calm_time):
    """What the wind does to the right race power, and to getting it wrong."""
    print("pacing: the wind makes the race longer, so the right power FALLS")
    print("  %-18s %9s %10s %12s %10s"
          % ("scenario", "race s", "best W", "if paced calm", "cost"))
    calm_power = optimal_pace(calm_time)
    for wind_speed, direction, naive, adapted, _moved in rows:
        duration = adapted
        best = optimal_pace(duration)
        # Pacing for the calm duration means spending W' at the calm rate,
        # which empties the reserve early on a longer race; the crew then
        # finishes on critical power alone.
        burn = ROWER_ANAEROBIC_WORK / max(calm_power - ROWER_CRITICAL_POWER,
                                          1.0)
        held = min(burn, duration)
        mean_power = ((calm_power * held
                       + ROWER_CRITICAL_POWER * max(duration - held, 0.0))
                      / duration)
        loss = -SPEED_EXPONENT * duration * (mean_power / best - 1.0)
        print("  %-18s %8.1f %10.1f %12.1f %+9.1fs"
              % ("%.0f m/s from %03d" % (wind_speed, direction), duration,
                 best, mean_power, loss))
    print()
    print("  the right power falls by under a watt across this whole table,")
    print("  because a head race is rowed only ~4%% above critical power and")
    print("  there is very little reserve to misallocate.  Pacing to the")
    print("  wind is not where the seconds are; steering to it might be.")


if __name__ == "__main__":
    raise SystemExit(main())
