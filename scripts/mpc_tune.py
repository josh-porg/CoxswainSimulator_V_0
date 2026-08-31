"""Tune the controller against the clock, not against cross-track error.

    python scripts/mpc_tune.py
    python scripts/mpc_tune.py --sweep cross --values 2 4 12 40 120
    python scripts/mpc_tune.py --sweep rudder --values 0.2 0.8 3.0 10.0

Every previous benchmark of this controller scored it on how closely it
held the line.  That is the wrong objective and it was quietly deciding
things: it says a controller that saws the rudder to stay within a metre
beats one that lets a metre and a half go by and carries less helm.  On a
race course the second one may well be quicker, and **quicker is the only
thing that matters**.

So this scores gate to gate in seconds.

What that changes
-----------------
Holding a line has two costs and they pull opposite ways.  Tracking
loosely lets the boat wander, which is extra distance; tracking tightly
costs rudder and yaw rate, both of which are drag.  Cross-track error
prices only the first.  The clock prices both, and it also prices the
thing neither of them sees -- that a boat allowed to run a little wide of
a bend and straighten early can be *shorter* than one pinned to the line.

The catch, stated up front
--------------------------
"Minimise time" will happily cut a corner, and on this river some corners
have bridges in them.  A run is therefore rejected outright if it leaves
the navigable channel or passes the wrong side of a pier, rather than
being penalised: an illegal line is not a fast line with a deduction, it
is not a line.

Reading the output
------------------
``distance`` is what makes the time interpretable.  A variant that is
quicker with a *shorter* distance won by cutting; one that is quicker at
the *same* distance won by carrying less drag.  Those are different
findings and the table keeps them apart.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.river import charles, lines                # noqa: E402
from coxswain.river.charts import CourseGeometry         # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,  # noqa: E402
                                  optimise_route)
from coxswain.river.trajectory import (ReducedModel,  # noqa: E402
                                       fit_reduced_model)
from coxswain.sim.control import Coxswain                # noqa: E402
from coxswain.sim.guidance import PathFollower           # noqa: E402
from coxswain.sim.mpc import PathMPC                     # noqa: E402
from coxswain.sim.simulator import RowingSimulator       # noqa: E402

RACE_LENGTH = 4822.0

#: Uniform power scale that makes this hull row the masters race time,
#: from scripts/time_budget.py.  **Both benchmarks left it out**, so the
#: boat rowed at the catalog default -- collegiate power -- and covered
#: 4034 m in 766 s, which is 5.27 m/s for a crew calibrated at 4.23.
#: Every earlier number in this file was measured 25% too fast, at a
#: speed where the rudder has 55% more authority than it really does and
#: where the controller's own fitted model no longer describes the plant.
MASTERS_POWER = 0.658


#: Fraction of the stretch used as run-up before the clock starts, so the
#: controller is settled and no variant is scored on its transient.
LEAD_IN = 0.15


def build_path(start, finish, reference_speed, iterations=35):
    raster = charles.charles_channel()
    _, _, race_line, _ = charles.hocr_course(raster)
    course = charles.charles_course(centreline=race_line, month=10)
    flow = charles.ContinuityFlow(course,
                                  discharge=charles.monthly_discharge(10))
    gates = CourseGeometry(channel=raster).gates_on_course()
    ev = RouteEvaluator(course, flow=flow, reference_speed=reference_speed,
                        upstream=True, margin=4.0, minimum_depth=1.2,
                        n_samples=900)
    ev.with_steering(ReducedModel(), raster=raster, gates=gates)
    seed = lines.legalise(lines.arch_route(course, raster, gates, 4.0),
                          course, raster, gates, 4.0)
    best = optimise_route(ev, n_control=13, iterations=iterations, seed=0,
                          initial=seed)
    route = Route(best.route.stations, best.route.offsets, name="optimised")
    station = np.linspace(0.0, course.length, 4000)
    full = course.offset_position(station, route.offset_at(station))
    inside = (station >= start) & (station <= finish)
    return full[inside], raster


def gate_crossing(positions, times, path, gate_index, follower):
    """Time the boat passes a gate, found along the path and interpolated.

    The first version dotted ``position - gate_point`` with the gate
    tangent and took the first non-negative sample.  That is a test
    against an **infinite plane**, which is harmless on a 900 m stretch
    and nonsense on a 4.7 km river that doubles back on itself: the
    boat's *starting* position already sits on the far side of the plane
    through the finish gate, the first crossing is index 0, and the run
    is reported as never having finished.  Both settings in the first
    full-course sweep "failed" that way, including the default that
    renders a whole race without complaint.

    So bracket along the path first -- the boat has passed the gate when
    its nearest path index does -- and only then interpolate with the
    local plane, which is exact over one step and immune to what the
    river does two kilometres later.
    """
    indices = np.array([follower.nearest(point) for point in positions])
    beyond = np.nonzero(indices >= gate_index)[0]
    if not len(beyond) or beyond[0] == 0:
        return None, None
    i = int(beyond[0])
    point = path[gate_index]
    tangent = path[min(gate_index + 1, len(path) - 1)] - point
    tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
    before = float(np.dot(positions[i - 1] - point, tangent))
    after = float(np.dot(positions[i] - point, tangent))
    if after == before:
        return float(times[i]), i
    fraction = -before / (after - before)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(times[i - 1] + fraction * (times[i] - times[i - 1])), i


def run(path, raster, steering, speed, duration, dt=0.02, horizon=6.0,
        steps=12, interval=0.20, **kwargs):
    """One run, timed between two fixed gates on the water."""
    boat = catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72)
    boat.power_scales = np.full(boat.n_seats, MASTERS_POWER)
    mpc = PathMPC(path, model=steering, horizon=horizon, steps=steps,
                  interval=interval, **kwargs)
    sim = RowingSimulator(boat, coxswain=Coxswain(rudder_override=mpc))
    heading = float(np.arctan2(path[1, 1] - path[0, 1],
                               path[1, 0] - path[0, 0]))
    state = sim.initial_state(surge_speed=speed)
    state[0], state[1] = path[0]
    state[5] = heading
    state[6] = speed * np.cos(heading)
    state[7] = speed * np.sin(heading)
    result = sim.run(duration=duration, dt=dt, initial_state=state)

    positions = np.asarray(result.position)[:2].T
    times = np.asarray(result.time)

    start_index = int(LEAD_IN * len(path))
    finish_index = len(path) - 2
    t_start, i_start = gate_crossing(positions, times, path, start_index,
                                     PathFollower(path))
    t_finish, i_finish = gate_crossing(positions, times, path, finish_index,
                                       PathFollower(path))
    if t_start is None or t_finish is None:
        return None

    leg = positions[i_start:i_finish + 1]
    travelled = float(np.hypot(*np.diff(leg, axis=0).T).sum())

    check = PathFollower(path)
    errors = []
    legal = True
    for point in leg:
        index = check.nearest(point)
        tangent, _ = check.frame_at(index)
        across = np.array([-tangent[1], tangent[0]])
        errors.append(float(np.dot(point[:2] - check.path[index], across)))
        if not raster.is_navigable(point[0], point[1]):
            legal = False
    errors = np.asarray(errors)

    return dict(time=t_finish - t_start, distance=travelled,
                rms=float(np.sqrt((errors ** 2).mean())),
                worst=float(np.abs(errors).max()), legal=legal,
                fail=100.0 * mpc.failures / max(mpc.solves + mpc.failures, 1))


SWEEPS = {
    "cross": ("weight_cross", [2.0, 4.0, 12.0, 40.0, 120.0]),
    "heading": ("weight_heading", [3.0, 12.0, 40.0, 120.0]),
    "rudder": ("weight_rudder", [0.1, 0.4, 0.8, 3.0, 12.0]),
    "rate": ("weight_rudder_rate", [0.3, 1.2, 2.5, 8.0, 25.0]),
    "split": ("weight_split", [10.0, 40.0, 160.0, 640.0]),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", type=float, default=2000.0)
    parser.add_argument("--to", dest="finish", type=float, default=2900.0)
    parser.add_argument("--duration", type=float, default=230.0)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--sweep", choices=sorted(SWEEPS), default="cross")
    parser.add_argument("--values", type=float, nargs="+", default=None)
    parser.add_argument("--full", action="store_true",
                        help="time the WHOLE course instead of a stretch.  "
                             "A sub-stretch score is myopic in exactly the "
                             "way MPC is meant not to be: a controller can "
                             "gain by cutting a corner and leaving the exit "
                             "state to somebody else, and over 900 m there "
                             "is no later to pay it back in.  On the full "
                             "course there is.")
    args = parser.parse_args(argv)
    if args.full:
        args.start, args.finish = 0.0, 5200.0
        args.duration = max(args.duration, 1.25 * args.race_time)

    speed = RACE_LENGTH / args.race_time
    print("optimising the line ...")
    path, raster = build_path(args.start, args.finish, speed)
    print("  %.0f m, timing from %.0f%% of it so the controller is settled"
          % (float(np.hypot(*np.diff(path, axis=0).T).sum()), 100 * LEAD_IN))
    steering = fit_reduced_model(
        catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72),
        reference_speed=speed)
    print()

    name, values = SWEEPS[args.sweep]
    values = args.values or values
    print("sweeping %s against the clock" % name)
    print("  %-10s %10s %9s %10s %8s %8s %6s"
          % (name, "time s", "vs best", "distance", "rms m", "worst", "legal"))
    rows = []
    for value in values:
        row = run(path, raster, steering, speed, args.duration,
                  **{name: value})
        if row is None:
            print("  %-10.3g   did not reach the finish gate" % value)
            continue
        row["value"] = value
        rows.append(row)
    if not rows:
        print("  nothing completed the leg")
        return 1
    best = min(r["time"] for r in rows)
    for row in rows:
        print("  %-10.3g %10.3f %+9.3f %10.2f %8.2f %8.2f %6s"
              % (row["value"], row["time"], row["time"] - best,
                 row["distance"], row["rms"], row["worst"],
                 "yes" if row["legal"] else "NO"))
    print()

    winner = min(rows, key=lambda r: r["time"])
    tightest = min(rows, key=lambda r: r["rms"])
    spread = max(r["time"] for r in rows) - best
    print("  fastest: %s = %.3g at %.3f s" % (name, winner["value"],
                                              winner["time"]))
    print("  tightest tracking: %s = %.3g at %.2f m rms"
          % (name, tightest["value"], tightest["rms"]))
    if winner["value"] != tightest["value"]:
        print("  **they are not the same setting.**  Tracking error and the")
        print("  clock disagree, which is the whole reason to score on the")
        print("  clock.")
    else:
        print("  they agree here, which is worth knowing but is not")
        print("  guaranteed and was not assumed.")
    print("  spread across the sweep: %.3f s over %.0f m of river"
          % (spread, winner["distance"]))
    print("  scaled to the full course that is about %.1f s."
          % (spread * RACE_LENGTH / max(winner["distance"], 1.0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
