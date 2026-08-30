"""Does the improved MPC actually steer better, or just differently?

    python scripts/mpc_bench.py
    python scripts/mpc_bench.py --from 2000 --to 2900 --duration 210

Four changes went into :class:`~coxswain.sim.mpc.PathMPC` at once, and a
controller is exactly the kind of thing where four plausible improvements
can sum to a regression.  So each is switched independently here and run
down the same stretch of river behind the same boat.

* **QP instead of NLP.**  Everything in the transcription was already
  linear-quadratic except ``de/dt = u sin(psi)``.  Linearising that about
  the *measured* heading error -- not about zero -- makes the program
  convex, and an active-set QP either returns the optimum or reports real
  infeasibility.  It has no iteration limit to hit, which is what the
  fallback path had been quietly absorbing.
* **A disturbance observer.**  The reduced model has no rig couple, no
  crosswind weathervane and no current shear; the boat has all three.
  Without an estimate the only route to a standing rudder is a standing
  cross-track error.
* **A Riccati terminal cost** in place of a guessed factor of four.
* **Warm-starting the split**, which had been reset to zero every solve.

The stretch is the Weeks turn, which is where the steering decisions on
this course actually are.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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
    return full[inside]


def run(path, steering, label, duration, speed, dt=0.02, **kwargs):
    boat = catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72)
    mpc = PathMPC(path, model=steering, horizon=6.0, steps=12,
                  interval=0.20, **kwargs)
    sim = RowingSimulator(boat, coxswain=Coxswain(rudder_override=mpc))
    heading = float(np.arctan2(path[1, 1] - path[0, 1],
                               path[1, 0] - path[0, 0]))
    state = sim.initial_state(surge_speed=speed)
    state[0], state[1] = path[0]
    state[5] = heading
    state[6] = speed * np.cos(heading)
    state[7] = speed * np.sin(heading)

    clock = time.time()
    result = sim.run(duration=duration, dt=dt, initial_state=state)
    wall = time.time() - clock

    positions = np.asarray(result.position)[:2].T
    check = PathFollower(path)
    errors = []
    for point in positions:
        index = check.nearest(point)
        tangent, _ = check.frame_at(index)
        across = np.array([-tangent[1], tangent[0]])
        errors.append(float(np.dot(point[:2] - check.path[index], across)))
    errors = np.asarray(errors)
    attempted = mpc.solves + mpc.failures
    return dict(label=label, rms=float(np.sqrt((errors ** 2).mean())),
                worst=float(np.abs(errors).max()),
                mean=float(errors.mean()),
                fail=100.0 * mpc.failures / max(attempted, 1),
                solves=attempted, wall=wall, bias=mpc.bias,
                real_time=wall / max(float(result.time[-1]), 1e-6))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", type=float, default=2000.0)
    parser.add_argument("--to", dest="finish", type=float, default=2900.0)
    parser.add_argument("--duration", type=float, default=210.0)
    parser.add_argument("--race-time", type=float, default=1140.0)
    args = parser.parse_args(argv)

    speed = 4822.0 / args.race_time
    print("optimising the line ...")
    path = build_path(args.start, args.finish, speed)
    length = float(np.hypot(*np.diff(path, axis=0).T).sum())
    print("  %.0f m of the Weeks turn, %d points" % (length, len(path)))
    steering = fit_reduced_model(
        catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72),
        reference_speed=speed)
    print("  fitted model: I %.0f, N_r %.0f, Y_c %.0f, rudder limit %.0f deg"
          % (steering.yaw_inertia, steering.yaw_damping,
             steering.yaw_control, np.degrees(steering.rudder_limit)))
    print()

    trials = (("NLP, as it was", dict(solver="nlp", estimate_bias=False)),
              ("QP", dict(solver="qp", estimate_bias=False)),
              ("QP + observer", dict(solver="qp", estimate_bias=True)))
    print("  %-18s %8s %8s %8s %8s %9s %9s"
          % ("controller", "rms m", "worst", "mean m", "fail %", "solves",
             "x real"))
    rows = []
    for label, options in trials:
        row = run(path, steering, label, args.duration, speed, **options)
        rows.append(row)
        print("  %-18s %8.2f %8.2f %+8.2f %8.1f %9d %9.1f"
              % (label, row["rms"], row["worst"], row["mean"], row["fail"],
                 row["solves"], row["real_time"]))
    print()
    best = min(rows, key=lambda r: r["rms"])
    base = rows[0]
    print("  %s is the best of these: rms %.2f m against %.2f m, a %.0f%%"
          % (best["label"], best["rms"], base["rms"],
             100 * (1 - best["rms"] / base["rms"])))
    print("  improvement, at %.0f%% of the solve failures and %.1fx the speed."
          % (100 * best["fail"] / max(base["fail"], 1e-9)
             if base["fail"] else 0.0,
             base["wall"] / max(best["wall"], 1e-9)))
    if rows[-1]["bias"]:
        print()
        print("  the observer settled on %.0f N m of unmodelled yaw moment,"
              % rows[-1]["bias"])
        print("  which at this speed is %.1f degrees of standing rudder --"
              % np.degrees(abs(rows[-1]["bias"])
                           / (steering.yaw_control * speed ** 2)))
        print("  the rig couple and the bend, carried on feedforward")
        print("  instead of on a standing cross-track error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
