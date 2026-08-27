"""Simulate the boat rowing the optimised line, and animate it.

    python scripts/animate_race.py --from 2100 --to 2800
    python scripts/animate_race.py --from 3600 --to 4200 --view course

Optimises a racing line, hands it to the full 6-DOF boat under
line-of-sight guidance, and draws the result as a video: the hull to
scale with its blades working, against the surveyed bank, the arches and
piers, and the boathouses a crew steers by.

``--from`` and ``--to`` are metres from the start line.  A whole race is
about 1000 seconds and 25,000 frames, which is neither quick to simulate
nor pleasant to watch, so pick the stretch that matters -- 2100 to 2800
covers the Weeks turn and Anderson, which is where the steering decisions
are.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.river import animate, charles             # noqa: E402
from coxswain.river.charts import CourseGeometry        # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,  # noqa: E402
                                  optimise_route)
from coxswain.river.trajectory import ReducedModel      # noqa: E402
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.guidance import PathFollower          # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", type=float, default=2100.0,
                        help="metres from the start line (default 2100)")
    parser.add_argument("--to", dest="finish", type=float, default=2800.0,
                        help="metres from the start line (default 2800)")
    parser.add_argument("--view", choices=("chase", "course"),
                        default="chase")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speed", type=float, default=8.0,
                        help="playback speed against real time")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--out", default="out/animation")
    args = parser.parse_args(argv)

    raster = charles.charles_channel()
    _, _, race_line, _ = charles.hocr_course(raster)
    course = charles.charles_course(centreline=race_line, month=10)
    flow = charles.ContinuityFlow(course,
                                  discharge=charles.monthly_discharge(10))
    gates = CourseGeometry(channel=raster).gates_on_course()

    print("optimising the line ...")
    ev = RouteEvaluator(course, flow=flow, reference_speed=5.2,
                        upstream=True, margin=4.0, minimum_depth=1.2,
                        n_samples=1200)
    ev.with_steering(ReducedModel(), raster=raster, gates=gates)
    ev.with_exertion()
    best = optimise_route(ev, n_control=13, iterations=70, seed=0)
    route = Route(best.route.stations, best.route.offsets, name="optimised")

    station = np.linspace(0.0, course.length, 4000)
    full_path = course.offset_position(station, route.offset_at(station))
    inside = (station >= args.start) & (station <= args.finish)
    path = full_path[inside]
    if len(path) < 10:
        raise SystemExit("that stretch is too short to animate")

    # Long enough to cover the leg even if the boat is slower than planned.
    leg = float(np.hypot(*np.diff(path, axis=0).T).sum())
    duration = 1.15 * leg / 4.6

    boat = catalog.eight(rate=28.0)
    follower = PathFollower(path, boundary_layer=25.0, gain=2.5,
                            rate_gain=1.2)
    sim = RowingSimulator(boat, coxswain=Coxswain(rudder_override=follower))
    heading = float(np.arctan2(path[8, 1] - path[0, 1],
                               path[8, 0] - path[0, 0]))
    state = sim.initial_state(surge_speed=4.7)
    state[0], state[1] = path[0]
    state[5] = heading

    print("simulating %.0f m of racing (%.0f s) ..." % (leg, duration))
    clock = time.time()
    result = sim.run(duration=duration, dt=args.dt, initial_state=state)
    print("   %.0f s of wall clock" % (time.time() - clock))

    positions = np.asarray(result.position)[:2].T
    headings = np.unwrap(np.asarray(result.attitude)[2])
    times = np.asarray(result.time)

    # Stop at the end of the line.  Run on past it and the guidance starts
    # steering at a point *behind* the boat -- the nearest path point can
    # only ever be the last one -- and it wanders off looking for a line
    # that has run out.  That produced a 47 m cross-track error and an
    # animation of a boat apparently losing its mind, from nothing worse
    # than a generous duration.
    finish = np.linalg.norm(positions - path[-1], axis=1)
    arrived = np.nonzero(finish < 12.0)[0]
    if len(arrived):
        cut = int(arrived[0]) + 1
        positions, headings, times = positions[:cut], headings[:cut], times[:cut]
        print("   reached the end of the leg at %.0f s" % times[-1])

    # cross-track, recomputed so the caption can show it
    check = PathFollower(path)
    errors = []
    for point in positions:
        index = check.nearest(point)
        tangent, _ = check.frame_at(index)
        across = np.array([-tangent[1], tangent[0]])
        errors.append(float(np.dot(point[:2] - check.path[index], across)))
    errors = np.asarray(errors)
    print("   cross-track rms %.2f m, worst %.2f m"
          % (np.sqrt((errors ** 2).mean()), np.abs(errors).max()))

    stride = max(int(round(args.speed / (args.fps * args.dt))), 1)
    figure, update, frames = animate.animate_run(
        positions, headings, path, raster, gates, path_length=leg,
        view=args.view, stride=stride, times=times,
        period=boat.timing.period, cross_track=errors)

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    target = os.path.join(args.out, "race_%04d_%04d_%s"
                          % (args.start, args.finish, args.view))
    print("writing %d frames ..." % frames)
    written = animate.write_animation(figure, update, frames, target,
                                      fps=args.fps)
    print("wrote", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
