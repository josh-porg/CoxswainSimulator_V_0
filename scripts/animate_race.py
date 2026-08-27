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
from coxswain.river.trajectory import (ReducedModel,  # noqa: E402
                                       fit_reduced_model)
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.guidance import PathFollower          # noqa: E402
from coxswain.sim.mpc import PathMPC                    # noqa: E402
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
    parser.add_argument("--dt", type=float, default=0.01,
                        help="simulation step; 0.02 halves the cost of a "
                             "full-course pass and the boat's yaw time "
                             "constant is 0.06 s, so it is still resolved")
    parser.add_argument("--full", action="store_true",
                        help="the whole race, start line to finish")
    parser.add_argument("--lead-in", dest="lead", type=float, default=140.0,
                        help="metres of run-up simulated before the "
                             "animated stretch, so the boat is already "
                             "tracking when it starts (default 140)")
    parser.add_argument("--controller", default="mpc",
                        choices=("mpc", "reactive"),
                        help="model predictive (anticipates the bend) or "
                             "reactive line-of-sight (corrects after)")
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
    # Simulate a run-up before the stretch being shown.  Dropping the boat
    # onto the line at rest, pointed along a chord, leaves the controller a
    # transient to settle -- which is real behaviour but not what anyone
    # wants to watch, and it is not how a crew arrives at any point of a
    # race.  Rowing in from behind means the boat is already tracking when
    # the picture starts.
    if args.full:
        # The whole race: station 0 is the start line off DeWolfe and the
        # course length is the finish.  A flag rather than remembering the
        # numbers -- and it was accepted and silently ignored for one
        # embarrassing pair of renders before this line existed.
        args.start, args.finish = 0.0, float(course.length)
        args.lead = 0.0                  # there is nothing before the start
    lead = max(float(args.lead), 0.0)
    inside = (station >= args.start - lead) & (station <= args.finish)
    path = full_path[inside]
    shown_from = float(np.searchsorted(
        station[inside], args.start, side="left")) if lead else 0.0
    if len(path) < 10:
        raise SystemExit("that stretch is too short to animate")

    # Long enough to cover the leg even if the boat is slower than planned.
    leg = float(np.hypot(*np.diff(path, axis=0).T).sum())
    duration = 1.15 * leg / 4.6

    boat = catalog.eight(rate=28.0)
    if args.controller == "mpc":
        # Fit the controller's internal model to the boat it is steering.
        steering = fit_reduced_model(catalog.eight(rate=28.0),
                                     reference_speed=4.7)
        follower = PathMPC(path, model=steering, horizon=6.0, steps=12,
                           interval=0.20)
    else:
        follower = PathFollower(path, boundary_layer=25.0, gain=2.5,
                                rate_gain=1.2)
    sim = RowingSimulator(boat, coxswain=Coxswain(rudder_override=follower))
    # The path tangent at the first point, not a chord across eight of
    # them: over a bend those differ by a degree or two, which is an
    # initial heading error the controller then has to work off.
    heading = float(np.arctan2(path[1, 1] - path[0, 1],
                               path[1, 0] - path[0, 0]))
    # ``initial_state`` sets the velocity in the **absolute** frame, so
    # ``surge_speed`` alone points the boat's motion due east regardless of
    # where it is heading.  Setting yaw without rotating the velocity to
    # match leaves the boat crabbing at the heading angle from the first
    # step -- 2.6 m of sideways displacement in the first second here,
    # which the controller then spends the whole run chasing.  It looked
    # like a controller oscillation and was an inconsistent initial state.
    state = sim.initial_state(surge_speed=4.7)
    state[0], state[1] = path[0]
    state[5] = heading
    state[6] = 4.7 * np.cos(heading)
    state[7] = 4.7 * np.sin(heading)

    print("simulating %.0f m of racing (%.0f s) under %s ..."
          % (leg, duration, args.controller))
    clock = time.time()
    result = sim.run(duration=duration, dt=args.dt, initial_state=state)
    print("   %.0f s of wall clock" % (time.time() - clock))
    if hasattr(follower, "solves"):
        print("   %d MPC solves, %d fell back"
              % (follower.solves, follower.failures))

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

    # Drop the run-up from the picture: it was simulated so the boat would
    # be settled, not so anyone would watch it.
    if lead > 0.0:
        gate = path[int(shown_from)] if shown_from < len(path) else path[0]
        reached = np.nonzero(np.linalg.norm(positions - gate, axis=1) < 8.0)[0]
        if len(reached):
            begin = int(reached[0])
            print("   settled over the %.0f m run-up; showing from %.0f s"
                  % (lead, times[begin]))
            positions = positions[begin:]
            headings = headings[begin:]
            times = times[begin:] - times[begin]

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
    target = os.path.join(args.out, "race_%04d_%04d_%s_%s"
                          % (args.start, args.finish, args.view,
                             args.controller))
    print("writing %d frames ..." % frames)
    written = animate.write_animation(figure, update, frames, target,
                                      fps=args.fps)
    print("wrote", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
