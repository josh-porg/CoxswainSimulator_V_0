"""Render the boat rowing the river in 3-D, from the coxswain's seat.

    python scripts/render3d.py --from 2100 --to 2800 --view cox
    python scripts/render3d.py --from 3600 --to 4200 --view chase3d --frames 200
    python scripts/render3d.py --controller reactive --stills

Simulates the full 6-DOF boat under model predictive control down an
optimised line, then draws it in the real river: surveyed bank, bridge
decks and piers, arches marked legal or penalised, and the planned line
on the water ahead.

``--view cox`` is the point of the exercise.  It puts the camera at the
coxswain's head, 0.7 m above the water in the stern, looking forward over
eight backs -- the only viewpoint from which the question "is this line
steerable?" means anything.  A plan view shows a coxswain geometry they
cannot see.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.river import charles                      # noqa: E402
from coxswain.river.charts import CourseGeometry        # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,  # noqa: E402
                                  optimise_route)
from coxswain.river.trajectory import (ReducedModel,  # noqa: E402
                                       fit_reduced_model)
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.guidance import PathFollower          # noqa: E402
from coxswain.sim.mpc import PathMPC                    # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402
from coxswain.viz.river3d import RiverScene             # noqa: E402


def simulate(start, finish, controller="mpc", dt=0.01):
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
    full = course.offset_position(station, route.offset_at(station))
    inside = (station >= start) & (station <= finish)
    path = full[inside]
    if len(path) < 10:
        raise SystemExit("that stretch is too short")
    leg = float(np.hypot(*np.diff(path, axis=0).T).sum())

    if controller == "mpc":
        steering = fit_reduced_model(catalog.eight(rate=28.0),
                                     reference_speed=4.7)
        driver = PathMPC(path, model=steering, horizon=6.0, steps=12,
                         interval=0.20)
    else:
        driver = PathFollower(path, boundary_layer=25.0)

    boat = catalog.eight(rate=28.0)
    sim = RowingSimulator(boat, coxswain=Coxswain(rudder_override=driver))
    heading = float(np.arctan2(path[8, 1] - path[0, 1],
                               path[8, 0] - path[0, 0]))
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

    print("simulating %.0f m under %s ..." % (leg, controller))
    clock = time.time()
    result = sim.run(duration=1.15 * leg / 4.6, dt=dt, initial_state=state)
    print("   %.0f s wall clock" % (time.time() - clock))
    if hasattr(driver, "solves"):
        print("   %d solves, %d fell back" % (driver.solves, driver.failures))

    positions = np.asarray(result.position)[:2].T
    finish_gap = np.linalg.norm(positions - path[-1], axis=1)
    arrived = np.nonzero(finish_gap < 12.0)[0]
    if len(arrived):
        result = result.truncate(float(result.time[int(arrived[0])])) \
            if hasattr(result, "truncate") else result
    return boat, result, raster, gates, path, driver


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", type=float, default=2100.0)
    parser.add_argument("--to", dest="finish", type=float, default=2800.0)
    parser.add_argument("--view", default="cox",
                        choices=("cox", "chase3d", "plan", "iso", "side",
                                 "stern", "top", "bow_quarter"))
    parser.add_argument("--controller", default="mpc",
                        choices=("mpc", "reactive"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stills", action="store_true",
                        help="write a few frames instead of a movie")
    parser.add_argument("--out", default="out/render3d")
    args = parser.parse_args(argv)

    boat, result, raster, gates, path, driver = simulate(
        args.start, args.finish, args.controller)

    scene = RiverScene(boat, result=result, channel=raster, gates=gates,
                       path=path, window=300.0, follow=True)
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    stem = "%s_%04d_%04d_%s" % (args.controller, args.start, args.finish,
                                args.view)
    if args.stills:
        for fraction in (0.0, 0.25, 0.5, 0.75, 0.98):
            when = fraction * scene.duration
            target = os.path.join(args.out, "%s_t%03d.png" % (stem, int(when)))
            scene.snapshot(t=when, path=target, view=args.view)
            print("wrote", target)
        return 0

    # mp4 needs imageio-ffmpeg; GIF needs nothing.  Try the better format
    # and fall back rather than failing after the frames are rendered.
    target = os.path.join(args.out, stem + ".mp4")
    print("rendering %d frames ..." % args.frames)
    # write_movie defaults to the last two stroke cycles, which is right
    # for checking technique and wrong for watching a leg of a race.
    try:
        scene.write_movie(target, n_frames=args.frames, view=args.view,
                          framerate=args.fps, t_start=0.0,
                          t_end=scene.duration,
                          window_size=(900, 600))
    except Exception as exc:
        print("   mp4 unavailable (%s); writing a GIF instead"
              % type(exc).__name__)
        target = os.path.join(args.out, stem + ".gif")
        scene.write_movie(target, n_frames=args.frames, view=args.view,
                          framerate=args.fps, t_start=0.0,
                          t_end=scene.duration,
                          window_size=(900, 600))
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
