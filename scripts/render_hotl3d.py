r"""Head of the Lake in 3-D, from the coxswain's seat.

    python scripts/render_hotl3d.py --stills
    python scripts/render_hotl3d.py --from 3600 --to 4600 --stills   # the Big Turn
    python scripts/render_hotl3d.py --view cox --frames 200

The same :class:`~coxswain.viz.river3d.RiverScene` that draws the Charles
and Tail of the Lake, pointed at the ship canal.  The default leg is the
Montlake Cut, 1700 to 2450 m: a 50 m slot between concrete walls with
the Montlake Bridge across the far end, the one stretch every crew on
this course remembers.

What is in the frame and where it comes from
--------------------------------------------
**The water** is the union of Lake Union, Portage Bay, the Montlake Cut
and Union Bay from OpenStreetMap, on USGS 3DEP bare-earth elevation.
Through the Cut the OSM water is 50-55 m wide between the walls, which
agrees with the Bridge Inventory's 45.7 m of navigation clearance at the
bridge to within the fenders.

**The bridges** are the three the course goes under -- I-5, the
University Bridge and the Montlake Bridge -- drawn from their
OpenStreetMap decks with heights from the Bridge Inventory, plus the
SR 520 bridge across Union Bay as a landmark.

**The line** is the optimised one, inside a corridor that has the docks,
the buoy lines and the bridge openings taken out of it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coxswain.boats import catalog                        # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,  # noqa: E402
                                  optimise_route)
from coxswain.river.seattle import (ship_canal_channel,   # noqa: E402
                                    load_obstructions)
from coxswain.river.structures import (seattle_structures,  # noqa: E402
                                       seattle_trees)
from coxswain.river.terrain import (seattle_imagery,       # noqa: E402
                                    seattle_terrain)
from coxswain.river.trajectory import ReducedModel        # noqa: E402
from coxswain.sim.control import Coxswain                 # noqa: E402
from coxswain.sim.guidance import PathFollower            # noqa: E402
from coxswain.sim.simulator import RowingSimulator        # noqa: E402
from coxswain.viz.river3d import RiverScene               # noqa: E402
from render_hotl import SPEED, hotl_course                # noqa: E402


def optimised_line(course, boat, samples: int = 500):
    """The raced line and the centreline it improves on."""
    evaluator = RouteEvaluator(course, boat=boat,
                               reference_speed=SPEED).with_steering(
        ReducedModel())
    centre = evaluator.evaluate(Route.centreline(course))
    best = optimise_route(evaluator, n_control=11, iterations=40)
    stations = np.linspace(0.0, course.length, samples)
    return course.offset_position(stations,
                                  best.route.offset_at(stations)), best, centre


def simulate(path, boat, start, finish, dt=0.02):
    """Row a leg of the line under the 6-DOF model."""
    station = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(path, axis=0).T))])
    inside = (station >= start) & (station <= finish)
    leg = path[inside]
    if len(leg) < 8:
        raise SystemExit("that stretch is too short")
    length = float(np.hypot(*np.diff(leg, axis=0).T).sum())

    # The follower gets the course from here on, not just this leg, and
    # the run stops when the leg does.
    #
    # It used to get the leg and run 1.1x as long as the leg takes.  For
    # the first 97% of that the boat held the line to a median 1.0 m --
    # and then it ran off the end of its own path, had nothing left to
    # follow, and carried on straight: 95 m off course by the last
    # frame.  On Lake Union that put the boat in open water and looked
    # like nothing; in the Montlake Cut it put it on the bank, and the
    # coxswain's view was a wall of hillside.  A rendered still is not
    # evidence of the scenery if the boat is not where the line is.
    ahead = path[station >= start]
    driver = PathFollower(ahead, boundary_layer=25.0)
    sim = RowingSimulator(boat, coxswain=Coxswain(rudder_override=driver))
    heading = float(np.arctan2(leg[6, 1] - leg[0, 1],
                               leg[6, 0] - leg[0, 0]))
    # ``initial_state`` sets velocity in the absolute frame, so the
    # heading has to be rotated into it or the boat crabs from step one.
    state = sim.initial_state(surge_speed=SPEED)
    state[0], state[1] = leg[0]
    state[5] = heading
    state[6] = SPEED * np.cos(heading)
    state[7] = SPEED * np.sin(heading)

    print("simulating %.0f m ..." % length)
    clock = time.time()
    result = sim.run(duration=OVERRUN * length / SPEED, dt=dt,
                     initial_state=state)
    print("   %.0f s wall clock" % (time.time() - clock))
    return trim_to(result, length), leg


#: How much longer than the nominal leg to integrate before trimming.
#: The 6-DOF hull settles near 4.7 m/s for this crew where the route
#: model assumes 3.9, so a run sized on ``SPEED`` alone finishes early.
OVERRUN = 1.30


def trim_to(result, length):
    """Cut a run at the moment the boat has covered ``length`` of track.

    The scene takes its duration from the result, so trimming here is
    what makes ``--from``/``--to`` mean what they say: every still and
    every movie frame then falls inside the leg that was asked for,
    however fast the hull turned out to be.
    """
    import dataclasses

    track = np.asarray(result.position)[:2].T
    covered = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(track, axis=0).T))])
    if covered[-1] < length:
        print("   the run covered %.0f m of the %.0f m leg"
              % (covered[-1], length))
        return result
    end = int(np.searchsorted(covered, length)) + 1
    print("   %.0f m in %.1f s (%.2f m/s); trimmed to the leg"
          % (covered[end - 1], result.time[end - 1],
             covered[end - 1] / max(result.time[end - 1], 1e-6)))
    return dataclasses.replace(result, time=result.time[:end],
                               states=result.states[:, :end])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", type=float, default=1700.0)
    parser.add_argument("--to", dest="finish", type=float, default=2450.0)
    parser.add_argument("--view", default="cox",
                        choices=("cox", "chase3d", "plan", "iso"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stills", action="store_true")
    parser.add_argument("--no-skyline", action="store_true",
                        help="draw only the near bank, for comparison")
    parser.add_argument("--no-imagery", action="store_true",
                        help="flat colour instead of the draped orthophoto")
    parser.add_argument("--out", default="out/hotl3d")
    args = parser.parse_args(argv)

    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    print("building the course ...")
    course = hotl_course()
    path, best, centre = optimised_line(course, boat)
    print("   %.0f m; optimised %.1f s against %.1f s as drawn"
          % (course.length, best.elapsed_clean, centre.elapsed_clean))

    result, leg = simulate(path, boat, args.start, args.finish)

    scene = RiverScene(
        # 4 m cells: the waterline in the near window comes straight off
        # this mask, and the Cut is only 50 m wide.  The window is 800 m
        # rather than Lake Union's 320 so the whole Cut is drawn at full
        # resolution from either end; beyond it the coarse mesh keeps a
        # slot open by taking block minima.
        boat, result=result, channel=ship_canal_channel(resolution=4.0),
        path=path, window=800.0, follow=True,
        show_skyline=not args.no_skyline,
        structures=seattle_structures(), terrain=seattle_terrain(),
        imagery=None if args.no_imagery else seattle_imagery(),
        obstructions=load_obstructions(), trees=seattle_trees())

    os.makedirs(args.out, exist_ok=True)
    stem = "hotl_%04d_%04d_%s" % (args.start, args.finish, args.view)
    if args.stills:
        for fraction in (0.02, 0.35, 0.7, 0.97):
            when = fraction * scene.duration
            target = os.path.join(args.out,
                                  "%s_t%03d.png" % (stem, int(when)))
            scene.snapshot(t=when, path=target, view=args.view,
                           window_size=(1100, 620), axes=False)
            print("wrote", target)
        return 0

    target = os.path.join(args.out, stem + ".mp4")
    print("rendering %d frames ..." % args.frames)
    try:
        scene.write_movie(target, n_frames=args.frames, view=args.view,
                          framerate=args.fps, t_start=0.0,
                          t_end=scene.duration, window_size=(900, 600))
    except Exception as error:
        print("   mp4 unavailable (%s); writing a GIF instead"
              % type(error).__name__)
        target = os.path.join(args.out, stem + ".gif")
        scene.write_movie(target, n_frames=args.frames, view=args.view,
                          framerate=args.fps, t_start=0.0,
                          t_end=scene.duration, window_size=(900, 600))
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
