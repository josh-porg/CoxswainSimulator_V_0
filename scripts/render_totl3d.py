r"""Tail of the Lake in 3-D, from the coxswain's seat.

    python scripts/render_totl3d.py --stills
    python scripts/render_totl3d.py --view cox --frames 200

The Charles renderer, pointed at Lake Union.  Not a copy of it: the same
:class:`~coxswain.viz.river3d.RiverScene`, with Seattle's elevation model
and footprints passed in where the Charles' are the default.  Anything
that looks course-specific in the picture -- the skyline, the bridge, the
shape of the bank -- is data, not code.

What is in the frame and where it comes from
--------------------------------------------
**The water and the bank** are the OpenStreetMap shoreline and USGS 3DEP
bare-earth elevation, cross-checked against each other: the shoreline has
to sit on low ground, and it does to within 1% of the lake's cells.

**The skyline** is downtown Seattle, 3 to 5 km down the lake, drawn
because that is what a crew looking down the course actually sees.  Only
buildings that subtend a real angle are drawn, and everything is mixed
toward the horizon by distance -- without that the towers render at the
same contrast as the near bank and read as small rather than far.

**The bridge** is the Aurora Bridge, which the course does not go under.
It is in the picture because it is the landmark on this lake, and its
deck height is taken from the elevation model at its abutments rather
than from a tag OpenStreetMap does not carry.

**The docks** -- 499 piers, 141 houseboats, the marinas and the
breakwaters -- are drawn because they, and not the shoreline, are what a
crew steers off.  They already removed 40% of the lake from the
optimiser's corridor; this puts them in the picture too.

**The line** is the optimised one, from the same optimiser the Charles
uses, inside a corridor that already has the docks and the course buoys
taken out of it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                        # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,  # noqa: E402
                                  optimise_route)
from coxswain.river.seattle import (lake_union_channel,   # noqa: E402
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
from render_totl import totl_course                       # noqa: E402

#: Cruising speed for a women's veteran coxed four, m/s.
SPEED = 3.9


def optimised_line(course, boat, samples: int = 400):
    """The raced line and the centreline it improves on."""
    evaluator = RouteEvaluator(course, boat=boat,
                               reference_speed=SPEED).with_steering(
        ReducedModel())
    centre = evaluator.evaluate(Route.centreline(course))
    best = optimise_route(evaluator, n_control=9, iterations=40)
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
    # 1150-1850 m by default: the one stretch of the course that runs
    # almost due south, so the crew is looking straight down the lake at
    # downtown Seattle.  Any leg renders; this is the one that shows what
    # the skyline work was for.
    parser.add_argument("--from", dest="start", type=float, default=1150.0)
    parser.add_argument("--to", dest="finish", type=float, default=1850.0)
    parser.add_argument("--view", default="cox",
                        choices=("cox", "chase3d", "plan", "iso"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stills", action="store_true")
    parser.add_argument("--no-skyline", action="store_true",
                        help="draw only the near bank, for comparison")
    parser.add_argument("--no-imagery", action="store_true",
                        help="flat colour instead of the draped orthophoto")
    parser.add_argument("--out", default="out/totl3d")
    args = parser.parse_args(argv)

    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    print("building the course ...")
    course = totl_course()
    path, best, centre = optimised_line(course, boat)
    print("   %.0f m; optimised %.1f s against %.1f s down the middle"
          % (course.length, best.elapsed, centre.elapsed))

    result, leg = simulate(path, boat, args.start, args.finish)

    scene = RiverScene(
        # 4 m rather than the 10 m default: the waterline in the near
        # window comes straight off this mask, so 10 m cells were
        # quantising the shore into ten-metre steps regardless of how
        # good the elevation model under them was.
        boat, result=result, channel=lake_union_channel(resolution=4.0),
        path=path, window=320.0, follow=True,
        show_skyline=not args.no_skyline,
        structures=seattle_structures(), terrain=seattle_terrain(),
        imagery=None if args.no_imagery else seattle_imagery(),
        obstructions=load_obstructions(), trees=seattle_trees())

    os.makedirs(args.out, exist_ok=True)
    stem = "totl_%04d_%04d_%s" % (args.start, args.finish, args.view)
    if args.stills:
        for fraction in (0.02, 0.35, 0.7, 0.97):
            when = fraction * scene.duration
            target = os.path.join(args.out,
                                  "%s_t%03d.png" % (stem, int(when)))
            # No axis widget: this is a picture of a lake, and an
            # x/y/z triad in the corner of a coxswain's view is noise.
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
