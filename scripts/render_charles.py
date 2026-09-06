r"""Head of the Charles: the same four maps the Seattle races get.

    python scripts/render_charles.py --out out/charles

Three miles up the river from the start off DeWolfe Boathouse, under six
bridges, to the finish above Eliot.  The plan, the depth profile, the
oblique and the coxswain's view, drawn by
:mod:`coxswain.viz.race_render` -- the same renderer Tail of the Lake and
Head of the Lake use, so the three courses are drawn by one piece of code
and can be compared without an asterisk.

What this adds to the Charles model
-----------------------------------
**The docks.**  The corridor was bounded by the shoreline and the depth
contours alone.  The reach is lined with boathouse floats -- DeWolfe, BU,
Riverside, Weld, Newell, the Cambridge Boat Club -- and none of them were
in it.  They narrow 52 of 732 stations, worst 10.5 m, and they do it
mostly in the first 400 m, which is exactly where a crew is squeezed
between the BU boathouses and the bridge (SOURCES sec. 120).

**The boat's own beam.**  The corridor is the clearance from the
*centreline* to the nearest thing a boat cannot row through, so a shell
whose blades reach 3.5 m either side may legally have its centreline
0.5 m off a dock.  Lake Union and the ship canal had the same bug and it
put the optimised line 5 m off the wall of the Montlake Cut.  Subtracted
here too.

**The arches.**  The Charles already had what neither Seattle course
needed: bridge gates with piers and named arches, and the Head of the
Charles rule about which arch a crew must take.  They are drawn.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                          # noqa: E402
from coxswain.river import charles                          # noqa: E402
from coxswain.river.charts import CourseGeometry            # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,     # noqa: E402
                                  optimise_route)
from coxswain.river.trajectory import ReducedModel           # noqa: E402
from coxswain.viz.race_render import (RaceScene, TraceLine,  # noqa: E402
                                      render_all)

#: Half the width of a shell with its blades out, m.  See the module
#: docstring: the clearance is to the hull centreline.
BOAT_HALF_SPAN = 3.5
#: Cruising speed for a women's veteran coxed four, m/s.  The same figure
#: the Seattle courses use, so the three are comparable.
SPEED = 3.9
#: The official three miles.
PUBLISHED_LENGTH = 4828.0


def charles_race_course(month: int = 10, resolution: float = 6.0):
    """The raced course, with the docks and the boat's beam taken out.

    Returns ``(course, geometry, rowable)``.
    """
    plain = charles.charles_channel(resolution=resolution)
    rowable = charles.rowable_channel(plain)
    removed = 100.0 * (1.0 - rowable.navigable_area / plain.navigable_area)
    docks = len(charles.load_obstructions())
    print("  %d mapped docks and floats; they take %.1f%% of the navigable "
          "water" % (docks, removed))

    geometry = CourseGeometry(channel=rowable, month=month)
    line = geometry.line

    bare = rowable.half_width_along(line)
    half = np.maximum(bare - BOAT_HALF_SPAN, 0.5)
    shore_only = plain.half_width_along(line)
    pinched = int((shore_only - bare > 1.0).sum())
    print("  the docks narrow %d of %d stations, worst %.1f m"
          % (pinched, len(bare), float((shore_only - bare).max())))
    print("  corridor half-width: min %.1f, median %.1f m (was %.1f median "
          "on the shoreline alone)"
          % (half.min(), np.median(half), np.median(shore_only)))

    course = charles.charles_course(centreline=line, half_width=half,
                                    month=month)
    return course, geometry, rowable


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/charles")
    parser.add_argument("--month", type=int, default=10,
                        help="for the discharge; 10 is regatta month")
    parser.add_argument("--resolution", type=float, default=6.0)
    parser.add_argument("--speed", type=float, default=SPEED)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args(argv)

    print("Head of the Charles")
    course, geometry, rowable = charles_race_course(args.month,
                                                    args.resolution)
    length = course.length
    print("  %.0f m against a published %.0f (%+.1f%%)"
          % (length, PUBLISHED_LENGTH,
             100.0 * (length / PUBLISHED_LENGTH - 1.0)))

    gates = geometry.gates_on_course()
    print("  %d bridges on the course: %s"
          % (len(gates), ", ".join("%s at %.0f m" % (g.name, d)
                                   for g, d in gates)))

    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    evaluator = RouteEvaluator(course, boat=boat, flow=geometry.flow,
                               reference_speed=args.speed, upstream=True,
                               minimum_depth=1.2)
    # ``gates`` are ``(gate, metres)`` pairs and the evaluator unpacks
    # them that way; handing it bare gates raises deep inside the arch
    # check, where the message names a tuple and not the caller.
    evaluator.with_steering(ReducedModel(), raster=rowable, gates=gates)
    centre = evaluator.evaluate(Route.centreline(course))
    best = optimise_route(evaluator, n_control=13, iterations=args.iterations,
                          seed=0)
    print()
    print("  %-12s %9s %9s %9s %10s %8s" % ("", "score s", "time s",
                                            "length m", "peak yaw", "aground"))
    for label, result in (("as drawn", centre), ("optimised", best)):
        print("  %-12s %9.1f %9.1f %9.0f %10.2f %7.1f%%"
              % (label, result.elapsed, result.elapsed_clean,
                 result.path_length, result.peak_yaw_rate,
                 100.0 * result.fraction_aground))
    print("  SAVED %.1f s of time (%.1f s of score)"
          % (centre.elapsed_clean - best.elapsed_clean,
             centre.elapsed - best.elapsed))

    stations = np.linspace(0.0, length, 600)
    optimised = course.offset_position(stations, best.route.offset_at(stations))

    try:
        from coxswain.river.structures import charles_structures
        structures = charles_structures()
    except Exception as error:                        # pragma: no cover
        print("  (no structures: %s)" % str(error)[:60])
        structures = None

    marks = [(course.centreline[0, 0], course.centreline[0, 1], "START"),
             (course.centreline[-1, 0], course.centreline[-1, 1], "FINISH")]
    for gate, _distance in gates:
        middle = 0.5 * (np.asarray(gate.start) + np.asarray(gate.end))
        marks.append((float(middle[0]), float(middle[1]), gate.name))

    scene = RaceScene(
        name="Head of the Charles",
        east=rowable.east, north=rowable.north, water=rowable.water,
        lines=[TraceLine(course.centreline, "course as drawn",
                         colour="#7d8f9c", width=1.4, style="--"),
               TraceLine(optimised, "optimised line", colour="#ff9248",
                         width=2.2)],
        obstructions=[p for _k, p in charles.load_obstructions()
                      if len(p) > 1],
        structures=structures,
        depth_at=lambda x, y: float(course.depth_at(x, y)),
        speed=args.speed, boat_length=boat.length,
        marks=marks,
    )
    os.makedirs(args.out, exist_ok=True)
    written = render_all(scene, args.out)
    print()
    for path in written:
        print("  wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
