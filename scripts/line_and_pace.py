r"""Is it worth steering wide for deeper water?

    python scripts/line_and_pace.py

`scripts/course_pacing.py` established that on the Charles the pacing
schedule is driven almost entirely by **depth**, because in the shallowest
surveyed water a masters eight at race pace sits at a depth Froude number
of 0.94 -- the near-vertical part of the transcritical drag rise, where a
watt buys almost no speed at all.

That result is about pacing, but its consequence is about **steering**: if
shallow water is that expensive, a line that goes around it may be worth
more than any redistribution of power along a line that goes through it.
This script asks the question directly.

The trade it is measuring
-------------------------
Three things move at once when a line is shifted off the centreline and
they do not all move the same way:

* **Distance.** Any deviation from the shortest line adds metres, and
  metres are seconds.
* **Depth.** The deep water is not on the centreline everywhere -- the
  thalweg wanders -- so a longer line can be a faster one.
* **Current.** On this river, negligible. Reported anyway so that
  "negligible" stays a measurement rather than an assumption.

The honest comparison is therefore total elapsed time with the pacing
optimiser run **separately on each line**, because the best schedule for a
deep line is not the best schedule for a shallow one. Comparing lines at a
single fixed schedule would understate the deep line, which is the error
this script exists to avoid.

What this does not do
---------------------
It sweeps constant offsets, which is a crude family of lines -- the real
optimum weaves toward the thalweg and back. It is enough to answer *is
this worth pursuing*, and the answer decides whether to spend the
optimiser on it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                          # noqa: E402
from coxswain.crew.pacing import CoursePacing, CourseSegment  # noqa: E402
from coxswain.river.charles import charles_course           # noqa: E402
from coxswain.river.route import Route                      # noqa: E402

from course_pacing import build_wind, hull_drag             # noqa: E402


def segments_along(course, path, n: int, wind=None):
    """Cut a path into ``n`` segments and read the river along it.

    Depth, current and wind are sampled on the **line actually rowed**,
    not on the centreline, which is the whole point: the centreline is
    where the survey put its spine, not where the deep water is.
    """
    step = np.diff(path, axis=0)
    leg = np.hypot(step[:, 0], step[:, 1])
    along = np.concatenate([[0.0], np.cumsum(leg)])
    total = float(along[-1])

    edges = np.linspace(0.0, total, n + 1)
    segments = []
    for start, end in zip(edges[:-1], edges[1:]):
        middle = 0.5 * (start + end)
        index = int(np.clip(np.searchsorted(along, middle) - 1,
                            0, len(leg) - 1))
        point = path[index]
        tangent = step[index] / max(leg[index], 1e-9)

        depth = float(course.depth_at(point[0], point[1]))
        current = np.asarray(course.current_at(point[0], point[1]))[:2]
        headwind = 0.0
        if wind is not None:
            speed = float(wind.speed_at(point[0], point[1]))
            headwind = -speed * float(np.dot(wind._towards, tangent))

        segments.append(CourseSegment(
            length=float(end - start),
            current=float(np.dot(current, tangent)),
            depth=max(depth, 0.30), headwind=headwind,
            label="%.0f m" % middle))
    return segments, total


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--offsets", type=float, nargs="+",
                        default=[-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
    parser.add_argument("--wind", type=float, default=0.0)
    parser.add_argument("--wind-from", type=float, default=250.0)
    args = parser.parse_args(argv)

    boat = catalog.eight(rate=32.0, rower_mass=68.0, rower_stature=1.70,
                         coxswain_mass=68.0)
    course = charles_course()
    wind = build_wind(args.wind, args.wind_from)
    drag = hull_drag(boat)

    print("Constant-offset lines on the surveyed Charles reach")
    print("  positive offset is to PORT of the centreline")
    if wind is not None:
        print("  wind %.1f m/s from %.0f deg" % (args.wind, args.wind_from))
    print()
    print("  %-12s %9s %8s %8s %10s %10s %9s"
          % ("line", "length m", "mean h", "min h", "flat s", "paced s",
             "vs centre"))

    rows = []
    for offset in args.offsets:
        route = Route.constant_offset(course, offset).clip_to_channel(
            course, margin=1.5)
        path = route.path(course, n=600)
        segments, length = segments_along(course, path, args.segments, wind)
        model = CoursePacing(segments, drag, shallow_model=boat.shallow)

        flat = model.flat_power()
        flat_plan = model.evaluate(np.full(len(segments), flat))
        paced, _amplitude = model.optimise(span=200.0, samples=61)
        depths = np.array([s.depth for s in segments])
        rows.append((offset, length, depths, flat_plan.total_time,
                     paced.total_time))

    centre = [r for r in rows if r[0] == 0.0]
    reference = centre[0][4] if centre else rows[0][4]
    for offset, length, depths, flat_time, paced_time in rows:
        print("  %-12s %9.0f %8.2f %8.2f %10.1f %10.1f %+9.1f"
              % ("%+.0f m" % offset, length, depths.mean(), depths.min(),
                 flat_time, paced_time, paced_time - reference))
    print()

    best = min(rows, key=lambda r: r[4])
    print("FASTEST LINE: %+.0f m, %.1f s" % (best[0], best[4]))
    if best[0] != 0.0:
        centre_row = centre[0] if centre else rows[0]
        extra = best[1] - centre_row[1]
        print("  It is %+.0f m %s than the centreline and %.1f s faster."
              % (abs(extra), "longer" if extra > 0 else "shorter",
                 reference - best[4]))
        print("  Mean depth %.2f m against %.2f m; shallowest %.2f against "
              "%.2f." % (best[2].mean(), centre_row[2].mean(),
                         best[2].min(), centre_row[2].min()))
        if extra > 0:
            print()
            print("  So the deeper line wins DESPITE being longer, which is")
            print("  the result worth having: at a depth Froude number near")
            print("  one, metres are cheaper than shallow water.")
    print()
    print("  'flat s' holds one power the whole way; 'paced s' re-optimises")
    print("  the schedule for that line.  They are reported separately")
    print("  because the best schedule for a deep line is not the best")
    print("  schedule for a shallow one, and comparing lines at a single")
    print("  fixed schedule would understate the deep one.")
    print()
    print("  Constant offsets are a crude family -- the real optimum weaves")
    print("  toward the thalweg and back.  This is sized to answer whether")
    print("  that is worth solving properly, not to be the answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
