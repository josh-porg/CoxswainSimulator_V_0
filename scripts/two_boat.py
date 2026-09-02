r"""Two boats on the course: does their wake change your line?

    python scripts/two_boat.py --boat 4+ --interval 15

A boat starts ahead of you at the regatta's usual interval and rows the
same speed, so the gap never closes and you sit in their water for the
whole race. Two questions follow, and only the second is interesting:

1. How much does their wake cost? (A little, and it is nearly free to
   avoid.)
2. **Does avoiding it change which line you should row?** That is the
   real question, because the line is already spoken for -- SOURCES
   sec. 66-67 found the fastest line is the thalweg, and being off it is
   expensive. If the leader is *on* the thalweg, dodging their puddles
   means leaving the deep water.

The competing pulls
-------------------
**Depth pulls you onto the centreline.** It is the deep line by
construction, and twenty metres off it costs a four about 54 s.

**Puddles push you off it**, but only just: `PuddleWake.lateral_overlap`
says the tracks are narrow. Less than a metre of separation clears them
completely, and a metre off the thalweg costs almost nothing.

So the answer is knowable in advance and the model should confirm it: take
the smallest offset that clears their blades and stay in the deep water.
What this script is really for is checking that, and finding the cases
where it fails -- a leader who is themselves off the line, or a gap short
enough that the hull wake matters.

The second wake lane
--------------------
One thing that is not obvious and falls out of the geometry: at a
separation of about **4.6 m** -- twice the blade track -- your port blades
land in their starboard puddles and you are back in dirty water. Between
about 0.8 m and 3.6 m you are clear; past 5.6 m you are clear again. A
coxswain drifting out to "get clear" can row straight through a second
band of their water on the way.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.crew.pacing import CoursePacing                  # noqa: E402
from coxswain.hydro.wake import PuddleWake, blade_track        # noqa: E402
from coxswain.river.charles import charles_course              # noqa: E402
from coxswain.river.route import Route                         # noqa: E402

from course_pacing import build_boat, build_wind, hull_drag    # noqa: E402
from line_and_pace import segments_along                       # noqa: E402


def leader_wake(boat, speed, drag):
    """The wake of the boat ahead, described by what it is doing."""
    return PuddleWake(drag=drag, speed=speed,
                      period=boat.timing.period,
                      n_blades=boat.n_seats)


def wake_multiplier(wake, gap, separation, track, hull_overlap=0.25):
    """Factor on this crew's required power from sitting in their water.

    Above one means the wake costs. The puddle term is scaled by the
    lateral overlap, so it disappears once the blade tracks separate; the
    hull-wake benefit is centreline-only and is given a Gaussian of the
    hull's own beam, because a boat a metre to the side is not in it
    either.
    """
    overlap = wake.lateral_overlap(gap, separation, blade_track=track)
    puddle = wake.power_penalty(gap) * overlap
    # The helpful half: narrow, on the centreline, and it decays.
    beam = 0.57
    lateral = np.exp(-(np.asarray(separation, float) / (2.0 * beam)) ** 2)
    benefit = wake.hull_benefit(gap, overlap=hull_overlap) * lateral
    return 1.0 + puddle - benefit


def evaluate(course, boat, drag, offsets, leader_offset, interval,
             segments_n, wind, speed_guess):
    """Time for each candidate line of mine, given the leader's line."""
    track = blade_track(boat)
    leader_route = Route.constant_offset(course, leader_offset).clip_to_channel(
        course, margin=1.5)
    leader_path = leader_route.path(course, n=600)

    rows = []
    for offset in offsets:
        route = Route.constant_offset(course, offset).clip_to_channel(
            course, margin=1.5)
        path = route.path(course, n=600)
        segments, _length = segments_along(course, path, segments_n, wind)

        # Same speed, so the along-course gap is fixed by the interval.
        gap = interval * speed_guess
        # Lateral separation, sampled at the same stations.
        step = max(len(path) // segments_n, 1)
        mine = path[::step][:segments_n]
        theirs = leader_path[::step][:segments_n]
        separation = np.hypot(mine[:, 0] - theirs[:, 0],
                              mine[:, 1] - theirs[:, 1])
        # For constant offsets the separation is essentially |dy|.
        separation = np.full(len(segments), abs(offset - leader_offset))

        wake = leader_wake(boat, speed_guess, drag(speed_guess))
        factor = wake_multiplier(wake, gap, separation, track)

        dirty = [type(s)(length=s.length, current=s.current,
                         headwind=s.headwind, depth=s.depth,
                         drag_factor=s.drag_factor * float(f),
                         label=s.label)
                 for s, f in zip(segments, np.atleast_1d(factor))]

        clean_model = CoursePacing(segments, drag, rowers=boat.n_seats,
                                   shallow_model=boat.shallow)
        dirty_model = CoursePacing(dirty, drag, rowers=boat.n_seats,
                                   shallow_model=boat.shallow)
        clean, _a = clean_model.optimise(span=200.0, samples=41)
        fouled, _b = dirty_model.optimise(span=200.0, samples=41)
        rows.append({
            "offset": offset,
            "separation": float(np.mean(separation)),
            "overlap": float(np.mean(
                wake.lateral_overlap(gap, separation, blade_track=track))),
            "factor": float(np.mean(np.atleast_1d(factor))),
            "clean": clean.total_time,
            "dirty": fouled.total_time,
        })
    return rows, interval * speed_guess


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boat", default="4+", choices=["8+", "4+"])
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--interval", type=float, default=15.0,
                        help="seconds the boat ahead started before you")
    parser.add_argument("--leader-offset", type=float, default=0.0,
                        help="the line they are rowing, m to port")
    parser.add_argument("--segments", type=int, default=12)
    parser.add_argument("--offsets", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 2.0, 3.0, 4.6, 6.0, 10.0,
                                 20.0])
    parser.add_argument("--wind", type=float, default=0.0)
    parser.add_argument("--wind-from", type=float, default=250.0)
    args = parser.parse_args(argv)
    if args.rate is None:
        args.rate = 30.0 if args.boat == "4+" else 32.0

    boat = build_boat(args.boat, args.rate)
    course = charles_course()
    wind = build_wind(args.wind, args.wind_from)
    drag = hull_drag(boat)
    speed = 3.895 if args.boat == "4+" else 4.490

    rows, gap = evaluate(course, boat, drag, args.offsets,
                         args.leader_offset, args.interval, args.segments,
                         wind, speed)

    print("Two boats, %s, %.0f s apart at %.2f m/s" % (args.boat,
                                                       args.interval, speed))
    print("  they started ahead of you and hold the same speed, so the")
    print("  gap stays at %.0f m (%.1f boat lengths) all race"
          % (gap, gap / boat.length))
    print("  they are rowing the %+.0f m line; blade track %.2f m"
          % (args.leader_offset, blade_track(boat)))
    print()
    print("  %-9s %8s %9s %8s %11s %11s %9s"
          % ("your line", "sep m", "in their", "power", "clean water",
             "their water", "wake cost"))
    print("  %-9s %8s %9s %8s %11s %11s %9s"
          % ("", "", "puddles", "factor", "s", "s", "s"))
    for row in rows:
        print("  %-9s %8.1f %9.3f %8.4f %11.1f %11.1f %9.2f"
              % ("%+.1f m" % row["offset"], row["separation"],
                 row["overlap"], row["factor"], row["clean"], row["dirty"],
                 row["dirty"] - row["clean"]))
    print()

    best_clean = min(rows, key=lambda r: r["clean"])
    best_dirty = min(rows, key=lambda r: r["dirty"])
    print("  fastest line in clean water: %+.1f m" % best_clean["offset"])
    print("  fastest line behind them:    %+.1f m" % best_dirty["offset"])
    if best_dirty["offset"] == best_clean["offset"]:
        print("  -- their wake does NOT change your line.")
    else:
        print("  -- their wake moves your line by %+.1f m, worth %.2f s."
              % (best_dirty["offset"] - best_clean["offset"],
                 best_clean["dirty"] - best_dirty["dirty"]))
    print()
    on_line = [r for r in rows if abs(r["offset"]
                                      - args.leader_offset) < 1e-9]
    if on_line:
        print("  sitting exactly on their line costs %.2f s over the reach."
              % (on_line[0]["dirty"] - on_line[0]["clean"]))
    print("  The puddle tracks are narrow: clearing them needs well under")
    print("  a metre, and a metre off the thalweg costs almost nothing.")
    print("  Watch the band near %.1f m, where your port blades meet their"
          % (2.0 * blade_track(boat)))
    print("  starboard puddles and you are back in dirty water.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
