r"""What split you need, and what that is on flat deep water.

    python scripts/splits.py --boat 4+

A crew trains on Lake Sammamish and races on the Charles, and the two are
not the same water. Sammamish is deep, still and (mostly) sheltered; the
Charles is shallow enough in places to put a hull near the critical depth
Froude number, runs against the crew, and is open to the wind across the
basin. **The same crew, rowing identically, produces a different split on
each.**

So "we need to go 2:08 at the Charles" is not a training target. This
converts it into one: find the power that produces the Charles time, then
ask what that power gives in flat deep water. That is the number to hold
on Sammamish.

Why it has to go through power
------------------------------
Split is not conserved across conditions and neither is speed; **power
is**, near enough, because it is what the crew produces. So the chain is

    Charles target time -> speed over ground -> speed through water
        -> resistance -> power -> deep-still speed -> Sammamish split

Every step but the last is already in this project and tested. Doing it
any other way -- scaling splits by a rule of thumb -- would be guessing at
exactly the nonlinearity (§66) that makes the Charles interesting.

The direction of the answer
---------------------------
The Charles is slower, so the equivalent Sammamish split is **faster**
than the Charles target. A crew that trains to the raw Charles number is
training too slow.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river.charles import (HOCR_COURSE_LENGTH,  # noqa: E402
                                    charles_course, hocr_course)

from course_pacing import (build_boat, build_segments,  # noqa: E402
                           build_wind, hull_drag)
from coxswain.crew.pacing import CoursePacing, CourseSegment  # noqa: E402

#: Categories worth quoting side by side.
CATEGORIES = {
    "4+": ("women", "Veteran Fours Age average of 60+ (no rower under 30)"),
    "8+": ("women", "Grand-Master Eights Age average of 50+ "
                    "(no rower under 30)"),
}


def stamp(seconds: float) -> str:
    return "%d:%04.1f" % (int(seconds // 60), seconds % 60)


def load_category(path, key):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if (r["gender"], r["event"]) == key]
    for row in rows:
        row["year"] = int(row["year"])
        row["seconds"] = float(row["seconds"])
    return sorted(rows, key=lambda r: r["year"])


def deep_still_speed(drag, rowers, power, efficiency=0.80):
    """Speed on flat, deep, still water at this per-rower power."""
    delivered = efficiency * power * rowers

    def excess(speed):
        air = 0.5 * 1.225 * 3.22 * speed * speed
        return (drag(speed) + air) * speed - delivered

    low, high = 0.5, 9.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if excess(mid) < 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def power_for_time(model, target, low=120.0, high=600.0, tol=0.05):
    """Per-rower power that produces ``target`` seconds on the course."""
    for _ in range(60):
        mid = 0.5 * (low + high)
        elapsed = model.evaluate(np.full(len(model.segments), mid)).total_time
        if abs(elapsed - target) < tol:
            return mid
        if elapsed > target:
            low = mid            # too slow, need more power
        else:
            high = mid
    return 0.5 * (low + high)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--boat", default="4+", choices=["8+", "4+"])
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--data", default="data/hocr_champions.csv")
    parser.add_argument("--segments", type=int, default=12)
    parser.add_argument("--wind", type=float, default=0.0)
    parser.add_argument("--wind-from", type=float, default=250.0)
    parser.add_argument("--target", type=float, default=None,
                        help="target Charles time in seconds; defaults to "
                             "the category's median winning time")
    args = parser.parse_args(argv)
    if args.rate is None:
        args.rate = 30.0 if args.boat == "4+" else 32.0

    key = CATEGORIES[args.boat]
    rows = load_category(args.data, key)
    times = np.array([r["seconds"] for r in rows])

    print("%s %s" % (key[0], key[1]))
    print("  %d winners, %d-%d" % (len(rows), rows[0]["year"],
                                   rows[-1]["year"]))
    print("  official course %.0f m, so a 500 m split is time / %.2f"
          % (HOCR_COURSE_LENGTH, HOCR_COURSE_LENGTH / 500.0))
    print()

    per = HOCR_COURSE_LENGTH / 500.0
    print("  %-8s %11s %11s %10s" % ("", "time", "split/500", "m/s"))
    for label, value in (("fastest", times.min()),
                         ("median", float(np.median(times))),
                         ("slowest", times.max())):
        print("  %-8s %11s %11s %10.3f"
              % (label, stamp(value), stamp(value / per),
                 HOCR_COURSE_LENGTH / value))
    print()
    print("  recent winners")
    print("  %-6s %11s %11s" % ("year", "time", "split/500"))
    for row in rows[-6:]:
        print("  %-6d %11s %11s"
              % (row["year"], stamp(row["seconds"]),
                 stamp(row["seconds"] / per)))
    print()

    target = args.target or float(np.median(times))
    boat = build_boat(args.boat, args.rate)
    drag = hull_drag(boat)
    course = charles_course()
    wind = build_wind(args.wind, args.wind_from)

    # Race-length slice of the reach, so the power is the power that wins
    # THIS race and not a 12 km paddle.
    _s, _f, _line, (start, finish) = hocr_course()
    segments = build_segments(course, args.segments, boat, wind)
    scale = HOCR_COURSE_LENGTH / sum(s.length for s in segments)
    raced = [CourseSegment(length=s.length * scale, current=s.current,
                           headwind=s.headwind, depth=s.depth,
                           drag_factor=s.drag_factor, label=s.label)
             for s in segments]
    model = CoursePacing(raced, drag, rowers=boat.n_seats,
                         shallow_model=boat.shallow)

    power = power_for_time(model, target)
    plan = model.evaluate(np.full(len(raced), power))
    flat = deep_still_speed(drag, boat.n_seats, power)

    print("WHAT THAT TAKES, AND WHAT IT IS ON FLAT WATER")
    print("  target on the Charles      %s   (%s per 500)"
          % (stamp(target), stamp(target / per)))
    print("  power it needs             %.1f W per rower" % power)
    print("  modelled Charles time      %s" % stamp(plan.total_time))
    print()
    print("  same power, deep still water")
    print("    speed                    %.3f m/s" % flat)
    print("    SPLIT ON SAMMAMISH       %s per 500 m"
          % stamp(500.0 / flat))
    print("    over %.0f m              %s"
          % (HOCR_COURSE_LENGTH, stamp(HOCR_COURSE_LENGTH / flat)))
    print()
    charles_split = target / per
    lake_split = 500.0 / flat
    print("  DIFFERENCE                 %.1f s per 500 m"
          % (charles_split - lake_split))
    print("  The lake split is %s than the river split, because the"
          % ("FASTER" if lake_split < charles_split else "SLOWER"))
    print("  Charles is shallow, adverse and open and Sammamish is not.")
    print("  A crew training to the raw Charles number trains too slow.")
    print()
    # A target is only a target if a crew can hold it.  Asking "what
    # power holds the still-air time in a headwind" returns a number
    # above critical power, and printing that as a lake split would be
    # actively misleading -- 1:44 is not a training target, it is proof
    # that the time is gone.
    from coxswain.crew.exertion import ROWER_CRITICAL_POWER
    duration = plan.total_time
    ceiling = ROWER_CRITICAL_POWER + model.capacity / max(duration, 1.0)
    print("  SUSTAINABILITY CHECK")
    print("    critical power %.1f W; over %s the ceiling is CP + W'/T = "
          "%.1f W" % (ROWER_CRITICAL_POWER, stamp(duration), ceiling))
    if power > ceiling:
        print("    ** %.1f W is ABOVE that ceiling. This target is not"
              % power)
        print("       reachable in these conditions -- the winning time")
        print("       itself will be slower, for everyone. The lake split")
        print("       above is arithmetic, not a plan. Re-run with a")
        print("       --target that a crew can actually hold.")
    else:
        print("    %.1f W is within it, so the target is holdable and the"
              % power)
        print("    lake split above is a real training number.")
    print()
    print("  Conditions assumed: %s"
          % ("still air" if args.wind <= 0 else
             "%.0f m/s from %.0f deg" % (args.wind, args.wind_from)))
    print("  Wind moves this a lot -- rerun with --wind to see how much.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
