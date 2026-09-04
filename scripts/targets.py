r"""What time wins, medals, and requalifies the boat -- with error bars.

    python scripts/targets.py

Three targets, because they are three different races:

* **Win** the 60-69 category.
* **Medal** -- top three.
* **Requalify** -- the Head of the Charles guarantees next year's entry to
  a sweep crew finishing "within the top half (based on entries) of the
  event-division in which they competed the preceding year". For a
  lottery entry that is the one that compounds: miss it and you are back
  in the draw.

Each is reported on the Charles *and* as the Lake Sammamish split that
produces it, because a training target on deep still water is not the
number on the river (SOURCES sec. 87).

Where the numbers come from
---------------------------
Two sources, doing different jobs.

**Full fields, 2022-2025** (``data/hocr_wvet4_results.csv``, 85 crews)
give the *shape* of the field: how far back third place and the top-half
cut sit relative to the winner. Those gaps are expressed as **ratios** to
the winning time, which is what makes them poolable across years whose
conditions differed.

**Winning times, 2014-2025** (``data/hocr_champions.csv``) give the
*level* and its year-to-year variability, over three times the span.

Combining them beats using either alone: four years is too few to
estimate the spread of winning times, and twelve years of winners says
nothing about where third place was.

The uncertainty is real and it is mostly weather
------------------------------------------------
SOURCES sec. 77 measured the year factor -- how fast the whole regatta ran
against its own norm -- and its spread over sixty years is **24.5%**, sd
2.6%. That is the dominant term, and it is why these come with bounds
rather than as single numbers. A target quoted to the tenth of a second
would be false precision about the wind.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river.charles import (HOCR_COURSE_LENGTH,  # noqa: E402
                                    charles_course)
from coxswain.crew.pacing import CoursePacing, CourseSegment  # noqa: E402

from course_pacing import (build_boat, build_segments,  # noqa: E402
                           build_wind, hull_drag)
from splits import deep_still_speed, power_for_time, stamp  # noqa: E402

CATEGORY = ("women", "Veteran Fours Age average of 60+ (no rower under 30)")


def load_fields(path):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["year"] = int(row["year"])
        row["place"] = int(row["place"])
        row["seconds"] = float(row["seconds"])
    by_year = defaultdict(list)
    for row in rows:
        by_year[row["year"]].append(row)
    for year in by_year:
        by_year[year].sort(key=lambda r: r["place"])
    return by_year


def load_winners(path, key):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if (r["gender"], r["event"]) == key]
    return sorted(((int(r["year"]), float(r["seconds"])) for r in rows))


def field_ratios(by_year):
    """Gap to 3rd and to the top-half cut, as ratios of the winning time.

    Ratios rather than seconds, because a year that was 3% slow was 3%
    slow for everyone and the *shape* of the field is what carries across.
    """
    medal, half, counts = [], [], []
    for year, rows in sorted(by_year.items()):
        winner = rows[0]["seconds"]
        n = len(rows)
        counts.append(n)
        if n >= 3:
            medal.append(rows[2]["seconds"] / winner)
        # "Top half based on entries": with n entries the cut is place
        # floor(n/2), and a crew must be at or inside it.
        cut = max(int(np.floor(n / 2)), 1)
        half.append(rows[cut - 1]["seconds"] / winner)
    return np.array(medal), np.array(half), np.array(counts)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fields", default="data/hocr_wvet4_results.csv")
    parser.add_argument("--winners", default="data/hocr_champions.csv")
    parser.add_argument("--boat", default="4+")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--segments", type=int, default=12)
    parser.add_argument("--wind", type=float, default=0.0)
    parser.add_argument("--wind-from", type=float, default=250.0)
    args = parser.parse_args(argv)

    by_year = load_fields(args.fields)
    winners = load_winners(args.winners, CATEGORY)
    medal, half, counts = field_ratios(by_year)

    print("WOMEN'S VETERAN FOURS 60+ -- WHAT IT TAKES")
    print("  full fields %d-%d (%d crews); winning times %d-%d (%d years)"
          % (min(by_year), max(by_year), sum(counts),
             winners[0][0], winners[-1][0], len(winners)))
    print()

    # -- the level, and how much it moves ------------------------------
    times = np.array([t for _y, t in winners])
    years = np.array([y for y, _t in winners])
    slope, intercept = np.polyfit(years, times, 1)
    trend = slope * 2026 + intercept
    residual = times - (slope * years + intercept)
    spread = float(residual.std(ddof=1))

    print("1. THE WINNING TIME")
    print("   mean %s, sd %s over %d years"
          % (stamp(times.mean()), stamp(times.std(ddof=1)), len(times)))
    print("   trend %+.1f s per year (%s), so 2026 centres on %s"
          % (slope, "getting faster" if slope < 0 else "getting slower",
             stamp(trend)))
    print("   residual sd about that trend: %.1f s" % spread)
    print()
    print("   The trend is weak against the scatter, and the scatter is")
    print("   mostly weather: SOURCES sec. 77 put the spread of year")
    print("   factors at 2.6% sd, which on a 20-minute race is about")
    print("   %.0f s -- the same size as the residual above.  Quoting a"
          % (0.026 * times.mean()))
    print("   target to the second would be false precision about wind.")
    print()

    # -- the three targets ----------------------------------------------
    print("2. THE THREE TARGETS, 2026")
    print("   field shape from %d-%d: 3rd is %.4f x winner (sd %.4f),"
          % (min(by_year), max(by_year), medal.mean(), medal.std(ddof=1)))
    print("   top-half cut is %.4f x winner (sd %.4f); fields of %d-%d"
          % (half.mean(), half.std(ddof=1), counts.min(), counts.max()))
    print()

    targets = [("WIN the category", 1.0, 0.0),
               ("MEDAL (top three)", float(medal.mean()),
                float(medal.std(ddof=1))),
               ("REQUALIFY (top half)", float(half.mean()),
                float(half.std(ddof=1)))]

    boat = build_boat(args.boat, args.rate)
    drag = hull_drag(boat)
    course = charles_course()
    wind = build_wind(args.wind, args.wind_from)
    segments = build_segments(course, args.segments, boat, wind)
    scale = HOCR_COURSE_LENGTH / sum(s.length for s in segments)
    raced = [CourseSegment(length=s.length * scale, current=s.current,
                           headwind=s.headwind, depth=s.depth,
                           drag_factor=s.drag_factor, label=s.label)
             for s in segments]
    model = CoursePacing(raced, drag, rowers=boat.n_seats,
                         shallow_model=boat.shallow)
    per = HOCR_COURSE_LENGTH / 500.0

    print("   %-22s %11s %13s %11s %13s"
          % ("target", "Charles", "80% range", "split/500", "Sammamish"))
    rows = []
    for label, ratio, ratio_sd in targets:
        centre = trend * ratio
        # Two independent sources of uncertainty: how fast the winning
        # time is that year, and where the cut sits relative to it.
        sigma = np.hypot(spread * ratio, trend * ratio_sd)
        low, high = centre - 1.2816 * sigma, centre + 1.2816 * sigma
        power = power_for_time(model, centre)
        lake = 500.0 / deep_still_speed(drag, boat.n_seats, power)
        rows.append((label, centre, low, high, power, lake))
        print("   %-22s %11s %6s-%-6s %11s %13s"
              % (label, stamp(centre), stamp(low), stamp(high),
                 stamp(centre / per), stamp(lake)))
    print()
    print("   'Charles' is the central estimate for 2026; the range is an")
    print("   80% interval, so a crew on the fast edge of it beats the")
    print("   target four years in five.")
    print()

    print("3. WHAT TO HOLD ON SAMMAMISH")
    for label, centre, _lo, _hi, power, lake in rows:
        print("   %-22s %.0f W per rower  ->  %s per 500 m"
              % (label, power, stamp(lake)))
    print()
    print("   Deep still water, so these are faster than the river splits")
    print("   by about %.1f s -- train to the lake number, not the river"
          % ((rows[0][1] / per) - rows[0][5]))
    print("   number (SOURCES sec. 87).")
    print()

    # -- where the local crews sat ---------------------------------------
    print("4. THE CREWS YOU KNOW")
    local = ("Sammamish", "Lake Washington", "Green Lake", "Conibear",
             "Montlake", "Lake Union", "Pocock", "Chinook")
    print("   %-32s %6s %6s %11s %9s"
          % ("club", "year", "place", "time", "of field"))
    for year, rows_year in sorted(by_year.items()):
        for row in rows_year:
            if any(name.lower() in row["club"].lower() for name in local):
                print("   %-32s %6d %6d %11s %8.0f%%"
                      % (row["club"][:32], year, row["place"], row["time"],
                         100.0 * row["place"] / len(rows_year)))
    print()
    print("   'of field' is the percentile that matters: at or under 50%")
    print("   guarantees entry the following year.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
