r"""Sixty years of Charles winners: which years were fast, and which crews.

    python scripts/hocr_history.py

A winning time is two things added together: how good the crew was, and
what the river did that day. Absolute times cannot separate them, which is
why comparing your 2025 result to your 2019 result tells you very little.

The separation is available for free, though, because **every category
races the same river on the same day**. If twenty events are all three
percent slow in one year, that is the weather, not twenty simultaneous bad
crews. So:

.. math::

    t_{ey} = \bar t_e \times f_y \times \varepsilon_{ey}

with ``t_ey`` the winning time in event ``e``, year ``y``; ``\bar t_e``
that event's own typical time; ``f_y`` a **year factor** common to all
events; and the residual the part that is actually about the crew.

Taking logs makes it a two-way additive fit, and the year factor is then
just the median residual across events -- robust, because a category with
one freak entry should not move the estimate of the day.

What the year factor is
-----------------------
It is a *conditions proxy*, not a measurement of wind or stream. It
absorbs anything common to the whole regatta: weather, water level, course
changes, and any year the timing or the route differed. That makes it
useful for exactly one thing -- putting a crew's time in context -- and
useless as physics. It is reported here, not fed into the simulator.

The caveat that matters
-----------------------
These are **winners only**. A category's winning time depends on who
entered, and a year when a national squad shows up is not a fast year, it
is a strong field. The year factor is a median across many categories
precisely to blunt that, but it cannot remove it. Full placings would fix
this and live behind per-event ids on RegattaCentral.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The event this project exists for.
TARGET = ("women", "Grand-Master Eights Age average of 50+ "
                   "(no rower under 30)")


def load(path):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["year"] = int(row["year"])
        row["seconds"] = float(row["seconds"])
        row["course_record"] = row["course_record"] == "True"
        row["key"] = (row["gender"], row["event"])
    return rows


def year_factors(rows, min_events=6):
    """Median log-residual per year, across every event that ran.

    Robust rather than least-squares: one category with an unusual field
    should not move the estimate of what the day was like.
    """
    by_event = defaultdict(list)
    for row in rows:
        by_event[row["key"]].append(row)

    typical = {}
    for key, group in by_event.items():
        if len(group) >= 5:
            typical[key] = float(np.median([np.log(r["seconds"])
                                            for r in group]))

    residuals = defaultdict(list)
    for row in rows:
        if row["key"] in typical:
            residuals[row["year"]].append(
                np.log(row["seconds"]) - typical[row["key"]])

    factors = {}
    for year, values in residuals.items():
        if len(values) >= min_events:
            factors[year] = (float(np.median(values)), len(values))
    return factors


def show_years(factors, recent):
    print("YEAR FACTOR -- how the whole regatta ran, against its own norm")
    print("  positive is a SLOW day; the units are percent of elapsed time")
    print()
    ordered = sorted(factors.items())
    tail = [(y, v) for y, v in ordered if y >= recent]
    print("  %-6s %9s %8s   %s" % ("year", "factor", "events", ""))
    for year, (value, count) in tail:
        percent = 100.0 * (np.exp(value) - 1.0)
        bar = "#" * int(min(abs(percent), 12) * 2)
        side = "slow" if percent > 0 else "fast"
        print("  %-6d %8.2f%% %8d   %-24s %s"
              % (year, percent, count, bar, side))
    print()

    values = np.array([100.0 * (np.exp(v) - 1.0)
                       for _y, (v, _n) in ordered])
    years = [y for y, _ in ordered]
    print("  over %d-%d the spread of year factors is %.1f%%, sd %.2f%%."
          % (years[0], years[-1], values.max() - values.min(), values.std()))
    print("  That is the size of the effect a crew's time carries before")
    print("  anyone has rowed differently.")
    fastest = years[int(np.argmin(values))]
    slowest = years[int(np.argmax(values))]
    print("  fastest day %d (%.2f%%), slowest %d (%+.2f%%)"
          % (fastest, values.min(), slowest, values.max()))
    print()


def show_category(rows, factors, key, recent):
    group = sorted([r for r in rows if r["key"] == key],
                   key=lambda r: r["year"])
    if not group:
        print("no rows for %s" % (key,))
        return
    print("CATEGORY -- %s %s" % (key[0], key[1]))
    print("  %-6s %10s %10s %-34s %s"
          % ("year", "time", "adjusted", "winner", ""))
    baseline = float(np.median([np.log(r["seconds"]) for r in group]))
    for row in group:
        if row["year"] < recent:
            continue
        factor = factors.get(row["year"], (0.0, 0))[0]
        adjusted = np.exp(np.log(row["seconds"]) - factor)
        mark = " CR" if row["course_record"] else ""
        print("  %-6d %10s %9.1fs %-34s%s"
              % (row["year"], row["time"], adjusted,
                 row["winner"][:34], mark))
    print()
    spread = np.exp(baseline)
    print("  median winning time %s" % stamp(spread))
    print()


def stamp(seconds):
    return "%d:%05.2f" % (int(seconds // 60), seconds % 60)


def show_programmes(rows, key, recent, top=8):
    """Who keeps winning this, and are they winning elsewhere too?"""
    group = [r for r in rows if r["key"] == key]
    counts = defaultdict(int)
    for row in group:
        counts[row["winner"]] += 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]

    print("PROGRAMMES -- repeat winners in this category")
    print("  %-34s %6s   %s" % ("crew", "wins", "also wins (other events)"))
    for name, wins in ranked:
        elsewhere = defaultdict(int)
        for row in rows:
            if row["key"] == key or row["winner"] != name:
                continue
            elsewhere[row["event"]] += 1
        others = sorted(elsewhere.items(), key=lambda kv: -kv[1])[:2]
        summary = ", ".join("%s x%d" % (e[:26], c) for e, c in others) or "-"
        print("  %-34s %6d   %s" % (name[:34], wins, summary))
    print()
    print("  A programme that wins across several categories in the same")
    print("  year is showing depth, and its result in one event carries")
    print("  information about the others.  That is the cross-category")
    print("  signal worth chasing once full placings are available.")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/hocr_champions.csv")
    parser.add_argument("--recent", type=int, default=2010)
    args = parser.parse_args(argv)

    if not os.path.exists(args.data):
        print("no %s -- run tools/scrape_hocr_champions.py first"
              % args.data)
        return 1

    rows = load(args.data)
    events = {r["key"] for r in rows}
    years = [r["year"] for r in rows]
    print("Head of the Charles champions, %d-%d"
          % (min(years), max(years)))
    print("  %d winners across %d events" % (len(rows), len(events)))
    print()

    factors = year_factors(rows)
    show_years(factors, args.recent)
    show_category(rows, factors, TARGET, args.recent)
    show_programmes(rows, TARGET, args.recent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
