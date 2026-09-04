r"""What time wins, what medals, and what gets the boat back next year.

    python scripts/target_times.py

Three thresholds matter to a crew entering the Head of the Charles, and
they are not the same question:

**Winning** is one crew's time, so it is the noisiest of the three -- it
depends entirely on who showed up.

**Medalling** is third place. Slightly less noisy, same dependence.

**Requalifying** is the top half of the field, and it is the one with real
consequences: Rule 3.2 grants guaranteed entry to institutions "finishing
within the top half (based on entries) of the sweep-oared event-division
in which they competed the preceding year." Miss it and the boat is back
in the lottery next year. As a lottery entry this season, that is the
threshold that decides whether there is a next season without another
draw.

Separating the crew from the day
--------------------------------
A raw threshold time confuses two things: how fast the field was, and what
the river did. §77's **year factor** -- the median log-residual across
~80 event-divisions that all raced the same water on the same day --
separates them. Each threshold is therefore reported twice: as it was
timed, and adjusted to neutral conditions.

The prediction for next year has to put the conditions variance back,
because next October's weather is not known. So the interval quoted is
wider than the spread of adjusted times, and deliberately so: it is
``sqrt(field variance + conditions variance)``.

Why a trend is not fitted
-------------------------
Six editions is not enough to fit a slope and mean it. The script tests
for one and reports the result; unless it is clearly significant, the
prediction is the adjusted mean with an interval, which is the honest
answer from six points.

A wrinkle in Rule 3.2 worth knowing
-----------------------------------
The rule adds: "Guaranteed entry acceptance to the **Veteran** Fours and
Eights will be determined on the basis of finishing within the respective
top half of all Veteran crews registered within the **Grand-Master** Fours
and Eights."

That reads as though Veteran crews are ranked among Grand-Master
entries rather than within their own event. The results published since
2019 show Women's Veteran Fours as a standalone event-division, so this
script uses the top half of the Veteran field and says so. If the regatta
scores it the other way the threshold moves, and it is worth asking
before relying on the number.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Head of the Charles to Tail of the Lake, from crews that raced both
#: (SOURCES sec. 92-93).  Four matched pairs, spread 2.2%.
TOTL_RATIO = 1.2175
TOTL_PAIRS = 4


def stamp(seconds: float) -> str:
    return "%d:%05.2f" % (int(seconds // 60), seconds % 60)


def load(path):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["year"] = int(row["year"])
        row["place"] = int(row["place"])
        row["seconds"] = float(row["seconds"])
    return rows


def thresholds(rows):
    """Winning, medal and requalifying time for each year."""
    by_year = defaultdict(list)
    for row in rows:
        by_year[row["year"]].append(row)

    out = {}
    for year, group in sorted(by_year.items()):
        group.sort(key=lambda r: r["place"])
        entries = len(group)
        # "Top half based on entries": with an odd field the halfway
        # place rounds down, which is the reading that does not hand a
        # guarantee to a crew in the bottom half.
        half = entries // 2
        out[year] = {
            "entries": entries,
            "win": group[0]["seconds"],
            "medal": group[min(2, entries - 1)]["seconds"],
            "requalify": group[half - 1]["seconds"] if half else np.nan,
            "half_place": half,
            "winner": group[0]["club"],
        }
    return out


def year_factor_map():
    """Conditions proxy per year, from the champions data (sec. 77)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from hocr_history import load as load_champions
        from hocr_history import year_factors
    except ImportError:
        return {}
    path = "data/hocr_champions.csv"
    if not os.path.exists(path):
        return {}
    return {year: value
            for year, (value, _n) in year_factors(load_champions(path)).items()}


def predict(adjusted, conditions_sd, level=0.80):
    """Mean, and a prediction interval that includes next year's weather.

    Two variances, not one.  The spread of adjusted thresholds is how much
    the *field* varies between years.  The conditions spread is how much
    the *river* varies.  Next October has both, so the interval is their
    sum in quadrature.
    """
    values = np.log(np.asarray(adjusted, dtype=float))
    mean = float(values.mean())
    field_sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    total = float(np.sqrt(field_sd ** 2 + conditions_sd ** 2))
    from scipy.stats import t as student
    dof = max(len(values) - 1, 1)
    # Prediction, not confidence: the extra 1/n is the uncertainty in the
    # mean itself, which matters with six points.
    width = student.ppf(0.5 + level / 2.0, dof) * total * np.sqrt(
        1.0 + 1.0 / len(values))
    return (float(np.exp(mean)), float(np.exp(mean - width)),
            float(np.exp(mean + width)), field_sd, total)


def trend(years, adjusted):
    """Is there a slope worth believing in six points?"""
    from scipy.stats import linregress
    result = linregress(np.asarray(years, float),
                        np.log(np.asarray(adjusted, float)))
    return result.slope, result.pvalue


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/hocr_wvet4_results.csv")
    parser.add_argument("--level", type=float, default=0.80)
    parser.add_argument("--target-year", type=int, default=2026)
    args = parser.parse_args(argv)

    if not os.path.exists(args.data):
        print("no %s" % args.data)
        return 1

    rows = load(args.data)
    table = thresholds(rows)
    factors = year_factor_map()
    known = [f for y, f in factors.items() if y >= 2015]
    conditions_sd = float(np.std(known, ddof=1)) if len(known) > 1 else 0.02

    print("Women's Veteran Fours (60+), Head of the Charles")
    print("  %d editions, %d results" % (len(table), len(rows)))
    print("  conditions spread (sec. 77 year factor, 2015-): sd %.2f%%"
          % (100 * conditions_sd))
    print()
    print("  %-6s %7s %10s %10s %12s   %s"
          % ("year", "entries", "win", "medal", "requalify", "winner"))
    for year, item in sorted(table.items()):
        print("  %-6d %7d %10s %10s %12s   %s"
              % (year, item["entries"], stamp(item["win"]),
                 stamp(item["medal"]), stamp(item["requalify"]),
                 item["winner"][:22]))
    print()
    print("  'requalify' is the slowest time still inside the top half,")
    print("  i.e. place %s in those fields."
          % ", ".join("%d" % t["half_place"] for _y, t in sorted(table.items())))
    print()

    years = sorted(table)
    print("ADJUSTED TO NEUTRAL CONDITIONS  (raw / year factor)")
    print("  %-6s %8s %11s %11s %13s"
          % ("year", "factor", "win", "medal", "requalify"))
    adjusted = defaultdict(list)
    for year in years:
        factor = factors.get(year, 0.0)
        row = []
        for key in ("win", "medal", "requalify"):
            value = float(np.exp(np.log(table[year][key]) - factor))
            adjusted[key].append(value)
            row.append(stamp(value))
        print("  %-6d %+7.2f%% %11s %11s %13s"
              % (year, 100 * (np.exp(factor) - 1.0), *row))
    print()

    print("PREDICTION FOR %d, %.0f%% interval" % (args.target_year,
                                                  100 * args.level))
    print("  %-12s %11s %24s %10s %10s"
          % ("threshold", "expected", "interval", "field sd", "total sd"))
    predictions = {}
    for key, label in (("win", "win"), ("medal", "medal (3rd)"),
                       ("requalify", "requalify (top half)")):
        mean, low, high, field_sd, total = predict(
            adjusted[key], conditions_sd, args.level)
        predictions[key] = (mean, low, high)
        print("  %-20s %11s   %10s to %-10s %6.2f%% %8.2f%%"
              % (label, stamp(mean), stamp(low), stamp(high),
                 100 * field_sd, 100 * total))
    print()
    for key, label in (("win", "win"), ("medal", "medal"),
                       ("requalify", "requalify")):
        slope, pvalue = trend(years, adjusted[key])
        verdict = ("significant" if pvalue < 0.05
                   else "not significant -- treat as flat")
        print("  trend in %-10s %+6.2f%% per year, p = %.2f  (%s)"
              % (label, 100 * slope, pvalue, verdict))
    print()

    print("THE SAME TARGETS AT TAIL OF THE LAKE")
    print("  Charles time / %.4f, from %d crews that raced both "
          "(sec. 92-93)." % (TOTL_RATIO, TOTL_PAIRS))
    print("  %-22s %11s %24s" % ("threshold", "TotL time", "interval"))
    for key, label in (("win", "win"), ("medal", "medal (3rd)"),
                       ("requalify", "requalify (top half)")):
        mean, low, high = predictions[key]
        print("  %-22s %11s   %10s to %-10s"
              % (label, stamp(mean / TOTL_RATIO), stamp(low / TOTL_RATIO),
                 stamp(high / TOTL_RATIO)))
    print()
    print("  Four matched pairs is a thin conversion and its own spread was")
    print("  2.2%, which is not carried into the intervals above.  Treat the")
    print("  Tail of the Lake column as indicative, not as a target to the")
    print("  second.")
    print()
    print("  And 500 m splits, which is what a crew actually races to:")
    print("  %-22s %11s %11s" % ("threshold", "Charles", "TotL"))
    for key, label in (("win", "win"), ("medal", "medal"),
                       ("requalify", "requalify")):
        mean = predictions[key][0]
        charles = mean / (4828.0 / 500.0)
        totl = (mean / TOTL_RATIO) / (4000.0 / 500.0)
        print("  %-22s %11s %11s" % (label, stamp(charles), stamp(totl)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
