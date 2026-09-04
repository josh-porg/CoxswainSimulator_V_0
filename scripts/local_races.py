r"""Does a local head race predict the Charles?

    python scripts/local_races.py

Lineups are set before the Charles and raced locally first, so a local
result is the earliest honest read on where a crew stands. Two things have
to be handled before it means anything.

**Local masters races rank on age-adjusted time.** Tail of the Lake applies
a handicap of up to 2:47 -- more than eight times the 20 s that separated
first from second in the 2024 Charles field. The Charles ranks on **raw**
time within an age band, so the comparison must use raw times and the
handicap must be stripped, not ignored.

**The courses are different lengths.** Rather than look up a distance and
hope it is the one raced, the conversion is measured from crews that
entered **both** races in the same year: their ratio absorbs the length,
the conditions and the day.

Matching crews honestly
-----------------------
Two traps, both of which produced wrong numbers before they were fixed.

*Club name alone is not a crew.* Sammamish entered five boats in one Tail
of the Lake event, ages 40 to 60; matching on club paired their age-40
crew with their 60+ Charles entry and produced a 1.40 ratio out of thin
air. Pairs are required to sit in the same **age band**.

*Normalisation order matters.* Stripping ``" association"`` before
``" rowing association"`` turns "Sammamish Rowing Association" into
"sammamish rowing", which then fails to match "Sammamish Association" --
silently dropping the single most relevant pair in the data.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splits import stamp                                        # noqa: E402

#: Tail of the Lake is a 4 km circuit of Lake Union.
#:
#: Worth noting how this was confirmed: the crew-matched ratio below
#: implies a length of ``4828 / 1.2175 = 3966 m`` from the race data
#: alone, and the published figure is **4000 m**.  Agreement to 0.9% is a
#: genuine independent check -- it says the ratio is capturing the course
#: length properly and that the conditions term riding along with it is
#: small.
TOTL_LENGTH = 4000.0
HOCR_LENGTH = 4828.0


def normalise(name: str) -> str:
    """Club name to a comparable key.

    Longest phrases first: the reverse order silently broke the Sammamish
    match, which was the pair the whole exercise existed for.
    """
    name = name.lower()
    # Compound suffixes first, then the bare word.  "Upper Valley Rowing"
    # and "Upper Valley" are one club, and not matching them cost the
    # only Textile-to-Charles pair in the 2024 data.
    name = re.sub(r"\b(rowing\s+)?(association|club|center|centre)\b",
                  " ", name)
    name = re.sub(r"\browing\b", " ", name)
    name = re.sub(r"\binc\b", " ", name)
    # Some sources repeat the club ("Westford Community,Westford Com...").
    name = re.sub(r"[^a-z ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"\s+b$", "", name)
    words = name.split()
    half = len(words) // 2
    if half and words[:half] == words[half:half * 2]:
        words = words[:half]
    return " ".join(words)


def load(path, **cast):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, fn in cast.items():
            if key in row:
                row[key] = fn(row[key])
    return rows


def matched_pairs(local, charles, year):
    """Crews in both races, same club and same age band."""
    field = [r for r in charles if r["year"] == year]
    entries = [r for r in local
               if "4+" in r["event"]
               and r["event"].lower().startswith("womens masters")]
    pairs = []
    for crew in field:
        band = 70 if "70" in crew["band"] else 60
        for other in entries:
            if normalise(other["club"]) != normalise(crew["club"]):
                continue
            age = int(other["age"])
            same = (age >= 68) if band == 70 else (58 <= age < 68)
            if not same:
                continue
            pairs.append((crew, other,
                          crew["seconds"] / other["raw_seconds"]))
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", default="data/totl_2024.csv")
    parser.add_argument("--charles", default="data/hocr_wvet4_results.csv")
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args(argv)

    local = load(args.local, raw_seconds=float, adjusted_seconds=float,
                 correction_seconds=float, place=int)
    charles = load(args.charles, year=int, place=int, seconds=float)

    print("TAIL OF THE LAKE %d -> HEAD OF THE CHARLES %d"
          % (args.year, args.year))
    print()
    handicap = np.array([r["correction_seconds"] for r in local
                         if "4+" in r["event"]])
    print("  the handicap is not a rounding error: local corrections in")
    print("  this file run from %.0f s to %.0f s (%s), and the Charles"
          % (handicap.min(), handicap.max(), stamp(handicap.max())))
    print("  applies none of it.  Raw times only, from here on.")
    print()

    pairs = matched_pairs(local, charles, args.year)
    if not pairs:
        print("  no age-matched crews in both races")
        return 1

    print("  %-24s %5s %11s %13s %9s"
          % ("club", "age", "local raw", "Charles raw", "ratio"))
    for crew, other, ratio in pairs:
        print("  %-24s %5s %11s %13s %9.4f"
              % (crew["club"][:24], other["age"], other["raw"],
                 crew["time"], ratio))
    ratios = np.array([r for _c, _o, r in pairs])
    print()
    print("  n = %d, mean %.4f, sd %.4f (%.1f%%)"
          % (len(ratios), ratios.mean(), ratios.std(ddof=1),
             100 * ratios.std(ddof=1) / ratios.mean()))
    print()
    print("  So: **Charles time is about %.3f x the Tail of the Lake raw"
          % ratios.mean())
    print("  time**, give or take %.1f%% -- roughly %.0f s on a 22-minute"
          % (100 * ratios.std(ddof=1) / ratios.mean(),
             ratios.std(ddof=1) / ratios.mean() * 22 * 60))
    print("  race.  That absorbs the course length, the conditions and")
    print("  the day, because it is measured from crews that rowed both.")
    print()

    print("LOCAL SPLITS -- Tail of the Lake, %.0f m" % TOTL_LENGTH)
    print("  the implied length from the ratio alone is %.0f m, against a"
          % (HOCR_LENGTH / ratios.mean()))
    print("  published %.0f m: agreement to %.1f%%, which is an independent"
          % (TOTL_LENGTH,
             100 * abs(HOCR_LENGTH / ratios.mean() - TOTL_LENGTH)
             / TOTL_LENGTH))
    print("  check that the conversion is doing what it claims.")
    print()
    per_local = TOTL_LENGTH / 500.0
    field = sorted([r for r in local
                    if r["event"].lower().startswith("womens masters")
                    and "4+" in r["event"]],
                   key=lambda r: r["place"])
    print("  %-5s %-26s %4s %10s %10s %11s"
          % ("place", "club", "age", "raw", "split/500", "adj split"))
    for row in field:
        print("  %-5d %-26s %4s %10s %10s %11s"
              % (row["place"], row["club"][:26], row["age"], row["raw"],
                 stamp(row["raw_seconds"] / per_local),
                 stamp(row["adjusted_seconds"] / per_local)))
    print()
    print("  'split/500' is the RAW split -- what the boat actually did.")
    print("  'adj split' is what the handicap turns it into for local")
    print("  scoring, and is the wrong number to take to the Charles.")
    print()

    print("USING IT: what a local time implies for the Charles")
    targets = [("win", 1195.9), ("medal", 1212.4), ("requalify", 1270.9)]
    print("  %-14s %11s %15s %14s"
          % ("target", "Charles", "local raw", "local split/500"))
    for label, seconds in targets:
        needed = seconds / ratios.mean()
        print("  %-14s %11s %15s %14s"
              % (label, stamp(seconds), stamp(needed),
                 stamp(needed / (TOTL_LENGTH / 500.0))))
    print()
    print("  These are the SOURCES sec. 89 targets divided by the ratio,")
    print("  so a crew reads its standing off a local race weeks before")
    print("  the entry list is even seeded.")
    print()
    print("  One caveat worth keeping: %d pairs is a small sample, and two"
          % len(ratios))
    print("  of them are the same club in adjacent age bands.  The 2.2%")
    print("  scatter is a lower bound on the real uncertainty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
