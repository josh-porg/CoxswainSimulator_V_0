r"""Parse a row2k results PDF into rows, with the handicap kept separate.

    python tools/parse_row2k_pdf.py data/raw/totl_2024.pdf \
        --out data/totl_2024.csv

Local masters races rank on **age-adjusted** time; the Head of the Charles
ranks on **raw** time within an age band. Comparing the two without
separating the handicap is meaningless, and the handicap is not a small
term -- up to 2:47 in one Tail of the Lake field, more than eight times
the 20 s between first and second at the Charles.

So this keeps `raw`, `correction` and `adjusted` as three columns and
never conflates them.

Why coordinates rather than text
--------------------------------
``extract_text`` on this PDF emits the crew names and the numeric columns
as separate runs and interleaves several events per page, so a naive parse
pairs the wrong time with the wrong crew. Reading the text **with its
positions** and rebuilding rows by ``y`` fixes that, because the layout is
what carries the association.

The check that makes it trustworthy
-----------------------------------
Every row satisfies ``raw - correction == adjusted`` by construction of
the regatta's own scoring, so the parser asserts it. A mis-associated row
fails that arithmetic almost every time, which turns a silent pairing bug
into a loud one.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

#: ``1 | 592 | 20:31.7 | 20:11.3`` -- place, bow, raw, adjusted.
RESULT = re.compile(r"^(\d+)\s*\|\s*(\d+)\s*\|\s*"
                    r"(\d{1,2}:\d{2}(?:\.\d+)?)\s*\|\s*"
                    r"(\d{1,2}:\d{2}(?:\.\d+)?)$")
#: ``Age: 43 -20.48`` or ``Age: 70 -2:39.75``
AGELINE = re.compile(r"Age:\s*(\d+)\s*(-?\d{0,2}:?\d{1,2}(?:\.\d+)?)")
#: ``67. Womens Masters Novice 4x``
HEADING = re.compile(r"^\d{1,3}\.\s+(.*\S)\s*$")


def seconds(stamp: str) -> float:
    stamp = stamp.strip().lstrip("-")
    if ":" in stamp:
        minutes, rest = stamp.split(":")
        return float(minutes) * 60.0 + float(rest)
    return float(stamp)


def page_rows(page):
    """``[(y, 'joined text')]`` for one page, top to bottom."""
    runs = []

    def visit(text, cm, tm, font, size):
        stripped = text.strip()
        if stripped:
            runs.append((round(tm[5], 1), round(tm[4], 1), stripped))

    page.extract_text(visitor_text=visit)
    grouped = defaultdict(list)
    for y, x, text in runs:
        grouped[y].append((x, text))
    return [(y, " | ".join(t for _x, t in sorted(grouped[y])))
            for y in sorted(grouped, reverse=True)]


def parse(path, tolerance=0.15):
    reader = __import__("pypdf").PdfReader(path)
    out, mismatched = [], 0
    for page in reader.pages:
        lines = page_rows(page)
        event = ""
        for index, (_y, text) in enumerate(lines):
            heading = HEADING.match(text)
            if heading:
                event = heading.group(1)
                continue
            match = RESULT.match(text)
            if not match:
                continue
            place, bow, raw, adjusted = match.groups()

            # The age line is the next run down; club and coxswain follow.
            age, correction = None, None
            club = ""
            for _y2, following in lines[index + 1:index + 6]:
                found = AGELINE.search(following)
                if found and age is None:
                    age, correction = int(found.group(1)), found.group(2)
                    continue
                if (age is not None and not club
                        and not following.startswith("(")
                        and not RESULT.match(following)
                        and not HEADING.match(following)):
                    club = following.split(" | ")[0].strip()
                    break
            if age is None:
                continue

            raw_s, adj_s, corr_s = (seconds(raw), seconds(adjusted),
                                    seconds(correction))
            # The regatta's own arithmetic. A row assembled from the wrong
            # pieces fails this, which is the point of checking it.
            if abs((raw_s - corr_s) - adj_s) > tolerance:
                mismatched += 1
                continue
            out.append({
                "event": event, "place": int(place), "bow": int(bow),
                "club": club, "age": age,
                "raw": raw, "raw_seconds": round(raw_s, 2),
                "correction_seconds": round(corr_s, 2),
                "adjusted": adjusted, "adjusted_seconds": round(adj_s, 2),
            })
    return out, mismatched


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf")
    parser.add_argument("--out", default=None)
    parser.add_argument("--event", default=None,
                        help="only rows whose event matches this substring")
    args = parser.parse_args(argv)

    rows, mismatched = parse(args.pdf)
    if args.event:
        rows = [r for r in rows
                if args.event.lower() in r["event"].lower()]

    print("parsed %s" % args.pdf)
    print("  %d rows kept, %d rejected by the raw-minus-handicap check"
          % (len(rows), mismatched))
    events = sorted({r["event"] for r in rows})
    print("  %d events" % len(events))

    if args.out and rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
