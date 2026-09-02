r"""Pull the Head of the Charles champions list, 1965 to present.

    python tools/scrape_hocr_champions.py --out data/hocr_champions.csv

The regatta publishes its champions as a single static page: one
accordion section per gender, one ``<h3>`` per event, and inside each a
paragraph of ``YEAR Winner TIME<br />`` lines with an asterisk marking a
course record. There is no API and no data file, so this parses the
markup.

Why the winners' list and not full results
------------------------------------------
Full placings live on RegattaCentral behind per-event job ids and are a
much larger scrape. The champions list is enough to answer the questions
that motivated it: how times in a category move over decades, whether a
year was fast or slow *across* categories, and which programmes recur.

That last point is the reason to keep every category rather than only the
one being raced. **A year's conditions are common to all events**, so the
spread of one crew's time against the field, or one category's winning
time against its own history, separates a fast crew from a fast day. A
single category cannot do that; twenty can.

What the times are, and are not
-------------------------------
They are elapsed times over a course whose length and layout have changed
more than once, rowed in whatever weather that October Saturday produced.
Comparing 1965 to 2025 as though they were the same test is meaningless.
Comparing categories *within* a year, or a category against its own
five-year rolling median, is not.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

URL = "https://hocr.org/the-regatta/champions/"

#: ``<h3> Championship Eights </h3>`` and friends.
HEADING = re.compile(r"<h3>\s*(.*?)\s*</h3>", re.S)
#: ``2025 Cambridge 13:41.692*`` -- winner may contain spaces, dots, commas,
#: ampersands and hyphens; the asterisk marks a course record.
ENTRY = re.compile(
    r"(?P<year>1[89]\d\d|20\d\d)\s+"
    r"(?P<winner>[^<\n]*?)\s+"
    r"(?P<time>\d{1,2}:\d{2}(?:\.\d+)?)(?P<record>\*?)"
)
#: Which accordion block we are inside.
GENDER = re.compile(r'id="accordion-item-\d+-(mens|womens|mixed)"')

TAGS = re.compile(r"<[^>]+>")
ENTITIES = (("&#8217;", "'"), ("&#8216;", "'"), ("&amp;", "&"),
            ("&#8211;", "-"), ("&#8212;", "-"), ("&nbsp;", " "),
            ("&quot;", '"'), ("&#039;", "'"))


def clean(text: str) -> str:
    for entity, plain in ENTITIES:
        text = text.replace(entity, plain)
    return re.sub(r"\s+", " ", TAGS.sub(" ", text)).strip()


def seconds(stamp: str) -> float:
    minutes, _, rest = stamp.partition(":")
    return float(minutes) * 60.0 + float(rest)


def parse(html: str):
    """Yield one record per champion."""
    # Split on headings, keeping track of which gender block we are in.
    marks = [(m.start(), "gender", m.group(1)) for m in GENDER.finditer(html)]
    marks += [(m.start(), "event", clean(m.group(1)))
              for m in HEADING.finditer(html)]
    marks.sort()

    for index, (start, kind, value) in enumerate(marks):
        if kind != "event":
            continue
        gender = "unknown"
        for position, other_kind, other in marks[:index][::-1]:
            if other_kind == "gender":
                gender = {"mens": "men", "womens": "women",
                          "mixed": "mixed"}[other]
                break
        end = marks[index + 1][0] if index + 1 < len(marks) else len(html)
        block = html[start:end]
        for match in ENTRY.finditer(block):
            winner = clean(match.group("winner"))
            if not winner:
                continue
            yield {
                "year": int(match.group("year")),
                "gender": gender,
                "event": value,
                "winner": winner,
                "time": match.group("time"),
                "seconds": round(seconds(match.group("time")), 3),
                "course_record": bool(match.group("record")),
            }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/hocr_champions.csv")
    parser.add_argument("--cache", default="data/raw/champions.html",
                        help="keep the fetched markup so a re-parse needs "
                             "no network")
    parser.add_argument("--offline", action="store_true",
                        help="parse the cached copy only")
    args = parser.parse_args(argv)

    html = None
    if args.offline or (os.path.exists(args.cache) and args.offline):
        with open(args.cache, encoding="utf-8") as handle:
            html = handle.read()
    if html is None:
        import requests
        print("fetching %s" % URL)
        response = requests.get(
            URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        html = response.text
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        with open(args.cache, "w", encoding="utf-8") as handle:
            handle.write(html)
        print("  cached %d bytes to %s" % (len(html), args.cache))

    rows = sorted(parse(html), key=lambda r: (r["gender"], r["event"],
                                              r["year"]))
    if not rows:
        print("parsed nothing -- the page markup has changed")
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    events = sorted({(r["gender"], r["event"]) for r in rows})
    years = [r["year"] for r in rows]
    print("wrote %s" % args.out)
    print("  %d rows, %d events, %d-%d"
          % (len(rows), len(events), min(years), max(years)))
    print("  %d course records marked"
          % sum(1 for r in rows if r["course_record"]))
    for gender in ("men", "women", "mixed"):
        count = sum(1 for r in rows if r["gender"] == gender)
        print("  %-6s %5d rows" % (gender, count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
