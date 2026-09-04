r"""Walk row2k's yearly results indexes and collect head-race PDFs.

    python tools/scrape_row2k_index.py --years 2014-2025 --download

row2k has published regatta results since the 1990s, one index page per
year, each linking to a results page that usually embeds a PDF. That is
the widest freely-available archive of American rowing results, and it is
the only realistic route to *many years, many venues*.

The single conversion this project needs -- local head race to Head of the
Charles -- is measured from crews that raced both (SOURCES sec. 92), and
its binding constraint is **how few crews do** (sec. 94). Four pairs from
one regatta is not a national model. The fix is not a cleverer method, it
is more regatta-years, and this fetches them.

What it does and does not do
----------------------------
It collects and downloads. It does **not** parse: the PDFs come from
several timing vendors with incompatible layouts, and
:mod:`tools.parse_row2k_pdf` handles one family of them. Which files parse
is reported so the gap is visible rather than assumed.

Politeness
----------
One request at a time with a pause between, and an index cached to disk so
re-runs cost nothing. row2k is a small operation that has hosted this
archive for thirty years; hammering it would be both rude and a good way
to be blocked.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time

import requests

INDEX = "https://www.row2k.com/results/index.cfm?year=%d"
HEADERS = {"User-Agent": "CoxswainSimulator/0.1 (research; contact via "
                         "github)"}
LINK = re.compile(r'href="(/results/resultspage\.cfm\?UID=[^"]+)"[^>]*>'
                  r'([^<]{3,80})')
PDF = re.compile(r'https://www\.row2k\.com/results/files/[^"\'\s>]+\.pdf')

#: Head races worth having: fall events with real masters fields, plus
#: anything whose name says "head of/on the".  Deliberately broad -- the
#: cost of an extra regatta is one request, and the cost of missing one is
#: a whole region.
WANTED = re.compile(
    r"head\s+(of|on)\s+the|tail\s+of|textile|hooch|schuylkill|housatonic|"
    r"occoquan|riverfront|christina|charles|chase|regatta\s+of|"
    r"fall\s+classic|frostbite|turkey|snowflake", re.I)


def year_index(year, cache_dir):
    """The raw HTML of one year's index, cached."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "row2k_%d.html" % year)
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    response = requests.get(INDEX % year, timeout=90, headers=HEADERS)
    response.raise_for_status()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(response.text)
    return response.text


def regattas(year, cache_dir):
    html = year_index(year, cache_dir)
    seen, out = set(), []
    for href, label in LINK.findall(html):
        name = re.sub(r"\s+", " ", label).strip()
        if not WANTED.search(name):
            continue
        uid = href.split("UID=")[1].split("&")[0]
        if uid in seen:
            continue
        seen.add(uid)
        out.append({"year": year, "name": name, "uid": uid})
    return out


def pdf_for(uid, pause):
    url = "https://www.row2k.com/results/resultspage.cfm?UID=%s&cat=6" % uid
    response = requests.get(url, timeout=90, headers=HEADERS)
    time.sleep(pause)
    if response.status_code != 200:
        return None
    found = PDF.findall(response.text)
    return found[0] if found else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", default="2014-2025")
    parser.add_argument("--out", default="data/row2k_index.csv")
    parser.add_argument("--pdf-dir", default="data/raw/row2k")
    parser.add_argument("--cache", default="data/raw/row2k_index")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many PDFs (0 = no limit)")
    args = parser.parse_args(argv)

    low, _, high = args.years.partition("-")
    years = range(int(low), int(high or low) + 1)

    rows = []
    for year in years:
        try:
            found = regattas(year, args.cache)
        except Exception as error:
            print("  %d: index failed (%s)" % (year, str(error)[:60]))
            continue
        print("%d: %d head races on the index" % (year, len(found)))
        rows.extend(found)

    if args.download:
        os.makedirs(args.pdf_dir, exist_ok=True)
        downloaded = 0
        for row in rows:
            if args.limit and downloaded >= args.limit:
                break
            slug = re.sub(r"[^a-z0-9]+", "_", row["name"].lower()).strip("_")
            target = os.path.join(args.pdf_dir,
                                  "%d_%s.pdf" % (row["year"], slug[:40]))
            if os.path.exists(target):
                row["pdf"] = target
                continue
            try:
                url = pdf_for(row["uid"], args.pause)
            except Exception:
                url = None
            if not url:
                row["pdf"] = ""
                continue
            try:
                blob = requests.get(url, timeout=120, headers=dict(
                    HEADERS, Referer="https://www.row2k.com/results/"))
                time.sleep(args.pause)
            except Exception:
                row["pdf"] = ""
                continue
            if blob.status_code == 200 and blob.content[:4] == b"%PDF":
                with open(target, "wb") as handle:
                    handle.write(blob.content)
                row["pdf"] = target
                downloaded += 1
                print("  saved %s (%.0f kB)" % (os.path.basename(target),
                                                len(blob.content) / 1000))
            else:
                row["pdf"] = ""

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["year", "name", "uid", "pdf"]
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row.setdefault("pdf", "")
            writer.writerow({k: row.get(k, "") for k in fields})
    print()
    print("wrote %s: %d regatta-years" % (args.out, len(rows)))
    have = sum(1 for r in rows if r.get("pdf"))
    if args.download:
        print("  %d with a PDF on disk" % have)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
