r"""Import an event's start order from the published heat sheet (pasted text) into the race card.

    python scripts/heat_sheet.py --crew four  --start "10:32" --paste data/local/raceday/heat_four.txt
    python scripts/heat_sheet.py --crew eight --start "09:43" --paste data/local/raceday/heat_eight.txt

The heat sheet is on RegattaCentral, which refuses scripted access, so the rows come in as pasted
text: one crew per line, the bow number first and the crew after it, in whatever spacing the page
copies with ("21  Sammamish Rowing Association", "21 | Sammamish Rowing Association (Kelly, J.)",
"21. Sammamish ..."). Lines without a leading bow number are skipped. The crew text is matched to
the published entries (``data/local/crew/feeders/hocr2026_*_entries.csv``) by club name, so the
field simulation's crews keep their evidence; the entries file is rewritten with the heat sheet's
bows (the old one kept beside it), and the start time is written to ``data/local/raceday/
start_times.json``, which ``scripts/race_card.py`` reads in place of the assumed time.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

FEEDERS = os.path.join(ROOT, "data", "local", "crew", "feeders")
RACEDAY = os.path.join(ROOT, "data", "local", "raceday")
ROW = re.compile(r"^\s*(\d{1,3})\s*[.|:)\-\t ]+\s*(.+?)\s*$")


def parse(text: str):
    out = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            crew = re.sub(r"\s*\([^)]*\)\s*$", "", m.group(2)).strip(" |\t")
            out.append((int(m.group(1)), crew))
    return out


def match(crew: str, clubs):
    best = difflib.get_close_matches(crew, clubs, n=1, cutoff=0.6)
    if best:
        return best[0]
    low = crew.lower()
    hits = [c for c in clubs if c.lower() in low or low in c.lower()]
    return hits[0] if len(hits) == 1 else None


def main(argv=None):
    import race_card as RC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crew", required=True, choices=sorted(RC.CREWS))
    ap.add_argument("--paste", required=True, help="text file with the event's heat-sheet rows")
    ap.add_argument("--start", default=None, help="the event's start on the heat sheet, Eastern: 'HH:MM' (Saturday 17 Oct) or 'YYYY-MM-DD HH:MM'")
    a = ap.parse_args(argv)
    spec = RC.CREWS[a.crew]
    path = os.path.join(FEEDERS, spec["entries"])
    header = [l for l in open(path, encoding="utf-8") if l.startswith("#")]
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    clubs = [r["club"].strip() for r in rows]
    sheet = parse(open(a.paste, encoding="utf-8").read())
    if not sheet:
        raise SystemExit("no rows with a leading bow number found in %s" % a.paste)
    by_club = {}
    unmatched = []
    for bow, crew in sheet:
        club = match(crew, clubs)
        if club is None:
            unmatched.append((bow, crew))
        else:
            by_club.setdefault(club, []).append(bow)
    missing = [c for c in clubs if c not in by_club]
    print("%d rows; matched %d; unmatched %s; entries not on the sheet %s"
          % (len(sheet), len(sheet) - len(unmatched), unmatched or "none", missing or "none"))
    if unmatched or missing:
        print("not written: resolve the names above (edit the paste to the entry's club name) and rerun")
        return 1
    used = {c: 0 for c in by_club}
    for r in rows:
        c = r["club"].strip()
        r["bow"] = str(by_club[c][used[c]])
        used[c] += 1
    # the rows keep their order: the field simulation's columns follow it (traffic.summarise)
    shutil.copy(path, path + ".before_heat_sheet")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.writelines(header)
        f.write("# bows from the published heat sheet (scripts/heat_sheet.py)\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if a.start:
        times_path = os.path.join(RACEDAY, "start_times.json")
        times = json.load(open(times_path)) if os.path.exists(times_path) else {}
        times[a.crew] = a.start
        json.dump(times, open(times_path, "w"), indent=1)
    ours = next(r for r in rows if r["club"].strip() == RC.OURS)
    print("written %s; our bow %s%s" % (path, ours["bow"], "; start %s ET" % a.start if a.start else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
