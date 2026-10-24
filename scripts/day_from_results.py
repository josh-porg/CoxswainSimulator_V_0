r"""Race morning: today's finished events -> the day factor input the race card reads (day_results.csv).

    python scripts/day_from_results.py due                 # which 2026 events to read now (URLs), as JSON
    python scripts/day_from_results.py add "<csv line>"     # store one event's summary (scripts/rc_event_summary.js)
    python scripts/day_from_results.py build               # write data/local/raceday/day_results.csv

The summaries live in data/local/raceday/results/hocr_event_summaries.csv, one line per year and event
(2024 to 2026, read from RegattaCentral in the browser). ``build`` pairs each official 2026 Saturday
event with the same event in 2025 (field medians; 2025's divided by BASE_2025, so the factor is against
the cards' still-air goals) and writes the rows ``scripts/race_card.py`` turns into the day factor (``coxswain.raceday.dayfactor``: the median of the per-event ratios). Backtested on
2024 -> 2025 and 2025 -> 2026 (results/backtest_dayfactor.py): with at least one finished event the
same day, the median ratio predicts an event's median to 1.2% on average, against 1.7% for last year's
time unchanged and 3.7% for the wind model driven by Logan's observed wind.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "local", "raceday", "results")
SUMMARIES = os.path.join(RES, "hocr_event_summaries.csv")
OUT = os.path.join(ROOT, "data", "local", "raceday", "day_results.csv")
EDT = timezone(timedelta(hours=-4))
#: 2025's Saturday ran this much slower than the race cards' still-air goals (field medians of events 17 and
#: 22 against the still-air requalify times, which sit at the field median: 1.021 and 1.018). The card applies
#: the day factor to still-air goals, so last year's times are divided by this before the ratio is taken.
BASE_2025 = 1.02
LAG_MIN = 25            # read an event this long after its start (the first finishers take ~17 min)
FIELDS = ["year", "event", "code", "status", "start_utc", "n", "winner", "q25", "median", "q75", "sp1", "sp2", "sp3"]


def _read():
    lines = open(SUMMARIES, encoding="utf-8").read().splitlines()
    head = [l for l in lines if l.startswith("#")]
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    return head, rows


def _write(head, rows):
    with open(SUMMARIES, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(head) + "\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def due(now=None):
    """2026 Saturday events started at least LAG_MIN ago and not yet stored as Official."""
    now = now or datetime.now(EDT)
    cfg = json.load(open(os.path.join(RES, "rc_events_2026.json")))
    _, rows = _read()
    done = {int(r["event"]) for r in rows if r["year"] == "2026" and r["status"] == "Official" and int(r["n"]) > 0}
    out = []
    for e in cfg["events"]:
        h, m = map(int, e["start_et"].split(":"))
        start = datetime(2026, 10, 17, h, m, tzinfo=EDT)
        if e["event"] not in done and start + timedelta(minutes=LAG_MIN) <= now:
            out.append(dict(event=e["event"], code=e["code"], url=cfg["url"] % e["event_id"]))
    return out


def add(line: str):
    vals = next(csv.reader([line.strip()]))
    if len(vals) != len(FIELDS):
        raise SystemExit("expected %d fields, got %d: %r" % (len(FIELDS), len(vals), line))
    new = dict(zip(FIELDS, vals))
    head, rows = _read()
    rows = [r for r in rows if not (r["year"] == new["year"] and r["event"] == new["event"])] + [new]
    rows.sort(key=lambda r: (int(r["year"]), int(r["event"])))
    _write(head, rows)
    print("stored %s event %s (%s, %s finishers)" % (new["year"], new["event"], new["status"], new["n"]))


def build():
    _, rows = _read()
    by = {(r["year"], r["event"]): r for r in rows}
    out = []
    for r in rows:
        if r["year"] != "2026" or int(r["event"]) < 9 or r["status"] != "Official" or int(r["n"] or 0) < 5:
            continue
        last = by.get(("2025", r["event"]))
        if last and last["code"] == r["code"]:
            out.append(("%s %s" % (r["event"], r["code"]), r["median"], "%.1f" % (float(last["median"]) / BASE_2025)))
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write("# Written by scripts/day_from_results.py at %s: 2026 field medians against 2025's / %.2f "
                 "(2025 against the still-air goals), official Saturday events only.\n" % (datetime.now(EDT).isoformat(timespec="minutes"), BASE_2025))
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["event", "today", "last_year"])
        w.writerows(out)
    for e, a, b in out:
        print("  %-14s %7s vs %7s  ratio %.4f" % (e, a, b, float(a) / float(b)))
    print("wrote %s: %d events" % (OUT, len(out)) if out else "wrote %s with no events yet (the wind model stands)" % OUT)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "due":
        print(json.dumps(due(), indent=1))
    elif cmd == "add":
        add(sys.argv[2])
    else:
        build()
