"""How fast do this squad's rowers actually improve? Longitudinal 5k erg history, every season since 2013.

    .venv/Scripts/python.exe research/training/squad_progression.py [--xlsx data/local/erg/Erg_and_Workouts_<date>.xlsx]

Reads the team's erg sheet ("Erg Test POST HERE": one row per rower, one column block per testing season; the season's
label sits above the column holding its 5k time). Every 5k is converted to watts with Concept2's relation
P = 2.80 / (t500)^3 (t500 the split in seconds per 500 m / 500, i.e. pace in s/m), so seasons compare on power. Output,
with rowers as anonymous ids (no names leave data/local):

- data/local/training/squad_5k_series.csv: id, sex, birth year, weight, season date, 5k seconds, watts;
- the yearly change in power against training age (years since the rower's first test) and against starting level,
  pooled over every rower with at least two tests a year or more apart: the empirical yardstick for a target.

Training age is measured from the first recorded test, not from when a rower began; for adult-onset club rowers who
join and test in their first season the two are close.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "local", "training")

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
          "nov": 11, "dec": 12}
SEASON = {"spring": 4, "summer": 7, "fall": 10, "winter": 12, "february": 2, "january": 1, "july": 7, "november": 11,
          "december": 12, "august": 8}


def season_date(label: str):
    """A testing season's label -> an approximate date (mid-month). None when no year can be read."""
    s = label.lower().replace("\xa0", " ")
    m = re.search(r"by (\d{1,2})/(\d{1,2})/(\d{2})", s)
    if m:
        return dt.date(2000 + int(m.group(3)), int(m.group(1)) if 1 <= int(m.group(1)) <= 12 else 8, 15)
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2})", s)
    if m:
        return dt.date(2000 + int(m.group(3)), int(m.group(1)), 15)
    year = None
    m = re.search(r"(20\d\d)", s)
    if m:
        year = int(m.group(1))
    else:
        m = re.search(r"'(\d\d)", s)
        if m:
            year = 2000 + int(m.group(1))
    if year is None:
        return None
    for word, month in SEASON.items():
        if word in s:
            return dt.date(year, month, 15)
    for word, month in MONTHS.items():
        if re.search(r"\b" + word, s):
            return dt.date(year, month, 15)
    return dt.date(year, 7, 1)


def seconds(v):
    """A 5k time as the sheet holds it: 'mm:ss.s' text, a time-of-day object (mm:ss read as hh:mm), or a timedelta."""
    if v is None:
        return None
    if isinstance(v, dt.timedelta):
        s = v.total_seconds()
    elif isinstance(v, dt.time):
        # a split typed as 22:17 is stored as 22 h 17 min: read hours as minutes and minutes as seconds
        s = v.hour * 60 + v.minute if v.hour else v.minute * 60 + v.second
    elif isinstance(v, dt.datetime):
        return None
    else:
        m = re.match(r"^\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*$", str(v))
        if not m:
            return None
        s = int(m.group(1)) * 60 + float(m.group(2))
    return s if 900.0 <= s <= 2400.0 else None                       # a 5k between 15 and 40 minutes


def watts(t5k):
    pace = t5k / 5000.0                                               # s per metre
    return 2.80 / pace ** 3


def read(xlsx):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    rows = list(wb["Erg Test POST HERE"].iter_rows(values_only=True))
    labels = rows[1]
    blocks = [(c, season_date(str(v))) for c, v in enumerate(labels)
              if v and "5k" in str(v).lower() and season_date(str(v)) is not None]
    sex = None
    out = []
    for r in rows[2:]:
        if not r or r[0] is None:
            continue
        head = str(r[0]).strip()
        if head.upper() in ("WOMEN", "MEN"):
            sex = "F" if head.upper() == "WOMEN" else "M"
            continue
        if sex is None:
            continue
        born = r[1] if isinstance(r[1], (int, float)) else None
        weight = r[4] if isinstance(r[4], (int, float)) else None
        tests = []
        for c, d in blocks:
            if c < len(r):
                t = seconds(r[c])
                if t:
                    tests.append((d, t))
        if tests:
            out.append(dict(name=head, sex=sex, born=born, weight=weight, tests=sorted(set(tests))))
    return out, blocks


def yearly_changes(rowers):
    """Pairs of tests 0.75-1.5 years apart: change in watts per year, the rower's training age and level at the start."""
    rows = []
    for k, r in enumerate(rowers):
        first = r["tests"][0][0]
        for i, (d0, t0) in enumerate(r["tests"]):
            for d1, t1 in r["tests"][i + 1:]:
                gap = (d1 - d0).days / 365.25
                if 0.75 <= gap <= 1.5:
                    w0, w1 = watts(t0), watts(t1)
                    rows.append(dict(id=k, sex=r["sex"], age0=(d0.year - r["born"]) if r["born"] else None,
                                     train_age=(d0 - first).days / 365.25, w0=w0,
                                     rel_per_year=(w1 / w0 - 1.0) / gap, sec_per_year=(t1 - t0) / gap))
                    break
    return rows


def summarise(rows, sex="F"):
    sel = [r for r in rows if r["sex"] == sex]
    out = {}
    for lo, hi, lab in ((0, 1.5, "first year (training age 0-1.5)"), (1.5, 3.5, "years 2-3"), (3.5, 99, "year 4 on")):
        g = np.array([r["rel_per_year"] for r in sel if lo <= r["train_age"] < hi])
        if g.size:
            out[lab] = dict(n=int(g.size), median=float(np.median(g)), p75=float(np.percentile(g, 75)),
                            p90=float(np.percentile(g, 90)), best=float(g.max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=sorted(glob.glob(os.path.join(ROOT, "data", "local", "erg", "Erg_and_Workouts_*.xlsx")))[-1])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rowers, blocks = read(a.xlsx)
    with open(os.path.join(OUT, "squad_5k_series.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "sex", "born", "weight_lb", "date", "t5k_s", "watts"])
        for k, r in enumerate(rowers):
            for d, t in r["tests"]:
                w.writerow([k, r["sex"], r["born"], r["weight"], d.isoformat(), round(t, 1), round(watts(t), 1)])
    # the id map stays private, beside the data
    json.dump({r["name"]: k for k, r in enumerate(rowers)}, open(os.path.join(OUT, "squad_ids_private.json"), "w"),
              indent=1)
    rows = yearly_changes(rowers)
    print("%d rowers (%d women), %d testing seasons with a 5k, %d year-on-year pairs" % (
        len(rowers), sum(r["sex"] == "F" for r in rowers), len(blocks), len(rows)))
    for sex in ("F", "M"):
        print("\n%s: yearly change in 5k power, by training age" % ("Women" if sex == "F" else "Men"))
        for lab, s in summarise(rows, sex).items():
            print("  %-34s n %3d  median %+5.1f%%  75th %+5.1f%%  90th %+5.1f%%  best %+5.1f%%" % (
                lab, s["n"], 100 * s["median"], 100 * s["p75"], 100 * s["p90"], 100 * s["best"]))
    json.dump(dict(pairs=rows, women=summarise(rows, "F"), men=summarise(rows, "M")),
              open(os.path.join(OUT, "squad_progression.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
