r"""Today's river from earlier races: the day factor (R9 of docs/RACEDAY.md).

Every crew races the same water within the hour, so how fast the river is today shows in the events
that have already finished (SOURCES sec. 77). For each earlier event the ratio of today's time to
the same event's time last year (the median of each field, or the times of the same placings) is one
reading of the day; the day factor is their median, with a bootstrap interval. A field that changed a
lot from last year is a noisy reading, so the more events the better, and an event-level spread is
reported with the factor.

Input, a CSV (``#`` comment lines allowed), one row per event or per placing:
``event,today,last_year`` with times as ``m:ss.s`` or seconds. The coxswain or anyone watching the
results can fill it in on the morning; no results service is assumed.

The race card applies the factor to the targets' totals in place of the wind model's estimate (it
measures the actual river, wind and all), keeping the wind model's split of time between stretches.
"""
from __future__ import annotations

import csv
from typing import List, Optional

import numpy as np


def seconds(text: str) -> float:
    text = str(text).strip()
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        out = 0.0
        for p in parts:
            out = out * 60.0 + p
        return out
    return float(text)


def read(path: str) -> List[dict]:
    rows = csv.DictReader(line for line in open(path, encoding="utf-8") if line.strip() and not line.startswith("#"))
    out = []
    for r in rows:
        try:
            out.append(dict(event=r["event"].strip(), today=seconds(r["today"]), last_year=seconds(r["last_year"])))
        except (KeyError, ValueError):
            continue
    return out


def factor(rows: List[dict], n_boot: int = 4000, seed: int = 3) -> Optional[dict]:
    """The day factor (today / last year) from per-event median ratios, with an 80% bootstrap interval."""
    if not rows:
        return None
    events = sorted({r["event"] for r in rows})
    per = np.array([np.median([r["today"] / r["last_year"] for r in rows if r["event"] == e]) for e in events])
    rng = np.random.default_rng(seed)
    boot = np.median(per[rng.integers(0, len(per), (n_boot, len(per)))], axis=1)
    return dict(factor=float(np.median(per)), low=float(np.percentile(boot, 10)), high=float(np.percentile(boot, 90)),
                events=len(events), readings=len(rows),
                per_event=[dict(event=e, ratio=float(x)) for e, x in zip(events, per)])
