r"""Who we meet on the course, and where: passes from the simulated field and the start order.

A head race starts crews in bow-number order at a fixed interval (about 15 s at the Charles, 15-20 s
in some years). With each crew's time for the course, every pair's encounter follows: a crew that
starts ``dt`` seconds after us and rows the course in ``T_i`` against our ``T`` draws level when
``t / T = (t - dt) / T_i``, at the distance ``x = L dt / (T - T_i)``; if ``0 < x < L``
there is a pass there. Run over the field simulation's draws (:mod:`coxswain.raceday.goals`: every
other crew's time on the same day as ours), that gives for each neighbour the chance of a pass
either way and where it happens, and for each bridge the chance a pass falls within the approach
to it, which is what decides whether the planned arch will be free.

Constant speed per crew is the simplification: a crew that starts hard and fades draws level a
little earlier than computed. The start interval and the course length are parameters.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

INTERVAL_S = 15.0
COURSE_M = 4822.0
BRIDGE_ZONE_M = 100.0    # a pass within this distance either side of a bridge contests the arches


def read_entries(path: str) -> List[dict]:
    """Bow and club from an entries CSV (``#`` comment lines skipped)."""
    rows = [r for r in csv.DictReader(line for line in open(path, encoding="utf-8") if not line.startswith("#"))]
    return [dict(bow=int(r["bow"]), club=r["club"].strip(), division=(r.get("age_division") or "").strip())
            for r in rows]


def own_draws(field, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Our crew's time per draw, on the same day as the field's draw."""
    rng = np.random.default_rng(7) if rng is None else rng
    return field.own_mean * field.day * np.exp(rng.normal(0.0, field.own_sd / field.own_mean, len(field.day)))


def encounters(own_t: np.ndarray, others_t: np.ndarray, own_bow: int, other_bows: Sequence[int],
               interval: float = INTERVAL_S, length: float = COURSE_M) -> Tuple[np.ndarray, np.ndarray]:
    """Where each pass happens, per draw and crew (NaN for none), and its sign: +1 we pass them, -1 they
    pass us. ``own_t``: (n,), ``others_t``: (n, k) seconds for the course."""
    dt = (np.asarray(other_bows, float) - own_bow) * interval          # their start minus ours
    # level when t / T = (t - dt) / T_i, i.e. at x = L dt / (T - T_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = length * dt[None, :] / (own_t[:, None] - others_t)
    where = np.where((x > 0) & (x < length), x, np.nan)
    sign = np.where(dt[None, :] > 0, -1.0, 1.0) * np.isfinite(where)
    return where, sign


@dataclass
class TrafficSummary:
    neighbours: List[dict]
    expected_passes_made: float
    expected_passed_by: float
    p_clear_race: float
    by_stretch: List[dict]
    bridges: List[dict]
    interval: float

    def as_dict(self) -> dict:
        return dict(neighbours=self.neighbours, expected_passes_made=self.expected_passes_made,
                    expected_passed_by=self.expected_passed_by, p_clear_race=self.p_clear_race,
                    by_stretch=self.by_stretch, bridges=self.bridges, interval=self.interval)


def summarise(field, entries: List[dict], ours: str, landmarks: Sequence[Tuple[str, float]],
              bridges: Dict[str, float], interval: float = INTERVAL_S, length: float = COURSE_M,
              window: int = 5, seed: int = 7) -> TrafficSummary:
    """The traffic picture for our crew: neighbours within ``window`` bows, passes by stretch and at
    the bridges. ``field.others`` columns are the entries in order without ours."""
    own = next(e for e in entries if e["club"] == ours)
    others = [e for e in entries if e["club"] != ours]
    if len(others) != field.others.shape[1]:
        raise ValueError("entries (%d others) do not match the field (%d)" % (len(others), field.others.shape[1]))
    t_own = own_draws(field, np.random.default_rng(seed))
    where, sign = encounters(t_own, field.others, own["bow"], [e["bow"] for e in others], interval, length)
    made = sign > 0
    taken = sign < 0
    neighbours = []
    for j, e in enumerate(others):
        if abs(e["bow"] - own["bow"]) > window:
            continue
        w = where[:, j]
        pm, pt = float(np.mean(made[:, j])), float(np.mean(taken[:, j]))
        neighbours.append(dict(bow=e["bow"], club=e["club"], division=e["division"],
                               starts_s=(e["bow"] - own["bow"]) * interval,
                               p_we_pass=pm, p_they_pass=pt,
                               median_at_m=float(np.nanmedian(w)) if np.isfinite(w).any() else None))
    neighbours.sort(key=lambda d: d["bow"])
    edges = [d for _, d in landmarks]
    names = [n for n, _ in landmarks]
    by = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        inside = (where >= a) & (where < b)
        by.append(dict(start=names[i], end=names[i + 1],
                       expected_made=float(np.mean(np.sum(inside & made, axis=1))),
                       expected_taken=float(np.mean(np.sum(inside & taken, axis=1)))))
    br = []
    for name, d in bridges.items():
        near = np.abs(where - d) <= BRIDGE_ZONE_M
        br.append(dict(bridge=name, at_m=d, p_pass_at=float(np.mean(np.any(near, axis=1))),
                       p_we_pass_at=float(np.mean(np.any(near & made, axis=1))),
                       p_passed_at=float(np.mean(np.any(near & taken, axis=1)))))
    any_pass = np.any(np.isfinite(where), axis=1)
    return TrafficSummary(neighbours, float(np.mean(np.sum(made, axis=1))), float(np.mean(np.sum(taken, axis=1))),
                          float(np.mean(~any_pass)), by, br, interval)
