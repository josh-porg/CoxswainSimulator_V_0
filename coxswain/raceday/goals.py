r"""Target times for race goals, with their uncertainty, from a simulated field.

A goal is a place: medal (3rd or better), beat last year's place, requalify (the top half). What
time it takes depends on who else is in the event and how fast they go this year, which is not
known. The field model (``research/crew/feeders/charles_update_2026.py``) simulates the event many
times, each crew from its own evidence (a 2025 Charles time, a 2026 feeder time, or a draw from last
year's field for a crew with neither), all on one day factor for the river. Saved as an ``.npz``:
``others`` (draws x crews, seconds), ``day`` (the day factor per draw), ``n``, ``nq``, and the crew's
own forecast ``own_mean``/``own_sd`` and ``last_place``.

For a time ``T`` rowed on the same day as each draw, the crew's place in that draw is one plus the
number of other crews faster than ``T x day``. Over the draws that gives the chance of achieving the
goal at ``T``. Inverted, it gives the **time for a goal at a stated chance**: the central target is
the 50% time, and the range runs from the 80% time (fast end, likely enough) to the 20% time (slow
end, a long shot). All times are on the field model's average-day scale; the race card scales them by
today's wind.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Field:
    others: np.ndarray
    day: np.ndarray
    n: int
    nq: int
    own_mean: float
    own_sd: float
    last_place: int        # -1 if the crew did not race last year

    @classmethod
    def load(cls, path: str) -> "Field":
        z = np.load(path)
        return cls(others=z["others"].astype(float), day=z["day"].astype(float), n=int(z["n"]), nq=int(z["nq"]),
                   own_mean=float(z["own_mean"]), own_sd=float(z["own_sd"]), last_place=int(z["last_place"]))

    def chance(self, time: float, place: int) -> float:
        """P(our place <= place) if we row ``time`` (average-day scale)."""
        faster = (self.others < (time * self.day)[:, None]).sum(axis=1)
        return float(np.mean(faster + 1 <= place))

    def time_for(self, place: int, chance: float, lo: float = 600.0, hi: float = 3000.0) -> float:
        """The time that achieves ``place`` or better with probability ``chance`` (bisection)."""
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.chance(mid, place) >= chance:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def goal(field: Field, place: int, name: str) -> dict:
    """Central (50%) target with its 80%-to-20% range, still-air seconds on the average-day scale."""
    return dict(name=name, place=place, central=field.time_for(place, 0.5),
                fast=field.time_for(place, 0.8), slow=field.time_for(place, 0.2),
                kind="field", note="50%% chance at the central time; 80%% at the fast end, 20%% at the slow end")


def forecast(field: Field, name: str = "best achievable") -> dict:
    """The crew's own forecast as a target: centre and an 80% range (+-1.28 sd)."""
    return dict(name=name, place=None, central=field.own_mean, fast=field.own_mean - 1.2816 * field.own_sd,
                slow=field.own_mean + 1.2816 * field.own_sd, kind="forecast",
                note="the crew's forecast; 80% range")


def chances(field: Field, places, seed: int = 7) -> dict:
    """Our chance of each place-or-better, rowing our forecast on the same day as the field
    (own time = forecast x day x crew noise, per draw)."""
    rng = np.random.default_rng(seed)
    own = field.own_mean * field.day * np.exp(rng.normal(0.0, field.own_sd / field.own_mean, len(field.day)))
    place = 1 + (field.others < own[:, None]).sum(axis=1)
    out = {int(k): float(np.mean(place <= k)) for k in places}
    out["median_place"] = float(np.median(place))
    out["place_80"] = [float(np.percentile(place, 10)), float(np.percentile(place, 90))]
    return out
