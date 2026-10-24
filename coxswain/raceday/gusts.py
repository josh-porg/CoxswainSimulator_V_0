"""Gusts: the forecast's peak wind turned into a time allowance, switchable on the card, map and report.

The race card times the line in the forecast's sustained wind. A gusty wind of the same mean costs
more, because the drag of the air grows with the square of the relative wind, so the strong moments
cost more than the lulls give back. Averaging the line's time over the spread of wind speeds gives the
allowance. The spread comes from the gust factor G = gust / sustained: for a peak gust over an hour,
G = 1 + g I with peak factor g ~ 3 (Wieringa 1973; ESDU 83045), so the turbulence intensity is
I = (G - 1) / g and the standard deviation of the speed is sigma = I U.

The NWS hourly forecast leaves ``windGust`` empty; the gridded forecast behind it carries it, in km/h,
as ISO intervals. Only the speed fluctuates here: swings in direction, and the steering they call
for, are not in the allowance (the rudder cost was measured on turns, not gust corrections).

The wind table (``scripts/wind_table.py``) holds the line's time factor against still air on a grid
of speed and direction. The allowance is half the table's curvature in speed times the variance of
the speed (exact for a cost quadratic in the wind); the per-section shares are averaged over a normal
spread of speeds with the same bilinear interpolation the pages use. ``scripts/raceday_site_loader.js`` and
``scripts/raceday_briefing.html`` mirror this in the browser.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

PEAK_FACTOR = 3.0                         # g in G = 1 + g I
MAX_INTENSITY = 0.5
NODES = np.linspace(-3.0, 3.0, 25)        # standard normal nodes for the average
WEIGHTS = np.exp(-0.5 * NODES ** 2) / np.exp(-0.5 * NODES ** 2).sum()


def intensity(speed: float, gust: Optional[float]) -> float:
    """Turbulence intensity from the sustained speed and the peak gust (0 when there is no gust)."""
    if gust is None or speed is None or not (speed > 0.3) or not math.isfinite(gust) or gust <= speed:
        return 0.0
    return min((gust / speed - 1.0) / PEAK_FACTOR, MAX_INTENSITY)


def table_at(table: dict, speed: float, direction: float) -> dict:
    """The wind table's row at any speed and direction (bilinear; speed clamped to the table)."""
    S, D = table["speeds"], table["directions"]
    n = len(D)
    v = max(0.0, min(speed or 0.0, S[-1]))
    i = 0
    while i < len(S) - 2 and v > S[i + 1]:
        i += 1
    ws = (v - S[i]) / (S[i + 1] - S[i])
    x = ((direction if direction is not None and math.isfinite(direction) else 0.0) % 360.0) / (360.0 / n)
    j = int(math.floor(x)) % n
    wd = x - math.floor(x)
    j1 = (j + 1) % n
    row = lambda a, b: table["rows"][a * n + b]           # noqa: E731
    return _mix([(row(i, j), (1 - ws) * (1 - wd)), (row(i, j1), (1 - ws) * wd),
                 (row(i + 1, j), ws * (1 - wd)), (row(i + 1, j1), ws * wd)])


def _mix(rows):
    out = dict(factor=0.0, share=None, felt=None, rudder=None)
    for r, w in rows:
        out["factor"] += w * r["factor"]
        for k in ("share", "felt", "rudder"):
            a = np.asarray(r[k], float) * w
            out[k] = a if out[k] is None else out[k] + a
    out["share"] = out["share"] / out["share"].sum()
    return out


def curvature(table: dict, speed: float, direction: float) -> float:
    """d2(factor)/d(speed)2 from a second difference one table step wide. Linear interpolation between
    the table's speeds would bias an average over gusts (low between nodes, high at them); the second
    difference of the interpolant is exact for a cost quadratic in the wind wherever it is taken."""
    S = table["speeds"]
    h = S[1] - S[0]
    u = min(max(speed or 0.0, S[0] + h), S[-1] - h)
    f = lambda v: table_at(table, v, direction)["factor"]       # noqa: E731
    return (f(u + h) - 2.0 * f(u) + f(u - h)) / (h * h)


def with_gusts(table: dict, speed: float, direction: float, gust: Optional[float]) -> dict:
    """The table's row with the gust allowance: the time factor gains half the curvature times the
    variance of the speed; the per-section shares, felt wind and turning are averaged over the spread.
    The steady row when there are no gusts."""
    sigma = intensity(speed, gust) * (speed or 0.0)
    steady = table_at(table, speed, direction)
    if sigma <= 0.0:
        return steady
    out = _mix([(table_at(table, max(0.0, speed + sigma * z), direction), w) for z, w in zip(NODES, WEIGHTS)])
    out["factor"] = steady["factor"] + 0.5 * curvature(table, speed, direction) * sigma ** 2
    return out


def allowance(table: dict, speed: float, direction: float, gust: Optional[float]) -> dict:
    """What counting the gusts does to the line: the time ratio, the wind-cost ratio, and seconds at
    the table's reference pace."""
    steady = table_at(table, speed, direction)["factor"]
    gusty = with_gusts(table, speed, direction, gust)["factor"]
    I = intensity(speed, gust)
    return dict(gust_ms=gust if gust is not None and math.isfinite(gust) else None, intensity=I,
                sigma_ms=I * (speed or 0.0), ratio=gusty / steady,
                wind_ratio=(gusty - 1.0) / (steady - 1.0) if abs(steady - 1.0) > 1e-6 else 1.0,
                added_s=(gusty - steady) * table["reference_s"])


# ---- the forecast's gusts (NWS gridded forecast) -------------------------------------------------
def _interval(text: str):
    start, dur = text.split("/")
    t0 = datetime.fromisoformat(start)
    m = re.match(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", dur)
    d, h, mi = (int(g or 0) for g in m.groups())
    return t0, t0 + timedelta(days=d, hours=h, minutes=mi)


def gust_at(when: datetime, point=None) -> Optional[float]:
    """Forecast peak gust, m/s, for the hour of ``when`` (timezone-aware); None beyond the forecast."""
    from coxswain.raceday.forecast import COURSE_POINT, POINTS, _get
    meta = _get(POINTS % (point or COURSE_POINT))
    grid = _get(meta["properties"]["forecastGridData"])["properties"]
    for v in grid.get("windGust", {}).get("values", []):
        t0, t1 = _interval(v["validTime"])
        if t0 <= when < t1 and v.get("value") is not None:
            return float(v["value"]) / 3.6
    return None
