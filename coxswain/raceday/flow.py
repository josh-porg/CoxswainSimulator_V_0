r"""The river's flow today and what it costs: USGS gauge 01104500 (Charles River at Waltham).

The current on the Charles comes from continuity, ``U = Q / A`` (:class:`coxswain.river.charles.
ContinuityFlow`): measured discharge over the surveyed cross-section. In October the basin is nearly
slack (median 2.8 m3/s over several hundred square metres), so the expected answer is "negligible";
the point of reading it live is to say so with a number, and to catch the rare high-water day. The
race runs upstream, so more flow is a little slower.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional, Tuple

from coxswain.river import charles

SITE = "01104500"
URL = ("https://waterservices.usgs.gov/nwis/iv/?sites=%s&parameterCd=00060&format=json&period=P1D" % SITE)
CFS = 0.0283168466


def latest(timeout: float = 20.0) -> Tuple[float, str]:
    """Latest discharge, m3/s, and its timestamp (provisional data)."""
    req = urllib.request.Request(URL, headers={"User-Agent": "CoxswainSimulator race card (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    v = d["value"]["timeSeries"][0]["values"][0]["value"][-1]
    return float(v["value"]) * CFS, v["dateTime"]


def _line_time(model, boat, reference_time: float) -> float:
    """Seconds for the line, integrated from the speed over the path (as the race card does; the
    evaluator's ``elapsed`` carries penalty terms and is not a race time)."""
    from coxswain.raceday.card import _evaluate, _sections
    return sum(s.time_s for s in _sections(model, _evaluate(model, boat, reference_time, None), None))


def effect_s(model, boat, reference_time: float, discharge: float, evaluate=None) -> float:
    """Seconds this discharge adds to the line against the October median, at ``reference_time``."""
    median = charles.monthly_discharge(10)
    keep = model.flow
    try:
        model.flow = charles.ContinuityFlow(model.course, discharge=discharge)
        today = _line_time(model, boat, reference_time)
        model.flow = charles.ContinuityFlow(model.course, discharge=median)
        usual = _line_time(model, boat, reference_time)
    finally:
        model.flow = keep
    return float(today - usual)


def card_entry(model, boat, reference_time: float, evaluate=None) -> Optional[dict]:
    try:
        q, when = latest()
    except Exception as error:                      # noqa: BLE001
        return dict(ok=False, error=type(error).__name__)
    return dict(ok=True, discharge_m3s=q, discharge_cfs=q / CFS, at=when,
                october_median_m3s=charles.monthly_discharge(10),
                effect_s=effect_s(model, boat, reference_time, q), site=SITE)


# ---- Saturday's river: forecast flow (National Water Model) and the basin level ------------------
NWM = "https://api.water.noaa.gov/nwps/v1/reaches/5868225/streamflow?series=%s"     # the Waltham reach
LEVEL_SITE = "01104715"            # New Charles River Dam, basin gage height (ft)
LEVEL_MEDIAN_FT = 107.32           # daily means 2025-09 to 2026-10 (5-95%: 106.88-107.78)


SHORT_REACH_H = 18.0               # how far past its last hour the short range may stand in


def forecast_at(when, timeout: float = 30.0):
    """National Water Model discharge for the hour of ``when`` (m3/s, and which run), short range
    (18 h) when it reaches, else the medium-range ensemble mean. When neither reaches (the medium range
    came back empty on 16 Oct), the short range's last hour if it is within SHORT_REACH_H of ``when``:
    a river already rising towards the start is better described by it than by the flow observed now.
    None beyond that."""
    from datetime import datetime
    last_short = None
    for series, key in (("short_range", "shortRange"), ("medium_range", "mediumRange")):
        try:
            req = urllib.request.Request(NWM % series, headers={"User-Agent": "CoxswainSimulator race card (research)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                block = json.load(r).get(key) or {}
        except Exception:                               # noqa: BLE001
            continue
        member = block.get("series") or block.get("mean") or next(iter(block.values()), None)
        data = (member or {}).get("data", [])
        best = None
        for x in data:
            t = datetime.fromisoformat(x["validTime"].replace("Z", "+00:00"))
            if best is None or abs((t - when).total_seconds()) < abs((best[0] - when).total_seconds()):
                best = (t, x["flow"])
        if best is not None and abs((best[0] - when).total_seconds()) <= 3 * 3600:
            return best[1] * CFS, "National Water Model, %s" % series.replace("_", " ")
        if series == "short_range" and data:
            t = datetime.fromisoformat(data[-1]["validTime"].replace("Z", "+00:00"))
            last_short = (t, data[-1]["flow"])
    if last_short and 0 <= (when - last_short[0]).total_seconds() <= SHORT_REACH_H * 3600:
        return last_short[1] * CFS, "National Water Model, short range's last hour (%s UTC; the start not yet in range)" \
            % last_short[0].strftime("%a %H:%M")
    return None


def basin_level(timeout: float = 30.0):
    """The basin's level now against its median pool, feet (USGS gage height at the dam)."""
    url = "https://waterservices.usgs.gov/nwis/iv/?sites=%s&parameterCd=00065&format=json&period=P1D" % LEVEL_SITE
    req = urllib.request.Request(url, headers={"User-Agent": "CoxswainSimulator race card (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        v = json.load(r)["value"]["timeSeries"][0]["values"][0]["value"][-1]
    return float(v["value"]) - LEVEL_MEDIAN_FT, v["dateTime"]
