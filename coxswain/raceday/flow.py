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
