"""The NWS hourly forecast for the Charles basin, at a given time.

The station module (:mod:`coxswain.river.stations`) reads what the wind *is*; on the night before
and on race morning the question is what it *will be* at the start time. The National Weather
Service publishes an hourly gridded forecast for any point; this reads it for the middle of the
course. The forecast wind is quoted at 10 m over open ground, the same reference the sheltering
model (:class:`coxswain.hydro.canopy.ShelteredWind`) takes, so it goes in without conversion.

Fails loudly on a network or parse error: a race-day tool that silently reports calm when it
cannot reach the forecast is worse than one that says it is broken.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from coxswain.river.stations import AGENT

#: Mid-course (near Weeks Footbridge), where the forecast grid cell is taken.
COURSE_POINT = (42.3700, -71.1220)
POINTS = "https://api.weather.gov/points/%.4f,%.4f"
COMPASS = {name: 22.5 * i for i, name in enumerate(
    "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split())}
MPH = 0.44704


@dataclass(frozen=True)
class HourlyWind:
    start: str            # ISO time, local offset
    speed: float          # m/s at 10 m, the forecast's upper figure when a range is given
    direction: float      # degrees FROM, NaN if variable
    gust: float           # m/s, NaN if not given
    temperature_c: float
    short: str            # the forecast's own wording, e.g. "Mostly Sunny"
    precipitation: float  # probability, %


def _get(url: str, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": "application/geo+json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _mph(text: str) -> float:
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]
    return max(numbers) * MPH if numbers else float("nan")


def hourly(point=COURSE_POINT) -> List[HourlyWind]:
    """Every hourly period the NWS currently forecasts for the point (about seven days)."""
    meta = _get(POINTS % point)
    periods = _get(meta["properties"]["forecastHourly"])["properties"]["periods"]
    out = []
    for p in periods:
        temp = p.get("temperature")
        if p.get("temperatureUnit") == "F" and temp is not None:
            temp = (temp - 32.0) * 5.0 / 9.0
        pop = (p.get("probabilityOfPrecipitation") or {}).get("value")
        out.append(HourlyWind(start=p["startTime"], speed=_mph(p.get("windSpeed", "")),
                              direction=COMPASS.get(p.get("windDirection", ""), float("nan")),
                              gust=_mph(p.get("windGust", "") or ""),
                              temperature_c=float("nan") if temp is None else float(temp),
                              short=p.get("shortForecast", ""), precipitation=float(pop or 0.0)))
    return out


def at(when: datetime, periods: Optional[List[HourlyWind]] = None) -> Optional[HourlyWind]:
    """The forecast hour containing ``when`` (timezone-aware), or None if beyond the forecast."""
    periods = periods if periods is not None else hourly()
    for p in periods:
        start = datetime.fromisoformat(p.start)
        if start <= when < start.replace(minute=59, second=59) or abs((start - when).total_seconds()) < 1800:
            return p
    return None


def _mid_mph(text: str) -> float:
    """The middle of a "10 to 15 mph" range (the 12-hour forecast's wind is a range, not an hour)."""
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]
    return 0.5 * (min(numbers) + max(numbers)) * MPH if numbers else float("nan")


def daily_at(when: datetime, point=COURSE_POINT) -> Optional[HourlyWind]:
    """The NWS 12-hour forecast period containing ``when`` (seven days out, a day or two beyond the
    hourly forecast). Its wind is a range ("10 to 15 mph"); the middle is used, so the card's 30%
    wind spread still applies. ``short`` carries the period's name ("Saturday")."""
    meta = _get(POINTS % point)
    for p in _get(meta["properties"]["forecast"])["properties"]["periods"]:
        start, end = datetime.fromisoformat(p["startTime"]), datetime.fromisoformat(p["endTime"])
        if start <= when < end:
            temp = p.get("temperature")
            if p.get("temperatureUnit") == "F" and temp is not None:
                temp = (temp - 32.0) * 5.0 / 9.0
            pop = (p.get("probabilityOfPrecipitation") or {}).get("value")
            return HourlyWind(start=p["startTime"], speed=_mid_mph(p.get("windSpeed", "")),
                              direction=COMPASS.get(p.get("windDirection", ""), float("nan")),
                              gust=float("nan"), temperature_c=float("nan") if temp is None else float(temp),
                              short="%s: %s (%s)" % (p.get("name", ""), p.get("shortForecast", ""), p.get("windSpeed", "")),
                              precipitation=float(pop or 0.0))
    return None
