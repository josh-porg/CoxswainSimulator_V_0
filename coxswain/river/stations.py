r"""Turning a weather station reading into a wind field over the Charles.

The canopy model needs a wind at the blending height.  What exists on race
morning is a number from an airport, and the two are not the same thing --
not because the airport is far away, but because a station's reading is
shaped by *its own* surroundings before it is shaped by the weather.

The stations that exist
-----------------------
From the NWS station list for the Charles basin, by distance from Weeks:

==========  ======  ====================================================
station     km      what it is
==========  ======  ====================================================
**KBOS**    8.9     Logan.  The only station close enough to matter, and
                    the one with a documented exposure and a long record.
                    Sits on a peninsula in the harbour: open water to the
                    east, downtown Boston to the west, so **its own
                    roughness depends strongly on wind direction.**
KBED        18.2    Hanscom Field, Bedford.  Inland and to the northwest.
KOWD        20.3    Norwood Memorial.  Inland and to the southwest.
==========  ======  ====================================================

KBOS alone gives the wind.  The other two earn their place by
**disagreeing** with it: see :func:`sea_breeze_risk`.

Why one station is not enough in October
----------------------------------------
A Boston sea breeze can put an easterly on the coast while the synoptic
flow inland is westerly, with the front somewhere in between -- and the
Charles basin is exactly "in between".  Reading KBOS alone on such a day
tells you the wind at Logan and nothing about the wind at Weeks.  Two
inland stations turn that from an unknown into a *detectable* condition:
if KBOS and KBED disagree by more than a sector, the reach is in
transition and no single-station prediction should be trusted.

The exposure correction
-----------------------
A wind measured over rough ground is slower than the same weather
measured over open ground, so station readings are not comparable until
they are corrected.  Wieringa's standard two-step [W86]_ does it by
taking the reading up to a blending height over the station's *own*
roughness and bringing it back down over the standard open exposure::

    u_pot = u_meas * ln(H/z0_stn) / ln(z_a/z0_stn)
                   * ln(z_a/z0_open) / ln(H/z0_open)

The result -- the **potential wind** -- is what the weather would read
over open short grass at 10 m, and is the quantity every downstream model
here already expects.

The per-sector roughnesses below are Davenport classes assigned from each
station's known geography rather than computed from footprints, because
the thing that matters most for Logan is which sectors are *harbour*, and
a building dataset cannot say that.  They are stated, not fitted.

References
----------
.. [W86] Wieringa, J. (1986) *Roughness-dependent geographical
   interpolation of surface wind speed averages*, Q. J. Royal Met.
   Society 112, 867-889.
.. [W92] Wieringa, J. (1992) *Updating the Davenport roughness
   classification*, J. Wind Eng. Ind. Aero. 41, 357-368.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

__all__ = ["Station", "STATIONS", "Observation", "latest", "potential_wind",
           "sea_breeze_risk", "charles_reference"]

API = "https://api.weather.gov/stations/%s/observations/latest"
AGENT = "CoxswainSimulator/0.1 (rowing research)"

#: Standard open exposure a potential wind refers to, m.
Z0_OPEN = 0.03
#: Blending height for the exposure correction, m.  Wieringa uses 60 m.
BLENDING = 60.0

_SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass(frozen=True)
class Station:
    """A reporting station and the ground around it."""

    identifier: str
    name: str
    latitude: float
    longitude: float
    distance_km: float
    anemometer_height: float = 10.0
    #: Roughness length by the sector the wind comes FROM, m.
    sector_z0: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    def roughness_for(self, wind_from: float) -> float:
        index = int(((float(wind_from) % 360.0) + 22.5) // 45.0) % 8
        return self.sector_z0.get(_SECTORS[index], Z0_OPEN)


#: Logan's fetch is harbour to the east and downtown to the west, which is
#: a factor of a thousand in roughness across two sectors.  Ignoring that
#: makes a westerly read as 30% weaker than the weather actually is.
KBOS = Station(
    "KBOS", "Boston, Logan International", 42.3606, -71.0106, 8.9,
    sector_z0={"N": 0.03, "NE": 0.0005, "E": 0.0005, "SE": 0.0005,
               "S": 0.01, "SW": 0.40, "W": 0.60, "NW": 0.30},
    note="peninsula in the harbour; open water east, city west")

KBED = Station(
    "KBED", "Hanscom Field, Bedford", 42.4681, -71.2890, 18.2,
    sector_z0={s: 0.10 for s in _SECTORS},
    note="inland airfield with woodland around; used to detect sea breeze")

KOWD = Station(
    "KOWD", "Norwood Memorial", 42.1908, -71.1728, 20.3,
    sector_z0={s: 0.10 for s in _SECTORS},
    note="inland airfield, wooded; the southern half of the sea-breeze check")

STATIONS = {s.identifier: s for s in (KBOS, KBED, KOWD)}


@dataclass(frozen=True)
class Observation:
    """One station reading."""

    station: str
    timestamp: str
    speed: float                 # m/s as reported
    direction: float             # degrees the wind comes FROM
    gust: float = float("nan")

    @property
    def calm(self) -> bool:
        return not np.isfinite(self.direction) or self.speed < 0.5


def latest(identifier: str, timeout: float = 30.0) -> Observation:
    """Current observation from the NWS API.

    Raises on a network failure rather than returning a fabricated calm --
    a race-day tool that silently reports no wind when it cannot reach the
    network is worse than one that says it is broken.
    """
    request = urllib.request.Request(API % identifier,
                                     headers={"User-Agent": AGENT,
                                              "Accept": "application/geo+json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    p = payload["properties"]

    def value(key, scale=1.0):
        entry = p.get(key) or {}
        raw = entry.get("value")
        return float("nan") if raw is None else float(raw) * scale

    return Observation(
        station=identifier, timestamp=p.get("timestamp", ""),
        speed=value("windSpeed", 1.0 / 3.6),          # km/h -> m/s
        direction=value("windDirection"),
        gust=value("windGust", 1.0 / 3.6))


def potential_wind(observation: Observation, station: Station = None) -> float:
    """The reading corrected to an open 10 m exposure, m/s, [W86]_."""
    station = station or STATIONS[observation.station]
    if observation.calm:
        return 0.0
    z0 = station.roughness_for(observation.direction)
    anemometer = station.anemometer_height
    return float(observation.speed
                 * np.log(BLENDING / z0) / np.log(anemometer / z0)
                 * np.log(anemometer / Z0_OPEN) / np.log(BLENDING / Z0_OPEN))


def sea_breeze_risk(observations) -> Tuple[bool, str]:
    """Do the stations agree about which way the wind is blowing?

    A Boston sea breeze puts an easterly on the coast under a westerly
    aloft, with the front inland of Logan and often right over the basin.
    On such a day KBOS describes the harbour and nothing else, and the
    right answer is to go and look at the water rather than to trust any
    of this.
    """
    usable = [o for o in observations if not o.calm]
    if len(usable) < 2:
        return False, "not enough stations reporting a direction"
    coastal = next((o for o in usable if o.station == "KBOS"), None)
    inland = [o for o in usable if o.station != "KBOS"]
    if coastal is None or not inland:
        return False, "no coastal/inland pair"
    spreads = [abs((o.direction - coastal.direction + 180.0) % 360.0 - 180.0)
               for o in inland]
    worst = max(spreads)
    if worst > 60.0:
        return True, ("KBOS and the inland stations disagree by %.0f degrees "
                      "-- sea-breeze front or a passing trough.  Any "
                      "single-station prediction for the basin is unsafe; "
                      "read the water on the warm-up." % worst)
    return False, ("stations agree within %.0f degrees -- one synoptic flow, "
                   "so KBOS can speak for the basin" % worst)


def charles_reference(observations=None) -> Tuple[float, float, str]:
    """Best estimate of the open-exposure 10 m wind over the Charles.

    Returns ``(speed, direction, commentary)``.  Speed is the potential
    wind, which is what :class:`~coxswain.hydro.canopy.ShelteredWind`
    expects as its reference; direction is the meteorological bearing the
    wind comes from.

    KBOS carries the estimate because it is twice as close as anything
    else.  The inland stations are used only to decide whether that is
    defensible on the day.
    """
    if observations is None:
        observations = []
        for identifier in STATIONS:
            try:
                observations.append(latest(identifier))
            except Exception as error:                   # noqa: BLE001
                observations.append(None)
                del error
        observations = [o for o in observations if o is not None]
    if not observations:
        raise RuntimeError("no station reported; check the network")

    # A METAR cycle can carry no wind direction at all -- variable, or
    # simply missing -- and KBOS did exactly that on the first live run of
    # this function.  Taking it anyway put a NaN into the wind field,
    # which then propagated silently through the whole race estimate and
    # printed "nan seconds" as though that were a forecast.  Fall through
    # to the next station instead, and say which one is speaking.
    usable = [o for o in observations if not o.calm]
    if not usable:
        return 0.0, float("nan"), ("every station reports calm or variable "
                                   "-- nothing to correct for")
    order = sorted(usable, key=lambda o: STATIONS[o.station].distance_km)
    chosen = order[0]
    speed = potential_wind(chosen)
    risky, why = sea_breeze_risk(observations)
    prefix = "CAUTION: " if risky else ""
    if chosen.station != "KBOS":
        prefix += ("KBOS had no usable wind this cycle, so %s (%.0f km) is "
                   "carrying the estimate -- treat it as weaker evidence. "
                   % (chosen.station, STATIONS[chosen.station].distance_km))
    return speed, chosen.direction, prefix + why
