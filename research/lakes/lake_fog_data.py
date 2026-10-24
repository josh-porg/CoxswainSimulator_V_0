"""Hourly morning table for lake fog on Lake Sammamish: the buoy, the water, the airports, the forecast as issued.

    .venv/Scripts/python.exe research/lakes/lake_fog_data.py data/local/sammamish/lake.json

Imported by lake_fog.py; run on its own it prints the table for the last few mornings.
* Buoy (King County, open water, south basin): air temperature, humidity, wind, solar radiation (15 min, hourly
  means). Times as served (Pacific daylight time all year, as the relay found); converted to local clock time.
* Water: the profiler's shallowest reading (<= 1.5 m) of each profile, else the parked reading at 10 m.
* Airports (KRNT and the cloud airports): visibility, fog codes, lowest broken/overcast base (ceiling).
* Forecast at the lake point (Open-Meteo historical forecast, gfs_seamless).
* Clear-sky solar (Haurwitz 1945) at the buoy, and the sun's elevation.
"""
import csv, json, math, os, sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

PT = ZoneInfo("America/Los_Angeles")
PDT = timezone(timedelta(hours=-7))
LAT, LON = 47.58167, -122.09


def _tab(path):
    rows = list(csv.reader(open(path, encoding="utf-8", errors="replace"), delimiter="\t"))
    return rows[0], rows[1:]


def local(t_pdt):
    """Buoy stamps are PDT all year: to local clock time (naive)."""
    return t_pdt.replace(tzinfo=PDT).astimezone(PT).replace(tzinfo=None)


def buoy_met(D):
    out = {}
    for y in range(2023, 2027):
        p = os.path.join(D, "buoy", "met_%d.csv" % y)
        if not os.path.exists(p):
            continue
        h, rows = _tab(p)
        col = {n.split(" (")[0]: i for i, n in enumerate(h)}
        for r in rows:
            try:
                t = local(datetime.strptime(r[0], "%m/%d/%Y %I:%M:%S %p"))
                v = dict(rh=float(r[col["Relative Humidity"]]), solar=float(r[col["Solar Radiation"]]), wind=float(r[col["Wind Speed"]]),
                         tair=float(r[col["Air Temperature"]]))
            except (ValueError, IndexError, KeyError):
                continue
            k = t.replace(minute=0, second=0)
            out.setdefault(k, []).append(v)
    return {k: {f: float(np.mean([x[f] for x in v])) for f in v[0]} for k, v in out.items()}


def water(D):
    """Hourly series of near-surface water temperature (deg C): profiles' top reading, gaps from the 10 m park."""
    pts = []
    for y in range(2023, 2027):
        p = os.path.join(D, "buoy", "profile_%d.csv" % y)
        if os.path.exists(p):
            h, rows = _tab(p)
            for r in rows:
                try:
                    d, T = float(r[1]), float(r[2])
                    if d <= 1.5 and -1 < T < 35:
                        pts.append((local(datetime.strptime(r[0], "%m/%d/%Y %I:%M:%S %p")), T, "top"))
                except (ValueError, IndexError):
                    pass
    park = []
    for y in range(2023, 2027):
        p = os.path.join(D, "buoy", "hourly_%d.csv" % y)
        if os.path.exists(p):
            h, rows = _tab(p)
            for r in rows:
                try:
                    park.append((local(datetime.strptime(r[0], "%m/%d/%Y %I:%M:%S %p")), float(r[2])))
                except (ValueError, IndexError):
                    pass
    top = sorted(pts)
    tt = np.array([a.timestamp() for a, _, _ in top]); TT = np.array([b for _, b, _ in top])
    pk = sorted(park)
    pt = np.array([a.timestamp() for a, _ in pk]); PTT = np.array([b for _, b in pk])
    return dict(top_t=tt, top=TT, park_t=pt, park=PTT)


def water_at(W, t, lag_h=0.0):
    """Surface water temperature known at time t (the latest top reading within 2 days, else the 10 m park)."""
    s = (t - timedelta(hours=lag_h)).timestamp()
    i = np.searchsorted(W["top_t"], s) - 1
    if i >= 0 and s - W["top_t"][i] < 2 * 86400:
        # the median of the last day's top readings (profiles a few times a day; the top metre warms in the afternoon)
        m = (W["top_t"] <= s) & (W["top_t"] > s - 86400)
        return float(np.median(W["top"][m])) if m.any() else float(W["top"][i])
    j = np.searchsorted(W["park_t"], s) - 1
    if j >= 0 and s - W["park_t"][j] < 2 * 86400:
        return float(W["park"][j])
    return float("nan")


def airports(D, ids):
    out = {}
    for ap in ids:
        p = os.path.join(D, "fog", "obs_%s.csv" % ap)
        for r in csv.DictReader(open(p)):
            try:
                t = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            k = (t + timedelta(minutes=30)).replace(minute=0)
            try:
                vis = float(r["vsby"])
            except ValueError:
                vis = float("nan")
            ceil = float("inf")
            for c, l in (("skyc1", "skyl1"), ("skyc2", "skyl2")):
                if r.get(c) in ("BKN", "OVC", "VV"):
                    try:
                        ceil = min(ceil, float(r[l]))
                    except ValueError:
                        pass
            wx = r.get("wxcodes") or ""
            fog = (vis < 0.62) or "FG" in wx
            low = ceil <= 1500.0
            cur = out.setdefault(k, {}).get(ap)
            rec = dict(vis=vis, ceil=ceil, fog=fog, low=low, rain=("RA" in wx or "SN" in wx))
            if cur is None or (fog and not cur["fog"]):
                out[k][ap] = rec
    return out


def forecast(D, where="lake"):
    h = json.load(open(os.path.join(D, "fog", "forecast_%s.json" % where)))
    out = {}
    for i, t in enumerate(h["time"]):
        out[datetime.fromisoformat(t)] = {k: h[k][i] for k in h if k != "time"}
    return out


def sun(t):
    """Solar elevation (deg) at the buoy for local naive time t; and Haurwitz clear-sky GHI (W/m2)."""
    tu = t.replace(tzinfo=PT).astimezone(timezone.utc)
    n = tu.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1 + (tu.hour - 12) / 24)
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    tst = tu.hour * 60 + tu.minute + eqt + 4 * LON
    ha = math.radians(tst / 4 - 180)
    la = math.radians(LAT)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    el = math.degrees(math.asin(max(-1, min(1, cz))))
    ghi = 1098 * cz * math.exp(-0.057 / cz) if cz > 0.02 else 0.0
    return el, ghi


def es(T):
    """Saturation vapour pressure over water, hPa (Bolton 1980)."""
    return 6.112 * np.exp(17.67 * T / (T + 243.5))


def mixing_excess(Ta, Td, Tw):
    """Saunders' mixing criterion for steam fog: the most supersaturated point (hPa) on the mixing line between the
    air (Ta, e(Td)) and air saturated at the water's temperature (Tw, es(Tw)). Positive: mixing alone makes fog."""
    f = np.linspace(0.0, 1.0, 101)
    T = f * Tw + (1 - f) * Ta
    e = f * es(Tw) + (1 - f) * es(Td)
    return float(np.max(e - es(T)))


def table(cfg_path):
    CFG = json.load(open(cfg_path)); D = os.path.dirname(os.path.abspath(cfg_path))
    B = buoy_met(D); W = water(D); A = airports(D, [CFG["fog"]["airport"]] + CFG["fog"].get("cloud_airports", [])); F = forecast(D)
    rows = []
    for t in sorted(F):
        if not (2 <= t.hour <= 12):
            continue
        f = F[t]
        if None in (f["temperature_2m"], f["dew_point_2m"]):
            continue
        b = B.get(t); a = A.get(t, {})
        el, ghi = sun(t)
        eve = F.get((t - timedelta(days=1)).replace(hour=21) if t.hour < 21 else t.replace(hour=21))
        tw = water_at(W, t, lag_h=t.hour + 3)            # what was known by 21:00 the evening before
        rows.append(dict(t=t, f=f, b=b, a=a, el=el, ghi=ghi, tw=tw,
                         eve_dpd=(eve["temperature_2m"] - eve["dew_point_2m"]) if eve and eve["temperature_2m"] is not None else None))
    return rows


if __name__ == "__main__":
    R = table(sys.argv[1])
    for r in R:
        if r["t"] >= datetime(2026, 10, 19) and 4 <= r["t"].hour <= 11:
            b = r["b"] or {}
            ap = " ".join("%s v%.1f c%s%s" % (k, v["vis"], "-" if v["ceil"] == float("inf") else int(v["ceil"]), " FG" if v["fog"] else "") for k, v in r["a"].items())
            print(r["t"].strftime("%m-%d %H"), "Tw %.1f" % r["tw"], "| buoy T %.1f RH %.0f wind %.1f solar %.0f/%.0f" % (b.get("tair", np.nan), b.get("rh", np.nan), b.get("wind", np.nan), b.get("solar", np.nan), r["ghi"]),
                  "| fc T %.1f Td %.1f wind %.1f low %s vis %.0f" % (r["f"]["temperature_2m"], r["f"]["dew_point_2m"], r["f"]["wind_speed_10m"] / 3.6, r["f"]["cloud_cover_low"], r["f"]["visibility"]),
                  "| mix %.2f" % mixing_excess(r["f"]["temperature_2m"], r["f"]["dew_point_2m"], r["tw"]), "|", ap)
