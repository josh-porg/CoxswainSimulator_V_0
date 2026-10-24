"""Does same-day information make the predictions better? A backtest on HOCR 2024 -> 2025 and 2025 -> 2026 (Friday).

Each event's field median (and its winner and first quartile) is predicted from the same event last year:

  persist    last year's time as it is
  model      last year's time x (this year's wind and river factor / last year's), from the race card's
             wind tables (four's for singles, doubles and fours; eight's for eights) at the KBOS hourly
             observation at the event's start, and the flow tables at the day's USGS mean discharge
  day        last year's time x the median ratio (this year / last year) of the events that had finished
             before this one started (results up 30 min after a start); the day before's ratio before then
  model+day  the model, corrected by the median of (actual / model) ratios of the events already finished

Data: hocr_event_summaries.csv beside this file (read from RegattaCentral in the browser, 16 Oct 2026);
KBOS observations from the Iowa Environmental Mesonet ASOS archive; USGS 01104500 daily discharge.
"""
import csv, io, json, os, sys, urllib.request
from datetime import datetime, timedelta
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "raceday", "results"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from coxswain.raceday import gusts as GU                # table_at: the race card's wind-table lookup

RD = os.path.join(ROOT, "data", "local", "raceday")
LAG_MIN = 30          # results of an event are taken as known this long after its start
UA = {"User-Agent": "CoxswainSimulator race card (research)"}


def rows():
    out = []
    for r in csv.DictReader(l for l in open(os.path.join(HERE, "hocr_event_summaries.csv")) if not l.startswith("#")):
        et = datetime.fromisoformat(r["start_utc"]) - timedelta(hours=7)          # Eastern wall clock
        out.append(dict(year=int(r["year"]), event=int(r["event"]), code=r["code"], et=et, day=et.date(),
                        n=int(r["n"]), winner=float(r["winner"]), q25=float(r["q25"]), median=float(r["median"])))
    return out


def kbos(day):
    """KBOS hourly-ish observations for a day: [(ET datetime, m/s, deg from)]."""
    cache = os.path.join(HERE, "kbos_%s.csv" % day)
    if not os.path.exists(cache):
        url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=BOS&data=sknt&data=drct&data=gust"
               "&year1={y}&month1={m}&day1={d}&year2={y}&month2={m}&day2={d2}&tz=America%2FNew_York&format=onlycomma"
               "&latlon=no&missing=M&trace=T&direct=no&report_type=3").format(y=day.year, m=day.month, d=day.day, d2=day.day + 1)
        open(cache, "w").write(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode())
    obs = []
    for r in csv.DictReader(io.StringIO(open(cache).read())):
        try:
            obs.append((datetime.fromisoformat(r["valid"]), float(r["sknt"]) * 0.514444, float(r["drct"])))
        except (ValueError, KeyError):
            pass
    return obs


def wind_at(obs, t):
    """Mean of the observations within 30 min of t (speed; direction as a vector mean)."""
    near = [(s, d) for v, s, d in obs if abs((v - t).total_seconds()) <= 1800]
    if not near:
        return 0.0, 0.0
    s = np.mean([x for x, _ in near])
    u = np.mean([x * np.sin(np.radians(d)) for x, d in near]); v = np.mean([x * np.cos(np.radians(d)) for x, d in near])
    return float(s), float(np.degrees(np.arctan2(u, v)) % 360)


def discharge(day):
    cache = os.path.join(HERE, "usgs_%s.json" % day)
    if not os.path.exists(cache):
        url = ("https://waterservices.usgs.gov/nwis/dv/?sites=01104500&parameterCd=00060&format=json&startDT=%s&endDT=%s" % (day, day))
        open(cache, "w").write(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode())
    ts = json.load(open(cache))["value"]["timeSeries"]
    v = ts[0]["values"][0]["value"] if ts else []
    if v:
        return float(v[0]["value"]) * 0.0283168466
    # today: no daily mean yet, so the mean of the instantaneous readings so far
    url = ("https://waterservices.usgs.gov/nwis/iv/?sites=01104500&parameterCd=00060&format=json&startDT=%s&endDT=%s" % (day, day))
    iv = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))["value"]["timeSeries"][0]["values"][0]["value"]
    os.remove(cache)
    return float(np.mean([float(x["value"]) for x in iv])) * 0.0283168466 if iv else None


def main():
    data = rows()
    bundle = json.load(open(os.path.join(RD, "site_bundle.json")))["crews"]
    tables = {c: bundle[c]["wind"] for c in ("four", "eight")}
    flows = {c: bundle[c]["flow"] for c in ("four", "eight")}
    obs = {d: kbos(d) for d in sorted({r["day"] for r in data})}
    q = {d: discharge(d) for d in obs}

    def model_factor(r):
        c = "eight" if "8+" in r["code"] else "four"
        s, d = wind_at(obs[r["day"]], r["et"])
        w = GU.table_at(tables[c], s, d)["factor"]
        fl = flows[c]
        f = 1.0 + np.interp(q[r["day"]], fl["discharge_m3s"], fl["effect_fraction"])
        r["wind"] = (round(s, 1), round(d)); r["q"] = q[r["day"]]
        return w * f

    for r in data:
        r["mf"] = model_factor(r)
    by = {(r["year"], r["event"]): r for r in data}
    results = []
    for r in data:
        prev = by.get((r["year"] - 1, r["event"]))
        if not prev or prev["code"] != r["code"]:
            continue
        for stat in ("median", "winner", "q25"):
            last = prev[stat]
            model = last * r["mf"] / prev["mf"]
            done = [(x, by[(x["year"] - 1, x["event"])]) for x in data
                    if x["year"] == r["year"] and x["day"] == r["day"] and (x["year"] - 1, x["event"]) in by
                    and x["et"] + timedelta(minutes=LAG_MIN) <= r["et"]]
            before = [(x, by[(x["year"] - 1, x["event"])]) for x in data
                      if x["year"] == r["year"] and x["day"] < r["day"] and (x["year"] - 1, x["event"]) in by]
            src = done or before
            ratio = np.median([x[stat] / p[stat] for x, p in src]) if src else 1.0
            resid = np.median([x[stat] / (p[stat] * x["mf"] / p["mf"]) for x, p in src]) if src else 1.0
            preds = dict(persist=last, model=model, day=last * ratio, model_day=model * resid)
            results.append(dict(year=r["year"], event=r["event"], code=r["code"], stat=stat, actual=r[stat],
                                same_day=len(done), friday=r["day"].weekday() == 4,
                                err={k: 100.0 * (v / r[stat] - 1.0) for k, v in preds.items()}))
    keys = ("persist", "model", "day", "model_day")
    print("Mean absolute error, %% of the time (n events); same-day info = at least one earlier event finished that day\n")
    for stat in ("median", "q25", "winner"):
        for label, sel in (("all", lambda x: True), ("with same-day info", lambda x: x["same_day"] > 0),
                           ("Saturday, with same-day info", lambda x: x["same_day"] > 0 and not x["friday"])):
            sub = [x for x in results if x["stat"] == stat and sel(x)]
            if sub:
                print("%-7s %-30s n=%2d  " % (stat, label, len(sub)) + "  ".join(
                    "%s %.2f" % (k, np.mean([abs(x["err"][k]) for x in sub])) for k in keys))
        print()
    print("Our events (median):")
    for x in results:
        if x["stat"] == "median" and x["event"] in (17, 22):
            print("  %d %-9s actual %.1f  errors %%: %s" % (x["year"], x["code"], x["actual"],
                  "  ".join("%s %+.2f" % (k, x["err"][k]) for k in keys)))
    print("\nWind at the starts (KBOS, m/s, from) and the day's discharge (m3/s):")
    for d in sorted(obs):
        evs = [r for r in data if r["day"] == d]
        print("  %s  q=%.1f  " % (d, q[d] or float("nan")) + " ".join("%d:%s" % (r["event"], r.get("wind")) for r in evs[:3]), "...")
    json.dump(results, open(os.path.join(HERE, "backtest_dayfactor.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
