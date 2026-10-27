"""HOCR 2026, Women's Veteran 4+: the line we rowed against the optimum in the conditions on the day.

    .venv/Scripts/python.exe research/raceday/race_review_2026.py

1. The CBGPS per-stroke record (time, distance, speed, rate, position) put on the course: station and
   offset from the centreline per stroke, the start and finish crossings, the arch taken at each bridge.
2. The conditions during the race, as observed: KBOS, KBED, KOWD wind (Iowa Environmental Mesonet ASOS
   archive) through the race window, turned into the open 10 m wind over the course by
   ``stations.charles_reference`` (the same function the race card uses live); USGS 01104500 discharge and
   01104715 basin level over the window.
3. The optimum through the same arches in those conditions: ``plan.optimised_time`` with free points
   through the big turn and after Eliot (as bend_reopt.py), at least STAND_OFF from the bank (a stand-in
   for the buoy lines, which are not in the model), at our pace, warm-started from our own line.
4. Both lines timed in the same model, cause by cause (distance, depth, current, wind, turning), and per
   stretch against what the GPS says we actually did.
Writes race_review_2026.json (everything the report and the animation need).
"""
import csv, io, json, os, sys, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C, plan as P
from coxswain.river import charles, bridges, lines, stations
from coxswain.river.course import local_tangent_plane
from coxswain.river.route import Route
import race_card as RC

from review_crews import CFG, STARBOARD, shore_name                                               # REVIEW_CREW picks the crew (default ours)
COXBOX = CFG["coxbox"]
EDT = timezone(timedelta(hours=-4))
START_ET = CFG["start_et"]                                       # the CBGPS session start, US Eastern
FT = 0.3048
STAND_OFF = 15.0
FREE = (1100.0, 1610.0, 2030.0, 2500.0, 3150.0, 3420.0, 3640.0, 3830.0, 4180.0, 4480.0)
UA = {"User-Agent": "CoxswainSimulator race review (research)"}


def strokes():
    rows = list(csv.reader(open(COXBOX, encoding="utf-8", errors="ignore")))
    i = next(k for k, r in enumerate(rows) if r and r[0].startswith("Per-Stroke Data"))
    out = []
    for r in rows[i + 3:]:
        if len(r) < 24 or not r[0].strip():
            continue
        try:
            h, m, s = r[3].split(":")
            out.append((int(h) * 3600 + int(m) * 60 + float(s), float(r[1]), float(r[5]), float(r[8]), float(r[22]), float(r[23])))
        except ValueError:
            continue
    return np.array(out)             # t, distance, speed, rate, lat, lon


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode()


def asos(station, t0, t1):
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=%s&data=sknt&data=drct&data=gust"
           "&year1=2026&month1=10&day1=17&year2=2026&month2=10&day2=18&tz=America%%2FNew_York&format=onlycomma"
           "&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4" % station)
    cache = os.path.join(HERE, "results", "asos_%s_20261017.csv" % station)
    if not os.path.exists(cache):
        open(cache, "w").write(get(url))
    obs = []
    for r in csv.DictReader(io.StringIO(open(cache).read())):
        try:
            t = datetime.fromisoformat(r["valid"]).replace(tzinfo=EDT)
            obs.append((t, float(r["sknt"]) * 0.514444, float(r["drct"]) if r["drct"] not in ("M", "") else float("nan"),
                        float(r["gust"]) * 0.514444 if r["gust"] not in ("M", "") else float("nan")))
        except (ValueError, KeyError):
            pass
    return [o for o in obs if t0 - timedelta(minutes=40) <= o[0] <= t1 + timedelta(minutes=20)]


def usgs(site, param, t0, t1):
    url = ("https://waterservices.usgs.gov/nwis/iv/?sites=%s&parameterCd=%s&format=json&startDT=2026-10-17T08:00-04:00"
           "&endDT=2026-10-17T14:00-04:00" % (site, param))
    v = json.loads(get(url))["value"]["timeSeries"][0]["values"][0]["value"]
    pts = [(datetime.fromisoformat(x["dateTime"]), float(x["value"])) for x in v]
    near = [y for t, y in pts if t0 - timedelta(minutes=30) <= t <= t1 + timedelta(minutes=30)]
    return float(np.mean(near)), pts


def main():
    model = C.CourseModel.build(starboard=STARBOARD)
    course = model.course
    L = float(course.length)
    S = strokes()
    x, y = local_tangent_plane(S[:, 4], S[:, 5], charles.CHARLES_ORIGIN)
    st = np.array([course.nearest_station(a, b) for a, b in zip(x, y)])
    off = np.array([course.offset_of(a, b, s) for a, b, s in zip(x, y, st)])
    t = S[:, 0]
    # start and finish crossings (station increasing through 0 and L)
    i0 = int(np.argmax(st > 5.0)); i1 = int(np.argmax(st >= L - 1.0)) if np.any(st >= L - 1.0) else len(st) - 1
    t_start = float(np.interp(0.0, st[max(i0 - 3, 0):i0 + 3], t[max(i0 - 3, 0):i0 + 3])) if i0 > 0 else float(t[0])
    t_finish = float(np.interp(L, st[i1 - 3:i1 + 2], t[i1 - 3:i1 + 2]))
    race_s = t_finish - t_start
    race = (t >= t_start) & (t <= t_finish + 1)
    win0, win1 = START_ET + timedelta(seconds=t_start), START_ET + timedelta(seconds=t_finish)
    print("race on the GPS: %.1f s (%d:%04.1f), %s to %s ET" % (race_s, race_s // 60, race_s % 60, win0.strftime("%H:%M:%S"), win1.strftime("%H:%M:%S")))

    # our line as a route: median offset per 25 m station bin, lightly smoothed
    bins = np.arange(0.0, L + 25.0, 25.0)
    so, oo = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = race & (st >= a) & (st < b)
        if m.sum():
            so.append(float(np.mean(st[m]))); oo.append(float(np.median(off[m])))
    so, oo = np.array(so), np.array(oo)
    oo = np.convolve(np.pad(oo, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    port, stbd = course.limits_at(so)                         # each side's own limit (shore corridor: asymmetric)
    oo = np.clip(oo, -(stbd - 2.0), port - 2.0)
    ours = Route(np.concatenate([[0.0], so, [L]]), np.concatenate([[oo[0]], oo, [oo[-1]]]), name="our line")

    # the arches we took: nearest arch centre to our offset at each bridge
    arches = {}
    for gate, metres in model.gates:
        cands = bridges.candidate_arches(gate, model.raster)
        if len(cands) < 2:
            continue
        ours_off = float(ours.offset_at(np.array([metres]))[0])
        centres = [(a.label, lines.to_offset(course, gate.point_at(a.centre))[1]) for a in cands]
        lab, c_off = min(centres, key=lambda z: abs(z[1] - ours_off))
        arches[gate.name] = dict(arch=lab, our_offset=ours_off, arch_offset=c_off, arches={k: v for k, v in centres})
    pins = {k: v["arch"] for k, v in arches.items()}
    print("arches taken:", pins)

    # the conditions in the race window
    obs = []
    for sid in ("BOS", "BED", "OWD"):
        for o in asos(sid, win0, win1):
            obs.append(dict(station="K" + sid, t=o[0].isoformat(), speed=o[1], direction=o[2], gust=o[3]))
    in_win = [o for o in obs if win0 - timedelta(minutes=35) <= datetime.fromisoformat(o["t"]) <= win1 + timedelta(minutes=10)]
    by_station = {}
    for o in in_win:
        by_station.setdefault(o["station"], []).append(o)
    readings = []
    for sid, os_ in by_station.items():
        sp = float(np.mean([o["speed"] for o in os_]))
        u = np.mean([o["speed"] * np.sin(np.radians(o["direction"])) for o in os_ if np.isfinite(o["direction"])])
        v = np.mean([o["speed"] * np.cos(np.radians(o["direction"])) for o in os_ if np.isfinite(o["direction"])])
        gusts = [o["gust"] for o in os_ if np.isfinite(o["gust"])]
        readings.append(stations.Observation(sid, win0.isoformat(), sp, float(np.degrees(np.arctan2(u, v)) % 360),
                                             max(gusts) if gusts else float("nan")))
    w_speed, w_dir, w_note = stations.charles_reference(readings)
    wind = (float(w_speed), float(w_dir))
    q_cfs, q_series = usgs("01104500", "00060", win0, win1)
    lvl_ft, _ = usgs("01104715", "00065", win0, win1)
    level_off = lvl_ft - 107.32
    print("wind %.1f m/s from %03.0f (%s); flow %.0f cfs; basin %+.2f ft" % (wind[0], wind[1], w_note, q_cfs, level_off))

    model = C.CourseModel.build(starboard=STARBOARD, level_offset=level_off * FT)
    deep = C.CourseModel.build(starboard=STARBOARD, level_offset=level_off * FT + 2.0)
    q = q_cfs * 0.0283168466
    spec = RC.CREWS[CFG["crew"]]
    boat = RC.build_boat(CFG["crew"], spec["rate"], profile="research")   # depth: Sretenskii, wave term only (PHYSICS_PROGRAMME, 2026-10-21)
    k = C.RUDDER_K[spec["boat"]]
    ref = race_s

    def timed(m, route, qq, w, kk):
        m.flow = charles.ContinuityFlow(m.course, discharge=qq)
        m.route = route
        sec = C._sections(m, C._evaluate(m, boat, ref, w), w, kk)
        return sec

    def breakdown(route):
        a = timed(deep, route, 0.05, None, 0.0); b = timed(model, route, 0.05, None, 0.0)
        c = timed(model, route, q, None, 0.0); d = timed(model, route, q, wind, 0.0); e = timed(model, route, q, wind, k)
        tot = lambda s: sum(z.time_s for z in s)                                  # noqa: E731
        return dict(total=tot(e), distance=tot(a), depth=tot(b) - tot(a), current=tot(c) - tot(b), wind=tot(d) - tot(c),
                    turning=sum(z.rudder_s for z in e), path_m=sum(z.path_m for z in e),
                    sections=[dict(start=z.start, end=z.end, from_m=z.from_m if hasattr(z, "from_m") else None,
                                   time_s=z.time_s, path_m=z.path_m, rudder_s=z.rudder_s) for z in e])

    print("optimising the line through our arches in the day's conditions (a few minutes)...")
    model.flow = charles.ContinuityFlow(model.course, discharge=q)
    opt = P.optimised_time(model, boat, ref, pins, k, wind=wind, free=FREE, margin=STAND_OFF, maxiter=900, start_route=ours)
    best = opt["route"]
    b_ours, b_best = breakdown(ours), breakdown(best)
    print("model time, our line %.1f s, optimum %.1f s (%+.1f s); path %.0f vs %.0f m"
          % (b_ours["total"], b_best["total"], b_ours["total"] - b_best["total"], b_ours["path_m"], b_best["path_m"]))

    # what the GPS says per stretch, against the model's time for our line (scaled to our total)
    card = json.load(open(os.path.join(RC.FIELDS, "card_%s.json" % CFG["crew"])))
    secs = card["sections"]
    gps = []
    for s in secs:
        ta = float(np.interp(s["from_m"], st[race], t[race])); tb = float(np.interp(s["to_m"], st[race], t[race]))
        m = race & (st >= s["from_m"]) & (st < s["to_m"])
        gps.append(dict(start=s["start"], end=s["end"], from_m=s["from_m"], to_m=s["to_m"], time_s=tb - ta,
                        rate=float(np.mean(S[m, 3])) if m.sum() else None))

    # per-stroke track for the animation (course frame and map frame)
    x0, y0 = card["map"]["origin"]
    bx, by_ = course.offset_position(np.linspace(0, L, 600), best.offset_at(np.linspace(0, L, 600))).T
    out = dict(
        race_s=race_s, start_et=win0.isoformat(), finish_et=win1.isoformat(),
        wind=dict(speed=wind[0], direction=wind[1], note=w_note, observations=in_win),
        flow_cfs=q_cfs, level_ft=level_off, arches=arches, pins=pins,
        ours=dict(b_ours, stations=list(map(float, ours.stations)), offsets=list(map(float, ours.offsets))),
        best=dict(b_best, stations=list(map(float, np.atleast_1d(best.stations))), offsets=list(map(float, np.atleast_1d(best.offsets))),
                  xy=np.round(np.column_stack([bx - x0, by_ - y0]), 1).tolist()),
        gps_sections=gps,
        track=dict(t=np.round(t[race] - t_start, 2).tolist(), x=np.round(x[race] - x0, 1).tolist(), y=np.round(y[race] - y0, 1).tolist(),
                   station=np.round(st[race], 1).tolist(), offset=np.round(off[race], 1).tolist(),
                   speed=np.round(S[race, 2], 2).tolist(), rate=np.round(S[race, 3], 1).tolist()),
        map=card["map"], card_targets=[dict(name=g["name"], today=g["today_central"]) for g in card["targets"]],
        stand_off_m=STAND_OFF)
    json.dump(out, open(os.path.join(HERE, CFG["json"]), "w"))
    print("written", CFG["json"])


if __name__ == "__main__":
    main()
