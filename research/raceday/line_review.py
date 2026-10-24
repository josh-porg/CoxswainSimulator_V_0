"""HOCR 2026: any crew's line on the Charles, against our four's line and the optimum, in that crew's own conditions.

    .venv/Scripts/python.exe research/raceday/line_review.py bridge

Generalises race_review_2026.py + race_review_opt.py (our four) to another CBGPS track:

1. The per-stroke record on the course: station and offset per stroke, the arch taken at each bridge with a
   choice of arch, and the time over the model's course (start crossing to the last stroke, extended to the
   model's finish line at the closing speed; both crews' CoxBoxes stop just short of it).
2. The conditions in that crew's race window: KBOS/KBED/KOWD wind (IEM ASOS) through
   ``stations.charles_reference``, USGS discharge and basin level. The wind speed is scaled so the four's
   whole-line factor equals the measured day factor for the crew's own event (field median 2026 / 2025,
   x BASE_2025 = 1.02, results/hocr_event_summaries.csv), as race_review_opt.py did for ours.
3. The model's pace set so the crew's own line takes its GPS time in those conditions.
4. In those same conditions and pace: the crew's line, our line, the optimum through the crew's arches and
   the optimum through ours (Powell, a free point every 150 m, 4 m inside the banks, from the crew's line).
   Only the line differs between them, so the differences are what the line is worth.
5. Stretch by stretch: what the GPS says each crew did, against what the model expects of its line.
Writes line_review_<crew>.json and line_review_<crew>.png beside this file.
"""
import json, os, sys
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.optimize import minimize, brentq

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import race_review_2026 as RR                                                  # noqa: E402  (strokes, asos, usgs)
from coxswain.raceday import card as C, plan as P, gusts as GU               # noqa: E402
from coxswain.river import charles, bridges, lines, stations                 # noqa: E402
from coxswain.river.course import local_tangent_plane                        # noqa: E402
from coxswain.river.route import Route                                       # noqa: E402
import race_card as RC                                                       # noqa: E402

EDT = timezone(timedelta(hours=-4))
FT = 0.3048
MARGIN, STEP = 4.0, 150.0
BASE_2025 = 1.02
CB = os.path.join(ROOT, "data", "local", "coxbox", "Sharing file  CoxBox %s.csv")
from review_crews import LINE_REVIEW as CREWS, alias           # noqa: E402  private: data/local/raceday/crews.json


def track(crew, model):
    RR.COXBOX = CREWS[crew]["csv"]
    S = RR.strokes()
    course = model.course
    L = float(course.length)
    x, y = local_tangent_plane(S[:, 4], S[:, 5], charles.CHARLES_ORIGIN)
    st = np.array([course.nearest_station(a, b) for a, b in zip(x, y)])
    off = np.array([course.offset_of(a, b, s) for a, b, s in zip(x, y, st)])
    t = S[:, 0]
    i0 = int(np.argmax(st > 5.0))
    t_start = float(np.interp(0.0, st[max(i0 - 3, 0):i0 + 3], t[max(i0 - 3, 0):i0 + 3])) if i0 > 0 else float(t[0])
    v_end = float(np.mean(S[-5:, 2]))
    t_end = float(t[-1] + max(L - st[-1], 0.0) / v_end)
    race = t >= t_start
    bins = np.arange(0.0, L + 25.0, 25.0)
    so, oo = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = race & (st >= a) & (st < b)
        if m.sum():
            so.append(float(np.mean(st[m]))); oo.append(float(np.median(off[m])))
    so, oo = np.array(so), np.array(oo)
    oo = np.convolve(np.pad(oo, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    hw = course.half_width_at(so)
    oo = np.clip(oo, -(hw - 2.0), hw - 2.0)
    route = Route(np.concatenate([[0.0], so, [L]]), np.concatenate([[oo[0]], oo, [oo[-1]]]), name=crew)
    arches = {}
    for gate, metres in model.gates:
        cands = bridges.candidate_arches(gate, model.raster)
        if len(cands) < 2:
            continue
        o = float(route.offset_at(np.array([metres]))[0])
        centres = [(a.label, lines.to_offset(course, gate.point_at(a.centre))[1]) for a in cands]
        lab, c_off = min(centres, key=lambda z: abs(z[1] - o))
        arches[gate.name] = dict(arch=lab, offset=o, arch_offset=c_off)
    return dict(S=S, x=x, y=y, st=st, off=off, t=t, t_start=t_start, race_s=t_end - t_start, end_station=float(st[-1]),
                route=route, arches=arches, pins={k: v["arch"] for k, v in arches.items()})


def conditions(crew, T):
    c = CREWS[crew]
    win0 = c["start"] + timedelta(seconds=float(T["t"][0])); win1 = c["start"] + timedelta(seconds=float(T["t"][-1]))
    by = {}
    for sid in ("BOS", "BED", "OWD"):
        for o in RR.asos(sid, win0, win1):
            if win0 - timedelta(minutes=35) <= o[0] <= win1 + timedelta(minutes=10):
                by.setdefault("K" + sid, []).append(o)
    readings = []
    for sid, os_ in by.items():
        sp = float(np.mean([o[1] for o in os_]))
        u = np.mean([o[1] * np.sin(np.radians(o[2])) for o in os_ if np.isfinite(o[2])])
        v = np.mean([o[1] * np.cos(np.radians(o[2])) for o in os_ if np.isfinite(o[2])])
        g = [o[3] for o in os_ if np.isfinite(o[3])]
        readings.append(stations.Observation(sid, win0.isoformat(), sp, float(np.degrees(np.arctan2(u, v)) % 360), max(g) if g else float("nan")))
    w_speed, w_dir, note = stations.charles_reference(readings)
    q_cfs, _ = RR.usgs("01104500", "00060", win0, win1)
    lvl, _ = RR.usgs("01104715", "00065", win0, win1)
    obs = {sid: dict(speed=float(np.mean([o[1] for o in os_])), gust=float(np.nanmax([o[3] for o in os_])) if any(np.isfinite(o[3]) for o in os_) else None,
                     n=len(os_)) for sid, os_ in by.items()}
    return dict(window=[win0.isoformat(), win1.isoformat()], wind_obs=(float(w_speed), float(w_dir)), note=note, stations=obs,
                q=q_cfs * 0.0283168466, q_cfs=q_cfs, level_ft=lvl - 107.32)


def main(crew=None):
    crew = crew or alias("msm_four")
    model0 = C.CourseModel.build()
    T, O = track(crew, model0), track("ours", model0)
    print("%s: model-course time %.1f s (official %.1f), last stroke at %.0f m of %.0f; arches %s"
          % (crew, T["race_s"], CREWS[crew]["official"], T["end_station"], model0.course.length, T["pins"]))
    print("ours: model-course time %.1f s (official %.1f), last stroke at %.0f m; arches %s"
          % (O["race_s"], CREWS["ours"]["official"], O["end_station"], O["pins"]))
    cond = conditions(crew, T)
    model = C.CourseModel.build(level_offset=cond["level_ft"] * FT)
    deep = C.CourseModel.build(level_offset=cond["level_ft"] * FT + 2.0)
    course, q = model.course, cond["q"]
    L = float(course.length)
    spec = RC.CREWS["four"]
    boat = RC.build_boat("four", spec["rate"], profile="research")   # depth: Sretenskii, wave term only (PHYSICS_PROGRAMME, 2026-10-21)
    k = C.RUDDER_K[spec["boat"]]
    Tw = json.load(open(os.path.join(RC.FIELDS, "wind_table_four.json")))
    bundle = json.load(open(os.path.join(RC.FIELDS, "site_bundle.json")))["crews"]["four"]
    river = 1.0 + float(np.interp(q, bundle["flow"]["discharge_m3s"], bundle["flow"]["effect_fraction"]))
    day = CREWS[crew]["day"]
    d = cond["wind_obs"][1]
    f = lambda v: GU.table_at(Tw, v, d)["factor"] * river - day                  # noqa: E731
    v_cal = brentq(f, 0.0, 15.0) if f(0.0) < 0 < f(15.0) else 0.0
    wind = (float(v_cal), float(d))
    print("window %s to %s ET; observed reference wind %.1f m/s from %03.0f; flow %.0f cfs; basin %+.2f ft; day factor %.3f -> wind %.2f m/s"
          % (cond["window"][0][11:19], cond["window"][1][11:19], cond["wind_obs"][0], d, cond["q_cfs"], cond["level_ft"], day, v_cal))

    def sections(m, route, qq, w, kk, ref):
        m.flow = charles.ContinuityFlow(m.course, discharge=qq)
        m.route = route
        return C._sections(m, C._evaluate(m, boat, ref, w), w, kk)

    total = lambda s: sum(z.time_s for z in s)                                  # noqa: E731
    ref = T["race_s"] / day
    for _ in range(3):
        ref *= T["race_s"] / total(sections(model, T["route"], q, wind, k, ref))
    print("model pace: still-air %.1f s for %s's line" % (ref, crew))

    def optimum(pins, warm):
        # plan.optimise_line (2026-10-22): bridge points free inside their arches, ends free, warm starts
        # from the crew's line and bestline's shortest legal line (TRACKING)
        model.flow = charles.ContinuityFlow(course, discharge=q)
        BLs = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_vs_model.json")))
        sh = next(x for x in BLs["lines"] if x["name"] == "their shortest line")
        shortest = Route(np.asarray(sh["stations"], float), np.asarray(sh["offsets"], float), name="shortest")
        return P.optimise_line(model, boat, ref, pins, k, wind, starts=(warm, shortest), step=STEP, margin=MARGIN)["route"]

    def breakdown(route):
        a = sections(deep, route, 0.05, None, 0.0, ref); b = sections(model, route, 0.05, None, 0.0, ref)
        c = sections(model, route, q, None, 0.0, ref); dd = sections(model, route, q, wind, 0.0, ref)
        e = sections(model, route, q, wind, k, ref)
        return dict(total=total(e), distance=total(a), depth=total(b) - total(a), current=total(c) - total(b),
                    wind=total(dd) - total(c), turning=sum(z.rudder_s for z in e), path_m=sum(z.path_m for z in e),
                    sections=[dict(start=z.start, end=z.end, time_s=z.time_s, path_m=z.path_m, rudder_s=z.rudder_s,
                                   wind=total([dd[i]]) - total([c[i]]), depth=total([b[i]]) - total([a[i]]),
                                   current=total([c[i]]) - total([b[i]])) for i, z in enumerate(e)])

    print("optimising through %s's arches..." % crew)
    best_own = optimum(T["pins"], T["route"])
    same = T["pins"] == O["pins"]
    if not same:
        print("optimising through our arches (in %s's conditions)..." % crew)
    best_ours = best_own if same else optimum(O["pins"], O["route"])
    B = {"line": breakdown(T["route"]), "our_line": breakdown(O["route"]), "optimum": breakdown(best_own),
         "optimum_our_arches": breakdown(best_ours)}
    for key, v in B.items():
        print("  %-20s %7.1f s  path %5.0f m  distance %+7.1f depth %+5.1f current %+5.1f wind %+5.1f turning %+4.1f"
              % (key, v["total"], v["path_m"], v["distance"], v["depth"], v["current"], v["wind"], v["turning"]))

    # her line in OUR conditions and pace (race_review_2026.json, calibrated)
    R = json.load(open(os.path.join(HERE, "race_review_2026.json")))
    Rc = R["calibrated"]
    m_ours = C.CourseModel.build(level_offset=R["level_ft"] * FT)
    q_ours = R["flow_cfs"] * 0.0283168466
    w_ours = (Rc["wind"]["speed"], Rc["wind"]["direction"])
    in_ours = {nm: total(sections(m_ours, r, q_ours, w_ours, k, Rc["ref_s"])) for nm, r in
               (("line", T["route"]), ("our_line", O["route"]))}
    print("in our conditions and pace: %s's line %.1f s, our line %.1f s" % (crew, in_ours["line"], in_ours["our_line"]))

    # stretch by stretch: GPS time against the model's time for the crew's own line
    card = json.load(open(os.path.join(RC.FIELDS, "card_four.json")))
    stretches = []
    for i, s in enumerate(card["sections"]):
        row = dict(start=s["start"], end=s["end"], from_m=s["from_m"], to_m=s["to_m"])
        for nm, X, b in ((crew, T, B["line"]), ("ours", O, None)):
            r = X["t"] >= X["t_start"]
            st_, t_ = X["st"][r], X["t"][r]
            if s["to_m"] <= X["end_station"] + 1:
                row[nm + "_gps_s"] = float(np.interp(s["to_m"], st_, t_) - np.interp(s["from_m"], st_, t_))
                m = r & (X["st"] >= s["from_m"]) & (X["st"] < s["to_m"])
                row[nm + "_rate"] = float(np.mean(X["S"][m, 3])) if m.sum() else None
                row[nm + "_offset"] = float(np.median(X["off"][m])) if m.sum() else None
        row["line_model_s"] = B["line"]["sections"][i]["time_s"]
        row["our_line_model_s"] = B["our_line"]["sections"][i]["time_s"]
        row["optimum_model_s"] = B["optimum"]["sections"][i]["time_s"]
        row["line_wind_s"] = B["line"]["sections"][i]["wind"]; row["line_depth_s"] = B["line"]["sections"][i]["depth"]
        stretches.append(row)

    s = np.linspace(0, L, 600)
    x0m, y0m = card["map"]["origin"]
    xy = lambda r: np.round(course.offset_position(s, r.offset_at(s)) - [x0m, y0m], 1).tolist()   # noqa: E731
    out = dict(crew=crew, label=CREWS[crew]["label"], official=CREWS[crew]["official"], splits=CREWS[crew].get("splits"),
               model_course_s=T["race_s"], ours_model_course_s=O["race_s"], end_station=T["end_station"],
               arches=T["arches"], pins=T["pins"], our_pins=O["pins"], conditions=cond,
               calibrated=dict(day_factor=day, wind=dict(speed=wind[0], direction=wind[1]), ref_s=ref),
               breakdown=B, in_our_conditions=in_ours, stretches=stretches,
               routes={nm: dict(stations=list(map(float, np.atleast_1d(r.stations))), offsets=list(map(float, np.atleast_1d(r.offsets))), xy=xy(r))
                       for nm, r in (("line", T["route"]), ("our_line", O["route"]), ("optimum", best_own), ("optimum_our_arches", best_ours))},
               track=dict(t=np.round(T["t"] - T["t_start"], 2).tolist(), station=np.round(T["st"], 1).tolist(),
                          offset=np.round(T["off"], 1).tolist(), speed=np.round(T["S"][:, 2], 2).tolist(), rate=np.round(T["S"][:, 3], 1).tolist()))
    json.dump(out, open(os.path.join(HERE, "line_review_%s.json" % crew), "w"))
    figure(out, course, model.gates, card)
    print("written line_review_%s.json and .png" % crew)


def figure(R, course, gates, card):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = float(course.length)
    s = np.linspace(0, L, 800)
    hw = course.half_width_at(s)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw=dict(height_ratios=[1.25, 1]), constrained_layout=True)
    a = ax[0]
    for sign in (-1, 1):
        b = course.offset_position(s, sign * hw)
        a.plot(b[:, 0], b[:, 1], color="#7a8a96", lw=0.8)
    sty = {"line": ("#c0392b", 1.8, "-", "Bridge (GPS line)"), "our_line": ("#1f4e79", 1.4, "-", "our four (GPS line)"),
           "optimum": ("#2e8b57", 1.2, "--", "optimum through her arches")}
    for nm, (c, lw, ls, lab) in sty.items():
        r = R["routes"][nm]
        p = course.offset_position(s, np.interp(s, r["stations"], r["offsets"]))
        a.plot(p[:, 0], p[:, 1], color=c, lw=lw, ls=ls, label=lab)
    for g, m in gates:
        p = course.offset_position(np.array([m]), np.array([0.0]))[0]
        a.annotate(g.name, p, fontsize=7, color="#444", xytext=(4, 4), textcoords="offset points")
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.legend(loc="lower left", fontsize=8, frameon=False)
    a.set_title("Head of the Charles 2026: %s and %s" % (CREWS[R["crew"]]["label"], CREWS["ours"]["label"]), fontsize=10)
    a = ax[1]
    a.fill_between(s, -hw, hw, color="#dfe7ee", lw=0, label="river (bank to bank)")
    for nm, (c, lw, ls, lab) in sty.items():
        r = R["routes"][nm]
        a.plot(s, np.interp(s, r["stations"], r["offsets"]), color=c, lw=lw, ls=ls, label=lab)
    for g, m in gates:
        a.axvline(m, color="#999", lw=0.6, ls=":")
        a.text(m, a.get_ylim()[1] if False else hw.max() * 0.95, g.name.replace(" Footbridge", "").replace(" Bridge", ""), rotation=90, fontsize=7, va="top", ha="right", color="#555")
    a.set_xlabel("distance along the course centreline (m)"); a.set_ylabel("offset from centreline (m)\n(negative = Cambridge)")
    a.set_xlim(0, L); a.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    fig.savefig(os.path.join(HERE, "line_review_%s.png" % R["crew"]), dpi=150)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
