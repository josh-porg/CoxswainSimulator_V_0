"""HOCR 2026: the club coxswains' lines compared on equal terms.

    .venv/Scripts/python.exe research/raceday/line_compare.py

Every crew in review_crews.CREWS that has its review JSON (race_review_2026.py + race_review_opt.py with
REVIEW_CREW=<crew>) takes part. Its line is its GPS track as a median offset per 25 m of course; its
setting is its boat, its pace (its own line reproduces its GPS time), its day-calibrated wind, flow and
level and the turning cost. Every line is timed in every setting, so within a setting only the line
differs. Also timed: each setting's best line (the optimiser's, through that crew's arches) and
bestline.row3r.com's shortest legal line. Boats are research-profile (shallow water on the wave drag only).
Writes line_compare.json (everything the comparison page needs) and line_compare.png.
"""
import json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C                                       # noqa: E402
from coxswain.river import charles                                           # noqa: E402
from coxswain.river.route import Route                                       # noqa: E402
import race_card as RC                                                       # noqa: E402
from review_crews import CREWS                                               # noqa: E402

FT = 0.3048
from review_crews import COX                                 # noqa: E402  names: data/local/raceday/crews.json (private)


def settings():
    out = {}
    for key, cfg in CREWS.items():
        p = os.path.join(HERE, cfg["json"])
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        cal = R["calibrated"]
        out[key] = dict(cox=COX.get(key, key), boat=cfg["crew"], event=cfg["event"], event_name=cfg["event_name"], bow=cfg["bow"],
                        place=cfg["place"], field=cfg["field"], official=cfg["official"], gps_s=R["race_s"],
                        level_ft=R["level_ft"], q=R["flow_cfs"] * 0.0283168466, flow_cfs=R["flow_cfs"],
                        wind=(cal["wind"]["speed"], cal["wind"]["direction"]), ref=cal["ref_s"],
                        line=(R["ours"]["stations"], R["ours"]["offsets"]), pins=R["pins"],
                        best=(cal["best"]["stations"], cal["best"]["offsets"]),
                        start_et=R.get("start_et"), speed=cal["ours"]["path_m"] / R["race_s"])
    return out


def main():
    SET = settings()
    keys = list(SET)
    card = json.load(open(os.path.join(RC.FIELDS, "card_four.json")))
    secs = [(s["start"], s["end"], s["from_m"], s["to_m"]) for s in card["sections"]]
    sl = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_vs_model.json")))["lines"][0]
    shortest = Route(np.asarray(sl["stations"], float), np.asarray(sl["offsets"], float), name="shortest")
    routes = {k: Route(np.asarray(SET[k]["line"][0], float), np.asarray(SET[k]["line"][1], float), name=k) for k in keys}
    out = dict(crews={k: {a: b for a, b in v.items() if a not in ("line", "best")} for k, v in SET.items()},
               sections=[dict(start=a, end=b, from_m=x, to_m=y) for a, b, x, y in secs], timed={})
    for setk, cfg in SET.items():
        model = C.CourseModel.build(level_offset=cfg["level_ft"] * FT)
        deep = C.CourseModel.build(level_offset=cfg["level_ft"] * FT + 2.0)
        spec = RC.CREWS[cfg["boat"]]
        boat = RC.build_boat(cfg["boat"], spec["rate"], profile="research")   # shallow water on the wave drag only
        k = C.RUDDER_K[spec["boat"]]

        def sections(m, route, qq, w, kk):
            m.flow = charles.ContinuityFlow(m.course, discharge=qq)
            m.route = route
            return C._sections(m, C._evaluate(m, boat, cfg["ref"], w), w, kk)

        tot = lambda s: sum(z.time_s for z in s)                               # noqa: E731
        lines = dict(routes)
        lines["best"] = Route(np.asarray(cfg["best"][0], float), np.asarray(cfg["best"][1], float), name="best")
        lines["shortest"] = shortest
        res = {}
        for ln, r in lines.items():
            a = sections(deep, r, 0.05, None, 0.0); b = sections(model, r, 0.05, None, 0.0)
            c = sections(model, r, cfg["q"], None, 0.0); d = sections(model, r, cfg["q"], cfg["wind"], 0.0)
            e = sections(model, r, cfg["q"], cfg["wind"], k)
            res[ln] = dict(total=tot(e), distance=tot(a), depth=tot(b) - tot(a), current=tot(c) - tot(b), wind=tot(d) - tot(c),
                           turning=sum(z.rudder_s for z in e), path_m=sum(z.path_m for z in e),
                           sections=[round(z.time_s, 2) for z in e])
        out["timed"][setk] = res
        print("setting %-7s (%s, %.2f m/s):" % (setk, cfg["boat"], cfg["speed"]))
        for ln, v in res.items():
            print("   %-9s %7.1f s (%+5.1f vs own)  path %4.0f m  depth %+5.1f wind %+5.1f" % (ln, v["total"], v["total"] - res[setk]["total"],
                                                                                         v["path_m"], v["depth"], v["wind"]))
    # where each line sits, stretch by stretch, and the drawn profiles: in the smooth frame
    # (coxswain/raceday/line_ocp.SmoothFrame: the centreline without its 150-200 m bank wiggles, which
    # every line's offset from the raw centreline inherits)
    from types import SimpleNamespace
    from coxswain.raceday import line_ocp as OCP
    course = C.CourseModel.build().course
    F = OCP.SmoothFrame(course)
    s_a = float(F.from_course([0.0], [0.0])[0][0]); s_b = float(F.from_course([course.length], [0.0])[0][0])
    s_grid = np.linspace(s_a, s_b, 400)
    st_dense = np.linspace(0.0, float(course.length), 2400)

    def in_frame(stations, offsets):
        fs, fn = F.from_course(st_dense, np.interp(st_dense, np.asarray(stations, float), np.asarray(offsets, float)))
        o = np.argsort(fs)
        return np.interp(s_grid, fs[o], fn[o])
    sec_f = [(float(F.from_course([x], [0.0])[0][0]), float(F.from_course([y], [0.0])[0][0])) for _, _, x, y in secs]
    prof, off = {}, {}
    for k in keys:
        for tag, (st, of) in (("", SET[k]["line"]), ("best:", SET[k]["best"])):
            v = in_frame(st, of)
            prof[tag + k] = np.round(v, 1).tolist()
            off[tag + k] = [round(float(np.median(v[(s_grid >= x) & (s_grid < y)])), 1) if ((s_grid >= x) & (s_grid < y)).any() else None for x, y in sec_f]
    prof["shortest"] = np.round(in_frame(sl["stations"], sl["offsets"]), 1).tolist()
    lo, hi = OCP.water_bounds(SimpleNamespace(course=course), F, s_grid, edge=0.0)
    out["profile"] = dict(s=np.round(s_grid - s_a, 1).tolist(), lines=prof, lo=np.round(lo, 1).tolist(), hi=np.round(hi, 1).tolist(),
                          half_width=np.round(np.maximum(-lo, hi), 1).tolist(), frame="smooth",
                          bridges=[dict(name=b["name"], s=round(float(F.from_course([b["at_m"]], [0.0])[0][0]) - s_a, 1)) for b in card["map"]["bridges"]])
    out["offsets_by_stretch"] = off
    out["pins"] = {k: SET[k]["pins"] for k in keys}
    json.dump(out, open(os.path.join(HERE, "line_compare.json"), "w"))
    figure(out, keys)


def figure(out, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = out["profile"]
    s, hw = np.asarray(P["s"]), np.asarray(P["half_width"])
    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    ax.fill_between(s, -hw, hw, color="#dfe7ee", lw=0)
    for i, k in enumerate(keys):
        ax.plot(s, P["lines"][k], lw=1.6, label=out["crews"][k]["cox"])
    for b in P["bridges"]:
        ax.axvline(b["s"], color="#999", lw=0.6, ls=":")
    ax.set_xlim(0, s[-1]); ax.set_xlabel("distance along the course centreline (m)"); ax.set_ylabel("offset (m), negative = Cambridge")
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=len(keys))
    fig.savefig(os.path.join(HERE, "line_compare.png"), dpi=150)
    print("written line_compare.json and line_compare.png (%d crews)" % len(keys))


if __name__ == "__main__":
    main()
