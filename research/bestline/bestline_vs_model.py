"""bestline.row3r.com's lines in our model: is the shortest line the fastest? (HOCR 2026 conditions.)

    .venv/Scripts/python.exe research/bestline/bestline_vs_model.py

The site's "best line" (shortest legal path through the bridges) and each published crew track are put
on the course (station, offset), and timed like our own line in research/raceday/race_review_opt.py:
the day-calibrated wind and the observed flow and level, at the four's pace (its still-air time on our
line), with the measured turning cost. Crew tracks are labelled by their rank on the site only.
Writes bestline_vs_model.json.
"""
import json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "bestline"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "research", "raceday"))
from review_crews import STARBOARD, shore_name                                   # noqa: E402
from coxswain.raceday import card as C
from coxswain.river import charles
from coxswain.river.course import local_tangent_plane
from coxswain.river.route import Route
import race_card as RC

FT = 0.3048


def to_route(course, latlon, name, L):
    a = np.asarray(latlon, float)
    x, y = local_tangent_plane(a[:, 0], a[:, 1], charles.CHARLES_ORIGIN)
    st = np.array([course.nearest_station(p, q) for p, q in zip(x, y)])
    off = np.array([course.offset_of(p, q, s) for p, q, s in zip(x, y, st)])
    keep = (st > 1.0) & (st < L - 1.0)
    st, off = st[keep], off[keep]
    bins = np.arange(0.0, L + 25.0, 25.0)
    so, oo = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (st >= lo) & (st < hi)
        if m.sum():
            so.append(float(st[m].mean())); oo.append(float(np.median(off[m])))
    so, oo = np.array(so), np.array(oo)
    port, stbd = course.limits_at(so)                         # each side's own limit (shore corridor: asymmetric)
    oo = np.clip(oo, -(stbd - 1.0), port - 1.0)
    return Route(np.concatenate([[0.0], so, [L]]), np.concatenate([[oo[0]], oo, [oo[-1]]]), name=name), float(so.min()), float(so.max())


def main():
    R = json.load(open(os.path.join(ROOT, "data", "local", "raceday", "race_review_2026.json")))
    cal = R["calibrated"]
    B = json.load(open(os.path.join(HERE, "bestline_2026.json")))
    model = C.CourseModel.build(starboard=STARBOARD, level_offset=R["level_ft"] * FT)
    course = model.course
    L = float(course.length)
    q = R["flow_cfs"] * 0.0283168466
    model.flow = charles.ContinuityFlow(course, discharge=q)
    boat = RC.build_boat("four", RC.CREWS["four"]["rate"], profile="research")   # depth: Sretenskii, wave term only (PHYSICS_PROGRAMME, 2026-10-21)
    k = C.RUDDER_K["4+"]
    wind = (cal["wind"]["speed"], cal["wind"]["direction"])
    ref = cal["ref_s"]

    def timed(route):
        model.route = route
        sec = C._sections(model, C._evaluate(model, boat, ref, wind), wind, k)
        return sum(s.time_s for s in sec), sum(s.path_m for s in sec), sum(s.rudder_s for s in sec)

    out = dict(conditions=dict(wind=wind, flow_cfs=R["flow_cfs"], ref_s=ref), lines=[])
    best, a, b = to_route(course, B["best"], "bestline.row3r.com shortest line", L)
    t, p, r = timed(best)
    out["lines"].append(dict(name="their shortest line", time_s=t, path_m=p, turning_s=r, covers=[a, b],
                             stations=list(map(float, best.stations)), offsets=list(map(float, best.offsets))))
    print("their shortest line: %.1f s, %.0f m (covers %.0f-%.0f m)" % (t, p, a, b))
    out["lines"].append(dict(name="our line", time_s=cal["ours"]["total"], path_m=cal["ours"]["path_m"], turning_s=cal["ours"]["turning"]))
    out["lines"].append(dict(name="our model's optimum (our arches)", time_s=cal["best"]["total"], path_m=cal["best"]["path_m"], turning_s=cal["best"]["turning"]))
    ranked = sorted(B["entries"].items(), key=lambda kv: float(kv[1]["row"][2].rstrip("%")))
    for i, (key, e) in enumerate(ranked, 1):
        if len(e["rowed"]) < 50:
            continue
        rt, a, b = to_route(course, [p[:2] for p in e["rowed"]], "site rank %d" % i, L)
        if a > 300 or b < L - 300:
            print("site rank %d: track covers only %.0f-%.0f m, skipped" % (i, a, b))
            continue
        t, p, r = timed(rt)
        out["lines"].append(dict(name="site rank %d (%s)" % (i, e["row"][1].split()[-1]), site_pct=e["row"][2], time_s=t, path_m=p,
                                 turning_s=r, stations=list(map(float, rt.stations)), offsets=list(map(float, rt.offsets))))
        print("site rank %2d %-8s %5s over: model %.1f s, %.0f m, turning %.1f s" % (i, e["row"][1].split()[-1], e["row"][2], t, p, r))
    json.dump(out, open(os.path.join(HERE, shore_name("bestline_vs_model.json")), "w"))


if __name__ == "__main__":
    main()
