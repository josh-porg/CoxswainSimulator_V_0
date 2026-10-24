"""Second pass of the race review: the optimum through our arches with the wind calibrated to the day.

    .venv/Scripts/python.exe research/raceday/race_review_opt.py

race_review_2026.py found (1) the model's wind cost on the observed wind (+26% on our line) far above what
the day's results show (fields 4.9% slower than 2025, about +7% on still water), and (2) a too-stiff
optimum (ten free points, 15 m from the bank). Here:

* the wind is the observed direction with its speed scaled so the four's whole-line factor (wind table,
  river included) equals the measured day factor, 1.0494 x 1.02 (results/hocr_event_summaries.csv);
* the model's pace is set so our own line, in those conditions, takes our GPS race time;
* the optimum keeps the arches we took, may move every 150 m, stays MARGIN m inside the banks (the
  course rule margin used throughout; buoys are not modelled), and is found by Powell from our own line.
Both lines are then broken down cause by cause and stretch by stretch. Updates race_review_2026.json.
"""
import csv, json, os, sys
import numpy as np
from scipy.optimize import minimize, brentq

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C, plan as P, gusts as GU
from coxswain.river import charles, lines
from coxswain.river.route import Route
import race_card as RC

FT = 0.3048
MARGIN = 4.0
STEP = 150.0
sys.path.insert(0, CODE)
from review_crews import CFG                                               # noqa: E402
DAY = CFG["day"]


def main():
    R = json.load(open(os.path.join(HERE, CFG["json"])))
    level, q = R["level_ft"], R["flow_cfs"] * 0.0283168466
    model = C.CourseModel.build(level_offset=level * FT)
    deep = C.CourseModel.build(level_offset=level * FT + 2.0)
    course = model.course
    L = float(course.length)
    spec = RC.CREWS[CFG["crew"]]
    boat = RC.build_boat(CFG["crew"], spec["rate"], profile="research")   # depth: Sretenskii, wave term only (PHYSICS_PROGRAMME, 2026-10-21)
    k = C.RUDDER_K[spec["boat"]]
    ours = Route(np.asarray(R["ours"]["stations"]), np.asarray(R["ours"]["offsets"]), name="our line")
    race_s = R["race_s"]

    # 1. the wind speed that makes the day's measured factor (river effect taken from the flow table)
    T = json.load(open(os.path.join(RC.FIELDS, "wind_table_%s.json" % CFG["crew"])))
    bundle = json.load(open(os.path.join(RC.FIELDS, "site_bundle.json")))["crews"][CFG["crew"]]
    river = 1.0 + float(np.interp(q, bundle["flow"]["discharge_m3s"], bundle["flow"]["effect_fraction"]))
    d = R["wind"]["direction"]
    f = lambda v: GU.table_at(T, v, d)["factor"] * river - DAY                  # noqa: E731
    v_cal = brentq(f, 0.0, 12.0) if f(0.0) < 0 < f(12.0) else 0.0
    wind = (float(v_cal), float(d))
    print("observed reference wind %.1f m/s from %03.0f; calibrated to the day (factor %.3f): %.2f m/s (x%.2f)"
          % (R["wind"]["speed"], d, DAY, v_cal, v_cal / R["wind"]["speed"]))

    def sections(m, route, qq, w, kk, ref):
        m.flow = charles.ContinuityFlow(m.course, discharge=qq)
        m.route = route
        return C._sections(m, C._evaluate(m, boat, ref, w), w, kk)

    total = lambda s: sum(z.time_s for z in s)                                  # noqa: E731
    # 2. the pace: our line in the day's conditions takes our GPS time
    ref = race_s / DAY
    for _ in range(3):
        ref *= race_s / total(sections(model, ours, q, wind, k, ref))
    print("model pace: still-air time %.1f s for our line (GPS race %.1f s)" % (ref, race_s))

    # 3. the optimum through our arches
    # plan.optimise_line (2026-10-22): bridge points free inside their arches, ends free, warm starts
    # from our line and bestline's shortest legal line; the old search fixed every arch point at the
    # arch centre, and the trestle rule was wrong (TRACKING)
    model.flow = charles.ContinuityFlow(course, discharge=q)
    BLs = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_vs_model.json")))
    sh = next(x for x in BLs["lines"] if x["name"] == "their shortest line")
    shortest = Route(np.asarray(sh["stations"], float), np.asarray(sh["offsets"], float), name="shortest")
    opt = P.optimise_line(model, boat, ref, R["pins"], k, wind, starts=(ours, shortest), step=STEP, margin=MARGIN)
    best = opt["route"]
    n = [opt["evaluations"]]
    print("point-search optimum: %.1f s from %.1f s at the warm start %r (%d evaluations)"
          % (opt["time_s"], opt["unoptimised_s"], opt["warm_start"], n[0]))
    # the smooth optimum (2026-10-22): Radau collocation, curvature as the control, the measured turning
    # loss for this boat, the water's edge and the arch windows as constraints (coxswain/raceday/line_ocp.py).
    # It is the "best line" from here on; the point search's kinked line is kept as best_points.
    from coxswain.raceday import line_ocp as OCP
    points_best = best
    rad = OCP.solve(model, boat, ref, wind, R["pins"], k, ours)
    from coxswain.river import lines as LN
    illegal = LN.illegal_bridges(rad["route"], course, model.raster, model.gates)
    print("Radau optimum: %s, %d iterations; %.1f s in the full model (point search %.1f s); illegal bridges %s"
          % (rad["status"], rad["iterations"], rad["time_s"], opt["time_s"], illegal))
    if rad["status"] == "solved" and not illegal:
        best = rad["route"]

    def breakdown(route):
        a = sections(deep, route, 0.05, None, 0.0, ref); b = sections(model, route, 0.05, None, 0.0, ref)
        c = sections(model, route, q, None, 0.0, ref); dd = sections(model, route, q, wind, 0.0, ref)
        e = sections(model, route, q, wind, k, ref)
        return dict(total=total(e), distance=total(a), depth=total(b) - total(a), current=total(c) - total(b),
                    wind=total(dd) - total(c), turning=sum(z.rudder_s for z in e), path_m=sum(z.path_m for z in e),
                    sections=[dict(start=z.start, end=z.end, time_s=z.time_s, path_m=z.path_m, rudder_s=z.rudder_s,
                                   wind=total([dd[i]]) - total([c[i]]), depth=total([b[i]]) - total([a[i]]),
                                   current=total([c[i]]) - total([b[i]])) for i, z in enumerate(e)])
    b_ours, b_best = breakdown(ours), breakdown(best)
    b_points = breakdown(points_best)
    print("our line %.1f s (%.0f m), optimum %.1f s (%.0f m): %+.1f s" % (b_ours["total"], b_ours["path_m"], b_best["total"],
          b_best["path_m"], b_ours["total"] - b_best["total"]))
    for key in ("distance", "depth", "current", "wind", "turning"):
        print("   %-8s ours %+7.1f  optimum %+7.1f  difference %+6.1f" % (key, b_ours[key], b_best[key], b_ours[key] - b_best[key]))
    s = np.linspace(0, L, 600)
    x0m, y0m = R["map"]["origin"]
    xy = course.offset_position(s, best.offset_at(s))
    R["calibrated"] = dict(day_factor=DAY, wind=dict(speed=wind[0], direction=wind[1], scale=v_cal / R["wind"]["speed"]),
                           ref_s=ref, margin_m=MARGIN, step_m=STEP, evaluations=n[0],
                           ours=b_ours, best=dict(b_best, stations=list(map(float, best.stations)), offsets=list(map(float, best.offsets)),
                                                  xy=np.round(np.column_stack([xy[:, 0] - x0m, xy[:, 1] - y0m]), 1).tolist(),
                                                  method="radau" if best is not points_best else "points"),
                           best_points=dict(b_points, stations=list(map(float, points_best.stations)), offsets=list(map(float, points_best.offsets))),
                           radau=dict(status=rad["status"], iterations=rad["iterations"], ocp_time_s=rad["ocp_time_s"], time_s=rad["time_s"],
                                      frame_s=rad["frame_s"], frame_n=rad["frame_n"], curvature=rad["curvature"], mesh=rad["mesh"],
                                      gates=rad["gates"], edge_m=rad["edge_m"]))
    # bestline.row3r.com's published lines, timed the same way (for this crew's pace and conditions)
    BL = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_vs_model.json")))
    R["calibrated"]["bestline"] = []
    for ln in [x for x in BL["lines"] if "stations" in x]:
        r = Route(np.asarray(ln["stations"], float), np.asarray(ln["offsets"], float), name=ln["name"])
        e = sections(model, r, q, wind, k, ref)
        R["calibrated"]["bestline"].append(dict(name=ln["name"], total=total(e), path_m=sum(z.path_m for z in e)))
    print("bestline lines: shortest %.1f s; %d of %d published lines faster than this crew's line"
          % (R["calibrated"]["bestline"][0]["total"], sum(b["total"] < b_ours["total"] for b in R["calibrated"]["bestline"] if b["name"].startswith("site rank")),
             sum(b["name"].startswith("site rank") for b in R["calibrated"]["bestline"])))
    oxy = course.offset_position(s, ours.offset_at(s))
    R["calibrated"]["ours"]["xy"] = np.round(np.column_stack([oxy[:, 0] - x0m, oxy[:, 1] - y0m]), 1).tolist()
    json.dump(R, open(os.path.join(HERE, CFG["json"]), "w"))
    print("updated", CFG["json"])


if __name__ == "__main__":
    main()
