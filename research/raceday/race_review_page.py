"""Build the HOCR 2026 race review page (race_review_2026.html beside this file) from the analysis files.

    .venv/Scripts/python.exe research/raceday/race_review_page.py

Inputs: race_review_2026.json (race_review_2026.py, race_review_opt.py), results/hocr_event_summaries.csv,
card_four.json (map, sections, pre-race targets), ../bestline/bestline_vs_model.json and bestline_2026.json.
The page is self-contained (data inline); the animation runs from the per-stroke GPS.
"""
import csv, json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from coxswain.river import charles
from coxswain.river.course import local_tangent_plane


def fmt(s):
    s = round(float(s), 1)
    m = int(s // 60)
    return "%d:%04.1f" % (m, s - 60 * m)


def main():
    R = json.load(open(os.path.join(HERE, "race_review_2026.json")))
    cal = R["calibrated"]
    card = json.load(open(os.path.join(HERE, "card_four.json")))
    mp = card["map"]
    x0, y0 = mp["origin"]
    rows = [r for r in csv.DictReader(l for l in open(os.path.join(HERE, "results", "hocr_event_summaries.csv")) if not l.startswith("#"))]
    by = {(r["year"], r["event"]): r for r in rows}
    ratios = sorted((int(e), float(by[("2026", e)]["median"]) / float(by[("2025", e)]["median"]))
                    for (y, e) in by if y == "2026" and int(e) >= 9 and ("2025", e) in by)
    day_ratio = float(np.median([r for _, r in ratios]))
    ev = by[("2026", "22")]
    bl = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_vs_model.json")))
    blraw = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_2026.json")))
    bb = np.asarray(blraw["best"], float)
    bx, by_ = local_tangent_plane(bb[:, 0], bb[:, 1], charles.CHARLES_ORIGIN)

    # the optimum's ghost: model time per stretch spread along the stretch by station
    secs = card["sections"]
    tb, tcum = [0.0], 0.0
    for s, b in zip(secs, cal["best"]["sections"]):
        tcum += b["time_s"]; tb.append(tcum)
    sb = [secs[0]["from_m"]] + [s["to_m"] for s in secs]
    scale = R["race_s"] / cal["ours"]["total"]          # the model reproduces our time on our line: 1.0
    bst = np.asarray(cal["best"]["stations"]); L = sb[-1]
    s_grid = np.linspace(0, L, 600)
    ghost_t = np.interp(s_grid, sb, np.asarray(tb) * scale)

    # stretch table: GPS against the model's expectation for our line at our steady pace
    stretch = []
    for g, m_o, m_b in zip(R["gps_sections"], cal["ours"]["sections"], cal["best"]["sections"]):
        stretch.append(dict(name="%s → %s" % (g["start"], g["end"]), metres=g["to_m"] - g["from_m"], gps=g["time_s"],
                            model=m_o["time_s"], best=m_b["time_s"], rate=g["rate"], wind=m_o["wind"]))
    # the water: both banks from the course model's water edges, as one outline (left bank out, right bank back)
    from coxswain.raceday import card as Cm
    course = Cm.CourseModel.build().course
    ss = np.linspace(0.0, float(course.length), 900)
    hw = course.water_half_width_at(ss) if hasattr(course, "water_half_width_at") else course.half_width_at(ss)
    left, right = course.offset_position(ss, hw), course.offset_position(ss, -hw)
    water = np.vstack([left, right[::-1]]) - np.array([x0, y0])
    tr = R["track"]
    data = dict(
        map=dict(w=mp["width"], h=mp["height"], water=np.round(water, 1).tolist(),
                 banks=[np.round(left - np.array([x0, y0]), 1).tolist(), np.round(right - np.array([x0, y0]), 1).tolist()],
                 bridges=mp["bridges"], marks=mp["marks"]),
        track=dict(t=tr["t"], x=tr["x"], y=tr["y"], v=tr["speed"], r=tr["rate"], s=tr["station"]),
        ours=cal["ours"]["xy"], best=cal["best"]["xy"],
        shortest=np.round(np.column_stack([bx - x0, by_ - y0]), 1).tolist(),
        ghost=dict(t=np.round(ghost_t, 2).tolist(), xy=cal["best"]["xy"]),
        offsets=dict(st=R["ours"]["stations"], ours=R["ours"]["offsets"], best_st=cal["best"]["stations"], best=cal["best"]["offsets"],
                     short_st=bl["lines"][0]["stations"], short=bl["lines"][0]["offsets"],
                     bridges=[dict(name=b["name"], s=b["at_m"]) for b in mp["bridges"]],
                     arches={k: v for k, v in R["arches"].items()}),
        stretch=stretch, ratios=ratios)
    n = dict(place=21, field=26, official=24 * 60 + 7.691, gps=R["race_s"], median=float(ev["median"]), winner=float(ev["winner"]),
             q25=float(ev["q25"]), day_ratio=day_ratio, day_factor=day_ratio * 1.02,
             wind_obs=R["wind"], wind_cal=cal["wind"], flow=R["flow_cfs"], level=R["level_ft"],
             ours=cal["ours"], best=cal["best"], bl=bl["lines"], ref=cal["ref_s"])
    model_req, model_fc = 24 * 60 + 52.8, 25 * 60 + 56.8
    still_req, still_fc = 1282.9, 1337.2
    page = open(os.path.join(HERE, "race_review_template.html"), encoding="utf-8").read()
    rep = {
        "%%DATA%%": json.dumps(data, separators=(",", ":")),
        "%%OFFICIAL%%": fmt(n["official"]), "%%GPS%%": fmt(n["gps"]),
        "%%MEDIAN%%": fmt(n["median"]), "%%WINNER%%": fmt(n["winner"]),
        "%%BEHIND_MEDIAN%%": "+%.0f s" % (n["official"] - n["median"]),
        "%%DAY_PCT%%": "%.1f%%" % (100 * (day_ratio - 1)), "%%DAY_FACTOR_PCT%%": "%.0f%%" % (100 * (n["day_factor"] - 1)),
        "%%MODEL_REQ%%": fmt(model_req), "%%DAY_REQ%%": fmt(still_req * n["day_factor"]),
        "%%MODEL_FC%%": fmt(model_fc), "%%DAY_FC%%": fmt(still_fc * n["day_factor"]),
        "%%KBOS%%": "%.0f mph" % (max(o["speed"] for o in R["wind"]["observations"] if o["station"] == "KBOS") / 0.44704),
        "%%GUST%%": "%.0f mph" % (max((o["gust"] for o in R["wind"]["observations"] if o["station"] == "KBOS" and o["gust"] == o["gust"]), default=0) / 0.44704),
        "%%WIND_DIR%%": "%03.0f°" % R["wind"]["direction"], "%%FLOW%%": "%.0f cfs" % n["flow"],
        "%%LINE_LOSS%%": "%.1f s" % (cal["ours"]["total"] - cal["best"]["total"]),
        "%%PATH_OURS%%": "%.0f m" % cal["ours"]["path_m"], "%%PATH_BEST%%": "%.0f m" % cal["best"]["path_m"],
        "%%D_DIST%%": "%+.1f" % (cal["ours"]["distance"] - cal["best"]["distance"]),
        "%%D_DEPTH%%": "%+.1f" % (cal["ours"]["depth"] - cal["best"]["depth"]),
        "%%D_CURR%%": "%+.1f" % (cal["ours"]["current"] - cal["best"]["current"]),
        "%%D_WIND%%": "%+.1f" % (cal["ours"]["wind"] - cal["best"]["wind"]),
        "%%D_TURN%%": "%+.1f" % (cal["ours"]["turning"] - cal["best"]["turning"]),
        "%%SHORT_T%%": "%+.1f s" % (bl["lines"][0]["time_s"] - cal["best"]["total"]),
        # the sign decides the sentence: since the depth fix (2026-10-21) the shortest line can beat the optimiser's
        "%%SHORT_SENTENCE%%": (("would have been %+.1f s against the optimum in the same model: shortest is not fastest once "
                                "wind, current, depth and turning count." % (bl["lines"][0]["time_s"] - cal["best"]["total"]))
                               if bl["lines"][0]["time_s"] >= cal["best"]["total"] else
                               ("would have been %.1f s faster than the optimum found here, in the same model. Once depth is "
                                "charged to the wave drag only, the shortest line is close to the fastest; the optimiser, which "
                                "moves the line every 150 m through our arches, does not reach it."
                                % (cal["best"]["total"] - bl["lines"][0]["time_s"]))),
        "%%SHORT_PATH%%": "%.0f m" % bl["lines"][0]["path_m"],
        "%%OUR_PCT%%": "%.1f%%" % (100 * (cal["ours"]["path_m"] / bl["lines"][0]["path_m"] - 1)),   # both in our course frame
        "%%WIND_CAL%%": "%.1f m/s" % cal["wind"]["speed"], "%%WIND_SCALE%%": "%.0f%%" % (100 * cal["wind"]["scale"]),
    }
    for k, v in rep.items():
        page = page.replace(k, v)
    open(os.path.join(HERE, "race_review_2026.html"), "w", encoding="utf-8").write(page)
    print("written race_review_2026.html (%.0f kB)" % (len(page) / 1024))
    for k in sorted(rep):
        if k != "%%DATA%%":
            print("  %s = %s" % (k, rep[k]))


if __name__ == "__main__":
    main()
