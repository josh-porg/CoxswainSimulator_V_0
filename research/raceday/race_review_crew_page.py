"""Race review page for any crew (REVIEW_CREW, review_crews.py), from race_review_template_crew.html.

    REVIEW_CREW=<key> .venv/Scripts/python.exe research/raceday/race_review_crew_page.py

Same page as ours (race_review_page.py builds that one, from race_review_template.html), with the crew's
event, place, day factor, race-card targets (the card's own goals for that boat), arches and bestline
comparison filled in. Inputs: <crew json> (race_review_2026.py + race_review_opt.py with REVIEW_CREW),
results/hocr_event_summaries.csv, results/event<N>_2026_places.json, card_<boat>.json, bestline files.
"""
import csv, json, os, re, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.river import charles                                           # noqa: E402
from coxswain.river.course import local_tangent_plane                        # noqa: E402
from review_crews import CFG                                                 # noqa: E402
import race_card as RC                                                       # noqa: E402

ARCH_LABEL = {"centre": "centre", "Cambridge shore": "Cambridge", "Boston shore": "Boston"}


def fmt(s):
    s = round(float(s), 1)
    m = int(s // 60)
    return "%d:%04.1f" % (m, s - 60 * m)


def ordinal(k):
    return "%d%s" % (k, "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th"))


def main():
    R = json.load(open(os.path.join(HERE, CFG["json"])))
    cal = R["calibrated"]
    card = json.load(open(os.path.join(RC.FIELDS, "card_%s.json" % CFG["crew"])))
    mp = card["map"]
    x0, y0 = mp["origin"]
    rows = [r for r in csv.DictReader(l for l in open(os.path.join(HERE, "results", "hocr_event_summaries.csv")) if not l.startswith("#"))]
    by = {(r["year"], r["event"]): r for r in rows}
    ev, ev25 = by[("2026", CFG["event"])], by.get(("2025", CFG["event"]))
    ev_ratio = float(ev["median"]) / float(ev25["median"]) if ev25 else None
    blraw = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_2026.json")))
    blm = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_vs_model.json")))
    bb = np.asarray(blraw["best"], float)
    bx, by_ = local_tangent_plane(bb[:, 0], bb[:, 1], charles.CHARLES_ORIGIN)

    # the optimum's ghost: model time per stretch spread along the stretch by station
    secs = card["sections"]
    tb, tcum = [0.0], 0.0
    for b in cal["best"]["sections"]:
        tcum += b["time_s"]; tb.append(tcum)
    sb = [secs[0]["from_m"]] + [x["to_m"] for x in secs]
    scale = R["race_s"] / cal["ours"]["total"]
    s_grid = np.linspace(0, sb[-1], 600)
    ghost_t = np.interp(s_grid, sb, np.asarray(tb) * scale)
    stretch = [dict(name="%s → %s" % (g["start"], g["end"]), metres=g["to_m"] - g["from_m"], gps=g["time_s"], model=m_o["time_s"],
                    best=m_b["time_s"], rate=g["rate"], wind=m_o["wind"])
               for g, m_o, m_b in zip(R["gps_sections"], cal["ours"]["sections"], cal["best"]["sections"])]
    from coxswain.raceday import card as Cm
    course = Cm.CourseModel.build().course
    ss = np.linspace(0.0, float(course.length), 900)
    hw = course.water_half_width_at(ss)
    left, right = course.offset_position(ss, hw), course.offset_position(ss, -hw)
    water = np.vstack([left, right[::-1]]) - np.array([x0, y0])
    tr = R["track"]
    data = dict(
        map=dict(w=mp["width"], h=mp["height"], water=np.round(water, 1).tolist(),
                 banks=[np.round(left - np.array([x0, y0]), 1).tolist(), np.round(right - np.array([x0, y0]), 1).tolist()],
                 bridges=mp["bridges"], marks=mp["marks"]),
        track=dict(t=tr["t"], x=tr["x"], y=tr["y"], v=tr["speed"], r=tr["rate"], s=tr["station"]),
        ours=cal["ours"]["xy"], best=cal["best"]["xy"], shortest=np.round(np.column_stack([bx - x0, by_ - y0]), 1).tolist(),
        ghost=dict(t=np.round(ghost_t, 2).tolist(), xy=cal["best"]["xy"]),
        offsets=dict(st=R["ours"]["stations"], ours=R["ours"]["offsets"], best_st=cal["best"]["stations"], best=cal["best"]["offsets"],
                     short_st=blm["lines"][0]["stations"], short=blm["lines"][0]["offsets"],
                     bridges=[dict(name=b["name"], s=b["at_m"]) for b in mp["bridges"]], arches=R["arches"]),
        stretch=stretch, boat_len=17.7 if CFG["crew"] == "eight" else 13.0)

    official, median = CFG["official"], float(ev["median"])
    d_med = official - median
    places = None
    pp = os.path.join(HERE, "results", "event%s_2026_places.json" % CFG["event"])
    if os.path.exists(pp):
        places = json.load(open(pp))["times"]                    # by finishing place (adjusted times)
    names = {"medal": "Medal", "requalify": "Requalify", "beat 2025": "Beat 2025's place"}
    trows = []
    for t in card["targets"]:
        if t["name"] == "forecast":
            label, happened = "Our own forecast", "%s (our time)" % fmt(official)
        else:
            k = int(t["place"])
            label = "%s (%s)" % (names.get(t["name"], t["name"].capitalize()), t["place_text"])
            happened = "%s (%s place)" % (fmt(places[k - 1]), ordinal(k)) if places else "–"
        trows.append('      <tr><td>%s</td><td class="n">%s</td><td class="n">%s</td><td class="n">%s</td></tr>'
                     % (label, fmt(t["today_central"]), fmt(t["central"] * cal["day_factor"]), happened))
    wind_pct = 100 * (card["targets"][0]["today_central"] / card["targets"][0]["central"] - 1)
    day_pct = 100 * (cal["day_factor"] - 1)
    pins = R["pins"]
    if all(v == "centre" for v in pins.values()):
        arch_text = "We took the centre arch at River Street, Western Avenue and Weeks."
    else:
        arch_text = "We took the %s arch." % ", the ".join("%s at %s" % (ARCH_LABEL.get(v, v), k.replace(" Footbridge", "")) for k, v in pins.items())
    bl = cal.get("bestline", [])
    pub = [b for b in bl if b["name"].startswith("site rank")]
    nfast = sum(b["total"] < cal["ours"]["total"] for b in pub)
    short = bl[0] if bl else None
    faster = "none comes out faster than ours" if nfast == 0 else ("only one comes out faster than ours" if nfast == 1 else "%d come out faster than ours" % nfast)
    bl_note = ("The shortest legal line (bestline.row3r.com, %.0f m) would have been %+.1f s against the optimum in the same model, "
               "timed at our pace: shortest is not fastest once wind, current, depth and turning count. Of the %d published crew lines, "
               "timed the same way, %s." % (short["path_m"], short["total"] - cal["best"]["total"], len(pub), faster)) if short else ""
    d_depth = cal["ours"]["depth"] - cal["best"]["depth"]
    # Corrected 2026-10-21 (PHYSICS_PROGRAMME, "Shallow water on the Charles"): the first version of this page
    # charged shallow water to the boat's whole drag and told the crew to leave the bank.
    correction = ("Corrected 21 October: an earlier version of this page put about 45 s of the gap on shallow water and said the "
                  "line along the bank cost more than its shorter distance saved. That came from a modelling error: the model "
                  "charged shallow water to the boat's whole drag instead of its wave drag only. On the day's GPS, our boat and "
                  "two other Sammamish crews showed no measurable slowdown along the banks.")
    d_dist = cal["ours"]["distance"] - cal["best"]["distance"]
    line_note = (("Keeping off the banks saves the optimum %.0f s of wave drag in shallow water (1 to 1.5 m deep on the start "
                  "stretch and after Eliot) %s. The GPS did not show that drag, so treat the saving as an upper bound. "
                  % (d_depth, "and gives %.0f s back in distance" % -d_dist if d_dist < -0.5 else "at no extra distance"))
                 if d_depth > 2 else "Shallow water is not where the time is (%+.0f s). " % d_depth) + correction
    rep = {
        "%%DATA%%": json.dumps(data, separators=(",", ":")),
        "%%TITLE%%": "HOCR 2026 Race Review · %s" % CFG["event_name"],
        "%%EYEBROW%%": "Head of the Charles · Saturday 17 October 2026 · %s · bow %d" % (CFG["event_name"], CFG["bow"]),
        "%%PLACE%%": str(CFG["place"]), "%%FIELD%%": str(CFG["field"]), "%%DIVISION%%": CFG["division"],
        "%%FIELD_TEXT%%": "%s, %d crews" % (CFG["event_name"], CFG["field"]),
        "%%OFFICIAL%%": fmt(official), "%%GPS%%": fmt(R["race_s"]), "%%MEDIAN%%": fmt(median), "%%WINNER%%": fmt(float(ev["winner"])),
        "%%VS_MEDIAN%%": ("we were %.0f s ahead of it" % -d_med) if d_med < 0 else ("we were %.0f s behind it" % d_med),
        "%%DAY_TEXT%%": ("Our event's field ran a median <b>%.1f%%</b> slower than in 2025, about <b>%.0f%%</b> on still water once "
                         "2025's own conditions are allowed for." % (100 * (ev_ratio - 1), day_pct)) if ev_ratio else "",
        "%%TARGET_ROWS%%": "\n".join(trows),
        "%%TARGET_NOTE%%": ("The race card's wind model put this wind at about +%.0f%% on our time; the day's results say about +%.0f%%. "
                            "For this report the wind model is calibrated to the day: the observed wind scaled to %.1f m/s (%.0f%% of "
                            "the open-water reference) reproduces it." % (wind_pct, day_pct, cal["wind"]["speed"], 100 * cal["wind"]["scale"])),
        "%%ARCH_TEXT%%": arch_text,
        "%%KBOS%%": "%.0f mph" % (max(o["speed"] for o in R["wind"]["observations"] if o["station"] == "KBOS") / 0.44704),
        "%%GUST%%": "%.0f mph" % (max((o["gust"] for o in R["wind"]["observations"] if o["station"] == "KBOS" and o["gust"] == o["gust"]), default=0) / 0.44704),
        "%%WIND_DIR%%": "%03.0f°" % R["wind"]["direction"], "%%FLOW%%": "%.0f cfs" % R["flow_cfs"],
        "%%LINE_LOSS%%": "%.1f s" % (cal["ours"]["total"] - cal["best"]["total"]),
        "%%PATH_OURS%%": "%.0f m" % cal["ours"]["path_m"], "%%PATH_BEST%%": "%.0f m" % cal["best"]["path_m"],
        "%%D_DIST%%": "%+.1f" % (cal["ours"]["distance"] - cal["best"]["distance"]),
        "%%D_DEPTH%%": "%+.1f" % (cal["ours"]["depth"] - cal["best"]["depth"]),
        "%%D_CURR%%": "%+.1f" % (cal["ours"]["current"] - cal["best"]["current"]),
        "%%D_WIND%%": "%+.1f" % (cal["ours"]["wind"] - cal["best"]["wind"]),
        "%%D_TURN%%": "%+.1f" % (cal["ours"]["turning"] - cal["best"]["turning"]),
        "%%BESTLINE_NOTE%%": bl_note,
        "%%LINE_NOTE%%": line_note,
    }
    page = open(os.path.join(HERE, "race_review_template_crew.html"), encoding="utf-8").read()
    for k, v in rep.items():
        page = page.replace(k, v)
    left_over = re.findall(r"%%[A-Z_0-9]+%%", page)
    assert not left_over, left_over
    open(os.path.join(HERE, CFG["html"]), "w", encoding="utf-8").write(page)
    print("written %s (%.0f kB)" % (CFG["html"], len(page) / 1024))
    for k in sorted(rep):
        if k != "%%DATA%%":
            print("  %s = %s" % (k, rep[k][:170]))


if __name__ == "__main__":
    main()
