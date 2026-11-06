"""Every number the experimental paper quotes, from the result files beside it (one place to check the text).

    .venv/Scripts/python.exe research/coxing/experiment/paper_numbers.py > data/local/coxing/experiment/paper_numbers.txt

The paper's primary analysis pools every piece of both coxswains (results.json); coxswain differences come from
coxswain_difference.py, and per-coxswain sets (SET=first, SET=race_07) serve what belongs to one coxswain (calling
profile, the debrief). Robustness: COURSE=1, SYNC=counts, and arousal from the lexical (ALT_AROUSAL=all) and
hypothesis-blind (ALT_SRC=blind) coders.
"""
import csv
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "experiment"))
SDIR = os.path.join(HERE, "..", "sessions")
J = lambda f: json.load(open(os.path.join(HERE, f)))
FIRST = ["race_01", "practice_01", "practice_02", "race_04", "race_05", "practice_03", "practice_04", "practice_05",
         "practice_06", "practice_07", "practice_08", "race_06", "race_08", "race_09"]
GROUP = {k: "800 m pieces 1-6" for k in FIRST if k.startswith("practice_0") and k not in ("practice_01", "practice_02")}


def mm(x):
    return 1000 * x


def table1():
    D = pd.read_csv(os.path.join(HERE, "episodes.csv"))
    rows = {}
    for sid in FIRST + ["race_07", "control_01", "control_02"]:
        b = list(csv.DictReader(open(os.path.join(SDIR, sid, "boat.csv"))))
        t = np.array([float(r["t"]) for r in b])
        sp = np.array([float(r["speed"]) for r in b])
        ph = json.load(open(os.path.join(SDIR, sid, "phrases.json"), encoding="utf-8"))
        meta = json.load(open(os.path.join(SDIR, sid, "meta.json")))
        g = GROUP.get(sid, sid)
        r = rows.setdefault(g, dict(boat=meta.get("boat_class"), strokes=0, minutes=0.0, phrases=0, calls=0, counts=0, dist=0.0))
        r["strokes"] += len(t)
        r["minutes"] += (t[-1] - t[0]) / 60
        r["phrases"] += sum(1 for p in ph if t[0] <= p["t"] <= t[-1])
        r["calls"] += int(((D.session == sid) & (D.target != "count")).sum())
        r["counts"] += int(((D.session == sid) & (D.target == "count")).sum())
        r["dist"] += float(np.sum(np.diff(t) * sp[1:]))
    print("TABLE 1 (piece, boat, strokes, minutes, phrases (per min), calls, count runs, mean speed m/s)")
    tot = dict(strokes=0, phrases=0, calls=0)
    for g, r in rows.items():
        print("  %-18s %-3s %4d %5.1f %4d (%4.1f) %4d %3d %.2f" % (g, r["boat"], r["strokes"], r["minutes"], r["phrases"],
              r["phrases"] / r["minutes"], r["calls"], r["counts"], r["dist"] / (r["minutes"] * 60)))
        if not g.startswith(("race_07", "control")):
            for k in tot:
                tot[k] += r[k]
    print("  first coxswain totals:", tot)


def prim(f):
    r = J(f)
    print("PRIMARY %s (calls %d)" % (f, r["n"]["calls"]))
    for x in r["primary"]:
        print("  %-36s n %4d  %+6.1f +- %4.1f p %.3f (%.2f) | local %+6.1f +- %4.1f p %.3f (%.2f)" % (
            x["label"], x["n"], mm(x["est"]), mm(x["sd"]), x["p"], x["p_holm"], mm(x["est_local"]), mm(x["sd_local"]),
            x["p_local"], x["p_local_holm"]))


def hyp(f, keys=None):
    r = J(f)
    H = r["hypotheses"]
    print("HYPOTHESES %s" % f)
    for k, v in H.items():
        if keys and k not in keys:
            continue
        e = np.atleast_1d(v["est"])
        if len(e) > 4:
            print("  %-34s n %s  global: %s" % (k, v.get("n"), " ".join("%+.0f" % mm(x) for x in e)))
            print("  %-34s       local:  %s" % ("", " ".join("%+.0f" % mm(x) for x in v["est_local"])))
            print("  %-34s       sd:     %s" % ("", " ".join("%.0f" % mm(x) for x in v["sd"])))
            continue
        nn = {a: b for a, b in v.items() if a.startswith("n")}
        sc = 1.0 if "rate_1_3" in k else 1000.0
        print("  %-34s %s  %s" % (k, nn, " | ".join("%+.2f (sd %.2f) p %.3f; local %+.2f (sd %.2f) p %.3f" % (
            sc * a, sc * b, c, sc * d, sc * g, h) for a, b, c, d, g, h in zip(
            e, np.atleast_1d(v["sd"]), np.atleast_1d(v["p"]), np.atleast_1d(v["est_local"]), np.atleast_1d(v["sd_local"]),
            np.atleast_1d(v["p_local"])))))
    h = r["heterogeneity"]
    print("  heterogeneity: per-call sd %.1f, call-free %.1f, reliability %.3f (upper %.3f)" % (
        mm(h["sd_calls"]), mm(h["sd_noise"]), h["reliability"], h["reliability_upper"]))
    if "shares" in r:
        print("  shares:", json.dumps({k: ({a: round(100 * b) for a, b in v.items()} if isinstance(v, dict) else round(100 * v))
                                       for k, v in r["shares"].items()}))
    if "today" in r:
        print("  practice:", json.dumps({k: v for k, v in r["today"].items() if k != "pieces"}, default=float)[:600])
    if "vocal_validity" in r:
        print("  vocal validity:", json.dumps(r["vocal_validity"]))
        v = r["hypotheses"]["vocal_arousal"]
        print("  vocal Holm:", v["p_holm"], v["p_local_holm"])
    if "crossover" in r:
        print("  crossover:", json.dumps(r["crossover"]))


KEY = ["vocal_arousal", "H1_power_speed_curve", "H1_all_calls_speed_curve", "H7_power_calls_speed_3_5", "H7_power_calls_rate_1_3",
       "H7_rate_calls_rate_1_3", "H7_rate_calls_dps_3_5", "H7_rate_calls_speed_3_5", "H7_length_calls_dps_3_5",
       "H7_length_calls_speed_3_5", "H7_technique_calls_speed_3_5", "H7_steer_calls_speed_3_5", "H4_power_11_20_first_half",
       "H4_power_11_20_second_half", "H3_contingency"]


def main():
    table1()
    prim("results.json")
    hyp("results.json")
    for f in ("results_course.json", "results_countsync055.json", "results_course_countsync055.json"):
        hyp(f, KEY)
    for f in ("results_course.json", "results_rulearousal_all.json", "results_blindarousal_all.json",
              "results_course_blindarousal_all.json", "results_countsync055.json"):
        prim(f)
    for f in ("coxswain_difference.json", "coxswain_difference_course.json"):
        print("COXSWAIN DIFFERENCE %s:" % f)
        for k, v in J(f).items():
            sc = 1.0 if "spm" in k else 1000.0
            print("  %-38s A %+.2f  B %+.2f  B-A %+.2f (sd %.2f) p %.3f  n %d/%d" % (
                k, sc * v["A"], sc * v["B"], sc * v["diff"], sc * v["diff_sd"], v["p"], v["n_A"], v["n_B"]))
    print("VOICE by coder agreement (all pieces):", json.dumps(J("blind_arousal/voice.json")))
    print("--- per coxswain: the first coxswain's pieces (debrief), the second's race ---")
    prim("results_set_first.json")
    hyp("results_set_first.json", KEY)
    hyp("results_course_set_first.json", KEY)
    prim("results_set_race_07.json")
    hyp("results_set_race_07.json", KEY)
    hyp("results_course_set_race_07.json", KEY)
    print("PER PIECE (power calls, strokes 3-5):", {k: (v["n"], round(mm(v["est"]), 1), round(v["p"], 3))
                                                    for k, v in J("per_piece.json").items()})
    print("RELIABILITY (lexical coder):", J("reliability.json"))
    print("BLIND AROUSAL agreement:", json.dumps({k: J("blind_arousal/agreement.json")[k] for k in ("first_coxswain", "pooled")}))
    print("PUSH MODEL:", json.dumps(J("push_response_model.json").get("_compare")))
    t = J("two_coxswains.json")
    print("TWO COXSWAINS policy:", json.dumps(t["policy"])[:3000])
    print("TWO COXSWAINS pushes:", json.dumps(t["pushes"]))
    print("TWO COXSWAINS gearing:", json.dumps(t["gearing"]))
    print("COUNT ANCHOR (0.55):", json.dumps({k: (v["shift_phase"], v["n"], v["phase"], v["R"], v["anchored"])
                                               for k, v in json.load(open(os.path.join(HERE, "..", "sync", "count_anchor.json")))["0.55"].items()}))


if __name__ == "__main__":
    main()
