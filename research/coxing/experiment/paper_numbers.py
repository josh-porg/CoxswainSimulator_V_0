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
    c = J("review/checks.json")                                 # examiner_checks.py (CODEBOOK addendum of 14 Nov 2026)
    mm1 = lambda x: round(1000 * x, 1)
    bhr = {r["test"]: r for r in c["bh"]["rows"]}
    print("REVIEW 1 BH: %d tests; q<0.05 both nulls: %s; power speed q %.3f / local %.3f; technique q %.3f / %.3f; second-half power q %.3f / %.3f" % (
        c["bh"]["n_tests"], c["bh"]["survive_both"], bhr["H7_power_calls_speed_3_5"]["q"], bhr["H7_power_calls_speed_3_5"]["q_local"],
        bhr["H7_technique_calls_speed_3_5"]["q"], bhr["H7_technique_calls_speed_3_5"]["q_local"],
        bhr["H4_power_3_5_second_half"]["q"], bhr["H4_power_3_5_second_half"]["q_local"]))
    for k in ("power_speed_3_5", "power_speed_3_5_course", "rate_rate_1_3", "length_dps_3_5"):
        sc = (lambda x: round(x, 2)) if k.startswith("rate") else mm1
        for null in ("global", "local"):
            m = c["meta"][k][null]
            print("REVIEW 2 META %s %s: k=%d RE %s CI [%s, %s] p %.3f tau %s I2 %.0f%% PI [%s, %s] LOO %s..%s positive %d/%d sign p %.3f; without PRR %s (p %.3f); without HOTL four %s (p %.3f)" % (
                k, null, m["k"], sc(m["mu"]), sc(m["ci"][0]), sc(m["ci"][1]), m["p"], sc(m["tau"]), 100 * m["I2"], sc(m["prediction"][0]), sc(m["prediction"][1]),
                sc(m["loo_range"][0]), sc(m["loo_range"][1]), m["n_positive"], m["k"], m["sign_p"], sc(m["without_pink_ribbon"]["mu"]), m["without_pink_ribbon"]["p"],
                sc(m["without_hotl_four"]["mu"]), m["without_hotl_four"]["p"]))
    print("REVIEW 2 occasions (global):", [(o["label"], mm1(o["est"]), mm1(o["se"]), o["n"]) for o in c["meta"]["power_speed_3_5"]["global"]["occasions"]])
    for k in ("pooled_all", "pooled_without_pink_ribbon", "pooled_without_hotl_four"):
        v = c["meta"][k]
        print("REVIEW 2 %s: n=%d %s (sd %s) p %.4f | local %s p %.4f" % (k, v["n"], mm1(v["est"]), mm1(v["sd"]), v["p"], mm1(v["est_local"]), v["p_local"]))
    for tag in ("momentum", "course"):
        e = c["event_study"][tag]
        print("REVIEW 3 EVENT STUDY %s n=%d: strokes %s global %s | local %s" % (tag, e["curve"]["n"], e["strokes"], [mm1(x) for x in e["curve"]["est"]], [mm1(x) for x in e["curve"]["est_local"]]))
        print("REVIEW 3 windows %s: %s" % (tag, {w: (mm1(v["est"]), mm1(v["sd"]), round(v["p"], 4), mm1(v["est_local"]), mm1(v["sd_local"]), round(v["p_local"], 4)) for w, v in e["windows"].items()}))
        sl = c["stroke_level"][tag]
        print("REVIEW 4 STROKE LEVEL %s (%d strokes): %s" % (tag, sl["n_strokes"], {m: {cl: (mm1(sl[m]["global"][cl]["est"]), mm1(sl[m]["global"][cl]["sd"]), round(sl[m]["global"][cl]["p"], 3),
                                                                                    mm1(sl[m]["local"][cl]["est"]), round(sl[m]["local"][cl]["p"], 3))
                                                                               for cl in ("power", "tech", "steer")} for m in ("alone", "with_neighbours", "before_only_unplanned", "after_only_unplanned")}))
        t = c["template"][tag]
        print("REVIEW 5 TEMPLATE %s: %s (sd %s) p %.4f | local %s (sd %s) p %.4f" % (tag, mm1(t["est"]), mm1(t["sd"]), t["p"], mm1(t["est_local"]), mm1(t["sd_local"]), t["p_local"]))
    sp = c["spec_curve"]
    print("REVIEW 6 SPEC CURVE: n=%d median %s 5-95%% %s..%s range %s..%s positive %.0f%% p<0.05 %.0f%%; windows only median %s p<0.05 %.0f%%; paper spec %s p %.4f" % (
        sp["n"], mm1(sp["median"]), mm1(sp["q05"]), mm1(sp["q95"]), mm1(sp["min"]), mm1(sp["max"]), 100 * sp["share_positive"], 100 * sp["share_sig"],
        mm1(sp["windows_only"]["median"]), 100 * sp["windows_only"]["share_sig"], mm1(sp["paper"]["est"]), sp["paper"]["p"]))
    print("REVIEW 6 by dimension:", {k: (mm1(v["median"]), round(100 * v["share_sig"])) for k, v in sp["by_dimension"].items()})
    f = c["forecast"]["summary"]
    print("REVIEW 7 FORECAST: BIC orders %s; Ljung-Box (lag 12) rejected at order four %d/15, at the selected order %d/15; one-step error sd %.1f -> %.1f mm/s; OOS R2 %s" % (
        {k: v for k, v in f["bic_orders"].items() if v}, f["ljung_box_rejected_ar4"], f["ljung_box_rejected_selected"], 1000 * f["resid_sd_ar4"], 1000 * f["resid_sd_selected"],
        {k: (round(v["median"], 2), [round(x, 2) for x in v["range"]]) for k, v in f.items() if k.startswith("oos")}))
    print("REVIEW 7 orders by piece:", J("results.json").get("ar_order"))
    for tag in ("momentum_global", "course_local"):
        m = c["crossfit"][tag]
        print("REVIEW 11 CROSSFIT %s: windows %s; RE %s CI [%s, %s] p %.3f tau %s; positive %d/%d; Stouffer z %.2f p %.4f; rows %s" % (
            tag, m["windows_chosen"], mm1(m["mu"]), mm1(m["ci"][0]), mm1(m["ci"][1]), m["p"], mm1(m["tau"]), m["n_positive"], m["k"], m["stouffer_z"], m["stouffer_p"],
            [(r["occasion"][:18], r["window"], mm1(r["est"]), mm1(r["sd"])) for r in m["rows"]]))
    for lab, v in c["labellers"].items():
        print("REVIEW 12 LABELLER %s: n=%d, %.0f%% at a primary power call; %s" % (lab, v["n"], 100 * v["share_at_a_primary_power_call"], {
            k: (mm1(x["est"]), mm1(x["sd"]), round(x["p"], 4), mm1(x["est_local"]), round(x["p_local"], 4)) for k, x in v.items() if isinstance(x, dict)}))
    ha = c["human_agreement"]
    print("REVIEW 13 HUMAN AGREEMENT:", json.dumps(ha)[:1800])
    # the originally specified order four beside the order-selected main analysis (CODEBOOK decision of 15 Nov 2026)
    for fa, fb in (("results.json", "results_ar4.json"), ("results_course.json", "results_course_ar4.json")):
        A, B = J(fa), J(fb)
        print("ORDER SENSITIVITY %s vs %s" % (fa, fb))
        for k in ("H7_power_calls_speed_3_5", "H7_power_calls_rate_1_3", "H7_rate_calls_rate_1_3", "H7_rate_calls_dps_3_5", "H7_rate_calls_speed_3_5",
                  "H7_length_calls_dps_3_5", "H7_length_calls_speed_3_5", "H7_technique_calls_speed_3_5", "H7_steer_calls_speed_3_5", "H3_contingency"):
            sc = 1.0 if "rate_1_3" in k else 1000.0
            x, y = A["hypotheses"][k], B["hypotheses"][k]
            print("  %-30s selected %+.2f (sd %.2f) p %.4f; local %+.2f p %.4f | order four %+.2f (sd %.2f) p %.4f; local %+.2f p %.4f" % (
                k, sc * x["est"][0], sc * x["sd"][0], x["p"][0], sc * x["est_local"][0], x["p_local"][0],
                sc * y["est"][0], sc * y["sd"][0], y["p"][0], sc * y["est_local"][0], y["p_local"][0]))
        for x, y in zip(A["primary"], B["primary"]):
            print("  %-36s selected %+6.1f p %.3f (%.2f); local %+6.1f p %.3f (%.2f) | order four %+6.1f p %.3f (%.2f); local %+6.1f p %.3f (%.2f)" % (
                x["label"], mm1(x["est"]), x["p"], x["p_holm"], mm1(x["est_local"]), x["p_local"], x["p_local_holm"],
                mm1(y["est"]), y["p"], y["p_holm"], mm1(y["est_local"]), y["p_local"], y["p_local_holm"]))
        print("  per-call sd: selected %.1f (call-free %.1f), order four %.1f (%.1f)" % (
            mm(A["heterogeneity"]["sd_calls"]), mm(A["heterogeneity"]["sd_noise"]), mm(B["heterogeneity"]["sd_calls"]), mm(B["heterogeneity"]["sd_noise"])))
    print("REVIEW 8 MDE primary (global, local):", {k: (mm1(v["mde80"]), mm1(v["mde80_local"])) for k, v in c["mde"]["primary"].items()})
    print("REVIEW 8 MDE coxswain differences:", {k: round(v["mde80"], 3) for k, v in c["mde"]["coxswain_difference"].items()})
    print("REVIEW 9 SAMPLE SIZE:", json.dumps(c["sample_size"]))
    ac = json.load(open(os.path.join(HERE, "..", "acoustic", "results.json")))     # research/coxing/acoustic (CODEBOOK addendum of 15 Nov 2026)
    q = ac["qc"]
    print("ACOUSTIC pieces: background %d locked of %d; calls %d; speech share median %.2f (%.2f-%.2f); thump talking minus quiet dB %s" % (
        len(ac["pieces_background"]), len(ac["pieces_voice"]), ac["n_calls"], float(np.median([v["speech_share"] for v in q.values()])),
        min(v["speech_share"] for v in q.values()), max(v["speech_share"] for v in q.values()),
        {k: round(v["thump_when_talking"] - v["thump_when_quiet"], 1) for k, v in q.items()}))
    print("ACOUSTIC A1: out-of-sample gain %.4f (null %.4f sd %.4f) p %.3f; correlations %s" % (
        ac["A1"]["oos_gain"], ac["A1"]["null_mean"], ac["A1"]["null_sd"], ac["A1"]["p"],
        [(r["feature"], r["with_"], round(r["r"], 3), round(r["p"], 4), round(r["q"], 3), round(r["p_local"], 4), round(r["q_local"], 3)) for r in ac["A1"]["correlations"]]))
    for name in ("A2", "A3"):
        rows = ac[name]
        print("ACOUSTIC %s: %d tests; q<0.05 both nulls %s; expected rows and p<0.01 rows: %s" % (
            name, len(rows), [(r["cls"], r["feature"]) for r in rows if r["q"] < 0.05 and r["q_local"] < 0.05],
            [(r["cls"], r["feature"], r["n"], round(r["est"], 3), round(r["p"], 4), round(r["q"], 3), round(r["est_local"], 3), round(r["p_local"], 4), round(r["q_local"], 3), r["expected"])
             for r in rows if r["expected"] or min(r["p"], r["p_local"]) < 0.01]))
    for key in ("V1", "V1_sound_only_unplanned"):
        print("ACOUSTIC %s: %s" % (key, {k: (v["n"], v["classes"], round(v["balanced_accuracy"], 3), round(v["null_mean"], 3), round(v["null_sd"], 3), round(v["p"], 4))
                                         for k, v in ac[key].items() if k != "means"}))
    print("ACOUSTIC V1 means (blind arousal raised):", {c: {f: round(x, 2) for f, x in v.items()} for c, v in ac["V1"]["means"]["blind arousal raised"].items()})
    print("ACOUSTIC V2: n tests %d; survive both %s; q<0.05 either: %s" % (
        len(ac["V2"]), [(r["state"], r["feature"]) for r in ac["V2"] if r["q"] < 0.05 and r["q_local"] < 0.05],
        [(r["state"], r["feature"], round(r["est"], 3), round(r["p"], 4), round(r["q"], 3), round(r["est_local"], 3), round(r["p_local"], 4), round(r["q_local"], 3))
         for r in ac["V2"] if r["q"] < 0.05 or r["q_local"] < 0.05]))
    print("ACOUSTIC V3:", ac["V3"])
    g = J("guide/results.json")                                 # guide_validation.py
    r, c, rep = g["rule_coder_on_guide"], g["guide_classifier_on_calls"], g["repertoire_first_coxswain"]
    coded = sum(v["n"] - v["rule_targets"].get("other", 0) for v in r["by_heading"].values())
    print("GUIDE: %d phrases, headings %s" % (g["n_guide"], g["by_heading"]))
    print("GUIDE rule coder: agreement %.2f chance %.2f p %.4f; among %d phrases it codes %.2f; by heading %s" % (
        r["agreement"], r["chance"], r["p"], coded, sum(v["agree"] * v["n"] for v in r["by_heading"].values()) / coded,
        {h: round(v["agree"], 2) for h, v in r["by_heading"].items()}))
    print("GUIDE classifier: all %s; close %s" % ({k: round(v, 4) if isinstance(v, float) else v for k, v in c["all"].items()},
                                                 {k: round(v, 4) if isinstance(v, float) else v for k, v in c["close"].items()}))
    print("GUIDE classifier by target:", {t: (v["n"], round(v["agree"], 2)) for t, v in c["by_target"].items()})
    print("GUIDE outside types:", c["outside_guide_types"])
    print("GUIDE repertoire (first coxswain): %d of %d matched; marked %d/%d, unmarked %d/%d, Fisher p %.3f; by heading %s" % (
        rep["guide_phrases_with_close_match"], g["n_guide"], rep["marked"]["with_match"], rep["marked"]["n"],
        rep["unmarked"]["with_match"], rep["unmarked"]["n"], rep["fisher_p"],
        {h: "%d/%d" % (v["with_match"], v["n"]) for h, v in rep["by_heading"].items()}))


if __name__ == "__main__":
    main()
