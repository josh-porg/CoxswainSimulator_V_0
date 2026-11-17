"""The human coder's power-call pack against the other coders (CODEBOOK addendum of 16 Nov 2026, part b).

    .venv/Scripts/python.exe research/coxing/human_coding/analyse_power.py

Reads data/local/coxing/human_coding_power/ (log_text.jsonl, key.json). Reports, for the phrases coded so far (first
showings): Cohen's kappa for "main target is power", "power among the targets" and "directive power call" (main target
power and stage dec) against the primary coder, the hypothesis-blind coder (if its files exist) and the lexical coder,
with bootstrap intervals; test-retest agreement from the repeats; how the human coded each source of phrase (primary
power calls, lexical-only power calls, controls); and the boat's response after the primary coder's power calls split
by whether the human also calls the lead phrase a directive power call (strokes 3-5, strokes 6-10, template; shift
nulls). Writes analysis.json beside the log.
"""
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, "..", "experiment"))
import examiner_checks as X  # noqa: E402
import rulecoder as RC  # noqa: E402
CP = X.CP

DATA = os.path.abspath(os.path.join(X.EXP, "..", "human_coding_power"))


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def boot(a, b, rng, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    ks = [kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))]


def main():
    log = [json.loads(l) for l in open(os.path.join(DATA, "log_text.jsonl"), encoding="utf-8")]
    key = json.load(open(os.path.join(DATA, "key.json")))["items"]
    rng = np.random.default_rng(5)
    D = X.Data()
    try:
        import blind_target as BT
        blind = {sid: BT.load_blind(sid) for sid in X.SES}
    except Exception:
        blind = None
    text = {sid: {p["i"]: p["text"] for p in D.s[sid]["phrases"]} for sid in X.SES}
    first = [r for r in log if "repeat_of" not in key[r["id"]]]
    H = {}
    for r in first:
        k = key[r["id"]]
        c = r["codes"]
        H[(k["session"], k["i"])] = dict(main=c["target"] == "power", any="power" in c.get("targets", [c["target"]]),
                                         dp=c["target"] == "power" and c.get("stage") == "dec", source=k["source"])
    out = dict(n=len(first), by_source={})
    for src in ("primary", "lexical", "control"):
        v = [h for h in H.values() if h["source"] == src]
        if v:
            out["by_source"][src] = dict(n=len(v), main_power=float(np.mean([h["main"] for h in v])), any_power=float(np.mean([h["any"] for h in v])),
                                         directive_power=float(np.mean([h["dp"] for h in v])))
    print("coded %d phrases; share the human calls a directive power call, by source: %s" % (
        len(first), {k: (v["n"], round(v["directive_power"], 2)) for k, v in out["by_source"].items()}))
    coders = {"primary": lambda s, i: D.codes[s][i]["target"] == "power" and D.codes[s][i]["stage"] == "dec",
              "lexical": lambda s, i: RC.code(text[s][i])["target"] == "power"}
    if blind:
        coders["blind"] = lambda s, i: blind[s].get(i, ("", ""))[0] == "power" and blind[s].get(i, ("", ""))[1] == "dec"
    out["kappa_directive_power"] = {}
    for name, f in coders.items():
        a = [h["dp"] for h in H.values()]
        b = [f(s, i) for (s, i) in H]
        out["kappa_directive_power"][name] = dict(kappa=kappa(a, b), ci=boot(a, b, rng), n=len(a))
        print("  directive power call, human vs %-8s kappa %.2f (95%% %.2f to %.2f), n=%d" % (name, kappa(a, b), *out["kappa_directive_power"][name]["ci"], len(a)))
    reps = [(H.get((key[key[r["id"]]["repeat_of"]]["session"], key[key[r["id"]]["repeat_of"]]["i"])), r) for r in log if "repeat_of" in key[r["id"]]]
    reps = [(a, r["codes"]["target"] == "power" and r["codes"].get("stage") == "dec") for a, r in reps if a]
    if reps:
        out["retest"] = dict(n=len(reps), agreement=float(np.mean([a["dp"] == b for a, b in reps])))
        print("  test-retest (directive power call): %d repeats, %.0f%% identical" % (len(reps), 100 * out["retest"]["agreement"]))
    # the boat's response after the primary coder's power episodes, by the human's code of the lead phrase
    sets = {"human agrees": {}, "human disagrees": {}}
    for sid in X.SES:
        lo, hi = D.valid(sid)
        ep = [e for e in D.episodes(sid) if e["cls"] == "power" and lo <= e["j"] < hi and (sid, e["i"]) in H]
        sets["human agrees"][sid] = np.array([e["j"] for e in ep if H[(sid, e["i"])]["dp"]], int)
        sets["human disagrees"][sid] = np.array([e["j"] for e in ep if not H[(sid, e["i"])]["dp"]], int)
    RNG = [D.valid(s) for s in X.SES]
    out["response"] = {}
    for name, js in sets.items():
        if sum(len(v) for v in js.values()) < 10:
            continue
        r = {wn: X.both([X.window_stat(D.R(s), w) for s in X.SES], [js[s] for s in X.SES], RNG, 1700) for wn, w in (("3_5", (3, 5)), ("6_10", (6, 10)), ("template", "template"))}
        out["response"][name] = r
        print("  power calls, %-16s n=%3d: 3-5 %+.1f (p %.3f; local %+.1f p %.3f) | 6-10 %+.1f (p %.3f) | template %+.1f (p %.3f)" % (
            name, r["3_5"]["n"], 1000 * r["3_5"]["est"], r["3_5"]["p"], 1000 * r["3_5"]["est_local"], r["3_5"]["p_local"],
            1000 * r["6_10"]["est"], r["6_10"]["p"], 1000 * r["template"]["est"], r["template"]["p"]))
    json.dump(out, open(os.path.join(DATA, "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
