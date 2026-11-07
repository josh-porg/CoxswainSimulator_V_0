"""Outside raters: agreement analysis fixed in the CODEBOOK addendum of 7 Nov 2026.

    .venv/Scripts/python.exe research/coxing/rating_site/analyse.py [--db path/to/ratings.db]

Download the deployed site's ratings.db (or set RATING_DATA to its folder) first. Reports:
- Krippendorff's alpha per field across outside raters (main target, stage, action, arousal: nominal; voice: interval),
  overall and within each role;
- per rater: agreement (share identical) with the leave-one-out consensus of the other raters on the same items, with
  the human coder (the study lead's codes from the human coding tool) and with the machine coder; check-item accuracy,
  flagging raters below 50% after at least five days;
- voice: Spearman correlation of each clip's mean rating with the human coder's rating and with the acoustic vocal
  arousal index.
Writes analysis.json beside the database.
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import Counter

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "..", "..", "callmodel"))
sys.path.insert(0, os.path.join(CODE, "..", "human_coding"))
import app  # noqa: E402  (alpha, pack)
import call_properties as CP  # noqa: E402

ROOT = os.path.abspath(os.path.join(CODE, "..", "..", ".."))
LOCAL = os.path.join(ROOT, "data", "local", "coxing")
FIELDS = ["target", "stage", "action", "arousal"]


def value(codes, f):
    return codes["targets"][0] if f == "target" else codes[f]


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(app.DATA, "ratings.db"))
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    con.row_factory = sqlite3.Row
    raters = {r["id"]: dict(r) for r in con.execute("SELECT * FROM raters")}
    rows = [dict(r) for r in con.execute("SELECT * FROM ratings")]
    p = app.pack()
    key = json.load(open(os.path.join(LOCAL, "human_coding", "key.json")))["items"]
    hlog = os.path.join(LOCAL, "human_coding", "log_text.jsonl")
    human = {r["id"]: r["codes"] for r in map(json.loads, open(hlog, encoding="utf-8"))} if os.path.exists(hlog) else {}
    codes_cache = {}

    def machine(item_id):
        k = key[item_id]
        if k["session"] not in codes_cache:
            codes_cache[k["session"]] = CP.load_codes(os.path.join(LOCAL, "experiment", "codes_%s.txt" % k["session"]))
        return CP.code_of(k["i"], codes_cache[k["session"]])

    text = [r for r in rows if r["kind"] == "text" and not r["is_check"] and not r["is_repeat"]]
    for r in text:
        r["c"] = json.loads(r["codes"])
    out = {"n_ratings": len(rows), "raters": {}}
    # alpha overall and by role
    for scope in ["all"] + sorted({raters[r["rater_id"]]["role"] for r in text if raters[r["rater_id"]]["role"]}):
        sel = [r for r in text if scope == "all" or raters[r["rater_id"]]["role"] == scope]
        out.setdefault("alpha", {})[scope] = {f: app.alpha(app._units([(r["item_id"], value(r["c"], f)) for r in sel])) for f in FIELDS}
    voice = [r for r in rows if r["kind"] == "voice" and not r["is_repeat"]]
    out["alpha"]["all"]["voice (interval)"] = app.alpha(app._units([(r["item_id"], r["rating"]) for r in voice]), "interval")
    print("alpha:", json.dumps(out["alpha"]))
    # per rater
    by_item = {}
    for r in text:
        by_item.setdefault(r["item_id"], []).append(r)
    for rid, info in raters.items():
        mine = [r for r in text if r["rater_id"] == rid]
        res = dict(label=info["label"], role=info["role"], n_text=len(mine))
        for f in FIELDS:
            loo, hum, mac = [], [], []
            for r in mine:
                others = [value(o["c"], f) for o in by_item[r["item_id"]] if o["rater_id"] != rid]
                if others:
                    top, n = Counter(others).most_common(1)[0]
                    if n > len(others) / 2:
                        loo.append(value(r["c"], f) == top)
                if r["item_id"] in human:
                    hv = human[r["item_id"]]
                    hum.append(value(r["c"], f) == (hv["target"] if f == "target" else hv[f]))
                mv = machine(r["item_id"])
                if f == "target" or mv["target"] not in ("count", "other"):
                    mac.append(value(r["c"], f) == mv[f])
            res[f] = dict(vs_consensus=float(np.mean(loo)) if loo else None, vs_human=float(np.mean(hum)) if hum else None,
                          vs_machine=float(np.mean(mac)) if mac else None)
        checks = [r for r in rows if r["rater_id"] == rid and r["is_check"]]
        acc = np.mean([json.loads(r["codes"])["targets"][0] == p["by_id"][r["item_id"]]["check"] for r in checks]) if checks else None
        days = len({r["day"] for r in rows if r["rater_id"] == rid})
        res.update(check_accuracy=None if acc is None else float(acc), n_checks=len(checks), days=days,
                   flagged=bool(acc is not None and days >= 5 and acc < 0.5))
        out["raters"][info["public_id"]] = res
        print("%-8s %-10s n=%3d checks %s  target vs consensus/human/machine %s" % (
            info["public_id"], info["role"] or "-", len(mine), "-" if acc is None else "%.0f%%" % (100 * acc),
            "/".join("-" if res["target"][k] is None else "%.2f" % res["target"][k] for k in ("vs_consensus", "vs_human", "vs_machine"))))
    # voice against the human coder and the acoustic index
    mean = {}
    for r in voice:
        mean.setdefault(r["item_id"], []).append(r["rating"])
    mean = {k: float(np.mean(v)) for k, v in mean.items()}
    vlog = os.path.join(LOCAL, "human_coding", "log_voice.jsonl")
    hv = {r["id"]: r["rating"] for r in map(json.loads, open(vlog, encoding="utf-8"))} if os.path.exists(vlog) else {}
    common = [k for k in mean if k in hv]
    out["voice"] = dict(clips=len(mean), spearman_with_human=spearman([mean[k] for k in common], [hv[k] for k in common]))
    try:
        import analyse as HC
        idx = HC.acoustic_index({(key[k]["session"], key[k]["i"]) for k in mean})
        pairs = [(mean[k], idx.get((key[k]["session"], key[k]["i"]))) for k in mean]
        pairs = [(x, y) for x, y in pairs if y is not None and np.isfinite(y)]
        out["voice"]["spearman_with_acoustic_index"] = spearman([x for x, _ in pairs], [y for _, y in pairs])
    except Exception as e:                                   # audio not available where this runs
        out["voice"]["spearman_with_acoustic_index"] = None
        out["voice"]["note"] = "acoustic index not computed: %s" % e
    print("voice:", json.dumps(out["voice"]))
    json.dump(out, open(os.path.join(os.path.dirname(a.db), "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
