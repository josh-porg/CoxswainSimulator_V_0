"""Outside raters: the agreement analysis fixed in the CODEBOOK addendum of 7 Nov 2026.

    .venv/Scripts/python.exe research/coxing/rating_site/analyse.py path/to/ratings_YYYY-MM-DD.json

Input: the JSON the site's admin view downloads (raters and ratings, as stored in the Google Sheet). Item codes are
mapped back to piece and phrase through data/local/coxing/rating_site/site_key.json (private). Reports:
- Krippendorff's alpha per field across outside raters (main target, stage, action, arousal: nominal; voice: interval),
  overall and within each role;
- per rater: agreement (share identical) with the leave-one-out majority of the other raters on the same items, with
  the human coder (the study lead's codes from the human coding tool) and with the machine coder; check-item accuracy,
  flagging raters below 50% after at least five days; number of devices used;
- voice: Spearman correlation of each clip's mean rating with the human coder's rating and with the acoustic vocal
  arousal index.
Writes analysis_<input name>.json beside the input.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, "..", "..", "callmodel"))
sys.path.insert(0, os.path.join(CODE, "..", "human_coding"))
import call_properties as CP  # noqa: E402

ROOT = os.path.abspath(os.path.join(CODE, "..", "..", ".."))
LOCAL = os.path.join(ROOT, "data", "local", "coxing")
FIELDS = ["target", "stage", "action", "arousal"]


def alpha(units, metric="nominal"):
    """Krippendorff's alpha from {unit: [values]}; units with fewer than two values are ignored."""
    units = [v for v in units.values() if len(v) >= 2]
    if not units:
        return None
    vals = [x for v in units for x in v]
    n = len(vals)
    d = (lambda a, b: 0.0 if a == b else 1.0) if metric == "nominal" else (lambda a, b: (a - b) ** 2)
    do = sum(sum(d(a, b) for i, a in enumerate(v) for j, b in enumerate(v) if i != j) / (len(v) - 1) for v in units) / n
    de = sum(d(a, b) for i, a in enumerate(vals) for j, b in enumerate(vals) if i != j) / (n * (n - 1))
    return None if de == 0 else round(1 - do / de, 3)


def units(pairs):
    u = {}
    for k, v in pairs:
        u.setdefault(k, []).append(v)
    return u


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    src = sys.argv[1]
    data = json.load(open(src, encoding="utf-8"))
    raters = {r["username"]: r for r in data["raters"]}
    rows = data["ratings"]
    K = json.load(open(os.path.join(LOCAL, "rating_site", "site_key.json")))
    codemap = K["codes"]
    hkey = json.load(open(os.path.join(LOCAL, "human_coding", "key.json")))["items"]
    hlog = os.path.join(LOCAL, "human_coding", "log_text.jsonl")
    human = {r["id"]: r["codes"] for r in map(json.loads, open(hlog, encoding="utf-8"))} if os.path.exists(hlog) else {}
    cache = {}

    def machine(item):
        sid, i = codemap[item]
        if sid not in cache:
            cache[sid] = CP.load_codes(os.path.join(LOCAL, "experiment", "codes_%s.txt" % sid))
        return CP.code_of(i, cache[sid])

    num = lambda v: int(float(v or 0))
    text = [r for r in rows if r["kind"] == "text" and not num(r["is_check"]) and not num(r["is_repeat"])]
    val = lambda r, f: str(r["targets"]).split("|")[0] if f == "target" else r[f]
    out = {"n_ratings": len(rows), "alpha": {}, "raters": {}}
    for scope in ["all"] + sorted({raters[r["username"]]["role"] for r in text if r["username"] in raters}):
        sel = [r for r in text if scope == "all" or raters.get(r["username"], {}).get("role") == scope]
        out["alpha"][scope] = {f: alpha(units([(r["item"], val(r, f)) for r in sel])) for f in FIELDS}
    voice = [r for r in rows if r["kind"] == "voice" and not num(r["is_repeat"])]
    out["alpha"]["all"]["voice (interval)"] = alpha(units([(r["item"], float(r["rating"])) for r in voice]), "interval")
    print("alpha:", json.dumps(out["alpha"]))
    by_item = units([(r["item"], r) for r in text])
    for user, info in raters.items():
        mine = [r for r in text if r["username"] == user]
        res = dict(role=info.get("role"), n_text=len(mine), devices=len([d for d in str(info.get("devices", "")).split(",") if d]))
        for f in FIELDS:
            loo, hum, mac = [], [], []
            for r in mine:
                others = [val(o, f) for o in by_item[r["item"]] if o["username"] != user]
                if others:
                    top, n = Counter(others).most_common(1)[0]
                    if n > len(others) / 2:
                        loo.append(val(r, f) == top)
                if r["item"] in human and r["item"] in hkey:
                    hv = human[r["item"]]
                    hum.append(val(r, f) == (hv["target"] if f == "target" else hv[f]))
                mv = machine(r["item"])
                if f == "target" or mv["target"] not in ("count", "other"):
                    mac.append(val(r, f) == mv[f])
            res[f] = {k: (float(np.mean(v)) if v else None) for k, v in (("vs_consensus", loo), ("vs_human", hum), ("vs_machine", mac))}
        checks = [r for r in rows if r["username"] == user and num(r["is_check"])]
        target_of = lambda item: "count" if machine(item)["target"] == "count" else "info"
        acc = float(np.mean([str(r["targets"]).split("|")[0] == target_of(r["item"]) for r in checks])) if checks else None
        days = len({r["day"] for r in rows if r["username"] == user})
        res.update(check_accuracy=acc, n_checks=len(checks), days=days, flagged=bool(acc is not None and days >= 5 and acc < 0.5))
        out["raters"][user] = res
        print("%-20s %-9s n=%4d checks %s devices %d  target vs consensus/human/machine %s" % (
            user, res["role"] or "-", len(mine), "-" if acc is None else "%.0f%%" % (100 * acc), res["devices"],
            "/".join("-" if res["target"][k] is None else "%.2f" % res["target"][k] for k in ("vs_consensus", "vs_human", "vs_machine"))))
    mean = {k: float(np.mean([float(r["rating"]) for r in v])) for k, v in units([(r["item"], r) for r in voice]).items()}
    vlog = os.path.join(LOCAL, "human_coding", "log_voice.jsonl")
    hv = {r["id"]: r["rating"] for r in map(json.loads, open(vlog, encoding="utf-8"))} if os.path.exists(vlog) else {}
    common = [k for k in mean if k in hv]
    out["voice"] = dict(clips=len(mean), spearman_with_human=spearman([mean[k] for k in common], [hv[k] for k in common]))
    try:
        import analyse as HC                                   # the human coding tool's acoustic index
        idx = HC.acoustic_index({tuple(codemap[k]) for k in mean})
        pairs = [(mean[k], idx.get(tuple(codemap[k]))) for k in mean]
        pairs = [(x, y) for x, y in pairs if y is not None and np.isfinite(y)]
        out["voice"]["spearman_with_acoustic_index"] = spearman([x for x, _ in pairs], [y for _, y in pairs])
    except Exception as e:
        out["voice"]["spearman_with_acoustic_index"] = None
        out["voice"]["note"] = "acoustic index not computed: %s" % e
    print("voice:", json.dumps(out["voice"]))
    dst = os.path.join(os.path.dirname(os.path.abspath(src)), "analysis_" + os.path.basename(src))
    json.dump(out, open(dst, "w"), indent=1)
    print("written", dst)


if __name__ == "__main__":
    main()
