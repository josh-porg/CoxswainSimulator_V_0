"""Outside labels from a practitioner guide: does an expert's sorting of calls by type agree with the target codes?

    .venv/Scripts/python.exe research/coxing/experiment/guide_validation.py

Design fixed in the CODEBOOK addendum of 8 Nov 2026 ("a practitioner guide's call types as outside labels"):
1. the lexical rule coder on the guide's printed phrases, agreement with the heading's mapped targets vs a permutation
   null (headings shuffled across phrases);
2. a nearest-guide-phrase classifier (TF-IDF cosine) on the study's call phrases, agreement with the primary target vs a
   permutation null (headings shuffled across guide phrases); all calls the guide covers, and close matches;
3. the first coxswain's repertoire: close matches for guide phrases the study lead marked vs unmarked (Fisher).
Inputs (private): data/local/coxing/literature/snarky_guide/phrases.tsv, sessions and primary codes.
Outputs (private): data/local/coxing/experiment/guide/ (results.json, nearest.tsv). Earlier results are untouched.
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import fisher_exact
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import rulecoder as RC  # noqa: E402
CP = RC.CP

LOCAL = os.path.abspath(os.path.join(CODE, "..", "..", "..", "data", "local", "coxing"))
GUIDE = os.path.join(LOCAL, "literature", "snarky_guide", "phrases.tsv")
SDIR = os.path.join(LOCAL, "sessions")
EXP = os.path.join(LOCAL, "experiment")
OUT = os.path.join(EXP, "guide")

MAP = {
    "Catches": {"catch"},
    "Drive": {"power"},
    "Finishes": {"finish"},
    "Recovery/Ratio": {"rate"},
    "Set/Timing": {"set", "together"},
    "Consistency/Body Position/Balance": {"body", "set"},
    "Rhythm/Swing": {"rate", "together", "body"},
    "Generic Motivation/Power": {"morale", "power", "tactic"},
}
COVERED = set().union(*MAP.values())
OUTSIDE = {"steer", "info", "length"}
PIECES = ["race_01", "practice_01", "practice_02", "race_04", "race_05", "practice_03", "practice_04", "practice_05",
          "practice_06", "practice_07", "practice_08", "race_06", "race_07", "race_08", "race_09"]
SECOND_COXSWAIN = {"race_07"}
N_PERM = 2000
CLOSE = 0.5


def load_guide():
    rows = [r for r in csv.DictReader((l for l in open(GUIDE, encoding="utf-8") if not l.startswith("#")), delimiter="\t")]
    return [r for r in rows if r["source"] == "guide"], [r for r in rows if r["source"] != "guide"]


def load_calls():
    calls = []
    for sid in PIECES:
        s = CP.load_session(os.path.join(SDIR, sid))
        codes = CP.load_codes(os.path.join(EXP, "codes_%s.txt" % sid))
        for p in s["phrases"]:
            if not s["t"][0] <= p["t"] <= s["t"][-1]:
                continue
            calls.append(dict(sid=sid, i=p["i"], text=p["text"], target=CP.code_of(p["i"], codes)["target"]))
    return calls


def vectoriser(texts):
    w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, sublinear_tf=True).fit(texts)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True).fit(texts)
    return lambda xs: hstack([w.transform(xs), c.transform(xs)]).tocsr()


def perm_p(observed, null):
    null = np.asarray(null)
    return float((1 + np.sum(null >= observed)) / (1 + len(null))), float(np.mean(null))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261108)
    guide, hand = load_guide()
    heads = [g["section"] for g in guide]
    out = dict(n_guide=len(guide), n_handwritten=len(hand), by_heading=dict(Counter(heads)))

    # 1. rule coder on the guide's phrases
    rule = [RC.code(g["phrase"])["target"] for g in guide]
    hit = lambda hs: np.mean([t in MAP[h] for t, h in zip(rule, hs)])
    obs = float(hit(heads))
    null = [hit(rng.permutation(heads)) for _ in range(N_PERM)]
    p, mu = perm_p(obs, null)
    per = {}
    for h in MAP:
        ix = [k for k, x in enumerate(heads) if x == h]
        per[h] = dict(n=len(ix), agree=float(np.mean([rule[k] in MAP[h] for k in ix])),
                      rule_targets=dict(Counter(rule[k] for k in ix).most_common()))
    out["rule_coder_on_guide"] = dict(agreement=obs, chance=mu, p=p, by_heading=per)
    print("1. rule coder on the guide: agreement %.2f (chance %.2f, p %.4f)" % (obs, mu, p))
    for h, v in per.items():
        print("   %-36s n=%3d  agree %.2f  %s" % (h, v["n"], v["agree"], v["rule_targets"]))

    # 2. nearest guide phrase for the study's calls
    calls = load_calls()
    coded = [c for c in calls if c["target"] not in ("count", "other")]
    inside = [c for c in coded if c["target"] in COVERED]
    vec = vectoriser([g["phrase"] for g in guide] + [c["text"] for c in coded])
    G, C = vec([g["phrase"] for g in guide]), vec([c["text"] for c in inside])
    S = cosine_similarity(C, G)
    nn, sim = S.argmax(1), S.max(1)
    tg = [c["target"] for c in inside]

    def agree(hs, mask):
        return float(np.mean([tg[k] in MAP[hs[nn[k]]] for k in np.flatnonzero(mask)])) if mask.any() else None

    res = {}
    for name, mask in (("all", np.ones(len(inside), bool)), ("close", sim >= CLOSE)):
        o = agree(heads, mask)
        nl = [agree(list(rng.permutation(heads)), mask) for _ in range(N_PERM)]
        pp, mm = perm_p(o, nl)
        res[name] = dict(n=int(mask.sum()), agreement=o, chance=mm, p=pp)
        print("2. guide classifier, %-5s calls: n=%4d agreement %.2f (chance %.2f, p %.4f)" % (name, mask.sum(), o, mm, pp))
    conf = Counter((tg[k], heads[nn[k]]) for k in range(len(inside)))
    res["by_target"] = {t: dict(n=sum(v for (a, _), v in conf.items() if a == t),
                                agree=float(np.mean([tg[k] in MAP[heads[nn[k]]] for k in range(len(inside)) if tg[k] == t])),
                                headings=dict(Counter(heads[nn[k]] for k in range(len(inside)) if tg[k] == t).most_common()))
                        for t in sorted(set(tg))}
    for t, v in res["by_target"].items():
        print("   %-9s n=%4d agree %.2f  nearest headings %s" % (t, v["n"], v["agree"], dict(list(v["headings"].items())[:3])))
    res["outside_guide_types"] = dict(n=sum(c["target"] in OUTSIDE for c in coded), of=len(coded),
                                      by_target=dict(Counter(c["target"] for c in coded if c["target"] in OUTSIDE)))
    print("   calls outside the guide's types: %d of %d coded calls %s" % (
        res["outside_guide_types"]["n"], len(coded), res["outside_guide_types"]["by_target"]))
    out["guide_classifier_on_calls"] = res
    with open(os.path.join(OUT, "nearest.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sid", "i", "call", "primary_target", "nearest_guide_phrase", "heading", "cosine", "agree"])
        for k, c in enumerate(inside):
            g = guide[nn[k]]
            w.writerow([c["sid"], c["i"], c["text"], c["target"], g["phrase"], g["section"], "%.3f" % sim[k], int(c["target"] in MAP[g["section"]])])

    # 3. the first coxswain's repertoire and the study lead's marks
    first = [c for c in coded if c["sid"] not in SECOND_COXSWAIN]
    F = vec([c["text"] for c in first])
    best = cosine_similarity(G, F).max(1)
    marked = np.array([g["mark"] != "-" for g in guide])
    used = best >= CLOSE
    table = [[int((used & marked).sum()), int((~used & marked).sum())], [int((used & ~marked).sum()), int((~used & ~marked).sum())]]
    odds, pf = fisher_exact(table)
    out["repertoire_first_coxswain"] = dict(
        n_calls=len(first), guide_phrases_with_close_match=int(used.sum()),
        marked=dict(n=int(marked.sum()), with_match=table[0][0]), unmarked=dict(n=int((~marked).sum()), with_match=table[1][0]),
        fisher_odds_ratio=float(odds), fisher_p=float(pf),
        by_heading={h: dict(n=int(sum(x == h for x in heads)), with_match=int(sum(used[k] for k, x in enumerate(heads) if x == h)))
                    for h in MAP})
    print("3. first coxswain: %d of %d guide phrases have a close match; marked %d/%d vs unmarked %d/%d (Fisher p %.3f, OR %.2f)" % (
        used.sum(), len(guide), table[0][0], marked.sum(), table[1][0], (~marked).sum(), pf, odds))
    for h, v in out["repertoire_first_coxswain"]["by_heading"].items():
        print("   %-36s %2d of %2d" % (h, v["with_match"], v["n"]))
    json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("written", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
