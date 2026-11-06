"""A hypothesis-blind second coding of one property, arousal, for every coded session.

    .venv/Scripts/python.exe research/coxing/experiment/blind_arousal.py pack
    .venv/Scripts/python.exe research/coxing/experiment/blind_arousal.py compare
    .venv/Scripts/python.exe research/coxing/experiment/blind_arousal.py voice     # which coder the voice agrees with

The primary coder knew the review's hypotheses, including H19 (calls that raise arousal raise speed), and
the lexical rule coder (rulecoder.py) reads words without context. The blind coder is a separate coder given
only the codebook's arousal rule and each session's transcript in order: nothing about the hypotheses, the
boat, the other codes or any result.

``pack`` writes data/local/coxing/experiment/blind_arousal/input_<session>.txt, one phrase per line in order:
``i|CODE|text`` for phrases the primary coder coded as calls (target not count/other: the unit the arousal
field applies to) and ``i|-|text`` for counts and incidental talk, kept for context only. The coder writes
arousal_<session>.txt beside it, one ``i up|down|-`` line per CODE phrase.

``compare`` reports, per session and pooled, the agreement (Cohen's kappa, and the share of calls each coder
codes up and down) between the primary coder, the lexical coder and the blind coder; written to
blind_arousal/agreement.json. run_experiment.py uses the blind codes with ALT_AROUSAL=all ALT_SRC=blind.
"""
import json
import os
import sys

import numpy as np

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "research", "callmodel"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import call_properties as CP  # noqa: E402

EXP = os.path.join(ROOT, "data", "local", "coxing", "experiment")
OUT = os.path.join(EXP, "blind_arousal")
SDIR = os.path.join(ROOT, "data", "local", "coxing", "sessions")
SES = ["race_01", "practice_01", "practice_02", "race_04", "race_05", "practice_03", "practice_04", "practice_05",
       "practice_06", "practice_07", "practice_08", "race_06", "race_08", "race_09", "race_07"]

RULE = """Arousal (the codebook's rule, verbatim):
  up   = urges intensity, urgency or aggression ("now!", "go", "attack", "hit it", "give me everything")
  down = calms, settles, conserves ("relax", "settle", "easy", "save it", "breathe")
  -    = neither
"""


def load_blind(sid):
    p = os.path.join(OUT, "arousal_%s.txt" % sid)
    out = {}
    for line in open(p, encoding="utf-8"):
        f = line.split()
        if len(f) == 2 and not line.startswith("#") and f[1] in ("up", "down", "-"):
            out[int(f[0])] = f[1]
    return out


def pack():
    os.makedirs(OUT, exist_ok=True)
    tot = 0
    for sid in SES:
        ph = json.load(open(os.path.join(SDIR, sid, "phrases.json"), encoding="utf-8"))
        codes = CP.load_codes(os.path.join(EXP, "codes_%s.txt" % sid))
        n = 0
        with open(os.path.join(OUT, "input_%s.txt" % sid), "w", encoding="utf-8") as f:
            for p in ph:
                c = codes.get(p["i"])
                call = c is not None and c["target"] not in ("count", "other")
                n += call
                f.write("%d|%s|%s\n" % (p["i"], "CODE" if call else "-", p["text"]))
        tot += n
        print("%-12s %4d phrases, %4d to code" % (sid, len(ph), n))
    open(os.path.join(OUT, "RULE.txt"), "w", encoding="utf-8").write(RULE)
    print("total to code:", tot)


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    ix = {c: k for k, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        M[ix[x], ix[y]] += 1
    n = M.sum()
    po = np.trace(M) / n
    pe = (M.sum(0) * M.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def compare():
    import rulecoder
    res, H, R, B = {}, [], [], []
    first = None
    for sid in SES:
        if sid == "race_07":                      # the second coxswain: pool the first coxswain's pieces before her
            first = (list(H), list(R), list(B))
        ph = {p["i"]: p["text"] for p in json.load(open(os.path.join(SDIR, sid, "phrases.json"), encoding="utf-8"))}
        codes = CP.load_codes(os.path.join(EXP, "codes_%s.txt" % sid))
        bl = load_blind(sid)
        ii = [i for i, c in codes.items() if c["target"] not in ("count", "other")]
        miss = [i for i in ii if i not in bl]
        h = [codes[i]["arousal"] for i in ii if i in bl]
        r = [rulecoder.code(ph[i])["arousal"] for i in ii if i in bl]
        b = [bl[i] for i in ii if i in bl]
        H += h
        R += r
        B += b
        share = lambda v, k: float(np.mean([x == k for x in v]))
        res[sid] = dict(n=len(b), missing=len(miss), kappa_hand_blind=kappa(h, b), kappa_rule_blind=kappa(r, b),
                        kappa_hand_rule=kappa(h, r), up=dict(hand=share(h, "up"), rule=share(r, "up"), blind=share(b, "up")),
                        down=dict(hand=share(h, "down"), rule=share(r, "down"), blind=share(b, "down")))
        v = res[sid]
        print("%-12s n=%4d (missing %d)  kappa hand-blind %.2f  rule-blind %.2f  hand-rule %.2f   up %% hand %.0f rule %.0f"
              " blind %.0f" % (sid, v["n"], v["missing"], v["kappa_hand_blind"], v["kappa_rule_blind"], v["kappa_hand_rule"],
                               100 * v["up"]["hand"], 100 * v["up"]["rule"], 100 * v["up"]["blind"]))
    if first:
        h, r, b = first
        share = lambda v, k: float(np.mean([x == k for x in v]))
        res["first_coxswain"] = dict(n=len(b), kappa_hand_blind=kappa(h, b), kappa_rule_blind=kappa(r, b),
                                     kappa_hand_rule=kappa(h, r), up=dict(hand=share(h, "up"), rule=share(r, "up"),
                                                                          blind=share(b, "up")))
        print("first coxswain n=%d  kappa hand-blind %.2f  rule-blind %.2f  hand-rule %.2f  up %% %.0f/%.0f/%.0f" % (
            len(b), kappa(h, b), kappa(r, b), kappa(h, r), 100 * share(h, "up"), 100 * share(r, "up"), 100 * share(b, "up")))
    res["pooled"] = dict(n=len(B), kappa_hand_blind=kappa(H, B), kappa_rule_blind=kappa(R, B), kappa_hand_rule=kappa(H, R))
    print("pooled n=%d  kappa hand-blind %.2f  rule-blind %.2f  hand-rule %.2f" % (
        len(B), res["pooled"]["kappa_hand_blind"], res["pooled"]["kappa_rule_blind"], res["pooled"]["kappa_hand_rule"]))
    json.dump(res, open(os.path.join(OUT, "agreement.json"), "w"), indent=1)


def voice(suffix=""):
    """Which coder's arousal the voice agrees with: the vocal arousal index (run_experiment.py, CODEBOOK addendum of
    5 Nov), by agreement between the primary and blind coders. Reads vocal<suffix>.csv and
    vocal_blindarousal_all<suffix>.csv (runs with and without ALT_AROUSAL=all ALT_SRC=blind; suffix "" = all pieces)."""
    import pandas as pd
    h = pd.read_csv(os.path.join(EXP, "vocal%s.csv" % suffix))
    b = pd.read_csv(os.path.join(EXP, "vocal_blindarousal_all%s.csv" % suffix))
    m = h.merge(b[["session", "i", "arousal"]], on=["session", "i"], suffixes=("_hand", "_blind"))
    out = {}
    for lab, sel in (("both up", (m.arousal_hand == "up") & (m.arousal_blind == "up")),
                     ("primary only up", (m.arousal_hand == "up") & (m.arousal_blind != "up")),
                     ("blind only up", (m.arousal_hand != "up") & (m.arousal_blind == "up")),
                     ("neither", (m.arousal_hand != "up") & (m.arousal_blind != "up"))):
        out[lab] = dict(n=int(sel.sum()), vocal=float(m.vocal[sel].mean()))
        print("%-16s n %4d  vocal arousal index %+.2f sd" % (lab, sel.sum(), m.vocal[sel].mean()))
    json.dump(out, open(os.path.join(OUT, "voice%s.json" % suffix), "w"), indent=1)


if __name__ == "__main__":
    {"pack": pack, "compare": compare, "voice": voice}[sys.argv[1]](*sys.argv[2:])
