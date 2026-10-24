"""Second coder: a lexical rule coder, written from the codebook definitions without reference to the
hand codes, for target, action, focus and arousal. Cohen's kappa against the hand codes per field.

    .venv/Scripts/python.exe research/coxing/experiment/rulecoder.py
"""
import json
import os
import re
import sys

import numpy as np

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "research", "callmodel"))
import call_properties as CP  # noqa: E402

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "experiment"))   # its data and outputs (gitignored)
SDIR = os.path.join(ROOT, "data", "local", "coxing", "sessions")
SES = ["race_01", "practice_01", "practice_02", "race_04", "race_05", "practice_03", "practice_04", "practice_05", "practice_06", "practice_07", "practice_08", "race_06"]

TARGET_RULES = [  # first match wins; order follows the codebook's specificity
    ("steer", r"\b(starboards?|starbuds|starving|ports?|porties|rudder|turn|turning|corner|yield|line)\b"),
    ("info", r"(\b\d{2,4}\b|\bsplit\b|\bmeters?\b|\bstroke rate\b|\bhalfway\b|\bto go\b|\bfinish line\b|\bk\b)"),
    ("rate", r"\b(ratio|rate|rhythm|build|quick|paddle|settle|take it up|go up)\b"),
    ("length", r"\b(long|length|lengthen|reach|stride|suspend\w*|hang\w*)\b"),
    ("together", r"\b(together|timing|one boat|by myself)\b"),
    ("catch", r"\b(catch\w*|direct\w*|sharp|hook|connect|attack|decisive|blades? (in|prep)|bury|back (them|the blades) in)\b"),
    ("finish", r"\b(finish\w*|send\w*|puddles?|release|complete|back end|tap|clear)\b"),
    ("set", r"\b(flat|set|level|heights?)\b"),
    ("body", r"\b(sit|chest|chins?|body|bodies|swing\w*|hips|cores?|glutes|shoulders|breathe?\w*|eyes|tall|sequenc\w*)\b"),
    ("power", r"\b(press\w*|legs?|push\w*|drive\w*|power\w*|pull|explode|stand\w*|bring it|harder|ten|10|20|effort|burn)\b"),
    ("tactic", r"\b(pass\w*|ahead|behind|walk\w*|gain\w*|clos\w*|them|seat|boats?)\b"),
    ("morale", r"\b(good|yes|yeah|yep|nice|got (this|it)|capable|let's go|come on|excellent|flying|hell|well done|that's it|there you go)\b"),
]
ACTION = r"^(let's|let us|i want|i need|we need|get|sit|swing|send|press|push|drive|hold|stay|keep|bring|take|find|lift|drop|move|reach|hang|suspend|squeeze|breathe|build|stand|connect|hook|attack|explode|commit|complete|back|tap|set|lock|swinging|sending|pressing|catch|long|chest|chins|flat|legs|hips|big|direct|sharp|quick)\b"
INT = r"\b(legs?|body|bodies|hands|chest|chins?|core|cores|glutes|shoulders|hips|knees|feet|lats|breath|breathe|breathing|arms|eyes|hamstrings?)\b"
EXT = r"\b(blades?|water|boat|puddles?|oars?|handle|run|splash|stern|bow)\b"
UP = r"\b(now|go|big|explode|attack|push|hard|harder|faster|black ?out|angry|hungry|burn|everything|power|hell|take it|move)\b"
DOWN = r"\b(relax|breathe|breathing|easy|calm|settle|save|smooth|steady|controlled|paddle|overcook|burn out)\b"


def code(text):
    t = text.lower()
    target = "other"
    for name, rx in TARGET_RULES:
        if re.search(rx, t):
            target = name
            break
    action = "y" if re.search(ACTION, t.strip()) else "n"
    fi, fe = bool(re.search(INT, t)), bool(re.search(EXT, t))
    focus = "int" if fi and not fe else ("ext" if fe and not fi else ("int" if fi else "-"))
    arousal = "down" if re.search(DOWN, t) else ("up" if re.search(UP, t) else "-")
    return dict(target=target, action=action, focus=focus, arousal=arousal)


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    ix = {c: k for k, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        M[ix[x], ix[y]] += 1
    n = M.sum()
    po = np.trace(M) / n
    pe = (M.sum(0) * M.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe), po


if __name__ == "__main__":
    hand, rule = {f: [] for f in ("target", "action", "focus", "arousal")}, {f: [] for f in ("target", "action", "focus", "arousal")}
    for sid in SES:
        phrases = json.load(open(os.path.join(SDIR, sid, "phrases.json"), encoding="utf-8"))
        codes = CP.load_codes(os.path.join(HERE, "codes_%s.txt" % sid))
        for p in phrases:
            if p["i"] not in codes or codes[p["i"]]["target"] in ("other", "count"):
                continue
            r = code(p["text"])
            for f in hand:
                hv = codes[p["i"]][f]
                hand[f].append("-" if hv in ("-", "mix") and f == "focus" and hv == "-" else hv)
                rule[f].append(r[f])
    out = {}
    for f in hand:
        kp, po = kappa(hand[f], rule[f])
        out[f] = dict(kappa=round(kp, 3), agreement=round(po, 3), n=len(hand[f]))
        print("%-8s kappa %.2f  raw agreement %.0f%%  (n=%d phrases)" % (f, kp, 100 * po, len(hand[f])))
    json.dump(out, open(os.path.join(HERE, "reliability.json"), "w"), indent=1)
