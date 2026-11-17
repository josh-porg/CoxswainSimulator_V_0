"""A hypothesis-blind second coding of target and stage for every coded piece (CODEBOOK addendum of 16 Nov 2026).

    .venv/Scripts/python.exe research/coxing/experiment/blind_target.py pack
    .venv/Scripts/python.exe research/coxing/experiment/blind_target.py compare
    .venv/Scripts/python.exe research/coxing/experiment/blind_target.py response

The early power-call response depended on the primary coder's reading of which calls demand power. The blind coder is a
separate coder given only the codebook's target and stage rules and each piece's transcript in order: nothing about the
hypotheses, the boat, the other codes or any result.

``pack`` writes data/local/coxing/experiment/blind_target/input_<session>.txt, one phrase per line in order:
``i|CODE|text`` for every phrase that is not a bare count (the primary coder's "other" is not revealed) and ``i|-|text``
for bare counts, kept for context. The coder writes target_<session>.txt beside it: ``i target stage`` per CODE phrase.
``compare``: Cohen's kappa with the primary coder for target, stage and "directive power call"; agreement.json.
``response``: the power-call response with the blind coder's target and stage in the episodes; response.json.
"""
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import examiner_checks as X  # noqa: E402
CP = X.CP

EXP = X.EXP
SDIR = X.SDIR
OUT = os.path.join(EXP, "blind_target")
SES = X.SES
TARGETS = ["power", "rate", "length", "catch", "finish", "body", "set", "together", "steer", "info", "morale", "tactic", "count", "other"]
STAGES = ["inf", "ass", "dec", "-"]
RULE = """Code two things for every line marked CODE, from the words alone (the lines before and after are context).

TARGET: the variable the call addresses. One of:
  power    = effort, pressure, drive, legs/push as force
  rate     = stroke rate, rhythm, ratio, slide speed, build/settle
  length   = reach, long, length
  catch    = entry, blade in, catch timing
  finish   = release, extraction, hands away
  body     = posture, sit up, core, shoulders, breathing
  set      = balance, blade height, level
  together = timing across crew, "together", "with stroke"
  steer    = course, turns, side press
  info     = split, rate/position/distance/opponent reports, time left
  morale   = encouragement or belief without a rowing target
  tactic   = move on a crew, push to gain, respond to a crew
  count    = a stroke count
  other    = anything else that is not a call to the crew
A fragment that continues the previous phrase's call takes that call's target.

STAGE: one of
  inf = states a fact
  ass = evaluates the crew's performance ("better", "too high", "that's it")
  dec = tells the crew what to do
  -   = does not apply (target count or other)
"""


def load_blind(sid):
    out = {}
    for line in open(os.path.join(OUT, "target_%s.txt" % sid), encoding="utf-8"):
        f = line.split()
        if len(f) == 3 and not line.startswith("#") and f[1] in TARGETS and f[2] in STAGES:
            out[int(f[0])] = (f[1], f[2])
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
                call = p["i"] in codes
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
    T, S_, P, miss, n = [], [], [], 0, 0
    for sid in SES:
        codes = CP.load_codes(os.path.join(EXP, "codes_%s.txt" % sid))
        bl = load_blind(sid)
        for i, c in codes.items():
            n += 1
            if i not in bl:
                miss += 1
                continue
            T.append((c["target"], bl[i][0]))
            calls = c["target"] not in ("count", "other") and bl[i][0] not in ("count", "other")
            if calls:
                S_.append((c["stage"], bl[i][1]))
            P.append((c["target"] == "power" and c["stage"] == "dec", bl[i][0] == "power" and bl[i][1] == "dec"))
    res = dict(n=n, missing=miss, kappa_target=kappa(*zip(*T)), n_stage=len(S_), kappa_stage=kappa(*zip(*S_)),
               kappa_directive_power=kappa(*zip(*P)), primary_power=int(sum(a for a, _ in P)), blind_power=int(sum(b for _, b in P)),
               both_power=int(sum(a and b for a, b in P)))
    json.dump(res, open(os.path.join(OUT, "agreement.json"), "w"), indent=1)
    print("blind target coder: %d phrases (%d missing); kappa target %.2f, stage %.2f (n %d), directive power call %.2f; power phrases primary %d, blind %d, both %d" % (
        n, miss, res["kappa_target"], res["kappa_stage"], res["n_stage"], res["kappa_directive_power"], res["primary_power"], res["blind_power"], res["both_power"]))


def response():
    D = X.Data()
    prim = {sid: X.calls_of(D, sid, "power", *D.valid(sid)) for sid in SES}
    Db = X.Data()
    for sid in SES:
        bl = load_blind(sid)
        for i, c in Db.codes[sid].items():
            if i in bl:
                c["target"], c["stage"] = bl[i]
    blind = {sid: X.calls_of(Db, sid, "power", *Db.valid(sid)) for sid in SES}
    near = lambda a, b: np.array([len(b) > 0 and np.min(np.abs(b - j)) <= 1 for j in a], bool)
    sets = {"primary": prim, "blind": blind,
            "both": {s: prim[s][near(prim[s], blind[s])] for s in SES},
            "primary only": {s: prim[s][~near(prim[s], blind[s])] for s in SES},
            "blind only": {s: blind[s][~near(blind[s], prim[s])] for s in SES}}
    RNG = [D.valid(s) for s in SES]
    out = {}
    for name, js in sets.items():
        r = dict(n=int(sum(len(js[s]) for s in SES)))
        for tag, course in (("momentum", False), ("course", True)):
            for wn, w in (("3_5", (3, 5)), ("6_10", (6, 10)), ("template", "template")):
                r["%s_%s" % (tag, wn)] = X.both([X.window_stat(D.R(s, "speed", None, course), w) for s in SES], [js[s] for s in SES], RNG, 1600)
        out[name] = r
        a, b, c = r["momentum_3_5"], r["momentum_6_10"], r["momentum_template"]
        d, e = r["course_3_5"], r["course_6_10"]
        print("%-13s n=%3d  3-5 %+6.1f (sd %.1f) p %.3f / local %+6.1f p %.3f | 6-10 %+6.1f p %.3f / local %+6.1f p %.3f | template %+6.1f p %.3f | course 3-5 %+6.1f p %.3f / local %+6.1f p %.3f; 6-10 %+6.1f p %.3f" % (
            name, r["n"], 1000 * a["est"], 1000 * a["sd"], a["p"], 1000 * a["est_local"], a["p_local"], 1000 * b["est"], b["p"], 1000 * b["est_local"], b["p_local"],
            1000 * c["est"], c["p"], 1000 * d["est"], d["p"], 1000 * d["est_local"], d["p_local"], 1000 * e["est"], e["p"]))
    json.dump(out, open(os.path.join(OUT, "response.json"), "w"), indent=1)


if __name__ == "__main__":
    {"pack": pack, "compare": compare, "response": response}[sys.argv[1]]()
