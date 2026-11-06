"""Do the two coxswains' calls get different responses? A heterogeneity test for the pooled analysis.

    .venv/Scripts/python.exe research/coxing/experiment/coxswain_difference.py [COURSE=1 in the environment]

The primary analysis pools every piece (piece fixed effects absorb crew, boat and course). Whether that pooling
hides a difference between coxswains is tested here, class by class: the mean response after one coxswain's calls
of a class minus the mean after the other's, against the study's shift null (each piece's call strokes shifted
together, 2000 shifts), whose spread is the sampling spread of the difference when neither coxswain's calls have
an effect. Coxswain A: the first coxswain's pieces; B: the second's (race_07). Classes and outcomes are the paper's
(target channels; announced pushes at strokes 8-12 from the announcement). Reads episodes.csv (run_experiment.py,
all pieces); writes coxswain_difference[_course].json.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "research", "callmodel"))
import call_properties as CP  # noqa: E402

EXP = os.path.join(ROOT, "data", "local", "coxing", "experiment")
SDIR = os.path.join(ROOT, "data", "local", "coxing", "sessions")
COX_B = ["race_07"]
COURSE = os.environ.get("COURSE") == "1"
ANN = re.compile(r"in two|take a\b|take (?:\d+|five|ten|twenty|three)|\bready\b|give me (?:\d+|five|ten)|last 10|last ten", re.I)
CLASSES = [  # label, selection, channel, h0, h1, scale (to report units)
    ("power calls, speed 3-5 (mm/s)", lambda d: (d.target == "power") & (d.stage == "dec"), "speed", 3, 5, 1000),
    ("power calls, rate 1-3 (spm)", lambda d: (d.target == "power") & (d.stage == "dec"), "rate", 1, 3, 1),
    ("announced pushes, speed 8-12 (mm/s)", lambda d: (d.target == "power") & (d.stage == "dec") & d.text.str.contains(ANN),
     "speed", 8, 12, 1000),
    ("rate calls, rate 1-3 (spm)", lambda d: (d.target == "rate") & (d.stage == "dec"), "rate", 1, 3, 1),
    ("rate calls, speed 3-5 (mm/s)", lambda d: (d.target == "rate") & (d.stage == "dec"), "speed", 3, 5, 1000),
    ("technique calls, speed 3-5 (mm/s)", lambda d: d.target.isin(["catch", "finish", "body", "set", "together"]) & (d.stage == "dec"),
     "speed", 3, 5, 1000),
    ("steering calls, speed 3-5 (mm/s)", lambda d: d.target == "steer", "speed", 3, 5, 1000),
    ("praise, speed 3-5 (mm/s)", lambda d: (d.stage == "ass") & (d.appraisal == "chal"), "speed", 3, 5, 1000),
    ("arousal raised, speed 3-5 (mm/s)", lambda d: d.arousal == "up", "speed", 3, 5, 1000),
    ("all calls, speed 3-5 (mm/s)", lambda d: d.target.notna(), "speed", 3, 5, 1000),
]


def main():
    D = pd.read_csv(os.path.join(EXP, "episodes.csv"))
    calls = D[D.target != "count"].reset_index(drop=True)
    sids = list(dict.fromkeys(calls.session))
    S = [CP.prepare(CP.load_session(os.path.join(SDIR, k)), course=COURSE) for k in sids]
    isB = np.array([k in COX_B for k in sids])
    out = {}
    print("%-38s %6s %6s  %9s %9s  %9s (sd)    p" % ("class", "n A", "n B", "A", "B", "B - A"))
    for lab, sel, ch, h0, h1, sc in CLASSES:
        d = calls[sel(calls)]
        jj = [d.loc[d.session == k, "j"].to_numpy() for k in sids]
        nA, nB = sum(len(j) for j, b in zip(jj, isB) if not b), sum(len(j) for j, b in zip(jj, isB) if b)
        if nA < 3 or nB < 3:
            continue

        def stat(q):
            a = np.concatenate([CP.gather([s], [j], ch, h0, h1) for s, j, b in zip(S, q, isB) if not b])
            b_ = np.concatenate([CP.gather([s], [j], ch, h0, h1) for s, j, b in zip(S, q, isB) if b])
            return np.array([np.nanmean(a), np.nanmean(b_), np.nanmean(b_) - np.nanmean(a)])
        og, mg, sg, pg = CP.permutation_test(stat, S, jj, 2000, 11)
        e = og - mg
        out[lab] = dict(n_A=nA, n_B=nB, A=float(e[0]), A_sd=float(sg[0]), B=float(e[1]), B_sd=float(sg[1]),
                        diff=float(e[2]), diff_sd=float(sg[2]), p=float(pg[2]))
        print("%-38s %6d %6d  %+9.2f %+9.2f  %+9.2f (%5.2f) %.3f" % (lab, nA, nB, sc * e[0], sc * e[1], sc * e[2],
                                                                     sc * sg[2], pg[2]))
    json.dump(out, open(os.path.join(EXP, "coxswain_difference%s.json" % ("_course" if COURSE else "")), "w"), indent=1)


if __name__ == "__main__":
    main()
