"""Do the LLM's source labels correspond to anything physical?

The labeller inferred, from wording alone, what information each call was based on.
That inference can be tested against measurement it never saw. Calls attributed to boat
feel should be better predicted by the vestibular signal than by the cox-box display,
and calls attributed to the cox-box should show the reverse. A crossed design gives a
double dissociation, which no single comparison can fake.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import sys

import numpy as np

sys.path.insert(0, "elite")
from vestib import VEST
from whycall import POSITION, BOX, blocked_auc

R = np.load("race.npz")
t = R["t"]; NS = len(t)
L = json.load(open("labels_llm.json"))["race1"]

src = {}
for s in ("boat_feel", "coxbox", "race_position", "plan"):
    a = np.zeros(NS)
    for d in L:
        if d["source"] == s:
            a[int(np.argmin(np.abs(t - (d["t"] + 1.0))))] += 1
    src[s] = (a > 0).astype(int)
    print("%-14s %d strokes carry such a call" % (s, src[s].sum()))

rng = np.random.default_rng(5)
print()
print("%-14s %5s | %-22s | %-22s" % ("call source", "n", "predicted by FELT", "predicted by BOX"))
print("%-14s %5s | %8s %8s %4s | %8s %8s %4s"
      % ("", "", "AUC", "null95", "sig", "AUC", "null95", "sig"))
print("-" * 72)
out = {}
for s, y in src.items():
    row = []
    for nm, X in (("FELT", VEST), ("BOX", BOX)):
        Z = np.column_stack([POSITION, X])
        a = blocked_auc(Z, y)
        null = []
        for _ in range(120):
            sh = int(rng.integers(30, NS - 30))
            null.append(blocked_auc(np.column_stack([POSITION, np.roll(X, sh, axis=0)]), y))
        thr = float(np.nanpercentile(null, 95))
        row += [a, thr, "*" if a > thr else " "]
    out[s] = row
    print("%-14s %5d | %8.3f %8.3f %4s | %8.3f %8.3f %4s"
          % (s, y.sum(), row[0], row[1], row[2], row[3], row[4], row[5]))
print()
bf, cb = out["boat_feel"], out["coxbox"]
print("double dissociation check:")
print("   boat_feel calls: FELT %+.3f vs BOX %+.3f   (difference %+.3f)"
      % (bf[0], bf[3], bf[0] - bf[3]))
print("   coxbox calls   : FELT %+.3f vs BOX %+.3f   (difference %+.3f)"
      % (cb[0], cb[3], cb[0] - cb[3]))
print("   interaction (the thing that would be hard to fake): %+.3f"
      % ((bf[0] - bf[3]) - (cb[0] - cb[3])))
