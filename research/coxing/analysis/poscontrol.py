"""Positive control: can this pipeline detect anything at all?

A string of nulls is only informative if the method can find structure that is really
there. Tactical calls -- "220 to go", "we're walking" -- are by definition tied to
where the boat is on the course, so race position must predict them. If it does not,
the nulls elsewhere say nothing about coxing and everything about the pipeline.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np

sys.path.insert(0, "elite")
from responses import CATS, hits
from whycall import POSITION, blocked_auc, S

rng = np.random.default_rng(97)
print("%-10s %5s %8s %10s %9s %7s" % ("category", "n", "AUC(pos)", "null mean", "null 95th", "p"))
print("-" * 56)
for c in CATS:
    y = (hits[c][S] > 0).astype(int)
    if y.sum() < 12: continue
    obs = blocked_auc(POSITION[S], y)
    null = []
    for _ in range(200):
        sh = int(rng.integers(20, len(S) - 20))
        null.append(blocked_auc(np.roll(POSITION[S], sh, axis=0), y))
    null = np.array(null, float)
    p = float(np.nanmean(null >= obs))
    star = "*" if p < 0.05 else " "
    print("%-10s %5d %8.3f %10.3f %9.3f %6.3f%s"
          % (c, y.sum(), obs, np.nanmean(null), np.nanpercentile(null, 95), p, star))
print()
print("Shifting race position against the calls preserves its shape and destroys only")
print("its alignment, so this asks whether WHERE the boat is predicts what is said.")
