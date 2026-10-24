"""The catch result: is it real after correcting for the whole search, and does the
physics make sense?

Sixteen tests were run (8 categories x 2 blocks). The family-wise null is the
distribution of the LARGEST gain anywhere in that grid when the block being tested
carries no real timing, so it prices the entire search at once.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "elite")
from responses import CATS, hits, NS, R
from vestib import VEST, FN
from whycall import POSITION, BOX, blocked_auc, speaking, S

FELT = VEST
rng = np.random.default_rng(29)
P, B, F = POSITION[S], BOX[S], FELT[S]
ycatch = (hits["catch"][S] > 0).astype(int)
a1 = blocked_auc(np.column_stack([P, B]), ycatch)
obs = blocked_auc(np.column_stack([P, B, F]), ycatch) - a1
print("catch: FELT gain over POSITION+BOX = %+.3f (n=%d positives)" % (obs, ycatch.sum()))

# FAMILY-WISE: largest gain anywhere in the 8x2 grid under shifted blocks
mx = []
for _ in range(60):
    best = -9
    for c in CATS:
        y = (hits[c][S] > 0).astype(int)
        if y.sum() < 12: continue
        b0 = blocked_auc(P, y)
        sb = int(rng.integers(20, len(S) - 20))
        g1 = blocked_auc(np.column_stack([P, np.roll(B, sb, axis=0)]), y) - b0
        b1 = blocked_auc(np.column_stack([P, B]), y)
        sf = int(rng.integers(20, len(S) - 20))
        g2 = blocked_auc(np.column_stack([P, B, np.roll(F, sf, axis=0)]), y) - b1
        best = max(best, np.nanmax([g1, g2]))
    mx.append(best)
mx = np.array(mx)
p = float(np.mean(mx >= obs))
print("family-wise null over the whole 8x2 grid: 95th %+.3f, max %+.3f, p = %.3f"
      % (np.nanpercentile(mx, 95), np.nanmax(mx), p))

# which sensations carry it
m = make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=4000, C=0.3, class_weight="balanced"))
m.fit(np.column_stack([P, B, F]), ycatch)
co = m[-1].coef_[0][P.shape[1] + B.shape[1]:]
order = np.argsort(-np.abs(co))
print()
print("strongest FELT predictors of a catch call:")
for i in order[:8]:
    print("   %-16s %+.3f" % (FN[i], co[i]))
print()
# single-channel ablation: which channel family alone reproduces the gain
fam = {}
for key in ("surge", "sway", "heave", "wroll", "wpitch", "wyaw"):
    idx = [i for i, n in enumerate(FN) if n.startswith(key)]
    fam[key] = blocked_auc(np.column_stack([P, B, F[:, idx]]), ycatch) - a1
print("gain from each sensation alone (over POSITION+BOX):")
for k, gv in sorted(fam.items(), key=lambda x: -x[1]):
    print("   %-7s %+.3f" % (k, gv))
