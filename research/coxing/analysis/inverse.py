"""Inverse clustering: group the boat's behaviour first, then look for calls.

Every analysis so far starts from the calls and asks what follows. This reverses it.
Each stroke is described by what the boat does over the following 12 strokes -- its
local response shape -- and those shapes are clustered without any reference to speech.
Only afterwards do we ask whether the clusters differ in what the coxswain was saying
when each one began.

If a call type reliably produced a characteristic response, the strokes carrying that
call should concentrate in the cluster with that shape.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, "elite")
from responses import CATS, hits, NS, R, detrend

H = 12
v = detrend(R["v"]); v = v - v.mean()
sr = detrend(R["sr"]); sr = sr - sr.mean()

# local response shape: speed and rate over the next H strokes, z-scored per channel
seg = []
for i in range(NS - H):
    seg.append(np.concatenate([v[i:i + H] / v.std(), sr[i:i + H] / sr.std()]))
X = np.array(seg)
print("%d stroke-windows of %d strokes, %d features (speed + rate trajectories)"
      % (X.shape[0], H, X.shape[1]))

rng = np.random.default_rng(5)
for K in (3, 4, 6):
    lab = KMeans(K, n_init=10, random_state=0).fit_predict(X)
    print()
    print("k=%d cluster sizes: %s" % (K, np.bincount(lab).tolist()))
    print("   mean speed change over the window, by cluster: %s"
          % ["%+.4f" % (v[np.flatnonzero(lab == c) + H - 1].mean() -
                        v[np.flatnonzero(lab == c)].mean()) for c in range(K)])
    print("   %-10s %s   chi2-style association" % ("category", " ".join("cl%d" % c for c in range(K))))
    for cat in CATS:
        y = (hits[cat][:NS - H] > 0).astype(int)
        if y.sum() < 12: continue
        obs = np.array([y[lab == c].mean() for c in range(K)])
        # null: rotate the call series, preserving its spacing, recompute the spread
        stat = obs.max() - obs.min()
        null = []
        for _ in range(300):
            sh = int(rng.integers(20, NS - H - 20))
            ys = np.roll(y, sh)
            o = np.array([ys[lab == c].mean() for c in range(K)])
            null.append(o.max() - o.min())
        p = float(np.mean(np.array(null) >= stat))
        star = "*" if p < 0.05 else " "
        print("   %-10s %s   spread %.3f  p=%.3f%s"
              % (cat, " ".join("%.2f" % x for x in obs), stat, p, star))
print()
print("p is the chance of a spread this large across clusters when the call series is")
print("circularly shifted, which keeps its density and spacing and breaks only its")
print("alignment with the boat's behaviour.")
