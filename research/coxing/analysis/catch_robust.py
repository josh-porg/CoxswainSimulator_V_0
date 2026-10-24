"""Robustness of the catch result.

A finding at lag 20 that exists only under one detrending choice is an artifact of the
detrending, not a call effect. The 61-stroke moving average leaves residual structure
on exactly that scale, so the window is varied, and a linear detrend included as a
version with no local smoothing at all.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np

sys.path.insert(0, "elite")
import vtt
from functional import FUNC, LOOKUP
from oos import blocked_oos, hits, NS, LAGS

R = np.load("race.npz")
t, v = R["t"], R["v"]
cat = hits["catch"]
X = np.column_stack([np.roll(cat, L) for L in LAGS])
X = X - X.mean(axis=0)


def resid(mode):
    if mode == "linear":
        A = np.column_stack([np.ones(NS), np.arange(NS)])
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    w = int(mode)
    k = np.ones(w) / w
    tr = np.convolve(np.pad(v, (w // 2, w // 2), mode="edge"), k, mode="valid")[:NS]
    return v - tr


rng = np.random.default_rng(5)
print("%-10s %12s %12s %10s" % ("detrend", "OOS R2", "null 95th", "peak mm/s"))
print("-" * 48)
for mode in ("31", "61", "101", "151", "linear"):
    yy = resid(mode); yy = yy - yy.mean()
    obs = blocked_oos(X, yy)
    null = []
    for _ in range(200):
        sh = np.roll(cat, int(rng.integers(26, NS - 26)))
        Xp = np.column_stack([np.roll(sh, L) for L in LAGS])
        null.append(blocked_oos(Xp - Xp.mean(axis=0), yy))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    b = Vt.T @ ((S / (S ** 2 + 96.2)) * (U.T @ yy))
    pk = b[int(np.argmax(np.abs(b)))]
    print("%-10s %+12.4f %+12.4f %10.2f  (lag %d)"
          % (mode, obs, np.percentile(null, 95), 1000 * pk, int(np.argmax(np.abs(b)))))

# influence: how much does any single call carry the result?
yy = resid("61"); yy = yy - yy.mean()
ev = np.flatnonzero(cat > 0)
base = blocked_oos(X, yy)
drops = []
for e in ev:
    c2 = cat.copy(); c2[e] = 0
    X2 = np.column_stack([np.roll(c2, L) for L in LAGS])
    drops.append(blocked_oos(X2 - X2.mean(axis=0), yy))
drops = np.array(drops)
print()
print("leave-one-call-out over %d catch calls: OOS R2 %+.4f -> min %+.4f, max %+.4f"
      % (len(ev), base, drops.min(), drops.max()))
print("   worst single call costs %.1f%% of the score" % (100 * (base - drops.min()) / base))
