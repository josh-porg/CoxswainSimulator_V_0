"""What makes the coxswain call? The inverse problem.

Asking whether calls move the boat is badly posed at n=1: the effects are small and the
response is noisy. The reverse is far better posed. Calls are frequent, the boat state
is measured precisely, and the coxswain is a sensor-and-decision system whose inputs we
can enumerate: what the boat is doing, and what the cox-box is showing them.

The confound that decides this analysis is race position. Calls cluster by phase and so
does boat state, so a model given only "how far through the race are we" can predict
calls without knowing anything about the boat. Every model below is therefore scored
against a race-position baseline, and what is reported is what boat dynamics add on
top of it.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, "elite")
import vtt
from responses import FUNC, LOOKUP, CATS, hits, NS, t, R

v, sr, dps, d = R["v"], R["sr"], R["dps"], R["d"]


def dev(x, w=61):
    k = np.ones(w) / w
    return x - np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), k, mode="valid")[:len(x)]


def lagdiff(x, n):
    y = np.empty_like(x); y[:n] = 0; y[n:] = x[n:] - x[:-n]
    return y


def roll_sd(x, w):
    out = np.zeros(len(x))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = x[lo:i + 1].std()
    return out


pos = d / d.max()                                   # race position, the baseline
BASE = np.column_stack([pos, pos ** 2, pos ** 3, np.sin(np.pi * pos), np.cos(np.pi * pos)])

feat, names = [], []
for nm, x in (("v", v), ("sr", sr), ("dps", dps)):
    feat += [dev(x)];                names += ["%s dev" % nm]
    for n in (1, 3, 5):
        feat += [lagdiff(x, n)];     names += ["d%s/%d" % (nm, n)]
    feat += [roll_sd(dev(x), 5)];    names += ["%s rough5" % nm]
BOAT = np.column_stack(feat)
print("race-position baseline: %d cols; boat-state block: %d cols (%s)"
      % (BASE.shape[1], BOAT.shape[1], ", ".join(names)))

# everything is causal: predictors at stroke s use data up to s, target is a call AT s
def blocked_auc(X, y, nfold=5):
    n = len(y); e = np.linspace(0, n, nfold + 1).astype(int)
    oof = np.full(n, np.nan)
    for i in range(nfold):
        te = np.zeros(n, bool); te[e[i]:e[i + 1]] = True; tr = ~te
        if y[tr].sum() < 3 or y[te].sum() < 1: continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    return roc_auc_score(y[ok], oof[ok]) if len(np.unique(y[ok])) > 1 else np.nan


rng = np.random.default_rng(7)
targets = [(c, (hits[c] > 0).astype(int)) for c in CATS]
targets.append(("ANY call", (sum(hits[c] for c in CATS) > 0).astype(int)))

print()
print("%-10s %6s %9s %9s %9s %10s" % ("target", "n", "pos only", "+boat", "gain", "null 95th"))
print("-" * 60)
for nm, y in targets:
    if y.sum() < 12: 
        print("%-10s %6d   (too few to score)" % (nm, y.sum())); continue
    a0 = blocked_auc(BASE, y)
    a1 = blocked_auc(np.column_stack([BASE, BOAT]), y)
    gains = []
    for _ in range(40):
        sh = int(rng.integers(26, NS - 26))
        gains.append(blocked_auc(np.column_stack([BASE, np.roll(BOAT, sh, axis=0)]), y) - a0)
    thr = float(np.nanpercentile(gains, 95))
    star = "*" if (a1 - a0) > thr else " "
    print("%-10s %6d %9.3f %9.3f %+9.3f%s %9.3f" % (nm, y.sum(), a0, a1, a1 - a0, star, thr))
print()
print("* = the gain from boat state exceeds the 95th percentile of 40 runs in which the")
print("  boat-state block is circularly shifted, breaking its timing against the calls")
print("  while preserving its own dynamics and the race-position baseline.")
