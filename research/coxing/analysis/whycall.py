"""Given that the coxswain is speaking, what determines WHAT they say?

At roughly one call per stroke, "is the coxswain talking" is nearly always yes, so
predicting call occurrence is dominated by a high base rate. Conditioning on speaking
and asking which category is selected is the better-posed question, and the one that
matches what a coxswain is actually doing: not deciding whether to speak but choosing
what to say next.

Three blocks of predictors, added in the order the coxswain gets them:
  POSITION  how far through the race -- the planned, scripted component
  BOX       what the cox-box displays: split (speed), rate, distance per stroke,
            distance, and their recent changes
  FELT      the vestibular input: surge, sway, heave and angular rates in boat
            coordinates, from the head-mounted IMU
Each block is scored by what it adds over the ones before, against a matched null in
which that block alone is circularly shifted.
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

v, sr, dps, d = R["v"], R["sr"], R["dps"], R["d"]
split = 500.0 / v                       # what the box actually shows


def dev(x, w=61):
    k = np.ones(w) / w
    return x - np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), k, mode="valid")[:len(x)]


def lagdiff(x, n):
    y = np.empty_like(x, dtype=float); y[:n] = 0; y[n:] = x[n:] - x[:-n]
    return y


pos = d / d.max()
POSITION = np.column_stack([pos, pos ** 2, pos ** 3, np.sin(np.pi * pos), np.cos(np.pi * pos)])
bx, bn = [], []
for nm, x in (("split", split), ("rate", sr), ("dps", dps)):
    bx.append(dev(x)); bn.append(nm + " dev")
    for n in (1, 3, 5):
        bx.append(lagdiff(x, n)); bn.append("d%s/%d" % (nm, n))
BOX = np.column_stack(bx)
FELT = VEST

speaking = (sum(hits[c] for c in CATS) > 0)
print("strokes with a call: %d of %d (%.0f%%)" % (speaking.sum(), NS, 100 * speaking.mean()))
print("blocks: POSITION %d cols, BOX %d cols, FELT %d cols"
      % (POSITION.shape[1], BOX.shape[1], FELT.shape[1]))


def blocked_auc(X, y, nfold=5):
    n = len(y); e = np.linspace(0, n, nfold + 1).astype(int)
    oof = np.full(n, np.nan)
    for i in range(nfold):
        te = np.zeros(n, bool); te[e[i]:e[i + 1]] = True; tr = ~te
        if y[tr].sum() < 3 or y[te].sum() < 1 or (~y[tr].astype(bool)).sum() < 3: continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=4000, C=0.3, class_weight="balanced"))
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2: return np.nan
    return roc_auc_score(y[ok], oof[ok])


rng = np.random.default_rng(13)
S = np.flatnonzero(speaking)
print()
print("%-10s %5s %8s %8s %9s %8s %9s" %
      ("category", "n", "POS", "+BOX", "null95", "+FELT", "null95"))
print("-" * 62)
for c in CATS:
    y = (hits[c][S] > 0).astype(int)
    if y.sum() < 12:
        print("%-10s %5d  (too few)" % (c, y.sum())); continue
    P, B, F = POSITION[S], BOX[S], FELT[S]
    a0 = blocked_auc(P, y)
    a1 = blocked_auc(np.column_stack([P, B]), y)
    a2 = blocked_auc(np.column_stack([P, B, F]), y)
    nb, nf = [], []
    for _ in range(40):
        s1 = int(rng.integers(20, len(S) - 20))
        nb.append(blocked_auc(np.column_stack([P, np.roll(B, s1, axis=0)]), y) - a0)
        s2 = int(rng.integers(20, len(S) - 20))
        nf.append(blocked_auc(np.column_stack([P, B, np.roll(F, s2, axis=0)]), y) - a1)
    tb = float(np.nanpercentile(nb, 95)); tf = float(np.nanpercentile(nf, 95))
    sb = "*" if (a1 - a0) > tb else " "
    sf = "*" if (a2 - a1) > tf else " "
    print("%-10s %5d %8.3f %+8.3f%s %8.3f %+8.3f%s %8.3f"
          % (c, y.sum(), a0, a1 - a0, sb, tb, a2 - a1, sf, tf))
print()
print("AUC is out-of-fold over contiguous blocks. Gains are over the preceding block;")
print("each null shifts only the block being tested. %d categories tested." % len(CATS))
