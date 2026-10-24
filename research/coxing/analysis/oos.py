"""Does any of this predict strokes the model has not seen?

In-sample R2 rises automatically with the number of regressors, and the functional
model has 208 of them. The earlier unvalidated classifier scored 0.275 in sample and
was spurious. Contiguous blocked folds are used rather than shuffled ones, because
speed is autocorrelated and shuffling leaks neighbouring strokes into training.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np

sys.path.insert(0, "elite")
import vtt
from functional import FUNC, LOOKUP  # noqa: F401  (reuse the same lexicon)

LAGS = np.arange(0, 26)
R = np.load("race.npz")
t, v = R["t"], R["v"]
NS = len(t)
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == "'")
words = vtt.parse_words("caps.en.vtt")

CATS = list(FUNC)
hits = {k: np.zeros(NS) for k in CATS}
for tt, w in words:
    k = LOOKUP.get(norm(w))
    if k:
        hits[k][int(np.argmin(np.abs(t - (tt + 1.0))))] += 1.0

kern = np.ones(61) / 61
trend = np.convolve(np.pad(v, (30, 30), mode="edge"), kern, mode="valid")[:NS]
y = v - trend
y = y - y.mean()


def design(keys):
    cols = []
    for k in keys:
        for L in LAGS:
            cols.append(np.roll(hits[k], L))
    X = np.column_stack(cols)
    return X - X.mean(axis=0)


def blocked_oos(X, y, nfold=5, lam=None):
    n = len(y)
    edges = np.linspace(0, n, nfold + 1).astype(int)
    num, den = 0.0, 0.0
    for i in range(nfold):
        te = np.zeros(n, bool); te[edges[i]:edges[i + 1]] = True
        tr = ~te
        Xtr, ytr = X[tr], y[tr]
        Xtr = Xtr - Xtr.mean(axis=0); m = ytr.mean()
        U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
        # pick lambda inside the fold if not given
        if lam is None:
            bestl, bestg = None, None
            for L in np.logspace(-2, 5, 40):
                b = Vt.T @ ((S / (S ** 2 + L)) * (U.T @ (ytr - m)))
                dof = float(np.sum(S ** 2 / (S ** 2 + L)))
                g = float(np.sum((ytr - m - Xtr @ b) ** 2)) / max(len(ytr) - dof, 1) ** 2
                if bestg is None or g < bestg:
                    bestg, bestl = g, L
            L = bestl
        else:
            L = lam
        b = Vt.T @ ((S / (S ** 2 + L)) * (U.T @ (ytr - m)))
        pred = X[te] @ b + m
        num += float(np.sum((y[te] - pred) ** 2))
        den += float(np.sum((y[te] - m) ** 2))
    return 1 - num / den


print("blocked 5-fold out-of-sample R2 (lambda chosen inside each fold)")
print()
full = design(CATS)
print("   8 functional categories, %3d regressors : %+.4f" % (full.shape[1], blocked_oos(full, y)))
for k in CATS:
    Xk = design([k])
    print("   %-9s alone,          %3d regressors : %+.4f"
          % (k, Xk.shape[1], blocked_oos(Xk, y)))

# a control with the same shape but no real timing
rng = np.random.default_rng(3)
scores = []
for _ in range(20):
    sh = {k: np.roll(hits[k], int(rng.integers(30, NS - 30))) for k in CATS}
    cols = [np.roll(sh[k], L) for k in CATS for L in LAGS]
    Xp = np.column_stack(cols); Xp = Xp - Xp.mean(axis=0)
    scores.append(blocked_oos(Xp, y))
scores = np.array(scores)
print()
print("   shifted control (same 208 regressors, timing destroyed):")
print("      mean %+.4f, range %+.4f to %+.4f over 20 draws"
      % (scores.mean(), scores.min(), scores.max()))
print()
print("A model that has learned something real scores above the shifted control.")
