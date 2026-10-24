"""Coupled autoregression with a representation the data can support.

The first version gave each channel 10-25 raw lags and overfit: the call model scored below
base rates on held-out data. Here each channel's history is summarised by exponentially
decayed sums at three timescales,
    h_c^tau(t) = sum_{k>=1} exp(-k / tau) * x_c(t-k),   tau in {2, 8, 30} strokes,
which is the linear form of a multivariate Hawkes process: recent events weigh most, and
three timescales span one stroke to a minute. The boat also keeps its first two raw lags,
because its autocorrelation is strongest there. Regularisation is chosen inside every
training fold on a held-back contiguous tail, so no test data reaches it.

Parameter counts per equation drop from 80-220 to about 30.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "joint"))   # its data and outputs (gitignored)
RACE = os.path.dirname(HERE)
sys.path.insert(0, RACE); sys.path.insert(0, os.path.join(RACE, "elite"))
os.chdir(RACE)
from responses import CATS, hits, NS, R, detrend

rng = np.random.default_rng(2026)
TAUS = (2.0, 8.0, 30.0)
WARM = 30


def decay_feats(X):
    """Exponentially-decayed past sums, strictly causal (excludes the current row)."""
    T, K = X.shape
    out = np.zeros((T, K * len(TAUS)))
    for i, tau in enumerate(TAUS):
        a = np.exp(-1.0 / tau)
        h = np.zeros(K)
        for t in range(T):
            out[t, i * K:(i + 1) * K] = h
            h = a * h + a * X[t]
    return out


v = detrend(R["v"]); v = (v - v.mean()) / v.std()
sr = detrend(R["sr"]); sr = (sr - sr.mean()) / sr.std()
B = np.column_stack([v, sr])
C = np.column_stack([(hits[c] > 0).astype(float) for c in CATS])

HB, HC = decay_feats(B), decay_feats(C)
RAW_B = np.column_stack([np.roll(B, 1, axis=0), np.roll(B, 2, axis=0)])

idx = np.arange(WARM, NS)
FOLDS = np.array_split(idx, 5)


def split_tail(tr, frac=0.2):
    n = int(len(tr) * frac)
    return tr[:-n], tr[-n:]


# ---------------- Gaussian (boat) ----------------
def ridge_fit(X, y, lam):
    mx, my = X.mean(0), y.mean()
    Xc = X - mx
    b = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - my))
    s2 = float(np.mean((y - my - Xc @ b) ** 2))
    return mx, my, b, s2


def gauss_cv(X, y, lams=np.logspace(-2, 4, 20)):
    ll = 0.0
    for te in FOLDS:
        tr = np.setdiff1d(idx, te)
        fit_i, val_i = split_tail(tr)
        best = None
        for lam in lams:
            mx, my, b, s2 = ridge_fit(X[fit_i], y[fit_i], lam)
            r = y[val_i] - ((X[val_i] - mx) @ b + my)
            vl = np.mean(-0.5 * np.log(2 * np.pi * s2) - 0.5 * r ** 2 / s2)
            if best is None or vl > best[0]:
                best = (vl, lam)
        mx, my, b, s2 = ridge_fit(X[tr], y[tr], best[1])
        r = y[te] - ((X[te] - mx) @ b + my)
        ll += float(np.sum(-0.5 * np.log(2 * np.pi * s2) - 0.5 * r ** 2 / s2))
    return ll / len(idx)


# ---------------- Bernoulli (calls) ----------------
def bern_cv(X, Y, Cs=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    ll = 0.0
    for te in FOLDS:
        tr = np.setdiff1d(idx, te)
        fit_i, val_i = split_tail(tr)
        for j in range(Y.shape[1]):
            def fitpred(rows_fit, rows_pred, Cr):
                yt = Y[rows_fit, j]
                if X is None or yt.sum() < 3 or yt.sum() > len(yt) - 3:
                    return np.full(len(rows_pred), np.clip(yt.mean(), 1e-3, 1 - 1e-3))
                mx, sx = X[rows_fit].mean(0), X[rows_fit].std(0) + 1e-9
                m = LogisticRegression(C=Cr, max_iter=4000)
                m.fit((X[rows_fit] - mx) / sx, yt)
                return np.clip(m.predict_proba((X[rows_pred] - mx) / sx)[:, 1], 1e-4, 1 - 1e-4)

            def nll(p, ye):
                return float(np.sum(ye * np.log(p) + (1 - ye) * np.log(1 - p)))
            if X is None:
                bestC = None
            else:
                bestC = max(Cs, key=lambda Cr: nll(fitpred(fit_i, val_i, Cr), Y[val_i, j]))
            p = fitpred(tr, te, bestC)
            ll += nll(p, Y[te, j])
    return ll / len(idx)


# ---------- self-prediction ----------
g_none = gauss_cv(np.zeros((NS, 1)) + rng.normal(0, 1e-6, (NS, 1)), v)
g_self = gauss_cv(np.column_stack([RAW_B, HB]), v)
b_base = bern_cv(None, C)
b_self = bern_cv(HC, C)

# ---------- coupling ----------
g_joint = gauss_cv(np.column_stack([RAW_B, HB, HC]), v)
b_joint = bern_cv(np.column_stack([HC, HB]), C)

NULL_B, NULL_C = 200, 60
nb, nc = [], []
for _ in range(NULL_B):
    sh = int(rng.integers(40, NS - 40))
    nb.append(gauss_cv(np.column_stack([RAW_B, HB, np.roll(HC, sh, axis=0)]), v) - g_self)
for _ in range(NULL_C):
    sh = int(rng.integers(40, NS - 40))
    nc.append(bern_cv(np.column_stack([HC, np.roll(HB, sh, axis=0)]), C) - b_self)
nb, nc = np.array(nb), np.array(nc)
gb, gc = g_joint - g_self, b_joint - b_self
pb = float((np.sum(nb >= gb) + 1) / (NULL_B + 1))
pc = float((np.sum(nc >= gc) + 1) / (NULL_C + 1))

print("Race 1: %d strokes scored, 5 contiguous folds; regularisation tuned inside each fold"
      % len(idx))
print("features per equation: boat %d, calls %d" % (RAW_B.shape[1] + HB.shape[1] + HC.shape[1],
                                                    HC.shape[1] + HB.shape[1]))
print()
print("SELF-PREDICTION                               held-out loglik per stroke")
print("  boat speed   own past   %+.3f   vs no predictors %+.3f   gain %+.3f"
      % (g_self, g_none, g_self - g_none))
print("  8 call chans own past   %+.3f   vs base rates    %+.3f   gain %+.3f"
      % (b_self, b_base, b_self - b_base))
print()
print("COUPLING, over and above each stream's own past")
print("  C -> B  calls into speed : %+.4f nats/stroke   null 95th %+.4f   p = %.3f"
      % (gb, np.percentile(nb, 95), pb))
print("  B -> C  boat into calls  : %+.4f nats/stroke   null 95th %+.4f   p = %.3f"
      % (gc, np.percentile(nc, 95), pc))
json.dump(dict(self_boat=g_self - g_none, self_calls=b_self - b_base,
               CtoB=gb, p_CtoB=pb, BtoC=gc, p_BtoC=pc,
               null_CtoB_95=float(np.percentile(nb, 95)),
               null_BtoC_95=float(np.percentile(nc, 95))),
          open(os.path.join(HERE, "coupling2.json"), "w"), indent=1)
