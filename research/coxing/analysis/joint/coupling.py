"""Coupled autoregression: does each stream shift the other's predictive distribution?

The joint process factorises as
    p(C, B) = prod_t  p(C_t | C_<t, B_<t) * p(B_t | B_<t, C_<t)
so there are two coupling questions, each a Granger test with held-out likelihood:

  C -> B  does the call history improve prediction of boat speed, GIVEN the boat's own
          history?  (The paper never gave speed its own lags, so its forward analysis could
          not separate "the call changed the boat" from "the boat was already changing and
          the coxswain saw it".)
  B -> C  does the boat history improve prediction of the next call, GIVEN the call
          history?

Each is scored as the gain in blocked out-of-sample log-likelihood from adding the other
stream, against a null in which the other stream is circularly shifted (own history kept
intact, cross-alignment destroyed).
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

import os
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "joint"))   # its data and outputs (gitignored)
RACE = os.path.dirname(HERE)
sys.path.insert(0, RACE)
sys.path.insert(0, os.path.join(RACE, "elite"))
os.chdir(RACE)
from responses import FUNC, CATS, hits, NS, R, detrend

rng = np.random.default_rng(2026)
KB, KC = 10, 25            # boat own-lags; call lags into the boat equation
KCC, KBC = 10, 10          # call own-lags; boat lags into the call equation

# ---------------- streams ----------------
v = detrend(R["v"]); v = (v - v.mean()) / v.std()
sr = detrend(R["sr"]); sr = (sr - sr.mean()) / sr.std()
B = np.column_stack([v, sr])                                  # boat stream, 2 channels
C = np.column_stack([(hits[c] > 0).astype(float) for c in CATS])  # call stream, 8 channels


def lags(X, k0, k1):
    """Columns X_{t-k} for k in [k0, k1]; rows before k1 are dropped by the caller."""
    return np.column_stack([np.roll(X, k, axis=0) for k in range(k0, k1 + 1)])


START = max(KB, KC, KCC, KBC)          # first row with a full history
idx = np.arange(START, NS)
FOLDS = np.array_split(idx, 5)          # contiguous blocks


# ---------------- C -> B : Gaussian log-likelihood of speed ----------------
def ridge_cv_ll(X, y, lams=np.logspace(-2, 4, 25)):
    """Blocked-CV Gaussian log-lik per row; lambda chosen inside each training fold by GCV."""
    ll = 0.0
    for te in FOLDS:
        tr = np.setdiff1d(idx, te)
        Xt, yt = X[tr], y[tr]
        mx, my = Xt.mean(0), yt.mean()
        Xc, yc = Xt - mx, yt - my
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        best = None
        for lam in lams:
            b = Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ yc))
            dof = np.sum(S ** 2 / (S ** 2 + lam))
            g = np.sum((yc - Xc @ b) ** 2) / max(len(yc) - dof, 1) ** 2
            if best is None or g < best[0]:
                best = (g, b, np.sum((yc - Xc @ b) ** 2) / max(len(yc) - dof, 1))
        _, b, s2 = best
        r = y[te] - ((X[te] - mx) @ b + my)
        ll += float(np.sum(-0.5 * np.log(2 * np.pi * s2) - 0.5 * r ** 2 / s2))
    return ll / len(idx)


y_b = v
X_own_b = lags(B, 1, KB)
X_c_into_b = lags(C, 1, KC)
ll_b0 = ridge_cv_ll(X_own_b, y_b)
ll_b1 = ridge_cv_ll(np.column_stack([X_own_b, X_c_into_b]), y_b)
null_b = []
for _ in range(200):
    Cs = np.roll(C, int(rng.integers(40, NS - 40)), axis=0)
    null_b.append(ridge_cv_ll(np.column_stack([X_own_b, lags(Cs, 1, KC)]), y_b) - ll_b0)
null_b = np.array(null_b)
gain_b = ll_b1 - ll_b0

# how much does the boat's own history carry?
ll_bnone = ridge_cv_ll(np.zeros((NS, 1)) + rng.normal(0, 1e-9, (NS, 1)), y_b)


# ---------------- B -> C : Bernoulli log-likelihood of each call channel ----------------
def logit_cv_ll(X, Y, C_=0.3):
    ll = 0.0
    for te in FOLDS:
        tr = np.setdiff1d(idx, te)
        mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-9
        for j in range(Y.shape[1]):
            yt = Y[tr, j]
            if yt.sum() < 2 or yt.sum() > len(yt) - 2:
                p = np.full(len(te), np.clip(yt.mean(), 1e-3, 1 - 1e-3))
            else:
                m = LogisticRegression(C=C_, max_iter=3000)
                m.fit((X[tr] - mx) / sx, yt)
                p = np.clip(m.predict_proba((X[te] - mx) / sx)[:, 1], 1e-4, 1 - 1e-4)
            ye = Y[te, j]
            ll += float(np.sum(ye * np.log(p) + (1 - ye) * np.log(1 - p)))
    return ll / len(idx)


X_own_c = lags(C, 1, KCC)
X_b_into_c = lags(B, 1, KBC)
ll_c0 = logit_cv_ll(X_own_c, C)
ll_c1 = logit_cv_ll(np.column_stack([X_own_c, X_b_into_c]), C)
null_c = []
for _ in range(60):
    Bs = np.roll(B, int(rng.integers(40, NS - 40)), axis=0)
    null_c.append(logit_cv_ll(np.column_stack([X_own_c, lags(Bs, 1, KBC)]), C) - ll_c0)
null_c = np.array(null_c)
gain_c = ll_c1 - ll_c0

base_rate_ll = float(np.mean([
    np.mean(C[idx, j] * np.log(np.clip(C[idx, j].mean(), 1e-3, 1)) +
            (1 - C[idx, j]) * np.log(np.clip(1 - C[idx, j].mean(), 1e-3, 1)))
    for j in range(C.shape[1])])) * C.shape[1]

out = {}
print("Race 1, %d strokes usable (%d dropped for lag history), 5 contiguous folds" % (len(idx), START))
print()
print("SELF-PREDICTION (each stream from its own past)")
print("  boat speed : held-out loglik/row %.3f vs %.3f with no predictors  (gain %+.3f)"
      % (ll_b0, ll_bnone, ll_b0 - ll_bnone))
print("  calls      : held-out loglik/row %.3f vs %.3f at base rates        (gain %+.3f)"
      % (ll_c0, base_rate_ll, ll_c0 - base_rate_ll))
print()
print("COUPLING (added to a model that already has the stream's own past)")
pb = float((np.sum(null_b >= gain_b) + 1) / (len(null_b) + 1))
pc = float((np.sum(null_c >= gain_c) + 1) / (len(null_c) + 1))
print("  C -> B  calls into speed : gain %+.4f nats/row   null 95th %+.4f   p = %.3f"
      % (gain_b, np.percentile(null_b, 95), pb))
print("  B -> C  boat into calls  : gain %+.4f nats/row   null 95th %+.4f   p = %.3f"
      % (gain_c, np.percentile(null_c, 95), pc))
json.dump(dict(ll_b0=ll_b0, ll_b1=ll_b1, gain_b=gain_b, p_b=pb,
               ll_c0=ll_c0, ll_c1=ll_c1, gain_c=gain_c, p_c=pc,
               self_b=ll_b0 - ll_bnone, self_c=ll_c0 - base_rate_ll),
          open("joint/coupling.json", "w"), indent=1)
