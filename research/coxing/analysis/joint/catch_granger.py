"""Does the paper's catch effect survive once speed's own history is a predictor?

The paper fitted speed on lagged catch calls with the trend removed, but never gave
speed its own past. The rival account -- the coxswain calls the catch because the boat is
already slowing, and it keeps slowing by its own dynamics -- predicts that catch calls
add nothing once that past is in the model. This is that test, for catch alone.
"""
import json, os, sys
import numpy as np
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "joint")); RACE = os.path.dirname(HERE)   # its data and outputs (gitignored)
sys.path.insert(0, RACE); sys.path.insert(0, os.path.join(RACE, "elite")); sys.path.insert(0, CODE)
os.chdir(RACE)
import importlib.util
spec = importlib.util.spec_from_file_location("c2", os.path.join(CODE, "coupling2.py"))
from responses import hits, NS, R, detrend

rng = np.random.default_rng(11)
v = detrend(R["v"]); v = (v - v.mean()) / v.std()
sr = detrend(R["sr"]); sr = (sr - sr.mean()) / sr.std()
B = np.column_stack([v, sr])
TAUS = (2.0, 8.0, 30.0); WARM = 30

def decay(X):
    T, K = X.shape; out = np.zeros((T, K * len(TAUS)))
    for i, tau in enumerate(TAUS):
        a = np.exp(-1 / tau); h = np.zeros(K)
        for t in range(T):
            out[t, i*K:(i+1)*K] = h; h = a*h + a*X[t]
    return out

HB = decay(B)
RAW = np.column_stack([np.roll(B, 1, 0), np.roll(B, 2, 0)])
catch = (hits["catch"] > 0).astype(float)[:, None]
LAGC = np.column_stack([np.roll(catch[:, 0], k) for k in range(1, 26)])   # the paper's 25 lags
idx = np.arange(WARM, NS); FOLDS = np.array_split(idx, 5)

def ridge_fit(X, y, lam):
    mx, my = X.mean(0), y.mean(); Xc = X - mx
    b = np.linalg.solve(Xc.T @ Xc + lam*np.eye(X.shape[1]), Xc.T @ (y - my))
    return mx, my, b, float(np.mean((y - my - Xc @ b)**2))

def gauss_cv(X, y, lams=np.logspace(-2, 4, 20)):
    ll = 0.0
    for te in FOLDS:
        tr = np.setdiff1d(idx, te); n = int(len(tr)*0.2); fi, va = tr[:-n], tr[-n:]
        best = max(lams, key=lambda l: np.mean(
            [(-0.5*np.log(2*np.pi*s2) - 0.5*r**2/s2) for s2 in [ridge_fit(X[fi], y[fi], l)[3]]
             for r in [y[va] - ((X[va]-ridge_fit(X[fi], y[fi], l)[0]) @ ridge_fit(X[fi], y[fi], l)[2]
                                + ridge_fit(X[fi], y[fi], l)[1])]]))
        mx, my, b, s2 = ridge_fit(X[tr], y[tr], best)
        r = y[te] - ((X[te]-mx) @ b + my)
        ll += float(np.sum(-0.5*np.log(2*np.pi*s2) - 0.5*r**2/s2))
    return ll / len(idx)

def r2_oos(X, y, lams=np.logspace(-2, 4, 20)):
    num = den = 0.0
    for te in FOLDS:
        tr = np.setdiff1d(idx, te); n = int(len(tr)*0.2); fi, va = tr[:-n], tr[-n:]
        def vl(l):
            mx, my, b, _ = ridge_fit(X[fi], y[fi], l)
            return -np.mean((y[va] - ((X[va]-mx) @ b + my))**2)
        mx, my, b, _ = ridge_fit(X[tr], y[tr], max(lams, key=vl))
        num += np.sum((y[te] - ((X[te]-mx) @ b + my))**2); den += np.sum((y[te]-y[tr].mean())**2)
    return 1 - num/den

res = {}
for label, Xc in (("decayed (3 timescales)", decay(catch)), ("25 raw lags, as in paper", LAGC)):
    # (a) the paper's model: catch only, no boat history
    alone = r2_oos(Xc, v)
    # (b) the control: boat history only
    own = r2_oos(np.column_stack([RAW, HB]), v)
    # (c) both
    both = r2_oos(np.column_stack([RAW, HB, Xc]), v)
    g0 = gauss_cv(np.column_stack([RAW, HB]), v)
    g1 = gauss_cv(np.column_stack([RAW, HB, Xc]), v)
    null = []
    for _ in range(200):
        sh = int(rng.integers(40, NS-40))
        null.append(gauss_cv(np.column_stack([RAW, HB, np.roll(Xc, sh, 0)]), v) - g0)
    null = np.array(null); gain = g1 - g0
    p = float((np.sum(null >= gain)+1)/(len(null)+1))
    res[label] = dict(r2_catch_alone=alone, r2_boat_own=own, r2_both=both,
                      gain=gain, null95=float(np.percentile(null, 95)), p=p)
    print("catch calls, %s" % label)
    print("  out-of-sample R2:  catch alone %+.3f | boat's own past %+.3f | both %+.3f"
          % (alone, own, both))
    print("  catch's added value over boat's own past: %+.4f nats/stroke,"
          " null 95th %+.4f, p = %.3f" % (gain, np.percentile(null, 95), p))
    print()
json.dump(res, open(os.path.join(HERE, "catch_granger.json"), "w"), indent=1)
