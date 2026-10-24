"""Is the null an artefact of the centred detrend?

detrend() subtracts a CENTRED 61-stroke moving average, so the residual at stroke t
depends on speed up to 30 strokes ahead. That is fine for the paper's static regression
but not for a Granger test, where it can inflate what the boat's own past appears to
predict. Repeat the catch test under three causal treatments of speed.
"""
import os, sys
import numpy as np
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "joint")); RACE = os.path.dirname(HERE)   # its data and outputs (gitignored)
sys.path.insert(0, RACE); sys.path.insert(0, os.path.join(RACE, "elite")); os.chdir(RACE)
from responses import hits, NS, R

rng = np.random.default_rng(3)
raw_v, raw_sr = R["v"].astype(float), R["sr"].astype(float)
catch = (hits["catch"] > 0).astype(float)
TAUS = (2.0, 8.0, 30.0); WARM = 62

def decay(X):
    X = X.reshape(len(X), -1); T, K = X.shape; out = np.zeros((T, K*len(TAUS)))
    for i, tau in enumerate(TAUS):
        a = np.exp(-1/tau); h = np.zeros(K)
        for t in range(T):
            out[t, i*K:(i+1)*K] = h; h = a*h + a*X[t]
    return out

def trailing(x, w):
    out = np.empty_like(x)
    for t in range(len(x)):
        out[t] = x[max(0, t-w):t].mean() if t > 0 else x[0]
    return x - out

idx = np.arange(WARM, NS); FOLDS = np.array_split(idx, 5)

def fit(X, y, lam):
    mx, my = X.mean(0), y.mean(); Xc = X - mx
    b = np.linalg.solve(Xc.T@Xc + lam*np.eye(X.shape[1]), Xc.T@(y-my))
    return mx, my, b, float(np.mean((y-my-Xc@b)**2))

def cv(X, y, lams=np.logspace(-2, 4, 16)):
    ll = 0.0; num = den = 0.0
    for te in FOLDS:
        tr = np.setdiff1d(idx, te); n = int(len(tr)*.2); fi, va = tr[:-n], tr[-n:]
        def score(l):
            mx, my, b, _ = fit(X[fi], y[fi], l); return -np.mean((y[va]-((X[va]-mx)@b+my))**2)
        mx, my, b, s2 = fit(X[tr], y[tr], max(lams, key=score))
        r = y[te]-((X[te]-mx)@b+my)
        ll += np.sum(-0.5*np.log(2*np.pi*s2)-0.5*r**2/s2)
        num += np.sum(r**2); den += np.sum((y[te]-y[tr].mean())**2)
    return ll/len(idx), 1-num/den

TREAT = {
  "raw speed, own lags carry the trend": (raw_v, raw_sr),
  "trailing 61-stroke detrend (causal)": (trailing(raw_v, 61), trailing(raw_sr, 61)),
  "first difference (causal)":           (np.r_[0, np.diff(raw_v)], np.r_[0, np.diff(raw_sr)]),
}
LAGC = np.column_stack([np.roll(catch, k) for k in range(1, 26)])
print("%-40s %9s %9s %9s %7s" % ("speed treatment", "R2 own", "R2 +catch", "gain", "p"))
print("-"*80)
for name, (v, sr) in TREAT.items():
    v = (v-v.mean())/v.std(); sr = (sr-sr.mean())/sr.std()
    B = np.column_stack([v, sr])
    OWN = np.column_stack([np.roll(B, 1, 0), np.roll(B, 2, 0), decay(B)])
    (l0, r0) = cv(OWN, v)
    (l1, r1) = cv(np.column_stack([OWN, LAGC]), v)
    null = [cv(np.column_stack([OWN, np.roll(LAGC, int(rng.integers(40, NS-40)), 0)]), v)[0] - l0
            for _ in range(150)]
    p = (np.sum(np.array(null) >= l1-l0)+1)/(len(null)+1)
    print("%-40s %+9.3f %+9.3f %+9.4f %7.3f" % (name, r0, r1, l1-l0, p))
