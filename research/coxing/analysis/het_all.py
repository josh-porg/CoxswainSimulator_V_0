"""Per-race ridge-regressed responses for every functional category.

The call-frequency profiles differ between races, but that is a statement about the
coxswain's input, not the boat's response. This estimates the response itself in each
race separately, on the same summary used for the catch comparison -- the mean
coefficient over the first 48 s after a call -- and tests each category for between-race
heterogeneity.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "elite")
import vtt
from responses import FUNC, LOOKUP, CATS, hits, NS, R, detrend, LAGS
from race2_readings import T as T2, V as V2
from race3_readings import T as T3, V as V3

STROKE = float(np.median(np.diff(R["t"])))
HORIZON = 48.0
BINS = [(a, a + 8) for a in range(0, int(HORIZON), 8)]
norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == "'")
NB = 400


def ridge(X, y, lam):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ y))


def race1(cat):
    y = detrend(R["v"]); y = y - y.mean()
    s = hits[cat]
    if s.sum() < 5:
        return np.nan, np.nan, int(s.sum())
    X = np.column_stack([np.roll(s, L) for L in LAGS]); X = X - X.mean(0)
    keep = LAGS * STROKE <= HORIZON
    theta = ridge(X, y, 96.2)[keep].mean()
    rng = np.random.default_rng(11)
    block = 40
    starts = np.arange(0, NS - block)
    d = np.empty(NB)
    for i in range(NB):
        idx = np.concatenate([np.arange(st, st + block)
                              for st in rng.choice(starts, NS // block)])
        idx = idx[idx < NS]
        d[i] = ridge(X[idx] - X[idx].mean(0), y[idx] - y[idx].mean(), 96.2)[keep].mean()
    return theta, d.std(), int(s.sum())


def sprint(cat, cap, T, V, lo, hi, seed):
    terms = set(FUNC[cat])
    ev = [t for t, w in vtt.parse_words(cap) if lo - HORIZON <= t <= hi and norm(w) in terms]
    if len(ev) < 5:
        return np.nan, np.nan, len(ev)
    ok = np.isfinite(V)
    t, v = T[ok], V[ok]
    w = 7
    tr = np.convolve(np.pad(v, (w // 2, w // 2), mode="edge"),
                     np.ones(w) / w, mode="valid")[:len(v)]
    y = v - tr; y = y - y.mean()
    X = np.zeros((len(t), len(BINS)))
    for r, ti in enumerate(t):
        for j, (a, b) in enumerate(BINS):
            X[r, j] = sum(1 for e in ev if a <= ti - e < b)
    X = X - X.mean(0)
    theta = ridge(X, y, 2.0).mean()
    rng = np.random.default_rng(seed)
    n = len(y); d = np.empty(NB)
    for i in range(NB):
        idx = rng.integers(0, n, n)
        d[i] = ridge(X[idx] - X[idx].mean(0), y[idx] - y[idx].mean(), 2.0).mean()
    return theta, d.std(), len(ev)


R2 = ("elite/nat8.en.vtt", T2, V2, 108, 330)
R3 = ("elite/cap_SQQTqYmb_Dg.en.vtt", T3, V3, 102, 312)

print("Mean response over the first %.0f s after a call, mm/s per call (se in brackets)."
      % HORIZON)
print("Categories with fewer than five calls in a race are not estimated.")
print()
print("%-12s %20s %20s %20s %10s" % ("category", "race 1", "race 2", "race 3", "Q test"))
print("-" * 88)
rows = []
for c in CATS:
    e1 = race1(c)
    e2 = sprint(c, *R2, seed=5)
    e3 = sprint(c, *R3, seed=6)
    cells = []
    for th, se, k in (e1, e2, e3):
        cells.append("  n<5" if np.isnan(th) else "%+7.1f (%5.1f) %2d" % (1000 * th, 1000 * se, k))
    ok = [(th, se) for th, se, k in (e1, e2, e3) if not np.isnan(th)]
    if len(ok) >= 2:
        th = np.array([a for a, _ in ok]); se = np.array([b for _, b in ok])
        w = 1 / se ** 2
        mu = (w * th).sum() / w.sum()
        Q = float((w * (th - mu) ** 2).sum()); df = len(ok) - 1
        pq = float(stats.chi2.sf(Q, df))
        qs = "%.2f p=%.2f" % (Q, pq)
    else:
        qs = "--"
    rows.append((c, e1, e2, e3, qs))
    print("%-12s %20s %20s %20s %10s" % (c, cells[0], cells[1], cells[2], qs))

print()
print("Weight carried by race 1 in a fixed-effect pool, per category:")
for c, e1, e2, e3, _ in rows:
    ok = [(th, se) for th, se, k in (e1, e2, e3) if not np.isnan(th)]
    if len(ok) < 2:
        continue
    se = np.array([b for _, b in ok]); w = 1 / se ** 2
    print("   %-12s %.1f%%" % (c, 100 * w[0] / w.sum()))
