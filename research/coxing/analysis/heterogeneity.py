"""Do the three races differ in how the boat responds to a call?

The races are not replicates. They differ in piece length (4050 m against 1000 m), boat
class (coxed four, eight, coxed four), crew, and season, and therefore in the coxswain's
own experience. If different crews respond differently to the same call, the per-race
effects should be more dispersed than their sampling error allows.

This is the standard meta-analytic question, so it is asked with the standard tools:
Cochran's Q against a chi-square null, and I-squared for the share of observed dispersion
not attributable to sampling error. A comparable summary is needed across two different
parameterisations, so each race is reduced to the mean response over the first 48 s after
a catch call, in mm/s per call.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "elite")
import vtt
from responses import hits, NS, R, detrend, LAGS
from race2_readings import T as T2, V as V2
from race3_readings import T as T3, V as V3

STROKE = float(np.median(np.diff(R["t"])))
HORIZON = 48.0
CATCH = set("catch catches direct entry front quick together".split())
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == "'")


def ridge(X, y, lam):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ y))


def race1():
    """Per-stroke parameterisation; mean coefficient over lags within 48 s."""
    y = detrend(R["v"]); y = y - y.mean()
    s = hits["catch"]
    X = np.column_stack([np.roll(s, L) for L in LAGS]); X = X - X.mean(0)
    keep = LAGS * STROKE <= HORIZON
    b = ridge(X, y, 96.2)
    theta = b[keep].mean()
    rng = np.random.default_rng(11)
    block, nb = 40, 600
    starts = np.arange(0, NS - block)
    draws = np.empty(nb)
    for i in range(nb):
        idx = np.concatenate([np.arange(st, st + block)
                              for st in rng.choice(starts, NS // block)])
        idx = idx[idx < NS]
        bb = ridge(X[idx] - X[idx].mean(0), y[idx] - y[idx].mean(), 96.2)
        draws[i] = bb[keep].mean()
    return theta, draws.std(), int(s.sum())


BINS = [(a, a + 8) for a in range(0, int(HORIZON), 8)]


def sprint(cap, T, V, lo, hi):
    """8 s-bin parameterisation; mean coefficient over the same 48 s."""
    ev = [t for t, w in vtt.parse_words(cap) if lo - HORIZON <= t <= hi and norm(w) in CATCH]
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
    rng = np.random.default_rng(5)
    n = len(y); draws = np.empty(600)
    for i in range(600):
        idx = rng.integers(0, n, n)
        draws[i] = ridge(X[idx] - X[idx].mean(0), y[idx] - y[idx].mean(), 2.0).mean()
    return theta, draws.std(), len(ev)


EST = [("Race 1  4050 m W4+, 2026 season", *race1()),
       ("Race 2  1000 m W8+, 2025 season",
        *sprint("elite/nat8.en.vtt", T2, V2, 108, 330)),
       ("Race 3  1000 m W4+, 2025 season",
        *sprint("elite/cap_SQQTqYmb_Dg.en.vtt", T3, V3, 102, 312))]

print("Mean response over the first %.0f s after a catch call, by race" % HORIZON)
print("%-34s %6s %10s %10s" % ("race", "calls", "mm/s", "se"))
print("-" * 64)
for nm, th, se, k in EST:
    print("%-34s %6d %10.2f %10.2f" % (nm, k, 1000 * th, 1000 * se))

th = np.array([e[1] for e in EST])
se = np.array([e[2] for e in EST])
w = 1.0 / se ** 2
mu = float((w * th).sum() / w.sum())
Q = float((w * (th - mu) ** 2).sum())
df = len(th) - 1
pQ = float(stats.chi2.sf(Q, df))
I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
print()
print("fixed-effect pooled estimate: %.2f mm/s (se %.2f)" % (1000 * mu, 1000 / np.sqrt(w.sum())))
print("Cochran Q = %.2f on %d df, p = %.3f" % (Q, df, pQ))
print("I-squared = %.1f%% of dispersion beyond sampling error" % I2)

# How large would a between-race sd have to be before this design detected it?
print()
print("power to detect between-race heterogeneity, given these standard errors:")
rng = np.random.default_rng(3)
print("%-16s %s" % ("true tau (mm/s)", "  ".join("%5d races" % k for k in (3, 6, 12, 24))))
for tau_mm in (5, 10, 20, 40):
    tau = tau_mm / 1000.0
    row = []
    for k in (3, 6, 12, 24):
        hit = 0
        for _ in range(400):
            s_k = rng.choice(se, k, replace=True)
            t_k = rng.normal(0.0, tau, k) + rng.normal(0, s_k)
            wk = 1 / s_k ** 2
            muk = (wk * t_k).sum() / wk.sum()
            Qk = (wk * (t_k - muk) ** 2).sum()
            hit += stats.chi2.sf(Qk, k - 1) < 0.05
        row.append("%5.0f%%" % (100 * hit / 400))
    print("%-16d %s" % (tau_mm, "       ".join(row)))
print()
print("Observed dispersion of the three point estimates: sd %.1f mm/s." % (1000 * th.std(ddof=1)))
