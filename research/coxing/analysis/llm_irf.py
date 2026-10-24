"""Do calls move the boat, using LLM labels instead of keyword matching?

The keyword lexicon was built from race 1's own vocabulary and matched only 85 of 405
race-2 words. LLM labels are assigned from context rather than word lists, so they
transfer and they capture calls no keyword list would catch. This reruns the forward
analysis -- does a call predict boat speed? -- on all four LLM axes.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import sys

import numpy as np

R = np.load("race.npz")
t, v = R["t"], R["v"]
NS = len(t)
L = json.load(open("labels_llm.json"))["race1"]
LAGS = np.arange(0, 26)

AX = {}
for ax in ("theme", "target", "pragmatic", "source"):
    for val in sorted({d[ax] for d in L}):
        s = np.zeros(NS)
        for d in L:
            if d[ax] == val:
                s[int(np.argmin(np.abs(t - (d["t"] + 1.0))))] += 1.0
        if s.sum() >= 12:
            AX["%s=%s" % (ax, val)] = s

k61 = np.ones(61) / 61
y = v - np.convolve(np.pad(v, (30, 30), mode="edge"), k61, mode="valid")[:NS]
y = y - y.mean()


def lagmat(s):
    X = np.column_stack([np.roll(s, L_) for L_ in LAGS])
    return X - X.mean(0)


def blocked_oos(X, yy, nfold=5):
    n = len(yy); e = np.linspace(0, n, nfold + 1).astype(int)
    num = den = 0.0
    for i in range(nfold):
        te = np.zeros(n, bool); te[e[i]:e[i + 1]] = True; tr = ~te
        Xt = X[tr] - X[tr].mean(0); m = yy[tr].mean()
        U, S, Vt = np.linalg.svd(Xt, full_matrices=False)
        bg = bl = None
        for lam in np.logspace(-2, 5, 30):
            b = Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ (yy[tr] - m)))
            dof = float(np.sum(S ** 2 / (S ** 2 + lam)))
            g = float(np.sum((yy[tr] - m - Xt @ b) ** 2)) / max(tr.sum() - dof, 1) ** 2
            if bg is None or g < bg: bg, bl = g, lam
        b = Vt.T @ ((S / (S ** 2 + bl)) * (U.T @ (yy[tr] - m)))
        num += float(np.sum((yy[te] - (X[te] @ b + m)) ** 2))
        den += float(np.sum((yy[te] - m) ** 2))
    return 1 - num / den


rng = np.random.default_rng(31)
print("%-26s %5s %10s %11s %8s" % ("LLM category", "n", "OOS R2", "null 95th", "p"))
print("-" * 64)
res = {}
for k, s in sorted(AX.items()):
    obs = blocked_oos(lagmat(s), y)
    null = np.array([blocked_oos(lagmat(np.roll(s, int(rng.integers(26, NS - 26)))), y)
                     for _ in range(200)])
    p = float(np.mean(null >= obs))
    res[k] = (obs, p)
    print("%-26s %5d %+10.4f %+11.4f %8.3f%s"
          % (k, int(s.sum()), obs, np.percentile(null, 95), p, "*" if p < 0.05 else ""))
best = max(res, key=lambda k: res[k][0])
print()
print("best: %s at OOS R2 %+.4f, uncorrected p=%.3f; %d categories tested, so the"
      % (best, res[best][0], res[best][1], len(res)))
print("Bonferroni threshold is p < %.4f" % (0.05 / len(res)))
json.dump({k: [float(a), float(b)] for k, (a, b) in res.items()}, open("llm_irf.json", "w"), indent=1)
