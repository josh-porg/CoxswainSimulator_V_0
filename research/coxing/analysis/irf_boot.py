"""Block bootstrap on the impulse responses, so the paper can quote uncertainty."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import re

import numpy as np

exec(open("irf.py").read().split("U, S, Vt = np.linalg.svd")[0])

LAM = 119.0
BLOCK = 40
NB = 300
rng = np.random.default_rng(7)


def fit(Xs, ys, lam=LAM):
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    return Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ ys))


beta0 = fit(X, y)
B0 = {}
for (name, L), b in zip(names, beta0):
    B0.setdefault(name, np.zeros(len(LAGS)))[L] = b

starts = np.arange(0, NS - BLOCK)
boots = {n: [] for n, _, _ in SUBS}
for _ in range(NB):
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, NS // BLOCK)])
    idx = idx[idx < NS]
    b = fit(X[idx], y[idx])
    for (name, L), val in zip(names, b):
        if L == 0:
            boots[name].append(np.zeros(len(LAGS)))
        boots[name][-1][L] = val

print("block bootstrap: %d resamples, %d-stroke blocks" % (NB, BLOCK))
print()
print("  %-16s %10s %9s %12s %s" % ("call type", "peak m/s", "at lag", "95%% CI", "sum lags 0-9 (m/s.strokes)"))
res = {}
for name, _, _ in SUBS:
    A = np.array(boots[name])
    b = B0[name]
    pk = int(np.argmax(np.abs(b)))
    peaks = A[:, pk]
    lo, hi = np.percentile(peaks, [2.5, 97.5])
    area = b[:10].sum()
    areas = A[:, :10].sum(axis=1)
    alo, ahi = np.percentile(areas, [2.5, 97.5])
    sig = "*" if (lo > 0) or (hi < 0) else " "
    sigA = "*" if (alo > 0) or (ahi < 0) else " "
    res[name] = dict(peak=b[pk], lag=pk, lo=lo, hi=hi, area=area, alo=alo, ahi=ahi)
    print("  %-16s %+10.4f %9d  [%+.4f,%+.4f]%s   %+.4f [%+.4f,%+.4f]%s"
          % (name, b[pk], pk, lo, hi, sig, area, alo, ahi, sigA))
print()
print("  * = 95%% bootstrap interval excludes zero")
json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()}, open("irf_ci.json", "w"))
