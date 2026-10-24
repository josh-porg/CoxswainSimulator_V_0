"""Impulse response of each call type, by distributed-lag regression.

The problem with a fixed window: calls come about once a stroke, so any window after one
call contains many others. A window estimate is therefore a blend of every call type
present, weighted by how often they co-occur.

The fix is to fit them jointly. For each call type c and lag L (strokes after the call),
build X[s, (c,L)] = number of type-c calls landing on stroke s-L, and regress the boat's
detrended response on all of them at once. The coefficient at (c, L) is then the effect
of ONE call of type c, L strokes later, holding every other call constant. The set of
coefficients over L is that call type's impulse response: where it starts (latency), how
big it gets, and how long it lasts (persistence).

Ridge-regularised because neighbouring lags are collinear; the penalty is chosen by
generalised cross-validation, not by hand.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import re

import numpy as np

D = np.load("race.npz")
t, v, sr, dps = D["t"], D["v"], D["sr"], D["dps"]
tag = json.load(open("tagged.json"))
OFF = 1.0
NS = len(t)
LAGS = np.arange(0, 26)

SUBS = [
    ("split", "tactical", r"\b2[0-3]\d\b"),
    ("in-two / take-N", "tactical", r"in two|take (a )?(10|ten|five|5)|that's one|one, two"),
    ("steering", "tactical", r"starboard|port\b|corner|turn|lane|yield"),
    ("ratio/length", "technical", r"ratio|rhythm|\blong\b|length|send|reach|distance per"),
    ("catch/press", "technical", r"catch|front end|attack|direct|sharp|suspend|hang|press|squeeze"),
    ("body/together", "technical", r"bod(y|ies)|swing|sit up|together|complete|clean|tidy|sequenc"),
    ("motivational", "motivational", r"."),
]


def stroke_of(vt):
    return int(np.argmin(np.abs(t - (vt + OFF))))


cols, names = [], []
for name, theme, pat in SUBS:
    rx = re.compile(pat, re.I)
    hits = [stroke_of(x[0]) for x in tag if x[2] == theme and rx.search(x[1])]
    imp = np.zeros(NS)
    for i in hits:
        imp[i] += 1.0
    for L in LAGS:
        cols.append(np.roll(imp, L))
        names.append((name, L))
    print("%-16s %s  n=%d" % (name, theme[:4], len(hits)))

X = np.column_stack(cols)
# response: speed with the slow fade removed, so we measure the call not the race
k = np.ones(61) / 61
trend = np.convolve(np.pad(v, (30, 30), mode="edge"), k, mode="valid")[:NS]
y = v - trend
X = X - X.mean(axis=0)
y = y - y.mean()

U, S, Vt = np.linalg.svd(X, full_matrices=False)
best = None
for lam in np.logspace(-3, 3, 40):
    d = S / (S ** 2 + lam)
    beta = Vt.T @ (d * (U.T @ y))
    resid = y - X @ beta
    dof = float(np.sum(S ** 2 / (S ** 2 + lam)))
    gcv = float(np.sum(resid ** 2)) / max(NS - dof, 1) ** 2
    if best is None or gcv < best[0]:
        best = (gcv, lam, beta, dof)
_, lam, beta, dof = best
print()
print("ridge lambda %.3g by GCV, effective dof %.1f of %d columns" % (lam, dof, X.shape[1]))
print("R^2 on detrended speed: %.3f" % (1 - np.var(y - X @ beta) / np.var(y)))

B = {}
for (name, L), b in zip(names, beta):
    B.setdefault(name, np.zeros(len(LAGS)))[L] = b
np.savez("irf.npz", lags=LAGS, **B)

print()
print("impulse response, m/s per call, by strokes after the call")
print("  %-16s %s" % ("call type", "".join("%6d" % L for L in LAGS[::2])))
for name, _, _ in SUBS:
    b = B[name]
    print("  %-16s %s" % (name, "".join("%+6.3f" % x for x in b[::2])))
print()
print("  %-16s %8s %8s %10s %10s" % ("call type", "peak", "at lag", "onset>25%", "half-life"))
for name, _, _ in SUBS:
    b = B[name]
    pk = int(np.argmax(np.abs(b)))
    sign = np.sign(b[pk])
    onset = next((L for L in LAGS if sign * b[L] > 0.25 * abs(b[pk])), None)
    after = [L for L in LAGS[pk:] if sign * b[L] < 0.5 * abs(b[pk])]
    half = (after[0] - pk) if after else None
    print("  %-16s %+8.4f %8d %10s %10s"
          % (name, b[pk], pk,
             ("%d" % onset) if onset is not None else "-",
             ("%d" % half) if half is not None else ">25"))
