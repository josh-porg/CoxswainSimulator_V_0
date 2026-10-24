"""Impulse responses using the validated continuous theme-intensity regressors.

The earlier fit used single-label call counts from a classifier that cross-validates at
0.52 macro F1 -- barely above the majority class. This uses the measures that passed
validation instead: technical by lexicon count (AUC 0.920), motivational and tactical
by learned one-vs-rest probability (0.798, 0.790). Each stroke carries a continuous
intensity for each theme, so nothing is forced into one category.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import sys

import numpy as np

sys.path.insert(0, "elite")
import units
import vtt
from labels import labels
from lexicon import counts
from theme_model import build

THEMES = ["technical", "motivational", "tactical"]
LAGS = np.arange(0, 26)

d = json.load(open("to_label.json"))
Xl = np.array([x[2] for x in d])
yl = np.array(labels())
models = {}
for th, c in (("motivational", 4), ("tactical", 8)):
    m = build(c)
    m.fit(Xl, (yl == th).astype(int))
    models[th] = m

R = np.load("race.npz")
t, v = R["t"], R["v"]
NS = len(t)
words = vtt.parse_words("caps.en.vtt")
OFF = 1.0

# theme intensity attached to the stroke each word lands on
inten = {th: np.zeros(NS) for th in THEMES}
win = units.windows(words, n=8, step=4)
wt = np.array([w[0] for w in win])
txt = [w[1] for w in win]
tech = np.array([counts(s)["technical"] for s in txt], dtype=float)
moti = models["motivational"].predict_proba(txt)[:, 1]
tact = models["tactical"].predict_proba(txt)[:, 1]
for arr, th in ((tech, "technical"), (moti, "motivational"), (tact, "tactical")):
    for tt, val in zip(wt, arr):
        i = int(np.argmin(np.abs(t - (tt + OFF))))
        inten[th][i] += float(val)

cols, names = [], []
for th in THEMES:
    imp = inten[th]
    print("%-14s total intensity %.1f over %d strokes" % (th, imp.sum(), NS))
    for L in LAGS:
        cols.append(np.roll(imp, L))
        names.append((th, L))
X = np.column_stack(cols)
k = np.ones(61) / 61
trend = np.convolve(np.pad(v, (30, 30), mode="edge"), k, mode="valid")[:NS]
y = v - trend
X = X - X.mean(axis=0)
y = y - y.mean()

U, S, Vt = np.linalg.svd(X, full_matrices=False)
best = None
for lam in np.logspace(-3, 4, 50):
    dd = S / (S ** 2 + lam)
    b = Vt.T @ (dd * (U.T @ y))
    resid = y - X @ b
    dof = float(np.sum(S ** 2 / (S ** 2 + lam)))
    gcv = float(np.sum(resid ** 2)) / max(NS - dof, 1) ** 2
    if best is None or gcv < best[0]:
        best = (gcv, lam, b, dof)
_, LAM, beta, dof = best
print()
print("ridge lambda %.3g (GCV), effective dof %.1f of %d, R2 %.3f"
      % (LAM, dof, X.shape[1], 1 - np.var(y - X @ beta) / np.var(y)))

B = {}
for (th, L), b in zip(names, beta):
    B.setdefault(th, np.zeros(len(LAGS)))[L] = b

rng = np.random.default_rng(7)
BLOCK, NB = 40, 400
starts = np.arange(0, NS - BLOCK)
boots = {th: [] for th in THEMES}
for _ in range(NB):
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, NS // BLOCK)])
    idx = idx[idx < NS]
    Us, Ss, Vts = np.linalg.svd(X[idx], full_matrices=False)
    bb = Vts.T @ ((Ss / (Ss ** 2 + LAM)) * (Us.T @ y[idx]))
    tmp = {}
    for (th, L), val in zip(names, bb):
        tmp.setdefault(th, np.zeros(len(LAGS)))[L] = val
    for th in THEMES:
        boots[th].append(tmp[th])

print()
print("%-14s %11s %7s %22s %24s" % ("theme", "peak m/s", "lag", "95% CI on peak", "cumulative lags 0-9"))
out = {}
for th in THEMES:
    A = np.array(boots[th])
    b = B[th]
    pk = int(np.argmax(np.abs(b)))
    lo, hi = np.percentile(A[:, pk], [2.5, 97.5])
    area = b[:10].sum()
    alo, ahi = np.percentile(A[:, :10].sum(axis=1), [2.5, 97.5])
    sig = "*" if lo > 0 or hi < 0 else " "
    sa = "*" if alo > 0 or ahi < 0 else " "
    out[th] = dict(peak=float(b[pk]), lag=pk, lo=float(lo), hi=float(hi),
                   area=float(area), alo=float(alo), ahi=float(ahi),
                   curve=[float(x) for x in b])
    print("%-14s %+11.5f %7d  [%+.5f,%+.5f]%s  %+.4f [%+.4f,%+.4f]%s"
          % (th, b[pk], pk, lo, hi, sig, area, alo, ahi, sa))
json.dump(out, open("irf2.json", "w"), indent=1)
print()
print("* = 95%% moving-block bootstrap interval excludes zero (%d resamples)" % NB)
