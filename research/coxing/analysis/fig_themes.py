"""Theme impulse responses with moving-block bootstrap intervals.

Uses the validated continuous theme intensities: technical by lexicon count (AUC 0.920),
motivational and tactical by cross-validated one-versus-rest probability (0.798, 0.790).
Intervals are pointwise 95% from 400 moving-block resamples with 40-stroke blocks.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

sys.path.insert(0, "elite")
import units
import vtt
from labels import labels
from lexicon import counts
from theme_model import build

THEMES = ["technical", "motivational", "tactical"]
LAGS = np.arange(0, 26)

d = json.load(open("to_label.json"))
Xl = np.array([x[2] for x in d]); yl = np.array(labels())
models = {}
for th, c in (("motivational", 4), ("tactical", 8)):
    m = build(c); m.fit(Xl, (yl == th).astype(int)); models[th] = m

R = np.load("race.npz")
t, v = R["t"], R["v"]
NS = len(t)
STROKE = float(np.median(np.diff(t)))
words = vtt.parse_words("caps.en.vtt")
inten = {th: np.zeros(NS) for th in THEMES}
win = units.windows(words, n=8, step=4)
wt = np.array([w[0] for w in win]); txt = [w[1] for w in win]
arrs = {"technical": np.array([counts(s)["technical"] for s in txt], dtype=float),
        "motivational": models["motivational"].predict_proba(txt)[:, 1],
        "tactical": models["tactical"].predict_proba(txt)[:, 1]}
for th, a in arrs.items():
    for tt, val in zip(wt, a):
        inten[th][int(np.argmin(np.abs(t - (tt + 1.0))))] += float(val)

cols, names = [], []
for th in THEMES:
    for L in LAGS:
        cols.append(np.roll(inten[th], L)); names.append((th, L))
X = np.column_stack(cols); X = X - X.mean(0)
k = np.ones(61) / 61
y = v - np.convolve(np.pad(v, (30, 30), mode="edge"), k, mode="valid")[:NS]
y = y - y.mean()


def ridge(Xm, yy, lam):
    U, S, Vt = np.linalg.svd(Xm, full_matrices=False)
    return Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ yy))


best = None
for lam in np.logspace(-3, 4, 50):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    b = Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ y))
    dof = float(np.sum(S ** 2 / (S ** 2 + lam)))
    gcv = float(np.sum((y - X @ b) ** 2)) / max(NS - dof, 1) ** 2
    if best is None or gcv < best[0]:
        best = (gcv, lam, b)
_, LAM, beta = best
B = {th: np.zeros(len(LAGS)) for th in THEMES}
for (th, L), c in zip(names, beta):
    B[th][L] = c

rng = np.random.default_rng(7)
block, nb = 40, 400
starts = np.arange(0, NS - block)
draws = {th: np.zeros((nb, len(LAGS))) for th in THEMES}
for i in range(nb):
    idx = np.concatenate([np.arange(st, st + block) for st in rng.choice(starts, NS // block)])
    idx = idx[idx < NS]
    bb = ridge(X[idx] - X[idx].mean(0), y[idx] - y[idx].mean(), LAM)
    tmp = {th: np.zeros(len(LAGS)) for th in THEMES}
    for (th, L), c in zip(names, bb):
        tmp[th][L] = c
    for th in THEMES:
        draws[th][i] = tmp[th]

COL = {"technical": "#c8622a", "motivational": "#3f7d78", "tactical": "#5b6b9a"}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
tt = LAGS * STROKE
for a, th in zip(ax, THEMES):
    lo = np.percentile(draws[th], 2.5, axis=0); hi = np.percentile(draws[th], 97.5, axis=0)
    a.axhline(0, color="0.5", lw=.8, ls="--")
    a.fill_between(tt, 1000 * lo, 1000 * hi, color=COL[th], alpha=.20, lw=0)
    a.plot(tt, 1000 * B[th], color=COL[th], lw=1.8)
    a.set_title(th, fontsize=9)
    a.set_xlabel("seconds after a call")
ax[0].set_ylabel("mm s$^{-1}$ per unit intensity")
fig.tight_layout()
fig.savefig("fig_themes.pdf"); fig.savefig("fig_themes.png", dpi=300)
for th in THEMES:
    pk = int(np.argmax(np.abs(B[th])))
    lo = np.percentile(draws[th][:, pk], 2.5); hi = np.percentile(draws[th][:, pk], 97.5)
    print("%-13s peak %+6.2f mm/s at %4.1f s  95%% CI [%+.2f, %+.2f]%s"
          % (th, 1000 * B[th][pk], tt[pk], 1000 * lo, 1000 * hi,
             "  excludes 0" if lo > 0 or hi < 0 else ""))
print("ridge lambda %.3g; R2 %.3f" % (LAM, 1 - np.var(y - X @ beta) / np.var(y)))
