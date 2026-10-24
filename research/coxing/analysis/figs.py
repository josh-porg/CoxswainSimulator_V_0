"""Figures for the paper.

fig_catch: the catch impulse response in race 1 with moving-block bootstrap intervals,
beside the same quantity estimated on the two independent races, both on a time axis so
they are directly comparable.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

sys.path.insert(0, "elite")
from responses import hits, NS, R, detrend, LAGS

STROKE = float(np.median(np.diff(R["t"])))      # seconds per stroke, race 1
y = detrend(R["v"]); y = y - y.mean()


def lagmat(s, lags=LAGS):
    X = np.column_stack([np.roll(s, L) for L in lags])
    return X - X.mean(0)


def ridge(X, yy, lam):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ yy))


def boot_ci(s, lam=96.2, nb=400, block=40, seed=7):
    rng = np.random.default_rng(seed)
    X = lagmat(s)
    b = ridge(X, y, lam)
    starts = np.arange(0, NS - block)
    B = np.zeros((nb, len(LAGS)))
    for i in range(nb):
        idx = np.concatenate([np.arange(st, st + block)
                              for st in rng.choice(starts, NS // block)])
        idx = idx[idx < NS]
        B[i] = ridge(X[idx] - X[idx].mean(0), y[idx] - y[idx].mean(), lam)
    return b, np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)


# ---------- right panel: pooled races 2 and 3 ----------
sys.path.insert(0, "elite")
import vtt
from race2_readings import T as T2, V as V2
from race3_readings import T as T3, V as V3

CATCH = set("catch catches front quick direct entry together".split())
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == "'")
RACES = [("elite/nat8.en.vtt", T2, V2, 108, 330),
         ("elite/cap_SQQTqYmb_Dg.en.vtt", T3, V3, 102, 312)]
BINS = [(a, a + 8) for a in range(0, 48, 8)]


def pooled_fit():
    Xs, ys = [], []
    for cap, T, V, lo, hi in RACES:
        ev = [t for t, w in vtt.parse_words(cap) if lo - 48 <= t <= hi and norm(w) in CATCH]
        ok = np.isfinite(V)
        t, v = T[ok], V[ok]
        w = 7
        tr = np.convolve(np.pad(v, (w // 2, w // 2), mode="edge"),
                         np.ones(w) / w, mode="valid")[:len(v)]
        yy = v - tr
        X = np.zeros((len(t), len(BINS)))
        for r, ti in enumerate(t):
            for j, (a, b) in enumerate(BINS):
                X[r, j] = sum(1 for e in ev if a <= ti - e < b)
        Xs.append(X - X.mean(0)); ys.append(yy - yy.mean())
    X = np.vstack(Xs); yy = np.concatenate(ys)
    b = ridge(X, yy, 2.0)
    rng = np.random.default_rng(4)
    B = np.zeros((400, len(BINS)))
    n = len(yy)
    for i in range(400):
        idx = rng.integers(0, n, n)
        B[i] = ridge(X[idx] - X[idx].mean(0), yy[idx] - yy[idx].mean(), 2.0)
    return b, np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)


fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))

b, lo, hi = boot_ci(hits["catch"])
tt = LAGS * STROKE
ax[0].axhline(0, color="0.5", lw=.8, ls="--")
ax[0].fill_between(tt, 1000 * lo, 1000 * hi, color="#c8622a", alpha=.20, lw=0)
ax[0].plot(tt, 1000 * b, color="#c8622a", lw=1.8)
pk = int(np.argmax(np.abs(b)))
ax[0].plot([tt[pk]], [1000 * b[pk]], "o", color="#c8622a", ms=4)
ax[0].annotate("$-9.7$ mm s$^{-1}$", (tt[pk], 1000 * b[pk]),
               textcoords="offset points", xytext=(-4, -16), fontsize=7.5, ha="right")
ax[0].set_title("Race 1: 21 catch calls, 576 strokes", fontsize=8.5)
ax[0].set_xlabel("seconds after a catch call")
ax[0].set_ylabel("mm s$^{-1}$ per call")

pb, plo, phi = pooled_fit()
ctr = np.array([(a + b_) / 2 for a, b_ in BINS])
ax[1].axhline(0, color="0.5", lw=.8, ls="--")
ax[1].fill_between(ctr, 1000 * plo, 1000 * phi, color="#3f7d78", alpha=.20, lw=0)
ax[1].plot(ctr, 1000 * pb, color="#3f7d78", lw=1.8, marker="o", ms=3)
ax[1].set_title("Races 2+3 pooled: 10 catch calls, 89 rows", fontsize=8.5)
ax[1].set_xlabel("seconds after a catch call")
ax[1].set_ylabel("mm s$^{-1}$ per call")
ax[1].axhline(1000 * b[pk], color="#c8622a", lw=1.1, ls=":")
ax[1].annotate("race 1 estimate", (1.0, 1000 * b[pk]), textcoords="offset points",
               xytext=(-4, 5), fontsize=7, color="#c8622a", ha="left")
ax[1].set_xlim(ax[0].get_xlim())

fig.tight_layout()
fig.savefig("fig_catch.pdf"); fig.savefig("fig_catch.png", dpi=300)
print("wrote fig_catch; race 1 peak %.1f mm/s at %.0f s; pooled range %.1f to %.1f mm/s"
      % (1000 * b[pk], tt[pk], 1000 * pb.min(), 1000 * pb.max()))
