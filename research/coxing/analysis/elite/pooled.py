"""Pooled replication: races 2 and 3, independent of race 1.

Race 1 found a catch effect twice, under keyword and under LLM labelling. Neither of
those is independent data. Races 2 and 3 are, and are pooled here because neither alone
has the events to test anything: race 2 carries 1 catch call, race 3 carries 9.

Both races are read at 4-5 s spacing, so the analysis runs in time. Lags are 8 s bins
out to 40 s, which spans race 1's peak at 21 strokes (about 42 s). Rows from the two
races are stacked with race-specific centring, and the null shifts each race's call
times independently, preserving their spacing within race.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import sys

import numpy as np

sys.path.insert(0, ".")
import vtt
from race2_readings import T as T2, V as V2
from race3_readings import T as T3, V as V3

CATCH = set("catch catches front quick direct entry together".split())
TECH = set(("catch catches front quick direct entry together finish finishes release "
            "hands tap draw long length reach lengthen full stretch ratio rhythm swing "
            "relax breathe patience slide recovery smooth press pressure power drive "
            "push legs squeeze send").split())
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == "'")


def calls(path, lo, hi, vocab):
    return [t for t, w in vtt.parse_words(path) if lo <= t <= hi and norm(w) in vocab]


RACES = [("race2", "nat8.en.vtt", T2, V2, 108, 330),
         ("race3", "cap_SQQTqYmb_Dg.en.vtt", T3, V3, 102, 312)]
BINS = [(a, a + 8) for a in range(0, 40, 8)]


def build(vocab):
    Xs, ys, gid = [], [], []
    for i, (nm, cap, T, V, lo, hi) in enumerate(RACES):
        ev = calls(cap, lo - 40, hi, vocab)
        ok = np.isfinite(V)
        t, v = T[ok], V[ok]
        w = 7
        tr = np.convolve(np.pad(v, (w // 2, w // 2), mode="edge"),
                         np.ones(w) / w, mode="valid")[:len(v)]
        y = v - tr
        X = np.zeros((len(t), len(BINS)))
        for r, ti in enumerate(t):
            for j, (a, b) in enumerate(BINS):
                X[r, j] = sum(1 for e in ev if a <= ti - e < b)
        Xs.append(X - X.mean(0)); ys.append(y - y.mean()); gid += [i] * len(t)
    return np.vstack(Xs), np.concatenate(ys), np.array(gid)


def blocked_oos(X, y, g, nfold=4, lam=2.0):
    num = den = 0.0
    for i in range(nfold):
        te = np.zeros(len(y), bool)
        for gi in np.unique(g):
            idx = np.flatnonzero(g == gi)
            e = np.linspace(0, len(idx), nfold + 1).astype(int)
            te[idx[e[i]:e[i + 1]]] = True
        tr = ~te
        Xt = X[tr] - X[tr].mean(0); m = y[tr].mean()
        U, S, Vt = np.linalg.svd(Xt, full_matrices=False)
        b = Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ (y[tr] - m)))
        num += float(np.sum((y[te] - (X[te] @ b + m)) ** 2))
        den += float(np.sum((y[te] - m) ** 2))
    return 1 - num / den


rng = np.random.default_rng(9)
for lbl, vocab in (("catch", CATCH), ("all technical", TECH)):
    ncall = sum(len(calls(c, lo - 40, hi, vocab)) for _, c, _, _, lo, hi in RACES)
    X, y, g = build(vocab)
    obs = blocked_oos(X, y, g)
    null = []
    for _ in range(600):
        Xs = []
        for gi in np.unique(g):
            idx = np.flatnonzero(g == gi)
            sh = int(rng.integers(3, len(idx) - 3))
            Xs.append(np.roll(X[idx], sh, axis=0))
        null.append(blocked_oos(np.vstack(Xs), y, g))
    null = np.array(null)
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    print("%-14s %3d calls, %3d rows: OOS R2 %+.4f, null 95th %+.4f, p = %.4f%s"
          % (lbl, ncall, len(y), obs, np.percentile(null, 95), p, "*" if p < 0.05 else ""))
