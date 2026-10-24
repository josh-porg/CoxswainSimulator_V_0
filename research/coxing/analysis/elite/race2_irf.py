"""Race 2 impulse responses, and whether race 1's catch result replicates.

Race 2 is read at about one sample per four seconds, so the analysis runs in time
rather than strokes: lags are four-second bins out to 44 s, which spans race 1's
peak at 21 strokes (about 42 s at that rate). With 50 usable readings the design is
far too small for a free search, so this is a targeted replication of one prior
hypothesis, with the other categories reported only descriptively.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import sys

import numpy as np

sys.path.insert(0, ".")
import vtt
from race2_readings import T, RATE, SPLIT, DPS, DIST, V

FUNC = {
    "power":   "press pressure power drive push legs hard strong squeeze send big"
               " more lean strength".split(),
    "catch":   "catch front quick direct together entry".split() + ["catches"],
    "finish":  "finish finishes through back release hands tap accelerate draw".split(),
    "length":  "long length reach lengthen full stretch longer out".split(),
    "ratio":   "ratio rhythm swing relax breathe patience slide recovery easy smooth"
               " control controlled".split(),
    "rate":    "rate rating stroke strokes timing time shift build settle sit".split()
               + ["28", "30", "32", "34", "26"],
    "motivat": "yes good great come fight dig believe now go well nice perfect"
               " brilliant lovely".split(),
    "tactical":"metres meters m gone left to go position walking seat seats open"
               " water bridge them their".split() + ["220", "500", "1000"],
}
LOOKUP = {}
for _k, _ws in FUNC.items():
    for _w in _ws:
        LOOKUP.setdefault(_w, _k)
CATS = list(FUNC)

words = vtt.parse_words("nat8.en.vtt")
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == "'")
ev = {c: [] for c in CATS}
for tt, w in words:
    k = LOOKUP.get(norm(w))
    if k and 100 <= tt <= 340:
        ev[k].append(tt)
print("race 2 call counts in the race window:")
print("   " + "  ".join("%s %d" % (c, len(ev[c])) for c in CATS))

ok = ~np.isnan(V)
t, v = T[ok], V[ok]
w = 9                                   # local trend over ~36 s
tr = np.convolve(np.pad(v, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")[:len(v)]
y = v - tr; y = y - y.mean()
print()
print("%d speed readings, stroke-scale sd after detrending %.4f m/s" % (len(y), y.std()))

BINS = [(a, a + 4) for a in range(0, 44, 4)]


def design(times):
    X = np.zeros((len(t), len(BINS)))
    for i, ti in enumerate(t):
        for j, (a, b) in enumerate(BINS):
            X[i, j] = sum(1 for e in times if a <= ti - e < b)
    return X - X.mean(0)


def ridge_fit(X, yy, lam):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ yy))


def blocked_oos(X, yy, nfold=5, lam=1.0):
    n = len(yy); e = np.linspace(0, n, nfold + 1).astype(int)
    num = den = 0.0
    for i in range(nfold):
        te = np.zeros(n, bool); te[e[i]:e[i + 1]] = True; tr_ = ~te
        Xt = X[tr_] - X[tr_].mean(0); m = yy[tr_].mean()
        b = ridge_fit(Xt, yy[tr_] - m, lam)
        num += float(np.sum((yy[te] - (X[te] @ b + m)) ** 2))
        den += float(np.sum((yy[te] - m) ** 2))
    return 1 - num / den


rng = np.random.default_rng(3)
print()
print("%-10s %6s %9s %10s %8s   %s" % ("category", "calls", "OOS R2", "null 95th", "p", "peak lag"))
print("-" * 68)
for c in CATS:
    if len(ev[c]) < 6:
        print("%-10s %6d   (too few)" % (c, len(ev[c]))); continue
    X = design(ev[c])
    obs = blocked_oos(X, y)
    null = []
    for _ in range(400):
        sh = float(rng.uniform(20, 200))
        null.append(blocked_oos(design([e + sh for e in ev[c]]), y))
    null = np.array(null)
    p = float(np.mean(null >= obs))
    b = ridge_fit(X, y, 1.0)
    pk = int(np.argmax(np.abs(b)))
    star = "*" if p < 0.05 else " "
    print("%-10s %6d %+9.4f %+10.4f %8.3f%s  %d-%d s (%+.1f mm/s)"
          % (c, len(ev[c]), obs, np.percentile(null, 95), p, star,
             BINS[pk][0], BINS[pk][1], 1000 * b[pk]))
print()
print("Shifting a category's call times by a random offset keeps its spacing and")
print("density and breaks only its alignment with the boat.")
