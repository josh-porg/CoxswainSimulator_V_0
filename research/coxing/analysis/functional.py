"""Functional call categories, finer than the three themes.

The theme split is inherited from thematic coding and pools calls that do quite
different jobs. "Technical" contains a catch call and a ratio call; if one moves the
boat and the other does not, averaging them hides both. These are the categories a
coach actually distinguishes -- power, catch, finish, length, ratio/rhythm, rate --
kept separate, with motivational and tactical retained for comparison.

Terms were chosen from this race's own vocabulary, not invented: every entry below
occurs in the transcript. Each term belongs to one category only; where a word is
genuinely ambiguous ("send" is both power and run) it is assigned to its dominant
sense and the joint ridge absorbs the residual correlation.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import sys

import numpy as np

sys.path.insert(0, "elite")
import vtt

FUNC = {
    "power":   "press pressure power drive push legs hard strong squeeze send big"
               " more lean strength".split(),
    "catch":   "catch front quick direct together entry in't".split() + ["catches"],
    "finish":  "finish finishes through back release hands tap accelerate draw".split(),
    "length":  "long length reach lengthen full stretch longer out".split(),
    "ratio":   "ratio rhythm swing relax breathe patience slide recovery easy smooth"
               " control controlled".split(),
    "rate":    "rate rating stroke strokes timing time shift build settle sit"
               " together'".split() + ["28", "30", "32", "34", "26"],
    "motivat": "yes good great come fight dig believe now go great well nice perfect"
               " brilliant lovely".split(),
    "tactical":"metres meters m gone left to go position walking seat seats open"
               " water bridge them they're their".split() + ["220", "500", "1000"],
}
CATS = list(FUNC)
LOOKUP = {}
for k, ws in FUNC.items():
    for w in ws:
        LOOKUP.setdefault(w, k)

LAGS = np.arange(0, 26)
R = np.load("race.npz")
t, v = R["t"], R["v"]
NS = len(t)
OFF = 1.0

norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == "'")
words = vtt.parse_words("caps.en.vtt")
hits = {k: np.zeros(NS) for k in CATS}
tot = {k: 0 for k in CATS}
for tt, w in words:
    k = LOOKUP.get(norm(w))
    if k is None:
        continue
    i = int(np.argmin(np.abs(t - (tt + OFF))))
    hits[k][i] += 1.0
    tot[k] += 1

print("category occurrences over %d strokes" % NS)
for k in CATS:
    print("   %-9s %4d   (%.2f per stroke)" % (k, tot[k], tot[k] / NS))
live = [k for k in CATS if tot[k] >= 10]
print("carried forward (>=10 occurrences): %s" % ", ".join(live))

cols, names = [], []
for k in live:
    for L in LAGS:
        cols.append(np.roll(hits[k], L)); names.append((k, L))
X = np.column_stack(cols)
kern = np.ones(61) / 61
trend = np.convolve(np.pad(v, (30, 30), mode="edge"), kern, mode="valid")[:NS]
y = v - trend
X = X - X.mean(axis=0)
y = y - y.mean()

U, S, Vt = np.linalg.svd(X, full_matrices=False)
best = None
for lam in np.logspace(-3, 4, 60):
    b = Vt.T @ ((S / (S ** 2 + lam)) * (U.T @ y))
    dof = float(np.sum(S ** 2 / (S ** 2 + lam)))
    gcv = float(np.sum((y - X @ b) ** 2)) / max(NS - dof, 1) ** 2
    if best is None or gcv < best[0]:
        best = (gcv, lam, b, dof)
_, LAM, beta, dof = best
R2 = 1 - np.var(y - X @ beta) / np.var(y)
print()
print("ridge lambda %.3g (GCV), effective dof %.1f of %d, R2 %.4f"
      % (LAM, dof, X.shape[1], R2))

B = {}
for (k, L), b in zip(names, beta):
    B.setdefault(k, np.zeros(len(LAGS)))[L] = b

def fit(idx):
    Us, Ss, Vts = np.linalg.svd(X[idx], full_matrices=False)
    return Vts.T @ ((Ss / (Ss ** 2 + LAM)) * (Us.T @ y[idx]))

rng = np.random.default_rng(11)
BLOCK, NB = 40, 400
starts = np.arange(0, NS - BLOCK)
boot = np.zeros((NB, X.shape[1]))
for i in range(NB):
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, NS // BLOCK)])
    boot[i] = fit(idx[idx < NS])

# FAMILY-WISE null: circularly shift each category's series independently, which
# destroys the call-to-speed timing while preserving each series' own autocorrelation
# and the speed autocorrelation. The max |t| over all categories and lags gives a
# threshold that already accounts for searching 8 categories x 26 lags.
NP = 400
maxstat = np.zeros(NP)
for i in range(NP):
    Xp = np.column_stack([np.roll(hits[k], int(rng.integers(30, NS - 30)))
                          for k in live for _ in [0]])
    cp = []
    for j, k in enumerate(live):
        sh = int(rng.integers(30, NS - 30))
        s_ = np.roll(hits[k], sh)
        for L in LAGS:
            cp.append(np.roll(s_, L))
    Xp = np.column_stack(cp); Xp = Xp - Xp.mean(axis=0)
    Up, Sp, Vtp = np.linalg.svd(Xp, full_matrices=False)
    bp = Vtp.T @ ((Sp / (Sp ** 2 + LAM)) * (Up.T @ y))
    maxstat[i] = np.abs(bp).max()
thr = float(np.percentile(maxstat, 95))

print()
print("%-9s %5s %11s %7s %24s %10s" % ("category", "n", "peak mm/s", "lag", "95% CI", "fam-wise"))
print("-" * 76)
out = {}
for k in live:
    A = np.array([bb[[names.index((k, L)) for L in LAGS]] for bb in boot])
    b = B[k]
    pk = int(np.argmax(np.abs(b)))
    lo, hi = np.percentile(A[:, pk], [2.5, 97.5])
    sig = "*" if (lo > 0 or hi < 0) else " "
    fam = "YES" if abs(b[pk]) > thr else "no"
    out[k] = dict(n=tot[k], peak=float(b[pk]), lag=pk, lo=float(lo), hi=float(hi),
                  fam=fam, curve=[float(x) for x in b])
    print("%-9s %5d %+11.3f %7d  [%+7.3f,%+7.3f]%s %9s"
          % (k, tot[k], 1000 * b[pk], pk, 1000 * lo, 1000 * hi, sig, fam))
print()
print("family-wise 95%% threshold on |beta| from 400 circular shifts: %.3f mm/s" % (1000 * thr))
print("* = pointwise bootstrap interval excludes zero; fam-wise = survives the")
print("   multiplicity correction for searching %d categories x %d lags."
      % (len(live), len(LAGS)))
json.dump(dict(R2=float(R2), thr=float(thr), cats=out), open("functional.json", "w"), indent=1)
