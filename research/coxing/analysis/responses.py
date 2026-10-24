"""Do calls change anything other than boat speed?

Speed is an identity: v = dps * sr / 60. A call that raises the rate and shortens the
stroke by the same proportion leaves speed untouched while genuinely changing what the
crew is doing. Speed is therefore the response least likely to show an effect, and the
cox-box logs four others. Each is run through the same validated machinery: functional
call categories, ridge-regularised distributed lag, blocked out-of-sample R2 scored
against a matched null of circular shifts.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np

sys.path.insert(0, "elite")
import vtt

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
for k, ws in FUNC.items():
    for w in ws:
        LOOKUP.setdefault(w, k)
CATS = list(FUNC)
LAGS = np.arange(0, 26)

R = np.load("race.npz")
t = R["t"]; NS = len(t)
norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum() or ch == "'")
hits = {k: np.zeros(NS) for k in CATS}
for tt, w in vtt.parse_words("caps.en.vtt"):
    k = LOOKUP.get(norm(w))
    if k:
        hits[k][int(np.argmin(np.abs(t - (tt + 1.0))))] += 1.0


def detrend(y, w=61):
    k = np.ones(w) / w
    return y - np.convolve(np.pad(y, (w // 2, w // 2), mode="edge"), k, mode="valid")[:len(y)]


def lagmat(s):
    X = np.column_stack([np.roll(s, L) for L in LAGS])
    return X - X.mean(axis=0)


def blocked_oos(X, y, nfold=5):
    n = len(y); e = np.linspace(0, n, nfold + 1).astype(int)
    num = den = 0.0
    for i in range(nfold):
        te = np.zeros(n, bool); te[e[i]:e[i + 1]] = True; tr = ~te
        Xtr = X[tr] - X[tr].mean(axis=0); m = y[tr].mean()
        U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
        bg = bl = None
        for L in np.logspace(-2, 5, 30):
            b = Vt.T @ ((S / (S ** 2 + L)) * (U.T @ (y[tr] - m)))
            dof = float(np.sum(S ** 2 / (S ** 2 + L)))
            g = float(np.sum((y[tr] - m - Xtr @ b) ** 2)) / max(tr.sum() - dof, 1) ** 2
            if bg is None or g < bg: bg, bl = g, L
        b = Vt.T @ ((S / (S ** 2 + bl)) * (U.T @ (y[tr] - m)))
        num += float(np.sum((y[te] - (X[te] @ b + m)) ** 2))
        den += float(np.sum((y[te] - m) ** 2))
    return 1 - num / den


RESP = [("speed  (m/s)", R["v"]), ("rate   (spm)", R["sr"]), ("dist/stroke (m)", R["dps"]),
        ("power  (W)", R["power"]), ("crew power (W)", R["crew_power"]),
        ("drag   (N)", R["drag"])]
rng = np.random.default_rng(101)
print("blocked out-of-sample R2 per call category, by response variable")
print("(bold-worthy = observed above the matched-null 95th percentile)")
print()
print("%-16s %8s | %s" % ("response", "sd", "  ".join("%-9s" % c for c in CATS)))
print("-" * 116)
store = {}
for nm, raw in RESP:
    y = detrend(np.asarray(raw, float)); y = y - y.mean()
    row = []
    for c in CATS:
        obs = blocked_oos(lagmat(hits[c]), y)
        null = [blocked_oos(lagmat(np.roll(hits[c], int(rng.integers(26, NS - 26)))), y)
                for _ in range(60)]
        thr = float(np.percentile(null, 95))
        star = "*" if obs > thr else " "
        row.append("%+.3f%s" % (obs, star))
        store[(nm, c)] = (obs, thr)
    print("%-16s %8.4f | %s" % (nm, y.std(), "  ".join("%-9s" % x for x in row)))
print()
print("* = above the 95th percentile of 60 circular shifts of that same call series")
np.save("_resp_store.npy", np.array([[store[(nm, c)][0] for c in CATS] for nm, _ in RESP]))
