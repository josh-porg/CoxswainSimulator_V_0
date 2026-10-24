"""Unsupervised call groupings, found without imposing any taxonomy.

The functional categories were hand-built, which risks finding what was put in. This
asks the transcript to group its own calls: windows are vectorised by their words and
clustered, with speed playing no part in forming the clusters. Only afterwards is each
cluster's impulse response fitted and validated.

Every cluster is scored by blocked out-of-sample R2 against a matched null of circular
shifts, and the best cluster is compared against the distribution of the best under
those shifts, so searching k clusters is paid for.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, "elite")
import units
import vtt

LAGS = np.arange(0, 26)
R = np.load("race.npz")
t, v = R["t"], R["v"]
NS = len(t)
words = vtt.parse_words("caps.en.vtt")
win = units.windows(words, n=8, step=4)
wt = np.array([w[0] for w in win])
txt = [w[1] for w in win]
print("%d windows of 8 words" % len(txt))

k61 = np.ones(61) / 61
trend = np.convolve(np.pad(v, (30, 30), mode="edge"), k61, mode="valid")[:NS]
y = v - trend
y = y - y.mean()


def blocked_oos(X, y, nfold=5):
    n = len(y)
    e = np.linspace(0, n, nfold + 1).astype(int)
    num = den = 0.0
    for i in range(nfold):
        te = np.zeros(n, bool); te[e[i]:e[i + 1]] = True
        tr = ~te
        Xtr = X[tr] - X[tr].mean(axis=0); m = y[tr].mean()
        U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
        bg, bl = None, None
        for L in np.logspace(-2, 5, 30):
            b = Vt.T @ ((S / (S ** 2 + L)) * (U.T @ (y[tr] - m)))
            dof = float(np.sum(S ** 2 / (S ** 2 + L)))
            g = float(np.sum((y[tr] - m - Xtr @ b) ** 2)) / max(tr.sum() - dof, 1) ** 2
            if bg is None or g < bg: bg, bl = g, L
        b = Vt.T @ ((S / (S ** 2 + bl)) * (U.T @ (y[tr] - m)))
        num += float(np.sum((y[te] - (X[te] @ b + m)) ** 2))
        den += float(np.sum((y[te] - m) ** 2))
    return 1 - num / den


def series(mask):
    s = np.zeros(NS)
    for tt in wt[mask]:
        s[int(np.argmin(np.abs(t - (tt + 1.0))))] += 1.0
    return s


def lagmat(s):
    X = np.column_stack([np.roll(s, L) for L in LAGS])
    return X - X.mean(axis=0)


vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english")
M = vec.fit_transform(txt)
rng = np.random.default_rng(31)

for K in (4, 6, 8, 10):
    lab = KMeans(K, n_init=10, random_state=0).fit_predict(M)
    obs, sizes = [], []
    for c in range(K):
        m = lab == c
        sizes.append(int(m.sum()))
        obs.append(blocked_oos(lagmat(series(m)), y))
    obs = np.array(obs)
    bi = int(np.argmax(obs))
    # null for the BEST of K, preserving each cluster's own event spacing
    nm = []
    for _ in range(120):
        sc = []
        for c in range(K):
            s = series(lab == c)
            sc.append(blocked_oos(lagmat(np.roll(s, int(rng.integers(26, NS - 26)))), y))
        nm.append(max(sc))
    nm = np.array(nm)
    p = float(np.mean(nm >= obs[bi]))
    top = vec.get_feature_names_out()[np.argsort(M[lab == bi].mean(axis=0).A1)[-6:]][::-1]
    print()
    print("k=%2d  best cluster %d (n=%d windows)  OOS R2 %+.4f   best-of-k null 95th %+.4f   p=%.3f"
          % (K, bi, sizes[bi], obs[bi], np.percentile(nm, 95), p))
    print("      all clusters: %s" % "  ".join("%+.3f(n=%d)" % (o, s) for o, s in zip(obs, sizes)))
    print("      top terms in best cluster: %s" % ", ".join(top))
