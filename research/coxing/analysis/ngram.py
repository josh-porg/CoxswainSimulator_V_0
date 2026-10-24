"""Sequence structure in the calls, and whether the boat adds anything to it.

This is the best-powered version of the trigger question. Rather than asking what
predicts a call from the boat alone, it first asks how much of the next call is
predictable from the calls that preceded it -- an n-gram / skip-gram model of the
coxswain's own speech -- and then asks what the boat state adds on top.

If coxing is largely scripted or habitual, call history should predict well and the
boat should add little. If it is reactive, the boat should add on top of history.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "elite")
from responses import CATS, hits, NS, R
from vestib import VEST
from whycall import POSITION, BOX, blocked_auc, S

M = np.column_stack([(hits[c][S] > 0).astype(float) for c in CATS])   # speaking strokes x cats
n, K = M.shape
print("%d speaking strokes, %d categories" % (n, K))
print("category frequencies:", dict(zip(CATS, M.sum(0).astype(int))))

# n-grams: the previous 1 and 2 calls.  skip-grams: calls 3 and 5 back.
HIST, HN = [], []
for lag in (1, 2, 3, 5):
    sh = np.vstack([np.zeros((lag, K)), M[:-lag]])
    HIST.append(sh); HN += ["%s[-%d]" % (c, lag) for c in CATS]
# local rate of each category over the previous 8 calls: the "what have I been on about"
win = np.zeros_like(M)
for i in range(n):
    lo = max(0, i - 8)
    if i > lo: win[i] = M[lo:i].mean(0)
HIST.append(win); HN += ["%s rate8" % c for c in CATS]
HIST = np.column_stack(HIST)
print("history block: %d columns (4 n/skip-gram lags + an 8-call running rate)" % HIST.shape[1])

P, B, F = POSITION[S], BOX[S], VEST[S]
rng = np.random.default_rng(41)
print()
print("%-10s %5s %8s %9s %9s %9s %9s" %
      ("category", "n", "POS", "+HIST", "null95", "+BOX+FELT", "null95"))
print("-" * 64)
res = {}
for j, c in enumerate(CATS):
    y = M[:, j].astype(int)
    if y.sum() < 12:
        print("%-10s %5d  (too few)" % (c, y.sum())); continue
    a0 = blocked_auc(P, y)
    a1 = blocked_auc(np.column_stack([P, HIST]), y)
    a2 = blocked_auc(np.column_stack([P, HIST, B, F]), y)
    nh, nx = [], []
    for _ in range(40):
        s1 = int(rng.integers(20, n - 20))
        nh.append(blocked_auc(np.column_stack([P, np.roll(HIST, s1, axis=0)]), y) - a0)
        s2 = int(rng.integers(20, n - 20))
        nx.append(blocked_auc(np.column_stack([P, HIST, np.roll(np.column_stack([B, F]),
                                                                s2, axis=0)]), y) - a1)
    th = float(np.nanpercentile(nh, 95)); tx = float(np.nanpercentile(nx, 95))
    sh_ = "*" if (a1 - a0) > th else " "
    sx = "*" if (a2 - a1) > tx else " "
    res[c] = (a0, a1, a2, th, tx)
    print("%-10s %5d %8.3f %+9.3f%s %9.3f %+9.3f%s %9.3f"
          % (c, y.sum(), a0, a1 - a0, sh_, th, a2 - a1, sx, tx))
print()
print("POS = race position only. +HIST = what the coxswain has just been saying")
print("(n-grams at lags 1-2, skip-grams at 3 and 5, plus an 8-call running rate).")
print("+BOX+FELT = the cox-box display and the vestibular input added on top.")
