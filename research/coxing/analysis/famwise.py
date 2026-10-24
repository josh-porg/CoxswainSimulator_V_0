"""A stable family-wise test for the whole trigger analysis.

Two earlier runs of this correction returned p = 0.167 and p = 0.020 for the same
observed effect, because each estimated a 95th percentile from only 50-60 draws. That
spread is a property of the null estimate, not of the data, and it decides whether the
catch result is reportable. This fixes the grid in advance -- 8 categories x 3 block
additions -- and estimates the distribution of the largest gain anywhere in it from
400 draws.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np

sys.path.insert(0, "elite")
from responses import CATS, hits
from vestib import VEST
from whycall import POSITION, BOX, blocked_auc, S

MOT = np.load("_wav_motion.npy"); SND = np.load("_wav_sound.npy")
W = np.nan_to_num(np.column_stack([MOT, SND]))
P, B, F, Wv = POSITION[S], BOX[S], VEST[S], W[S]
Y = {c: (hits[c][S] > 0).astype(int) for c in CATS if (hits[c][S] > 0).sum() >= 12}
print("grid: %d categories x 3 block additions = %d tests" % (len(Y), 3 * len(Y)))

def gains(Bx, Fx, Wx):
    out = []
    for c, y in Y.items():
        a0 = blocked_auc(P, y)
        a1 = blocked_auc(np.column_stack([P, Bx]), y)
        a2 = blocked_auc(np.column_stack([P, Bx, Fx]), y)
        a3 = blocked_auc(np.column_stack([P, Bx, Fx, Wx]), y)
        out += [a1 - a0, a2 - a1, a3 - a2]
    return np.array(out, float)

obs = gains(B, F, Wv)
names = [(c, blk) for c in Y for blk in ("BOX", "FELT", "WAVE")]
i = int(np.nanargmax(obs))
print("largest observed gain: %+.3f  (%s, %s)" % (obs[i], names[i][0], names[i][1]))

rng = np.random.default_rng(2024)
mx = np.zeros(400)
for k in range(400):
    s1, s2, s3 = (int(rng.integers(20, len(S) - 20)) for _ in range(3))
    g = gains(np.roll(B, s1, axis=0), np.roll(F, s2, axis=0), np.roll(Wv, s3, axis=0))
    mx[k] = np.nanmax(g)
p = float(np.mean(mx >= obs[i]))
print("family-wise null from 400 draws: mean %+.3f, 95th %+.3f, 99th %+.3f, max %+.3f"
      % (mx.mean(), np.percentile(mx, 95), np.percentile(mx, 99), mx.max()))
print("corrected p for the largest gain = %.3f" % p)
se = float(np.sqrt(p * (1 - p) / 400))
print("Monte Carlo standard error on that p: %.3f (so p is %.3f +- %.3f)" % (se, p, 1.96 * se))
np.save("_famwise_null.npy", mx)
