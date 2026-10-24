"""Does a multi-resolution description of the sensory input predict calls better?

The earlier test summarised each stroke's motion by three numbers per channel. This
replaces that with octave-band energies, which separate the stroke rhythm from the
check and the blade vibration, and adds the acoustic bands the coxswain hears with
their own voice removed. If a coxswain calls the catch because of what a bad catch
feels and sounds like, the frequency content should carry more than the amplitude.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np

sys.path.insert(0, "elite")
from responses import CATS, hits
from vestib import VEST
from whycall import POSITION, BOX, blocked_auc, S

MOT = np.load("_wav_motion.npy")
SND = np.load("_wav_sound.npy")
W = np.column_stack([MOT, SND])
W = np.nan_to_num(W, nan=np.nanmedian(W))
print("blocks: POSITION %d, BOX %d, FELT-simple %d, WAVELET %d (%d motion + %d sound)"
      % (POSITION.shape[1], BOX.shape[1], VEST.shape[1], W.shape[1],
         MOT.shape[1], SND.shape[1]))

P, B, F, Wv = POSITION[S], BOX[S], VEST[S], W[S]
rng = np.random.default_rng(17)
print()
print("%-10s %5s %8s %9s %9s %9s %9s" %
      ("category", "n", "P+B", "+FELT", "null95", "+WAVE", "null95"))
print("-" * 64)
obs_grid = {}
for c in CATS:
    y = (hits[c][S] > 0).astype(int)
    if y.sum() < 12:
        continue
    a0 = blocked_auc(np.column_stack([P, B]), y)
    a1 = blocked_auc(np.column_stack([P, B, F]), y)
    a2 = blocked_auc(np.column_stack([P, B, F, Wv]), y)
    nf, nw = [], []
    for _ in range(40):
        s1 = int(rng.integers(20, len(S) - 20))
        nf.append(blocked_auc(np.column_stack([P, B, np.roll(F, s1, axis=0)]), y) - a0)
        s2 = int(rng.integers(20, len(S) - 20))
        nw.append(blocked_auc(np.column_stack([P, B, F, np.roll(Wv, s2, axis=0)]), y) - a1)
    tf, tw = float(np.nanpercentile(nf, 95)), float(np.nanpercentile(nw, 95))
    obs_grid[c] = (a1 - a0, a2 - a1)
    print("%-10s %5d %8.3f %+9.3f%s %9.3f %+9.3f%s %9.3f"
          % (c, y.sum(), a0, a1 - a0, "*" if (a1 - a0) > tf else " ", tf,
             a2 - a1, "*" if (a2 - a1) > tw else " ", tw))

# family-wise: the largest gain anywhere in the grid under shifted blocks
best_obs = max(max(v) for v in obs_grid.values())
mx = []
for _ in range(50):
    b = -9
    for c in CATS:
        y = (hits[c][S] > 0).astype(int)
        if y.sum() < 12:
            continue
        a0 = blocked_auc(np.column_stack([P, B]), y)
        a1 = blocked_auc(np.column_stack([P, B, F]), y)
        s1 = int(rng.integers(20, len(S) - 20)); s2 = int(rng.integers(20, len(S) - 20))
        g1 = blocked_auc(np.column_stack([P, B, np.roll(F, s1, axis=0)]), y) - a0
        g2 = blocked_auc(np.column_stack([P, B, F, np.roll(Wv, s2, axis=0)]), y) - a1
        b = max(b, np.nanmax([g1, g2]))
    mx.append(b)
mx = np.array(mx)
print()
print("largest observed gain anywhere: %+.3f" % best_obs)
print("family-wise null over the whole grid: 95th %+.3f, max %+.3f, p = %.3f"
      % (np.nanpercentile(mx, 95), np.nanmax(mx), float(np.mean(mx >= best_obs))))
