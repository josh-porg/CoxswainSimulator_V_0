"""Sync the video clock to the boat data using the cox's own split callouts.

The cox reads the CoxBox aloud through the race -- "you're at a 228", "220", "219".
Those are instrument reads, so they should match the measured split trace exactly at
the right offset. Sliding the called splits against the measured ones finds the offset,
and how well they match at the optimum says how reliable the sync is.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np

D = np.load("race.npz")
t, v = D["t"], D["v"]
split = 500.0 / v          # seconds per 500 m, per stroke


def vsec(s):
    m, ss = s.split(":")
    return int(m) * 60 + int(ss)


# (video time, split the cox read out) -- instrument reads only, not targets or
# references to other crews
CALLS = [("1:19", 148), ("3:56", 146), ("5:34", 144), ("5:48", 140), ("6:28", 144),
         ("6:49", 139), ("6:55", 142), ("7:56", 140), ("9:17", 146), ("9:37", 140),
         ("11:22", 143), ("11:57", 140), ("15:39", 146), ("15:52", 148),
         ("16:37", 141), ("18:27", 141), ("18:43", 138), ("19:10", 138)]
# splits are mm:ss -> 2:28 = 148 s etc.

vt = np.array([vsec(a) for a, _ in CALLS], float)
cs = np.array([b for _, b in CALLS], float)

best = None
for off in np.arange(-60.0, 60.1, 0.5):
    pred = np.interp(vt + off, t, split)
    err = float(np.sqrt(np.mean((pred - cs) ** 2)))
    if best is None or err < best[1]:
        best = (off, err)
off, err = best
print("best offset %+.1f s, rms mismatch %.1f s per 500" % (off, err))
zero = float(np.sqrt(np.mean((np.interp(vt, t, split) - cs) ** 2)))
print("at zero offset the mismatch is %.1f s, so the sync is worth %.1f s" % (zero, zero - err))
print()
print("  %-8s %-9s %-11s %s" % ("video", "called", "measured", "diff"))
pred = np.interp(vt + off, t, split)
for (a, b), p in zip(CALLS, pred):
    print("  %-8s %-9s %-11s %+.1f s"
          % (a, "%d:%02d" % (b // 60, b % 60), "%d:%04.1f" % (p // 60, p % 60), p - b))
print()
print("mean |diff| %.1f s per 500" % np.mean(np.abs(pred - cs)))
np.save("offset.npy", np.array([off]))
