"""Is the catch category's positive out-of-sample score real?

It is the only one of eight above zero (+0.0844). The control it must beat is not the
eight-category one but a matched one: the same 26-regressor single-category model with
the same number and spacing of events, differing only in when they occur relative to
the boat. Circular shifts preserve the series' own spacing exactly.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import sys

import numpy as np

sys.path.insert(0, "elite")
import vtt
from functional import FUNC, LOOKUP
from oos import blocked_oos, design, hits, y, NS, LAGS  # noqa: F401

rng = np.random.default_rng(23)
obs = blocked_oos(design(["catch"]), y)
print("observed catch OOS R2: %+.4f  (n=%d events)" % (obs, int(hits["catch"].sum())))

null = []
base = hits["catch"].copy()
for _ in range(500):
    sh = np.roll(base, int(rng.integers(26, NS - 26)))
    X = np.column_stack([np.roll(sh, L) for L in LAGS])
    null.append(blocked_oos(X - X.mean(axis=0), y))
null = np.array(null)
p = float(np.mean(null >= obs))
print("matched null (500 circular shifts of the same series):")
print("   mean %+.4f, 95th pct %+.4f, max %+.4f" % (null.mean(), np.percentile(null, 95), null.max()))
print("   p = %.3f  (one-sided, proportion of shifts scoring at least as high)" % p)

# across all eight categories, corrected for having picked the best of eight
allobs = {k: blocked_oos(design([k]), y) for k in FUNC}
best = max(allobs, key=allobs.get)
nullmax = []
for _ in range(300):
    sc = []
    for k in FUNC:
        sh = np.roll(hits[k], int(rng.integers(26, NS - 26)))
        X = np.column_stack([np.roll(sh, L) for L in LAGS])
        sc.append(blocked_oos(X - X.mean(axis=0), y))
    nullmax.append(max(sc))
nullmax = np.array(nullmax)
pc = float(np.mean(nullmax >= allobs[best]))
print()
print("best of eight was '%s' at %+.4f" % (best, allobs[best]))
print("null of the BEST of eight: 95th pct %+.4f, max %+.4f" % (np.percentile(nullmax, 95), nullmax.max()))
print("   corrected p = %.3f" % pc)
