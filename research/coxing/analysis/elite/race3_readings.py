"""Race 3: Masters Nationals 2025 Women's Open D4+, CoxBox read from video.

Display mounted facing the coxswain, so the camera sees it inverted; frames are rotated
180 degrees before reading. Field labels (RATE, SPLIT, DIST) are legible here, which
they were not in race 2.

Speed is taken as rate x dps / 60 rather than from the split field. The three fields are
redundant, and the split is the one most often clipped at the edge of the tracked crop,
so deriving it from the two that read cleanly loses nothing and recovers rows that
would otherwise be lost.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import numpy as np

N = np.nan
# t, rate(spm), split(s, where legible), dps(m), dist(m)
RAW = [
 (102,58.0,225,2.66,12),  (107,38.0,130,6.15,33),  (110,61.5,117,4.28,45),
 (116,39.5,111,6.80,70),  (118,37.5,111,7.10,79),  (128,37.5,110,7.28,124),
 (132,37.5,113,6.94,146), (142,33.0,N,7.95,190),   (148,33.5,N,7.73,211),
 (153,32.5,N,7.66,235),   (158,31.5,N,8.01,255),   (163,32.0,N,7.80,277),
 (169,33.5,122,7.25,301), (174,34.0,N,6.83,318),   (182,32.5,N,7.40,350),
 (184,33.0,N,7.21,359),   (188,32.5,125,7.24,376), (194,32.5,N,7.35,408),
 (198,31.5,N,7.76,419),   (202,32.0,126,7.34,435), (210,32.0,N,7.38,463),
 (215,32.0,N,7.38,486),   (219,31.5,N,7.20,504),   (224,32.5,125,7.37,521),
 (240,32.0,N,7.26,585),   (244,33.0,N,7.20,602),   (252,32.5,127,7.19,632),
 (254,32.5,127,7.25,641), (260,33.0,N,7.12,661),   (266,34.0,123,7.04,682),
 (270,33.5,125,7.03,701), (275,33.0,N,7.25,723),   (278,34.0,N,6.92,733),
 (282,32.0,N,N,752),      (290,32.0,125,7.41,782), (296,32.0,N,7.27,803),
 (300,32.0,125,7.36,823), (307,31.5,N,7.38,848),   (310,31.5,130,7.11,862),
 (312,32.0,N,7.18,870),
]
A = np.array(RAW, float)
T, RATE, SPLIT, DPS, DIST = A.T
V = RATE * DPS / 60.0                     # speed from the two cleanest fields

if __name__ == "__main__":
    print("%d readings, %.0f-%.0f s" % (len(T), T.min(), T.max()))
    ok = np.isfinite(V)
    print("speed available for %d of them: mean %.3f m/s, sd %.3f"
          % (ok.sum(), V[ok].mean(), V[ok].std()))
    # cross-check against the split field where it was legible
    both = np.isfinite(SPLIT) & ok
    vs = 500.0 / SPLIT[both]
    err = 100 * (V[both] - vs) / vs
    print()
    print("CHECK 1  speed from rate*dps vs speed from the split field (%d readings):"
          % both.sum())
    print("   mean %+.2f%%, median |err| %.2f%%, max %.2f%%"
          % (err.mean(), np.median(np.abs(err)), np.abs(err).max()))
    # cross-check against distance differences
    d = np.isfinite(DIST)
    print()
    print("CHECK 2  distance strictly increasing: %s (%d readings, %.0f to %.0f m)"
          % (bool(np.all(np.diff(DIST[d]) > 0)), d.sum(), DIST[d][0], DIST[d][-1]))
    steady = (T > 140) & ok & d
    span = (DIST[steady][-1] - DIST[steady][0]) / (T[steady][-1] - T[steady][0])
    print("   mean speed over the steady phase: %.3f m/s from distance, %.3f from rate*dps"
          % (span, V[steady].mean()))
