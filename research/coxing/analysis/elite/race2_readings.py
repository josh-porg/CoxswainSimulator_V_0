"""Race 2: CoxBox display read from video, Masters Nationals 2025 women's club 8+.

Values transcribed from the tracked, normalised display crops. Fields the image did not
support are left as NaN rather than guessed. Times are video seconds, which are also
transcript seconds, so speech and boat share one clock with no synchronisation step.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import numpy as np

# t, rate(spm), split(s per 500m), dps(m), dist(m)
RAW = [
 (108,37.5,136,5.28,18),  (112,39.5,116,6.69,36),  (117,39.5,105,7.12,60),
 (120,39.5,np.nan,np.nan,73), (122,39.5,103,7.37,87), (128,38.5,106,7.32,110),
 (132,39.0,105,7.23,np.nan), (138,38.0,105,7.55,np.nan), (139,39.5,np.nan,7.18,np.nan),
 (143,36.0,103,7.62,np.nan), (148,34.5,104,8.23,205), (152,34.5,np.nan,8.07,np.nan),
 (157,34.0,np.nan,np.nan,np.nan), (160,33.0,108,8.22,260), (163,34.5,109,7.91,275),
 (170,33.0,110,8.23,303), (171,33.0,110,8.23,313), (179,33.0,112,8.08,np.nan),
 (184,33.0,113,8.06,367), (188,33.5,115,7.71,388), (191,33.0,115,np.nan,399),
 (197,33.5,np.nan,np.nan,np.nan), (198,33.5,112,7.84,431), (204,33.0,110,7.82,np.nan),
 (208,33.0,113,7.92,np.nan), (210,33.0,112,8.12,np.nan), (214,np.nan,114,7.70,np.nan),
 (220,33.0,113,np.nan,np.nan), (224,33.0,115,7.81,545), (229,34.0,112,7.78,567),
 (233,33.0,113,7.96,np.nan), (234,34.0,113,7.70,592), (240,33.5,113,7.92,611),
 (244,33.5,113,7.83,634), (248,33.0,114,7.92,np.nan), (252,33.5,113,np.nan,np.nan),
 (254,33.0,114,7.91,np.nan), (261,33.0,113,7.89,705), (264,33.0,114,7.92,724),
 (268,34.0,110,np.nan,np.nan), (272,33.0,112,8.00,np.nan), (277,34.0,110,np.nan,np.nan),
 (280,34.5,110,7.89,np.nan), (286,34.5,109,8.05,np.nan), (287,34.5,109,8.05,np.nan),
 (291,34.5,108,7.85,np.nan), (295,35.5,108,7.71,862), (300,34.5,108,8.08,886),
 (305,35.5,108,7.81,np.nan), (306,np.nan,108,7.81,np.nan), (312,34.5,103,8.10,np.nan),
 (315,35.5,107,7.79,956), (321,34.5,108,8.14,np.nan), (323,34.0,108,np.nan,np.nan),
 (326,35.0,np.nan,np.nan,np.nan), (330,35.0,109,7.81,1025),
]
A = np.array(RAW, float)
T, RATE, SPLIT, DPS, DIST = A.T
V = 500.0 / SPLIT                       # boat speed, m/s

if __name__ == "__main__":
    print("%d readings over %.0f-%.0f s" % (len(T), T.min(), T.max()))
    for nm, x in (("rate", RATE), ("split", SPLIT), ("dps", DPS), ("dist", DIST)):
        print("   %-6s %2d missing, range %.1f-%.1f" %
              (nm, int(np.isnan(x).sum()), np.nanmin(x), np.nanmax(x)))
    print()
    ok = ~np.isnan(RATE) & ~np.isnan(SPLIT) & ~np.isnan(DPS)
    pred = V[ok] * 60 / RATE[ok]
    err = 100 * (pred - DPS[ok]) / DPS[ok]
    print("INTERNAL CHECK  dps = v*60/rate, on %d complete readings:" % ok.sum())
    print("   mean error %+.2f%%, median |error| %.2f%%, 90th pct %.2f%%, max %.2f%%"
          % (err.mean(), np.median(np.abs(err)), np.percentile(np.abs(err), 90),
             np.abs(err).max()))
    bad = np.flatnonzero(np.abs(err) > 5)
    print("   readings off by >5%%: %d %s"
          % (len(bad), [("%.0fs" % T[ok][b]) for b in bad]))
    print()
    d = ~np.isnan(DIST)
    print("distance monotonic: %s (%d readings, %.0f to %.0f m)"
          % (bool(np.all(np.diff(DIST[d]) > 0)), d.sum(), DIST[d][0], DIST[d][-1]))
    mv = np.diff(DIST[d]) / np.diff(T[d])
    print("   implied mean speed from distance %.2f m/s vs %.2f from split"
          % (mv.mean(), np.nanmean(V)))
    print()
    print("speed: mean %.3f m/s, sd %.3f; rate: mean %.1f spm, sd %.2f"
          % (np.nanmean(V), np.nanstd(V), np.nanmean(RATE), np.nanstd(RATE)))
