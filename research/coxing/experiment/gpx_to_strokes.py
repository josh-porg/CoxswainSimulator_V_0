"""A CoxBox GPX (one trackpoint per stroke) as per-stroke t, speed, rate, distance; validated against the CSV of the same piece."""
import re, sys, numpy as np
from datetime import datetime

def load_gpx(path):
    s = open(path, encoding="utf-8").read()
    pts = re.findall(r'<trkpt lat="([-\d.]+)" lon="([-\d.]+)"><time>([^<]+)</time>', s)
    lat = np.radians([float(a) for a, _, _ in pts]); lon = np.radians([float(b) for _, b, _ in pts])
    ts = np.array([datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() for _, _, t in pts])
    R = 6371000.0
    dlat, dlon = np.diff(lat), np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(a))
    dt = np.diff(ts)
    t = ts[1:] - ts[0]
    return t, d / dt, 60.0 / dt, np.cumsum(d), ts[0]

if __name__ == "__main__":
    sys.path.insert(0, "tools/overlay")
    import coxbox_data as C
    base = "data/local/coxbox/Sharing file  CoxBox 011009704 20261007 0948AM"
    t, sp, ra, dist, t0 = load_gpx(base + ".gpx")
    c = C.load_csv(base + ".csv")
    n = min(len(t), len(c.t))
    print("gpx strokes %d, csv strokes %d" % (len(t), len(c.t)))
    for lag in range(-2, 3):
        a, b = sp[max(0, lag):n + min(0, lag)], c.speed[max(0, -lag):n - max(0, lag)]
        m = min(len(a), len(b))
        print("lag %+d: speed r %.3f, mean diff %.3f m/s; rate r %.3f" % (lag, np.corrcoef(a[:m], b[:m])[0, 1], np.mean(a[:m] - b[:m]),
              np.corrcoef(ra[max(0, lag):][:m], c.rate[max(0, -lag):][:m])[0, 1]))
