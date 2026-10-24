"""Lake Sammamish routes for the conditions page, from the CoxBox GPS of SRA practices (data/local/coxbox).

    .venv/Scripts/python.exe research/lakes/sammamish/make_routes.py

1. North arm loop: the 25 Sep 2026 05:55 practice piece as rowed (3.9 km), thinned.
2. North end to mid-lake: the median line of the 7 Oct 2026 pieces, binned by latitude (47.649 to 47.614 N).
3. Whole lake: the lake's own centreline (midpoint of the water across each latitude) from the north end to the
   south basin, for longer rows.
Writes routes.json. Edit freely: waypoints are [lat, lon].
"""
import csv, glob, json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "sammamish"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "lakes"))
import lake_geometry as G
from matplotlib.path import Path


def track(name):
    rows = list(csv.reader(open(os.path.join(ROOT, "data", "local", "coxbox", name), encoding="utf-8", errors="ignore")))
    k = next(j for j, r in enumerate(rows) if r and r[0].startswith("Per-Stroke Data"))
    pts = []
    for r in rows[k + 3:]:
        try:
            pts.append((float(r[22]), float(r[23])))
        except (ValueError, IndexError):
            pass
    return np.array(pts)


def thin(P, step_m=120.0):
    out = [P[0]]
    for p in P[1:]:
        if np.hypot((p[0] - out[-1][0]) * 111000, (p[1] - out[-1][1]) * 75000) >= step_m:
            out.append(p)
    if np.hypot((P[-1][0] - out[-1][0]) * 111000, (P[-1][1] - out[-1][1]) * 75000) > 20:
        out.append(P[-1])
    return [[round(float(a), 5), round(float(b), 5)] for a, b in out]


def main():
    outer, _ = G.rings(os.path.join(HERE, "osm_lake.json"))
    poly = Path(np.array(outer[0])[:, ::-1])
    loop = thin(track("Sharing file  CoxBox 011009704 20260925 0555AM.csv"))
    pts = np.vstack([track(os.path.basename(p)) for p in glob.glob(os.path.join(ROOT, "data", "local", "coxbox", "*20261007*.csv"))])
    pts = pts[pts[:, 0] > 47.0]
    mid = []
    for la in np.arange(47.6490, 47.6135, -0.0015):
        m = np.abs(pts[:, 0] - la) < 0.00075
        if m.sum() >= 3:
            mid.append([round(float(la), 5), round(float(np.median(pts[m, 1])), 5)])
    lon_s = np.convolve(np.pad([m[1] for m in mid], 2, mode="edge"), np.ones(5) / 5, mode="valid")   # pieces on both sides: smooth
    mid = [[m[0], round(float(x), 5)] for m, x in zip(mid, lon_s)]
    centre = []
    lons = np.arange(-122.125, -122.045, 0.0002)
    for la in np.arange(47.6480, 47.5640, -0.0030):
        wet = poly.contains_points(np.column_stack([lons, np.full(len(lons), la)]))
        if wet.any():
            idx = np.where(wet)[0]
            runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            run = max(runs, key=len)
            centre.append([round(float(la), 5), round(float(lons[run].mean()), 5)])
    routes = [dict(key="loop", name="North arm loop (25 Sep piece)", colour="#d6247a", points=loop),
              dict(key="mid", name="North end to mid-lake (7 Oct pieces)", colour="#e8b10c", points=mid),
              dict(key="lake", name="Whole lake, north end to the south basin", colour="#2f6f9f", points=centre)]
    for r in routes:
        P = np.array(r["points"])
        dry = [tuple(p) for p in P if not poly.contains_point((p[1], p[0]))]
        L = np.sum(np.hypot(np.diff(P[:, 0]) * 111000, np.diff(P[:, 1]) * 75000))
        print("%s: %d points, %.1f km, points on land: %d" % (r["name"], len(P), L / 1000, len(dry)))
    json.dump(dict(note="From SRA CoxBox GPS and the OSM shoreline (Oct 2026); approximate. Waypoints are [lat, lon].", routes=routes),
              open(os.path.join(HERE, "routes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
