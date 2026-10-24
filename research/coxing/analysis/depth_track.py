"""Depth along the race track, and what the shallow water cost.

The soundings ship with the simulator (coxswain/data/lake_union_depth.npz) and cover
this water. Shallow water raises wave resistance sharply once the depth Froude number
approaches one, which is the effect coxswain.hydro.finite_depth_michell exists for.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

z = np.load(os.path.join(REPO, "coxswain", "data", "lake_union_depth.npz"))
xy, dep, src = z["depth_xy"], z["depth"], z["depth_source"]   # (lat, lon), metres
R = np.load("race.npz")
lat, lon, d, v, t = R["lat"], R["lon"], R["d"], R["v"], R["t"]

# metric scaling so nearest-neighbour is in metres, not degrees
lat0 = np.radians(float(lat.mean()))
sx, sy = 111320.0 * np.cos(lat0), 110540.0
tree = cKDTree(np.c_[xy[:, 1] * sx, xy[:, 0] * sy])
dist, idx = tree.query(np.c_[lon * sx, lat * sy], k=4)
w = 1.0 / np.maximum(dist, 0.5) ** 2
depth = (dep[idx] * w).sum(axis=1) / w.sum(axis=1)
near = dist[:, 0]
np.save("depth_along.npy", depth)

print("nearest sounding: median %.0f m, 90th %.0f m, worst %.0f m"
      % (np.median(near), np.percentile(near, 90), near.max()))
print("depth along the track: min %.1f, median %.1f, max %.1f m"
      % (depth.min(), np.median(depth), depth.max()))
g = 9.80665
Frh = v / np.sqrt(g * depth)
print("depth Froude number: median %.3f, max %.3f" % (np.median(Frh), Frh.max()))
print()
print("  %-10s %-9s %8s %8s %8s" % ("distance", "clock", "depth m", "Fr_h", "speed"))
for q in (0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000):
    i = int(np.argmin(np.abs(d - q)))
    print("  %-10s %-9s %8.1f %8.3f %8.3f"
          % ("%.0f m" % d[i], "%d:%02d" % (t[i] // 60, t[i] % 60), depth[i], Frh[i], v[i]))

shallow = depth < 5.0
print()
print("  strokes in water under 5 m: %d of %d (%.0f%%)" % (shallow.sum(), len(depth),
                                                           100 * shallow.mean()))
if shallow.any():
    print("  mean speed there %.3f vs %.3f elsewhere (%+.3f m/s)"
          % (v[shallow].mean(), v[~shallow].mean(), v[shallow].mean() - v[~shallow].mean()))
