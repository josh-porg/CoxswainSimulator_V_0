"""Parse the CoxBox race file into a clean per-stroke record, and add the physics.

The CoxBox gives GPS speed, stroke rate, distance and position per stroke. It does NOT
give force, power or oar angles -- there were no instrumented oarlocks. That missing
half is what the simulator supplies: given the boat and the measured speed, the hull
model says what drag it was overcoming, and therefore what power went into the water.
"""
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from course_pacing import build_boat, hull_drag      # noqa: E402

MEDIA = json.load(open(os.path.join(REPO, "data", "local", "coxing", "local_paths.json"), encoding="utf-8"))["media_root"]   # private, machine-specific
SRC = os.path.join(MEDIA, "CoxBox 011009704 20260920 0806AM.csv")
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis"))   # its data and outputs (gitignored)


def tosec(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


lines = open(SRC, encoding="utf-8-sig").read().splitlines()
i = lines.index("Per-Stroke Data:")
rows = [r for r in csv.DictReader(lines[i + 2:])
        if r.get("Distance (GPS)") not in (None, "", "(Meters)")]

d = np.array([float(r["Distance (GPS)"]) for r in rows])
t = np.array([tosec(r["Elapsed Time"]) for r in rows])
v = np.array([float(r["Speed (GPS)"]) for r in rows])
sr = np.array([float(r["Stroke Rate"]) for r in rows])
dps = np.array([float(r["Distance/Stroke (GPS)"]) for r in rows])
lat = np.array([float(r["GPS Lat."]) for r in rows])
lon = np.array([float(r["GPS Lon."]) for r in rows])

boat = build_boat("4+", 32.0)
drag_fn = hull_drag(boat)
drag = np.array([float(drag_fn(x)) for x in v])
power = drag * v                       # W at the hull, i.e. useful propulsive power

# crew power at the handle: divide by blade efficiency. The research profile's tier-1
# four sits at 0.681; that is the model's own number, not a fit to this race.
ETA = 0.681
crew_power = power / ETA

np.savez(os.path.join(HERE, "race.npz"), d=d, t=t, v=v, sr=sr, dps=dps,
         lat=lat, lon=lon, drag=drag, power=power, crew_power=crew_power)

print("strokes %d, %.1f m in %s" % (len(d), d[-1], "%d:%04.1f" % (int(t[-1] // 60), t[-1] % 60)))
print("speed %.3f +- %.3f m/s (%.3f to %.3f)" % (v.mean(), v.std(), v.min(), v.max()))
print("rate  %.1f +- %.1f spm (%.1f to %.1f)" % (sr.mean(), sr.std(), sr.min(), sr.max()))
print("distance/stroke %.2f +- %.2f m" % (dps.mean(), dps.std()))
print()
print("reconstructed by the hull model:")
print("  hull drag   %.1f +- %.1f N  (%.1f to %.1f)" % (drag.mean(), drag.std(), drag.min(), drag.max()))
print("  useful power %.0f +- %.0f W" % (power.mean(), power.std()))
print("  crew power at the handle, at eta = %.3f: %.0f W total, %.0f W per rower"
      % (ETA, crew_power.mean(), crew_power.mean() / 4.0))
print()
print("=== the race in 500 m pieces ===")
print("  %-12s %7s %7s %7s %8s %9s" % ("piece", "split", "speed", "rate", "m/stroke", "crew W"))
edges = np.arange(0, d[-1] + 1, 500.0)
for a, b in zip(edges[:-1], edges[1:]):
    m = (d >= a) & (d < b)
    if m.sum() < 3:
        continue
    seg_t = t[m].max() - t[m].min()
    seg_d = d[m].max() - d[m].min()
    sp = 500.0 / (seg_d / seg_t) if seg_d > 0 else float("nan")
    print("  %-12s %7s %7.3f %7.1f %8.2f %9.0f"
          % ("%.0f-%.0f" % (a, b), "%d:%04.1f" % (int(sp // 60), sp % 60),
             v[m].mean(), sr[m].mean(), dps[m].mean(), crew_power[m].mean()))
