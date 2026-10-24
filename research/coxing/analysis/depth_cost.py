"""What the shallow water cost, from the project's finite-depth wave table."""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from coxswain.hydro.finite_depth_michell import wave_table_for    # noqa: E402
from course_pacing import build_boat, hull_drag                    # noqa: E402

R = np.load("race.npz")
v, d = R["v"], R["d"]
depth = np.load("depth_along.npy")
boat = build_boat("4+", 32.0)
table = wave_table_for(boat.offsets)

DMIN = 2.0
n_clip = int((depth < DMIN).sum())
h = np.maximum(depth, DMIN)
print("soundings below %.1f m clipped: %d strokes (%.1f%%) -- near-shore artefacts"
      % (DMIN, n_clip, 100 * n_clip / len(depth)))

Rdeep = np.array([float(table(x)) for x in v])
Rshal = np.array([float(table.at_depth(x, hh)) for x, hh in zip(v, h)])
extra = Rshal - Rdeep

drag_fn = hull_drag(boat)
base = np.array([float(drag_fn(x)) for x in v])
seg = np.gradient(d)
lost = float(np.sum(seg * extra / np.maximum(base, 1e-6) / v))

print()
print("  %-9s %8s %10s %10s %9s" % ("depth m", "Fr_h", "R deep", "R shallow", "extra"))
for hh in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
    vv = float(np.median(v))
    a, b = float(table(vv)), float(table.at_depth(vv, hh))
    print("  %-9.1f %8.3f %10.2f %10.2f %+9.2f N" % (hh, vv / np.sqrt(9.80665 * hh), a, b, b - a))
print()
print("  along the track: extra wave resistance mean %+.2f N, max %+.2f N"
      % (extra.mean(), extra.max()))
print("  as a share of total hull drag: mean %+.2f%%" % (100 * (extra / base).mean()))
print("  INTEGRATED TIME COST OF THE SHALLOW WATER: %+.1f s" % lost)
shal = h < 5.0
print()
print("  under 5 m for %d of %d strokes (%.0f%%); extra there %+.2f N"
      % (shal.sum(), len(h), 100 * shal.mean(), extra[shal].mean()))
