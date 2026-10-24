"""Phase 4.3a: where does the entry's 1.72 deg come from?

The catch-lag decomposition put 1.72 deg of the model's 4.26 deg drive lag in place
by t = 0.10 s, before the hull deficit could accumulate. The mechanism is in
DynamicOarSimulator._integrate_sweep_catch: while the blade is in the air the oar is
FORCED onto the prescribed sweep boat.oar_sweep after every step, and handed to the
torque drive only on the step its blade's normal velocity reaches zero ([CR06] eq. 16).

So during the air phase the model's oar angle is the catalog ARC, not hers. This
compares the two directly -- no simulation needed, the arc is analytic -- and asks:

  1. how far the prescribed arc has drifted from her measured angle by entry,
  2. whether the arc's RATE at entry matches hers, since the torque drive inherits
     the sweep's angle AND rate at that instant,
  3. when [CR06]'s own zero-normal-velocity rule fires on her measured data.

Nothing here changes the model.
"""
import csv
import os
import warnings

import numpy as np

warnings.simplefilter("ignore")
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "cr06_study"))   # its data and outputs (gitignored)
MARKER = "\n" + "grid = np.linspace(0.0, 1.0, 400"
exec(open(os.path.join(CODE, "female_segments_cr06.py")).read().split(MARKER)[0])

boat = research.apply(her_boat(stature))
sweep, timing = boat.oar_sweep, boat.timing
lock = boat.rig.seats[0].oarlocks[0]
ELL = float(lock.oar.blade_centre_outboard)


def load(name):
    a, b = [], []
    with open(os.path.join(HERE, "cr06_fig3_measured.csv")) as fh:
        for row in csv.DictReader(r for r in fh if not r.startswith("#") and not r.startswith('"#')):
            if row["series"] == name:
                a.append(float(row["t_s"])); b.append(float(row["value"]))
    a, b = np.array(a), np.array(b)
    o = np.argsort(a)
    return a[o], b[o]


def fourier(ta, ya, nh, grid):
    wq = 2 * np.pi / T_CR06
    k = np.arange(1, nh + 1)
    A = np.hstack([np.ones((ta.size, 1)), np.cos(np.outer(ta, k) * wq), np.sin(np.outer(ta, k) * wq)])
    c = np.linalg.lstsq(A, ya, rcond=None)[0]
    a_, b_ = c[1:nh + 1], c[nh + 1:]
    cg, sg = np.cos(np.outer(grid, k) * wq), np.sin(np.outer(grid, k) * wq)
    return (c[0] + cg @ a_ + sg @ b_, (-sg @ (a_ * k * wq)) + (cg @ (b_ * k * wq)))


g = np.linspace(0.0, 0.20, 400)
tA, A = load("oar_angle_deg")
tv, V = load("boat_velocity_mps")
her_d, her_rate_d = fourier(tA, A, 10, g)
her_v, _ = fourier(tv, V, 10, g)

arc_d = np.degrees([float(sweep(x, timing)) for x in g])
arc_rate_d = np.degrees([float(sweep.rate(x, timing)) for x in g])

print("her catch %+.2f deg at t=0; the prescribed arc starts at %+.2f deg" % (her_d[0], arc_d[0]))
print("catch angle of the arc: %+.2f deg" % np.degrees(float(sweep.catch_angle)))
print()
print("     t     her    arc   (arc-her)  | her rate  arc rate  (deg/s)  | her slip")
for q in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
    i = int(np.argmin(np.abs(g - q)))
    w = ELL * np.radians(her_rate_d[i]) + her_v[i] * np.cos(np.radians(her_d[i]))
    print("  %.02f  %+6.2f %+6.2f  %+7.2f  | %8.1f %9.1f %8.1f  | %+7.3f"
          % (q, her_d[i], arc_d[i], arc_d[i] - her_d[i],
             her_rate_d[i], arc_rate_d[i], arc_rate_d[i] - her_rate_d[i], w))

# [CR06] eq. 16 on her own measured data: entry when the normal velocity reaches zero
w_her = ELL * np.radians(her_rate_d) + her_v * np.cos(np.radians(her_d))
k = np.flatnonzero(w_her <= 0.0)
print()
if k.size:
    ke = k[0]
    print("[CR06] eq. 16 on HER measured trace: normal velocity reaches zero at t=%.4f s,"
          % g[ke])
    print("  her angle there %+.2f deg, her rate %.1f deg/s" % (her_d[ke], her_rate_d[ke]))
    print("  the arc is at %+.2f deg, rate %.1f deg/s  -> the torque drive inherits"
          % (arc_d[ke], arc_rate_d[ke]))
    print("  an angle %+.2f deg and a rate %+.1f deg/s away from hers."
          % (arc_d[ke] - her_d[ke], arc_rate_d[ke] - her_rate_d[ke]))
else:
    print("her slip never reaches zero on 0-0.20 s (it starts at %+.3f)" % w_her[0])
print()
print("the model's own run reports the blade in the air 0.000-0.042 of the cycle,")
print("i.e. to t = %.4f s." % (0.042 * T_CR06))
i = int(np.argmin(np.abs(g - 0.042 * T_CR06)))
print("  at that instant: her %+.2f deg, arc %+.2f deg, gap %+.2f deg"
      % (her_d[i], arc_d[i], arc_d[i] - her_d[i]))
print("  rates: her %.1f, arc %.1f deg/s, gap %+.1f" % (her_rate_d[i], arc_rate_d[i], arc_rate_d[i] - her_rate_d[i]))
