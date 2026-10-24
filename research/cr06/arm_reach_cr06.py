"""Is the oar constrained by the arm's reach at the two ends of the stroke?

The entry fix and the finish defect look like one thing: the oar is under-constrained
by the body at both ends. The physical constraint is that the ARM HAS FINITE LENGTH,
so the handle cannot go where the arm cannot reach. If that is right then

    at the catch   the arms are near straight   -> the oar's angle is set by the body
    mid-drive      the arm is slack             -> the oar is free, torque-driven
    at the finish  the arms are fully drawn in  -> the oar is constrained again

and the sweep the rower can make is bounded, with no free parameter beyond the
anthropometry already in the model.

This measures shoulder-to-handle distance against arm length through the cycle, using
the model's own tracks, and asks whether the ends really are where the arm runs out.
"""
import os
import warnings

import numpy as np

warnings.simplefilter("ignore")
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "cr06_study"))   # its data and outputs (gitignored)
MARKER = "\n" + "grid = np.linspace(0.0, 1.0, 400"
exec(open(os.path.join(CODE, "female_segments_cr06.py")).read().split(MARKER)[0])

boat = research.apply(her_boat(stature))
rower = boat.crew[0].rower
sweep, timing = boat.oar_sweep, boat.timing
L_arm = rower.upper_arm_length + rower.forearm_length
side = sorted(rower._hand_tracks)[0]
hand = rower._hand_tracks[side]
sh = rower._shoulder_tracks[side]

g = np.linspace(0.0, T_CR06, 400, endpoint=False)
def vec(pt):
    return np.array([float(getattr(c, "value", c)) for c in np.atleast_1d(pt)], dtype=float)


d = np.array([float(np.linalg.norm(vec(hand(float(x))) - vec(sh(float(x))))) for x in g])
ang = np.degrees([float(sweep(float(x), timing)) for x in g])

print("arm length %.3f m (upper %.3f + forearm&hand %.3f)"
      % (L_arm, rower.upper_arm_length, rower.forearm_length))
print("reach margin recorded by the elbow solve: %s"
      % {k: round(float(v), 4) for k, v in rower._arm_reach_margin.items()})
print("drive fraction %.3f -> drive ends at t=%.3f s" % (timing.drive_fraction, timing.drive_duration))
print()
print("     t    oar deg   shoulder->handle   d / L_arm")
for q in (0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.89, 0.95, 1.05, 1.30, 1.60):
    i = int(np.argmin(np.abs(g - q)))
    print("  %.02f   %+8.1f %14.3f %13.3f" % (q, ang[i], d[i], d[i] / L_arm))

i_max, i_min = int(np.argmax(d)), int(np.argmin(d))
print()
print("  arms STRAIGHTEST  d/L %.3f at t=%.3f s, oar %+.1f deg" % (d[i_max] / L_arm, g[i_max], ang[i_max]))
print("  arms MOST DRAWN   d/L %.3f at t=%.3f s, oar %+.1f deg" % (d[i_min] / L_arm, g[i_min], ang[i_min]))
print()
print("  her catch is +60.4 deg at t=0, her release -39.1 at 0.894, her turn-round -44.3 at 1.025")
print("  the model with HIP_ENTRY reaches -42.2 at release and overruns to -49.4 at the turn")
print()
if d[i_max] / L_arm > 0.97:
    print("  -> the arms DO run out at one end: the reach limit is a real constraint there.")
else:
    print("  -> the arms never approach full extension (max %.1f%% of length), so a pure" % (100 * d[i_max] / L_arm))
    print("     reach limit does NOT bound this stroke, and cannot by itself stop the oar.")
