"""Phase 4.3a: could the BODY drive the entry, instead of a prescribed arc?

The study showed a correct entry is worth 3.2 deg of drive lag and 1.1 points of
boat speed, but it got there by feeding the model her measured hand path, which is
not predictive. The principled fix is that the hands are on the handle, so the oar
angle through the air phase follows from where the rower's hands are.

There is an architectural catch. When hand_targets is set, kinematics.py constrains
the arms TO the oar -- "the hand track IS the handle" -- so the hands follow the oar
and cannot drive it. The free-arm branch builds the hand from the shoulder through
upper_arm and forearm angles, and THAT path is independent of the oar.

The question this answers: does the free-arm body's own hand path imply an oar rate
near hers, or does it reproduce the arc's 24 deg/s deficit? If the body is as gentle
as the arc, coupling the oar to it fixes nothing and the defect is in the body, not
the coupling.

Rate relation, convention-free in magnitude: the handle sits at radius r_h from the
pin, so the hand's fore-aft speed is |dx/dt| = r_h |cos(theta)| |dtheta/dt|, hence

    dtheta/dt = (dx_hand/dt) / (r_h cos(theta))

evaluated at her measured theta, so both candidates are read at the same oar angle.
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
rower = boat.crew[0].rower
sweep, timing = boat.oar_sweep, boat.timing
lock = boat.rig.seats[0].oarlocks[0]
R_H = float(lock.oar.inboard)

g = np.linspace(0.0, 0.16, 321)
J = [rower.joint_positions(float(x)) for x in g]
hand_x = np.array([j["hand"][0] for j in J])
sh_x = np.array([j["shoulder"][0] for j in J])
hip_x = np.array([j["hip"][0] for j in J])
has_targets = getattr(rower, "hand_targets", None) is not None
print("rower hand_targets set: %s  (if True the hands are slaved to the oar)" % has_targets)

dt = g[1] - g[0]
v_hand = np.gradient(hand_x, dt)
v_sh = np.gradient(sh_x, dt)
v_hip = np.gradient(hip_x, dt)


def load(name):
    a, b = [], []
    with open(os.path.join(HERE, "cr06_fig3_measured.csv")) as fh:
        for row in csv.DictReader(r for r in fh if not r.startswith("#") and not r.startswith('"#')):
            if row["series"] == name:
                a.append(float(row["t_s"])); b.append(float(row["value"]))
    a, b = np.array(a), np.array(b)
    o = np.argsort(a)
    return a[o], b[o]


NH, W = 10, 2 * np.pi / T_CR06
KH = np.arange(1, NH + 1)
tA, A = load("oar_angle_deg")
M = np.hstack([np.ones((tA.size, 1)), np.cos(np.outer(tA, KH) * W), np.sin(np.outer(tA, KH) * W)])
C = np.linalg.lstsq(M, A, rcond=None)[0]
her_d = C[0] + np.cos(np.outer(g, KH) * W) @ C[1:NH + 1] + np.sin(np.outer(g, KH) * W) @ C[NH + 1:]
her_rate_d = (-np.sin(np.outer(g, KH) * W) @ (C[1:NH + 1] * KH * W)
              + np.cos(np.outer(g, KH) * W) @ (C[NH + 1:] * KH * W))
arc_rate_d = np.degrees([float(sweep.rate(float(x), timing)) for x in g])

cos_t = np.cos(np.radians(her_d))
implied_d = np.degrees(v_hand / (R_H * cos_t))
her_v_hand = np.radians(her_rate_d) * R_H * cos_t

print()
print("     t   her theta | hand speed m/s  body   her  | implied rate  arc   her  (deg/s)")
for q in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
    i = int(np.argmin(np.abs(g - q)))
    print("  %.02f   %+7.2f  | %12.3f %6.3f | %10.1f %6.1f %6.1f"
          % (q, her_d[i], v_hand[i], her_v_hand[i], implied_d[i], arc_rate_d[i], her_rate_d[i]))

m = (g >= 0.02) & (g <= 0.10)
print()
print("over the air phase 0.02-0.10 s, mean rate (deg/s):")
print("  body's own hand path   %8.1f" % implied_d[m].mean())
print("  prescribed arc         %8.1f" % arc_rate_d[m].mean())
print("  her measured oar       %8.1f" % her_rate_d[m].mean())
print()
print("  hand speed, body %.3f m/s vs hers %.3f m/s  (ratio %.2f)"
      % (v_hand[m].mean(), her_v_hand[m].mean(), v_hand[m].mean() / her_v_hand[m].mean()))
print("  seat/hip %.3f, shoulder %.3f m/s -- where the hand speed comes from"
      % (v_hip[m].mean(), v_sh[m].mean()))


# ---------------------------------------------------------------- the free-arm body
# The run above is circular: hand_targets is set, so those hands ARE the handle track.
# Rebuild the same athlete with hand_targets=None so the hand comes from the stroke
# dataset's own upper-arm and forearm angles, independent of the oar, and ask what oar
# rate THAT path would impose if the hands were on the handle.
from coxswain.crew.kinematics import JointDrivenRower, RowerStation

seat = boat.rig.seats[0]
free = JointDrivenRower(rower.anthropometry, rower.station, boat.timing,
                        dataset=rower.dataset, phase_offset=0.0, hand_targets=None,
                        recovery_arrival=rower.recovery_arrival, phase_warp=rower.phase_warp)
Jf = [free.joint_positions(float(x)) for x in g]
fh_x = np.array([j['hand'][0] for j in Jf])
fv = np.gradient(fh_x, dt)
free_implied = np.degrees(fv / (R_H * cos_t))

print()
print('=== the free-arm body, hands from the dataset rather than from the oar ===')
print('     t   free-arm hand m/s | implied rate | arc    her')
for q in (0.02, 0.04, 0.06, 0.08, 0.10):
    i = int(np.argmin(np.abs(g - q)))
    print('  %.02f   %16.3f | %12.1f | %6.1f %6.1f'
          % (q, fv[i], free_implied[i], arc_rate_d[i], her_rate_d[i]))
print()
print('over 0.02-0.10 s, mean |rate| deg/s:  free-arm body %.1f   arc %.1f   her %.1f'
      % (abs(free_implied[m].mean()), abs(arc_rate_d[m].mean()), abs(her_rate_d[m].mean())))
print('  free-arm body / her  %.2f      arc / her  %.2f'
      % (abs(free_implied[m].mean()) / abs(her_rate_d[m].mean()),
         abs(arc_rate_d[m].mean()) / abs(her_rate_d[m].mean())))
