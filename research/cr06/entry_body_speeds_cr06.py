"""Phase 4.3a: does HER OWN body move fast enough to make her oar's catch?

The entry defect is a rate: the prescribed arc sweeps at 0.65x her measured oar rate
through the air phase, and that alone costs 3.2 deg of drive lag and 1.1 points of
boat speed. The oar is moved by the hands, and the hands sit at the end of
ankle -> hip -> shoulder -> arm. So the question underneath is whether a body can
supply the hand speed the oar demands.

This answers it from measurement alone -- no model, no simulation, nothing fitted:

    her hip speed       = d/dt (leg displacement)      [her measured leg time-law]
    her shoulder speed  = hip speed + d/dt (back disp) [plus her measured back law]
    her hand speed      = r_h cos(theta) * |dtheta/dt| [from her measured oar angle]

If her shoulder already moves at the hand speed her oar needs, the arms are near
rigid through the catch and the oar's rate IS the body's rate -- so a body that
reverses correctly would produce the correct entry, and the entry defect and the
known leg-reversal defect are one thing. If her shoulder moves much faster, the arms
absorb the difference and the entry needs the arms modelled, not just the legs.
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
lock = boat.rig.seats[0].oarlocks[0]
R_H = float(lock.oar.inboard)
rower = boat.crew[0].rower

NH, W = 10, 2 * np.pi / T_CR06
KH = np.arange(1, NH + 1)


def series(path, name, col="value_m"):
    a, b = [], []
    with open(os.path.join(HERE, path)) as fh:
        for row in csv.DictReader(r for r in fh if not r.startswith("#") and not r.startswith('"#')):
            if row["series"] == name:
                a.append(float(row["t_s"])); b.append(float(row[col]))
    a, b = np.array(a), np.array(b)
    o = np.argsort(a)
    return a[o], b[o]


def fit(ta, ya):
    A = np.hstack([np.ones((ta.size, 1)), np.cos(np.outer(ta, KH) * W), np.sin(np.outer(ta, KH) * W)])
    return np.linalg.lstsq(A, ya, rcond=None)[0]


def val(c, g):
    return c[0] + np.cos(np.outer(g, KH) * W) @ c[1:NH + 1] + np.sin(np.outer(g, KH) * W) @ c[NH + 1:]


def rate(c, g):
    return (-np.sin(np.outer(g, KH) * W) @ (c[1:NH + 1] * KH * W)
            + np.cos(np.outer(g, KH) * W) @ (c[NH + 1:] * KH * W))


g = np.linspace(0.0, 0.16, 321)
c_leg = fit(*series("cr06_fig3_body.csv", "leg_disp_m"))
c_back = fit(*series("cr06_fig3_body.csv", "back_disp_m"))
c_ang = fit(*series("cr06_fig3_measured.csv", "oar_angle_deg", "value"))

v_hip = rate(c_leg, g)                       # ankle is fixed in the hull
v_sh = v_hip + rate(c_back, g)               # shoulder = hip + back displacement
th = np.radians(val(c_ang, g))
dth = np.radians(rate(c_ang, g))
v_hand = np.abs(dth) * R_H * np.cos(th)      # hand speed the oar demands

# the model's own body, for contrast
J = [rower.joint_positions(float(x)) for x in g]
dt = g[1] - g[0]
m_hip = np.gradient(np.array([j["hip"][0] for j in J]), dt)
m_sh = np.gradient(np.array([j["shoulder"][0] for j in J]), dt)

print("her measured body against her measured oar, through the air phase")
print("     t   | her hip  her shoulder  hand the oar needs | shoulder - hand | model hip  model sh")
for q in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
    i = int(np.argmin(np.abs(g - q)))
    print("  %.02f  | %7.3f %13.3f %18.3f | %15.3f | %9.3f %9.3f"
          % (q, v_hip[i], v_sh[i], v_hand[i], v_sh[i] - v_hand[i], m_hip[i], m_sh[i]))

m = (g >= 0.02) & (g <= 0.10)
print()
print("means over 0.02-0.10 s (m/s):")
print("  her hip          %6.3f        model hip       %6.3f   (model/her %.2f)"
      % (v_hip[m].mean(), m_hip[m].mean(), m_hip[m].mean() / v_hip[m].mean()))
print("  her shoulder     %6.3f        model shoulder  %6.3f   (model/her %.2f)"
      % (v_sh[m].mean(), m_sh[m].mean(), m_sh[m].mean() / v_sh[m].mean()))
print("  hand her oar needs %6.3f" % v_hand[m].mean())
print()
print("  her shoulder / hand the oar needs  %.2f" % (v_sh[m].mean() / v_hand[m].mean()))
print("  if that is near 1, her arms are near rigid through the catch and the oar's")
print("  rate IS her body's rate; if it is well above 1, the arms absorb the rest.")
