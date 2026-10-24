"""Test the prediction: correct the trunk travel and the catch posture closes.

dec6d9d found the bent catch is the known trunk over-travel -- the rower's own
kinematics run a 0.515 m trunk where she measures 0.398 -- and predicted that scaling
it to hers makes the catch arm posture right, with d_L/F following from 0.300 m.

This scales (shoulder - hip) by K = 0.398 / native about the hip, which is what
body_field already does for the momentum books, and then works [CR06] eq. 8:

    x_H/S(t) = d_L/F - s sin(theta) - x_B/F - x_S/B

For each candidate d_L/F the arm posture is fully determined at every instant, so the
question is which value gives a real catch (arms near full reach) AND a real finish
(hands drawn to the body), with the arm never asked to exceed its length in between.
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
r = boat.crew[0].rower
lock = boat.rig.seats[0].oarlocks[0]
S = float(lock.oar.inboard)
L = r.upper_arm_length + r.forearm_length
MARGIN = float(list(r._arm_reach_margin.values())[0])
CAP = MARGIN * L
side = sorted(r._hand_tracks)[0]
ank = float(r.joint_positions(0.0)["ankle"][0])
pin = float(np.asarray(lock.position, dtype=float)[0])
T = float(boat.timing.period)
t_drive = float(boat.timing.drive_duration)


def vec(p):
    return np.array([float(getattr(c, "value", c)) for c in np.atleast_1d(p)], dtype=float)


ts = np.linspace(0.0, t_drive, 120)
hip = np.array([float(r.joint_positions(float(t))["hip"][0]) for t in ts])
sh = np.array([vec(r._shoulder_tracks[side](float(t))) for t in ts])
hand = np.array([vec(r._hand_tracks[side](float(t))) for t in ts])
theta = np.array([float(boat.oar_sweep(float(t), boat.timing)) for t in ts])

native_back = (sh[:, 0] - hip).max() - (sh[:, 0] - hip).min()
K = 0.398 / native_back
xBF = hip - ank
xSB_native = sh[:, 0] - hip
xSB = K * xSB_native                       # her measured trunk travel
sh_x_scaled = hip + xSB

# lateral and vertical offsets shoulder->hand stay as the rig and posture give them
lat = np.abs(hand[:, 1] - sh[:, 1])
drop = np.abs(hand[:, 2] - sh[:, 2])
reach = np.sqrt(np.maximum(CAP ** 2 - lat ** 2 - drop ** 2, 1e-9))

print("native trunk travel %.3f m -> K = %.3f to reach her 0.398" % (native_back, K))
print("arm %.3f m, elbow-solve cap %.3f m" % (L, CAP))
print()
print(" d_L/F |  catch x_H/S   as %% of reach |  finish x_H/S  as %% |  max overrun")
print("-------|------------------------------|---------------------|-------------")
best = None
for dlf in (0.300, 0.250, 0.200, 0.184, 0.158, 0.120, 0.080):
    xHS = dlf - S * np.sin(theta) - xBF - xSB
    frac = np.abs(xHS) / reach
    over = float(np.max(frac))
    c, f = float(frac[0]), float(frac[-1])
    flag = ""
    if over > 1.0:
        flag = "  <- arm too short"
    elif c > 0.93:
        flag = "  <- catch at full reach"
    print(" %.3f | %+7.3f m      %5.1f%%       | %+7.3f m   %5.1f%%  |  %5.1f%%%s"
          % (dlf, xHS[0], 100 * c, xHS[-1], 100 * f, 100 * over, flag))
    if over <= 1.0 and (best is None or c > best[1]):
        best = (dlf, c, f, over)

# solve for the d_L/F that puts the catch exactly at the cap
xHS0_at = lambda d: abs(d - S * np.sin(theta[0]) - xBF[0] - xSB[0])
lo, hi = -0.2, 0.5
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if xHS0_at(mid) > reach[0]:
        lo = mid
    else:
        hi = mid
dlf_star = hi
xHS = dlf_star - S * np.sin(theta) - xBF - xSB
print()
print("d_L/F that puts the CATCH exactly at the arm's cap: %.3f m" % dlf_star)
print("  catch  x_H/S %+.3f m, %.0f%% of reach, %.0f%% of arm length in 3-D"
      % (xHS[0], 100 * abs(xHS[0]) / reach[0],
         100 * np.sqrt(xHS[0] ** 2 + lat[0] ** 2 + drop[0] ** 2) / L))
print("  finish x_H/S %+.3f m, %.0f%% of reach, %.0f%% of arm length in 3-D"
      % (xHS[-1], 100 * abs(xHS[-1]) / reach[-1],
         100 * np.sqrt(xHS[-1] ** 2 + lat[-1] ** 2 + drop[-1] ** 2) / L))
print("  worst point in the drive: %.0f%% of reach" % (100 * np.max(np.abs(xHS) / reach)))
print()
print("  the model hard-codes d_L/F = %.3f m; dec6d9d predicted %.3f from the travel balance"
      % (pin - ank, 0.184))

print()
print("=== where does the arm overrun? reach fraction through the drive ===")
print("   t/T_drive |  native trunk @0.300 | her trunk @0.300 | her trunk @0.184")
for frac_t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
    i = int(round(frac_t * (len(ts) - 1)))
    a = abs(0.300 - S * np.sin(theta[i]) - xBF[i] - xSB_native[i]) / reach[i]
    b = abs(0.300 - S * np.sin(theta[i]) - xBF[i] - xSB[i]) / reach[i]
    c2 = abs(0.184 - S * np.sin(theta[i]) - xBF[i] - xSB[i]) / reach[i]
    print("      %.2f     |        %5.1f%%        |      %5.1f%%     |     %5.1f%%"
          % (frac_t, 100 * a, 100 * b, 100 * c2))
i_nat = int(np.argmax(np.abs(0.300 - S * np.sin(theta) - xBF - xSB_native) / reach))
i_her = int(np.argmax(np.abs(0.300 - S * np.sin(theta) - xBF - xSB) / reach))
print()
print("  native trunk peaks at %.0f%% of reach, at t/T_drive %.2f"
      % (100 * np.max(np.abs(0.300 - S * np.sin(theta) - xBF - xSB_native) / reach), ts[i_nat] / t_drive))
print("  her trunk    peaks at %.0f%% of reach, at t/T_drive %.2f"
      % (100 * np.max(np.abs(0.300 - S * np.sin(theta) - xBF - xSB) / reach), ts[i_her] / t_drive))
print()
print("  shoulder position through the drive, native vs hers (m from the stretcher):")
for frac_t in (0.0, 0.25, 0.5, 0.75, 1.0):
    i = int(round(frac_t * (len(ts) - 1)))
    print("      %.2f   native %+.3f   hers %+.3f   (difference %+.3f)"
          % (frac_t, xBF[i] + xSB_native[i], xBF[i] + xSB[i], xSB[i] - xSB_native[i]))
