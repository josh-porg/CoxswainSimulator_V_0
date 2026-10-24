"""Can ONE stretcher position close both the catch and the finish?

The stretcher is adjustable, so pin-to-feet is a rigging choice rather than fixed
geometry. The question is whether any choice of it makes the model's catch AND finish
both physically right, or whether the body and the rig are inconsistent with each
other whatever the setting.

Moving the stretcher moves the body (the rower is referenced to the ankle) while the
pin stays with the hull, so a shift d bow-ward adds d to every body x and leaves the
handle where it is.

  catch  wants the arms near straight: |hand - shoulder| ~ the arm length
  finish wants the handle at the body: it should arrive between hip and shoulder
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
L = r.upper_arm_length + r.forearm_length
MARGIN = float(list(r._arm_reach_margin.values())[0])


def vec(p):
    return np.array([float(getattr(c, "value", c)) for c in np.atleast_1d(p)], dtype=float)


side = sorted(r._hand_tracks)[0]
t_catch, t_fin = 0.0, float(boat.timing.drive_duration)
h_c, s_c = vec(r._hand_tracks[side](t_catch)), vec(r._shoulder_tracks[side](t_catch))
h_f, s_f = vec(r._hand_tracks[side](t_fin)), vec(r._shoulder_tracks[side](t_fin))
hip_c = r.joint_positions(t_catch)["hip"]
hip_f = r.joint_positions(t_fin)["hip"]
ank = float(r.joint_positions(t_catch)["ankle"][0])
pin = float(np.asarray(lock.position, dtype=float)[0])

print("pin x %.3f, ankle x %.3f -> pin is %.3f m bow-ward of the feet" % (pin, ank, pin - ank))
print("arm length %.3f m; the elbow solve caps reach at %.4f of it (%.3f m)" % (L, MARGIN, MARGIN * L))
print()

# --- what shift d closes the catch? -------------------------------------------
drop = abs(h_c[2] - s_c[2])
lat = abs(h_c[1] - s_c[1])
target = MARGIN * L
need_x = float(np.sqrt(max(target ** 2 - drop ** 2 - lat ** 2, 0.0)))
have_x = abs(h_c[0] - s_c[0])
d_catch = need_x - have_x
print("CATCH: fore-aft shoulder->hand is %.3f m, needs %.3f m for the capped reach"
      % (have_x, need_x))
print("  -> move the body %+.3f m bow-ward; the pin would then sit %.3f m from the feet"
      % (d_catch, pin - (ank + d_catch)))

# --- what shift does the finish tolerate? -------------------------------------
print()
print("FINISH: handle x %.3f, hip x %.3f, shoulder x %.3f" % (h_f[0], hip_f[0], s_f[0]))
print("  the handle arrives %.3f m bow-ward of the hip (it should reach the body)"
      % (h_f[0] - hip_f[0]))
print("  moving the body %+.3f m bow-ward would leave it %.3f m SHORT of the hip"
      % (d_catch, (h_f[0] - (hip_f[0] + d_catch))))

# --- the underlying mismatch ---------------------------------------------------
oar_travel = float(lock.oar.inboard) * (np.sin(float(boat.oar_sweep.catch_angle))
                                        - np.sin(float(boat.oar_sweep.finish_angle)))
sh_travel = s_f[0] - s_c[0]
arm_c_x = need_x                      # if the arms were straight at the catch
arm_f_x = abs(h_f[0] - s_f[0])
body_travel = sh_travel + arm_c_x - arm_f_x
print()
print("THE MISMATCH, independent of where the stretcher goes:")
print("  the oar offers   r_h (sin cat - sin fin) = %.3f m of handle travel in x" % oar_travel)
print("  the body wants   shoulder %.3f + arms (%.3f at a straight catch - %.3f drawn) = %.3f m"
      % (sh_travel, arm_c_x, arm_f_x, body_travel))
print("  difference %.3f m" % (body_travel - oar_travel))
print()
if abs(d_catch) > 0.05 and abs(body_travel - oar_travel) > 0.05:
    print("  -> no single stretcher position closes both ends: the body's stroke and the")
    print("     oar's sweep are inconsistent by the difference above, and the model absorbs")
    print("     it by bending the arms at the catch.")
else:
    print("  -> a stretcher shift can close both ends.")

# --- the same sum with HER measured back travel --------------------------------
# joint_positions uses the rower's native trunk (back travel 0.516); her measured
# value is 0.398, which body_field applies as K. Redo the mismatch with hers.
back_native = (s_f[0] - hip_f[0]) - (s_c[0] - hip_c[0])
K_back = 0.398 / back_native
sh_travel_her = (hip_f[0] - hip_c[0]) + 0.398
s_f_her = hip_f[0] + (s_f[0] - hip_f[0]) * K_back
arm_f_her = abs(h_f[0] - s_f_her)
body_her = sh_travel_her + arm_c_x - arm_f_her
print()
print("SAME SUM WITH HER MEASURED BACK TRAVEL (0.398, not the native %.3f):" % back_native)
print("  shoulder travel %.3f + arms (%.3f - %.3f) = %.3f m against the oar's %.3f"
      % (sh_travel_her, arm_c_x, arm_f_her, body_her, oar_travel))
print("  difference %.3f m" % (body_her - oar_travel))
print()
print("  so the mismatch is %.2f-%.2f m depending on whose trunk is used, and it does"
      % (min(body_her, body_travel) - oar_travel, max(body_her, body_travel) - oar_travel))
print("  not go away: the body's stroke is longer than the oar's sweep either way.")
