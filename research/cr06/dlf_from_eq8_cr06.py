"""Solve [CR06] eq. 8 for d_L/F instead of assuming a straight catch.

The previous pass asserted "no single stretcher position closes both ends". That was
wrong, and it was wrong because it FIXED the catch arm at full extension and then
found the finish would not fit. Eq. 8 does not work that way: everything in it is
measured relative to the foot stretcher, so the travel balance fixes the arms and the
stretcher position follows.

    handle, relative to the stretcher :  d_L/F - s sin(theta)
    hand,   relative to the stretcher :  x_B/F + x_S/B + x_H/S

Taking catch to finish, d_L/F cancels out of the travel:

    s (sin th_c - sin th_f)  =  dx_B/F + dx_S/B + dx_H/S

Her leg and back travels are measured and the sweep is measured, so dx_H/S -- how much
the arms draw in -- is DETERMINED, not free. Then d_L/F follows from eq. 8 at either
end, and by construction both ends close.
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


def vec(p):
    return np.array([float(getattr(c, "value", c)) for c in np.atleast_1d(p)], dtype=float)


side = sorted(r._hand_tracks)[0]
t_c, t_f = 0.0, float(boat.timing.drive_duration)
h_c, s_c = vec(r._hand_tracks[side](t_c)), vec(r._shoulder_tracks[side](t_c))
h_f, s_f = vec(r._hand_tracks[side](t_f)), vec(r._shoulder_tracks[side](t_f))
th_c = float(boat.oar_sweep.catch_angle)
th_f = float(boat.oar_sweep.finish_angle)
ank = float(r.joint_positions(t_c)["ankle"][0])
pin = float(np.asarray(lock.position, dtype=float)[0])
hip_c = float(r.joint_positions(t_c)["hip"][0])
hip_f = float(r.joint_positions(t_f)["hip"][0])

LEG, BACK = 0.581, 0.398          # her measured travels
oar_travel = S * (np.sin(th_c) - np.sin(th_f))
dx_arms = oar_travel - (LEG + BACK)

# fore-aft reach the arm can offer at each end, given the 3-D offsets there
def reach(h, sh):
    return float(np.sqrt(max((MARGIN * L) ** 2 - (h[2] - sh[2]) ** 2 - (h[1] - sh[1]) ** 2, 0.0)))


reach_c, reach_f = reach(h_c, s_c), reach(h_f, s_f)
xHS_f = -abs(h_f[0] - s_f[0])                      # hand aft of shoulder, negative
xHS_c = xHS_f - dx_arms

print("her measured: leg travel %.3f, back travel %.3f, sweep %+.1f to %+.1f deg"
      % (LEG, BACK, np.degrees(th_c), np.degrees(th_f)))
print("s = %.3f m (CR06's force radius, = actual inboard - 0.06)" % S)
print()
print("TRAVEL BALANCE (d_L/F cancels):")
print("  the oar gives  s (sin c - sin f)      = %.3f m" % oar_travel)
print("  legs + back                           = %.3f m" % (LEG + BACK))
print("  so the arms MUST draw in by           = %.3f m" % dx_arms)
print()
print("  fore-aft reach available at the catch  %.3f m (capped arm %.3f, drop %.3f, lateral %.3f)"
      % (reach_c, MARGIN * L, abs(h_c[2] - s_c[2]), abs(h_c[1] - s_c[1])))
print("  x_H/S at the finish                    %+.3f m" % xHS_f)
print("  -> x_H/S at the catch must be          %+.3f m" % xHS_c)
print("     that is %.0f%% of the fore-aft reach, i.e. %.0f%% of arm length in 3-D"
      % (100 * abs(xHS_c) / reach_c,
         100 * np.sqrt(xHS_c ** 2 + (h_c[1] - s_c[1]) ** 2 + (h_c[2] - s_c[2]) ** 2) / L))
if abs(xHS_c) <= reach_c:
    print("     -> it FITS: the arm is long enough, so the geometry does close.")
else:
    print("     -> it does NOT fit: the arm is too short by %.3f m." % (abs(xHS_c) - reach_c))

# d_L/F from eq. 8 at the catch, all relative to the stretcher
xBF_c = hip_c - ank
xSB_c = s_c[0] - hip_c
dlf = S * np.sin(th_c) + xBF_c + xSB_c + xHS_c
print()
print("EQ. 8 AT THE CATCH:  d_L/F = s sin(th) + x_B/F + x_S/B + x_H/S")
print("                          = %.3f + %.3f + %.3f + %.3f = %.3f m"
      % (S * np.sin(th_c), xBF_c, xSB_c, xHS_c, dlf))
xBF_f = hip_f - ank
xSB_f = s_f[0] - hip_f
print("  check at the finish:      %.3f + %.3f + %.3f + %.3f = %.3f m"
      % (S * np.sin(th_f), xBF_f, xSB_f, xHS_f, S * np.sin(th_f) + xBF_f + xSB_f + xHS_f))
print()
print("  the model hard-codes d_L/F = %.3f m (station_x + 0.30)" % (pin - ank))
print("  derived value              = %.3f m   -> the stretcher is %.3f m out" % (dlf, (pin - ank) - dlf))
