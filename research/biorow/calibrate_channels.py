"""Calibrate [BR24]'s seat and trunk channels from the data, and derive the rower's
centre-of-mass travel relative to the hull (docs/SOURCES.md sec. 156).

    python research/biorow/calibrate_channels.py

The file gives seat position Ls and trunk position Lt without their origins or frame.
Four tests, each using only the data, de Leva (1996) and published postures:

1. The velocity channels are the derivatives of the positions, so the origins are
   constant offsets and drop out of any momentum balance.
2. Ls is zero at the catch. Anchoring the hip over the ankle with a published catch
   posture predicts the seat travel, which is then compared with the measured 0.597 m.
3. Lt cannot be in the hull's frame: read that way, trunk-relative-to-hip is more
   forward at the finish than at the catch. Read relative to the seat it is not.
4. The height of the point Lt tracks is fitted by whole-system horizontal momentum,

       d/dt [ M v + sum_i m_i xdot_i ] = lam * sum_oars H cos(A) - k v^2,

   integrated once in time (the averaged data do not survive two derivatives):

       M v + sum_i m_i xdot_i = lam * int F dt - k * int v^2 dt + C.

   With the trunk rigid about the hip, every trunk and head mass moves as
   (its height / tracked-point height) * Vt, so the tracked height enters linearly.

The data are commercial (gitignored in data/local/biorow/) and only derived numbers
are printed. Legs use de Leva's placements: thigh CM 40.95% from the hip, shank 44.59%
from the knee, foot fixed on the stretcher.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = np.genfromtxt(os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv"),
                  delimiter=",", names=True)
X = D[:-1]                                   # the last row closes the cycle
N = len(X)
T = float(D["Ti"][-1])
dt = T / N

# ---- athlete and rig, from the file's header
m_body, stature, m_hull, m_oar = 97.0, 1.91, 18.0, 1.2      # scull mass: [CR06] Table 1
inboard, oar_len, blade_len = 0.875, 2.885, 0.43
s = stature / 1.741                                          # de Leva male reference stature
L_th, L_sh = 0.4222 * s, 0.4340 * s
lower, mid, upper, head_len = 0.1457 * s, 0.2155 * s, 0.2421 * s, 0.2033 * s
pct = dict(head=6.94, upper=15.96, mid=16.33, lower=11.17, uarm=2.71, farm=1.62,
           hand=0.61, thigh=14.16, shank=4.33, foot=1.37)
m = {k: v / 100.0 * m_body for k, v in pct.items()}
stack = lower + mid + upper
h_sh = stack * (1 - 0.085)                  # shoulder joint, kinematics.SHOULDER_DROP_FRACTION
h_parts = {"lower": lower * (1 - 0.6115), "mid": lower + mid * (1 - 0.4502),
           "upper": lower + mid + upper * (1 - 0.5066), "head": stack + head_len * 0.5976}
m_TH = sum(m[k] for k in h_parts)
h_TH = sum(m[k] * h for k, h in h_parts.items()) / m_TH
m_arm = 2 * (m["uarm"] + m["farm"] + m["hand"])
ua = 0.2817 * s
arm_com = (m["uarm"] * 0.5772 * ua + m["farm"] * (ua + 0.4574 * 0.2689 * s)
           + m["hand"] * (ua + 0.2689 * s + 0.79 * 0.0862 * s)) / (m["uarm"] + m["farm"] + m["hand"])
alpha = arm_com / (ua + (0.2689 + 0.0862) * s)             # straight-arm CoM fraction to the hand
r_oar_com = oar_len / 2 - inboard
M_tot = m_hull + 2 * m_oar + m_body

A1, A2 = np.radians(X["A1"]), np.radians(X["A2"])
v = X["Vs"]
F = X["H1"] * np.cos(A1) + X["H2"] * np.cos(A2)


def deriv(y):
    k = np.fft.rfftfreq(N, d=dt) * 2 * np.pi
    return np.fft.irfft(np.fft.rfft(y) * 1j * k, n=N)


def cumint(y):
    return np.concatenate([[0], np.cumsum((y[1:] + y[:-1]) / 2 * dt)])


# ---- 1. velocities are derivatives of positions
print("1. corr(dLs/dt, Vs_2) %.3f   corr(dLt/dt, Vt) %.3f"
      % (np.corrcoef(deriv(X["Ls"]), X["Vs_2"])[0, 1], np.corrcoef(deriv(X["Lt"]), X["Vt"])[0, 1]))
w = deriv(X["A1"] * np.pi / 180)
r_h = float(np.polyfit(w, X["Vh1"], 1)[0])
print("   handle radius from Vh1 / (dA1/dt): %.3f m (inboard %.3f)" % (r_h, inboard))


# ---- 2. seat origin from a published catch posture
def catch_hip(shank_deg, knee_deg):
    a_s = np.radians(180 - shank_deg)
    kx, ky = L_sh * np.cos(a_s), L_sh * np.sin(a_s)
    a_t = a_s + np.pi + np.radians(knee_deg)
    return ky + L_th * np.sin(a_t), kx + L_th * np.cos(a_t)


def finish_hip_x(H, knee_deg=171.2):
    Dd = np.sqrt(L_th ** 2 + L_sh ** 2 - 2 * L_th * L_sh * np.cos(np.radians(knee_deg)))
    return np.sqrt(Dd ** 2 - H ** 2)


i_catch = int(np.argmin(X["A1"]))
print("2. Ls at the catch %.3f m, range %.3f m" % (X["Ls"][i_catch], np.ptp(X["Ls"])))
anchors = {}
for name, shank, knee in (("Caplan & Gardner catch (shank 91.6, knee 41.0)", 91.6, 41.0),
                          ("Kleshnev on-water catch knee 45.4, shank 91.6", 91.6, 45.4)):
    H, d_c = catch_hip(shank, knee)
    anchors[name] = (H, d_c)
    print("   %-48s predicts seat travel %.3f m" % (name, finish_hip_x(H) - d_c))

# ---- 3. frame of the trunk channel
i_fin = int(np.argmax(X["A1"]))
print("3. hull-frame reading: trunk rel. hip %.3f at catch -> %.3f at finish (must increase)"
      % (X["Lt"][i_catch] - X["Ls"][i_catch], X["Lt"][i_fin] - X["Ls"][i_fin]))
print("   seat-relative reading: %.3f -> %.3f" % (X["Lt"][i_catch], X["Lt"][i_fin]))


# ---- 4. momentum balance, velocity level
def legs_velocity(anchor):
    H, d_c = anchors[anchor]
    hx = d_c + X["Ls"]
    Dd = np.hypot(hx, H)
    ang = np.arctan2(H, hx) + np.arccos(np.clip((L_sh ** 2 + Dd ** 2 - L_th ** 2) / (2 * L_sh * Dd), -1, 1))
    kx_d = deriv(L_sh * np.cos(ang))
    return X["Vs_2"] + 0.4095 * (kx_d - X["Vs_2"]), (1 - 0.4459) * kx_d


hand_d = (X["Vh1"] * np.cos(A1) + X["Vh2"] * np.cos(A2)) / 2 * (r_h / r_h)
oar_d = -(r_oar_com / r_h) * (X["Vh1"] * np.cos(A1) + X["Vh2"] * np.cos(A2))
c_t = m_TH * h_TH + m_arm * (1 - alpha) * h_sh          # multiplies Vt / h_p


def fit(anchor, h_p=None, m_hull_=m_hull, absolute=False):
    thigh_d, shank_d = legs_velocity(anchor)
    Mt = m_hull_ + 2 * m_oar + m_body
    base = (Mt * v + 2 * m["thigh"] * thigh_d + 2 * m["shank"] * shank_d
            + (m_TH + m_arm * (1 - alpha)) * X["Vs_2"] + m_arm * alpha * hand_d + m_oar * oar_d)
    cols = [cumint(F), -cumint(v ** 2), np.ones(N)]
    if absolute:                     # Lt as the shoulder in the hull frame
        base = (Mt * v + 2 * m["thigh"] * thigh_d + 2 * m["shank"] * shank_d
                + m_TH * (X["Vs_2"] + h_TH / h_sh * (X["Vt"] - X["Vs_2"]))
                + m_arm * ((1 - alpha) * X["Vt"] + alpha * hand_d) + m_oar * oar_d)
    elif h_p is None:
        cols.append(-c_t * X["Vt"])
    else:
        base = base + c_t / h_p * X["Vt"]
    Z = np.column_stack(cols)
    cf, *_ = np.linalg.lstsq(Z, base, rcond=None)
    res = base - Z @ cf
    return cf, float(res.std()), 1 - res.var() / base.var()


anchor = "Kleshnev on-water catch knee 45.4, shank 91.6"
cf, rms, r2 = fit(anchor)
h_fit = 1 / cf[3]
print("4. free fit: tracked point %.3f m above the hip (shoulder joint %.3f); lever %.3f, "
      "k %.2f, rms %.1f N s, R2 %.3f" % (h_fit, h_sh, cf[0], cf[1], rms, r2))
for label, kw in (("hull 15 kg", dict(m_hull_=15.0)), ("hull 21 kg", dict(m_hull_=21.0)),
                  ("Caplan & Gardner anchor", dict(anchor="Caplan & Gardner catch (shank 91.6, knee 41.0)"))):
    kw.setdefault("anchor", anchor)
    print("   %-26s tracked point %.3f m" % (label, 1 / fit(**kw)[0][3]))
for label, h in (("shoulder joint", h_sh), ("cervicale", stack), ("0.40 m", 0.40)):
    print("   fixed at %-16s rms %.1f N s" % (label, fit(anchor, h_p=h)[1]))
print("   hull-frame reading     rms %.1f N s" % fit(anchor, absolute=True)[1])

# ---- the centre of mass
thigh_d, shank_d = legs_velocity(anchor)
for label, h_p in (("fitted", h_fit), ("shoulder joint", h_sh)):
    xd = (2 * m["thigh"] * thigh_d + 2 * m["shank"] * shank_d
          + m_TH * (X["Vs_2"] + h_TH / h_p * X["Vt"])
          + m_arm * ((1 - alpha) * (X["Vs_2"] + h_sh / h_p * X["Vt"]) + alpha * hand_d)) / m_body
    x = cumint(xd)
    print("CoM travel relative to the hull (%s): %.3f m" % (label, np.ptp(x)))
