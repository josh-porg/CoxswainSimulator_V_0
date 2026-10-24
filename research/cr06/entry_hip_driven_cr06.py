"""Phase 4.3a, a PREDICTIVE candidate fix: let the seat carry the hands into the catch.

entry_body_speeds_cr06.py found the thing that matters. Through the air phase:

    her hip                0.487 m/s      the model's hip   0.510 m/s   (1.05x hers)
    her shoulder           0.982 m/s
    the hand her oar needs 0.455 m/s

so HER HAND SPEED IS HER HIP SPEED (1.07x), not her shoulder's (2.16x) -- at the
catch the arms are straight and the hands ride the seat. And the model's hip is
already right. Yet the model's oar sweeps at 0.65x her rate, because the prescribed
arc drags the hands at 0.284 m/s -- 0.58x the model's OWN hip -- and the architecture
makes the arms absorb the mismatch instead of the oar.

So: during the air phase, put the handle where the seat has carried it. With the
hand at radius r_h from the pin, d(hand_x) = -r_h cos(theta) d(theta), hence

    sin(theta(t)) = sin(theta_catch) - dx_hip(t) / r_h
    dtheta/dt     = -v_hip(t) / (r_h cos(theta))

This is predictive: dx_hip comes from the model's own body, never from her oar. It
is also sourced in the plainest way -- the catch is taken with straight arms, so the
hands go where the seat goes -- rather than fitted to anything.

Run as a study on her stroke. Nothing in the repository is modified.
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

K = 0.398 / back
NH, W = 10, 2 * np.pi / T_CR06
KH = np.arange(1, NH + 1)


def _srs(path, name, col):
    a, b = [], []
    with open(os.path.join(HERE, path)) as fh:
        for row in csv.DictReader(r for r in fh if not r.startswith("#") and not r.startswith('"#')):
            if row["series"] == name:
                a.append(float(row["t_s"])); b.append(float(row[col]))
    a, b = np.array(a), np.array(b)
    o = np.argsort(a)
    return a[o], b[o]


def _fit(ta, ya):
    A = np.hstack([np.ones((ta.size, 1)), np.cos(np.outer(ta, KH) * W), np.sin(np.outer(ta, KH) * W)])
    return np.linalg.lstsq(A, ya, rcond=None)[0]


c_ang = _fit(*_srs("cr06_fig3_measured.csv", "oar_angle_deg", "value"))


def her_deg(tau):
    x = np.mod(float(tau), T_CR06)
    return float(c_ang[0] + np.cos(x * KH * W) @ c_ang[1:NH + 1] + np.sin(x * KH * W) @ c_ang[NH + 1:])


def run(mode):
    oardynamics.InertiaProfile.of = staticmethod(oar_only)
    try:
        boat = research.apply(her_boat(stature))
        boat.power_scales = np.ones(boat.n_seats)
        lock = boat.rig.seats[0].oarlocks[0]
        r_h = float(lock.oar.inboard)
        rower = boat.crew[0].rower
        sim = DynamicOarSimulator(boat, peak_torque=1.0, catch=research.catch,
                                  coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
        tq = lambda t: 0.5 * max(float(np.interp(np.mod(t, T_CR06), t_F, F_hand, period=T_CR06)), 0.0) * r_h
        sim._torque = lambda slot, ang, state, t: tq(t)
        sim.crew_field = body_field(boat, True, K)

        if mode == "hip":
            th0 = float(boat.oar_sweep.catch_angle)
            hip0 = float(rower.joint_positions(0.0)["hip"][0])
            eps = 1e-4 * T_CR06

            def hip_pose(tau):
                x = float(np.mod(float(tau), T_CR06))
                dx = float(rower.joint_positions(x)["hip"][0]) - hip0
                v = (float(rower.joint_positions(min(x + eps, T_CR06))["hip"][0])
                     - float(rower.joint_positions(max(x - eps, 0.0))["hip"][0])) / (2 * eps)
                sn = np.clip(np.sin(th0) - dx / r_h, -0.999, 0.999)
                th = float(np.arcsin(sn))
                return th, float(-v / (r_h * np.cos(th)))
            sim._sweep_pose = hip_pose

        r = sim.run_strokes(20, surge_speed=4.10, dt=integrators.estimate_step(T_CR06) / 4.0)
        return sim, r
    finally:
        oardynamics.InertiaProfile.of = REAL_OF


print("%-28s %9s %9s %9s %9s %9s" % ("", "entry s", "0.30", "0.50", "0.70", "speed"))
for label, mode in (("prescribed arc (as built)", "arc"), ("seat carries the hands", "hip")):
    sim, r = run(mode)
    t, y = r.last_time, r.last_states
    th = np.degrees(y[STATE_SIZE])
    tt = np.mod(t - t[0], T_CR06)
    v = float(np.mean(np.hypot(y[6], y[7])))
    ent = np.mod(sim.entries[-1][1] - t[0], T_CR06) if sim.entries else float("nan")
    o = [th[int(np.argmin(np.abs(tt - q)))] - her_deg(q) for q in (0.30, 0.50, 0.70)]
    print("%-28s %9.4f %+9.2f %+9.2f %+9.2f %9.4f" % (label, ent, o[0], o[1], o[2], v))

print()
print("her own entry by [CR06] eq. 16 is 0.0521 s; her mean speed is 4.1911 m/s")
print("her measured hand path gave  0.0516   +0.50  +1.02  +1.03   4.1375  (a study, not predictive)")
