"""Phase 4.3a, the decisive study: give the air phase her measured hand path.

entry_arc_vs_her.py located the entry defect. It is not the catch angle -- hers is
+60.42 deg and the prescribed arc starts at +60.49 -- it is the RATE. The arc
accelerates too gently, running about 24 deg/s slow through the whole air phase, and
that one defect does three things:

    the blade enters 29 ms late   ([CR06] eq. 16 fires at 0.0521 s on her measured
                                   trace, at 0.0815 s on the arc, a 56% longer air
                                   phase, because the slower rate delays the slip
                                   reaching zero)
    the torque drive inherits an angle 1.44 deg behind hers
    and inherits a rate 24 deg/s slower than hers

This replaces ONLY the air phase's prescribed pose with her measured angle and rate,
by overriding DynamicOarSimulator._sweep_pose -- the single point where
_integrate_sweep_catch reads the arc. Everything else in the simulation is untouched:
the same torque from her measured handle force, the same tier-1 blade at C2 = 140.88,
the same body field, the same hull. The entry criterion is still [CR06] eq. 16, now
evaluated on her rate.

This is a STUDY, not a fix. Feeding the model her measured hand path is not a
predictive model -- it measures how much of the drive lag the entry is worth, and so
how much a body-driven catch would have to buy back. A real fix has to come from the
rower's body driving the oar through the entry, which is phase 4.3's subject.
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


def load(name):
    a, b = [], []
    with open(os.path.join(HERE, "cr06_fig3_measured.csv")) as fh:
        for row in csv.DictReader(r for r in fh if not r.startswith("#") and not r.startswith('"#')):
            if row["series"] == name:
                a.append(float(row["t_s"])); b.append(float(row["value"]))
    a, b = np.array(a), np.array(b)
    o = np.argsort(a)
    return a[o], b[o]


NH = 10
W = 2 * np.pi / T_CR06
KH = np.arange(1, NH + 1)


def fit(ta, ya):
    A = np.hstack([np.ones((ta.size, 1)), np.cos(np.outer(ta, KH) * W), np.sin(np.outer(ta, KH) * W)])
    return np.linalg.lstsq(A, ya, rcond=None)[0]


tA, A = load("oar_angle_deg")
CA = fit(tA, A)


def her_angle(tau):
    x = np.mod(float(tau), T_CR06)
    return float(np.radians(CA[0] + np.cos(x * KH * W) @ CA[1:NH + 1] + np.sin(x * KH * W) @ CA[NH + 1:]))


def her_rate(tau):
    x = np.mod(float(tau), T_CR06)
    return float(np.radians(-np.sin(x * KH * W) @ (CA[1:NH + 1] * KH * W)
                            + np.cos(x * KH * W) @ (CA[NH + 1:] * KH * W)))


def run(use_her_entry):
    oardynamics.InertiaProfile.of = staticmethod(oar_only)
    try:
        boat = research.apply(her_boat(stature))
        boat.power_scales = np.ones(boat.n_seats)
        lock = boat.rig.seats[0].oarlocks[0]
        r_h = float(lock.oar.inboard)
        sim = DynamicOarSimulator(boat, peak_torque=1.0, catch=research.catch,
                                  coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
        tq = lambda t: 0.5 * max(float(np.interp(np.mod(t, T_CR06), t_F, F_hand, period=T_CR06)), 0.0) * r_h
        sim._torque = lambda slot, ang, state, t: tq(t)
        sim.crew_field = body_field(boat, True, K)
        if use_her_entry:
            sim._sweep_pose = lambda tau: (her_angle(tau), her_rate(tau))
        r = sim.run_strokes(20, surge_speed=4.10, dt=integrators.estimate_step(T_CR06) / 4.0)
        return sim, r
    finally:
        oardynamics.InertiaProfile.of = REAL_OF


print("%-26s %9s %9s %9s %9s %9s" % ("", "entry s", "0.30 deg", "0.50 deg", "0.70 deg", "speed"))
for label, flag in (("prescribed arc (as built)", False), ("her measured hand path", True)):
    sim, r = run(flag)
    t, y = r.last_time, r.last_states
    th = np.degrees(y[STATE_SIZE])
    tt = np.mod(t - t[0], T_CR06)
    v = float(np.mean(np.hypot(y[6], y[7])))
    ent = [e for e in sim.entries][-1][1] if sim.entries else float("nan")
    ent_tau = np.mod(ent - t[0], T_CR06) if sim.entries else float("nan")
    outs = []
    for q in (0.30, 0.50, 0.70):
        i = int(np.argmin(np.abs(tt - q)))
        outs.append(th[i] - np.degrees(her_angle(tt[i])))
    print("%-26s %9.4f %+9.2f %+9.2f %+9.2f %9.4f" % (label, ent_tau, outs[0], outs[1], outs[2], v))

print()
print("her own entry by [CR06] eq. 16 is 0.0521 s; her trace's mean speed is 4.1911 m/s")
print("the recorded gate lags +2.66 / +3.89 / +4.26 deg at 4.0901 m/s")
