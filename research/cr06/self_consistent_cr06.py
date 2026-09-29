"""One athlete, one stroke: does the model, driven by [CR06]'s own measured inputs, predict her boat?

measured_inputs_vs_legge.py mixed athletes: [CR06]'s leg time-law (one women's single at
30.9 spm) under [LE26]'s force (25 elite scullers at 34-37 spm), with the body's
accelerations rescaled by (1.94/P)^2. This removes that assumption. Every input is from
[CR06] Fig. 3, the same stroke, and the model runs at her own period, T = 1.94 s.

Rig, [CR06] Table 1 and A.5 (singles):
  rower 75 kg; boat 19.7 kg (15.8 rigged + 3.9 telemetry)
  oar mass 1.2 kg; force point s = 0.83 m from the lock; blade centre l = 1.805 m
    -> Oar(length = 0.83 + 1.805 + 0.43/2 = 2.85, inboard 0.83, blade_length 0.43)
       uniform-rod inertia about the lock 1.237 kg m^2 against [CR06]'s I_G + m d^2 = 1.233
  oar arc: her measured 60.49 deg at the catch, -44.35 deg at the finish
  drive fraction: the model's own rate law gives 0.462 at 30.9 spm; her release marker
    sits at 0.894 / 1.94 = 0.461, so it is left alone
Not given by [CR06], so marked: stature (1.787 m, [LE26]'s elite women), water (the
model's fresh-water default), hull shape (the catalog 1x).

Inputs, both as functions of time from her catch (t = 0, maximum oar angle):
  handle torque = F_hand(t) * s, clipped at zero. [CR06]'s model scales "the oar force
  and mass by 2" for sculling, so Fig. 3's F_hand is read as PER OAR. The other reading
  (both hands summed, so F/2 per oar) is run too, and the predicted speed decides.
  body: the catalog body warped onto her measured leg displacement (warp lam 1e-4, as
  before) -- now at its native rate, no rescaling.

Nothing is fitted: the force scale is 1. Scored against her measured boat velocity:
  mean speed (4.19 m/s), rms of the velocity trace about its mean, and the oar angle.
"""
import csv
import os
import sys
import time

import numpy as np
from scipy.interpolate import make_smoothing_spline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

from coxswain import physics                                # noqa: E402
from coxswain.boats import catalog                          # noqa: E402
from coxswain.boats.boat import Boat                        # noqa: E402
from coxswain.boats.rig import Oar, build_sculling_rig      # noqa: E402
from coxswain.core.state import STATE_SIZE                  # noqa: E402
from coxswain.crew import oardynamics                       # noqa: E402
from coxswain.sim.control import Coxswain                   # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T_CR06 = 1.94
RATE = 60.0 / T_CR06
CR06_OAR = Oar(length=0.83 + 1.805 + 0.215, inboard=0.83, blade_area=0.0903, blade_length=0.43, mass=1.2)
ARC = catalog.SCULLING_ARC.__class__(catch_angle=np.radians(60.49), finish_angle=np.radians(-44.35))
research = physics.resolve("research")
REAL_OF = oardynamics.InertiaProfile.__dict__["of"]


def series(name, path, key):
    rows = csv.DictReader(l for l in open(path) if not l.lstrip('"').startswith("#"))
    d = np.array(sorted((float(r["t_s"]), float(r[key])) for r in rows if r["series"] == name))
    s = np.mod(d[:, 0], T_CR06)
    o = np.argsort(s)
    s, v = s[o], d[o, 1]
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    return s[keep], v[keep]


# the committed re-extraction of Fig. 3 (research/cr06/extract_fig3.py); body curves in the same file
MEAS = os.path.join(ROOT, "data", "literature", "cr06_fig3_measured.csv")
t_F, F_hand = series("handle_force_N", MEAS, "value")
t_v, v_boat = series("boat_velocity_mps", MEAS, "value")
t_a, angle = series("oar_angle_deg", MEAS, "value")

# measured leg, pre-smoothed and normalised, as in warped_body_vs_legge.py
t_L, L_raw = series("leg_disp_m", MEAS, "value")
k0 = int(np.argmin(L_raw))
s_raw = np.mod(t_L - t_L[k0], T_CR06) / T_CR06
o = np.argsort(s_raw)
s_raw, L_raw = s_raw[o], L_raw[o]
_sp = make_smoothing_spline(np.concatenate([s_raw - 1, s_raw, s_raw + 1]) * T_CR06,
                            np.concatenate([L_raw, L_raw, L_raw]), lam=3e-6)
S_MEAS = np.linspace(0.0, 1.0, 4001, endpoint=False)
L_MEAS = _sp(S_MEAS * T_CR06)
L_MEAS = (L_MEAS - L_MEAS.min()) / np.ptp(L_MEAS)
WARP_LAM = 1e-4


def cr06_single():
    base = catalog.single_scull(rate=RATE, rower_mass=75.0, rower_stature=1.787)
    rig = build_sculling_rig(n_seats=1, spacing=1.22, stern_station=-0.35, span=0.80,
                             oarlock_height=0.32, oar=CR06_OAR)
    return Boat(name="CR06 women's single", offsets=base.offsets, rig=rig, hull_mass=19.7,
                hull_inertia=base.hull_inertia, timing=base.timing, appendages=base.appendages,
                water=base.water, default_anthropometry=catalog.RowerAnthropometry(mass=75.0, stature=1.787),
                force_profile=base.force_profile, oar_sweep=ARC)


def oar_only(boat, seat=0, **kwargs):
    return float(boat.rig.seats[seat].oarlocks[0].oar.inertia_about_lock)


def build_warp(boat):
    P = float(boat.timing.period)
    rower = boat.crew[0].rower
    t = np.linspace(0.0, P, 1601, endpoint=False)
    leg = np.array([(lambda j: j["hip"][0] - j["ankle"][0])(rower.joint_positions(float(x))) for x in t])
    kmin = int(np.argmin(leg))
    tau_grid = np.mod(t - t[kmin], P) / P
    oo = np.argsort(tau_grid)
    tau_grid, L_model = tau_grid[oo], (leg[oo] - leg.min()) / np.ptp(leg)
    k_max_m, k_max_c = int(np.argmax(L_model)), int(np.argmax(L_MEAS))
    drive_m = np.maximum.accumulate(L_model[:k_max_m + 1]) + np.arange(k_max_m + 1) * 1e-9
    rec_m = np.minimum.accumulate(L_model[k_max_m:])[::-1] + np.arange(L_model.size - k_max_m) * 1e-9
    tau_rec = tau_grid[k_max_m:][::-1]
    tau = np.where(S_MEAS <= S_MEAS[k_max_c], np.interp(L_MEAS, drive_m, tau_grid[:k_max_m + 1]),
                   np.interp(L_MEAS, rec_m, tau_rec))
    tau = np.maximum.accumulate(tau)
    spline = make_smoothing_spline(np.concatenate([S_MEAS - 1, S_MEAS, S_MEAS + 1]),
                                   np.concatenate([tau - 1, tau, tau + 1]), lam=WARP_LAM)
    return spline, t[kmin] / P, P, np.ptp(leg)


def warped_field(boat, spline, t_min_offset, P):
    d1, d2 = spline.derivative(1), spline.derivative(2)

    def field(t, exact=False):
        real = t / P - t_min_offset
        k = np.floor(real)
        s = real - k
        T = P * (k + float(spline(s)) + t_min_offset)
        tp, tpp = float(d1(s)), float(d2(s))
        mass, pos, vel, acc = boat.crew_field(T, exact=True)
        return mass, pos, vel * tp, acc * tp ** 2 + vel * tpp / P
    return field


#: Half the default fixed step: the 1.2 kg oar under the oar-only balance diverges at T/80
#: (capped 0.5/80 s); at /2 and /4 the mean speed agrees to 0.6 mm/s (diag_cr06_step.py).
from coxswain.core import integrators                        # noqa: E402
DT = integrators.estimate_step(T_CR06) / 2.0
BUILDS = (("summed, smooth body", 0.5, False), ("summed, measured leg", 0.5, True),
          ("per oar, measured leg", 1.0, True))
v_mean_meas = float(np.trapezoid(np.interp(np.linspace(0, T_CR06, 2001), t_v, v_boat, period=T_CR06),
                                 np.linspace(0, T_CR06, 2001)) / T_CR06)
grid = np.linspace(0.0, 1.0, 400, endpoint=False)
v_meas_g = np.interp(grid * T_CR06, t_v, v_boat, period=T_CR06)
ang_meas_g = np.interp(grid * T_CR06, t_a, angle, period=T_CR06)
fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
axes[0].plot(grid, v_meas_g, "k", lw=2.2, label="CR06 measured")
axes[1].plot(grid, ang_meas_g, "k", lw=2.2, label="CR06 measured")
axes[2].plot(grid, np.interp(grid * T_CR06, t_F, F_hand, period=T_CR06), "k", lw=2.2, label="CR06 F_hand (input)")
print("CR06 measured: mean speed %.3f m/s, velocity range %.2f..%.2f, leg travel %.3f m"
      % (v_mean_meas, v_boat.min(), v_boat.max(), np.ptp(_sp(S_MEAS * T_CR06))), flush=True)
print("%-24s | %6s %6s | %6s %6s %6s | %6s %6s | %s" % ("build", "speed", "err%", "v min", "v max", "v rms",
                                                        "ang c", "ang f", "power"), flush=True)
for (label, per_oar, warped), colour in zip(BUILDS, ("tab:orange", "tab:green", "tab:blue")):
    t0 = time.time()
    oardynamics.InertiaProfile.of = staticmethod(oar_only)
    try:
        boat = research.apply(cr06_single())
        boat.power_scales = np.ones(boat.n_seats)
        r_h = float(boat.rig.seats[0].oarlocks[0].oar.inboard)
        if warped:
            spline, t_min_offset, P, leg_travel = build_warp(boat)
            field = warped_field(boat, spline, t_min_offset, P)

        def torque(slot, ang, state, t, _k=per_oar):
            return _k * max(float(np.interp(np.mod(t, T_CR06), t_F, F_hand, period=T_CR06)), 0.0) * r_h

        sim = DynamicOarSimulator(boat, peak_torque=1.0, catch=research.catch,
                                  coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
        sim._torque = torque
        if warped:
            sim.crew_field = field
        run = sim.run_strokes(20, surge_speed=v_mean_meas, dt=DT)
    finally:
        oardynamics.InertiaProfile.of = REAL_OF
    times, states = run.last_time, run.last_states
    P = float(boat.timing.period)
    frac = np.mod(times - times[0], P) / P
    o = np.argsort(frac)
    speed = np.hypot(states[6], states[7])
    v_m = np.interp(grid, frac[o], speed[o], period=1.0)
    ang_m = np.degrees(np.interp(grid, frac[o], states[STATE_SIZE + 0][o], period=1.0))
    rms = float(np.sqrt(np.mean(((v_m - v_m.mean()) - (v_meas_g - v_meas_g.mean())) ** 2)))
    print("%-24s | %6.3f %+5.1f%% | %6.2f %6.2f %6.3f | %6.1f %6.1f | %6.1fW  (%.0f s)"
          % (label, run.settled_speed(), 100 * (run.settled_speed() / v_mean_meas - 1), v_m.min(), v_m.max(), rms,
             ang_m.max(), ang_m.min(), run.settled_power(), time.time() - t0), flush=True)
    axes[0].plot(grid, v_m, color=colour, lw=1.3, label=label)
    axes[1].plot(grid, ang_m, color=colour, lw=1.3, label=label)
axes[0].set_ylabel("boat velocity (m/s)")
axes[1].set_ylabel("oar angle (deg)")
axes[2].set_ylabel("handle force (N)")
axes[2].set_xlabel("fraction of the cycle from the catch")
for ax in axes:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "self_consistent_cr06.png"), dpi=110)
print("wrote self_consistent_cr06.png")
