"""Why does the model's oar lag hers by 2.7-4.3 deg through the drive?

The end-to-end gate (her_via_profile.out) reports the model's oar angle running
AHEAD of hers in value -- +2.66 deg at 0.30 s, +3.89 at 0.50, +4.26 at 0.70 --
and since her angle DECREASES through the drive, that means the model's oar is
sweeping too slowly. Her measured handle force drives it, so the candidates are:

    (a) the blade load is too high through the drive,
    (b) the oar's inertia is too high,
    (c) the model's slower hull (4.09 vs her 4.19) feeds back as extra slip.

(b) is already at its floor: the run uses the oar-only inertia profile with a
1.2 kg oar, and adding the rower's arms would only raise it. This measures (a)
and (c) directly, by comparing, at the SAME instant:

    the model's blade load,  F = -sign(w) C2 w^2,  w = l*thetadot + v cos(theta)
    her implied blade load,  F = (I_lock*thetadd + 0.5*F_hand*S + m d a_hull cos theta)/l

both on her clock, plus the slip each implies. If the model's load sits above
hers through the drive, (a) is the cause; if the loads agree but the slips differ,
(c) is.
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
oardynamics.InertiaProfile.of = staticmethod(oar_only)
try:
    boat = research.apply(her_boat(stature))
    boat.power_scales = np.ones(boat.n_seats)
    lock = boat.rig.seats[0].oarlocks[0]
    r_h = float(lock.oar.inboard)
    sim = DynamicOarSimulator(boat, peak_torque=1.0, catch=research.catch,
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    tq = lambda t: 0.5 * max(float(np.interp(np.mod(t, T_CR06), t_F, F_hand, period=T_CR06)), 0.0) * r_h
    ELL = float(sim._oars[0].outboard)
    C2 = float(sim._oars[0].blade.c2)
    sim._torque = lambda slot, ang, state, t: tq(t)
    sim.crew_field = body_field(boat, True, K)
    run = sim.run_strokes(20, surge_speed=4.10, dt=integrators.estimate_step(T_CR06) / 4.0)
    t, y = run.last_time, run.last_states
finally:
    oardynamics.InertiaProfile.of = REAL_OF

n = sim.n_oar_states
theta = y[STATE_SIZE]
dtheta = y[STATE_SIZE + n]
v_h = np.hypot(y[6], y[7])
ph = np.mod(t - t[0], T_CR06) / T_CR06
tt = ph * T_CR06

w_model = ELL * dtheta + v_h * np.cos(theta)
F_model = -np.sign(w_model) * C2 * w_model ** 2

# ---- her side, from the measured traces on the same clock ---------------------
S_ARM, I_LOCK, M_OAR = 0.83, 1.233, 1.2
D_CG = 0.83 + 1.805 + 0.215 - 2.85 / 2.0 - 0.83


def load(name):
    a, b = [], []
    with open(os.path.join(HERE, "cr06_fig3_measured.csv")) as fh:
        for row in csv.DictReader(r for r in fh if not r.startswith("#") and not r.startswith('"#')):
            if row["series"] == name:
                a.append(float(row["t_s"])); b.append(float(row["value"]))
    a, b = np.array(a), np.array(b)
    o = np.argsort(a)
    return a[o], b[o]


def fourier(ta, ya, nh, grid):
    wq = 2 * np.pi / T_CR06
    k = np.arange(1, nh + 1)
    A = np.hstack([np.ones((ta.size, 1)), np.cos(np.outer(ta, k) * wq), np.sin(np.outer(ta, k) * wq)])
    c = np.linalg.lstsq(A, ya, rcond=None)[0]
    a_, b_ = c[1:nh + 1], c[nh + 1:]
    cg, sg = np.cos(np.outer(grid, k) * wq), np.sin(np.outer(grid, k) * wq)
    return (c[0] + cg @ a_ + sg @ b_,
            (-sg @ (a_ * k * wq)) + (cg @ (b_ * k * wq)),
            (-cg @ (a_ * (k * wq) ** 2)) - (sg @ (b_ * (k * wq) ** 2)))


tA, A = load("oar_angle_deg")
tv, V = load("boat_velocity_mps")
tF, Fh = load("handle_force_N")
hd, hdd, hddd = fourier(tA, A, 10, tt)
her_th, her_dth, her_ddth = np.radians(hd), np.radians(hdd), np.radians(hddd)
her_v, her_a, _ = fourier(tv, V, 10, tt)
her_F, _, _ = fourier(tF, Fh, 10, tt)

her_blade = (I_LOCK * her_ddth + 0.5 * her_F * S_ARM + M_OAR * D_CG * her_a * np.cos(her_th)) / ELL
her_w = ELL * her_dth + her_v * np.cos(her_th)

print("blade centre outboard %.3f m, C2 %.2f, inboard %.3f m" % (ELL, C2, r_h))
print("model mean hull speed %.4f m/s (her trace %.4f)" % (v_h.mean(), her_v.mean()))
print()
print("    t   theta model/her  (lag)  | slip model/her  | blade load model/her  | hull v m/h")
for q in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
    i = int(np.argmin(np.abs(tt - q)))
    print("  %.2f   %+6.1f %+6.1f %+6.2f | %6.2f %6.2f | %7.0f %7.0f | %.2f %.2f"
          % (q, np.degrees(theta[i]), np.degrees(her_th[i]),
             np.degrees(theta[i]) - np.degrees(her_th[i]),
             w_model[i], her_w[i], F_model[i], her_blade[i], v_h[i], her_v[i]))

d = (tt > 0.10) & (tt < 0.80)
print()
print("over the drive 0.10-0.80 s:")
print("  model blade load mean %.0f N, hers %.0f N   -> model/her %.3f"
      % (F_model[d].mean(), her_blade[d].mean(), F_model[d].mean() / her_blade[d].mean()))
print("  model slip mean %.3f m/s, hers %.3f m/s     -> model/her %.3f"
      % (w_model[d].mean(), her_w[d].mean(), w_model[d].mean() / her_w[d].mean()))
print("  model hull mean %.3f m/s, hers %.3f m/s" % (v_h[d].mean(), her_v[d].mean()))
print()
print("  if the loads agree and the slips differ, the lag is the slower hull (c);")
print("  if the model's load sits above hers, it is the blade law in the drive (a).")

# ---------------------------------------------------------------- the causal test
# Integrate the oar ALONE on her clock, from her catch, under her measured handle
# force and the SAME tier-1 law, changing only the hull velocity it sees:
#   (i)  her measured hull velocity
#   (ii) the model's hull velocity from the run above (3.2% slower through the drive)
# If (i) tracks her angle and (ii) reproduces the +4 deg lag, the lag IS the hull
# deficit -- the catch dip -- and not the blade law, the torque or the inertia.
np.savetxt(os.path.join(HERE, 'catch_lag_hull_cr06.csv'),
           np.column_stack([tt, v_h, np.degrees(theta)]), delimiter=',',
           header='t_s,model_hull_mps,model_oar_deg', comments='')

from scipy.integrate import solve_ivp

order = np.argsort(tt)
ts, vs = tt[order], v_h[order]


def oar_ode(v_of_t):
    def rhs(x, z):
        th, dth = z
        F_h = float(np.interp(np.mod(x, T_CR06), t_F, F_hand, period=T_CR06))
        vv = float(v_of_t(x))
        aa = float(np.interp(x, tt[order], her_a[order]))
        w = ELL * dth + vv * np.cos(th)
        Fb = -np.sign(w) * C2 * w ** 2
        return [dth, (ELL * Fb - 0.5 * F_h * S_ARM - M_OAR * D_CG * aa * np.cos(th)) / I_LOCK]
    return rhs


i0 = int(np.argmin(np.abs(tt - 0.02)))
z0 = [her_th[i0], her_dth[i0]]
span = (float(tt[i0]), 0.85)
grid = np.linspace(span[0], span[1], 200)
her_v_f = lambda x: np.interp(x, tt[order], her_v[order])
mod_v_f = lambda x: np.interp(x, ts, vs)

sol_her = solve_ivp(oar_ode(her_v_f), span, z0, t_eval=grid, rtol=1e-8, atol=1e-10)
sol_mod = solve_ivp(oar_ode(mod_v_f), span, z0, t_eval=grid, rtol=1e-8, atol=1e-10)
her_on_grid = np.interp(grid, tt[order], np.degrees(her_th)[order])
a_her = np.degrees(sol_her.y[0]); a_mod = np.degrees(sol_mod.y[0])

print()
print('=== the oar integrated alone, changing ONLY the hull velocity it sees ===')
print('     t    her    with her hull  (err)   with model hull  (lag)')
for q in (0.10, 0.30, 0.50, 0.70, 0.80):
    k = int(np.argmin(np.abs(grid - q)))
    print('  %.2f  %+6.1f   %+9.1f %+7.2f   %+11.1f %+7.2f'
          % (q, her_on_grid[k], a_her[k], a_her[k] - her_on_grid[k],
             a_mod[k], a_mod[k] - her_on_grid[k]))
m = grid > 0.05
print()
print('  rms vs her, 0.05-0.85 s:  her hull %.2f deg   model hull %.2f deg'
      % (np.sqrt(np.mean((a_her[m] - her_on_grid[m]) ** 2)),
         np.sqrt(np.mean((a_mod[m] - her_on_grid[m]) ** 2))))
print('  the full simulation lags by +2.66 / +3.89 / +4.26 deg at 0.30 / 0.50 / 0.70 s')
