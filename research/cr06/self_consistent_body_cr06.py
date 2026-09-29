"""One-athlete test with her body on her clock: leg AND back on her measured time-laws.

self_consistent_cr06.py warped the leg so her measured leg minimum fell at the MODEL's leg-minimum
time (0.993 of the cycle), while the imposed force runs on her clock (catch at t = 0, leg minimum at
0.004): the body reversed about 21 ms before the force. Its trunk was the catalog body's, which comes
forward 32 ms late and opens 62 ms early against her measured back displacement.

Two channels, each a monotone time warp of the catalog body onto one measured curve, with her
extreme placed at HER time on the force's clock:
  leg  channel  x_B/F = hip - ankle       onto her leg displacement
  back channel  x_S/B = shoulder - hip    onto her back displacement
The whole body follows the leg channel; the upper body (head, trunk above the lower trunk, arms)
takes its motion RELATIVE to the lower trunk from the back channel:
  a_upper = a_lower_trunk(leg time) + K * [a_upper - a_lower_trunk](back time)
K = 1, or 0.398 / 0.513 = her back travel over the model's at 1.787 m.

Builds (her rig, summed force, nothing fitted, half step):
  leg on her clock
  leg on her clock + back time-law
  leg on her clock + back time-law and travel
"""
import os
import warnings

import numpy as np
from scipy.interpolate import make_smoothing_spline

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "self_consistent_cr06.py")).read().split("BUILDS = (")[0]
exec(_src)

import matplotlib.pyplot as plt                              # noqa: E402
from coxswain.core import integrators                        # noqa: E402

DT = integrators.estimate_step(T_CR06) / 2.0
UPPER, REF = [0, 1, 2, 4, 5, 6, 7], 3
K_BACK = 0.398 / 0.513
BODY = MEAS                                  # body curves live in the measured file
FINE = np.linspace(0.0, T_CR06, 4000, endpoint=False)


def measured_channel(name):
    """Her curve, smoothed as before, normalised; its minimum's phase on her clock; phase grid from that minimum."""
    tt, vv = series(name, BODY, "value")
    sp = make_smoothing_spline(np.concatenate([tt - T_CR06, tt, tt + T_CR06]), np.concatenate([vv, vv, vv]), lam=3e-6)
    m = float(FINE[np.argmin(sp(FINE))]) / T_CR06
    s = np.linspace(0.0, 1.0, 4001, endpoint=False)
    curve = sp(np.mod(m + s, 1.0) * T_CR06)
    return m, s, (curve - curve.min()) / np.ptp(curve)


def model_channel(boat, fn):
    P = float(boat.timing.period)
    rower = boat.crew[0].rower
    t = np.linspace(0.0, P, 1601, endpoint=False)
    c = np.array([fn(rower.joint_positions(float(x))) for x in t])
    kmin = int(np.argmin(c))
    tau = np.mod(t - t[kmin], P) / P
    o = np.argsort(tau)
    return t[kmin] / P, tau[o], (c[o] - c.min()) / np.ptp(c)


def warp(s_meas, c_meas, tau_grid, c_model):
    """Monotone map from her phase (since her minimum) to model phase (since its minimum), as build_warp."""
    k_max_m, k_max_c = int(np.argmax(c_model)), int(np.argmax(c_meas))
    drive_m = np.maximum.accumulate(c_model[:k_max_m + 1]) + np.arange(k_max_m + 1) * 1e-9
    rec_m = np.minimum.accumulate(c_model[k_max_m:])[::-1] + np.arange(c_model.size - k_max_m) * 1e-9
    tau = np.where(s_meas <= s_meas[k_max_c], np.interp(c_meas, drive_m, tau_grid[:k_max_m + 1]),
                   np.interp(c_meas, rec_m, tau_grid[k_max_m:][::-1]))
    tau = np.maximum.accumulate(tau)
    return make_smoothing_spline(np.concatenate([s_meas - 1, s_meas, s_meas + 1]),
                                 np.concatenate([tau - 1, tau, tau + 1]), lam=WARP_LAM)


def channel_field(boat, spline, meas_min, model_min):
    P = float(boat.timing.period)
    d1, d2 = spline.derivative(1), spline.derivative(2)

    def field(t):
        real = t / P - meas_min
        k = np.floor(real)
        s = real - k
        T = P * (k + float(spline(s)) + model_min)
        tp, tpp = float(d1(s)), float(d2(s))
        mass, pos, vel, acc = boat.crew_field(T, exact=True)
        return mass, np.array(pos, float), np.array(vel, float) * tp, np.array(acc, float) * tp ** 2 + np.array(vel, float) * tpp / P
    return field


def body_field(boat, back, k_back):
    m_leg, s_leg, c_leg = measured_channel("leg_disp_m")
    mm_leg, tau_leg, cm_leg = model_channel(boat, lambda j: j["hip"][0] - j["ankle"][0])
    leg = channel_field(boat, warp(s_leg, c_leg, tau_leg, cm_leg), m_leg, mm_leg)
    if not back:
        return lambda t, exact=False: leg(t)
    m_b, s_b, c_b = measured_channel("back_disp_m")
    mm_b, tau_b, cm_b = model_channel(boat, lambda j: j["shoulder"][0] - j["hip"][0])
    bk = channel_field(boat, warp(s_b, c_b, tau_b, cm_b), m_b, mm_b)

    def field(t, exact=False):
        mass, pos, vel, acc = leg(t)
        _, pos_b, vel_b, acc_b = bk(t)
        for off in range(0, vel.shape[0] - vel.shape[0] % 12, 12):
            idx = [off + i for i in UPPER]
            r = off + REF
            pos[idx] = pos[r] + k_back * (pos_b[idx] - pos_b[r])
            vel[idx] = vel[r] + k_back * (vel_b[idx] - vel_b[r])
            acc[idx] = acc[r] + k_back * (acc_b[idx] - acc_b[r])
        return mass, pos, vel, acc
    return field


grid = np.linspace(0.0, 1.0, 400, endpoint=False)
v_meas_g = np.interp(grid * T_CR06, t_v, v_boat, period=T_CR06)
early = grid < 0.3
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(grid, v_meas_g, "k", lw=2.2, label="CR06 measured")
print("%-34s | %6s %6s | %10s %6s | %6s | %s" % ("build", "speed", "err%", "v min @", "v max", "v rms", "power"), flush=True)
print("%-34s | %6.3f %6s | %4.2f @%.3f %6.2f | %6s |" % ("CR06 measured", 4.191, "-", v_meas_g[early].min(),
                                                       grid[early][np.argmin(v_meas_g[early])], v_meas_g.max(), "-"), flush=True)
for (label, back, k_back), colour in zip((("leg on her clock", False, 1.0),
                                          ("leg on clock + back time-law", True, 1.0),
                                          ("leg on clock + back law & travel", True, K_BACK)),
                                         ("tab:green", "tab:purple", "tab:red")):
    oardynamics.InertiaProfile.of = staticmethod(oar_only)
    try:
        boat = research.apply(cr06_single())
        boat.power_scales = np.ones(boat.n_seats)
        r_h = float(boat.rig.seats[0].oarlocks[0].oar.inboard)
        sim = DynamicOarSimulator(boat, peak_torque=1.0, catch=research.catch,
                                  coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
        sim._torque = lambda slot, ang, state, t: 0.5 * max(
            float(np.interp(np.mod(t, T_CR06), t_F, F_hand, period=T_CR06)), 0.0) * r_h
        sim.crew_field = body_field(boat, back, k_back)
        run = sim.run_strokes(20, surge_speed=4.191, dt=DT)
    finally:
        oardynamics.InertiaProfile.of = REAL_OF
    times, states = run.last_time, run.last_states
    P = float(boat.timing.period)
    frac = np.mod(times - times[0], P) / P
    o = np.argsort(frac)
    v_m = np.interp(grid, frac[o], np.hypot(states[6], states[7])[o], period=1.0)
    rms = float(np.sqrt(np.mean(((v_m - v_m.mean()) - (v_meas_g - v_meas_g.mean())) ** 2)))
    print("%-34s | %6.3f %+5.1f%% | %4.2f @%.3f %6.2f | %6.3f | %5.1fW  last strokes %s" % (
        label, run.settled_speed(), 100 * (run.settled_speed() / 4.191 - 1), v_m[early].min(),
        grid[early][np.argmin(v_m[early])], v_m.max(), rms, run.settled_power(),
        np.round([s.mean_speed for s in run.strokes[-4:]], 3)), flush=True)
    ax.plot(grid, v_m, color=colour, lw=1.3, label=label)
ax.set_xlabel("fraction of the cycle from the catch")
ax.set_ylabel("boat velocity (m/s)")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "self_consistent_body_cr06.png"), dpi=110)
print("wrote self_consistent_body_cr06.png")
