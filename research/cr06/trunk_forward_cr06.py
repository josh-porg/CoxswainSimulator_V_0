"""Step 4.3, rungs 3 and 4 (rebuilt 2026-10-24): her trunk driven forward by the hip torque, open loop and
under a delayed feedback controller; and rung 5's linear stability check.

    .venv/Scripts/python.exe research/cr06/trunk_forward_cr06.py

The originals (cr06/forward_hip_cr06.py, tracking_hip_cr06.py, delayed_pd_linear.py, 2026-09-14) were lost
with a session scratchpad; their results are in docs/TRACKING.md ("Step 4.3, third rung", "fourth rung" and
the "Linear check" under the fifth). This rebuild follows the record and prints the recorded numbers beside
its own.

Rung 3. The trunk angle becomes a state; the hip's motion, the hull's acceleration, the handle force and the
arms stay prescribed (one-way: the hull does not feel the trunk). The hip balance of hip_topdown_cr06.py is
affine in the trunk's angular acceleration, M_ext = M0(t, a, a') + s alpha with s = -(effective inertia), so
alpha is solved exactly at every RK4 stage. Feedforward torque = the rung-2 moment of her own motion.
Checks: the round trip returns her trunk angle; open loop, a 0.1 deg start error grows (an inverted
pendulum) and its e-folding time is the controller's requirement.

Rung 4. Hip torque = feedforward + s (-Kp e(t - d) - Kd e'(t - d)), e the trunk-angle error, d the sensory
delay; gains from the plant, not tuned: Kp = wn^2 + 1/tau^2 with tau the open-loop e-folding time,
Kd = 2 zeta wn, zeta = 0.7. Delays 0, 45, 75, 100, 117 ms (SOURCES, "Sensory and reflex delays"), wn 5 and
10 rad/s; disturbances a 0.1 deg start offset and a 10 N.m torque bias; one stroke, as the record.

Rung 5's linear check. e'' = lam^2 e - Kp e(t - d) - Kd e'(t - d), lam = 1/tau: the rightmost root of its
characteristic equation s^2 - lam^2 + (Kp + Kd s) exp(-s d) = 0 gives the growth rate and the delay at which
the loop loses stability.

Writes data/local/cr06_study/trunk_forward_cr06.json.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import fsolve

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import her_stroke as H                                         # noqa: E402
import hip_topdown_cr06 as HT                                  # noqa: E402

ZETA = 0.7
DELAYS_MS = (0, 45, 75, 100, 117)
RECORDED = dict(
    rung3=dict(round_trip_deg=0.0028, open_loop_end_deg=72.0, growth=720.0, tau_s=0.29, eff_inertia=(8.8, 8.9)),
    rung4={(5, 0): (0.10, 0.000, 0.6, "3.8-3.9 / 3.3-3.5, 22-23"), (5, 45): (0.10, 0.000, 0.6, "3.8-3.9 / 3.3-3.5, 22-23"),
           (5, 75): (0.10, 0.000, 0.6, "3.8-3.9 / 3.3-3.5, 22-23"), (5, 100): (0.10, 0.002, None, "4.0 / 3.2, 25"),
           (5, 117): (0.10, 0.013, None, "4.6 / 3.0, 30"), (10, 0): (0.10, 0.000, 1.7, "0.7 / 0.7, 13-19"),
           (10, 45): (0.10, 0.000, 1.7, "0.7 / 0.7, 13-19"), (10, 75): (0.13, 0.13, None, "1.5 / 1.5, 40 (edge)"),
           (10, 100): (8.0, 6.3, 235, "56 / 41, 1640 (fails)"), (10, 117): (60, None, 1010, "91, diverged (fails)")},
    linear={5: dict(growth={0: -3.5, 100: -1.75, 117: -0.70}, boundary_ms=132), 10: dict(growth={45: -6.8, 75: 0.45}, boundary_ms=72)})


class Plant:
    """The hip balance at any instant: prescribed inputs interpolated periodically from the grid."""

    def __init__(self, S, hip):
        self.S, self.hip = S, hip
        self.body = HT.upper_body()
        hx, self.hz, hxdd = hip
        # smooth inputs, evaluated at any instant: the open loop amplifies an error ~1000x per stroke, so linear
        # interpolation between grid points (kinks in the acceleration) is enough to spoil the round trip
        tv, v = H.series("boat_velocity_mps")
        self._ab = H.periodic_spline(tv, v, S.lam_v).derivative(1)
        self._hxdd = S.legS.derivative(2)                       # the hip follows her leg channel
        self.F = S.F_handle * np.cos(np.radians(S.oar_deg))

    def m0_s(self, tq, a, ad):
        """(M0, s): extension moment with zero angular acceleration, and d(moment)/d(alpha)."""
        tm = tq % H.T
        ab, hxdd = float(self._ab(tm)), float(self._hxdd(tm))
        F = float(np.interp(tm, self.S.t, self.F, period=H.T))   # enters M0 only additively: cancels along her path
        M0 = (HT.Z_HAND - self.hz) * F                          # handle: r_hand x (-F, 0) = rz F
        s = 0.0
        for _, m, d, I in self.body:
            rx, rz = d * np.cos(a), d * np.sin(a)
            ax0 = ab + hxdd - d * np.cos(a) * ad ** 2           # the alpha-free parts of the acceleration
            az0 = -d * np.sin(a) * ad ** 2
            M0 += rx * (-m * H.G)                               # gravity: r x m g
            M0 -= rx * m * az0 - rz * m * ax0                    # inertia, alpha-free
            # the alpha part: a = d(-sin a, cos a) alpha  ->  -(r x m a) = -m d^2 alpha; spin -I alpha
            s -= m * d ** 2 + I
        return M0, s


N_HARMONICS = 15


def fourier(S, n=N_HARMONICS):
    """Her back channel as a truncated Fourier series of the smoothed spline: periodic and smooth in every
    derivative. A cubic spline's third derivative jumps at each knot, and the open-loop trunk amplifies the
    integrator's error at those jumps ~1000x per stroke; the series removes that. Returns (f, f', f'', rms
    difference from the spline in m)."""
    tt = np.linspace(0, H.T, 2048, endpoint=False)
    y = S.backS(tt)
    c = np.fft.rfft(y) / len(tt)
    w = 2 * np.pi / H.T
    k = np.arange(n + 1)
    def ev(tq, d=0):
        tq = np.asarray(tq, float)
        ph = np.outer(np.atleast_1d(tq), k) * w
        fac = (1j * k * w) ** d
        z = (c[:n + 1] * fac * np.exp(1j * ph))
        val = z[:, 0].real + 2 * z[:, 1:].real.sum(1)
        return val if val.size > 1 else float(val[0])
    rms = float(np.sqrt(np.mean((ev(tt) - y) ** 2)))
    return ev, rms


def her_angle(S, x_catch):
    """Her trunk angle and its rates from the Fourier series of her back channel (analytic)."""
    k0 = int(np.argmin(S.leg))
    ev, _ = fourier(S)
    b0 = ev(S.t[k0])
    def ang(tq, n=0):
        x = x_catch + ev(tq) - b0
        a = np.arccos(np.clip(x / HT.L_TRUNK, -1, 1))
        if n == 0:
            return a
        xd, xdd = ev(tq, 1), ev(tq, 2)
        ad = -xd / (HT.L_TRUNK * np.sin(a))
        if n == 1:
            return ad
        return (-xdd - HT.L_TRUNK * np.cos(a) * ad ** 2) / (HT.L_TRUNK * np.sin(a))
    return ang


def run(P, ang, tau_ff, n_steps, a0_offset_deg=0.0, bias=0.0, wn=None, delay_ms=0.0, tau=0.29):
    """RK4 over one stroke from t = 0. Returns (max |error| deg, end error deg, peak |feedback| N.m, errors)."""
    dt = H.T / n_steps
    Kp = (wn ** 2 + 1 / tau ** 2) if wn else 0.0
    Kd = 2 * ZETA * wn if wn else 0.0
    d = delay_ms / 1000.0
    hist_t, hist_e, hist_ed = [], [], []

    def delayed(tq):
        if not hist_t or d == 0:
            return None
        tt = tq - d
        if tt <= hist_t[0]:
            return hist_e[0], hist_ed[0]
        return float(np.interp(tt, hist_t, hist_e)), float(np.interp(tt, hist_t, hist_ed))

    fb_peak = [0.0]

    def f(tq, y):
        a, ad = y
        e, ed = a - ang(tq), ad - ang(tq, 1)
        M0, s = P.m0_s(tq, a, ad)
        tq_torque = tau_ff(tq) + bias
        fb = 0.0
        if wn:
            if d == 0:
                fb = s * (-Kp * e - Kd * ed)
            else:
                de = delayed(tq)
                fb = 0.0 if de is None else s * (-Kp * de[0] - Kd * de[1])
            fb_peak[0] = max(fb_peak[0], abs(fb))
        alpha = (tq_torque + fb - M0) / s
        return np.array([ad, alpha])

    y = np.array([ang(0.0) + np.radians(a0_offset_deg), ang(0.0, 1)])
    errs = []
    for k in range(n_steps):
        tq = k * dt
        e = y[0] - ang(tq); errs.append(e)
        hist_t.append(tq); hist_e.append(e); hist_ed.append(y[1] - ang(tq, 1))
        k1 = f(tq, y); k2 = f(tq + dt / 2, y + dt / 2 * k1); k3 = f(tq + dt / 2, y + dt / 2 * k2); k4 = f(tq + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(y)) or abs(y[0]) > 10:
            break
    e_end = y[0] - ang(H.T)
    errs.append(e_end)
    errs = np.degrees(np.array(errs))
    return float(np.max(np.abs(errs))), float(errs[-1]), fb_peak[0], errs


def dde_root(lam, Kp, Kd, delay):
    """Rightmost root of s^2 - lam^2 + (Kp + Kd s) exp(-s delay) = 0, from a grid of starting points."""
    best = None
    for re0 in np.linspace(-15, 8, 12):
        for im0 in np.linspace(0, 40, 21):
            def F(v):
                z = complex(v[0], v[1])
                r = z ** 2 - lam ** 2 + (Kp + Kd * z) * np.exp(-z * delay)
                return [r.real, r.imag]
            sol, info, ier, _ = fsolve(F, [re0, im0], full_output=True)
            if ier == 1 and abs(complex(*F(sol))) < 1e-8 and (best is None or sol[0] > best[0] + 1e-9):
                best = sol
    return float(best[0]) if best is not None else float("nan")


def main():
    S = H.Stroke()
    x_catch, hip = HT.setup(S)
    ang = her_angle(S, x_catch)
    P = Plant(S, hip)
    # feedforward: the rung-2 moment of her own motion, M_ext = M0 + s alpha along her path
    def tau_ff(tq):
        M0, s = P.m0_s(tq, ang(tq), ang(tq, 1))
        return M0 + s * ang(tq, 2)
    s_range = [P.m0_s(tq, ang(tq), ang(tq, 1))[1] for tq in np.linspace(0, H.T, 50, endpoint=False)]
    out = dict(eff_inertia=[-max(s_range), -min(s_range)], rung3={}, rung4={}, linear={})
    print("rung 3: effective inertia %.2f to %.2f kg.m^2 (recorded 8.8-8.9; rung 4's slope 8.67-9.15)" % tuple(out["eff_inertia"]))
    for n in (1000, 2000):
        mx, end, _, _ = run(P, ang, tau_ff, n)
        out["rung3"]["round_trip_%d" % n] = mx
        print("  round trip, RK4 %d steps: max trunk-angle error %.4f deg (recorded 0.0028)" % (n, mx))
    mx, end, _, errs = run(P, ang, tau_ff, 1000, a0_offset_deg=0.1)
    growth = abs(end) / 0.1
    tau = H.T / np.log(growth) if growth > 1 else float("nan")
    out["rung3"].update(open_loop_end_deg=end, growth=growth, tau_s=tau)
    print("  open loop from a 0.1 deg offset: %.1f deg after one stroke, growth %.0fx, e-folding %.3f s (recorded 72 deg, 720x, 0.29 s)" % (end, growth, tau))
    tau_use = tau if np.isfinite(tau) else 0.29
    print("rung 4 (gains from tau = %.3f s, zeta 0.7): max / end error, peak feedback" % tau_use)
    print("  %-8s %-6s %-30s %-34s %s" % ("wn", "delay", "0.1 deg offset", "10 N.m bias", "recorded (offset; bias)"))
    for wn in (20, 5, 10):
        for dly in ((0,) if wn == 20 else DELAYS_MS):
            a1 = run(P, ang, tau_ff, 1000, a0_offset_deg=0.1, wn=wn, delay_ms=dly, tau=tau_use)
            a2 = run(P, ang, tau_ff, 1000, bias=10.0, wn=wn, delay_ms=dly, tau=tau_use)
            out["rung4"]["%d/%d" % (wn, dly)] = dict(offset=a1[:3], bias=a2[:3])
            rec = RECORDED["rung4"].get((wn, dly))
            print("  %-8s %-6s %-30s %-34s %s" % ("%d rad/s" % wn, "%d ms" % dly, "%.2f / %.3f deg, %.1f N.m" % a1[:3], "%.2f / %.2f deg, %.1f N.m (%.0f%%)" % (a2[0], a2[1], a2[2], 100 * a2[2] / 267),
                  ("%s / %s, %s; %s" % rec) if rec else ("sanity: decays to 0, 6.4 N.m" if wn == 20 else "")))
    lam = 1 / tau_use
    print("rung 5's linear check: e'' = lam^2 e - Kp e(t-d) - Kd e'(t-d), lam = 1/%.3f s" % tau_use)
    for wn in (5, 10):
        Kp, Kd = wn ** 2 + lam ** 2, 2 * ZETA * wn
        rates = {dly: dde_root(lam, Kp, Kd, dly / 1000.0) for dly in (0, 45, 75, 100, 117, 132)}
        lo, hi = 0.0, 0.3
        for _ in range(40):                                     # the delay where the rightmost root crosses zero
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if dde_root(lam, Kp, Kd, mid) < 0 else (lo, mid)
        out["linear"][str(wn)] = dict(growth={str(k): v for k, v in rates.items()}, boundary_ms=1000 * lo)
        print("  %2d rad/s: growth rate (1/s) %s; stability boundary %.0f ms (recorded %s ms; growth %s)"
              % (wn, ", ".join("%d ms %+.2f" % kv for kv in rates.items()), 1000 * lo, RECORDED["linear"][wn]["boundary_ms"], RECORDED["linear"][wn]["growth"]))
    json.dump(out, open(os.path.join(H.OUT, "trunk_forward_cr06.json"), "w"), indent=1, default=float)
    print("written", os.path.join(H.OUT, "trunk_forward_cr06.json"))


if __name__ == "__main__":
    main()
