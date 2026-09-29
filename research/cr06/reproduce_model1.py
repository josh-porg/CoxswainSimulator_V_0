"""[CR06]'s Model 1, as printed, driven by their own measured stroke: does it give their result?

    python research/cr06/reproduce_model1.py [--harmonics 12]

Their trial (c) (Table 2, text p. 895): with the rower's leg, back and oar-angle motion fitted to
the measurements, "the simulated average speed is less than the measured average speed (3.83 m/s
compared to 4.18 m/s)". Here the same model is driven by the measured curves themselves
(data/literature/cr06_fig3_measured.csv, re-extracted from their Fig. 3).

The model, summing their eqs. (1)-(3) with (5)-(7) for a sculler ("multiplying the oar force
and mass by 2"), is one ODE for the boat velocity v once the body and oar are prescribed:

    (m_R + m_b + 2 m_O) dv/dt = -C1 v^2 + 2 F_oar cos(phi)
                                - m_R (x''_B/F + r x''_S/B) - 2 m_O d (phi'' cos phi - phi'^2 sin phi)
    F_oar = C2 (l phi' + v cos phi)^2 while that normal velocity is driving (< 0), else 0 (eq. 16)

Every constant is their Table 2.7 (singles). The only choice is how the digitised curves are
differentiated: a periodic Fourier fit with N harmonics, swept (their own fits used 16-knot
splines). Nothing is fitted to the boat.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "literature", "cr06_fig3_measured.csv")

# [CR06] Table 2.7, singles
T = 1.94
M_R, M_B, M_O = 75.0, 19.7, 1.2
S, L = 0.83, 1.805          # modified inboard and outboard (force point, blade centre)
D = 0.565                   # oarlock to oar centre of mass
R = 0.4                     # rower CoM height over shoulder height
C1 = 3.16                   # boat drag, N/(m/s)^2
C2 = 58.7                   # oar blade, N/(m/s)^2 (Hoerner's C = 1.3, top edge just below the surface)

PRINTED = {"trial_c_speed": 3.83, "measured_speed": 4.18}


class Fourier:
    """Periodic least-squares Fourier fit, differentiable exactly."""

    def __init__(self, t, y, harmonics):
        w = 2 * np.pi / T
        k = np.arange(1, harmonics + 1)
        A = np.column_stack([np.ones_like(t)] + [f(kk * w * t) for kk in k for f in (np.cos, np.sin)])
        self.c, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.w, self.k = w, k

    def __call__(self, t, nu=0):
        t = np.asarray(t, dtype=float)
        out = np.full_like(t, self.c[0] if nu == 0 else 0.0)
        for i, kk in enumerate(self.k):
            a, b = self.c[1 + 2 * i], self.c[2 + 2 * i]
            wk = kk * self.w
            # derivative nu of a cos(wk t) + b sin(wk t)
            for _ in range(nu):
                a, b = b * wk, -a * wk
            out = out + a * np.cos(wk * t) + b * np.sin(wk * t)
        return out


def series(name):
    rows = [r for r in csv.DictReader(l for l in open(DATA, encoding="utf-8") if not l.startswith("#"))
            if r["series"] == name]
    t = np.mod(np.array([float(r["t_s"]) for r in rows]), T)
    return t, np.array([float(r["value"]) for r in rows])


def run(harmonics=12, c2=C2, periods=30):
    phi = Fourier(*(lambda t, y: (t, np.radians(y)))(*series("oar_angle_deg")), harmonics)
    leg = Fourier(*series("leg_disp_m"), harmonics)
    back = Fourier(*series("back_disp_m"), harmonics)
    mass = M_R + M_B + 2 * M_O

    def rhs(t, y):
        v = y[0]
        p, pd, pdd = float(phi(t)), float(phi(t, 1)), float(phi(t, 2))
        normal = L * pd + v * np.cos(p)
        f_oar = c2 * normal ** 2 if normal < 0.0 else 0.0
        body = M_R * (float(leg(t, 2)) + R * float(back(t, 2)))
        oar = 2 * M_O * D * (pdd * np.cos(p) - pd ** 2 * np.sin(p))
        return [(-C1 * v * abs(v) + 2 * f_oar * np.cos(p) - body - oar) / mass]

    t_end = periods * T
    sol = solve_ivp(rhs, (0.0, t_end), [4.0], max_step=T / 400, rtol=1e-8, atol=1e-10, dense_output=True)
    tt = np.linspace(t_end - T, t_end, 2000, endpoint=False)
    v = sol.sol(tt)[0]
    tv, vm = series("boat_velocity_mps")
    return dict(speed=float(v.mean()), v_min=float(v.min()), v_max=float(v.max()),
                measured_speed=float(np.mean(Fourier(tv, vm, harmonics)(np.linspace(0, T, 2000)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmonics", default="8,12,16")
    a = ap.parse_args()
    print("[CR06] Model 1, their Table 2.7 constants, driven by their measured stroke")
    print("printed: trial (c) %.2f m/s against measured %.2f" % (PRINTED["trial_c_speed"], PRINTED["measured_speed"]))
    for n in (int(x) for x in a.harmonics.split(",")):
        for scale in (1.0, 2.4):
            r = run(n, C2 * scale)
            print("  %2d harmonics, C2 x%.1f: mean %.3f m/s (%.2f-%.2f); measured trace mean %.3f"
                  % (n, scale, r["speed"], r["v_min"], r["v_max"], r["measured_speed"]))


if __name__ == "__main__":
    main()
