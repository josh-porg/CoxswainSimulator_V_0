"""Every project blade law on the Nantes towing tank's flat blade ([LJH12]; the Barre & Kobus 2010 benchmark).

    .venv/Scripts/python.exe research/blade/leroyer_jh12_quasisteady.py

A third measured load on a known motion, independent of [GTW21] and [RLBQV19]: a flat plate (0.302 x 0.139 m, AR
0.46, 420 cm^2) on a rigid shaft, top edge at the surface, carriage V_b = 2.91 m/s, theta_dot = K sin(theta) with K =
3 rad/s, eta_0 = 0.742, so the arm to the blade centre is L_e = V_b / (K eta_0) = 1.307 m (data/literature/
leroyer_jh12_flat_blade.csv). No bending (rigid shaft); every law's pieces fixed before this test.

The motion in the blade's frame (n normal, t along the shaft): a point at radius r slips through the water at
w_n = r theta_dot - V_b sin(theta) and w_t = V_b cos(theta). The paper reduces the measured force with the velocity at
the centre I: C = F / (1/2 rho S_b |V_I|^2), lift and drag about -V_I, against the pseudo-incidence i (the angle
between -V_I and the shaft). Each law is integrated in strips and reduced the same way; F_x, F_y in the carriage
frame. Signs: one convention for every law, fixed by [CG07]'s correlation with the measurement (printed).

Outputs: data/local/blade/leroyer_jh12_quasisteady.json and .png.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import helmbold                                                   # noqa: E402
from coxswain.crew.liftdrag import LiftDragBlade                  # noqa: E402

LIT = os.path.join(ROOT, "data", "literature", "leroyer_jh12_flat_blade.csv")
OUT = os.path.join(ROOT, "data", "local", "blade")
RHO, VB, K, ETA0 = 1000.0, 2.91, 3.0, 0.742
LE = VB / (K * ETA0)
AREA = 0.042
HEIGHT = np.sqrt(AREA * 0.46)                                     # AR = height / length = 0.46
SPAN = AREA / HEIGHT
STRIPS = 24
WINDOWS = ((0, 45), (45, 90), (90, 135), (135, 180))


def load():
    rows = {}
    with open(LIT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("series"):
                continue
            s, _v, a, b = line.strip().split(",")
            rows.setdefault(s, []).append((float(a), float(b)))
    return {k: np.array(v) for k, v in rows.items()}


def slip_law(c0):
    return lambda wn, wt, dA: (-np.sign(wn) * 0.5 * RHO * dA * c0 * wn * wn, 0.0)


def liftdrag_law(blade):
    def law(wn, wt, dA):
        q = 0.5 * RHO * dA * (wn * wn + wt * wt)
        if q <= 0.0:
            return 0.0, 0.0
        c_n, c_t = blade.coefficients(np.arctan2(abs(wn), abs(wt)))
        return -np.sign(wn) * q * c_n, -np.sign(wt) * q * c_t
    return law


def run(law, theta):
    """Per theta: (F_n, F_t) on the blade (water's force), centre (w_n, w_t)."""
    rate = K * np.sin(theta)
    r = LE + (np.arange(STRIPS) + 0.5 - 0.5 * STRIPS) * SPAN / STRIPS
    dA = AREA / STRIPS
    out = np.zeros((theta.size, 4))
    for i, (th, om) in enumerate(zip(theta, rate)):
        fn = ft = 0.0
        for ri in r:
            a, b = law(ri * om - VB * np.sin(th), VB * np.cos(th), dA)
            fn += a; ft += b
        out[i] = (fn, ft, LE * om - VB * np.sin(th), VB * np.cos(th))
    return out


def reduce(R, theta):
    """C_l, C_d about the centre's relative flow, the pseudo-incidence i, and F_x, F_y in the carriage frame."""
    fn, ft, wn, wt = R.T
    u = -np.stack([wt, wn], 1)                                    # water relative to the blade, (t, n)
    q = 0.5 * RHO * AREA * (wn ** 2 + wt ** 2)
    uh = u / np.linalg.norm(u, axis=1)[:, None]
    F = np.stack([ft, fn], 1)
    cd = (F * uh).sum(1) / q
    cl = (F * np.stack([-uh[:, 1], uh[:, 0]], 1)).sum(1) / q
    inc = np.degrees(np.arccos(np.clip(uh[:, 0], -1, 1)))         # angle between -V_I and the shaft (t)
    fx = ft * np.cos(theta) - fn * np.sin(theta)
    fy = ft * np.sin(theta) + fn * np.cos(theta)
    return cl, cd, inc, fx, fy


def windows(x, model, meas):
    return [float(np.mean(model[(x >= a) & (x < b)] - meas[(x >= a) & (x < b)]) / np.max(np.abs(meas)))
            for a, b in WINDOWS]


def main():
    os.makedirs(OUT, exist_ok=True)
    D = load()
    theta = np.radians(np.linspace(1.0, 179.0, 357))
    laws = {
        "slip C0 1.3 (Hoerner, [CR06])": slip_law(1.3),
        "slip, research level (C0 3.12)": slip_law(140.88 / (0.5 * RHO * 0.0903)),
        "lift-drag CG07": liftdrag_law(LiftDragBlade.big_blade(outboard=LE, area=AREA)),
        "lift-drag Coppel": liftdrag_law(LiftDragBlade.big_blade_full_scale(outboard=LE, area=AREA, source="coppel")),
        "cross-flow (fins' law), C_Dc 1.3": liftdrag_law(helmbold.crossflow(HEIGHT, SPAN, AREA, False, 1.3)),
        "cross-flow (fins' law), C_Dc 2.05": liftdrag_law(helmbold.crossflow(HEIGHT, SPAN, AREA, False, 2.05)),
    }
    ref = None
    res, traces = {}, {}
    td = np.degrees(theta)
    for name, law in laws.items():
        cl, cd, inc, fx, fy = reduce(run(law, theta), theta)
        if ref is None:                                            # one sign convention, from CG07 (below)
            ref = name
        traces[name] = (cl, cd, inc, fx, fy)
    # sign convention: fixed once from CG07 against the measurement, applied to every law
    cl, cd, inc, fx, fy = traces["lift-drag CG07"]
    o = np.argsort(inc)
    s_cl = np.sign(np.corrcoef(cl[o], np.interp(inc[o], *D["Cl"].T))[0, 1])
    s_fx = np.sign(np.corrcoef(fx, np.interp(td, *D["Fx_N"].T))[0, 1])
    s_fy = np.sign(np.corrcoef(fy, np.interp(td, *D["Fy_N"].T))[0, 1])
    print("[LJH12] flat blade: L_e %.3f m, blade %.3f x %.3f m; measured C_d max %.2f, C_l %.2f / %.2f, F_x min %.1f N; "
          "signs from CG07: C_l %+d, F_x %+d, F_y %+d"
          % (LE, SPAN, HEIGHT, D["Cd"][:, 1].max(), D["Cl"][:, 1].max(), D["Cl"][:, 1].min(), D["Fx_N"][:, 1].min(),
             s_cl, s_fx, s_fy))
    print("\n%-38s C_d err/peak by i 0-45/45-90/90-135/135-180   C_l rms  C_d rms  |  F_x rms N  F_y rms N  F_x impulse"
          % "law")
    for name, (cl, cd, inc, fx, fy) in traces.items():
        cl, fx, fy = s_cl * cl, s_fx * fx, s_fy * fy
        o = np.argsort(inc)
        mcl, mcd = np.interp(inc[o], *D["Cl"].T), np.interp(inc[o], *D["Cd"].T)
        mfx, mfy = np.interp(td, *D["Fx_N"].T), np.interp(td, *D["Fy_N"].T)
        w = windows(inc[o], cd[o], mcd)
        r = dict(cd_windows=w, cl_rms=float(np.sqrt(np.mean((cl[o] - mcl) ** 2))),
                 cd_rms=float(np.sqrt(np.mean((cd[o] - mcd) ** 2))), fx_rms=float(np.sqrt(np.mean((fx - mfx) ** 2))),
                 fy_rms=float(np.sqrt(np.mean((fy - mfy) ** 2))), fx_impulse=float(np.trapezoid(fx, td) / np.trapezoid(mfx, td)),
                 cd_max=float(cd.max()), cl_max=float(cl.max()))
        res[name] = r
        traces[name] = (cl, cd, inc, fx, fy)
        print("%-38s %s       %5.2f    %5.2f   |   %5.1f      %5.1f      %5.2f" % (
            name, " ".join("%+5.2f" % v for v in w), r["cl_rms"], r["cd_rms"], r["fx_rms"], r["fy_rms"], r["fx_impulse"]))
    json.dump(dict(results=res, L_e=LE, span=SPAN, height=HEIGHT), open(os.path.join(OUT, "leroyer_jh12_quasisteady.json"), "w"),
              indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for a, key, xv in ((ax[0, 0], "Cl", "i"), (ax[0, 1], "Cd", "i"), (ax[1, 0], "Fy_N", "t"), (ax[1, 1], "Fx_N", "t")):
            a.plot(*D[key].T, "k", lw=2, label="measured [LJH12]")
            for name, (cl, cd, inc, fx, fy) in traces.items():
                y = dict(Cl=cl, Cd=cd, Fy_N=fy, Fx_N=fx)[key]
                x = inc if xv == "i" else td
                o = np.argsort(x)
                a.plot(x[o], y[o], lw=1, label=name)
            a.set_xlabel("pseudo-incidence i (deg)" if xv == "i" else "theta (deg)"); a.set_ylabel(key); a.grid(alpha=0.3)
        ax[0, 0].legend(fontsize=6)
        fig.savefig(os.path.join(OUT, "leroyer_jh12_quasisteady.png"), dpi=120)
    except Exception as e:                                         # the figure is a convenience
        print("figure skipped: %s" % e)


if __name__ == "__main__":
    main()
