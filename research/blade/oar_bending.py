"""Oar-shaft bending as a quasi-static hinge, solved against the blade's own load. Tested on [RLBQV19].

    .venv/Scripts/python.exe research/blade/oar_bending.py

[RLBQV19]'s bench law (sec. 2.3.1, eq. 1): the blade's linear and angular deflections are proportional to the
bending moment at the oarlock, f = M/K_lin and phi = M/K_ang (sweep oar: K_lin = 3424.1 N, K_ang = 2948 N m). So
f / phi = K_ang / K_lin = 0.861 m whatever the load: the blade moves as if the shaft were rigid with a hinge
h = L_e - K_ang/K_lin from the pin (0.697 m on their oar) and a rotational spring that turns phi = M/K_ang.

Quasi-static, as they treat it (the oar's inertia is small against the fluid load): at every instant

    K_ang phi = M(phi, phi_dot) = sum over strips of r cos(theta_1) F_n(w_n(r)),
    w_n(r) = r cos(theta_1) psi_dot - (r - h) phi_dot - V_b sin(psi - phi),

so the water is the damper and the shaft the spring: given phi, phi_dot is the rate whose slip makes the load's
moment equal the spring's. Solved per step by bracketing; phi integrated forward (Heun). Nothing is taken from the
measured load: the test is whether the predicted bending, and the forces with it, match what [RLBQV19] measured,
and what the open-loop run (bending from their measured M_z; robert2019_quasisteady.py) gave.

Outputs: data/local/blade/oar_bending.json and .png.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import robert2019_quasisteady as RQ                                # noqa: E402
from coxswain.crew.liftdrag import LiftDragBlade                  # noqa: E402

OUT = RQ.OUT
HINGE = RQ.L_E - RQ.K_ANG / RQ.K_LIN                              # 0.697 m from the pin


def moment(psi, rate, phi, phi_dot, law):
    """(M about the pin as printed (negative while driving), F_x, F_y) for this bending state."""
    ds = RQ.SPAN / RQ.STRIPS
    r = RQ.L_E + (np.arange(RQ.STRIPS) + 0.5) * ds
    dA = RQ.AREA / RQ.STRIPS
    pb = psi - phi
    fn = ft = mz = 0.0
    for ri in r:
        wn = ri * np.cos(RQ.THETA1) * rate - (ri - HINGE) * phi_dot - RQ.VB * np.sin(pb)
        wa = RQ.VB * np.cos(pb)
        a, b = law(wn, wa, dA)
        fn += a; ft += b; mz += ri * np.cos(RQ.THETA1) * a
    return mz, -fn * np.sin(pb) + ft * np.cos(pb), fn * np.cos(pb) + ft * np.sin(pb)


def phi_rate(psi, rate, phi, law):
    """The phi_dot at which the load's moment equals the spring's: K_ang phi = -M (M printed negative)."""
    g = lambda pd: -moment(psi, rate, phi, pd, law)[0] - RQ.K_ANG * phi   # noqa: E731
    lo, hi = -40.0, 40.0
    glo, ghi = g(lo), g(hi)
    if glo * ghi > 0:                                              # no balance in range: the closer end
        return lo if abs(glo) < abs(ghi) else hi
    return brentq(g, lo, hi, xtol=1e-6)


def solve(K, law):
    t, psi, rate = K["t"], K["psi"], K["rate"]
    phi = np.zeros_like(t)
    for i in range(t.size - 1):
        dt = t[i + 1] - t[i]
        k1 = phi_rate(psi[i], rate[i], phi[i], law)
        k2 = phi_rate(psi[i + 1], rate[i + 1], phi[i] + dt * k1, law)
        phi[i + 1] = phi[i] + 0.5 * dt * (k1 + k2)
    phid = np.gradient(phi, t)
    F = np.array([moment(psi[i], rate[i], phi[i], phid[i], law) for i in range(t.size)])
    return phi, np.stack([F[:, 1], F[:, 2], F[:, 0]], 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    D = RQ.load()
    K = RQ.kinematics(D, rigid=True, n=400)                        # the motor's motion; bending is solved here
    meas = np.stack([np.interp(K["psi_deg"], *D[k].T) for k in ("Fx_N", "Fy_N", "Mz_Nm")], 1)
    phi_meas = np.abs(meas[:, 2]) / RQ.K_ANG
    laws = {
        "slip C0 1.3 ([CR06] computed)": RQ.slip_law(1.3),
        "lift-drag Coppel": RQ.liftdrag_law(LiftDragBlade.big_blade_full_scale(outboard=1.818, area=RQ.AREA,
                                                                              source="coppel")),
        "lift-drag CG07": RQ.liftdrag_law(LiftDragBlade.big_blade(outboard=1.818, area=RQ.AREA)),
    }
    res, traces = {}, {}
    print("hinge %.3f m from the pin; measured bending (their law on the measured M_z) peaks at %.2f deg"
          % (HINGE, np.degrees(phi_meas.max())))
    print("\n%-34s predicted bending peak  F_x peak @deg   F_x rms  M_z rms   (open loop: F_x rms)" % "law")
    open_loop = RQ.score(K, RQ.loads(RQ.kinematics(D, rigid=False, n=400), RQ.slip_law(1.3)), meas)
    for name, law in laws.items():
        phi, F = solve(K, law)
        sc = RQ.score(K, F, meas)
        ol = RQ.score(K, RQ.loads(RQ.kinematics(D, rigid=False, n=400), law), meas)
        res[name] = dict(score=sc, phi_peak_deg=float(np.degrees(phi.max())),
                         phi_rms_deg=float(np.degrees(np.sqrt(np.mean((phi - phi_meas) ** 2)))), open_loop=ol)
        traces[name] = (phi, F)
        print("%-34s %6.2f deg (rms %4.2f)   %4.0f @%3.0f     %5.1f   %6.1f        %5.1f" % (
            name, np.degrees(phi.max()), res[name]["phi_rms_deg"], sc["Fx"]["peak"], sc["Fx"]["peak_at"],
            sc["Fx"]["rms"], sc["Mz"]["rms"], ol["Fx"]["rms"]))
    json.dump(dict(hinge_m=HINGE, results=res), open(os.path.join(OUT, "oar_bending.json"), "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        ax[0].plot(K["psi_deg"], np.degrees(phi_meas), "k", lw=2, label="from measured M_z (their law)")
        ax[1].plot(K["psi_deg"], meas[:, 0], "k", lw=2, label="measured")
        ax[2].plot(K["psi_deg"], meas[:, 2], "k", lw=2, label="measured")
        for name, (phi, F) in traces.items():
            ax[0].plot(K["psi_deg"], np.degrees(phi), lw=1, label=name)
            ax[1].plot(K["psi_deg"], F[:, 0], lw=1, label=name)
            ax[2].plot(K["psi_deg"], F[:, 2], lw=1, label=name)
        for a, lab in zip(ax, ("bending phi (deg)", "F_x (N)", "M_z (N m)")):
            a.set_xlabel("sweep angle psi_0 (deg)"); a.set_ylabel(lab); a.grid(alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.savefig(os.path.join(OUT, "oar_bending.png"), dpi=130)
    except Exception as e:                                         # the figure is a convenience
        print("figure skipped: %s" % e)


if __name__ == "__main__":
    main()
