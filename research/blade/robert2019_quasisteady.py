"""Every project blade law on Robert et al. (2019)'s full-size blade, its motion known and its oar bending.

    .venv/Scripts/python.exe research/blade/robert2019_quasisteady.py

The cross-check to [GTW21] (research/blade/grift2021_quasisteady.py; SOURCES §184): a different motion (a towing
carriage at a constant 4.202 m/s, the oar swept by a motor, Fig. 3's angular velocity against angle), a full-size
hatchet blade on a real, flexible oar, and measured F_x, F_y and M_z (data/literature/robert2019_trial154.csv).

**The oar bends** ([RLBQV19] sec. 2.3.1, their "simplified model", eq. 1): the blade's linear and angular deflections
are proportional to the bending moment at the oarlock, f = M/K_lin and phi = M/K_ang, K_lin = 3424.1 N and
K_ang = 2948 N m, measured on a bench. The measured M_z (about the rotation axis, at the oarlock) gives the bending
directly, so the blade's motion is known open loop: it lags the motor by phi (up to 7.5 deg here) and f (0.11 m),
and its velocity through the water loses f_dot + (r - L_e) phi_dot while the load builds and gains it back as the
load falls. ``--rigid`` drops it, to show what bending is worth.

Geometry (horizontal plane, the oar inclined theta_1 = 9.5 deg, so radii project by cos theta_1): oarlock to blade
root L_e = 1.558 m; the project's sweep blade, 0.52 m along the shaft, 0.110 m^2 (centre 1.818 m against the paper's
reference point 1.815 m). A blade point at radius r moves through the water, normal to the blade and along it:

    w_n = r cos(theta_1) psi_dot - f_dot - (r - L_e) phi_dot - V_b sin(psi_b),    w_a = V_b cos(psi_b),

psi_b = psi - phi. Loads are integrated in strips; F_x = -F_norm sin psi_b + F_t cos psi_b,
F_y = F_norm cos psi_b + F_t sin psi_b, M_z = sum r cos(theta_1) F_norm (signs as the paper prints them).

Outputs: data/local/blade/robert2019_quasisteady.json and .png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from coxswain.crew.blade_added_mass import labbe_added_mass, patton_added_mass   # noqa: E402
from coxswain.crew.liftdrag import LiftDragBlade                  # noqa: E402

LIT = os.path.join(ROOT, "data", "literature", "robert2019_trial154.csv")
OUT = os.path.join(ROOT, "data", "local", "blade")
RHO = 1000.0
VB = 4.202
THETA1 = np.radians(9.5)
L_E = 1.558
SPAN, AREA, HEIGHT = 0.52, 0.110, 0.25
K_LIN, K_ANG = 3424.1, 2948.0
STRIPS = 26
WINDOWS = ((20, 50), (50, 80), (80, 110), (110, 150))


def load():
    rows = {}
    with open(LIT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("series"):
                continue
            s, a, b = line.strip().split(",")
            rows.setdefault(s, []).append((float(a), float(b)))
    return {k: np.array(v) for k, v in rows.items()}


def kinematics(D, rigid, n=600):
    psi_deg = np.linspace(20.0, 150.0, n)
    psi = np.radians(psi_deg)
    rate = np.interp(psi_deg, *D["psi0dot_radps"].T)
    t = np.concatenate([[0.0], np.cumsum(np.diff(psi) / (0.5 * (rate[1:] + rate[:-1])))])
    m = np.abs(np.interp(psi_deg, *D["Mz_Nm"].T))
    if rigid:
        m = np.zeros_like(m)
    phi, f = m / K_ANG, m / K_LIN
    return dict(psi_deg=psi_deg, psi=psi, rate=rate, t=t, phi=phi, f=f,
                phi_dot=np.gradient(phi, t), f_dot=np.gradient(f, t))


def slip_law(c0):
    return lambda wn, wa, dA: (-np.sign(wn) * 0.5 * RHO * dA * c0 * wn * wn, 0.0)


def liftdrag_law(blade):
    def law(wn, wa, dA):
        q = 0.5 * RHO * dA * (wn * wn + wa * wa)
        if q <= 0.0:
            return 0.0, 0.0
        c_n, c_t = blade.coefficients(np.arctan2(abs(wn), abs(wa)))
        return -np.sign(wn) * q * c_n, -np.sign(wa) * q * c_t
    return law


def loads(K, law):
    ds = SPAN / STRIPS
    r = L_E + (np.arange(STRIPS) + 0.5) * ds
    dA = AREA / STRIPS
    out = np.zeros((K["t"].size, 3))
    for i in range(K["t"].size):
        pb = K["psi"][i] - K["phi"][i]
        fn = ft = mz = 0.0
        for ri in r:
            wn = ri * np.cos(THETA1) * K["rate"][i] - K["f_dot"][i] - (ri - L_E) * K["phi_dot"][i] - VB * np.sin(pb)
            wa = VB * np.cos(pb)
            a, b = law(wn, wa, dA)
            fn += a; ft += b; mz += ri * np.cos(THETA1) * a
        out[i] = (-fn * np.sin(pb) + ft * np.cos(pb), fn * np.cos(pb) + ft * np.sin(pb), mz)
    return out


def added_mass(K, m_a):
    """Kirchhoff's form at the blade centre (normal added mass only): F_norm = -m_a dU_n/dt, F_t = -m_a U_n Omega."""
    rc = L_E + 0.5 * SPAN
    pb = K["psi"] - K["phi"]
    un = rc * np.cos(THETA1) * K["rate"] - K["f_dot"] - 0.5 * SPAN * K["phi_dot"] - VB * np.sin(pb)
    om = K["rate"] - K["phi_dot"]
    fn = -m_a * np.gradient(un, K["t"])
    ft = -m_a * un * om
    return np.stack([-fn * np.sin(pb) + ft * np.cos(pb), fn * np.cos(pb) + ft * np.sin(pb), rc * np.cos(THETA1) * fn], 1)


def score(K, model, meas):
    out = {}
    for j, comp in enumerate(("Fx", "Fy", "Mz")):
        w = []
        for a, b in WINDOWS:
            m = (K["psi_deg"] >= a) & (K["psi_deg"] < b)
            # mean error over the window, as a fraction of the measured peak (ratios blow up where the load is ~0)
            w.append(float(np.mean(model[m, j] - meas[m, j]) / np.max(np.abs(meas[:, j]))))
        out[comp] = dict(windows=w, impulse_ratio=float(np.trapezoid(model[:, j], K["t"]) / np.trapezoid(meas[:, j], K["t"])),
                         rms=float(np.sqrt(np.mean((model[:, j] - meas[:, j]) ** 2))),
                         peak=float(model[np.argmax(np.abs(model[:, j])), j]),
                         peak_at=float(K["psi_deg"][np.argmax(np.abs(model[:, j]))]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigid", action="store_true", help="ignore the oar's bending")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    D = load()
    K = kinematics(D, a.rigid)
    meas = np.stack([np.interp(K["psi_deg"], *D[k].T) for k in ("Fx_N", "Fy_N", "Mz_Nm")], 1)
    laws = {
        "slip C0 1.3 ([CR06] computed)": slip_law(1.3),
        "slip, research level (C2 140.88 scull -> C0 3.12)": slip_law(140.88 / (0.5 * RHO * 0.0903)),
        "lift-drag CG07": liftdrag_law(LiftDragBlade.big_blade(outboard=1.818, area=AREA)),
        "lift-drag Coppel": liftdrag_law(LiftDragBlade.big_blade_full_scale(outboard=1.818, area=AREA, source="coppel")),
        "lift-drag Sliasas-Tullis": liftdrag_law(LiftDragBlade.big_blade_full_scale(outboard=1.818, area=AREA,
                                                                                    source="sliasas_tullis")),
    }
    traces = {name: loads(K, law) for name, law in laws.items()}
    for lab, m_a in (("Patton", patton_added_mass(SPAN, HEIGHT, RHO)), ("[LB19]", labbe_added_mass(SPAN, HEIGHT, RHO))):
        traces["CG07 + added mass %s (%.1f kg)" % (lab, m_a)] = traces["lift-drag CG07"] + added_mass(K, m_a)
    res = {name: score(K, tr, meas) for name, tr in traces.items()}
    print("[RLBQV19] trial 154 %s: measured F_x peak %.0f N at %.0f deg, M_z %.0f N m; bending up to %.1f deg, %.3f m; "
          "stroke %.2f s (20-150 deg)" % ("RIGID OAR" if a.rigid else "with the oar's bending", meas[:, 0].max(),
                                          K["psi_deg"][np.argmax(meas[:, 0])], meas[:, 2].min(), np.degrees(K["phi"].max()),
                                          K["f"].max(), K["t"][-1]))
    print("\n%-50s F_x mean error / peak, by window %s   F_x impulse  peak N @deg   F_y rms  M_z rms  F_x rms"
          % ("law", "/".join("%d-%d" % w for w in WINDOWS)))
    for name, r in res.items():
        print("%-50s %s          %5.2f    %5.0f @%3.0f   %5.1f   %6.1f   %5.1f" % (
            name, " ".join("%+5.2f" % v for v in r["Fx"]["windows"]), r["Fx"]["impulse_ratio"], r["Fx"]["peak"],
            r["Fx"]["peak_at"], r["Fy"]["rms"], r["Mz"]["rms"], r["Fx"]["rms"]))
    tag = "_rigid" if a.rigid else ""
    json.dump(dict(results=res, rigid=a.rigid), open(os.path.join(OUT, "robert2019_quasisteady%s.json" % tag), "w"),
              indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for j, lab in enumerate(("F_x (N)", "F_y (N)", "M_z (N m)")):
            ax[j].plot(K["psi_deg"], meas[:, j], "k", lw=2, label="measured [RLBQV19]")
            for name, tr in traces.items():
                ax[j].plot(K["psi_deg"], tr[:, j], lw=1, label=name)
            ax[j].set_xlabel("sweep angle psi_0 (deg)"); ax[j].set_ylabel(lab); ax[j].grid(alpha=0.3)
        ax[0].legend(fontsize=6)
        fig.savefig(os.path.join(OUT, "robert2019_quasisteady%s.png" % tag), dpi=130)
    except Exception as e:                                         # the figure is a convenience
        print("figure skipped: %s" % e)


if __name__ == "__main__":
    main()
