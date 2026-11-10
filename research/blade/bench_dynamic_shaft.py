"""The dynamic shaft on [RLBQV19]'s tank trial: does a bent oar carrying its blade's water bend when theirs did? (§228)

The validation ladder's in-water shaft rung (docs/VALIDATION_LADDER.md, rung B1). [RLBQV19]'s trial 154 sweeps a
full-size sweep oar through the water by motor while a carriage tows it at 4.202 m/s, and measures F_x, F_y and the
oarlock moment M_z; their bench law phi = |M_z| / K_ang turns the measured moment into the measured bend. The motor's
motion is known, so the bend is a prediction: here from the same blade (strips, the research cross-flow law, face-on
2.2, normal load only, as rung 2's ``crossflow_normal``) on three shafts:

* ``quasi-static``: the massless hinge, K_ang phi = the load's moment about the pin (research/blade/oar_bending.py);
* ``dynamic``: the blade as a body on the hinge with its dry mass and its added mass (Patton, or [LB19]) and the
  measured structural damping, as ``BendDynamics`` does in the boat: the water's force is the rate of change of its
  momentum along the normal, -m_a d(u_n)/dt, taken about the hinge::

      u_n   = r_c cos(theta_1) psi_dot - e phi_dot - V_b sin(psi - phi)          (blade centre, normal)
      m e u_n_dot = sum_i (r_i - h) F_i + k_h phi + c_h phi_dot,    m = m_dry + m_a

  with F_i the strips' loads (negative while driving), k_h = K_ang e / (r_c cos theta_1) so that the static limit is
  their law, c_h = 2 zeta sqrt(k_h m_dry e^2). ``--munk`` adds the ideal-flow Munk term m u_n V_b cos(psi - phi) (the
  equation from the Lagrangian with the water's energy 0.5 m_a u_n^2), which the boat model does not carry and a
  separated blade does not feel -- shown only to say what it is worth;
* ``rigid``.

Scored against the measurement: the bend (rms, peak and where), F_x and M_z (rms, impulse ratio, peak and where).

    python research/blade/bench_dynamic_shaft.py

Output: printed; data/local/blade/bench_dynamic_shaft.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import robert2019_quasisteady as RQ                              # noqa: E402

from coxswain.crew.blade_added_mass import labbe_added_mass, patton_added_mass   # noqa: E402
from coxswain.crew.liftdrag import LiftDragBlade                 # noqa: E402
from coxswain.crew.oar_bending import BLADE_MASS_ASSUMED, STRUCTURAL_DAMPING_BL2010   # noqa: E402

HINGE = RQ.L_E - RQ.K_ANG / RQ.K_LIN                             # 0.697 m from the pin ([RLBQV19] eq. 1)
RC = RQ.L_E + 0.5 * RQ.SPAN                                      # blade centre, 1.818 m
STRIP_R = RQ.L_E + (np.arange(RQ.STRIPS) + 0.5) * RQ.SPAN / RQ.STRIPS
COS1 = np.cos(RQ.THETA1)


def research_blade(face_on=2.2, lift_scale=1.0):
    """Rung 2's ``crossflow_normal`` on this blade: the cross-flow form, face-on 2.2, normal load only."""
    cf = LiftDragBlade.crossflow(outboard=RC, area=RQ.AREA, length=RQ.SPAN, face_on=face_on, density=RQ.RHO,
                                 lift_scale=lift_scale)

    def law(wn, wa, dA):
        q = 0.5 * RQ.RHO * dA * (wn * wn + wa * wa)
        if q <= 0.0:
            return 0.0
        c_n, _c_t = cf.coefficients(np.arctan2(abs(wn), abs(wa)))
        return -np.sign(wn) * q * c_n
    return law


def strip_loads(psi, rate, phi, phid, law):
    """(sum (r - h) F_i, F_x, F_y, M_z) at this state; F_i normal, negative while driving (as [RLBQV19] print)."""
    dA = RQ.AREA / RQ.STRIPS
    pb = psi - phi
    hinge_m = fn = mz = 0.0
    for r in STRIP_R:
        wn = r * COS1 * rate - (r - HINGE) * phid - RQ.VB * np.sin(pb)
        f = law(wn, RQ.VB * np.cos(pb), dA)
        fn += f
        hinge_m += (r - HINGE) * f
        mz += r * COS1 * f
    return hinge_m, -fn * np.sin(pb), fn * np.cos(pb), mz


def run_dynamic(K, law, m_a, m_dry=BLADE_MASS_ASSUMED, zeta=STRUCTURAL_DAMPING_BL2010, substeps=20, munk=False):
    """The bend through the trial, RK4 on the motor's time base (``substeps`` per sample)."""
    t, psi_k, rate_k = K["t"], K["psi"], K["rate"]
    acc_k = np.gradient(rate_k, t)
    e = RC - HINGE
    k_h = RQ.K_ANG * e / (RC * COS1)
    m = m_dry + m_a
    c_h = 2.0 * zeta * np.sqrt(k_h * m_dry * e * e)

    def motor(tt):
        return (np.interp(tt, t, psi_k), np.interp(tt, t, rate_k), np.interp(tt, t, acc_k))

    def rhs(tt, y):
        phi, phid = y
        psi, rate, acc = motor(tt)
        un = RC * COS1 * rate - e * phid - RQ.VB * np.sin(psi - phi)
        hinge_m = strip_loads(psi, rate, phi, phid, law)[0]
        c = np.cos(psi - phi)
        # m e (r_c cos psi_ddot - e phi_ddot - V_b c (psi_dot - phi_dot)) + m u_n V_b c = hinge_m + k_h phi + c_h phi_dot
        num = m * e * (RC * COS1 * acc - RQ.VB * c * (rate - phid)) + (m * un * RQ.VB * c if munk else 0.0) - hinge_m - k_h * phi - c_h * phid
        return np.array([phid, num / (m * e * e)])

    # start from the quasi-static bend at the first sample, at rest relative to it
    y = np.array([float(K["phi_qs0"]), 0.0])
    out = np.zeros((t.size, 2))
    out[0] = y
    for i in range(t.size - 1):
        h = (t[i + 1] - t[i]) / substeps
        tt = t[i]
        for _ in range(substeps):
            k1 = rhs(tt, y); k2 = rhs(tt + h / 2, y + h / 2 * k1)
            k3 = rhs(tt + h / 2, y + h / 2 * k2); k4 = rhs(tt + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            tt += h
        out[i + 1] = y
    return out[:, 0], out[:, 1]


def run_quasistatic(K, law):
    """The massless hinge: k_h phi + sum (r - h) F_i(phi, phi_dot) = 0 at every instant, phi_dot solved."""
    from scipy.optimize import brentq
    e = RC - HINGE
    k_h = RQ.K_ANG * e / (RC * COS1)
    t, psi, rate = K["t"], K["psi"], K["rate"]

    def phid_at(i, phi):
        g = lambda pd: strip_loads(psi[i], rate[i], phi, pd, law)[0] + k_h * phi   # noqa: E731
        lo, hi = -40.0, 40.0
        if g(lo) * g(hi) > 0:
            return lo if abs(g(lo)) < abs(g(hi)) else hi
        return brentq(g, lo, hi, xtol=1e-7)

    phi = np.zeros(t.size)
    g0 = lambda p: strip_loads(psi[0], rate[0], p, 0.0, law)[0] + k_h * p              # noqa: E731
    from scipy.optimize import brentq as bq
    phi[0] = bq(g0, -0.5, 0.5) if g0(-0.5) * g0(0.5) < 0 else 0.0
    for i in range(t.size - 1):
        dt = t[i + 1] - t[i]
        k1 = phid_at(i, phi[i])
        k2 = phid_at(i + 1, phi[i] + dt * k1)
        phi[i + 1] = phi[i] + 0.5 * dt * (k1 + k2)
    return phi, np.gradient(phi, t)


def forces(K, phi, phid, law):
    return np.array([strip_loads(K["psi"][i], K["rate"][i], phi[i], phid[i], law)[1:] for i in range(K["t"].size)])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-on", type=float, default=2.2,
                    help="the cross-flow face-on value (2.2 rung 2's; 1.3 [RLBQV19]'s own at Fr 1.0, SOURCES §190-§191)")
    ap.add_argument("--munk", action="store_true", help="add the ideal-flow Munk term (not in the boat model)")
    ap.add_argument("--lift-scale", type=float, default=1.0, help="the cross-flow lift slope over Helmbold's (§231)")
    a = ap.parse_args()
    D = RQ.load()
    K = RQ.kinematics(D, rigid=True, n=400)                       # the motor's motion
    meas = np.stack([np.interp(K["psi_deg"], *D[k].T) for k in ("Fx_N", "Fy_N", "Mz_Nm")], 1)
    phi_meas = np.abs(meas[:, 2]) / RQ.K_ANG
    law = research_blade(a.face_on, a.lift_scale)
    qs_phi, qs_phid = run_quasistatic(K, law)
    K["phi_qs0"] = qs_phi[0]
    m_p = patton_added_mass(RQ.SPAN, RQ.HEIGHT, RQ.RHO)
    m_l = labbe_added_mass(RQ.SPAN, RQ.HEIGHT, RQ.RHO)
    cases = {"rigid": (np.zeros_like(qs_phi), np.zeros_like(qs_phi)), "quasi-static hinge": (qs_phi, qs_phid)}
    # the surface factor kappa: [GR20]'s surface / submerged ratio ~0.5; the population force curves' bound < 0.1 (§173)
    for lab, m_a in (("Patton x0.1 %.1f kg" % (0.1 * m_p), 0.1 * m_p), ("Patton x0.5 %.1f kg" % (0.5 * m_p), 0.5 * m_p),
                     ("Patton %.1f kg" % m_p, m_p), ("[LB19] %.1f kg" % m_l, m_l)):
        cases["dynamic, " + lab] = run_dynamic(K, law, m_a, munk=a.munk)
    print("[RLBQV19] trial 154, research cross-flow blade (face-on %.2f, normal load)%s; measured bend peak %.2f deg at "
          "%.0f deg, F_x peak %.0f N at %.0f deg" % (a.face_on, ", with the Munk term" if a.munk else "",
                                                   np.degrees(phi_meas.max()), K["psi_deg"][np.argmax(phi_meas)],
                                                   meas[:, 0].max(), K["psi_deg"][np.argmax(meas[:, 0])]))
    print("\n%-26s | bend peak @deg  rms deg | F_x peak @deg  rms N  impulse | M_z rms  impulse" % "shaft")
    res = {}
    for name, (phi, phid) in cases.items():
        F = forces(K, phi, phid, law)
        sc = RQ.score(K, F, meas)
        r = dict(bend_peak_deg=float(np.degrees(phi.max())), bend_peak_at=float(K["psi_deg"][np.argmax(phi)]),
                 bend_rms_deg=float(np.degrees(np.sqrt(np.mean((phi - phi_meas) ** 2)))), score=sc)
        res[name] = r
        print("%-26s | %5.2f @%4.0f   %5.2f   | %5.0f @%4.0f  %5.1f   %5.2f  | %6.1f  %5.2f" % (
            name, r["bend_peak_deg"], r["bend_peak_at"], r["bend_rms_deg"], sc["Fx"]["peak"], sc["Fx"]["peak_at"],
            sc["Fx"]["rms"], sc["Fx"]["impulse_ratio"], sc["Mz"]["rms"], sc["Mz"]["impulse_ratio"]))
    print("\nbend through the trial (deg), measured and each shaft:")
    print("  psi  | measured | " + " | ".join("%-14s" % n[:14] for n in cases))
    for p in range(20, 151, 10):
        i = int(np.argmin(abs(K["psi_deg"] - p)))
        print("  %4d | %6.2f   | " % (p, np.degrees(phi_meas[i])) +
              " | ".join("%14.2f" % np.degrees(cases[n][0][i]) for n in cases))
    os.makedirs(RQ.OUT, exist_ok=True)
    json.dump(dict(results=res, added_mass=dict(patton=m_p, labbe=m_l)),
              open(os.path.join(RQ.OUT, "bench_dynamic_shaft_faceon%g%s%s.json" % (a.face_on, "_munk" if a.munk else "", "" if a.lift_scale == 1.0 else "_lift%g" % a.lift_scale)), "w"), indent=1)


if __name__ == "__main__":
    main()
