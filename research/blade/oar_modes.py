"""Vibration modes of a scull in the water: is the quasi-static bending hinge valid? (SOURCES §224)

The dynamic oar's bending (coxswain.crew.oar_bending) is a massless hinge: the shaft is in static equilibrium with the
blade's load at every instant. That holds only if the shaft's own bending mode -- with the blade's mass and the water
it carries -- is much faster than the loads that drive it. This finds those modes for [CR06]'s scull with a 1-D
Euler-Bernoulli beam (Hermite cubic elements, consistent mass), verified against closed forms, and with:

* **the shaft:** [L14]'s measured medium scull EI (2099 N m^2) along the shaft; the blade 50x stiffer (rigid);
  the oar's 1.2 kg spread uniformly (a 0.3 kg blade is assumed; reported: the beam's inertia about the pin against
  the rig's 1.237 kg m^2);
* **the pin:** no lateral motion at the oarlock, free rotation;
* **the hands:** ``held`` (no lateral motion at the handle: the crew's inertia dominates), ``free`` (no hands), or
  ``crew`` (a lumped mass at the handle on the arms' stretch spring, SOURCES §210: ~1000 N per 12 cm both arms);
* **the water:** the blade's added mass for motion normal to its face, spread along the blade, Patton (13 kg) to
  Labbe et al. (22 kg) (coxswain.crew.blade_added_mass), and a linearised drag damping c = rho C_n A u at the blade
  (C_n 2.2, the research blade's face-on coefficient; u the normal slip speed).

Closed forms printed alongside: a uniform cantilever's beta_n L = 1.8751, 4.6941, 7.8548; and the outboard as a
cantilever from the pin with the blade's (added) mass at its centre, omega = sqrt(3 EI / (c^3 (m + 0.24 m_shaft))).

    python research/blade/oar_modes.py [--slip 1.5]
"""
import argparse
import os
import sys

import numpy as np
from scipy.linalg import eig, eigh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT]
from coxswain.crew.blade_added_mass import labbe_added_mass, patton_added_mass   # noqa: E402
from coxswain.crew.oar_bending import L14_MEDIUM_EI                               # noqa: E402

RHO = 1000.0
#: [CR06]'s scull in the model (handle_rung1 rig): length, inboard, blade length, blade centre from the pin, area, mass
SCULL = dict(length=2.85, inboard=0.83, blade_length=0.43, blade_centre=1.805, blade_area=0.0903, mass=1.2,
             blade_width=0.215, inertia_about_lock=1.237)
BLADE_MASS = 0.3                 # kg, assumed (Concept2 does not publish it); small beside the added mass
ARM_STIFFNESS = 1000.0 / 0.12    # N/m, both arms: SOURCES §210's stretch, ~12 cm at ~1000 N handle force
CN_FACE_ON = 2.2                 # the research blade's face-on normal coefficient (SOURCES §201)


def beam_matrices(nodes, EI, m):
    """Global stiffness and consistent mass for Hermite beam elements; ``EI``, ``m`` per element. DOF (w, theta)."""
    n = len(nodes)
    K = np.zeros((2 * n, 2 * n)); M = np.zeros((2 * n, 2 * n))
    for e in range(n - 1):
        L = nodes[e + 1] - nodes[e]
        k = EI[e] / L ** 3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                                        [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
        mm = m[e] * L / 420.0 * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                                           [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L * L, -22 * L, 4 * L * L]])
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(idx, idx)] += k; M[np.ix_(idx, idx)] += mm
    return K, M


def verify(n=60):
    """A uniform clamped-free beam against beta_n L: returns the ratios FE / exact for the first three modes."""
    L, EI, m = 1.0, 1.0, 1.0
    x = np.linspace(0, L, n + 1)
    K, M = beam_matrices(x, np.full(n, EI), np.full(n, m))
    keep = np.arange(2, 2 * (n + 1))                       # clamp w and theta at x = 0
    w2 = eigh(K[np.ix_(keep, keep)], M[np.ix_(keep, keep)], eigvals_only=True)
    exact = np.array([1.8751, 4.6941, 7.8548]) ** 2
    return np.sqrt(w2[:3]) / exact


def oar_model(hands, added, slip, n_el=120, EI=L14_MEDIUM_EI):
    """Nodes from the handle (x = 0) to the tip; returns the reduced (K, M, C), node x, the pin's x and the blade span."""
    s = SCULL
    L, x_pin = s["length"], s["inboard"]
    root = x_pin + s["blade_centre"] - 0.5 * s["blade_length"]
    x = np.linspace(0.0, L, n_el + 1)
    for p in (x_pin, root):                                 # put nodes exactly at the pin and the blade root
        x[np.argmin(abs(x - p))] = p
    x = np.unique(x)
    mid = 0.5 * (x[1:] + x[:-1])
    on_blade = mid >= root
    m_shaft = (s["mass"] - BLADE_MASS) / (root - 0.0)
    m_line = np.where(on_blade, BLADE_MASS / s["blade_length"] + added / s["blade_length"], m_shaft)
    EI_line = np.where(on_blade, 50.0 * EI, EI)
    K, M = beam_matrices(x, EI_line, m_line)
    C = np.zeros_like(K)
    c_total = RHO * CN_FACE_ON * s["blade_area"] * slip     # N s/m, linearised drag on the blade
    if c_total > 0.0:                                       # spread over the blade's nodes as a lumped damper
        blade_nodes = np.where(x >= root - 1e-12)[0]
        for i in blade_nodes:
            C[2 * i, 2 * i] += c_total / len(blade_nodes)
    i_pin, i_hand = int(np.argmin(abs(x - x_pin))), 0
    fixed = [2 * i_pin]                                     # the pin: no lateral motion
    if hands == "held":
        fixed.append(2 * i_hand)
    elif hands == "crew":                                   # crew mass on the arms' spring, grounded through the crew
        # the crew's reflected mass at the handle, per oar: a third DOF-free lumped mass is folded into a spring to
        # ground in series -- here as the spring alone (the crew's inertia >> the oar's, SOURCES §218)
        K[2 * i_hand, 2 * i_hand] += ARM_STIFFNESS / 2.0
    keep = np.setdiff1d(np.arange(K.shape[0]), fixed)
    return K[np.ix_(keep, keep)], M[np.ix_(keep, keep)], C[np.ix_(keep, keep)], x, x_pin, root


def modes(K, M, C, count=3):
    """Undamped frequencies (Hz) and the damped modes' frequency (Hz) and damping ratio, lowest first (rigid modes
    with frequency < 0.05 Hz dropped)."""
    w2 = eigh(K, M, eigvals_only=True)
    f_und = np.sqrt(np.maximum(w2, 0.0)) / (2 * np.pi)
    f_und = f_und[f_und > 0.05][:count]
    n = K.shape[0]
    A = np.block([[np.zeros((n, n)), np.eye(n)], [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])
    lam = eig(A, right=False)
    lam = lam[np.abs(lam) > 2 * np.pi * 0.05]
    osc = lam[lam.imag > 1e-6]
    osc = osc[np.argsort(np.abs(osc))][:count]
    over = lam[np.abs(lam.imag) <= 1e-6]                    # overdamped: real roots
    return f_und, [(abs(l) / (2 * np.pi), -l.real / abs(l)) for l in osc], np.sort(-over.real)[:count]


#: Breitschädel & Løset (2010), Procedia Eng. 2:3011 (data/local/literature/breitschadel2010_oar_vibration/): eight
#: reference sculls clamped behind the collar, free oscillation after a 50 N tip load released: outboard length (m),
#: outboard mass (kg, assumed uniform by the authors), first natural frequency (Hz), damping ratio. Table 1.
BL2010 = [(2.030, 1.293, 8.27, 0.029), (2.030, 1.301, 8.19, 0.038), (2.032, 1.139, 8.08, 0.037),
          (2.032, 1.127, 8.27, 0.029), (2.032, 1.093, 8.40, 0.030), (2.032, 1.124, 8.18, 0.034),
          (2.018, 1.110, 8.44, 0.037), (2.023, 1.090, 8.43, 0.036)]


def bl2010_check(EI=L14_MEDIUM_EI, blade_length=0.43, n_el=80):
    """The measured in-air frequencies against this beam: clamped at the collar, outboard L_o, uniform mass m_o (as
    the authors assumed), [L14]'s shaft EI, the blade 50x stiffer. Returns (measured f, predicted f) per oar."""
    out = []
    for L, m_o, f_meas, _zeta in BL2010:
        x = np.linspace(0.0, L, n_el + 1)
        mid = 0.5 * (x[1:] + x[:-1])
        EI_line = np.where(mid >= L - blade_length, 50.0 * EI, EI)
        K, M = beam_matrices(x, EI_line, np.full(n_el, m_o / L))
        keep = np.arange(2, K.shape[0])
        f = np.sqrt(eigh(K[np.ix_(keep, keep)], M[np.ix_(keep, keep)], eigvals_only=True)[0]) / (2 * np.pi)
        out.append((f_meas, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip", type=float, default=1.5, help="blade normal slip speed for the drag damping, m/s")
    a = ap.parse_args()
    s = SCULL
    print("verification, uniform cantilever FE / exact (modes 1-3):", " ".join("%.5f" % r for r in verify()))
    chk = np.array(bl2010_check())
    print("Breitschädel & Løset 2010, in air, clamped at the collar: measured %.2f-%.2f Hz, predicted with [L14] EI "
          "%.2f-%.2f Hz (mean ratio %.3f); measured damping ratio %.1f-%.1f %%"
          % (chk[:, 0].min(), chk[:, 0].max(), chk[:, 1].min(), chk[:, 1].max(), np.mean(chk[:, 1] / chk[:, 0]),
             100 * min(z for *_, z in BL2010), 100 * max(z for *_, z in BL2010)))
    # the EI that reproduces each measured frequency (f scales with sqrt(EI))
    ei_fit = L14_MEDIUM_EI * (chk[:, 0] / chk[:, 1]) ** 2
    print("  EI that reproduces them: %.0f-%.0f N m^2 (mean %.0f) against [L14]'s %.0f"
          % (ei_fit.min(), ei_fit.max(), ei_fit.mean(), L14_MEDIUM_EI))
    K, M, C, x, x_pin, root = oar_model("free", 0.0, 0.0)
    # the beam's inertia about the pin, against the rig's
    m_shaft = (s["mass"] - BLADE_MASS) / root
    I_beam = m_shaft * ((x_pin ** 3) + (root - x_pin) ** 3) / 3.0 + BLADE_MASS * s["blade_centre"] ** 2
    print("beam inertia about the pin %.3f kg m^2 (rig %.3f)" % (I_beam, s["inertia_about_lock"]))
    ma = dict(patton=patton_added_mass(s["blade_length"], s["blade_width"], RHO),
              labbe=labbe_added_mass(s["blade_length"], s["blade_width"], RHO))
    c = s["blade_centre"]
    k_tip = 3.0 * L14_MEDIUM_EI / c ** 3
    print("outboard as a cantilever from the pin, load at the blade centre: k = %.0f N/m" % k_tip)
    m_out = m_shaft * (root - x_pin)
    print("\n%-8s %-7s %8s | closed form f1 (Hz) | FE undamped f (Hz)       | damped (slip %.1f m/s): f Hz, zeta"
          % ("hands", "water", "added kg", a.slip))
    for added_name, added in (("air", 0.0), ("patton", ma["patton"]), ("labbe", ma["labbe"])):
        f_cf = np.sqrt(k_tip / (added + BLADE_MASS + 0.24 * m_out)) / (2 * np.pi)
        for hands in ("held", "free", "crew"):
            K, M, C, *_ = oar_model(hands, added, a.slip if added > 0 else 0.0)
            f_und, damped, over = modes(K, M, C)
            dtxt = ", ".join("%.2f/%.2f" % d for d in damped[:2]) if damped else "none oscillatory"
            if len(over):
                dtxt += "; overdamped rates %s 1/s" % ", ".join("%.1f" % r for r in over[:2])
            print("%-8s %-7s %8.1f | %8.2f            | %s | %s" % (hands, added_name, added, f_cf,
                                                                    " ".join("%6.2f" % f for f in f_und), dtxt))


if __name__ == "__main__":
    main()
