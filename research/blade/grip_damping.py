"""Does a real grip damp the oar's in-air bending mode? The post-release ringing with a sourced hand model (§250).

After the release the model's dry blade rings on the shaft's hinge at ~9 Hz, damped only by the shaft's structural 3.4 %
(Breitschadel & Loset 2010), because the hands are imposed as a rigid motion (§240). Here the oar is free to rotate
about the pin against a sourced hand: Dong, Dong, Wu & Rakheja (2007), J. Biomech. 40:2335-2340, Table 1, the 4-DOF
fingers-hand-arm model gripping a handle (50 N), identified from driving-point responses along the forearm axis:
palm and finger skin masses M3, M4 moving with the handle; finger mass M2 and palm-wrist-arm mass M1, coupled
handle-M2 (k4, c4), handle-M1 (k3, c3), M2-M1 (k2, c2) and M1-body (k1, c1). The handle's motion perpendicular to
the shaft is taken along the model's axis (the forearm, near the boat's axis at the finish) -- an assumption.

The oar: the inner part rotating about the pin (theta; inertia about the pin less the blade's), the blade's dry mass on
the shaft's hinge (phi; the simulator's [L14] law: k_h, lever e, structural damping 3.4 %), the hand at the inboard
force point. Linear, in air, small motions. Prints each mode's frequency and damping ratio, rigid hands against the
gripping hand, for the grip and grip+push conditions, and the ringing's decay time (to 10 %).

    python research/blade/grip_damping.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from coxswain.crew.oar_bending import BLADE_MASS_ASSUMED, STRUCTURAL_DAMPING_BL2010, OarBending  # noqa: E402

#: Dong et al. (2007) Table 1, 4-DOF model: (M1, M2, M3, M4) kg, (k1, k2, k3, k4) N/m, (c1, c2, c3, c4) N s/m
DONG2007 = {
    "grip 50 N": ((1.4329, 0.0897, 0.0230, 0.0147), (3377.0, 12710.0, 29906.0, 190041.0), (50.6, 35.2, 74.5, 127.6)),
    "grip + push 50 N": ((1.4145, 0.0820, 0.0267, 0.0140), (4207.0, 6523.0, 58555.0, 207964.0), (85.9, 37.9, 118.3, 120.8)),
}

# [CR06]'s scull (the simulator's catalogue values): blade centre 1.805 m, blade 0.43 m, inboard to the force point 0.83 m,
# oar inertia about the pin 1.237 kg m^2
CENTRE, BLADE_LEN, INBOARD, I_OAR = 1.805, 0.43, 0.83, 1.237


def oar_matrices():
    law = OarBending.laschowski2014_scull(CENTRE, BLADE_LEN)
    h = law.hinge_radius(CENTRE, BLADE_LEN)
    e = CENTRE - h
    k_h = law.k_ang * e / CENTRE
    m_b = BLADE_MASS_ASSUMED
    c_h = 2.0 * STRUCTURAL_DAMPING_BL2010 * np.sqrt(k_h * m_b * e * e)
    i_in = I_OAR - m_b * CENTRE ** 2
    return dict(e=e, k_h=k_h, c_h=c_h, m_b=m_b, i_in=i_in, l=CENTRE)


def modes(M, C, K):
    n = M.shape[0]
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    Mi = np.linalg.inv(M)
    A[n:, :n] = -Mi @ K
    A[n:, n:] = -Mi @ C
    lam = np.linalg.eigvals(A)
    lam = lam[np.imag(lam) > 1e-6]
    w = np.abs(lam)
    zeta = -np.real(lam) / w
    o = np.argsort(w)
    return w[o] / (2 * np.pi), zeta[o]


def rigid_hands(p):
    """Hands imposed (theta fixed): the blade alone on the hinge."""
    return modes(np.array([[p["m_b"] * p["e"] ** 2]]), np.array([[p["c_h"]]]), np.array([[p["k_h"]]]))


def gripping(p, hand):
    """DOFs: theta (oar about the pin), phi (bend), x1 (palm-wrist-arm), x2 (fingers); handle displacement L theta."""
    (m1, m2, m3, m4), (k1, k2, k3, k4), (c1, c2, c3, c4) = hand
    L, l, e, mb = INBOARD, p["l"], p["e"], p["m_b"]
    M = np.zeros((4, 4))
    M[0, 0] = p["i_in"] + mb * l * l + (m3 + m4) * L * L
    M[0, 1] = M[1, 0] = mb * l * e
    M[1, 1] = mb * e * e
    M[2, 2], M[3, 3] = m1, m2

    def pair(mat, i, j, val, ri=1.0, rj=1.0):          # element val between coordinate i (scale ri) and j (scale rj)
        mat[i, i] += val * ri * ri
        mat[j, j] += val * rj * rj
        mat[i, j] -= val * ri * rj
        mat[j, i] -= val * ri * rj
    K, C = np.zeros((4, 4)), np.zeros((4, 4))
    K[1, 1] += p["k_h"]
    C[1, 1] += p["c_h"]
    for mat, vals in ((K, (k1, k2, k3, k4)), (C, (c1, c2, c3, c4))):
        v1, v2, v3, v4 = vals
        mat[2, 2] += v1                                  # M1 to the body
        pair(mat, 3, 2, v2)                              # M2 - M1
        pair(mat, 0, 2, v3, ri=L)                        # handle - M1 (palm)
        pair(mat, 0, 3, v4, ri=L)                        # handle - M2 (fingers)
    return modes(M, C, K)


def main():
    p = oar_matrices()
    print("oar: hinge lever e %.3f m, k_h %.0f N m, dry blade %.2f kg, inner oar %.3f kg m^2 about the pin"
          % (p["e"], p["k_h"], p["m_b"], p["i_in"]))
    f, z = rigid_hands(p)
    print("\nrigid hands (the simulator): bend mode %.1f Hz, damping ratio %.3f, decays to 10 %% in %.2f s"
          % (f[0], z[0], np.log(10) / (z[0] * 2 * np.pi * f[0])))
    for name, hand in DONG2007.items():
        f, z = gripping(p, hand)
        print("\n%s (Dong et al. 2007, Table 1):" % name)
        for fi, zi in zip(f, z):
            print("  mode %6.1f Hz, damping ratio %.3f, decays to 10 %% in %.3f s"
                  % (fi, zi, np.log(10) / (zi * 2 * np.pi * fi) if zi > 0 else float("inf")))


if __name__ == "__main__":
    main()
