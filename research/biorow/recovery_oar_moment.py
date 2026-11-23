"""The oar's weight moment from the feathered recovery: where is the oar's centre of mass? (§260)

Kleshnev (email 2026-11-22): BioRow's force sensor is on the shaft and turns with the feathering, so with the blade
feathered in mid-recovery it measures the force in the *vertical* plane. There the hands hold the oar's weight about
the pin, so the vertical-plane balance about the pin is

    F_h(t) l_h = m g x_cg cos(psi) + I_v psi_ddot(t)        (psi the vertical oar angle, l_h the force point's lever)

with no blade load and no horizontal inertia in that plane. Over the mid-recovery window (feathered, blade clear of
the water, away from the squaring before the catch and the feathering after the finish) a least-squares fit of
F_h l_h on (g cos psi, psi_ddot) gives the static moment S = m x_cg and the vertical inertia I_v. With Concept2's own
oar mass (Ultralight / Skinny sculls ~1.375 kg, Concept2's Bantam page) S gives x_cg, the centre of mass outboard of
the pin, which sets the oar's inertia about the pin, I = m (k^2 + x_cg^2).

Compared with the model's scull: a uniform rod (research profile, scull_mass 1.2 kg from [CR06] Table 1) has its
centre at L/2 - inboard from the pin.

Caveats, printed: the sensor's zero (set by BioRow with the oar unloaded in the horizontal plane), the sign
convention in the vertical plane, and that the file is a ~20-stroke average at 27 Hz.

    python -I research/biorow/recovery_oar_moment.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv")

G = 9.81
L_OAR, INBOARD = 2.885, 0.875             # [BR24]'s sculls (BioRow file header)
L_H = INBOARD - 0.06                       # the force point, 0.06 m from the handle's top (Kleshnev 2026-10-28)
M_CONCEPT2 = 1.375                         # kg, Concept2: Ultralight racing sculls ~1375 g; Skinny "nearly the same"
M_MODEL = 1.2                              # kg, the research profile's scull ([CR06] Table 1)


def load():
    rows = list(csv.DictReader(open(DATA)))
    col = lambda k: np.array([float(r[k]) for r in rows])
    return col("Ti"), col("A1"), col("A2"), col("V1"), col("V2"), col("H1"), col("H2")


def periodic_second_derivative(t, y):
    """Second derivative of a periodic sampled series (the file's last sample repeats the first)."""
    T = t[-1] - t[0]
    tt, yy = t[:-1], y[:-1]
    n = tt.size
    dt = T / n
    return (np.roll(yy, -1) - 2 * yy + np.roll(yy, 1)) / dt ** 2, tt


def main():
    t, a1, a2, v1, v2, h1, h2 = load()
    T = t[-1]
    finish = t[int(np.argmax(a1))]
    catch = t[int(np.argmin(a1))]
    print("[BR24] cycle %.3f s; finish (max angle) %.3f s, catch (min angle) %.3f s" % (T, finish, catch))
    # mid-recovery: from 0.06 s after the finish's turn to 0.20 s before the catch (squaring and feathering out)
    lo, hi = finish + 0.06, catch - 0.20 + T
    print("mid-recovery window: %.3f s to %.3f s (wrapping), the oar feathered and clear of the water" % (lo, hi - T))
    res = {}
    for side, v, h in (("oar 1", v1, h1), ("oar 2", v2, h2)):
        psi = np.radians(v)
        psidd, tt = periodic_second_derivative(t, psi)
        hh = h[:-1]
        tw = np.where(tt < lo, tt + T, tt)
        sel = (tw >= lo) & (tw <= hi)
        y = hh[sel] * L_H
        X = np.column_stack([G * np.cos(psi[:-1][sel]), psidd[sel]])
        (S, Iv), *_ = np.linalg.lstsq(X, y, rcond=None)
        S0 = float(np.mean(y) / G)                                # static only
        res[side] = (S, Iv, S0)
        print("  %s: %d samples, force %.1f-%.1f N (mean %.1f); fit S = m x_cg %.3f kg m, I_v %.2f kg m^2; static "
              "only S %.3f kg m" % (side, sel.sum(), hh[sel].min(), hh[sel].max(), hh[sel].mean(), S, Iv, S0))
    S = np.mean([r[2] for r in res.values()])
    lo_s = min(r[2] for r in res.values())
    hi_s = max(r[2] for r in res.values())
    x_c2 = S / M_CONCEPT2
    x_rod = L_OAR / 2.0 - INBOARD
    print("\nstatic moment m x_cg: %.3f kg m (oars %.3f-%.3f)" % (S, lo_s, hi_s))
    print("  with Concept2's 1.375 kg: x_cg %.2f m outboard of the pin (%.2f m from the handle end)" % (x_c2,
                                                                                                    x_c2 + INBOARD))
    print("  the model's uniform rod: x_cg %.3f m; static moment %.3f kg m at 1.2 kg, %.3f at 1.375 kg"
          % (x_rod, M_MODEL * x_rod, M_CONCEPT2 * x_rod))
    k2 = L_OAR ** 2 / 12.0
    print("  inertia about the pin, uniform rod: %.2f kg m^2 at 1.2 kg, %.2f at 1.375 kg (x_cg %.3f)"
          % (M_MODEL * (k2 + x_rod ** 2), M_CONCEPT2 * (k2 + x_rod ** 2), x_rod))
    print("  inertia about the pin with the measured x_cg and a rod's radius of gyration: %.2f kg m^2"
          % (M_CONCEPT2 * (k2 + x_c2 ** 2)))
    print("\ncaveats: the sensor's zero and its vertical-plane sign are BioRow's; a 20-stroke average at 27 Hz; the "
          "fit's I_v is poorly conditioned (psi varies by ~2 deg), the static moment is the robust number")


if __name__ == "__main__":
    sys.exit(main())
