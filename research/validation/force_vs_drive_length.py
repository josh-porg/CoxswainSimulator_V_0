"""Handle force against drive length: model, [BR24] and BioRow's population reference curve (SOURCES §234).

Kleshnev (RBN 2023/12, "Averaged biomechanical curves", >50 000 samples) gives the population's handle force as a
percentage of its peak at points of drive length (catch -> finish by oar angle): 13 % at the catch (the oar's inertia),
22 % at 10 % (the blade centre below the water), ~60 % at 20 % (fully buried at -3 deg), 100 % at 40 %, 74 % at 60 %
(handle velocity peak, oar near perpendicular), ~0 at 90 %. Reads the per-sample tables research/blade/
flex_gap_diagnosis.py saves (``--surface-factor`` runs; columns t, gate, bend, bend rate, speed, air, power, rate,
force) and his BioRow cycle.

    python research/validation/force_vs_drive_length.py [npz ...]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
#: RBN 2023/12's population force, % of peak, against drive length (stated in its text)
BIOROW_2023 = ((0.0, 13.0), (0.1, 22.0), (0.2, 60.0), (0.4, 100.0), (0.6, 74.0), (0.9, 0.0))


def measured_br24():
    D = np.genfromtxt(os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv"), delimiter=",", names=True)[:-1]
    A, H = 0.5 * (D["A1"] + D["A2"]), 0.5 * (D["H1"] + D["H2"])
    k, f = int(np.argmin(A)), int(np.argmax(A))
    a, h = A[k:f + 1], H[k:f + 1]
    return (a - a[0]) / (a[-1] - a[0]), h / h.max()


def model(table):
    gate, force = table[:, 1], table[:, 8]
    i0, i1 = int(np.argmax(gate)), int(np.argmin(gate))       # catch (largest angle) to finish
    g, F = gate[i0:i1 + 1], force[i0:i1 + 1]
    return (g[0] - g) / (g[0] - g[-1]), F / F.max()


def main():
    paths = sys.argv[1:] or [os.path.join(ROOT, "data", "local", "validation", "flex_gap_br24_kappa0.1.npz")]
    rows = {"[BR24] measured": measured_br24()}
    for p in paths:
        d = np.load(p)
        for lab in d.files:
            rows["%s %s" % (os.path.basename(p)[:-4], lab)] = model(d[lab])
    print("drive length | RBN 2023/12 | " + " | ".join(rows))
    for q, b in BIOROW_2023:
        print("   %3.0f %%     |   %3.0f %%    | " % (100 * q, b) +
              " | ".join("%5.0f %%" % (100 * np.interp(q, x, y)) for x, y in rows.values()))
    for name, (x, y) in rows.items():
        print("%s: peak at %.0f %% of drive length (population 40 %%)" % (name, 100 * x[np.argmax(y)]))


if __name__ == "__main__":
    main()
