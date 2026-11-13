"""Handle force through the drive: model, [BR24] and Kleshnev's population reference curve (SOURCES §234, §237).

RBN 2023/12 Fig. 1a (>50 000 samples) gives the population's handle force as % of its peak per 10 % of the drive's
**time** (rbn2023.py; the newsletter's text calls the axis "drive length", its figure "time of drive phase", and its
panels d-e settle it: §237 — §234-236 read it as length, which was wrong). Here every curve is put on both axes: time
directly, and drive length through each curve's own time-to-length relation (the population's from its handle
velocity). Reads the per-sample tables research/blade/flex_gap_diagnosis.py saves and [BR24]'s BioRow cycle.

    python research/validation/force_vs_drive_length.py [npz ...]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import rbn2023                                                   # noqa: E402


def measured_br24():
    """(time fraction, drive-length fraction, force / peak) of his measured drive, catch to finish by oar angle."""
    D = np.genfromtxt(os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv"), delimiter=",", names=True)[:-1]
    A, H = 0.5 * (D["A1"] + D["A2"]), 0.5 * (D["H1"] + D["H2"])
    k, f = int(np.argmin(A)), int(np.argmax(A))
    a, h = A[k:f + 1], H[k:f + 1]
    return np.linspace(0.0, 1.0, a.size), (a - a[0]) / (a[-1] - a[0]), h / h.max()


def model(table):
    t, gate, force = table[:, 0], table[:, 1], table[:, 8]
    i0, i1 = int(np.argmax(gate)), int(np.argmin(gate))           # catch (largest angle) to finish
    g, F, tt = gate[i0:i1 + 1], force[i0:i1 + 1], t[i0:i1 + 1] - t[i0]
    return tt / tt[-1], (g[0] - g) / (g[0] - g[-1]), F / F.max()


def main():
    paths = sys.argv[1:] or [os.path.join(ROOT, "data", "local", "validation", "flex_gap_br24_kappa0.1.npz")]
    ref = rbn2023.reference()
    rows = {"population": (ref["time"] / 100.0, ref["length"] / 100.0, ref["force"] / 100.0),
            "[BR24] measured": measured_br24()}
    for p in paths:
        d = np.load(p)
        for lab in d.files:
            rows["%s %s" % (os.path.basename(p)[:-4].replace("flex_gap_", ""), lab)] = model(d[lab])
    for axis, j, grid in (("time", 0, ref["time"] / 100.0), ("length", 1, np.arange(0.0, 1.01, 0.1))):
        print("\nhandle force, %% of peak, by drive %s:" % axis)
        print(" %6s | " % axis + " | ".join("%-22s" % n[:22] for n in rows))
        for q in grid:
            print("  %3.0f %% | " % (100 * q) + " | ".join("%22.0f" % (100 * np.interp(q, r[j], r[2])) for r in rows.values()))
    for name, (tf, dl, f) in rows.items():
        k = int(np.argmax(f))
        print("%s: peak at %.0f %% of the drive's time, %.0f %% of its length" % (name, 100 * tf[k], 100 * dl[k]))


if __name__ == "__main__":
    main()
