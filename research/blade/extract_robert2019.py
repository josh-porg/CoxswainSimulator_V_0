"""Extract a full-size blade's measured load on a known motion from Robert et al. (2019), vector figures.

    .venv/Scripts/python.exe research/blade/extract_robert2019.py

Robert, Leroyer, Barre, Queutey & Visonneau (2019), *Validation of CFD simulations of the flow around a full-scale
rowing blade with realistic kinematics*, J. Mar. Sci. Technol. 24, 1105-1118 (open, HAL hal-02394848, CC BY 4.0).
A real sweep oar (hatchet blade) held at the handle and the oarlock (inboard 0.800 m) on a motor-driven arm on a
towing carriage; trial 154: carriage 4.202 m/s, immersion angle 9.5 deg, roll 6 deg. Six-component forces at 500 Hz,
low-passed at 11 Hz, inertial and aerodynamic parts removed. Two figures:

- Fig. 3, p. 4: the imposed angular velocity of the oar against its sweep angle psi_0 (0 = along the carriage's
  travel, 90 = square).
- Fig. 7, p. 10: F_x (propulsive, carriage frame), F_y (transverse), F_z and the moment M_z about the rotation axis,
  against psi_0, for trial 154: the experiment (solid, 2198 segments) and three CFD grids.

The tick labels are outlined glyphs, not text, so the axes are calibrated from the gridlines with the printed tick
values (read from the rendered figure), and each calibration is checked against the label glyphs' positions.

Writes data/literature/robert2019_trial154.csv (series, psi0_deg, value).
"""
from __future__ import annotations

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.path.join(ROOT, "data", "local", "literature", "robert2019", "robert2019.pdf")
OUT = os.path.join(ROOT, "data", "literature", "robert2019_trial154.csv")

# page index, drawing index of the curve, (x_page_a, x_val_a, x_page_b, x_val_b), (y_page_a, y_val_a, y_page_b, y_val_b)
# Fig. 3: grid every 10 deg from 0 (x 383.40) to 180 (489.87); labels 0..3 rad/s at y 157.09, 130.66, 104.21, 77.76.
# Fig. 7: x labels 50, 150 deg at the gridlines x 249.0 / 349.7 (left panels), 433.9 / 534.6 (right panels).
PANELS = {
    "psi0dot_radps": (3, 308, (383.40, 0.0, 489.87, 180.0), (157.12, 0.0, 77.76, 3.0)),
    "Fx_N": (9, 10, (249.0, 50.0, 349.7, 150.0), (187.5, 0.0, 96.8, 200.0)),
    "Fy_N": (9, 76, (433.9, 50.0, 534.6, 150.0), (176.3, -50.0, 96.2, 50.0)),
    "Fz_N": (9, 129, (249.0, 50.0, 349.7, 150.0), (359.2, 0.0, 271.6, 20.0)),
    "Mz_Nm": (9, 184, (433.9, 50.0, 534.6, 150.0), (251.1, 0.0, 356.1, -400.0)),
}
FRAMES = {"Fx_N": 3, "Fy_N": 68, "Fz_N": 122, "Mz_Nm": 175, "psi0dot_radps": 299}


def lin(a_page, a_val, b_page, b_val):
    k = (b_val - a_val) / (b_page - a_page)
    return lambda p: a_val + k * (np.asarray(p) - a_page)


def main():
    import pymupdf
    doc = pymupdf.open(PDF)
    rows = []
    for name, (page, idx, xc, yc) in PANELS.items():
        drawings = doc[page].get_drawings()
        dr = drawings[idx]
        frame = drawings[FRAMES[name]]["rect"]
        pts = np.array([(it[1].x, it[1].y) for it in dr["items"]] + [(dr["items"][-1][2].x, dr["items"][-1][2].y)])
        inside = (pts[:, 0] >= frame.x0 - 0.5) & (pts[:, 0] <= frame.x1 + 0.5)
        pts = pts[inside]
        fx, fy = lin(*xc), lin(*yc)
        x, y = fx(pts[:, 0]), fy(pts[:, 1])
        o = np.argsort(x, kind="stable")
        x, y = x[o], y[o]
        keep = np.concatenate([[True], np.diff(x) > 1e-6])
        x, y = x[keep], y[keep]
        rows += [(name, a, b) for a, b in zip(x, y)]
        print("%-14s %4d points, psi0 %.1f-%.1f deg, value %.1f to %.1f" % (name, x.size, x[0], x[-1], y.min(), y.max()))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Robert, Leroyer, Barre, Queutey & Visonneau (2019) J. Mar. Sci. Technol. 24:1105-1118 (CC BY 4.0, HAL\n"
                "# hal-02394848). Fig. 3: imposed oar angular velocity (rad/s) against sweep angle; Fig. 7: trial 154's MEASURED\n"
                "# F_x (propulsive), F_y (transverse), F_z (N) and M_z about the rotation axis (N m) against sweep angle psi_0\n"
                "# (0 = along the carriage's travel). Carriage 4.202 m/s, immersion 9.5 deg, roll 6 deg; inboard 0.800 m,\n"
                "# oarlock to blade root 1.558 m. Extracted from the vector figures by research/blade/extract_robert2019.py;\n"
                "# axes from the gridlines and printed tick values; inside each frame only (Fig. 7: 20-150 deg).\n"
                "series,psi0_deg,value\n")
        for s, a, b in rows:
            f.write("%s,%.4f,%.4f\n" % (s, a, b))
    print("written %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
