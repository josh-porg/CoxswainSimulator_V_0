"""Extract the Nantes towing-tank measurements of a flat blade from Leroyer et al. (12th Journees de l'Hydrodynamique).

    .venv/Scripts/python.exe research/blade/extract_leroyer_jh12.py

Leroyer, Barre, Wackers, Kobus & Queutey, *Vers le calcul des efforts sur les palettes d'aviron en fonctionnement
reel* (open, French). The experiment is the LMF towing tank's (the benchmark of Barre & Kobus 2010, Proc. IMechE P
224:37-50, and of Leroyer et al. 2010): a flat plate on a RIGID shaft, aspect ratio 0.46, area S_b = 420 cm^2 (a
scull blade at scale 0.7), its top edge at the undisturbed surface, carriage V_b = 2.91 m/s, the blade swept by
theta_dot = K sin(theta) with K = 3 rad/s, so the advance parameter eta_0 = V_b sin(theta) / (L_e theta_dot) is held at
0.742 (Froude 0.59 at 90 deg by their definition; the real-stroke values 2.51 rad/s and 3.48 m/s scaled at constant
Froude and eta_0). Sec. III-2.1. The figures (p. 10, vector; the experiment is the heavy solid line, "Exp."):

- Fig. 8: C_l against the pseudo-incidence i at the blade centre I (the angle between -V_I and the shaft);
- Fig. 9: C_d against i (C = F / (1/2 rho S_b |V_I|^2));
- Fig. 10: F_y (transverse, carriage frame) against theta;  Fig. 11: F_x (propulsive) against theta.

Axes from the printed tick labels (text on this page). Writes data/literature/leroyer_jh12_flat_blade.csv.
"""
from __future__ import annotations

import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.path.join(ROOT, "data", "local", "literature", "leroyer_jh12", "leroyer_jh12.pdf")
OUT = os.path.join(ROOT, "data", "literature", "leroyer_jh12_flat_blade.csv")
NUM = re.compile(r"^-?\d+(\.\d+)?$")

# name: (x range of the panel incl. its labels, y range, x-axis variable)
PANELS = {
    "Cl": ((80, 280), (65, 240), "i_deg"),
    "Cd": ((320, 525), (65, 240), "i_deg"),
    "Fy_N": ((80, 285), (285, 455), "theta_deg"),
    "Fx_N": ((320, 525), (285, 455), "theta_deg"),
}


def axes(words, xr, yr):
    nums = [(0.5 * (w[0] + w[2]), 0.5 * (w[1] + w[3]), float(w[4]), w) for w in words
            if NUM.match(w[4]) and xr[0] <= w[0] <= xr[1] and yr[0] <= w[1] <= yr[1]]
    ybot = max(n[1] for n in nums)                                # the x tick row is the lowest row of numbers
    xt = sorted((n[0], n[2]) for n in nums if abs(n[1] - ybot) < 2.0)
    xleft = min(n[3][2] for n in nums if abs(n[1] - ybot) >= 2.0)
    yt = sorted((n[1], n[2]) for n in nums if abs(n[1] - ybot) >= 2.0 and abs(n[3][2] - xleft) < 1.5)
    px, vx = np.array(xt).T
    py, vy = np.array(yt).T
    cx, cy = np.polyfit(px, vx, 1), np.polyfit(py, vy, 1)
    res = max(np.abs(np.polyval(cx, px) - vx).max() / np.ptp(vx), np.abs(np.polyval(cy, py) - vy).max() / np.ptp(vy))
    return cx, cy, res, (px.min(), px.max(), py.min(), py.max())


def main():
    import pymupdf
    page = pymupdf.open(PDF)[9]
    words = page.get_text("words")
    drawings = page.get_drawings()
    rows = []
    for name, (xr, yr, xvar) in PANELS.items():
        cx, cy, res, (x0, x1, y0, y1) = axes(words, xr, yr)
        pts = []
        for dr in drawings:
            r = dr["rect"]
            if (dr.get("width") or 0) < 0.85 or (dr.get("dashes") or "").replace(" ", "") != "[]0":
                continue
            if len(dr["items"]) < 50 or r.x0 < x0 - 3 or r.x1 > x1 + 3 or r.y0 < yr[0] or r.y1 > y1 + 3:
                continue
            for it in dr["items"]:
                if it[0] == "l":
                    pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        P = np.array(pts)
        x, y = np.polyval(cx, P[:, 0]), np.polyval(cy, P[:, 1])
        o = np.argsort(x, kind="stable")
        x, y = x[o], y[o]
        keep = np.concatenate([[True], np.diff(x) > 1e-6])
        x, y = x[keep], y[keep]
        rows += [(name, xvar, a, b) for a, b in zip(x, y)]
        k = np.argmax(np.abs(y))
        print("%-5s %4d points, %s %.0f-%.0f; extreme %.2f at %.0f; tick fit %.4f of range" % (
            name, x.size, xvar, x[0], x[-1], y[k], x[k], res))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Leroyer, Barre, Wackers, Kobus & Queutey, 12th Journees de l'Hydrodynamique (open), Figs 8-11, the\n"
                "# EXPERIMENT (LMF towing tank): a flat blade, AR 0.46, S_b 420 cm^2, rigid shaft, top edge at the surface,\n"
                "# V_b 2.91 m/s, theta_dot = 3 sin(theta) rad/s, eta_0 = 0.742 (L_e = V_b / (K eta_0) = 1.307 m). C_l, C_d against\n"
                "# the pseudo-incidence at the blade centre; F_y, F_x (N, carriage frame) against theta. Extracted by\n"
                "# research/blade/extract_leroyer_jh12.py from the vector figure; axes from the printed tick labels.\n"
                "series,x_variable,x,value\n")
        for s, v, a, b in rows:
            f.write("%s,%s,%.4f,%.5f\n" % (s, v, a, b))
    print("written %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
