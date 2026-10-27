"""Extract a realistic blade motion and its measured forces from Grift, Tummers & Westerweel (2021), vector figures.

    .venv/Scripts/python.exe research/blade/extract_grift2021.py

Grift, Tummers & Westerweel (2021), *Hydrodynamics of rowing propulsion*, J. Fluid Mech. 918, A29 (open access).
A 1:2 scale sweep blade driven by a robot along an on-water blade path (a men's coxless four at standard pace),
in still tank water, with the hydrodynamic force measured (ATI 6-DOF transducer). The motion is known exactly,
so it tests a blade law with no uncertainty in the boat's speed through the water. Three figures, all vector:

- Fig. 4(b), p. 8: oar angle theta against time t, full scale timing; "M4- standard" (solid) is the case run.
- Fig. 7(b), p. 12: the blade tip's x and y against t as the robot reproduced them (black; scaled 1:2).
- Fig. 9(a), p. 14: F_n and F_t against t* (the drive, 0 to 1), velocity scaling kappa = 1.00; three realisations
  (solid, dashed, dotted), which the paper says overlap almost perfectly.

Each panel's axes are calibrated from its printed tick labels (the words on the page) against their positions, not
from the frame. Curves are the stroked paths inside the panel, picked by colour and dash pattern.

Writes data/literature/grift2021_kinematics.csv (series, t_s, value) and grift2021_forces.csv (series, t_star,
value_N). Acceptance checks, printed: the catch near theta = -50 deg and the release near +28 deg (text, sec. 2),
the tip path length near 0.96 m at scale (sec. 3.3), and the three force realisations agreeing.
"""
from __future__ import annotations

import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.path.join(ROOT, "data", "local", "literature", "grift2021", "grift2021_jfm.pdf")
OUT = os.path.join(ROOT, "data", "literature")
NUM = re.compile(r"^[−\-–]?\d+(\.\d+)?$")


def number(s):
    return float(s.replace("−", "-").replace("–", "-"))


def panel_axes(page, box, xbox=None):
    """Linear maps (page x -> value, page y -> value) from the tick labels beside a panel's frame. ``xbox``: the
    panel whose x tick labels to use, for a panel that shares its time axis with the one below it."""
    xbox = box if xbox is None else xbox
    words = page.get_text("words")
    left, bottom = [], []
    for x0, y0, x1, y1, w, *_ in words:
        if not NUM.match(w):
            continue
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if x1 <= box.x0 + 1 and box.x0 - 30 < x1 and box.y0 - 6 <= cy <= box.y1 + 6:
            left.append((cy, number(w)))
        if y0 >= xbox.y1 - 1 and y0 < xbox.y1 + 14 and xbox.x0 - 8 <= cx <= xbox.x1 + 8:
            bottom.append((cx, number(w)))
    left = sorted(set(left)); bottom = sorted(set(bottom))
    if len(left) < 3 or len(bottom) < 3:
        raise RuntimeError("tick labels not found for panel %s: %s %s" % (box, left, bottom))
    py, vy = np.array(left).T
    px, vx = np.array(bottom).T
    ay = np.polyfit(py, vy, 1); ax = np.polyfit(px, vx, 1)
    res = max(np.max(np.abs(np.polyval(ay, py) - vy)), np.max(np.abs(np.polyval(ax, px) - vx)))
    return (lambda x: np.polyval(ax, x)), (lambda y: np.polyval(ay, y)), float(res), (vx.min(), vx.max()), (vy.min(), vy.max())


def curves(page, box, colour, dashes):
    """Points of the stroked paths inside ``box`` with this colour and dash pattern, in drawing order."""
    out = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if dr.get("fill") is not None or not dr.get("color"):
            continue
        if not (r.x0 >= box.x0 - 1 and r.x1 <= box.x1 + 1 and r.y0 >= box.y0 - 1 and r.y1 <= box.y1 + 1):
            continue
        if len(dr["items"]) < 20:
            continue                                             # legend samples, ticks, arrows
        if tuple(round(c, 2) for c in dr["color"]) != colour:
            continue
        if (dr.get("dashes") or "").replace(" ", "") != dashes.replace(" ", ""):
            continue
        pts = []
        for it in dr["items"]:
            if it[0] == "l":
                pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
            elif it[0] == "c":
                pts += [(it[1].x, it[1].y), (it[4].x, it[4].y)]
        out.append(np.array(pts))
    return out


def frame(page, near):
    """The panel frame (the white filled box) whose top-left is nearest ``near``."""
    boxes = [dr["rect"] for dr in page.get_drawings() if dr.get("fill") and len(dr["items"]) == 1
             and dr["rect"].width > 100]
    return min(boxes, key=lambda b: abs(b.x0 - near[0]) + abs(b.y0 - near[1]))


def series(page, box, colour, dashes, which=0, xbox=None):
    fx, fy, res, xr, yr = panel_axes(page, box, xbox)
    cs = curves(page, box, colour, dashes)
    if len(cs) <= which:
        raise RuntimeError("curve %s %s not found" % (colour, dashes))
    p = cs[which]
    x, y = fx(p[:, 0]), fy(p[:, 1])
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    keep = np.concatenate([[True], np.diff(x) > 1e-6])
    return x[keep], y[keep], res


def main():
    import pymupdf
    doc = pymupdf.open(PDF)
    rows_k, rows_f = [], []
    # Fig. 4(b): oar angle, M4- standard (solid)
    pg = doc[7]
    box = frame(pg, (285, 67))
    t, th, res = series(pg, box, (0.0, 0.0, 0.0), "[] 0")
    rows_k += [("theta_deg", a, b) for a, b in zip(t, th)]
    print("Fig. 4(b) theta: %d points, t %.3f-%.3f s, catch %.1f, release %.1f deg (text: about -50 and +28); "
          "tick fit %.3g" % (t.size, t[0], t[-1], th[0], th[-1], res))
    # Fig. 7(b): tip x (solid) and y (dashed), robot (black)
    pg = doc[11]
    box = frame(pg, (283, 66))
    tx, x, res1 = series(pg, box, (0.0, 0.0, 0.0), "[] 0")
    ty, y, res2 = series(pg, box, (0.0, 0.0, 0.0), "[ 4.0989 4.0989 ] 0")
    rows_k += [("tip_x_m_scaled", a, b) for a, b in zip(tx, x)] + [("tip_y_m_scaled", a, b) for a, b in zip(ty, y)]
    g = np.linspace(max(tx[0], ty[0]), min(tx[-1], ty[-1]), 2000)
    L = np.sum(np.hypot(np.diff(np.interp(g, tx, x)), np.diff(np.interp(g, ty, y))))
    print("Fig. 7(b) tip: x %d, y %d points, t to %.3f s; path length %.3f m at scale (text: 0.96); tick fit %.3g %.3g"
          % (tx.size, ty.size, min(tx[-1], ty[-1]), L, res1, res2))
    # Fig. 9(a): F_n (black) and F_t (grey), three realisations
    pg = doc[13]
    box = frame(pg, (103, 67))
    below = frame(pg, (103, 172))                                # panel (c): the shared time axis's labels
    got = {}
    for lab, dash in (("1", "[] 0"), ("2", "[ 5.7948 3.4769 ] 0"), ("3", "[ .57948 1.7385 ] 0")):
        for comp, col in (("Fn", (0.0, 0.0, 0.0)), ("Ft", (0.5, 0.5, 0.5))):
            ts, f, res = series(pg, box, col, dash, xbox=below)
            got[comp + lab] = (ts, f)
            rows_f += [("%s_N_run%s" % (comp, lab), a, b) for a, b in zip(ts, f)]
    g = np.linspace(0.0, 1.0, 400)
    fn = np.array([np.interp(g, *got["Fn%d" % i]) for i in (1, 2, 3)])
    ft = np.array([np.interp(g, *got["Ft%d" % i]) for i in (1, 2, 3)])
    print("Fig. 9(a) forces: F_n min %.1f N at t* %.2f; realisations differ by rms %.2f N (F_n), %.2f N (F_t); "
          "impulse over the drive F_n %.2f, F_t %.2f N (t* units)"
          % (fn.mean(0).min(), g[np.argmin(fn.mean(0))], np.sqrt(np.mean(fn.std(0) ** 2)),
             np.sqrt(np.mean(ft.std(0) ** 2)), np.trapezoid(fn.mean(0), g), np.trapezoid(ft.mean(0), g)))
    head = ("# Grift, Tummers & Westerweel (2021) J. Fluid Mech. 918 A29, open access. Extracted from the vector figures by\n"
            "# research/blade/extract_grift2021.py; axes calibrated from the printed tick labels.\n")
    with open(os.path.join(OUT, "grift2021_kinematics.csv"), "w", encoding="utf-8") as f:
        f.write(head + "# Fig. 4(b): oar angle theta (deg, M4- standard, full-scale timing); Fig. 7(b): blade tip x, y (m, 1:2 scale,\n"
                "# robot-reproduced, x along the boat's motion). Time t in s from the catch.\nseries,t_s,value\n")
        for s, a, b in rows_k:
            f.write("%s,%.5f,%.5f\n" % (s, a, b))
    with open(os.path.join(OUT, "grift2021_forces.csv"), "w", encoding="utf-8") as f:
        f.write(head + "# Fig. 9(a): measured F_n and F_t (N, 1:2 scale blade, kappa = 1.00) against t* (0 catch, 1 release),\n"
                "# three realisations. Sign as printed: F_n negative while the blade drives.\nseries,t_star,value_N\n")
        for s, a, b in rows_f:
            f.write("%s,%.5f,%.5f\n" % (s, a, b))
    print("written grift2021_kinematics.csv, grift2021_forces.csv")


if __name__ == "__main__":
    main()
