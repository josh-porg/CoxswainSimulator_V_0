"""Extract the measured traces of [CR06] Fig. 3 from the paper's vector figure.

    python research/cr06/extract_fig3.py

Cabrera, Ruina & Kleshnev (2006), *A simple 1+ dimensional model of rowing mimics observed
forces and motions*, Human Movement Science 25(2) 192-220. The open preprint
(ruina.tam.cornell.edu/.../simple_1plus_d_model.pdf) carries Fig. 3 on page 16 as vector
PostScript: a women's single, one stroke, measured (heavy lines) against the model (thin).

Five panels: boat velocity, handle force, leg displacement x_B/F, oar angle, back
displacement x_S/B. The page is converted to SVG with poppler's pdftocairo; each panel's
frame comes from its dashed grid, its time axis runs 0-2 s across the frame, and its value
axis is calibrated from the tick marks against the printed tick labels. Measured curves
are the stroke-width-10 paths.

Writes data/literature/cr06_fig3_measured.csv (series, t_s, value). Acceptance check:
mean boat speed over the stroke, recorded as 4.19 m/s in SOURCES against the paper's
stated 4.18.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.path.join(ROOT, "data", "local", "literature", "cabrera_ruina_kleshnev_2006.pdf")
OUT = os.path.join(ROOT, "data", "literature", "cr06_fig3_measured.csv")
PAGE = 16
T_STROKE = 1.94

# printed tick labels, bottom to top, per panel (identified by the frame's position)
PANELS = {
    "boat_velocity_mps": dict(ticks=[0, 1, 2, 3, 4, 5], where=("left", "top")),
    "handle_force_N": dict(ticks=[-100, 0, 100, 200, 300, 400, 500, 600], where=("right", "top")),
    "leg_disp_m": dict(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], where=("left", "middle")),
    "oar_angle_deg": dict(ticks=[-60, -40, -20, 0, 20, 40, 60], where=("right", "middle")),
    "back_disp_m": dict(ticks=[0, 0.1, 0.2, 0.3, 0.4], where=("centre", "bottom")),
}


def svg_paths():
    with tempfile.TemporaryDirectory() as tmp:
        svg = os.path.join(tmp, "p.svg")
        subprocess.run(["pdftocairo", "-svg", "-f", str(PAGE), "-l", str(PAGE), PDF, svg], check=True)
        text = open(svg, encoding="utf-8").read()
    out = []
    for p in re.findall(r"<path ([^>]*)/>", text):
        if "transform" not in p:
            continue                       # glyph outlines
        d = re.search(r' d="([^"]+)"', p).group(1)
        width = float(re.search(r'stroke-width="([\d.]+)"', p).group(1))
        dashed = "dasharray" in p
        segments = []
        for part in re.split(r"(?=M)", d):
            nums = [float(x) for x in re.findall(r"-?[\d.]+", part)]
            if len(nums) >= 2:
                segments.append(np.array(nums).reshape(-1, 2))
        out.append((width, dashed, segments))
    return out


def frames(paths):
    """Panel frames from the dashed grid: (x0, x1, y0, y1) in raw units, y up."""
    h, v = [], []
    for width, dashed, segs in paths:
        if not dashed:
            continue
        for s in segs:
            (xa, ya), (xb, yb) = s[0], s[-1]
            if abs(ya - yb) < 1e-6:
                h.append((min(xa, xb), max(xa, xb), ya))
            elif abs(xa - xb) < 1e-6:
                v.append((xa, min(ya, yb), max(ya, yb)))
    boxes = {}
    for x0, x1, y in h:
        key = (round(x0), round(x1))
        boxes.setdefault(key, []).append(y)
    result = []
    for (x0, x1), ys in boxes.items():
        ys = np.array(ys)
        cols = [c for c in v if x0 - 1 <= c[0] <= x1 + 1
                and c[1] <= ys.max() + 1 and c[2] >= ys.min() - 1]
        y0 = min(c[1] for c in cols) if cols else ys.min()
        y1 = max(c[2] for c in cols) if cols else ys.max()
        result.append((float(x0), float(x1), float(y0), float(y1)))
    return result


def ticks_in(paths, frame):
    """Distinct y of the short horizontal tick marks on the frame's left edge."""
    x0, x1, y0, y1 = frame
    ys = []
    for width, dashed, segs in paths:
        if dashed:
            continue
        for s in segs:
            (xa, ya), (xb, yb) = s[0], s[-1]
            if len(s) == 2 and abs(ya - yb) < 1e-6 and y0 - 2 <= ya <= y1 + 2:
                if min(xa, xb) >= x0 - 1 and max(xa, xb) <= x0 + 60:
                    ys.append(ya)
    ys = np.unique(np.round(ys, 2))
    return ys


def classify(frame, allframes):
    """Column by the frame's centre (label widths shift the edges), row by its bottom."""
    x0, x1, y0, y1 = frame
    centres = [0.5 * (f[0] + f[1]) for f in allframes]
    lo, hi = min(centres), max(centres)
    cx = 0.5 * (x0 + x1)
    col = "left" if cx < lo + 0.25 * (hi - lo) else ("right" if cx > hi - 0.25 * (hi - lo) else "centre")
    ys = sorted({round(f[2]) for f in allframes})
    row = "bottom" if round(y0) == ys[0] else ("top" if round(y0) == ys[-1] else "middle")
    return col, row


def main():
    if not os.path.exists(PDF):
        sys.exit("fetch the preprint into %s first" % PDF)
    paths = svg_paths()
    fr = frames(paths)
    rows = []
    summary = {}
    for name, spec in PANELS.items():
        frame = next(f for f in fr if classify(f, fr) == spec["where"])
        x0, x1, y0, y1 = frame
        tick_y = ticks_in(paths, frame)
        if len(tick_y) != len(spec["ticks"]):
            sys.exit("%s: found %d ticks, expected %d" % (name, len(tick_y), len(spec["ticks"])))
        a, b = np.polyfit(tick_y, spec["ticks"], 1)
        resid = np.abs(np.polyval([a, b], tick_y) - spec["ticks"]).max()
        # the measured curve can be drawn as several sub-paths; the catch/release markers are
        # thinner filled circles, so every heavy segment inside the frame is the curve
        heavy = [s for w, dsh, segs in paths if w == 10.0 and not dsh for s in segs
                 if s[:, 0].min() >= x0 - 1 and s[:, 0].max() <= x1 + 1
                 and s[:, 1].min() >= y0 - 2 and s[:, 1].max() <= y1 + 2]
        if not heavy:
            sys.exit("%s: no measured curve found" % name)
        pts = np.vstack(heavy)
        pts = pts[np.argsort(pts[:, 0], kind="stable")]
        keep = np.concatenate([[True], np.diff(pts[:, 0]) > 1e-6])
        pts = pts[keep]
        if np.ptp(pts[:, 0]) < 0.9 * (x1 - x0) * T_STROKE / 2.0:
            sys.exit("%s: measured curve spans too little of the stroke" % name)
        t = (pts[:, 0] - x0) / (x1 - x0) * 2.0
        val = a * pts[:, 1] + b
        for tt, vv in zip(t, val):
            rows.append((name, tt, vv))
        summary[name] = (len(pts), t.min(), t.max(), val.min(), val.max(), resid)
    for k, (n, t0, t1, v0, v1, r) in summary.items():
        print("%-18s %3d points  t %.3f-%.3f s  value %.3f-%.3f  tick fit residual %.1e" % (k, n, t0, t1, v0, v1, r))
    tv = np.array([(t, v) for s, t, v in rows if s == "boat_velocity_mps"])
    o = np.argsort(tv[:, 0])
    tv = tv[o]
    mean_v = np.trapezoid(tv[:, 1], tv[:, 0]) / (tv[-1, 0] - tv[0, 0])
    print("mean boat speed %.3f m/s (recorded 4.19; the paper states 4.18)" % mean_v)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Cabrera, Ruina & Kleshnev (2006) Human Movement Science 25(2) 192-220, Fig. 3, measured curves\n")
        f.write("# (heavy lines): women's single, one stroke, T = 1.94 s. Extracted from the open preprint's vector\n")
        f.write("# figure (page 16) by research/cr06/extract_fig3.py; axes calibrated on the printed ticks.\n")
        f.write("# leg_disp_m = x_B/F (seat relative to foot), back_disp_m = x_S/B (shoulder relative to seat).\n")
        f.write("series,t_s,value\n")
        for s_, t_, v_ in rows:
            f.write("%s,%.5f,%.5f\n" % (s_, t_, v_))
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
