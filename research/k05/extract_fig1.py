"""Digitise [K05] Fig. 1's on-water ("Boat") drive curves from the scanned paper.

    python research/k05/extract_fig1.py [path/to/k05.pdf]

Kleshnev (2005), ISBS XXIII 130-133, Fig. 1: average patterns of five female scullers at racing
rate (1.80 m, 72.2 kg; 32.3 spm on the water), against drive length (%). The page is a 300 dpi
grayscale scan: the on-water curves are black, the two machines grey, so the black curve is the
darkest ink. Per panel the drive branch is the upper one (positive velocity), traced as the
topmost black run in each pixel column, with titles and legends masked.

Calibration: y from the panel's zero line and the spacing of its printed labels (a linear fit of
the label centres); x from 0% at the y-axis to 100% at the curves' far end (the loop closes at
the end of the drive), which puts the printed "20%" and "40%" within ~8 px (1.2%) of where they
are drawn.

Checks printed, none fitted: legs + trunk + arms against the handle speed, and the segment
travels the curves imply (travel = integral of v / v_handle over the handle path) against
[K05]'s own table (legs 0.51, trunk 0.48, arms 0.62 m over a 1.59 m drive).

Output: data/literature/k05_fig1_onwater.csv, one row per 1% of drive length.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.path.join(ROOT, "data", "local", "literature", "k05.pdf")
OUT = os.path.join(ROOT, "data", "literature", "k05_fig1_onwater.csv")

#: per panel: y-axis column, the panel's right frame, zero-line row, pixels per m/s (from the label fits), rows
#: that may carry data (titles and legends outside), and masked boxes (x0, x1, y0, y1)
PANELS = {
    # force panel: scale in px per newton (label fit 800 ... 0 N); its drive end dips below zero,
    # so the far end is looked for down to 60 px under the zero line
    "handle_force": dict(axis=984, right=1655, zero=1674.6, scale=0.51592, rows=(1250, 1672),
                         mask=[(1010, 1330, 1185, 1250), (1340, 1650, 1185, 1345),
                               (1340, 1560, 1600, 1660)], below=60),
    "handle_speed": dict(axis=1780, right=2445, zero=1468.0, scale=86.5, rows=(1255, 1466),
                         mask=[(2020, 2330, 1340, 1440)]),
    "legs_velocity": dict(axis=974, right=1655, zero=2050.0, scale=168.0, rows=(1840, 2048),
                          mask=[(1345, 1640, 1820, 1990)]),
    "trunk_velocity": dict(axis=1769, right=2445, zero=2050.0, scale=168.0, rows=(1840, 2048), mask=[]),
    "arms_velocity": dict(axis=974, right=1655, zero=2684.0, scale=153.6, rows=(2420, 2682), mask=[]),
}
BLACK = 80


def page_image(pdf, tmp):
    stem = os.path.join(tmp, "k05")
    subprocess.run(["pdfimages", "-png", "-f", "3", "-l", "3", pdf, stem], check=True)
    return np.asarray(Image.open(stem + "-000.png").convert("L")).astype(int)


def trace(a, axis, right, zero, scale, rows, mask, below=None):
    ink = a < BLACK
    for x0, x1, y0, y1 in mask:
        ink[y0:y1, x0:x1] = False
    band = ink[rows[0]:rows[1], :]
    # the drive ends where the loop closes: the far end of either branch, which may sit on or
    # below the zero line, so look at both branches (down to 1.2 units below zero)
    both = ink[rows[0]:int(zero + (1.2 * scale if below is None else below)), :]
    cols = np.flatnonzero(both.any(axis=0))
    cols = cols[(cols > axis + 2) & (cols < right)]          # inside this panel's frame
    right = cols.max()
    xs, vs = [], []
    for x in range(axis + 3, right + 1):
        ys = np.flatnonzero(band[:, x])
        if ys.size == 0:
            continue
        # topmost run of black: the drive branch
        top = ys[0]
        run = ys[ys <= top + 8]
        y = rows[0] + float(run.mean())
        xs.append(100.0 * (x - axis) / (right - axis))
        vs.append((zero - y) / scale)
    return np.array(xs), np.array(vs)


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else PDF
    with tempfile.TemporaryDirectory() as tmp:
        a = page_image(pdf, tmp)
    grid = np.arange(0.0, 101.0, 1.0)
    table = {"length_pct": grid}
    for name, p in PANELS.items():
        x, v = trace(a, **p)
        order = np.argsort(x)
        table[name] = np.interp(grid, x[order], v[order])
    h = table["handle_speed"]
    total = table["legs_velocity"] + table["trunk_velocity"] + table["arms_velocity"]
    inner = (grid >= 10) & (grid <= 90)
    print("segments sum against handle speed, 10-90%% of the drive: rms %.3f m/s, mean %+.3f m/s"
          % (np.sqrt(np.mean((total - h)[inner] ** 2)), np.mean((total - h)[inner])))
    f = table["handle_force"]
    k = int(np.argmax(f))
    ok_v = h > 0.2
    dt = np.where(ok_v, 1.0 / np.where(ok_v, h, 1.0), 0.0)
    print("handle force: max %.0f N at %.1f%% of length ([K05] table 602 N at 34.7%%)" % (f[k], grid[k]))
    print("  average / max over length %.1f%%, over time %.1f%% ([K05] table 56.9%%)"
          % (100 * f.mean() / f[k], 100 * np.sum(f * dt) / np.sum(dt) / f[k]))
    length = 1.59
    ok = h > 0.2
    for seg, printed in (("legs_velocity", 0.51), ("trunk_velocity", 0.48), ("arms_velocity", 0.62)):
        travel = np.trapezoid(np.where(ok, table[seg] / np.where(ok, h, 1.0), 0.0), grid / 100.0 * length)
        print("  %-15s travel over the drive %.3f m ([K05] table %.2f)" % (seg, travel, printed))
    dt = np.trapezoid(np.where(ok, 1.0 / np.where(ok, h, 1.0), 0.0), grid / 100.0 * length)
    print("  drive time implied by the handle speed: %.3f s" % dt)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("# Kleshnev (2005) ISBS XXIII 130-133, Fig. 1, on-water ('Boat') drive branch: five female\n"
                "# scullers, 1.80 m, 72.2 kg, racing rate (32.3 spm), drive length 1.59 m. Digitised from the\n"
                "# 300 dpi scan by research/k05/extract_fig1.py (darkest ink, topmost run per column); velocities\n"
                "# in m/s against drive length in %; about +-0.03 m/s from the pixel scale and the line width.\n"
                "# handle_force_N: the on-water handle force panel, same digitisation, about +-10 N.\n")
        f.write("length_pct,handle_speed,legs_velocity,trunk_velocity,arms_velocity,handle_force_N\n")
        for i in range(grid.size):
            f.write("%.0f,%.3f,%.3f,%.3f,%.3f,%.1f\n" % tuple(table[k][i] for k in
                    ("length_pct", "handle_speed", "legs_velocity", "trunk_velocity", "arms_velocity",
                     "handle_force")))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
