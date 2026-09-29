"""Digitise Coppel (2010)'s Big Blade figures from the thesis PDF (open access, etheses.bham.ac.uk/793).

    python research/coppel/extract_coppel.py [path/to/coppel.pdf]

Two figure pairs, both embedded rasters:

  Figs 3.19 / 3.20 (PDF page 105)  Caplan & Gardner's measured C_L and C_D markers for the curved
                                    Big Blade -> data/literature/cg07_bigblade_via_coppel2010.csv
  Figs 3.25 / 3.26 (PDF page 115)  CFD at quarter scale (0.75 m/s) and full size (5 m/s): printed
                                    here per computed angle; the CSV keeps the values that match
                                    Table 3.7's differences (the curves overlap where they agree)

Markers are found by a morphological opening that keeps filled discs and drops the thin curve
lines; legend markers are dropped by position. Axes are calibrated on the plot frame, whose
edges are the axis limits. Needs poppler's pdfimages on the path.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage as nd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF = os.path.join(ROOT, "data", "local", "literature", "coppel.pdf")


def images(pdf, page, tmp):
    stem = os.path.join(tmp, "p%d" % page)
    subprocess.run(["pdfimages", "-png", "-f", str(page), "-l", str(page), pdf, stem], check=True)
    return sorted(os.path.join(tmp, f) for f in os.listdir(tmp) if f.startswith("p%d" % page))


def frame(a):
    dark = a < 140
    h, w = a.shape
    rows = [r for r in range(h) if dark[r].sum() > 0.6 * w]
    cols = [c for c in range(w) if dark[:, c].sum() > 0.5 * h]
    return min(cols), max(cols), min(rows), max(rows)


def markers(path, vtop, vbot, legend_x):
    a = np.asarray(Image.open(path).convert("L")).astype(float)
    x0, x1, y0, y1 = frame(a)
    lab, n = nd.label(nd.binary_opening(a < 110, structure=np.ones((4, 4))))
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if not (3 <= np.ptp(ys) + 1 <= 14 and 3 <= np.ptp(xs) + 1 <= 14) or len(xs) < 30:
            continue
        ang = 180.0 * (xs.mean() - x0) / (x1 - x0)
        val = vtop + (vbot - vtop) * (ys.mean() - y0) / (y1 - y0)
        if abs(ang - round(ang / 5) * 5) > 1.5 or legend_x[0] < xs.mean() < legend_x[1] and ys.mean() < y0 + 0.35 * (y1 - y0):
            continue                                   # legend marker, or not on a tested angle
        out.append((round(ang), round(val, 3)))
    return sorted(out)


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else PDF
    with tempfile.TemporaryDirectory() as tmp:
        lift, drag = images(pdf, 105, tmp)[:2]
        print("Caplan & Gardner markers, curved Big Blade (Figs 3.19, 3.20)")
        print("  lift:", markers(lift, 3.0, -3.0, (380, 620)))
        print("  drag:", markers(drag, 3.0, 0.0, (500, 820)))


if __name__ == "__main__":
    main()
