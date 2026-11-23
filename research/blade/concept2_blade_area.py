"""Scull blade areas from Concept2's face-on product photographs, scaled by Concept2's published dimensions (§259).

Concept2 publishes each blade's outline (length, width at the tip, widest width; concept2.com/oars/sculls/blades) but
not its area. Its product photographs show each blade face-on, white on a plain blue ground. Per image:

* the blade mask: pixels markedly brighter in red than the blue ground (the dark shaft is excluded by brightness);
* the pixel scale from the published blade length (the mask's extent along the shaft, which is horizontal in the
  photographs), checked against the published widest width (the mask's largest extent across): the two scales agree
  when the photograph is face-on and the published dimensions are of the same outline;
* the projected area = mask pixels x scale^2, and the fill fraction area / (length x widest width), which carries the
  shape to a blade of the same family at other dimensions.

The Smoothie2 scull is not photographed face-on on the page (its images are sweep blades). Its area is estimated as the
Smoothie2 sweep photograph's fill fraction on the Smoothie2 scull's published outline (46 x 21.5 cm, Plain Edge), with
the Big Blade scull's fill fraction as the cross-check: the Plain Edge scull has the Big Blade's widths (21.5 / 21.5 cm)
and 2 cm more length. The Vortex Edge scull (tip 16.5 cm) is narrower at the tip and so smaller; it has no face-on
photograph and is bracketed below the Plain Edge.

Images (downloaded 2026-11-23 with the user's permission): data/local/literature/concept2_blades/.

    python -I research/blade/concept2_blade_area.py
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IMG = os.path.join(ROOT, "data", "local", "literature", "concept2_blades")

#: Concept2's published scull blade dimensions, cm: (length, width at tip, widest width)
PUBLISHED = {"Big Blade": (44.0, 21.5, 21.5), "Fat2": (46.0, 16.5, 23.0), "Comp": (37.0, 16.5, 23.0),
             "Smoothie2 Plain Edge": (46.0, 21.5, 21.5), "Smoothie2 Vortex Edge": (46.0, 16.5, 21.5)}
PHOTOS = {"Big Blade": "big_blade_scull.jpg", "Fat2": "fat2_scull.jpg", "Comp": "comp_scull.jpg",
          "Smoothie2 sweep (shape only)": "smoothie2_pe_sweep.jpg"}


def mask_of(path):
    a = np.asarray(Image.open(path).convert("RGB"), float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 120) & (g > 120)                       # white / grey blade; the ground is (17, 74, 189)-like blue


def measure(path):
    m = mask_of(path)
    # the shaft's highlight stripe is ~10-25 px tall; the blade's columns are far taller
    cols = np.flatnonzero(m.sum(axis=0) > 40)
    x0, x1 = cols[0], cols[-1]
    m[:, :x0] = False
    length_px = x1 - x0 + 1
    widths = []
    for x in range(x0, x1 + 1):
        ys = np.flatnonzero(m[:, x])
        widths.append(ys[-1] - ys[0] + 1 if ys.size else 0)
    # "width at the tip": across the blade just inside its tip (the tip edge is slanted, so at 97 % of the length)
    return dict(px=int(m.sum()), length_px=int(length_px), widest_px=int(max(widths)),
                tip_px=int(widths[int(0.97 * (len(widths) - 1))]))


def main():
    print("Concept2 scull blades: projected face area from the photographs, scaled by the published length")
    print("%-28s %8s %8s | %10s %10s | %9s %6s" % ("blade", "L cm", "W cm", "W meas cm", "tip meas", "area m^2",
                                                   "fill"))
    fills = {}
    for name, f in PHOTOS.items():
        r = measure(os.path.join(IMG, f))
        if name in PUBLISHED:
            L, tip, W = PUBLISHED[name]
            s = L / r["length_px"]                                  # cm per pixel
            area = r["px"] * s * s / 1e4
            fill = area * 1e4 / (L * W)
            fills[name] = fill
            print("%-28s %8.1f %8.1f | %10.1f %10.1f | %9.4f %6.3f"
                  % (name, L, W, r["widest_px"] * s, r["tip_px"] * s, area, fill))
        else:
            fill = r["px"] / (r["length_px"] * r["widest_px"])
            fills[name] = fill
            print("%-28s %8s %8s | aspect L/W %.2f              | %9s %6.3f"
                  % (name, "-", "-", r["length_px"] / r["widest_px"], "-", fill))
    L, _tip, W = PUBLISHED["Smoothie2 Plain Edge"]
    a_sm = fills["Smoothie2 sweep (shape only)"] * L * W / 1e4
    a_bb = fills["Big Blade"] * L * W / 1e4
    print("\nSmoothie2 Plain Edge scull (46 x 21.5 cm): %.4f m^2 on the Smoothie2 sweep's fill, %.4f on the Big Blade "
          "scull's" % (a_sm, a_bb))
    print("Smoothie2 Vortex Edge scull: below these (tip 16.5 cm against 21.5); no face-on photograph")
    print("model's catalogue scull: 0.083 m^2 (assumed)")


if __name__ == "__main__":
    sys.exit(main())
