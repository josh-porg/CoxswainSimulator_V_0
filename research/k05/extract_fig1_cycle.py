"""Digitise the rest of [K05] Fig. 1 on water: the recovery branches and the boat's acceleration.

    python research/k05/extract_fig1_cycle.py [path/to/k05.pdf] [--overlay out.png]

extract_fig1.py traced the drive branch of each velocity panel (the topmost black run) and the
handle force. A whole stroke also needs the recovery: the same panels' lower loop, the bottom-most
black run per column, on the same length calibration (0% at the y-axis, 100% at the curves' far
end). Legends and axis titles are masked; where a mask covers the curve it is interpolated across.

The boat acceleration panel is against cycle time (%), single-valued; it is traced as the mean row
of the black ink per column (steep stretches are near-vertical). Its time origin is not the catch
(the catch dip falls near 27% of the cycle), so comparisons align the dip.

Checks printed, none fitted: each segment's travel over the recovery (integral of v / v_handle
along the handle's return) against its travel over the drive and [K05]'s table (legs 0.51, trunk
0.48, arms 0.62 m), and the acceleration's extremes against the table (min -7.92, max 3.39 m/s^2).

Output: data/literature/k05_fig1_recovery.csv, data/literature/k05_fig1_boat_acceleration.csv.
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

import extract_fig1 as E

REC_OUT = os.path.join(E.ROOT, "data", "literature", "k05_fig1_recovery.csv")
ACC_OUT = os.path.join(E.ROOT, "data", "literature", "k05_fig1_boat_acceleration.csv")

#: per velocity panel: rows the recovery branch may occupy, and extra masks (legends and axis
#: titles below the zero line), (x0, x1, y0, y1)
#: the tick labels ("20% ... 80%") sit just under the zero line, inside the recovery band;
#: found from the glyph clusters, padded 3 px
TICKS = {
    "handle_speed": (1494, 1529, [(1871, 1949), (1996, 2074), (2119, 2197), (2243, 2321)]),
    "legs_velocity": (2080, 2118, [(1068, 1153), (1196, 1282), (1323, 1408), (1452, 1537)]),
    "trunk_velocity": (2079, 2122, [(1863, 1941), (1991, 2068), (2116, 2193), (2243, 2320)]),
    "arms_velocity": (2718, 2760, [(1057, 1153), (1192, 1283), (1322, 1408), (1452, 1536)]),
}
RECOVERY = {
    "handle_speed": dict(rows=(1472, 1700), mask=[(2250, 2445, 1690, 1730)]),
    "legs_velocity": dict(rows=(2054, 2262), mask=[(1400, 1655, 2235, 2300)]),
    "trunk_velocity": dict(rows=(2054, 2262), mask=[(1790, 2110, 2130, 2262), (2130, 2445, 2235, 2310)]),
    # arms: the legend lines and words only, so the curve passing above them is kept
    "arms_velocity": dict(rows=(2688, 2880), mask=[(1015, 1180, 2785, 2818), (1015, 1305, 2830, 2865),
                                                   (1015, 1250, 2870, 2910), (1350, 1655, 2860, 2930)]),
}
ACC = dict(axis=1758, right=2445, zero=2533.0, scale=38.8, rows=(2380, 2900),
           mask=[(1760, 2110, 2340, 2405), (2110, 2410, 2690, 2790), (2040, 2330, 2840, 2905),
                 (1855, 1945, 2560, 2602), (2005, 2090, 2560, 2602), (2145, 2235, 2560, 2602),
                 (2290, 2380, 2560, 2602)])


def far_end(ink, p):
    """The curves' far end, as extract_fig1.trace finds it (both branches)."""
    both = ink[p["rows"][0]:int(p["zero"] + (1.2 * p["scale"] if p.get("below") is None else p["below"])), :]
    cols = np.flatnonzero(both.any(axis=0))
    cols = cols[(cols > p["axis"] + 2) & (cols < p["right"])]
    return int(cols.max())


def trace_bottom(a, name):
    p = E.PANELS[name]
    rec = RECOVERY[name]
    ink = a < E.BLACK
    right = far_end(ink, p)                       # before masking, as the drive calibration
    ticks = [(x0, x1, TICKS[name][0], TICKS[name][1]) for x0, x1 in TICKS[name][2]] if name in TICKS else []
    for x0, x1, y0, y1 in list(p["mask"]) + rec["mask"] + ticks:
        ink[y0:y1, x0:x1] = False
    band = ink[rec["rows"][0]:rec["rows"][1], :]
    xs, vs = [], []
    for x in range(p["axis"] + 3, right + 1):
        ys = np.flatnonzero(band[:, x])
        if ys.size == 0:
            continue
        bottom = ys[-1]
        run = ys[ys >= bottom - 8]
        y = rec["rows"][0] + float(run.mean())
        xs.append(100.0 * (x - p["axis"]) / (right - p["axis"]))
        vs.append((p["zero"] - y) / p["scale"])
    return np.array(xs), np.array(vs)


def trace_acceleration(a):
    p = ACC
    ink = a < E.BLACK
    for x0, x1, y0, y1 in p["mask"]:
        ink[y0:y1, x0:x1] = False
    band = ink[p["rows"][0]:p["rows"][1], :]
    xs, vs = [], []
    for x in range(p["axis"] + 3, p["right"] - 2):
        ys = np.flatnonzero(band[:, x])
        if ys.size == 0:
            continue
        y = p["rows"][0] + float(ys.mean())
        xs.append(100.0 * (x - p["axis"]) / (p["right"] - p["axis"]))
        vs.append((p["zero"] - y) / p["scale"])
    return np.array(xs), np.array(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=E.PDF)
    ap.add_argument("--overlay", default=None, help="write the traced points over the page, for a look")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        a = E.page_image(args.pdf, tmp)
    grid = np.arange(0.0, 101.0, 1.0)
    table = {}
    traced = {}
    for name in RECOVERY:
        x, v = trace_bottom(a, name)
        o = np.argsort(x)
        traced[name] = (x[o], v[o])
        table[name] = np.interp(grid, x[o], v[o])
    h = table["handle_speed"]
    h[0] = h[-1] = 0.0                                  # turning points, as the drive branch
    ok = h < -0.2
    for seg, printed in (("legs_velocity", 0.51), ("trunk_velocity", 0.48), ("arms_velocity", 0.62)):
        travel = np.trapezoid(np.where(ok, table[seg] / np.where(ok, h, -1.0), 0.0), grid / 100.0 * 1.59)
        print("  %-15s travel over the recovery %.3f m ([K05] table %.2f)" % (seg, travel, printed))
    total = table["legs_velocity"] + table["trunk_velocity"] + table["arms_velocity"]
    inner = (grid >= 10) & (grid <= 90)
    print("segments sum against handle speed on the recovery, 10-90%%: rms %.3f m/s, mean %+.3f m/s"
          % (np.sqrt(np.mean((total - h)[inner] ** 2)), np.mean((total - h)[inner])))
    dt = np.trapezoid(np.where(ok, -1.0 / np.where(ok, h, -1.0), 0.0), grid / 100.0 * 1.59)
    print("  recovery time implied by the handle speed: %.3f s ([K05]: cycle 1.858 s, rhythm 54%%: 0.855 s)" % dt)
    xa, va = trace_acceleration(a)
    acc = np.interp(grid, xa, va)
    print("boat acceleration: min %.2f at %.0f%%, max %.2f m/s^2 ([K05] table -7.92, 3.39)"
          % (va.min(), xa[np.argmin(va)], va.max()))
    with open(REC_OUT, "w", encoding="utf-8", newline="") as f:
        f.write("# Kleshnev (2005) ISBS XXIII 130-133, Fig. 1, on-water ('Boat') recovery branch: velocities in m/s\n"
                "# (negative: towards the bow) against drive length in %, the handle returning 100% -> 0%. Digitised\n"
                "# by research/k05/extract_fig1_cycle.py (bottom-most black run per column); about +-0.03 m/s.\n")
        f.write("length_pct,handle_speed,legs_velocity,trunk_velocity,arms_velocity\n")
        for i in range(grid.size):
            f.write("%.0f,%.3f,%.3f,%.3f,%.3f\n" % (grid[i], h[i], table["legs_velocity"][i],
                                                   table["trunk_velocity"][i], table["arms_velocity"][i]))
    with open(ACC_OUT, "w", encoding="utf-8", newline="") as f:
        f.write("# Kleshnev (2005) ISBS XXIII 130-133, Fig. 1, boat acceleration on water against cycle time (%).\n"
                "# The time origin is not the catch (the catch dip is near 27%). Digitised by\n"
                "# research/k05/extract_fig1_cycle.py (mean row of the black ink per column); about +-0.2 m/s^2.\n")
        f.write("cycle_pct,acceleration_mps2\n")
        for i in range(grid.size):
            f.write("%.0f,%.3f\n" % (grid[i], acc[i]))
    print("wrote", REC_OUT, ACC_OUT)
    if args.overlay:
        from PIL import Image, ImageDraw
        img = Image.fromarray(a.astype(np.uint8)).convert("RGB")
        dr = ImageDraw.Draw(img)
        for name, (x, v) in traced.items():
            p = E.PANELS[name]
            right = far_end(a < E.BLACK, p)
            for xi, vi in zip(x, v):
                px = p["axis"] + xi / 100.0 * (right - p["axis"])
                py = p["zero"] - vi * p["scale"]
                dr.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 0, 0))
        for xi, vi in zip(xa, va):
            px = ACC["axis"] + xi / 100.0 * (ACC["right"] - ACC["axis"])
            py = ACC["zero"] - vi * ACC["scale"]
            dr.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(0, 160, 255))
        img.save(args.overlay)


if __name__ == "__main__":
    main()
