"""Digitise the footplate centre of pressure from Vieira et al. (2017), PLoS ONE 12(11):e0187202.

    .venv/Scripts/python.exe research/cr06/extract_vieira2017.py

Figures from PLOS (CC-BY), large PNGs kept in data/local/literature/vieira2017/ (g003.png, g004.png).
CoP is the pressure-weighted fore-rear coordinate of the Pedar insole, mm from the heel end (Fig 2B shows
the insole on that scale). Read here:
- Fig 3, control insole (thick grey): one rower (participant 6, male, 1.85 m, 84 kg, left foot, 32 spm),
  CoP and plantar force against % of the rowing cycle; catch at the grey dashed line, peak force at the
  black dashed line.
- Fig 4C and 4F, control insole (black bars): the group (11 elite rowers, 4 women) mean and SD of the CoP at
  peak force and averaged over the drive, left and right feet.
Axes are calibrated on the printed ticks (found as short dark marks against the axis lines). Writes
data/literature/vieira2017_cop.csv.
"""
import os

import numpy as np
from PIL import Image

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CODE, "..", ".."))
FIG = os.path.join(ROOT, "data", "local", "literature", "vieira2017")
OUT = os.path.join(ROOT, "data", "literature", "vieira2017_cop.csv")


def load(name):
    return np.asarray(Image.open(os.path.join(FIG, name)).convert("RGB")).astype(int)


def ticks_y(dark, x0, x1, r0, r1):
    rows = [y for y in range(r0, r1) if dark[y, x0:x1].sum() >= 4]
    out = []
    for y in rows:                                            # merge adjacent rows into one tick
        if out and y - out[-1][-1] <= 2:
            out[-1].append(y)
        else:
            out.append([y])
    return [float(np.mean(g)) for g in out]


def fig3():
    im = load("g003.png")
    dark = im.sum(2) < 250
    grey = (np.abs(im[..., 0] - im[..., 1]) < 12) & (np.abs(im[..., 1] - im[..., 2]) < 12) & (im[..., 0] > 140) & (im[..., 0] < 205)
    # CoP panel: left axis at column ~101, bottom axis at row ~759, ticks every 50 mm from 0 to 250
    yt = ticks_y(dark, 102, 108, 500, 765)
    mm = np.linspace(250, 0, len(yt))
    fy = np.polyfit(yt, mm, 1)
    # x: the printed ticks at 40, 60, 80, 100 % (20 is hidden under the curve)
    xt = {40: 296.0, 60: 396.5, 80: 497.5, 100: 598.0}
    fx = np.polyfit(list(xt.values()), list(xt.keys()), 1)
    # force panel above: its own y ticks (0..600 N) left of the same axis
    yf = ticks_y(dark, 102, 108, 0, 232)
    fN = np.polyfit(yf, np.linspace(600, 0, len(yf)), 1) if len(yf) >= 4 else None
    rows = []
    for x in range(105, 597):
        col = np.flatnonzero(grey[548:757, x]) + 548
        if len(col) < 3 or np.ptp(col) > 40:                    # thick grey line: several pixels deep, but not a dashed
            continue                                            # vertical marker spanning the panel
        pct = float(np.polyval(fx, x)); cop = float(np.polyval(fy, np.median(col)))
        fcol = np.flatnonzero(grey[0:230, x])
        force = float(np.polyval(fN, np.median(fcol))) if fN is not None and len(fcol) >= 3 else np.nan
        rows.append((pct, cop, force))
    return np.array(rows), (len(yt), len(yf))


def main():
    R, nt = fig3()
    print("Fig 3: %d samples; y ticks found %s" % (len(R), nt))
    drive = R[(R[:, 0] >= 57.5) & (R[:, 0] <= 92)]
    for lo, hi in ((57.5, 65), (65, 75), (75, 80), (80, 85), (85, 88), (88, 92)):
        m = drive[(drive[:, 0] >= lo) & (drive[:, 0] < hi)]
        if len(m):
            print("  %4.1f-%4.1f %% of cycle: CoP %5.1f mm (range %5.1f-%5.1f), force %s" % (lo, hi, np.median(m[:, 1]), m[:, 1].min(), m[:, 1].max(),
                  "%.0f N" % np.nanmedian(m[:, 2]) if np.isfinite(np.nanmedian(m[:, 2])) else "-"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Vieira, Botter, Gastaldi, Sacco, Martelli & Giacomozzi (2017), PLoS ONE 12(11):e0187202 (CC-BY).\n"
                "# Fore-rear centre of pressure on the Pedar insole, mm from the heel end; Concept2 Model E.\n"
                "# Fig 3, control insole: participant 6 (male, 1.85 m, 84 kg), left foot, 32 spm; digitised by\n"
                "# research/cr06/extract_vieira2017.py from the PLOS large PNG, axes calibrated on the printed ticks.\n"
                "# Group values (Fig 4C, 4F) are read by eye from the black (control) bars, +-3 mm.\n")
        f.write("series,pct_cycle,cop_mm,force_N\n")
        for p, c, fN in R:
            f.write("fig3_control,%.2f,%.1f,%s\n" % (p, c, "" if not np.isfinite(fN) else "%.0f" % fN))
        f.write("group_peak_force_left_mean,,197,\ngroup_peak_force_left_sd,,14,\ngroup_peak_force_right_mean,,195,\ngroup_peak_force_right_sd,,16,\n"
                "group_drive_average_left_mean,,158,\ngroup_drive_average_left_sd,,34,\ngroup_drive_average_right_mean,,158,\ngroup_drive_average_right_sd,,33,\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
