"""Kleshnev's population reference curves (RBN 2023/12, No 273) from the newsletter's vector figure, as numbers (§237).

Fig. 1a-c are drawn through 11 data points each, one per 10 % of the **time of the drive phase** (panel a's axis label;
the text calls the same axis "drive length", but panels d-e, which are explicitly against drive length, put the blade's
burial at -3 deg near 10 % of length where a-c put it at 20 %). Each curve is a chain of smoothed spans whose joints are
the data points; the joints are read here and scaled by the axes (the y scale from the tick labels' spacing, the x axis
from the first to the last point). Panels d (by rate) and e (sweep / scull, rate 20+ / 30+) give the vertical oar angle
against drive length, drive and recovery: their joints are the same 10 %-of-time samples, so they also give the
population's drive length against drive time.

    python research/validation/extract_rbn2023.py

Reads data/local/literature/biorow_newsletters/2023_12_BioRowNews.pdf (shared by the author; not in the public repo);
writes rbn2023_fig1.csv beside it and prints the table.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "data", "local", "literature", "biorow_newsletters")
RED, BLUE, BLACK, MAGENTA = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 1.0)


def chains(page, colour, box, width=0.6):
    """Joints of the smoothed polylines of ``colour`` inside ``box`` (x0, y0, x1, y1): a list of point chains."""
    spans = []
    for d in page.get_drawings():
        c = d.get("color")
        if c is None or d.get("fill") is not None or abs((d.get("width") or 0) - width) > 0.05:
            continue
        if any(abs(a - b) > 0.02 for a, b in zip(c, colour)):
            continue
        pts = []
        for it in d["items"]:
            if it[0] == "l":
                pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        if not pts:
            continue
        a, b = pts[0], pts[-1]
        if not (box[0] - 1 <= a[0] <= box[2] + 1 and box[1] - 1 <= a[1] <= box[3] + 1):
            continue
        spans.append((a, b))
    out = []
    for a, b in spans:                                           # consecutive spans share a joint
        if out and abs(out[-1][-1][0] - a[0]) < 0.3 and abs(out[-1][-1][1] - a[1]) < 0.3:
            out[-1].append(b)
        else:
            out.append([a, b])
    return [np.array(c) for c in out]


def main():
    import pymupdf
    page = pymupdf.open(os.path.join(DIR, "2023_12_BioRowNews.pdf"))[0]
    # Fig. 1a: force %, y axis 0 % at 194.6, 100 % at 130.0 (the axis line's ends); x 50.5 .. 278.8
    force = max(chains(page, RED, (48, 128, 282, 199)), key=len)
    x = (force[:, 0] - force[0, 0]) / (force[-1, 0] - force[0, 0])
    f = (194.6 - force[:, 1]) / (194.6 - 130.0) * 100.0
    # Fig. 1b: vertical oar angle, deg: 3 at 208.0, -6 at 256.4 (tick labels' centres), 0 at 224.0 (the axis line)
    vert = max(chains(page, BLUE, (48, 198, 282, 261)), key=len)
    va = (224.0 - vert[:, 1]) / ((256.4 - 208.0) / 9.0)
    # Fig. 1c: velocity %, 0 at 347.7 (the axis line); 20 % per 17.05 pt (tick labels)
    scale = 17.05 / 20.0
    box = (48, 261, 282, 356)
    handle = max(chains(page, BLACK, box), key=len)
    seat = max(chains(page, BLUE, box), key=len)
    upper = max(chains(page, RED, box), key=len)
    vh, vs, vu = [(347.7 - c[:, 1]) / scale for c in (handle, seat, upper)]
    # Fig. 1e: scull, rate 30+ (magenta): vertical angle against drive length; the drive is the lower branch
    loops = chains(page, MAGENTA, (316, 270, 562, 360))
    drive = min(loops, key=lambda c: -c[:, 1].max())             # the branch that goes deepest (largest y)
    xl = (drive[:, 0] - 318.4) / (558.8 - 318.4)
    n = min(len(x), len(va), len(vh), len(vs), len(vu))
    t = np.round(x[:n] * 100.0)
    print("RBN 2023/12 Fig. 1 (population, >50 000 samples), per 10 % of the drive's time:")
    print(" time %% | force %%  vertical deg | handle v %%  seat v %%  upper body v %%")
    rows = []
    for i in range(n):
        rows.append((t[i], f[i], va[i], vh[i], vs[i], vu[i]))
        print("  %4.0f   |  %5.1f     %5.2f      |   %5.1f     %5.1f      %5.1f" % rows[-1])
    # drive length against time: the cumulative integral of the handle velocity (its own check: panel e's joints)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (vh[1:n] + vh[:n - 1]))])
    length = 100.0 * cum / cum[-1]
    print("\ndrive length %% at those times, from the handle velocity's integral: " + " ".join("%.1f" % v for v in length))
    print("Fig. 1e scull R30+ drive branch joints, drive length %%:              " +
          " ".join("%.1f" % (100 * v) for v in np.sort(xl)[:n]))
    out = os.path.join(DIR, "rbn2023_fig1.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# Kleshnev, RBN 2023/12 (No 273) Fig. 1a-c: population reference curves per 10 % of the drive's time,\n"
                 "# read from the newsletter's vector figure (research/validation/extract_rbn2023.py; SOURCES §237).\n"
                 "# force and handle velocity as % of their peaks (per-sample normalised, so the mean peaks below 100);\n"
                 "# seat and upper-body velocity as % of the peak handle velocity; vertical oar angle in degrees;\n"
                 "# length_pct: drive length at that time, the handle velocity's cumulative integral.\n"
                 "time_pct,force_pct,vertical_deg,handle_v_pct,seat_v_pct,upper_v_pct,length_pct\n")
        for r, L in zip(rows, length):
            fh.write(",".join("%.2f" % v for v in r + (L,)) + "\n")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
