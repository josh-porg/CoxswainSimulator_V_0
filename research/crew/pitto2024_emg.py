"""On-water rowing EMG as numbers, and the multi-clock body's excitation modules from it (review §12 step 4; §253).

Source: Pitto, Ertel, Simon, Gauchard & Mornieux (2024), Appl. Sci. 14:9055 (CC BY), 12 national-level male scullers
on water at race rate, 13 muscles, Fig. 3: mean linear envelope (normalised to each muscle's peak) against the cycle
from the catch, the finish marked by a dashed line. The figure is a raster (data/local/literature/pitto2024_scull_emg/
fig3_emg.png, 2440 x 2952); each panel's black mean curve is read column by column, calibrated on the panel's own
axes (0 at the bottom axis, 1 at the frame's top, the 0.5 grid line midway; the cycle from the left axis to the right
frame, the finish at the dashed line, 52.1 % of the cycle). Reading error ~ the curve's half-thickness, ±0.02.

Modules (review §12 step 4: three module waveforms, legs / trunk / arms, plus the recovery's flexors), each the mean of
its muscles' envelopes, renormalised to peak 1:
* trunk extensors: thoracic and lumbar erector spinae, gluteus maximus;
* arm pullers: biceps brachii, brachioradialis, deltoideus posterior;
* trunk flexors (finish and recovery): rectus abdominis, external and internal obliquus;
* arm return (recovery): deltoideus anterior; triceps brachii longus (arms held straight in the early drive);
* **leg extensors: not recorded by Pitto et al.** (no vastus or rectus femoris) — the gap is reported, and the module
  is left absent rather than invented (the knee's demand timing from the model is the placeholder, flagged as such).

    python research/crew/pitto2024_emg.py      # digitise -> data/literature/pitto2024_emg.csv; modules; overlay check
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(ROOT, "data", "local", "literature", "pitto2024_scull_emg", "fig3_emg.png")
CSV = os.path.join(ROOT, "data", "literature", "pitto2024_emg.csv")

#: panel axes in the original pixels: column (left axis x, right frame x, dashed finish x), row (bottom axis y, top y)
COLS = ((57.5, 752.5, 419.5), (873.5, 1568.5, 1235.5), (1686.5, 2381.5, 2048.5))
ROWS = ((480.0, 72.0), (1065.0, 654.5), (1648.0, 1239.0), (2234.0, 1825.0), (2820.0, 2408.0))
PANELS = {  # name: (row, col)
    "biceps_brachii": (0, 0), "brachioradialis": (0, 1), "flexor_carpi_ulnaris": (0, 2),
    "triceps_brachii_longus": (1, 0), "deltoideus_anterior": (1, 1), "deltoideus_posterior": (1, 2),
    "thoracic_erector_spinae": (2, 0), "lumbar_erector_spinae": (2, 1), "external_obliquus": (2, 2),
    "internal_obliquus": (3, 0), "rectus_abdominis": (3, 1), "tensor_fasciae_latae": (3, 2),
    "gluteus_maximus": (4, 2),
}
FINISH = (419.5 - 57.5) / (752.5 - 57.5)                       # the finish's cycle fraction (the dashed line)
MODULES = {
    "trunk_extensors": ("thoracic_erector_spinae", "lumbar_erector_spinae", "gluteus_maximus"),
    "arm_pullers": ("biceps_brachii", "brachioradialis", "deltoideus_posterior"),
    "trunk_flexors": ("rectus_abdominis", "external_obliquus", "internal_obliquus"),
    "arm_return": ("deltoideus_anterior",),
    "arm_extensors": ("triceps_brachii_longus",),
}
GRID = np.linspace(0.0, 1.0, 101)


def digitise():
    """``{muscle: envelope on GRID}`` read from the figure."""
    from PIL import Image
    im = np.asarray(Image.open(FIG).convert("L")).astype(int)
    out = {}
    for name, (r, c) in PANELS.items():
        x0, x1, xd = COLS[c]
        yb, yt = ROWS[r]
        xs, vs = [], []
        for x in range(int(x0) + 5, int(x1) - 2):
            if abs(x - xd) < 9:                                   # the dashed finish line
                continue
            col = im[int(yt) + 2:int(yb) - 4, x]
            dark = np.flatnonzero(col < 80)
            if dark.size < 3:
                continue
            # the thickest run of dark pixels is the mean curve (the grey band and grid lines are lighter)
            runs = np.split(dark, np.flatnonzero(np.diff(dark) > 1) + 1)
            best = max(runs, key=len)
            if len(best) < 3:
                continue
            y = int(yt) + 2 + float(np.mean(best))
            xs.append((x - x0) / (x1 - x0))
            vs.append((yb - y) / (yb - yt))
        xs, vs = np.array(xs), np.array(vs)
        out[name] = np.interp(GRID, xs, vs)
    return out


def write_csv(env):
    names = list(env)
    with open(CSV, "w", encoding="utf-8") as fh:
        fh.write("# Pitto, Ertel, Simon, Gauchard & Mornieux (2024), Appl. Sci. 14:9055 (CC BY), Fig. 3: on-water mean EMG\n"
                 "# envelopes (each normalised to its peak), 12 national-level male scullers at race rate; digitised from\n"
                 "# the raster figure by research/crew/pitto2024_emg.py (SOURCES §253), +-0.02. cycle: fraction from the\n"
                 "# catch; the finish at %.3f.\n" % FINISH)
        fh.write("cycle," + ",".join(names) + "\n")
        for i, x in enumerate(GRID):
            fh.write("%.2f," % x + ",".join("%.3f" % env[n][i] for n in names) + "\n")


def read_csv():
    import csv
    rows = list(csv.DictReader(l for l in open(CSV, encoding="utf-8") if not l.startswith("#")))
    cyc = np.array([float(r["cycle"]) for r in rows])
    return cyc, {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "cycle"}


def modules():
    """``(cycle, {module: waveform, peak 1})`` from the digitised envelopes (the leg extensors absent: not recorded)."""
    cyc, env = read_csv()
    out = {}
    for m, muscles in MODULES.items():
        w = np.mean([env[k] for k in muscles], axis=0)
        out[m] = w / w.max()
    return cyc, out


def main():
    env = digitise()
    write_csv(env)
    print("wrote", CSV)
    cyc, mods = modules()
    print("finish at %.3f of the cycle" % FINISH)
    print("%-26s %s | peak at (cycle, %% of drive)" % ("envelope", " ".join("%4.0f" % (100 * q) for q in np.arange(0, 1.0, 0.1))))
    for name, w in list(env.items()) + [("MODULE " + k, v) for k, v in mods.items()]:
        k = int(np.argmax(w))
        print("%-26s %s | %.2f (%.0f %%)" % (name[:26], " ".join("%4.2f" % np.interp(q, GRID, w) for q in np.arange(0, 1.0, 0.1)),
                                            cyc[k], 100 * cyc[k] / FINISH if cyc[k] <= FINISH else float("nan")))
    # an overlay for a one-time visual check of the reading
    try:
        from PIL import Image, ImageDraw
        im = Image.open(FIG).convert("RGB")
        dr = ImageDraw.Draw(im)
        for name, (r, c) in PANELS.items():
            x0, x1, _xd = COLS[c]
            yb, yt = ROWS[r]
            pts = [(x0 + q * (x1 - x0), yb - v * (yb - yt)) for q, v in zip(GRID, env[name])]
            dr.line(pts, fill=(255, 0, 0), width=3)
        out = os.path.join(os.path.dirname(FIG), "fig3_emg_overlay.png")
        im.resize((im.width // 2, im.height // 2)).save(out)
        print("overlay:", out)
    except Exception as e:                                        # noqa: BLE001
        print("overlay skipped:", e)


if __name__ == "__main__":
    main()
