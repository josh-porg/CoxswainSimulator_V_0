"""The leg extensors' EMG envelope from an open indoor-rowing study, for the multi-clock body's leg module (§255).

Source: Vieira, Cerone, Stocchi, Lalli, Andrews & Gazzoni (2020), Sensors 20(6):1666 (CC BY 4.0), PMC7147320: surface
EMG of eight leg muscles on an indoor rower at 18-28 spm, envelopes averaged over cycles (Fig. 2; right side black,
left side grey; the cycle from the catch, the finish at the dashed line). Pitto et al. (2024)'s on-water set has no knee
extensor (MULTI_CLOCK_BODY_PLAN blocker B2); this is the open source that fills it, **on an ergometer**, and is flagged
as such wherever it is used.

The right side's thin black envelope of vastus lateralis and vastus medialis is read column by column on the figure's
own axis (ticks at the catch 0 %, the finish, the catch 100 %: x 117, 345, 718 px; the drive 37.9 % of the cycle), each
normalised (min to max) to 0-1 (the panels carry no zero line). Under the labels the envelope is taken as the lowest dark
trace (the glyphs sit above it; skipping and interpolating instead drew a false ramp through the recovery).

    python research/crew/vieira2020_emg.py    # -> data/literature/vieira2020_leg_emg.csv; prints the peaks
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(ROOT, "data", "local", "literature", "vieira2020_leg_emg", "sensors-20-01666-g002.jpg")
CSV = os.path.join(ROOT, "data", "literature", "vieira2020_leg_emg.csv")
X_CATCH, X_FINISH, X_END = 117.0, 345.0, 718.0
FINISH = (X_FINISH - X_CATCH) / (X_END - X_CATCH)
#: panel: (y window, label columns to skip)
PANELS = {"vastus_lateralis": ((505, 655), (440, 700)), "vastus_medialis": ((648, 790), (440, 700)),
          # the hamstrings (§273): the trunk's hip-extension moment has a second, leg-timed contributor
          "semitendinosus": ((790, 935), (440, 700)), "biceps_femoris": ((935, 1075), (440, 700))}
GRID = np.linspace(0.0, 1.0, 101)


def digitise():
    from PIL import Image
    im = np.asarray(Image.open(FIG).convert("L")).astype(int)
    out = {}
    for name, ((y0, y1), (l0, l1)) in PANELS.items():
        xs, ys = [], []
        for x in range(int(X_CATCH) + 3, int(X_END) - 2):
            if abs(x - X_FINISH) < 3:
                continue
            col = im[y0:y1, x]
            dark = np.flatnonzero(col < 90)
            if dark.size == 0:
                continue
            runs = np.split(dark, np.flatnonzero(np.diff(dark) > 1) + 1)
            # under the label the envelope is the lowest dark trace (the glyphs sit above it); elsewhere the thickest
            best = runs[-1] if l0 <= x <= l1 else max(runs, key=len)
            xs.append((x - X_CATCH) / (X_END - X_CATCH))
            ys.append(y0 + float(np.mean(best)))
        ys = np.array(ys)
        v = (ys.max() - ys) / (ys.max() - ys.min())               # up is larger activity
        out[name] = np.interp(GRID, np.array(xs), v, period=1.0)
    return out


def main():
    env = digitise()
    with open(CSV, "w", encoding="utf-8") as fh:
        fh.write("# Vieira et al. (2020), Sensors 20(6):1666 (CC BY 4.0), Fig. 2: indoor-rowing EMG envelopes (right side),\n"
                 "# 18-28 spm, normalised min-max to 0-1; digitised by research/crew/vieira2020_emg.py (SOURCES §255).\n"
                 "# cycle: fraction from the catch; the finish at %.3f. ERGOMETER, not on water.\n" % FINISH)
        fh.write("cycle," + ",".join(env) + "\n")
        for i, x in enumerate(GRID):
            fh.write("%.2f," % x + ",".join("%.3f" % env[n][i] for n in env) + "\n")
    print("wrote", CSV, "; finish at %.3f of the cycle" % FINISH)
    for name, w in env.items():
        k = int(np.argmax(w))
        print("%-18s peak at %.2f of the cycle = %.0f %% of the drive; above half from %.0f to %.0f %% of the drive"
              % (name, GRID[k], 100 * GRID[k] / FINISH, 100 * GRID[np.flatnonzero(w >= 0.5)[0]] / FINISH,
                 100 * GRID[np.flatnonzero(w[:60] >= 0.5)[-1]] / FINISH))


if __name__ == "__main__":
    main()
