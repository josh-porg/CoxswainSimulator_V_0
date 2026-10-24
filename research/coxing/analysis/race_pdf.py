"""Race analysis sheet for the 4+, Pink Ribbon Row 2026."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

D = np.load("race.npz")
d, t, v, sr, dps, cw = D["d"], D["t"], D["v"], D["sr"], D["dps"], D["crew_power"]
TEAL, AMBER, INK, MUTED = "#1C6B63", "#B9761F", "#1F2421", "#6B7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "grid.color": "#E3E6E5",
                     "grid.linewidth": 0.6})


def roll(a, n=21):
    k = np.ones(n) / n
    return np.convolve(np.pad(a, (n // 2, n // 2), mode="edge"), k, mode="valid")[:len(a)]


def clock(x):
    return "%d:%02d" % (int(x // 60), int(x % 60))


LOSS = [(79.2, 283), (252.6, 876), (382.4, 1324), (550.5, 1910), (1065.7, 3695)]
CUT = (1060, 1650)

fig = plt.figure(figsize=(8.27, 11.69))
fig.patch.set_facecolor("white")
fig.text(0.07, 0.960, "Pink Ribbon Row 2026 — the 4+, stroke by stroke",
         fontsize=16, fontweight="bold", color=INK)
fig.text(0.07, 0.940, "4050 m in 19:26.5 · 2:24.0 average · 576 strokes · 29.7 spm · "
                      "GPS and rate measured, force and power supplied by the model",
         fontsize=8.3, color=MUTED)

panels = [("Boat speed (m/s)", v, 0.760, TEAL, True),
          ("Stroke rate (spm)", sr, 0.585, AMBER, False),
          ("Metres per stroke", dps, 0.410, TEAL, False)]
for title, y, bottom, colour, mark in panels:
    ax = fig.add_axes([0.10, bottom, 0.84, 0.150])
    ax.plot(d, y, color=colour, lw=0.5, alpha=0.30)
    ax.plot(d, roll(y), color=colour, lw=2.0, solid_capstyle="round")
    ax.axvspan(*CUT, color=MUTED, alpha=0.10, lw=0)
    if mark:
        for tt, dd in LOSS:
            ax.axvline(dd, color=AMBER, lw=1.0, ls=(0, (3, 3)), alpha=0.8)
            ax.text(dd, ax.get_ylim()[1], clock(tt), fontsize=6.5, color=AMBER,
                    rotation=90, va="top", ha="right")
    ax.set_ylabel(title)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlim(0, d[-1])
    if bottom == 0.410:
        ax.set_xlabel("distance (m)")
fig.text(0.115, 0.918, "shaded = Montlake Cut · dashed = the five places the boat dropped off its own trend",
         fontsize=7.5, color=MUTED)

ax = fig.add_axes([0.10, 0.225, 0.84, 0.130])
edges = np.arange(0, d[-1] + 1, 500.0)
sp, ctr = [], []
for a, b in zip(edges[:-1], edges[1:]):
    m = (d >= a) & (d < b)
    if m.sum() < 3:
        continue
    sp.append(500.0 / ((d[m].max() - d[m].min()) / (t[m].max() - t[m].min())))
    ctr.append(0.5 * (a + b))
bars = ax.bar(ctr, sp, width=420, color=TEAL, alpha=0.85)
bars[int(np.argmax(sp))].set_color(AMBER)
ax.axhline(144.0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
ax.text(120, 146.9, "race average 2:24.0", fontsize=7.5, color=MUTED, va="top")
ax.set_ylim(140, 148)
ax.set_yticks([141, 143, 145, 147])
ax.set_yticklabels(["2:21", "2:23", "2:25", "2:27"])
ax.set_ylabel("split per 500 m")
ax.set_xlabel("distance (m)")
ax.set_title("Where the time went", fontsize=10, fontweight="bold", color=INK, loc="left", pad=6)
ax.grid(axis="y")
ax.set_axisbelow(True)

txt = (
    "What the data says\n\n"
    "• The start was not the problem. The first 500 m was 2:22.6, the second-fastest piece of the race.\n"
    "• The slowest piece was 500–1000 m at 2:26.5 — the settle after the start, not the start itself.\n"
    "• The Cut cost nothing measurable: 3.474 m/s through it against a race mean of 3.472. Whatever the\n"
    "   chop did, it did not show up as lost boat speed.\n"
    "• They faded in LENGTH, not power. Metres per stroke fell 7.44 → 6.82 (−8%) while the rate rose\n"
    "   28.3 → 30.8. Reconstructed crew power stayed flat at 518–558 W, about 136 W per rower.\n"
    "• That is consistent with having something left: the engine held, the stroke shortened."
)
fig.text(0.07, 0.178, txt, fontsize=8.6, color=INK, va="top", linespacing=1.55)
fig.text(0.07, 0.030,
         "Speed, rate, distance and position are measured by the CoxBox. Drag and power are the model's: hull "
         "resistance at the measured speed,\nwith blade efficiency 0.681 from the research profile. There were no "
         "instrumented oarlocks, so no force or oar angle was recorded.",
         fontsize=6.8, color=MUTED, va="top", linespacing=1.5)

with PdfPages("race_analysis.pdf") as pdf:
    pdf.savefig(fig)
fig.savefig("race_preview.png", dpi=110)
print("ok")
