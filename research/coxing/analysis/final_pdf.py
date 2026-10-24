"""Call breakdown and full loss decomposition."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

TEAL, AMBER, INK, MUTED = "#1C6B63", "#B9761F", "#1F2421", "#6B7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "grid.color": "#E3E6E5", "grid.linewidth": 0.6})

ROWS = [
 ("TECHNICAL", [("ratio / rhythm", 7, .0123), ("breathe / loose", 10, .0018),
                ("length / send", 41, .0002), ("finish / release", 26, -.0084),
                ("suspend / hang", 32, -.0140), ("body / swing", 39, -.0151),
                ("catch / front end", 36, -.0207), ("together / clean", 13, -.0344)]),
 ("TACTICAL", [("split stated", 42, .0286), ("in two / take a N", 28, .0229),
               ("other crews", 24, .0016), ("rate stated", 8, -.0013),
               ("distance / position", 14, -.0165), ("steering (turns)", 25, -.0232)]),
 ("MOTIVATIONAL", [("exhortation", 10, .0432), ("identity / shared", 15, .0295),
                   ("praise", 50, .0077)]),
]

fig = plt.figure(figsize=(8.27, 11.69))
fig.patch.set_facecolor("white")
fig.text(0.07, 0.966, "Which calls worked, and what the water cost",
         fontsize=16, fontweight="bold", color=INK)
fig.text(0.07, 0.938, "Pink Ribbon Row 2026, the 4+ · 595 calls classified against Nugent et al. (2025) · "
                      "each scored on the 10 strokes after it\nagainst the 10 before, relative to the average "
                      "call (+0.0036 m/s)", fontsize=8.3, color=MUTED, linespacing=1.5)

labels, vals, cols = [], [], []
for gi, (theme, subs) in enumerate(ROWS):
    if gi:
        labels.append(""); vals.append(0.0); cols.append("none")
    labels.append(theme); vals.append(0.0); cols.append("none")
    for name, n, rel in subs:
        labels.append("   %s   (n=%d)" % (name, n))
        vals.append(rel)
        cols.append(TEAL if rel > 0 else AMBER)
ax = fig.add_axes([0.33, 0.500, 0.60, 0.408])
yp = np.arange(len(labels))[::-1]
ax.barh(yp, vals, color=cols, height=0.68)
ax.axvline(0, color=MUTED, lw=1.0)
ax.set_yticks(yp)
ax.set_yticklabels(labels, fontsize=7.4)
ax.set_xlabel("boat speed gained or lost against the average call (m/s)")
ax.grid(axis="x")
ax.set_axisbelow(True)
for lab, tick in zip(labels, ax.get_yticklabels()):
    if lab and not lab.startswith("   "):
        tick.set_fontweight("bold"); tick.set_color(INK); tick.set_fontsize(8.2)

txt = (
 "The pattern, in one line: the calls that work are the ones you use least.\n\n"
 "• EVERY motivational sub-type is positive. Exhortation (+0.043) and shared identity (+0.030) are the two\n"
 "   strongest things you did all race — and motivational calls were only 15–17% of your total, against 38.6%\n"
 "   for the elite coxes in Nugent. Praise is your most-used motivational call and your weakest.\n"
 "• MOST technical sub-types are negative. Only ratio helps, and it helps a lot (+0.012, and n is only 7).\n"
 "• Splits are your best tactical call by far (+0.029) and your rate calls do nothing (−0.001).\n"
 "• Steering shows as the worst tactical call (−0.023), but that is the physical cost of the two turns, not\n"
 "   a bad call. Priced separately below at 1.6 s.\n\n"
 "Are technical calls just reactive? Partly — but it does not rescue them.\n\n"
 "   theme            boat BEFORE the call     AFTER        recovery\n"
 "   technical            −0.0025             −0.0080        −0.0054\n"
 "   tactical             −0.0016             +0.0069        +0.0085\n"
 "   motivational         +0.0053             +0.0191        +0.0138\n\n"
 "Technical calls are indeed made when the boat is already falling. So are tactical ones — and those reverse\n"
 "it. Technical calls do not arrest the decline they were answering.\n\n"
 "What the water cost, measured\n\n"
 "   shallow water (soundings + Sretenskii)      +4.3 s      49% of strokes under 5 m; Fr_h to 0.79\n"
 "   windage (1.7 m/s from 33°, observed)        +4.3 s      net headwind component −0.34 m/s\n"
 "   the two turns (rudder + pressure turn)      +1.6 s      53° at 14:04, 41° at 17:33\n"
 "   wind-driven chop                            ~0.3 s      H_s 0.039 m — negligible\n"
 "   motorboat wake in the Cut                     ~0 s      +0.002 m/s vs the rest of the race\n"
 "   pacing                                      −2.8 s      steadier than constant-energy pace\n"
 "   ------------------------------------------------------------\n"
 "   everything above                            ~8 s        of a 66 s gap to the requalification line\n"
 "   length fade                                up to 67 s   7.44 → 6.82 m per stroke"
)
fig.text(0.07, 0.462, txt, fontsize=7.6, color=INK, va="top", linespacing=1.5, family="DejaVu Sans")
FOOT = [
    "Classification is rule-based, approximating Nugent's human thematic analysis. 21% of",
    "calls stayed unclassified and the shares moved several points when one rule changed, so",
    "treat them as indicative. Sub-types with n<3 omitted. Windage assumes Cd.A = 0.18 m2,",
    "order-of-magnitude only. Depth soundings are coarse (median 16 m to the nearest) and 29%",
    "were clipped as near-shore artefacts. Correlation, not proof.",
]
fig.text(0.07, 0.034, "\n".join(FOOT), fontsize=6.6, color=MUTED, va="top", linespacing=1.55)

with PdfPages("calls_and_water.pdf") as pdf:
    pdf.savefig(fig)
fig.savefig("final_preview.png", dpi=105)
print("ok")
