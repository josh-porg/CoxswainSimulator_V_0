"""Coxing sheet: which calls moved the boat."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

exec(open("calls.py").read().split("print(\"%-7s")[0])   # reuse MOVES, data, OFF, N

TEAL, AMBER, INK, MUTED = "#1C6B63", "#B9761F", "#1F2421", "#6B7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "grid.color": "#E3E6E5", "grid.linewidth": 0.6})

rec = []
for vt, text, kind in MOVES:
    i = int(np.argmin(np.abs(t - (vsec(vt) + OFF))))
    if i < N or i > len(t) - N - 1:
        continue
    rec.append(dict(vt=vt, text=text, kind=kind,
                    dv=v[i:i+N].mean() - v[i-N:i].mean(),
                    dl=dps[i:i+N].mean() - dps[i-N:i].mean()))

fig = plt.figure(figsize=(8.27, 11.69))
fig.patch.set_facecolor("white")
fig.text(0.07, 0.966, "Which calls moved the boat", fontsize=16, fontweight="bold", color=INK)
fig.text(0.07, 0.932, "Pink Ribbon Row 2026, the 4+ · 22 explicit moves · each measured over the 10 strokes "
                      "after the call against the 10 before\nVideo synced to the boat data by your own split "
                      "callouts: offset +1.0 s, matching the measured split to 1.4 s on average",
         fontsize=8.3, color=MUTED, linespacing=1.5)

ax = fig.add_axes([0.30, 0.530, 0.63, 0.375])
ypos = np.arange(len(rec))[::-1]
cols = [TEAL if r["dv"] > 0.015 else (MUTED if r["dv"] > 0 else AMBER) for r in rec]
ax.barh(ypos, [r["dv"] for r in rec], color=cols, height=0.65)
ax.axvline(0, color=MUTED, lw=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels(["%s  %s" % (r["vt"], r["text"][:30]) for r in rec], fontsize=7.2)
ax.set_xlabel("change in boat speed over the next 10 strokes (m/s)")
ax.set_title("Every move, in order", fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)
ax.grid(axis="x")
ax.set_axisbelow(True)

ax2 = fig.add_axes([0.30, 0.352, 0.63, 0.112])
kinds = ["ratio", "press", "length", "catch", "focus", "burst"]
mv = [np.mean([r["dl"] for r in rec if r["kind"] == k]) for k in kinds]
cols2 = [TEAL if m > 0.02 else AMBER for m in mv]
ax2.bar(kinds, mv, color=cols2, width=0.6)
ax2.axhline(0, color=MUTED, lw=0.9)
ax2.set_ylabel("mean Δ metres/stroke")
ax2.set_title("By type of call — length is what answered", fontsize=11, fontweight="bold",
              color=INK, loc="left", pad=8)
ax2.grid(axis="y")
ax2.set_axisbelow(True)

txt = (
    "What the boat actually did\n\n"
    "• 12 of 22 moves answered, 3 partly, 7 not.\n"
    "• RATIO calls are far and away your best: +0.034 m/s and +0.27 m per stroke on average, the only\n"
    "   type that bought speed AND length. \"Ratio, get long\" at 5:40 gave +0.080 m/s and +0.39 m/stroke.\n"
    "• The TENS did not work: four bursts, −0.010 m/s and −0.06 m/stroke on average.\n"
    "• That is consistent with the race as a whole — this crew faded in length, not power, so the calls\n"
    "   that restored length moved the boat and the ones that asked for more power did not.\n"
    "• Two fair defences: 14:10 was the starboard yield to be passed, and 17:28's five were into a\n"
    "   corner. Both lose speed by geometry, not by being wrong.\n\n"
    "And what it was not\n\n"
    "• Steering: NOT DETECTABLE. The raw GPS path is 60 m longer than a smoothed line, but expected\n"
    "   jitter inflation alone is 60.5 m — the wiggle is noise, not helm.\n"
    "• Pacing: −2.8 s, i.e. they held a steadier speed than the same energy at constant pace would give.\n"
    "• The Cut: +0.002 m/s versus the rest of the race. No measurable environmental loss.\n"
    "• Length fade: up to 67 s. Holding the opening 7.44 m/stroke for all 576 strokes would have\n"
    "   covered the course 234 m sooner. That is the race."
)
fig.text(0.07, 0.312, txt, fontsize=8.3, color=INK, va="top", linespacing=1.5)
fig.text(0.07, 0.018,
         "Speed, rate and position measured by the CoxBox; transcript from the video's auto-captions. A 10-stroke "
         "window catches the move but also whatever\nelse was happening, so single rows are indicative and the "
         "by-type means are the reliable part. Correlation, not proof: the crew also tires.",
         fontsize=6.8, color=MUTED, va="top", linespacing=1.5)

with PdfPages("coxing_calls.pdf") as pdf:
    pdf.savefig(fig)
fig.savefig("calls_preview.png", dpi=105)
print("ok")
