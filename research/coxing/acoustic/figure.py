"""Figure for the acoustic strand: what the background sound tracks, and what the voice alone tells.

    .venv/Scripts/python.exe research/coxing/acoustic/figure.py

Left: correlation of each background feature with stroke rate and with the boat's one-step speed error (filled:
q < 0.05 under both nulls). Right: balanced accuracy of predicting the kind of call from the voice alone, with all ten
features and with the six that do not depend on the number of words, against the label-permutation null (mean and 95%).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.abspath(os.path.join(CODE, "..", "..", "..", "data", "local", "coxing"))
R = json.load(open(os.path.join(LOCAL, "acoustic", "results.json")))
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
INK, ACC, GREY = "#1f2a36", "#b0442c", "#8c96a0"
NAMES = {"thump": "thump", "sharp": "thump sharpness", "splash": "catch splash", "spread": "catch spread", "water": "water", "rumble": "rumble"}

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
rows = R["A1"]["correlations"]
feats = list(NAMES)
y = np.arange(len(feats))[::-1]
for off, w, col in ((0.17, "stroke rate", INK), (-0.17, "speed error", ACC)):
    for yy, f in zip(y, feats):
        r = next(x for x in rows if x["feature"] == f and x["with_"] == w)
        sig = r["q"] < 0.05 and r["q_local"] < 0.05
        ax[0].barh(yy + off, r["r"], height=0.3, color=col if sig else "none", edgecolor=col, lw=0.8)
ax[0].axvline(0, color=INK, lw=0.5)
ax[0].set_yticks(y)
ax[0].set_yticklabels([NAMES[f] for f in feats])
ax[0].set_xlabel("correlation within pieces")
ax[0].barh(np.nan, 0, color=INK, label="with stroke rate")
ax[0].barh(np.nan, 0, color=ACC, label="with the speed error")
ax[0].legend(frameon=False, fontsize=7, loc="lower right")
ax[0].set_title("Background sound", fontsize=8.5)
tasks = [("target group", "target group\n(7 classes)"), ("directive", "directive\nor not"), ("blind arousal raised", "raises intensity\n(blind coder)")]
x = np.arange(len(tasks))
for off, key, col, lab in ((-0.17, "V1", INK, "all ten voice features"), (0.17, "V1_sound_only_unplanned", ACC, "pitch, level and quality only")):
    for xx, (t, _) in zip(x, tasks):
        v = R[key][t]
        ax[1].plot([xx + off - 0.13, xx + off + 0.13], [v["null_mean"]] * 2, color=GREY, lw=1)
        ax[1].fill_between([xx + off - 0.13, xx + off + 0.13], v["null_mean"] - 1.96 * v["null_sd"], v["null_mean"] + 1.96 * v["null_sd"], color=GREY, alpha=0.3, lw=0)
        ax[1].plot(xx + off, v["balanced_accuracy"], "o", color=col, ms=4.5, label=lab if xx == 0 else None)
ax[1].set_xticks(x)
ax[1].set_xticklabels([l for _, l in tasks], fontsize=7.5)
ax[1].set_ylabel("balanced accuracy")
ax[1].legend(frameon=False, fontsize=7, loc="upper left")
ax[1].set_title("Kind of call from the voice alone", fontsize=8.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(LOCAL, "experiment", "fig_acoustic." + ext), dpi=200)
print("written fig_acoustic")
