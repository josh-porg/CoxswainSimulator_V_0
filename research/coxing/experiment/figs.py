"""Figures for the experimental paper, from results.json and the control analysis."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "experiment"))   # its data and outputs (gitignored)
R = json.load(open(os.path.join(HERE, "results.json")))
SH = json.load(open(os.path.join(HERE, "..", "short_horizon.json")))
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
INK, ACC, GREY = "#1f2a36", "#b0442c", "#8c96a0"

# Figure 1: response curves
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
h = np.arange(1, 21)
for a, key, title in ((ax[0], "H1_power_speed_curve", "Directive power calls"),
                      (ax[1], "H1_all_calls_speed_curve", "All calls")):
    c = R["hypotheses"][key]
    e, s = 1000 * np.array(c["est"]), 1000 * np.array(c["sd"])
    a.fill_between(h, -1.96 * s, 1.96 * s, color=GREY, alpha=0.25, lw=0, label="95% of global null")
    a.plot(h, e, color=ACC, lw=1.6, label="estimate, global null")
    a.plot(h, 1000 * np.array(c["est_local"]), color=INK, lw=1, ls="--", label="estimate, local null")
    a.axhline(0, color=INK, lw=0.5)
    a.set_title("%s (n = %d)" % (title, c["n"]), fontsize=8.5)
    a.set_xlabel("strokes after the call")
    a.set_xticks([1, 5, 10, 15, 20])
ax[0].set_ylabel("speed beyond momentum (mm s$^{-1}$)")
ax[0].legend(frameon=False, fontsize=7, loc="upper left")
b = SH["control"]["build (rate up)"]
ax[2].plot(np.arange(1, 9), b["rate"]["mean"], color=ACC, lw=1.6, label="stroke rate (spm)")
a2 = ax[2].twinx()
a2.plot(np.arange(1, 9), np.array(b["speed"]["mean"]), color=INK, lw=1.2, ls="--", label="speed (m s$^{-1}$)")
a2.spines["top"].set_visible(False)
ax[2].set_title("Control: scripted builds (n = %d)" % b["n"], fontsize=8.5)
ax[2].set_xlabel("strokes after the call")
ax[2].set_ylabel("change in rate (spm)")
a2.set_ylabel("change in speed (m s$^{-1}$)")
h1, l1 = ax[2].get_legend_handles_labels()
h2, l2 = a2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="lower right")
fig.savefig(os.path.join(HERE, "fig_curves.pdf"))
fig.savefig(os.path.join(HERE, "fig_curves.png"), dpi=300)

# Figure 2: primary contrasts
P = R["primary"]
fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
y = np.arange(len(P))[::-1]
for yy, r in zip(y, P):
    ax.errorbar(1000 * r["est"], yy + 0.12, xerr=1.96 * 1000 * r["sd"], fmt="o", color=ACC, ms=3.5, lw=1, capsize=0)
    ax.errorbar(1000 * r["est_local"], yy - 0.12, xerr=1.96 * 1000 * r["sd_local"], fmt="s", color=INK, ms=3, lw=0.8, capsize=0)
ax.set_yticks(y)
ax.set_yticklabels(["%s (n=%d)" % (r["label"], r["n"]) for r in P], fontsize=7.5)
ax.axvline(0, color=INK, lw=0.5)
ax.set_xlabel("difference (mm s$^{-1}$), ±1.96 null sd")
ax.plot([], [], "o", color=ACC, label="global null")
ax.plot([], [], "s", color=INK, label="local null")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
fig.savefig(os.path.join(HERE, "fig_contrasts.pdf"))
fig.savefig(os.path.join(HERE, "fig_contrasts.png"), dpi=300)
print("figures written")
