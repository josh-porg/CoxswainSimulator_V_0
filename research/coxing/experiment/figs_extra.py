"""Two further figures for the experimental paper.

    .venv/Scripts/python.exe research/coxing/experiment/figs_extra.py

fig_example: one race as the analysis sees it (speed and its one-step forecast, stroke rate, and every call by class).
fig_channels: what directive power, rate and length calls are followed by in stroke rate, distance per stroke and speed
(strokes 1-10 beyond the boat's own forecast; grey band = 95% of the global shift null; dashed = against the local null).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import examiner_checks as X  # noqa: E402
CP = X.CP

EXP = X.EXP
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
INK, ACC, GREY = "#1f2a36", "#b0442c", "#8c96a0"
EXAMPLE = "race_05"                      # Tail of the Lake, four


def example(D):
    s = D.s[EXAMPLE]
    n = D.n(EXAMPLE)
    k = np.arange(n)
    p = D.order(EXAMPLE)
    fc = s["speed"] - D.R(EXAMPLE)[:, 0]                 # one-step forecast
    rows = [("power", "power"), ("rate", "rate, length"), ("length", "rate, length"), ("tech", "technique"), ("steer", "steering"),
            ("info", "information"), ("morale", "morale, tactics"), ("tactic", "morale, tactics"), ("other", "other calls"), ("count", "counts")]
    labels = list(dict.fromkeys(l for _, l in rows))
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True, constrained_layout=True, gridspec_kw=dict(height_ratios=[1.3, 0.7, 1.4]))
    ax[0].plot(k, s["speed"], color=INK, lw=0.7, label="speed, each stroke")
    ax[0].plot(k, fc, color=ACC, lw=0.6, alpha=0.8, label="one-step forecast (order %d)" % p)
    ax[0].set_ylabel("speed (m s$^{-1}$)")
    ax[0].set_ylim(np.percentile(s["speed"], 1) - 0.1, np.percentile(s["speed"], 99.5) + 0.1)
    ax[0].legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    ax[1].plot(k, s["rate"], color=INK, lw=0.7)
    ax[1].set_ylabel("rate (spm)")
    ax[1].set_ylim(np.percentile(s["rate"], 1) - 1, np.percentile(s["rate"], 99.5) + 1)
    for e in D.episodes(EXAMPLE):
        if 0 <= e["j"] < n:
            y = len(labels) - 1 - labels.index(dict(rows)[e["cls"]])
            ax[2].plot([e["j"], e["j"]], [y - 0.36, y + 0.36], color=ACC if e["cls"] == "power" else INK, lw=0.7)
    ax[2].set_yticks(range(len(labels)))
    ax[2].set_yticklabels(labels[::-1], fontsize=7.5)
    ax[2].set_ylim(-0.6, len(labels) - 0.4)
    ax[2].set_xlabel("stroke")
    ax[2].set_xlim(0, n)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(EXP, "fig_example." + ext), dpi=200)
    return dict(piece=EXAMPLE, strokes=int(n), order=int(p), calls=int(sum(e["cls"] != "count" for e in D.episodes(EXAMPLE) if 0 <= e["j"] < n)))


def channels(D):
    RNG = [D.valid(s_) for s_ in X.SES]
    chans = [("rate", "stroke rate (spm)", 1.0), ("dps", "distance per stroke (mm)", 1000.0), ("speed", "speed (mm s$^{-1}$)", 1000.0)]
    classes = [("power", "Power calls"), ("rate", "Rate and ratio calls"), ("length", "Length calls")]
    fig, ax = plt.subplots(3, 3, figsize=(7.2, 5.2), sharex=True, constrained_layout=True)
    out, h = {}, np.arange(1, 11)
    for a, (cls, title) in enumerate(classes):
        JS = [X.calls_of(D, s_, cls, lo, hi) for s_, (lo, hi) in zip(X.SES, RNG)]
        for b, (ch, lab, sc) in enumerate(chans):
            V = [D.R(s_, ch)[:, :10] for s_ in X.SES]
            eg, sg, pg = X.perm(V, JS, RNG, False, 900 + 10 * a + b)
            el, _, _ = X.perm(V, JS, RNG, True, 950 + 10 * a + b)
            x = ax[a, b]
            x.fill_between(h, -1.96 * sc * sg, 1.96 * sc * sg, color=GREY, alpha=0.25, lw=0)
            x.plot(h, sc * eg, color=ACC, lw=1.5)
            x.plot(h, sc * el, color=INK, lw=1, ls="--")
            x.axhline(0, color=INK, lw=0.5)
            x.set_xticks([1, 5, 10])
            if a == 0:
                x.set_title(lab, fontsize=8.5)
            if b == 0:
                x.set_ylabel("%s\n(n = %d)" % (title, sum(len(j) for j in JS)), fontsize=8)
            if a == 2:
                x.set_xlabel("strokes after the call")
            out["%s_%s" % (cls, ch)] = dict(est=(sc * eg).tolist(), sd=(sc * sg).tolist(), p=pg.tolist(), est_local=(sc * el).tolist())
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(EXP, "fig_channels." + ext), dpi=200)
    return out


def main():
    D = X.Data()
    out = dict(example=example(D), channels=channels(D))
    json.dump(out, open(os.path.join(EXP, "review", "figs_extra.json"), "w"), indent=1)
    print("written fig_example, fig_channels")


if __name__ == "__main__":
    main()
