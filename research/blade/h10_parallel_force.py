"""The full-scale parallel blade force ([H10]) against the review blade's tangential load on measured scull paths.

    .venv/Scripts/python.exe research/blade/h10_parallel_force.py

[H10] (Hofmijster, de Koning & van Soest 2010): one world-class female sculler, instrumented oars and oarlocks,
30-32 spm. Fig. 9 (traced: data/literature/hofmijster2010_fig9_parallel.csv) is the force along the shaft; Figs 8 and
10 (read by eye: hofmijster2010_fig8_fig10_read.csv) the blade angle and the lift and drag. Negative = inward on the
blade, the convention of LiftDragBlade.loads (F_t along the shaft, outward positive).

The review blade (the fins' cross-flow law, C_Dc 2.2) is evaluated on two measured scull paths ([CR06], another elite
female sculler at a similar rate; [BR24]) with research/blade/blade_path_in_boat.py's formula, over the loaded drive:
from the catch to the release by [CR06]'s rule (the blade leaves when it stops driving, w_n >= 0, after the load peak).
Compared by fraction of the drive:

1. the ratio of the tangential load to the whole blade force, F_t / |F|;
2. where it changes sign, against where the blade is square to the boat;
3. the tangential share of blade losses ([H10] Table I: 45.7 W without, 53.7 W with the parallel force).

And the one number a still-water law cannot supply: the along-shaft speed of the water at [H10]'s sign change that
would cancel the blade's still-water sliding there, against the blade's own speed l theta_dot. If the water near the
blade is carried outward (H10's first explanation; the spanwise flow of a revolving plate), this is how fast.

Outputs: data/local/blade/h10_parallel_force.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import blade_path_in_boat as B                                   # noqa: E402
from coxswain.crew.liftdrag import LiftDragBlade                  # noqa: E402

LIT = os.path.join(ROOT, "data", "literature")
OUT = os.path.join(ROOT, "data", "local", "blade")
H10_LOSS = (45.7, 53.7)                                           # W, both blades, Table I (deformable or rigid alike)
RIGS = {"cr06": dict(l=1.805, area=0.0903, length=0.43), "br24": dict(l=None, area=0.083, length=0.43)}


def read_csv(name):
    rows = [l.strip().split(",") for l in open(os.path.join(LIT, name), encoding="utf-8") if not l.startswith("#")]
    head, body = rows[0], np.array(rows[1:], float)
    return {h: body[:, k] for k, h in enumerate(head)}


def h10():
    p = read_csv("hofmijster2010_fig9_parallel.csv")
    k = read_csv("hofmijster2010_fig8_fig10_read.csv")
    t = p["t_s"]
    lift, drag = np.interp(t, k["t_s"], k["lift_N"]), np.interp(t, k["t_s"], k["drag_N"])
    total = np.hypot(lift, drag)
    angle = np.interp(t, k["t_s"], k["blade_angle_deg"])
    start = k["t_s"][np.argmax(k["blade_angle_deg"] > 30.5)] - 0.05      # the blade starts to turn (~0.07 s)
    frac = (t - start) / (t[-1] - start)
    cross = float(t[np.argmax(p["F_parallel_N"] > 0)])
    square = float(np.interp(90.0, angle, t))
    return dict(t=t, frac=frac, f_par=p["F_parallel_N"], total=total, angle=angle, cross=cross, square=square,
                frac_cross=float(np.interp(cross, t, frac)), frac_square=float(np.interp(square, t, frac)),
                angle_at_cross=float(np.interp(cross, t, angle)), share=1 - H10_LOSS[0] / H10_LOSS[1])


def law_on_path(name):
    t, a, ad, u = B.measured_path(name)
    rig = RIGS[name]
    l = rig["l"] or 1.8
    law = LiftDragBlade.crossflow(outboard=l, area=rig["area"], length=rig["length"], face_on=2.2)
    i0 = int(np.argmax(a))
    t, a, ad, u = t[i0:], a[i0:], ad[i0:], u[i0:]
    w = np.array([law.relative_velocity(x, r, uu, 1) for x, r, uu in zip(a, ad, u)])
    f = np.array([law.loads(x, r, uu, 1) for x, r, uu in zip(a, ad, u)])
    peak = int(np.argmax(np.abs(f[:, 0]) * (np.arange(len(t)) < len(t) // 2)))
    rel = peak + int(np.argmax(w[peak:, 0] >= 0.0))                 # release: stops driving after the load peak
    s = slice(0, rel)
    t, a, ad, u, w, f = t[s], a[s], ad[s], u[s], w[s], f[s]
    frac = (t - t[0]) / (t[-1] - t[0])
    total = np.hypot(f[:, 0], f[:, 1])
    pn, pt = np.abs(f[:, 0] * w[:, 0]), np.abs(f[:, 1] * w[:, 1])
    sq = float(frac[np.argmax(a <= 0.0)])
    cross = float(frac[np.argmax((f[:, 1] > 0) & (frac > 0.2))])
    return dict(frac=frac, ratio=f[:, 1] / np.maximum(total, 1e-9), total=total, w_a=w[:, 1], blade_speed=l * np.abs(ad),
                angle=np.degrees(a), drive_s=float(t[-1] - t[0]), frac_square=sq, frac_cross=cross,
                share=float(np.trapezoid(pt, t) / np.trapezoid(pn + pt, t)))


def main():
    os.makedirs(OUT, exist_ok=True)
    H = h10()
    print("[H10] parallel force %.1f to %+.1f N; changes sign at %.2f s (%.0f%% of the drive, blade at %.0f deg to the boat);"
          " square to the boat at %.2f s (%.0f%%); share of blade losses %.1f%%" % (
              H["f_par"].min(), H["f_par"].max(), H["cross"], 100 * H["frac_cross"], H["angle_at_cross"], H["square"],
              100 * H["frac_square"], 100 * H["share"]))
    out = {"h10": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in H.items()}}
    bins = np.linspace(0, 1, 11)
    hr = [np.mean((H["f_par"] / np.maximum(H["total"], 1))[(H["frac"] >= a) & (H["frac"] < b)]) for a, b in zip(bins[:-1], bins[1:])]
    print("  F_par/|F| by tenths: " + " ".join("%+.2f" % x for x in hr))
    out["h10"]["ratio_tenths"] = hr
    for name in ("cr06", "br24"):
        L = law_on_path(name)
        lr = [np.mean(L["ratio"][(L["frac"] >= a) & (L["frac"] < b)]) for a, b in zip(bins[:-1], bins[1:])]
        i = int(np.argmin(np.abs(L["frac"] - H["frac_cross"])))
        need, speed = -L["w_a"][i], L["blade_speed"][i]
        print("\n%s path, review blade (loaded drive %.2f s): F_t/|F| by tenths: %s" % (
            name, L["drive_s"], " ".join("%+.2f" % x for x in lr)))
        print("  changes sign at %.0f%% of the drive; blade square at %.0f%%. Tangential share of blade losses %.1f%%"
              " (H10 %.1f%%)" % (100 * L["frac_cross"], 100 * L["frac_square"], 100 * L["share"], 100 * H["share"]))
        print("  at H10's sign change (%.0f%% of the drive, oar %+.0f deg): still-water sliding %.2f m/s along the shaft;"
              " the water would have to move %+.2f m/s along the shaft to cancel it, %.0f%% of the blade's own speed %.2f m/s"
              % (100 * H["frac_cross"], L["angle"][i], L["w_a"][i], need, 100 * abs(need) / speed, speed))
        out[name] = dict(ratio_tenths=lr, frac_cross=L["frac_cross"], frac_square=L["frac_square"], share=L["share"],
                         sliding_at_h10_cross=float(L["w_a"][i]), blade_speed_at_h10_cross=float(speed),
                         drive_s=L["drive_s"])
    json.dump(out, open(os.path.join(OUT, "h10_parallel_force.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
