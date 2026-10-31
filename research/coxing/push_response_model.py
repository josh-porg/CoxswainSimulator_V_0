"""How fast can a push show up in the boat's speed? The simulator's step response against the measured push curve.

    .venv/Scripts/python.exe research/coxing/push_response_model.py [--settle 16] [--after 16] [--step 0.05]

A coxed four (catalogue 4+, shipped surge physics, 6-DOF) rows steady at the race rate; from one stroke on, every
seat's force rises by ``--step`` (power_scales). The per-stroke mean speed of that run minus the same run without the
step is the boat's own build-up to a sustained push: hull, crew and entrained water have to be accelerated against a
drag that grows with speed, so the gain approaches its plateau with a time constant tau = m v / (2 F_drag) in the
linearised surge balance. The baseline force is scaled so the steady speed matches each measured crew's mean speed
(tau grows as speed falls). Compared with the measured announced-push curve from the push's first stroke
(two_coxswains.py), normalised by its plateau: if the measured build is slower than the boat's, the crew is building
the effort, not just the boat its speed.

A second pair of runs at rate +-1 spm with the force profile unchanged gives the model's gearing at constant force,
d ln v / d ln r (compare with the crews' response to rate calls, two_coxswains.py).

Outputs: data/local/coxing/experiment/push_response_model.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from coxswain.boats import coxed_four  # noqa: E402
from coxswain.sim.simulator import RowingSimulator  # noqa: E402

EXP = os.path.join(ROOT, "data", "local", "coxing", "experiment")
SDIR = os.path.join(ROOT, "data", "local", "coxing", "sessions")


def stroke_means(res, period, n):
    t, u = res.time, res.states[6]
    return np.array([u[(t >= k * period) & (t < (k + 1) * period)].mean() for k in range(n)])


def run(rate, scale, n, step_at=None, step=0.0, v0=4.0):
    boat = coxed_four(rate=rate)
    sim = RowingSimulator(boat)
    period = float(boat.timing.period)

    def on_stroke(k, b):
        s = scale * (1.0 + (step if step_at is not None and k >= step_at else 0.0))
        b.power_scales = np.full(len(b.power_scales), s)
    res = sim.run(n * period, surge_speed=v0, on_stroke=on_stroke)
    return stroke_means(res, period, n), period


def crew_speed(sid):
    import csv
    rows = list(csv.DictReader(open(os.path.join(SDIR, sid, "boat.csv"))))
    v = np.array([float(r["speed"]) for r in rows])
    ra = np.array([float(r["rate"]) for r in rows])
    return float(np.median(v)), float(np.median(ra))


def step_shape(h, tau, T):
    """Per-stroke mean of 1 - exp(-t / tau) over stroke h (1-based) of period T: the linearised surge response."""
    return 1.0 - (tau / T) * (1.0 - np.exp(-T / tau)) * np.exp(-(h - 1) * T / tau)


def compare():
    """Fit tau to each model run, then the model's shape (one amplitude) to the measured push curves."""
    mod = json.load(open(os.path.join(EXP, "push_response_model.json")))
    meas = json.load(open(os.path.join(EXP, "two_coxswains.json")))["pushes"]
    out = {}
    for sid, m in mod.items():
        f = np.array(m["fraction"])
        h = np.arange(1, len(f) + 1)
        taus = np.linspace(1.0, 20.0, 400)
        err = [np.sum((step_shape(h, tau, m["period"]) - f) ** 2) for tau in taus]
        tau = float(taus[int(np.argmin(err))])
        mod[sid]["tau_s"] = tau
        print("%s model: tau %.1f s = %.1f strokes; share of the plateau at strokes 3-5: %.2f, by stroke 10: %.2f" % (
            sid, tau, tau / m["period"], f[2:5].mean(), f[9]))
    shape = np.mean([np.array(m["fraction"][:12]) for m in mod.values()], 0)
    for key in ("all_j_exec", "A_j_exec", "B_j_exec", "all_j_ann"):
        if key not in meas:
            continue
        c = np.array(meas[key]["curve"])
        amp = float(c @ shape / (shape @ shape))
        resid = c - amp * shape
        out[key] = dict(amplitude_mm_s=amp, rms_mm_s=float(np.sqrt(np.mean(resid ** 2))),
                        early_resid=float(resid[:4].mean()))
        print("measured %-10s plateau fitted %+.0f mm/s; residual rms %.0f mm/s; strokes 1-4 %+.0f mm/s against the boat's"
              " own build" % (key, amp, out[key]["rms_mm_s"], out[key]["early_resid"]))
    mod["_compare"] = out
    json.dump(mod, open(os.path.join(EXP, "push_response_model.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=int, default=16)
    ap.add_argument("--after", type=int, default=16)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--compare", action="store_true", help="only fit and compare the saved runs")
    a = ap.parse_args()
    if a.compare:
        return compare()
    out = {}
    n = a.settle + a.after
    targets = {"race_06": crew_speed("race_06"), "race_07": crew_speed("race_07")}
    for sid, (v_meas, r_meas) in targets.items():
        t0 = time.time()
        base, period = run(r_meas, 1.0, 8, v0=v_meas)          # short run: the default force's speed
        scale = (v_meas / base[-3:].mean()) ** 2                # drag ~ v^2, so force scales as v^2
        flat, period = run(r_meas, scale, n, v0=v_meas)
        stepd, _ = run(r_meas, scale, n, step_at=a.settle, step=a.step, v0=v_meas)
        resp = stepd[a.settle:] - flat[a.settle:]
        plateau = resp[-4:].mean()
        frac = resp / plateau
        # time constant from the stroke where the response passes 1 - 1/e
        k63 = int(np.argmax(frac >= 1 - np.exp(-1)))
        out[sid] = dict(measured_speed=v_meas, rate=r_meas, model_speed=float(flat[a.settle - 4:a.settle].mean()),
                        force_scale=float(scale), period=period, step=a.step, response_mm_s=(1000 * resp).tolist(),
                        plateau_mm_s=float(1000 * plateau), fraction=frac.tolist(), stroke_63=k63 + 1,
                        drift_mm_s=float(1000 * (flat[a.settle - 1] - flat[a.settle - 5])))
        print("%s: crew %.2f m/s at %.1f spm; model %.2f m/s (force x%.2f). +%.0f%% force -> plateau %+.0f mm/s;"
              " fraction by stroke: %s; 63%% at stroke %d (%.1f s); baseline drift %+.0f mm/s over 4 strokes [%.0f s]" % (
                  sid, v_meas, r_meas, out[sid]["model_speed"], scale, 100 * a.step, 1000 * plateau,
                  " ".join("%.2f" % f for f in frac[:10]), k63 + 1, (k63 + 1) * period, out[sid]["drift_mm_s"],
                  time.time() - t0), flush=True)
        # gearing at constant force: +-1 spm
        g = {}
        for dr in (-1.0, 1.0):
            vv, _ = run(r_meas + dr, scale, a.settle, v0=v_meas)
            g[dr] = vv[-4:].mean()
        el = np.log(g[1.0] / g[-1.0]) / np.log((r_meas + 1) / (r_meas - 1))
        out[sid]["gearing_constant_force"] = float(el)
        print("   gearing at constant force: %.2f m/s at %.1f spm, %.2f at %.1f -> d ln v / d ln r = %.2f" % (
            g[-1.0], r_meas - 1, g[1.0], r_meas + 1, el), flush=True)
    json.dump(out, open(os.path.join(EXP, "push_response_model.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
