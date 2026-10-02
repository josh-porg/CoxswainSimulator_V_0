"""Given a population's hands, does the model produce that population's handle force?

    python research/crew/k05_population_test.py [--strokes 10]

[K05] Fig. 1 measured, for the same five women on the water at racing rate, both the handle's
speed and the handle's force against drive length (both digitised into
``data/literature/k05_fig1_onwater.csv``; the force panel reproduces the paper's own table:
612 N at 34.0% against 602 N at 34.7%, average / max 58.4% against 56.9%). So the blade and hull
physics can be tested on a population with nothing fitted: put the population's hands on the
handle (rung 1, [K05]'s rhythm and drive law) in a single like theirs and compare the handle
force that comes out with the one they measured.

The single: [CR06]'s women's rig (arc 104.8 deg, 1.52 m of handle path against [K05]'s 1.59 m,
so forces are compared against the fraction of drive length), with [K05]'s crew, 1.80 m and
72.2 kg, at 32.3 spm. Boat speed is not published, so the force *shape* is the test; the level is
reported against [K05]'s table (max 602 N, average 342 N, 391 W) with the caveat that the paper's
power and force are not mutually consistent on a single oar or two (SOURCES sec. 174).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import handle_rung1 as R                                         # noqa: E402

from coxswain.crew.blade_immersion import EntryImmersion       # noqa: E402
from coxswain.crew.drive_law import K05_FIG1                    # noqa: E402
from coxswain.crew.stroke import OnWaterTiming                  # noqa: E402

V = R.V
K05_RATE, K05_MASS, K05_STATURE = 32.3, 72.2, 1.80

CONFIGS = [
    ("slip, research C2 (one-athlete fit)", dict()),
    ("K05 body + slip, research C2", dict(body=True)),
    ("K05 body + tier 2 [CG07] + strips + immersion", dict(body=True, blade_law="liftdrag", strips=True,
                                                          immersion=True)),
    ("K05 body + tier 2 Coppel + strips + immersion", dict(body=True, blade_law="liftdrag",
                                                          blade_coefficients="coppel", strips=True,
                                                          immersion=True)),
    ("tier 2 [CG07]", dict(blade_law="liftdrag")),
    ("tier 2 [CG07] + strips", dict(blade_law="liftdrag", strips=True)),
    ("tier 2 [CG07] + strips + immersion", dict(blade_law="liftdrag", strips=True, immersion=True)),
    ("tier 2 Coppel + strips + immersion", dict(blade_law="liftdrag", blade_coefficients="coppel",
                                                strips=True, immersion=True)),
]


def k05_acceleration():
    path = os.path.join(os.path.dirname(K05_FIG1), "k05_fig1_boat_acceleration.csv")
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    return (np.array([float(r["cycle_pct"]) for r in rows]) / 100.0,
            np.array([float(r["acceleration_mps2"]) for r in rows]))


def acceleration_match(result):
    """The model's hull surge acceleration over the last stroke against [K05]'s, both on a cycle
    clock with the catch dip (minimum) aligned: rms, and the extremes."""
    run = result["run"]
    t, y = run.last_time, run.last_states
    v = np.hypot(y[6], y[7])
    a = np.gradient(v, t)
    ph = (t - t[0]) / (t[-1] - t[0])
    pk, ak = k05_acceleration()
    grid = np.linspace(0.0, 1.0, 200, endpoint=False)
    am = np.interp(np.mod(grid + ph[int(np.argmin(a))], 1.0), ph, a)
    ar = np.interp(np.mod(grid + pk[int(np.argmin(ak))], 1.0), pk, ak, period=1.0)
    return float(np.sqrt(np.mean((am - ar) ** 2))), float(a.min()), float(a.max())


def k05_force():
    rows = list(csv.DictReader(l for l in open(K05_FIG1, encoding="utf-8") if not l.startswith("#")))
    s = np.array([float(r["length_pct"]) for r in rows]) / 100.0
    return s, np.array([float(r["handle_force_N"]) for r in rows])


def model_force(result):
    """Handle force per oar against drive-length fraction over the drive, from the constraint."""
    sim, run = result["sim"], result["run"]
    t, y = run.last_time, run.last_states
    lock = sim.boat.rig.seats[sim._seats[0]].oarlocks[0]
    inboard = float(lock.oar.inboard)
    oar = sim._oars[0]
    catch, finish = float(oar.catch_angle), float(oar.finish_angle)
    tau = t - t[0]
    drive = tau <= float(sim.boat.timing.drive_duration)
    angle = np.array([sim._sweep_pose(x)[0] for x in tau])
    force = np.array([-sim.handle_torques(t[k], y[:, k], sim._air_mask[:, k])[0, 0] / inboard
                      for k in range(t.size)])
    s = (catch - angle) / (catch - finish)
    keep = drive & (s >= 0.0) & (s <= 1.0)
    return s[keep], force[keep], tau[keep]


def describe(s, f, tau):
    k = int(np.argmax(f))
    mean_t = np.trapezoid(f, tau) / (tau[-1] - tau[0])
    return dict(peak=float(f[k]), at=float(100 * s[k]), avg_max=float(100 * mean_t / f[k]),
                early=float(np.interp(0.05, s, f)))


def predict(boat, strokes, body=False, **options):
    """handle_rung1.predict, with the body optionally re-timed onto [K05]'s legs and trunk."""
    from coxswain.sim.control import Coxswain
    from coxswain.sim.dynamic_oar import DynamicOarSimulator
    boat.power_scales = np.ones(boat.n_seats)
    sim = DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="handle",
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True, **options)
    if body:
        import k05_body
        sim.crew_field, _K = k05_body.body_field(boat)
    run = sim.run_strokes(int(strokes), surge_speed=4.3)
    last = run.strokes[-3:]
    return dict(speed=float(np.mean([s.mean_speed for s in last])),
                ivv=float(np.mean([s.surge_swing for s in last])),
                power=float(np.mean([s.handle_power for s in last])), sim=sim, run=run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--match", default="", help="run only configurations whose label contains this")
    a = ap.parse_args()
    sk, fk = k05_force()
    grid = np.linspace(0.02, 0.98, 49)
    ref = np.interp(grid, sk, fk)
    print("[K05] measured: max %.0f N at %.1f%%, average/max 56.9%% (table), force at 5%% of length %.0f N"
          % (fk.max(), 100 * sk[int(np.argmax(fk))], np.interp(0.05, sk, fk)), flush=True)
    measured = dict(leg_travel=None)
    for label, opts in CONFIGS:
        if a.match and a.match not in label:
            continue
        boat = V.cr06_boat(measured, timing=OnWaterTiming(K05_RATE), stature=K05_STATURE, drive_law=True,
                           mass=K05_MASS)
        kw = {k: v for k, v in opts.items() if k not in ("strips", "immersion", "body")}
        if opts.get("strips"):
            kw["blade_span"] = float(boat.rig.seats[0].oarlocks[0].oar.blade_length)
        if opts.get("immersion"):
            kw["blade_immersion"] = EntryImmersion(4.0, 0.0)
        r = predict(boat, a.strokes, body=opts.get("body", False), **kw)
        s, f, tau = model_force(r)
        d = describe(s, f, tau)
        model = np.interp(grid, s, f)
        shape = np.sqrt(np.mean((model / model.max() - ref / ref.max()) ** 2))
        early = np.sqrt(np.mean(((model / model.max() - ref / ref.max())[grid <= 0.3]) ** 2))
        late = np.sqrt(np.mean(((model / model.max() - ref / ref.max())[grid >= 0.6]) ** 2))
        print("  %-40s %.3f m/s, IVV %.1f%%, %.0f W | max %.0f N at %.0f%%, avg/max %.0f%%, at 5%% %.0f N | "
              "shape rms %.3f (0-30%% %.3f, 60-100%% %.3f)"
              % (label, r["speed"], 100 * r["ivv"], r["power"], d["peak"], d["at"], d["avg_max"], d["early"],
                 shape, early, late), flush=True)
        rms, amin, amax = acceleration_match(r)
        print("      boat acceleration: min %.1f, max %.1f m/s^2, rms against [K05] %.2f ([K05] -7.9 / 3.4)"
              % (amin, amax, rms), flush=True)
        print("      force/max at 10/20/35/50/70/90%%: model %s | [K05] %s"
              % (" ".join("%.2f" % (np.interp(x, s, f) / d["peak"]) for x in (0.1, 0.2, 0.35, 0.5, 0.7, 0.9)),
                 " ".join("%.2f" % (np.interp(x, sk, fk) / fk.max()) for x in (0.1, 0.2, 0.35, 0.5, 0.7, 0.9))),
              flush=True)


if __name__ == "__main__":
    main()
