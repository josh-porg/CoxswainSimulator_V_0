"""Phase 4.3, rung 1: hands on the handle, on two measured scullers. Nothing is fitted.

    python research/crew/handle_rung1.py [--strokes 14]

``DynamicOarSimulator(crew="handle")``: the body is the research profile's prescribed body and
its hands stay on the handle, so the oar angle is the hands' sweep for the whole stroke and the
handle force is the constraint's reaction ([CR06]'s architecture, reproduced in
research/cr06/reproduce_model1.py). The blade enters and leaves at zero normal velocity. Speed,
speed fluctuation and handle power are all *predicted*: no power is imposed.

Each athlete's own rig and arc; the stroke timing is either the model's ergometer law
(Telfer) or [K05]'s on-water single-scull rhythm (sourced, predicts both athletes' drive
fractions to 0.006). Also reported: the speed the prediction implies at the athlete's measured
power by the cube law, so speed is compared at equal power.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "research", "blade"), os.path.join(ROOT, "research", "biorow")]
import validate_blade as V                                       # noqa: E402

from coxswain.crew.stroke import OnWaterTiming, StrokeTiming     # noqa: E402
from coxswain.sim.control import Coxswain                        # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator         # noqa: E402


def boat_for(name, measured, timing_kind):
    if name == "br24":
        import like_for_like as L
        timing = OnWaterTiming(L.RATE) if timing_kind == "on-water" else StrokeTiming(L.RATE)
        return L.build("arc", timing=timing)
    rate = 60.0 / V.CR06_T
    timing = OnWaterTiming(rate) if timing_kind == "on-water" else StrokeTiming(rate)
    return V.cr06_boat(measured, timing)


def predict(boat, strokes, **options):
    boat.power_scales = np.ones(boat.n_seats)
    sim = DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="handle",
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True,
                              **options)
    run = sim.run_strokes(int(strokes), surge_speed=4.3)
    last = run.strokes[-3:]
    return dict(speed=float(np.mean([s.mean_speed for s in last])),
                ivv=float(np.mean([s.surge_swing for s in last])),
                power=float(np.mean([s.handle_power for s in last])),
                drive_fraction=float(boat.timing.drive_fraction), sim=sim, run=run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=14)
    a = ap.parse_args()
    for name in ("br24", "cr06"):
        measure, _build = V.ATHLETES[name]
        m = measure()
        print("\n%s measured: %.3f m/s, IVV %.1f%%, %.0f W" % (name, m["speed"], 100 * m["ivv"], m["power"]))
        for kind in ("ergometer", "on-water"):
            r = predict(boat_for(name, m, kind), a.strokes)
            at_power = r["speed"] * (m["power"] / r["power"]) ** (1.0 / 3.0)
            print("  %-9s timing (drive %.3f): %.3f m/s, IVV %.1f%%, %.0f W; at his/her %.0f W: %.3f m/s (%+.1f%%)"
                  % (kind, r["drive_fraction"], r["speed"], 100 * r["ivv"], r["power"], m["power"],
                     at_power, 100 * (at_power / m["speed"] - 1)), flush=True)


if __name__ == "__main__":
    main()


def force_descriptors(result):
    """Handle force per oar through the last stroke, from the constraint, and the descriptors
    the population sources print: force at blade entry over peak, entry-to-peak time,
    peak / mean over the drive ([LE26]: 0.17, 0.38-0.41 s, 1.61-1.68; [H20]: -, 0.39-0.43 s,
    1.87-1.90; [K05]: peak at 34.7% of the drive)."""
    sim, run = result["sim"], result["run"]
    t, y = run.last_time, run.last_states
    start = t[0]
    oar = sim._oars[0]
    lock = sim.boat.rig.seats[sim._seats[0]].oarlocks[0]
    inertia = float(getattr(oar.inertia, "oar_inertia", 0.0))
    inboard = float(lock.oar.inboard)
    from coxswain.core.state import State
    force = np.zeros_like(t)
    for k in range(t.size):
        rate, accel = sim._sweep_motion(t[k] - start)
        angle = sim._sweep_pose(t[k] - start)[0]
        torque = inertia * accel
        if not sim._air_mask[0, k]:
            f_n, _f, arm = sim._blade_loads_arm(0, angle, rate, State.from_vector(y[:len(y) - 2 * sim.n_oar_states, k]), lock)
            torque -= arm * f_n
        force[k] = -torque / inboard                    # pull on the handle, N per oar
    wet = ~sim._air_mask[0]
    tw = t[wet]
    fw = force[wet]
    k_peak = int(np.argmax(fw))
    return dict(entry_fraction=float(fw[0] / fw[k_peak]), entry_to_peak=float(tw[k_peak] - tw[0]),
                peak_over_mean=float(fw[k_peak] / fw.mean()), peak_at=float((tw[k_peak] - tw[0]) / (tw[-1] - tw[0])),
                peak=float(fw[k_peak]))
