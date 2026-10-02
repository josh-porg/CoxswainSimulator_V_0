"""Blade immersion at the catch, validated against on-water populations. Nothing is fitted.

    python research/crew/immersion_study.py [--strokes 10] [--athletes br24,cr06]

Rung 1 (hands on the handle, [K05] on-water rhythm and drive law, [CG07] tier 2 with strips)
with the blade going in over BioRow's norm of 4 deg of oar travel after entry
(:mod:`coxswain.crew.blade_immersion`), with and without blade added mass scaled by the wetted
height, and a surface factor on it from Grift's surface/submerged ratio.

Scored against population measures only, each as its source defines it:

* [LE26] force at the catch / peak, 0.17 (about 100 N of 597 / 458 N per gate);
* [H20] catch slip / finish slip: oar travel with gate force below 196 N / 98 N, M1x 7.7 / 14.1
  deg, W1x 9.7 / 18.1 deg; gate force from the handle force by the lever, (r_h + l) / l;
* peak / mean force over the drive, [H20] 1.87-1.90, [K05] on water 1.76, [LE26] 1.61-1.68;
* catch to peak, [H20] 0.39-0.43 s, [LE26] 0.38-0.41 s, and, because the model's force has a
  plateau, the plateau-robust time to 90% of peak alongside;
* BioRow's timing norms, predicted rather than imposed: the blade's velocity turns driving about
  65 ms after the catch, and the catch slip (catch to fully buried) targets 6 deg;
* IVV, the athletes' own 49.1 / 49.4%.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import handle_rung1 as R                                         # noqa: E402

from coxswain.crew.blade_immersion import EntryImmersion       # noqa: E402

V = R.V

CONFIGS = [
    ("baseline (instant, no mass)", "none", None),
    ("immersion 4 deg", "none", EntryImmersion(4.0, 1.0)),
    ("immersion 4 deg + Patton", "patton", EntryImmersion(4.0, 1.0)),
    ("immersion 4 deg + Patton x0.5", "patton", EntryImmersion(4.0, 0.5)),
    ("immersion 4 deg + LB19", "labbe", EntryImmersion(4.0, 1.0)),
    ("immersion 3 deg + Patton x0.5", "patton", EntryImmersion(3.0, 0.5)),
    ("immersion 6 deg + Patton x0.5", "patton", EntryImmersion(6.0, 0.5)),
    ("instant + Patton", "patton", None),
]

#: second pass: how much added mass the population force curves tolerate, and how the catch
#: slip responds to a longer burial ([BR24]'s own is 7.5 deg, one athlete)
FOLLOWUP = [
    ("immersion 6 deg", "none", EntryImmersion(6.0, 1.0)),
    ("immersion 8 deg", "none", EntryImmersion(8.0, 1.0)),
    ("immersion 4 deg + Patton x0.1", "patton", EntryImmersion(4.0, 0.1)),
    ("immersion 4 deg + Patton x0.2", "patton", EntryImmersion(4.0, 0.2)),
]


def descriptors(result):
    """Population measures from the last stroke, per oar."""
    sim, run = result["sim"], result["run"]
    t, y = run.last_time, run.last_states
    lock = sim.boat.rig.seats[sim._seats[0]].oarlocks[0]
    inboard = float(lock.oar.inboard)
    lever = float(sim._oars[0].outboard)
    handle = np.array([-sim.handle_torques(t[k], y[:, k], sim._air_mask[:, k])[0, 0] / inboard
                       for k in range(t.size)])
    gate = handle * (inboard + lever) / lever
    tau = t - t[0]
    angle = np.array([sim._sweep_pose(x)[0] for x in tau])
    wet = ~sim._air_mask[0]
    drive = tau <= float(sim.boat.timing.drive_duration)
    k_peak = int(np.argmax(np.where(wet, handle, -np.inf)))
    peak = handle[k_peak]
    tw = tau[wet]
    first90 = tau[np.flatnonzero(wet & (handle >= 0.9 * peak))[0]]
    catch = float(angle[0])
    above = np.flatnonzero(drive & (gate >= 196.0))
    catch_slip = np.degrees(catch - angle[above[0]]) if above.size else np.nan
    below = np.flatnonzero(drive & (gate >= 98.0))
    finish = float(np.min(angle[drive]))
    finish_slip = np.degrees(angle[below[-1]] - finish) if below.size else np.nan
    entry_t = sim.entries[-1][1] - t[0] if sim.entries else np.nan
    entry_angle = sim.entries[-1][2] if sim.entries else np.nan
    buried = np.nan
    if sim.blade_immersion is not None and np.isfinite(entry_angle):
        buried = np.degrees(catch - entry_angle) + sim.blade_immersion.bury_deg
    return dict(entry_over_peak=float(handle[wet][0] / peak), peak=float(peak),
                peak_over_mean=float(peak / handle[wet].mean()),
                catch_to_peak=float(tau[k_peak]), catch_to_90=float(first90),
                catch_slip=float(catch_slip), finish_slip=float(finish_slip),
                entry_ms=1000.0 * float(entry_t), bioslip=float(buried),
                wet_s=float(tw[-1] - tw[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    ap.add_argument("--athletes", default="br24,cr06")
    ap.add_argument("--followup", action="store_true", help="run the second-pass configurations")
    a = ap.parse_args()
    configs = FOLLOWUP if a.followup else CONFIGS
    print("targets: entry/peak 0.17 [LE26]; catch slip 7.7-9.7, finish slip 14.1-18.1 deg [H20]; "
          "peak/mean 1.61-1.90; catch-to-peak 0.38-0.43 s; blade driving ~65 ms after the catch, "
          "catch slip 6 deg (BioRow norms)", flush=True)
    for name in a.athletes.split(","):
        measure, _build = V.ATHLETES[name]
        m = measure()
        print("\n%s measured: %.3f m/s, IVV %.1f%%, %.0f W" % (name, m["speed"], 100 * m["ivv"], m["power"]),
              flush=True)
        for label, mass, law in configs:
            boat = R.boat_for(name, m, "on-water", True)
            span = float(boat.rig.seats[0].oarlocks[0].oar.blade_length)
            r = R.predict(boat, a.strokes, blade_law="liftdrag", blade_span=span, blade_added_mass=mass,
                          blade_immersion=law)
            d = descriptors(r)
            at_power = r["speed"] * (m["power"] / r["power"]) ** (1.0 / 3.0)
            print("  %-30s IVV %4.1f%% %4.0f W (%+.1f%% at his/her W) | entry/peak %.2f, peak %4.0f N, "
                  "peak/mean %.2f, catch->peak %.2f s, ->90%% %.2f s | slips %.1f / %.1f deg | "
                  "entry %3.0f ms, catch-to-buried %.1f deg"
                  % (label, 100 * r["ivv"], r["power"], 100 * (at_power / m["speed"] - 1),
                     d["entry_over_peak"], d["peak"], d["peak_over_mean"], d["catch_to_peak"],
                     d["catch_to_90"], d["catch_slip"], d["finish_slip"], d["entry_ms"], d["bioslip"]),
                  flush=True)


if __name__ == "__main__":
    main()
