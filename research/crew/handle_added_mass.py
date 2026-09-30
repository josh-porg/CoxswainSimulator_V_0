"""Phase 4.3, rung 1 with blade added mass: does the entrained water shape the force curve?

    python research/crew/handle_added_mass.py [--strokes 14]

Rung 1 (hands on the handle, research/crew/handle_rung1.py) with the [K05] on-water rhythm and
drive law and the tier 2 blade ([CG07] with strip integration) peaks too sharply: peak / mean
2.1-2.2 against the populations' 1.6-1.9. A quadratic slip law whose slip is zero at both ends
gives about 2 by construction (a sin^2 curve). Blade added mass loads the blade by its
acceleration instead, so it is not zero at entry. The pairing is clean: [CR06]'s rule puts the
blade in and out at zero normal velocity, so the entrained momentum m w_n is zero at both
switches (no impulse) and its work over the drive, -m [w_n^2 / 2], is zero.

Added mass as sourced: Patton (1965), AR-2 plate in unbounded fluid, or [LB19]'s C_m 0.7
measured near a surface. Nothing fitted. Prints speed, IVV and power per athlete, and the force
descriptors the populations print, plus the added mass's net work per stroke as a check.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import handle_rung1 as R                                         # noqa: E402

V = R.V


def added_mass_work(result):
    """Net work the entrained water does on the blades over the last stroke, J per oar."""
    sim, run = result["sim"], result["run"]
    if sim.blade_added_mass == "none":
        return 0.0
    t, y = run.last_time, run.last_states
    saved = sim.blade_added_mass
    power_with = np.array([sim.handle_torques(t[k], y[:, k], sim._air_mask[:, k])[0, 0] for k in range(t.size)])
    sim.blade_added_mass = "none"
    try:
        power_without = np.array([sim.handle_torques(t[k], y[:, k], sim._air_mask[:, k])[0, 0]
                                  for k in range(t.size)])
    finally:
        sim.blade_added_mass = saved
    rate = np.array([sim._sweep_motion(t[k] - sim._stroke_start)[0] for k in range(t.size)])
    return float(np.trapezoid((power_with - power_without) * rate, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=14)
    a = ap.parse_args()
    for name in ("br24", "cr06"):
        measure, _build = V.ATHLETES[name]
        m = measure()
        print("\n%s measured: %.3f m/s, IVV %.1f%%, %.0f W" % (name, m["speed"], 100 * m["ivv"], m["power"]))
        for mass in ("none", "patton", "labbe"):
            boat = R.boat_for(name, m, "on-water", True)
            span = float(boat.rig.seats[0].oarlocks[0].oar.blade_length)
            r = R.predict(boat, a.strokes, blade_law="liftdrag", blade_span=span, blade_added_mass=mass)
            d = R.force_descriptors(r)
            at_power = r["speed"] * (m["power"] / r["power"]) ** (1.0 / 3.0)
            print("  %-6s %.3f m/s, IVV %.1f%%, %.0f W; at %.0f W: %+.1f%%; force: entry/peak %.2f, "
                  "catch-to-peak %.2f s, peak at %.0f%% of blade-in, peak/mean %.2f, peak %.0f N; "
                  "added-mass work %+.2f J"
                  % (mass, r["speed"], 100 * r["ivv"], r["power"], m["power"],
                     100 * (at_power / m["speed"] - 1), d["entry_fraction"], d["catch_to_peak"],
                     100 * d["peak_at"], d["peak_over_mean"], d["peak"], added_mass_work(r)), flush=True)


if __name__ == "__main__":
    main()
