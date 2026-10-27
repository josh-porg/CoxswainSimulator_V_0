"""The cross-flow blade inside the simulator, closed loop: phase 4.3 rung 1 on both athletes.

    .venv/Scripts/python.exe research/blade/crossflow_in_simulator.py [--strokes 10]

The review's blade (`DynamicOarSimulator(blade_law="liftdrag", blade_coefficients="crossflow@<face-on>")`, the fins'
cross-flow law, SOURCES §190-191) against the research profile's slip law and tier 2 [CG07], in the same run as
research/crew/handle_rung1.py: hands on the handle, [K05]'s on-water drive law on each athlete's own arc and rate,
nothing fitted; speed, IVV and handle power predicted, and the speed the athlete's measured power implies by the
cube law. Strips across the blade (the review's laws were tested in strips; the research slip law keeps its
centre read, as fitted). The face-on value comes from the Froude ordering (§191: ~2.2 below Fr ~0.9; [CR06] 0.61);
1.3 is run as the other end of the bracket.

Outputs: data/local/blade/crossflow_in_simulator.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "research", "crew"), ROOT]
import handle_rung1 as R                                          # noqa: E402

V = R.V
OUT = os.path.join(ROOT, "data", "local", "blade")
CONFIGS = {
    "research slip (fitted C2, centre)": {},
    "tier 2 CG07, strips": dict(blade_law="liftdrag", blade_coefficients="cg07", strips=True),
    "cross-flow C_Dc 2.2, strips": dict(blade_law="liftdrag", blade_coefficients="crossflow@2.2", strips=True),
    "cross-flow C_Dc 1.3, strips": dict(blade_law="liftdrag", blade_coefficients="crossflow@1.3", strips=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for name in ("cr06", "br24"):
        m = V.ATHLETES[name][0]()
        print("\n%s measured: %.3f m/s, IVV %.1f%%, %.0f W" % (name, m["speed"], 100 * m["ivv"], m["power"]), flush=True)
        for label, cfg in CONFIGS.items():
            cfg = dict(cfg)
            boat = R.boat_for(name, m, "on-water", drive_law=True)
            if cfg.pop("strips", False):
                cfg["blade_span"] = float(boat.rig.seats[0].oarlocks[0].oar.blade_length)
            r = R.predict(boat, a.strokes, **cfg)
            d = R.force_descriptors(r)
            at_power = r["speed"] * (m["power"] / r["power"]) ** (1.0 / 3.0)
            effs = [s.blade_efficiency for s in r["run"].strokes[-3:] if s.blade_efficiency is not None]
            eff = float(np.mean(effs)) if effs else float("nan")
            res = dict(speed=r["speed"], ivv=r["ivv"], power=r["power"], speed_at_power=at_power,
                       speed_error=at_power / m["speed"] - 1.0, ivv_points=100 * (r["ivv"] - m["ivv"]),
                       entry_to_peak=d["entry_to_peak"], peak_over_mean=d["peak_over_mean"], peak_at=d["peak_at"],
                       blade_efficiency=eff)
            out.setdefault(name, {})[label] = res
            print("  %-36s %.3f m/s, IVV %.1f%% (%+.1f pts), %3.0f W; at his/her power %+.1f%%; force peak at %.0f%% of "
                  "the drive (entry-to-peak %.2f s), peak/mean %.2f; blade efficiency %.2f (Kleshnev's band 0.75-0.82)" % (
                      label, r["speed"], 100 * r["ivv"], res["ivv_points"], r["power"], 100 * res["speed_error"],
                      100 * d["peak_at"], d["entry_to_peak"], d["peak_over_mean"], eff), flush=True)
    json.dump(out, open(os.path.join(OUT, "crossflow_in_simulator.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
