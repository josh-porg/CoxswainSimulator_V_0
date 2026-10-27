"""Why is the cross-flow blade slow in the boat? Its pieces separated, closed loop, with and without the oar's bending.

    .venv/Scripts/python.exe research/blade/crossflow_decomposition.py [--strokes 10]

SOURCES §192: closed loop (rung 1, both athletes) the fins' cross-flow law is 7-9% slow at the athletes' power while
tier 2 [CG07] is fast. The two laws differ in two pieces: the normal load (Helmbold's low-AR lift and a face-on
cross-flow term against [CG07]'s sin forms) and the tangential load (drag-like in the cross-flow law, as [GTW21]
measures; thrust-like in [CG07]). Each is swapped on its own here, by blade objects that take one piece from each law:

- cross-flow normal, no tangential load;
- cross-flow normal, [CG07]'s tangential;
- [CG07] normal, the cross-flow law's (drag-like) tangential.

And each configuration with the scull's derived bending (coxswain/crew/oar_bending.py: [LNA15] through a uniform
cantilever; ~1 deg at 400 N m). Rung 1 as research/crew/handle_rung1.py (hands on the handle, [K05]'s drive law on each
athlete's own arc and rate), strips across the blade; speed at the athlete's measured power by the cube law.

Outputs: data/local/blade/crossflow_decomposition.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "research", "crew"), ROOT]
import handle_rung1 as R                                          # noqa: E402

from coxswain.crew.liftdrag import LiftDragBlade                  # noqa: E402
from coxswain.crew.oar_bending import OarBending                  # noqa: E402
from coxswain.sim.control import Coxswain                         # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator          # noqa: E402

V = R.V
OUT = os.path.join(ROOT, "data", "local", "blade")


@dataclass(frozen=True)
class Mixed(LiftDragBlade):
    """Normal coefficient from one blade, tangential from another (``None``: no tangential load)."""
    normal_from: object = None
    tangential_from: object = None

    def coefficients(self, alpha):
        c_n = self.normal_from.coefficients(alpha)[0]
        c_t = 0.0 if self.tangential_from is None else self.tangential_from.coefficients(alpha)[1]
        return float(c_n), float(c_t)


def blades(oar, area, density, face_on=2.2):
    cg = LiftDragBlade.big_blade(outboard=float(oar.blade_centre_outboard), area=area, density=density)
    cf = LiftDragBlade.crossflow(outboard=float(oar.blade_centre_outboard), area=area, length=float(oar.blade_length),
                                 face_on=face_on, density=density)
    base = dict(lift_amplitude=cg.lift_amplitude, drag_amplitude=cg.drag_amplitude, area=area,
                outboard=float(oar.blade_centre_outboard), density=density)
    return {
        "tier 2 CG07": cg,
        "cross-flow (fins' law), C_Dc 2.2": cf,
        "cross-flow normal, no tangential": Mixed(**base, normal_from=cf, tangential_from=None),
        "cross-flow normal, CG07 tangential": Mixed(**base, normal_from=cf, tangential_from=cg),
        "CG07 normal, cross-flow tangential": Mixed(**base, normal_from=cg, tangential_from=cf),
    }


def run(boat, blade, strokes, bending):
    boat.power_scales = np.ones(boat.n_seats)
    lock = boat.rig.seats[0].oarlocks[0]
    sim = DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="handle", blade_law="liftdrag",
                              blade_span=float(lock.oar.blade_length), oar_bending=bending, liftdrag_blade=blade,
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    r = sim.run_strokes(int(strokes), surge_speed=4.3)
    last = r.strokes[-3:]
    return (float(np.mean([s.mean_speed for s in last])), float(np.mean([s.surge_swing for s in last])),
            float(np.mean([s.handle_power for s in last])), float(np.max(np.abs(sim._bend))) if bending else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for name in ("cr06", "br24"):
        m = V.ATHLETES[name][0]()
        print("\n%s measured: %.3f m/s, IVV %.1f%%, %.0f W" % (name, m["speed"], 100 * m["ivv"], m["power"]), flush=True)
        probe = R.boat_for(name, m, "on-water", drive_law=True)
        oar = probe.rig.seats[0].oarlocks[0].oar
        bend = OarBending.laschowski2015_scull(float(oar.blade_centre_outboard), float(oar.blade_length))
        for label, blade in blades(oar, float(oar.blade_area), float(probe.water.density)).items():
            for bending in (None, bend):
                boat = R.boat_for(name, m, "on-water", drive_law=True)
                speed, ivv, power, phi = run(boat, blade, a.strokes, bending)
                at = speed * (m["power"] / power) ** (1.0 / 3.0)
                tag = label + (", bending" if bending else "")
                out.setdefault(name, {})[tag] = dict(speed=speed, ivv=ivv, power=power, speed_error=at / m["speed"] - 1.0,
                                                    ivv_points=100 * (ivv - m["ivv"]), bend_max_deg=float(np.degrees(phi)))
                print("  %-46s %.3f m/s, IVV %.1f%% (%+.1f), %3.0f W; at his/her power %+.1f%%%s" % (
                    tag, speed, 100 * ivv, 100 * (ivv - m["ivv"]), power, 100 * (at / m["speed"] - 1.0),
                    ("; last bend %.2f deg" % np.degrees(phi)) if bending else ""), flush=True)
    json.dump(out, open(os.path.join(OUT, "crossflow_decomposition.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
