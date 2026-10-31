"""The review blade closed loop with its tangential load scaled, read at [H10]'s measured share of blade losses.

    .venv/Scripts/python.exe research/blade/tangential_level_sweep.py [--strokes 10]

SOURCES §199-200. The quasi-steady tangential load is too large mid-drive on both measured loads (about 2x in
[GTW21]'s tank, about 5x on the water in [H10]) and out of phase with the sliding on both. Its level for a full-scale
blade on the water is what [H10] measured: 14.9 % of blade losses (Table I, 45.7 -> 53.7 W). This script does not fit
anything to the test athletes: it runs rung 1 (research/crew/handle_rung1.py: hands on the handle, [K05]'s drive law,
each athlete's own arc and rate; strips; rigid oar) with the cross-flow normal load and the cross-flow tangential load
times k, k = 0 ... 1, and records for each run its speed at the athlete's measured power and the tangential share of
blade losses on its own simulated path (blade centre, loaded drive: catch to the release by [CR06]'s rule). The speed at
[H10]'s share is then read by interpolation in the share. The level comes from an independent sculler; the athletes
only test it.

Outputs: data/local/blade/tangential_level_sweep.json.
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
sys.path[:0] = [HERE, os.path.join(ROOT, "research", "crew"), ROOT]
import crossflow_decomposition as D                               # noqa: E402
import handle_rung1 as R                                          # noqa: E402

from coxswain.core.state import STATE_SIZE, State                 # noqa: E402
from coxswain.crew.liftdrag import LiftDragBlade                  # noqa: E402
from coxswain.sim.control import Coxswain                         # noqa: E402
from coxswain.sim.dynamic_oar import DynamicOarSimulator          # noqa: E402

V = R.V
OUT = os.path.join(ROOT, "data", "local", "blade")
H10_SHARE = 1.0 - 45.7 / 53.7
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class ScaledTangential(D.Mixed):
    """The cross-flow law with its tangential coefficient times ``tangential_scale``."""
    tangential_scale: float = 1.0

    def coefficients(self, alpha):
        c_n, c_t = D.Mixed.coefficients(self, alpha)
        return c_n, float(self.tangential_scale * c_t)


def share_on_path(sim, run, blade):
    t, Y = run.last_time, run.last_states
    n = sim.n_oar_states
    angle, rate = Y[STATE_SIZE], Y[STATE_SIZE + n]
    lock = sim.boat.rig.seats[sim._seats[0]].oarlocks[0]
    u = np.array([sim._lock_velocity(State.from_vector(Y[:STATE_SIZE, k]), lock)[:2] for k in range(len(t))])
    u = u * np.array([1.0, int(lock.side)])
    i0 = int(np.argmax(angle))
    t, angle, rate, u = t[i0:], angle[i0:], rate[i0:], u[i0:]
    w = np.array([blade.relative_velocity(a, r, uu, 1) for a, r, uu in zip(angle, rate, u)])
    f = np.array([blade.loads(a, r, uu, 1) for a, r, uu in zip(angle, rate, u)])
    peak = int(np.argmax(np.abs(f[:, 0]) * (np.arange(len(t)) < len(t) // 2)))
    rel = peak + int(np.argmax(w[peak:, 0] >= 0.0))
    s = slice(0, max(rel, 2))
    pn = np.trapezoid(np.abs(f[s, 0] * w[s, 0]), t[s])
    pt = np.trapezoid(np.abs(f[s, 1] * w[s, 1]), t[s])
    return float(pt / (pn + pt)) if pn + pt > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for name in ("cr06", "br24"):
        m = V.ATHLETES[name][0]()
        print("\n%s measured: %.3f m/s, %.0f W" % (name, m["speed"], m["power"]), flush=True)
        probe = R.boat_for(name, m, "on-water", drive_law=True)
        oar = probe.rig.seats[0].oarlocks[0].oar
        area, rho = float(oar.blade_area), float(probe.water.density)
        cf = LiftDragBlade.crossflow(outboard=float(oar.blade_centre_outboard), area=area,
                                     length=float(oar.blade_length), face_on=2.2, density=rho)
        rows = []
        for k in SCALES:
            blade = ScaledTangential(lift_amplitude=cf.lift_amplitude, drag_amplitude=cf.drag_amplitude, area=area,
                                     outboard=float(oar.blade_centre_outboard), density=rho, normal_from=cf,
                                     tangential_from=cf, tangential_scale=k)
            boat = R.boat_for(name, m, "on-water", drive_law=True)
            boat.power_scales = np.ones(boat.n_seats)
            sim = DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="handle", blade_law="liftdrag",
                                      blade_span=float(oar.blade_length), liftdrag_blade=blade,
                                      coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
            run = sim.run_strokes(a.strokes, surge_speed=4.3)
            last = run.strokes[-3:]
            speed = float(np.mean([s.mean_speed for s in last]))
            power = float(np.mean([s.handle_power for s in last]))
            ivv = float(np.mean([s.surge_swing for s in last]))
            err = speed * (m["power"] / power) ** (1 / 3) / m["speed"] - 1.0
            sh = share_on_path(sim, run, blade)
            rows.append(dict(scale=k, speed=speed, power=power, ivv=ivv, speed_error=err, share=sh))
            print("  tangential x%.2f: %.3f m/s, %3.0f W, IVV %.1f%%; at the measured power %+.1f%%; tangential share of"
                  " blade losses %.1f%%" % (k, speed, power, 100 * ivv, 100 * err, 100 * sh), flush=True)
        sh = np.array([r["share"] for r in rows])
        er = np.array([r["speed_error"] for r in rows])
        o = np.argsort(sh)
        at = float(np.interp(H10_SHARE, sh[o], er[o]))
        print("  at [H10]'s share (%.1f%%): speed at the measured power %+.1f%%" % (100 * H10_SHARE, 100 * at), flush=True)
        out[name] = dict(rows=rows, h10_share=H10_SHARE, speed_error_at_h10=at, measured=m)
    json.dump(out, open(os.path.join(OUT, "tangential_level_sweep.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
