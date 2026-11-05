"""How often the quasi-static bending solve finds its root in a boat (SOURCES §217).

``DynamicOarSimulator._update_bending`` solves ``K (phi + dt phi') = l F_n(angle + phi + dt phi', rate + k phi')`` for
the hinge rate ``phi'`` by bracketing in [-phi/dt - 60, 60] rad/s; with no sign change it takes the nearer end. This
counts those fallbacks per stroke and where in the drive they happen, for a boat under the research profile.

    python research/blade/bending_solve_check.py [--boat 8+] [--torque 500] [--strokes 6] [--dt 0.001]
"""
import argparse

import numpy as np
from scipy import optimize

from coxswain import physics
from coxswain.boats import catalog
from coxswain.sim.control import Coxswain
from coxswain.sim.dynamic_oar import STATE_SIZE, DynamicOarSimulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boat", default="8+")
    ap.add_argument("--torque", type=float, default=500.0, help="peak handle torque, N m")
    ap.add_argument("--strokes", type=int, default=6)
    ap.add_argument("--bending", default="auto", choices=["auto", "sweep", "scull"],
                    help="the law: auto = [RLBQV19] on a sweep rig, [L14] on a scull (no profile bends by default)")
    ap.add_argument("--dt", type=float, default=None, help="integration step, s (default: the simulator's own)")
    args = ap.parse_args()

    profile = physics.resolve("research")
    boat = profile.apply(catalog.build(args.boat, rate=28.0))
    boat.power_scales = np.ones(boat.n_seats)
    own = profile.blade_options(boat)
    from coxswain.crew.oar_bending import OarBending
    oar = boat.rig.seats[0].oarlocks[0].oar
    law = args.bending if args.bending != "auto" else ("sweep" if boat.rig.is_sweep else "scull")
    bending = (OarBending.robert2019_sweep() if law == "sweep"
               else OarBending.laschowski2014_scull(float(oar.blade_centre_outboard), float(oar.blade_length)))
    sim = DynamicOarSimulator(boat, peak_torque=args.torque, catch=profile.catch, blade_law=own["blade_law"],
                              oar_bending=bending,
                              blade_span=own["blade_span"], blade_coefficients=own["blade_coefficients"],
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)

    calls = {"solved": 0, "fallback": 0, "where": []}
    real = optimize.brentq

    def counting(f, a, b, *pa, **kw):
        calls["solved"] += 1
        return real(f, a, b, *pa, **kw)

    optimize.brentq = counting              # _update_bending imports brentq at call time
    original = sim._update_bending

    def watched(y, step):
        before = calls["solved"]
        original(y, step)
        wet = sum(1 for s in range(sim.n_oar_states) if sim._in_air is None or not sim._in_air[s])
        missed = wet - (calls["solved"] - before)
        if missed > 0:
            calls["fallback"] += missed
            calls["where"].append(float(np.degrees(y[STATE_SIZE])))

    sim._update_bending = watched
    run = sim.run_strokes(args.strokes, surge_speed=5.0, dt=args.dt)
    optimize.brentq = real
    print("%s, peak torque %.0f N m, K_ang %.0f: %.2f m/s; bend solves %d, fallbacks to a bracket end %d"
          % (args.boat, args.torque, sim.oar_bending.k_ang, run.settled_speed(), calls["solved"], calls["fallback"]))
    if calls["where"]:
        w = np.asarray(calls["where"])
        print("  fallbacks at oar angles (deg): min %.1f, median %.1f, max %.1f" % (w.min(), np.median(w), w.max()))
    phi = np.abs(sim._bend_history[0])
    print("  last stroke: peak bend %.1f deg, peak bend rate %.1f rad/s"
          % (np.degrees(phi.max()), np.abs(sim._bend_history[1]).max()))
    # where the bend peaks: the oar's angle against its catch and finish, and the time since the catch
    t = run.last_time
    slot, k = np.unravel_index(np.argmax(phi), phi.shape)
    oar = sim._oars[slot]
    angle = np.degrees(run.last_states[STATE_SIZE + slot, k])
    entry = [e for e in sim.entries if e[0] == slot and e[1] <= t[k]]
    print("  peak on oar %d at %.1f deg (catch %.1f, finish %.1f), %.3f s after the stroke start, %s"
          % (slot, angle, np.degrees(oar.catch_angle), np.degrees(oar.finish_angle), t[k] - t[0],
             "%.3f s after the blade's entry at %.1f deg" % (t[k] - entry[-1][1], np.degrees(entry[-1][2]))
             if entry else "before the entry"))
    for frac in (0.0, 0.1, 0.25, 0.5, 0.75):
        j = int(frac * (len(t) - 1) * 0.6)
        print("    t+%.3f s: oar %.1f deg, bend %.1f deg" % (t[j] - t[0], np.degrees(run.last_states[STATE_SIZE + slot, j]),
                                                         np.degrees(sim._bend_history[0, slot, j])))


if __name__ == "__main__":
    main()
