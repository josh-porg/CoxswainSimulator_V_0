"""Where the model's blade wastes its energy through the drive (SOURCES §221).

Like for like, the model's blade is ~0.05 less efficient than Kleshnev's measured blades (§220). This settles a boat
under the research profile (optionally bending with the crew's hold, as §220) and splits the last stroke's handle
energy and blade waste, on the blade's own track and on the gate track, into tenths of the drive (catch 0 -> finish 1),
summed over every oar. A loss concentrated at the catch or the release points at the entry, immersion or release; a
loss spread through the drive points at the blade law's level.

    python research/blade/blade_waste_profile.py [--boat 8+] [--bending] [--hold 4] [--watts 360]
"""
import argparse

import numpy as np

from coxswain import physics
from coxswain.boats import catalog
from coxswain.core.state import STATE_SIZE, State
from coxswain.crew.oar_bending import OarBending
from coxswain.sim.control import Coxswain
from coxswain.sim.dynamic_oar import DynamicOarSimulator
from coxswain.validation import scorecard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boat", default="8+")
    ap.add_argument("--bending", action="store_true", help="the Concept2-derived sweep shaft (or [L14] on sculls)")
    ap.add_argument("--hold", type=float, default=None, help="the crew's hold on its planned speed, beta")
    ap.add_argument("--watts", type=float, default=360.0)
    args = ap.parse_args()

    profile = physics.resolve("research")
    boat = profile.apply(catalog.build(args.boat, rate=28.0))
    boat.power_scales = np.ones(boat.n_seats)
    oar0 = boat.rig.seats[0].oarlocks[0].oar
    if args.bending:
        boat.oar_bending = (OarBending.concept2_sweep(float(oar0.blade_centre_outboard), float(oar0.blade_length))
                            if boat.rig.is_sweep else
                            OarBending.laschowski2014_scull(float(oar0.blade_centre_outboard), float(oar0.blade_length)))
    fv = None
    if args.hold is not None:
        from coxswain.crew.handle_fv import PlannedSpeedHold
        fv = PlannedSpeedHold(args.hold)
    own = profile.blade_options(boat)
    kw = dict(catch=profile.catch, blade_law=own["blade_law"], blade_span=own["blade_span"],
              blade_coefficients=own["blade_coefficients"])
    torque = DynamicOarSimulator.torque_for_power(boat, args.watts, start=5.0, handle_fv=fv, **kw)
    sim = DynamicOarSimulator(boat, peak_torque=torque, handle_fv=fv,
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True, **kw)
    run = sim.run_strokes(scorecard.SETTLE_STROKES, surge_speed=5.0)
    t, y = run.last_time, run.last_states
    n = sim.n_oar_states
    air = sim._air_mask
    hist = sim._bend_history if sim.oar_bending is not None else None
    edges = np.linspace(0.0, 1.0, 11)
    handle = np.zeros(10)                 # gate track: F_n u, u from the shaft's rate (the handle's work)
    handle_blade = np.zeros(10)           # blade track: F_n u, u from the bent blade's own rate
    waste_blade = np.zeros(10)
    waste_gate = np.zeros(10)
    dt = float(np.mean(np.diff(t)))
    saved = (sim._bend.copy(), sim._bend_rate.copy())
    for k in range(t.size):
        state = State.from_vector(y[:STATE_SIZE, k])
        if hist is not None:
            sim._bend[:], sim._bend_rate[:] = hist[0, :, k], hist[1, :, k]
        for slot, seat in enumerate(sim._seats):
            oar = sim._oars[slot]
            angle, rate = float(y[STATE_SIZE + slot, k]), float(y[STATE_SIZE + n + slot, k])
            if angle <= oar.finish_angle or (air is not None and air[slot, k]):
                continue
            span = float(oar.catch_angle) - float(oar.finish_angle)
            u_prog = min(max((float(oar.catch_angle) - angle) / span, 0.0), 0.999)
            b = int(u_prog * 10)
            b_angle, b_rate = angle, rate
            if hist is not None:
                b_angle += float(sim._bend[slot])
                b_rate += float(sim._bend_lever[slot] * sim._bend_rate[slot])
            for lock in boat.rig.seats[seat].oarlocks:
                load = abs(float(sim._blade_loads_arm(slot, angle, rate, state, lock)[0]))
                if load <= 0.0:
                    continue
                # each track signed and normalised by its own slip and sweep, as the scorecard's tally is
                for (a, r), target, work in (((angle, rate), waste_gate, handle),
                                             ((b_angle, b_rate), waste_blade, handle_blade)):
                    u = abs(float(oar.outboard) * r)
                    v_n = sim._lock_speed_on_normal(state, lock, a) * float(np.cos(a))
                    slip = u - v_n
                    f = load if slip >= 0.0 else -load
                    target[b] += f * slip * dt
                    work[b] += f * u * dt
    sim._bend[:], sim._bend_rate[:] = saved
    total = handle.sum()
    print("%s %s%s at %.0f W per rower: %.2f m/s; blade efficiency (energy) gate %.4f, blade %.4f"
          % (args.boat, "bending" if args.bending else "rigid", "" if fv is None else ", hold %.1f" % args.hold,
             args.watts, run.settled_speed(), 1 - waste_gate.sum() / total,
             1 - waste_blade.sum() / handle_blade.sum()))
    print("  drive tenth   handle %   waste (blade) % of handle   waste (gate) % of handle   local efficiency (blade)")
    for i in range(10):
        local = 1 - waste_blade[i] / handle_blade[i] if handle_blade[i] > 0 else float("nan")
        print("  %.1f-%.1f      %5.1f      %6.2f                      %6.2f                      %.3f"
              % (edges[i], edges[i + 1], 100 * handle[i] / total, 100 * waste_blade[i] / total,
                 100 * waste_gate[i] / total, local))


if __name__ == "__main__":
    main()
