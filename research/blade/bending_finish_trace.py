"""What loads the blade at the finish: the crew's torque, or the oar's own momentum? (SOURCES §218)

With the sweep bending law the eight's blades lag 20-38 deg into the finish (§217), and the blade's moment about the
pin, K phi, reaches twice the peak handle torque. A massless oar cannot do that, so either the oar's (and the crew's
reflected) inertia is being stopped by the water through the blade, or something else is. This traces one oar through
the last stroke's drive, rigid and bending, at the same peak torque:

    progress  oar angle  oar rate  oar accel  handle torque  blade moment (arm F_n)  bend

    python research/blade/bending_finish_trace.py [--boat 8+] [--torque 849] [--strokes 4]
"""
import argparse

import numpy as np

from coxswain import physics
from coxswain.boats import catalog
from coxswain.core.state import State
from coxswain.crew.oar_bending import OarBending
from coxswain.sim.control import Coxswain
from coxswain.sim.dynamic_oar import STATE_SIZE, DynamicOarSimulator


def trace(boat_name, torque, strokes, bend, fv=None):
    profile = physics.resolve("research")
    boat = profile.apply(catalog.build(boat_name, rate=28.0))
    boat.power_scales = np.ones(boat.n_seats)
    own = profile.blade_options(boat)
    oar0 = boat.rig.seats[0].oarlocks[0].oar
    law = None
    if bend:
        law = (OarBending.robert2019_sweep() if boat.rig.is_sweep
               else OarBending.laschowski2014_scull(float(oar0.blade_centre_outboard), float(oar0.blade_length)))
    sim = DynamicOarSimulator(boat, peak_torque=torque, catch=profile.catch, blade_law=own["blade_law"],
                              blade_span=own["blade_span"], blade_coefficients=own["blade_coefficients"],
                              oar_bending=law if bend else False, handle_fv=fv,
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    run = sim.run_strokes(strokes, surge_speed=5.0)
    t, y = run.last_time, run.last_states
    n = sim.n_oar_states
    slot = 0
    oar = sim._oars[slot]
    seat = sim._seats[slot]
    lock = boat.rig.seats[seat].oarlocks[0]
    angle = y[STATE_SIZE + slot]
    rate = y[STATE_SIZE + n + slot]
    accel = np.gradient(rate, t)
    air = sim._air_mask
    hist = sim._bend_history
    wet = np.flatnonzero(~air[slot] & (angle > oar.finish_angle))
    if wet.size == 0:
        print("no wet samples")
        return
    first, last = wet[0], wet[-1]
    print("%s %s%s, peak torque %.0f N m: %.2f m/s, %.0f W per rower, oar %d (catch %.1f, finish %.1f deg)"
          % (boat_name, "bending" if bend else "rigid",
             "" if fv is None else ", crew %s" % (fv.key(),),
             torque, run.settled_speed(), run.settled_power(), slot,
             np.degrees(oar.catch_angle), np.degrees(oar.finish_angle)))
    print("  I = the seat's inertia about the pin: the rower's reflected inertia plus the oar (kg m^2)")
    print("  progress  angle   rate    accel    I     I*accel  handle   blade(arm F_n)   bend")
    saved = (sim._bend.copy(), sim._bend_rate.copy())
    for k in np.linspace(first, min(last + 3, t.size - 1), 22).astype(int):
        if hist is not None:
            sim._bend[:], sim._bend_rate[:] = hist[0, :, k], hist[1, :, k]
        state = State.from_vector(y[:STATE_SIZE, k])
        handle = sim._torque(slot, float(angle[k]), state, float(t[k]), float(rate[k]))   # the clock crew's pull
        f_n, _f_t, arm = sim._blade_loads_arm(slot, float(angle[k]), float(rate[k]), state, lock)
        phi = float(sim._bend[slot]) if hist is not None else 0.0
        moment = float(oar.inertia_at(float(angle[k]))[0])
        print("  %6.3f  %6.1f  %6.2f  %7.1f  %5.1f  %7.0f  %7.0f   %7.0f          %5.1f%s"
              % ((k - first) / max(last - first, 1), np.degrees(angle[k]), rate[k], accel[k], moment,
                 moment * accel[k], handle, arm * f_n, np.degrees(phi),
                 "  (out)" if air[slot, k] or angle[k] <= oar.finish_angle else ""))
    sim._bend[:], sim._bend_rate[:] = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boat", default="8+")
    ap.add_argument("--torque", type=float, default=849.0)
    ap.add_argument("--strokes", type=int, default=4)
    ap.add_argument("--fv", type=float, default=None,
                    help="the crew's handle force-velocity with this V0, m/s (coxswain.crew.handle_fv; SOURCES §218)")
    ap.add_argument("--hold", type=float, default=None,
                    help="the crew's hold on its planned speed, beta (coxswain.crew.handle_fv.PlannedSpeedHold; §220)")
    args = ap.parse_args()
    fv = None
    if args.fv is not None:
        from coxswain.crew.handle_fv import HandleForceVelocity
        fv = HandleForceVelocity(args.fv, "--fv")
    if args.hold is not None:
        from coxswain.crew.handle_fv import PlannedSpeedHold
        fv = PlannedSpeedHold(args.hold, "--hold")
    for bend in (False, True):
        trace(args.boat, args.torque, args.strokes, bend, fv)


if __name__ == "__main__":
    main()
