"""Blade efficiency on the published definition, with and without oar bending (SOURCES §217).

Kleshnev's band (78.5 +/- 3.1% for a single) was measured with the blade track from the gate angle and boat velocity
and the blade force from handle force x gearing: a rigid oar's track.  The simulator's own tally takes the bent blade's
motion.  This settles each scorecard boat at the scorecard's race-pace point under the research profile, once with
the profile's sourced bending and once rigid, and prints both tallies.

    python research/blade/efficiency_definition.py [--boats 8+ 4+] [--watts 360]
    python research/blade/efficiency_definition.py --bend      # the bent oar's peak bend and oarlock moment
"""
import argparse
import time

import numpy as np

from coxswain import physics
from coxswain.boats import catalog
from coxswain.validation import scorecard
from coxswain.validation.targets import by_key


#: the sweep oar's bending law: "robert" ([RLBQV19]'s bench) or "concept2" ([L14] scull EI x Concept2's ratio, §220)
SWEEP_LAW = "robert"


def bending_law(boat):
    """The bending law under test: [RLBQV19]'s bench law on a sweep rig, [L14]'s measured scull shaft on a scull. No
    profile bends the oars (withdrawn, SOURCES §217), so the study sets it on the boat itself."""
    from coxswain.crew.oar_bending import OarBending
    oar = boat.rig.seats[0].oarlocks[0].oar
    if boat.rig.is_sweep:
        if SWEEP_LAW == "concept2":
            return OarBending.concept2_sweep(float(oar.blade_centre_outboard), float(oar.blade_length))
        return OarBending.robert2019_sweep()
    return OarBending.laschowski2014_scull(float(oar.blade_centre_outboard), float(oar.blade_length))


def bend_report(name, watts, rate):
    """Settle the bending boat as the scorecard does, keep the simulator, and print the last stroke's peak blade
    rotation and oarlock moment (K_ang * phi, the bench law's own quantity)."""
    from coxswain.sim.control import Coxswain
    from coxswain.sim.dynamic_oar import DynamicOarSimulator
    profile = physics.resolve("research")
    boat = profile.apply(catalog.build(name, rate=rate))
    boat.oar_bending = bending_law(boat)
    boat.power_scales = np.ones(boat.n_seats)
    own = profile.blade_options(boat)
    kw = dict(catch=profile.catch, blade_law=own["blade_law"], blade_span=own["blade_span"],
              blade_coefficients=own["blade_coefficients"])
    torque = DynamicOarSimulator.torque_for_power(boat, watts, start=5.0, **kw)
    sim = DynamicOarSimulator(boat, peak_torque=torque, coxswain=Coxswain(rudder_override=lambda t, s: 0.0),
                              fast=True, **kw)
    sim.run_strokes(scorecard.SETTLE_STROKES, surge_speed=5.0)
    phi = np.abs(sim._bend_history[0])
    k = sim.oar_bending.k_ang
    print("%-3s bend: peak blade rotation %.1f deg (oar mean %.1f), oarlock moment %.0f N m, K_ang %.0f N m/rad, "
          "bend rate peak %.2f rad/s" % (name, np.degrees(phi.max()), np.degrees(phi.max(axis=1).mean()),
                                         k * phi.max(), k, np.abs(sim._bend_history[1]).max()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boats", nargs="+", default=["8+", "4+"])
    ap.add_argument("--watts", type=float, default=360.0)
    ap.add_argument("--rate", type=float, default=28.0)
    ap.add_argument("--bend", action="store_true", help="report the bent oar's peak bend instead")
    ap.add_argument("--modes", nargs="+", default=["bending", "rigid"], choices=["bending", "rigid"])
    ap.add_argument("--fv", type=float, default=None,
                    help="the crew's handle force-velocity with this V0, m/s (coxswain.crew.handle_fv; SOURCES §218)")
    ap.add_argument("--hold", type=float, default=None,
                    help="the crew's hold on its planned speed, beta (coxswain.crew.handle_fv.PlannedSpeedHold; §220)")
    ap.add_argument("--sweep-law", default="robert", choices=["robert", "concept2"])
    args = ap.parse_args()
    global SWEEP_LAW
    SWEEP_LAW = args.sweep_law
    if args.bend:
        for name in args.boats:
            bend_report(name, args.watts, args.rate)
        return
    profile = physics.resolve("research")
    target = by_key("blade_efficiency_level")
    fv, label = None, ""
    if args.fv is not None:
        from coxswain.crew.handle_fv import HandleForceVelocity
        fv, label = HandleForceVelocity(args.fv, "--fv"), "; crew force-velocity V0 %.2f m/s" % args.fv
    if args.hold is not None:
        from coxswain.crew.handle_fv import PlannedSpeedHold
        fv, label = PlannedSpeedHold(args.hold, "--hold"), "; crew hold beta %.2f" % args.hold
    for name in args.boats:
        print("%s band %.3f-%.3f (Kleshnev Table 2, gate-angle track)%s" % (
            (name,) + target.band_for(name) + (label,)))
        for bend in args.modes:
            boat = profile.apply(catalog.build(name, rate=args.rate))
            boat.oar_bending = bending_law(boat) if bend == "bending" else None
            t0 = time.time()
            got = scorecard.settle_dynamic(boat, args.watts, start=5.0, handle_fv=fv)
            print("%-3s %-7s %.2f m/s  slip ratio %.4f  K07 energy (gate track) %.4f  swing %.1f%%  (%.0f s)" % (
                name, bend, got.speed, got.blade_efficiency, got.blade_efficiency_k07,
                100.0 * got.surge_swing, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
