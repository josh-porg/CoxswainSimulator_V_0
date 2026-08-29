"""Is the Cambridge arch worth it on the Powerhouse Stretch?

    python scripts/powerhouse.py
    python scripts/powerhouse.py --race-time 1140 --gap 17.3

The rules leave both the centre and the Cambridge arch open at River
Street and Western Avenue, and almost everybody takes the centre.  That
is the argument *for* the Cambridge arch: not that it is shorter, because
it is not, but that it is empty.

Two costs, one comparison
-------------------------
Taking the Cambridge arch costs whatever the line costs -- extra metres,
a worse entry to the bend that follows -- and :mod:`coxswain.river.route`
already prices that, at this crew's speed rather than a collegiate one.

Rowing the centre arches costs whatever being in another crew's water
costs, and :mod:`coxswain.hydro.wake` prices that.

Neither number is worth quoting alone.  What this script reports is the
**break-even**: how much of the Powerhouse Stretch you would have to
spend in somebody's puddles before the empty arch pays for itself.  That
converts a physics question into one a coxswain can actually answer from
the start list and the warm-up.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import racing_line as RL                                # noqa: E402
from coxswain.hydro.resistance import hull_resistance   # noqa: E402
from coxswain.hydro.wake import (PuddleWake,  # noqa: E402
                                 TurbulentWater)
from coxswain.river import lines                        # noqa: E402
from coxswain.river.route import Route, optimise_route  # noqa: E402
from coxswain.river.trajectory import ReducedModel      # noqa: E402

RACE_LENGTH = 4822.0
SPEED_EXPONENT = 0.498

#: Only these two.  Carrying the Cambridge arch on through Weeks is a
#: different and much worse decision -- it leaves the boat outside the
#: turn -- and the report already prices it separately.
POWERHOUSE = ("River Street", "Western Avenue")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--rate", type=float, default=28.0)
    parser.add_argument("--gap", type=float, default=17.3,
                        help="how far astern of the crew ahead you would be "
                             "on the crowded line, m (default one length)")
    parser.add_argument("--iterations", type=int, default=45)
    args = parser.parse_args(argv)

    speed = RACE_LENGTH / args.race_time
    raster, course, flow, gates = RL.build(month=10)

    print("optimising a line inside each arch choice at %.3f m/s ..." % speed)
    scored = {}
    for name, pins in (("centre arches",
                        {b: "centre" for b in POWERHOUSE}),
                       ("Cambridge arches",
                        {b: "Cambridge shore" for b in POWERHOUSE})):
        ev = RL.evaluator(course, flow, raster, gates, ReducedModel(),
                          pins=pins)
        ev.reference_speed = speed
        start = lines.pinned_arch_route(course, raster, gates, pins,
                                        margin=4.0, name=name)
        best = optimise_route(ev, n_control=13, iterations=args.iterations,
                              seed=0, initial=start)
        route = Route(best.route.stations, best.route.offsets, name=name)
        scored[name] = (route, ev.evaluate(route))

    print()
    print("  %-18s %10s %9s %9s" % ("arch choice", "time", "distance",
                                    "peak yaw"))
    for name, (_route, r) in scored.items():
        print("  %-18s %9.1fs %8.0fm %9.2f"
              % (name, r.elapsed_clean + 60.0 * r.illegal_arches,
                 r.path_length, r.peak_yaw_rate))
    penalty = ((scored["Cambridge arches"][1].elapsed_clean
                + 60.0 * scored["Cambridge arches"][1].illegal_arches)
               - (scored["centre arches"][1].elapsed_clean
                  + 60.0 * scored["centre arches"][1].illegal_arches))
    print()
    if penalty > 0:
        print("  the Cambridge arch costs %.1f s of line." % penalty)
    else:
        print("  the Cambridge arch is %.1f s FASTER on the line alone, "
              "before any" % -penalty)
        print("  clean-water argument at all.")

    # -- what clean water is worth ---------------------------------------
    from coxswain.boats import catalog                   # noqa: E402
    b = catalog.eight(rate=args.rate, rower_mass=72.0, rower_stature=1.72)
    submerged = b.mesh.submerged(np.array([0.0, 0.0, b.equilibrium_heave()]),
                                 np.zeros(3), rho=b.water.density,
                                 gravity=9.80665, water_level=0.0)
    force, _ = hull_resistance(np.array([speed, 0.0, 0.0]), submerged,
                               mean_wetted_length=b.length, water=b.water,
                               coefficients=b.resistance)
    drag = abs(float(force[0]))
    wake = PuddleWake(drag=drag, speed=speed, period=60.0 / args.rate)
    dirty = TurbulentWater(wake, length=b.length)
    blades = float(wake.power_penalty(args.gap)) * SPEED_EXPONENT
    friction = float(dirty.drag_penalty(args.gap)) * SPEED_EXPONENT
    hull_back = float(wake.hull_benefit(args.gap)) * SPEED_EXPONENT
    net = blades + friction - hull_back

    print()
    print("what the crowded line costs, %0.1f m astern" % args.gap)
    print("  blades in their puddles:  %+.4f s per second of racing" % blades)
    print("  hull in their turbulence: %+.4f s   (Tu %.1f%%, b %.2f, "
          "Cf up %.1f%%)"
          % (friction, 100 * float(dirty.intensity(args.gap)),
             float(dirty.parameter(args.gap)),
             100 * float(dirty.drag_penalty(args.gap))
             / dirty.viscous_fraction))
    print("  hull in their wake:       %+.4f s   (the one that helps)"
          % -hull_back)
    print("  net:                      %+.4f s per second (%.2f%% of speed)"
          % (net, 100 * net))

    stretch = _powerhouse_length(course, gates)
    seconds_of_stretch = stretch / speed
    print()
    print("the Powerhouse Stretch, River Street to Western Avenue")
    print("  %.0f m, which is %.0f s of racing at this speed"
          % (stretch, seconds_of_stretch))
    print("  clean water for all of it is worth %.1f s" % (net *
                                                           seconds_of_stretch))
    print()
    if net <= 0:
        print("  the model says following is a net GAIN at this gap, so the")
        print("  clean-water argument does not carry -- take the centre arch.")
        return 0
    breakeven = penalty / net if net > 0 else float("inf")
    print("  BREAK-EVEN: the Cambridge arch pays if you would otherwise")
    if breakeven <= 0:
        print("  spend any time at all in traffic -- it is free.")
    elif breakeven > seconds_of_stretch:
        print("  spend %.0f s in somebody's water, and the stretch is only"
              % breakeven)
        print("  %.0f s long.  It does not pay on wake grounds alone."
              % seconds_of_stretch)
    else:
        print("  spend more than %.0f s of the %.0f s stretch in somebody's"
              % (breakeven, seconds_of_stretch))
        print("  water -- that is %.0f%% of it, or about %.0f m."
              % (100 * breakeven / seconds_of_stretch, breakeven * speed))
    return 0


def _powerhouse_length(course, gates):
    """Station distance between River Street and Western Avenue, m."""
    stations = {}
    for gate in gates:
        name = getattr(gate, "bridge", None) or getattr(gate, "name", "")
        station = getattr(gate, "station", None)
        if station is not None:
            stations.setdefault(str(name), float(station))
    try:
        return abs(stations["Western Avenue"] - stations["River Street"])
    except KeyError:
        # Surveyed positions if the gate objects do not carry stations.
        return 1050.0


if __name__ == "__main__":
    raise SystemExit(main())
