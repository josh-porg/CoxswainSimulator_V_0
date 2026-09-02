r"""Two boats on the course: does the crew ahead change your line?

    python scripts/two_boats.py

Every line this project has optimised assumed an empty river. A Head of
the Charles entry never has one. This is the first step of the traffic
problem, and deliberately the simplest: **one boat ahead, same speed,
fixed start interval, and the question of what their wake does to your
choice of line.**

The passing rules are not modelled yet
--------------------------------------
HOCR's rules say the Passer declares a side within one boat length of open
water and the Passee must have yielded by half a length, with 60 s, then
120 s, then disqualification for failing to. None of that is here yet,
because none of it binds while both boats hold the same speed: with equal
speeds nobody closes and nobody passes. That is the next step, and it is a
different kind of problem -- a game between two coxswains rather than an
optimisation against a river.

What this does model
--------------------
With equal speeds the gap is constant, so every puddle you meet has
exactly the same age and the leader's wake becomes a **static drag field
painted along their track**. Your line then optimises against a river that
has an extra layer on it, and the existing optimiser handles that without
modification.

The interesting structure is lateral. Their puddles sit in two lines at
their blade track, roughly 3.15 m either side of their centreline; yours
sweep the same distance either side of you. Coincidences happen at
**zero** offset and at **twice the blade track**, so there are two bad
places to sit and a clean window between them.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                          # noqa: E402
from coxswain.hydro.resistance import hull_resistance       # noqa: E402
from coxswain.hydro.wake import blade_track                 # noqa: E402
from coxswain.river.charles import charles_course           # noqa: E402
from coxswain.river.route import (Route, RouteEvaluator,    # noqa: E402
                                  optimise_route)
from coxswain.river.traffic import LeadBoat                 # noqa: E402


def leader_drag(boat, speed):
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)
    force, _ = hull_resistance(
        np.array([float(speed), 0.0, 0.0]), submerged,
        mean_wetted_length=boat.length, water=boat.water,
        coefficients=boat.resistance)
    return abs(float(force[0]))


def lateral_profile(lead, gaps):
    """The wake's shape across the river, at several start intervals."""
    print("THE WAKE ACROSS THE RIVER")
    print("  blade track %.2f m, so puddle lines sit that far either side"
          % lead.wake.track)
    print()
    offsets = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0,
                        14.0])
    header = "  %-9s" % "offset m"
    for interval in gaps:
        header += " %11s" % ("%.0f s" % interval)
    print(header)
    for offset in offsets:
        row = "  %-9.1f" % offset
        for interval in gaps:
            factor = lead.wake.drag_factor(offset, interval * lead.speed)
            row += " %11.4f" % factor
        print(row)
    print()
    print("  Two peaks, not one: at zero offset both of your blade tracks")
    print("  sit on both of theirs, and at about %.1f m your inside blade"
          % (2 * lead.wake.track))
    print("  drops into their far puddle line.  The window between them is")
    print("  the place to be, and it is only about a metre wide.")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float, default=15.0,
                        help="leader's head start, s")
    parser.add_argument("--intervals", type=float, nargs="+",
                        default=[10.0, 15.0, 30.0, 60.0])
    parser.add_argument("--speed", type=float, default=4.23)
    parser.add_argument("--n-control", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args(argv)

    boat = catalog.eight(rate=32.0, rower_mass=68.0, rower_stature=1.70,
                         coxswain_mass=68.0)
    course = charles_course()
    drag = leader_drag(boat, args.speed)

    print("Two boats on the surveyed Charles reach")
    print("  leader drag %.0f N at %.2f m/s, stroke period %.2f s"
          % (drag, args.speed, boat.timing.period))
    print("  start interval %.0f s = %.0f m of water between the boats"
          % (args.interval, args.interval * args.speed))
    print()

    # -- the empty river, for reference ---------------------------------
    clean = RouteEvaluator(course, boat=boat)
    solo = optimise_route(clean, n_control=args.n_control,
                          iterations=args.iterations)
    solo_offsets = solo.route.offset_at(solo.station)

    # -- a leader on the centreline --------------------------------------
    centre_lead = LeadBoat.build(Route.centreline(course), course, boat,
                                 drag=drag, interval=args.interval,
                                 speed=args.speed)
    lateral_profile(centre_lead, args.intervals)

    print("HOW THE LINE RESPONDS")
    print("  %-26s %10s %10s %11s %10s"
          % ("leader's line", "your time", "vs solo", "max offset", "mean sep"))

    rows = []
    for label, lead_route in (("centreline", Route.centreline(course)),
                              ("the fast line", solo.route)):
        lead = LeadBoat.build(lead_route, course, boat, drag=drag,
                              interval=args.interval, speed=args.speed)
        evaluator = RouteEvaluator(course, boat=boat).with_traffic(lead)
        best = optimise_route(evaluator, n_control=args.n_control,
                              iterations=args.iterations)
        offsets = best.route.offset_at(best.station)
        separation = offsets - lead.offset_at(best.station)
        rows.append((label, best, offsets, separation))
        print("  %-26s %9.1fs %+9.1fs %10.1f %10.1f"
              % (label, best.elapsed_clean,
                 best.elapsed_clean - solo.elapsed_clean,
                 float(np.abs(offsets).max()),
                 float(np.abs(separation).mean())))

    print("  %-26s %9.1fs %+9.1fs %10.1f %10s"
          % ("(nobody -- empty river)", solo.elapsed_clean, 0.0,
             float(np.abs(solo_offsets).max()), "--"))
    print()

    print("WHAT IT COSTS TO SIT IN IT")
    print("  Rowing the SOLO fast line while a boat ahead rows the same")
    print("  line, against re-optimising around them:")
    lead = LeadBoat.build(solo.route, course, boat, drag=drag,
                          interval=args.interval, speed=args.speed)
    dirty = RouteEvaluator(course, boat=boat).with_traffic(lead)
    stuck = dirty.evaluate(solo.route)
    dodged = rows[1][1]
    print("    same line, in their water   %9.1f s" % stuck.elapsed_clean)
    print("    re-optimised around them    %9.1f s" % dodged.elapsed_clean)
    print("    empty river                 %9.1f s" % solo.elapsed_clean)
    print("    cost of the traffic         %9.1f s"
          % (stuck.elapsed_clean - solo.elapsed_clean))
    print("    recovered by moving over    %9.1f s"
          % (stuck.elapsed_clean - dodged.elapsed_clean))
    print()
    print("  The second number is the coxswain's decision and the first is")
    print("  not.  A crew that sits directly on the stern in front pays the")
    print("  whole cost; the lateral shift that avoids it is small.")
    print()
    print("  Both boats are held at the same speed here, so nobody closes")
    print("  and the passing rules never bind.  That is the next problem,")
    print("  and it is a game between two coxswains rather than an")
    print("  optimisation against a river.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
