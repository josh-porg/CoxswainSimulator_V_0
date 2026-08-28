"""Seat a crew of unequal rowers, for a straight course or for a turn.

    python scripts/seating.py
    python scripts/seating.py --crew 1.00 0.97 1.02 0.95 1.01 0.99 0.93 1.04
    python scripts/seating.py --verify 3

Given rowers of known relative strength, some of whom can only row one
side and only some of whom can take the stroke seat, this searches every
legal arrangement for the one that suits a given course -- straight, or
bending to port, or bending to starboard.

Why this can be searched exhaustively
-------------------------------------
The crew's total power does not depend on how they are seated, so the
propulsive cost of a weak rower is the same in every arrangement and
cancels out of the comparison.  What arrangement *does* change is the
boat's standing yaw, through two couples:

``side``     ``sum(side_i * p_i) * span`` -- the port/starboard force
             imbalance acting through the 0.85 m oarlock offset.
``stagger``  ``sum(side_i * p_i * x_i) * k`` -- each rower's blade side
             force acting at their own station, which is why an
             alternating rig has a standing bias at all.

Both are linear in the power scales, so a whole arrangement scores in
microseconds and all of them can be enumerated.  The 6-DOF model is then
used to *verify* the handful of finalists rather than to search, which is
the only reason this is tractable: a single 20 s trial costs 20 s of wall
clock, and there are tens of thousands of arrangements.

What a turn changes
-------------------
On a straight course the target yaw is zero and the best crew is the one
that cancels its own bias.  In a bend the target is **not** zero -- the
boat must come round at the rate the course demands -- so an arrangement
that leans into the turn is doing free work the rudder would otherwise do,
and rudder is drag.  The same crew therefore has a different best seating
for a port turn than for a starboard one, which is the interesting claim
and the one worth testing against a coach's intuition.

The model has nothing to say about rhythm, confidence, or whether a rower
can actually follow the person in front of them.  It answers the narrow
mechanical question only.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.progress import progress                  # noqa: E402
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402

RACE_LENGTH = 4822.0
PORT, STARBOARD = +1, -1

#: Yaw rate the course asks for, deg/s, in each scenario.  The straight is
#: self-explanatory; the turn figures are the Charles at its two hardest
#: bends -- Weeks to port, the Cambridge Boat Club bend to starboard --
#: taken from the curvature of the optimised racing line at 4.7 m/s.
SCENARIOS = {"straight": 0.0, "port turn": +1.30, "starboard turn": -1.30}


def couples(sides, powers, stations, span=0.85):
    """The two yaw couples an arrangement produces, per unit force.

    Returned separately rather than summed because they have different
    lever arms and the calibration below has to weigh them.
    """
    sides = np.asarray(sides, dtype=float)
    powers = np.asarray(powers, dtype=float)
    stations = np.asarray(stations, dtype=float)
    return (float(np.sum(sides * powers) * span),
            float(np.sum(sides * powers * stations)))


def trial(boat, duration=20.0, dt=0.01):
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0)
    sim = RowingSimulator(boat, coxswain=cox)
    result = sim.run(duration=duration, dt=dt, surge_speed=4.7)
    t = np.asarray(result.time)
    keep = t >= 0.5 * t[-1]
    velocity = np.asarray(result.velocity)[:2].T[keep]
    return (float(np.hypot(*velocity.T).mean()),
            float(np.degrees(np.asarray(result.omega)[2])[keep].mean()))


def calibrate(stations, duration):
    """Fit yaw = a * side_couple + b * stagger_couple on three arrangements.

    Three points for two coefficients and an intercept, measured with the
    real boat, so the fast score is anchored to the 6-DOF rather than to
    an assumed lever.
    """
    probes = [
        [PORT, STARBOARD] * 4,                       # standard
        [PORT, STARBOARD, STARBOARD, PORT] * 2,      # german
        [PORT] * 4 + [STARBOARD] * 4,                # ends-loaded
    ]
    rows, yaws = [], []
    bar = progress(total=len(probes), desc="calibrating", unit="probe")
    for sides in probes:
        boat = catalog.eight(rate=28.0)
        for seat, side in zip(boat.rig.seats, sides):
            for lock in seat.oarlocks:
                lock.position[1] = abs(lock.position[1]) * side
                object.__setattr__(lock, "side", side)
        _speed, yaw = trial(boat, duration)
        side_c, stagger_c = couples(sides, np.ones(len(sides)), stations)
        rows.append([side_c, stagger_c, 1.0])
        yaws.append(yaw)
        bar.update(1)
    bar.close()
    coefficients, *_ = np.linalg.lstsq(np.array(rows), np.array(yaws),
                                       rcond=None)
    return coefficients


def enumerate_seatings(crew, stations, coefficients, target, limit=8):
    """Every legal arrangement, scored by how near it lands the target yaw."""
    n = len(crew)
    names = [r["name"] for r in crew]
    powers = {r["name"]: r["power"] for r in crew}
    best = []
    seats = list(range(n))

    for order in itertools.permutations(range(n)):
        # order[k] is the rower index sitting in seat k (0 = stroke)
        rower = [crew[i] for i in order]
        if not rower[0]["can_stroke"]:
            continue
        # sides follow from each rower's own constraint where fixed;
        # rowers who can row either take whatever the balance needs.
        fixed = [r["side"] for r in rower]
        flexible = [k for k, s in enumerate(fixed) if s == 0]
        forced_port = sum(1 for s in fixed if s == PORT)
        forced_stbd = sum(1 for s in fixed if s == STARBOARD)
        need_port = n // 2 - forced_port
        if need_port < 0 or need_port > len(flexible):
            continue
        for port_choice in itertools.combinations(flexible, need_port):
            sides = list(fixed)
            for k in flexible:
                sides[k] = PORT if k in port_choice else STARBOARD
            side_c, stagger_c = couples(
                sides, [r["power"] for r in rower], stations)
            yaw = (coefficients[0] * side_c + coefficients[1] * stagger_c
                   + coefficients[2])
            best.append((abs(yaw - target), yaw,
                         tuple(r["name"] for r in rower), tuple(sides)))
    best.sort(key=lambda row: row[0])
    return best[:limit]


def parse_crew(values, stroke_capable, sides):
    crew = []
    for index, power in enumerate(values):
        crew.append({
            "name": chr(ord("A") + index),
            "power": float(power),
            "can_stroke": (index in stroke_capable) if stroke_capable else True,
            "side": sides[index] if sides else 0,
        })
    return crew


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crew", type=float, nargs="+",
                        default=[1.00, 0.97, 1.03, 0.95, 1.01, 0.99, 0.92,
                                 1.04],
                        help="relative power of each rower, stern to bow "
                             "order is irrelevant -- they are candidates, "
                             "not a seating")
    parser.add_argument("--stroke-capable", type=int, nargs="+",
                        default=[0, 2, 4, 7],
                        help="indices of rowers who can take stroke")
    parser.add_argument("--sides", type=int, nargs="+", default=None,
                        help="per rower: 1 port only, -1 starboard only, "
                             "0 either. Default: all bisidal")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--verify", type=int, default=2,
                        help="how many finalists to check with the 6-DOF")
    args = parser.parse_args(argv)

    reference = catalog.eight(rate=28.0)
    stations = [seat.station_x for seat in reference.rig.seats]
    crew = parse_crew(args.crew, set(args.stroke_capable), args.sides)

    print("crew: " + ", ".join(
        "%s %.2f%s%s" % (r["name"], r["power"],
                         "" if r["side"] == 0 else
                         (" P-only" if r["side"] > 0 else " S-only"),
                         "*" if r["can_stroke"] else "")
        for r in crew))
    print("(* = can row stroke)")
    print()

    coefficients = calibrate(stations, args.duration)
    print("yaw = %.4f x side couple + %.4f x stagger couple %+.4f"
          % tuple(coefficients))
    print()

    for scenario, target in SCENARIOS.items():
        finalists = enumerate_seatings(crew, stations, coefficients, target)
        print("%s (target yaw %+.2f deg/s)" % (scenario, target))
        print("  %-26s %-26s %9s" % ("stern -> bow", "sides", "yaw"))
        for _error, yaw, names, sides in finalists[:3]:
            print("  %-26s %-26s %+9.3f"
                  % (" ".join(names),
                     "".join("P" if s > 0 else "S" for s in sides), yaw))
        print()

    if args.verify:
        print("verifying the straight-course finalists with the 6-DOF:")
        finalists = enumerate_seatings(crew, stations, coefficients, 0.0)
        bar = progress(total=min(args.verify, len(finalists)),
                       desc="verifying", unit="seating")
        for _error, predicted, names, sides in finalists[:args.verify]:
            boat = catalog.eight(rate=28.0)
            order = {r["name"]: r["power"] for r in crew}
            boat.power_scales = np.array([order[n] for n in names])
            for seat, side in zip(boat.rig.seats, sides):
                for lock in seat.oarlocks:
                    lock.position[1] = abs(lock.position[1]) * side
                    object.__setattr__(lock, "side", side)
            speed, yaw = trial(boat, args.duration)
            print("  %-26s predicted %+.3f, measured %+.3f, %.4f m/s"
                  % (" ".join(names), predicted, yaw, speed))
            bar.update(1)
        bar.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
