"""Compare sweep seating patterns on the racing course.

    python scripts/rigging.py
    python scripts/rigging.py --patterns standard german
    python scripts/rigging.py --duration 40

An alternating sweep rig carries a residual yaw couple: every oarlock
sits a third of a metre bow-ward of its rower, so summing side times
station over the crew leaves a net moment arm -- 4.9 m of it on a
standard eight.  The boat pulls one way, the coxswain carries standing
rudder to hold it, and standing rudder is drag for 4.8 km.

Bucket rigs exist to cancel that arm.  This measures what each named
pattern actually buys, with the full 6-DOF boat rather than the
arithmetic: the zero-helm yaw bias, the standing rudder needed to hold a
line, and what that rudder costs in seconds over the Head of the Charles.

Every pattern is also checked on the same optimised line, because the
stagger couple is not the whole story -- pairing seats changes the roll
forcing within the stroke, which the balance controller then has to
spend authority on.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.boats.rig import RIG_PATTERNS             # noqa: E402
from coxswain.progress import progress                  # noqa: E402
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402

RACE_LENGTH = 4822.0


def steady(boat, rudder_deg, duration, settle_fraction=0.5, dt=0.01):
    """Steady yaw rate, mean speed and roll swing under a held rudder."""
    cox = Coxswain(rudder_override=lambda t, s: np.radians(rudder_deg))
    sim = RowingSimulator(boat, coxswain=cox)
    result = sim.run(duration=duration, dt=dt, surge_speed=4.7)
    t = np.asarray(result.time)
    keep = t >= settle_fraction * t[-1]
    omega = np.degrees(np.asarray(result.omega)[2])[keep]
    velocity = np.asarray(result.velocity)[:2].T[keep]
    speed = float(np.hypot(*velocity.T).mean())
    roll = np.degrees(np.asarray(result.attitude)[0])[keep]
    return float(omega.mean()), speed, float(np.ptp(roll))


def trim_rudder(boat, duration):
    """Rudder angle that holds the boat straight, by secant on the bias."""
    a_deg, (a_rate, _s, _r) = 0.0, (None, None, None)
    a_rate, speed0, roll0 = steady(boat, 0.0, duration)
    if abs(a_rate) < 0.02:
        return 0.0, a_rate, speed0, roll0
    b_deg = np.clip(-a_rate / 0.075, -20.0, 20.0)   # authority ~0.075 deg/s per deg
    b_rate, speed1, roll1 = steady(boat, b_deg, duration)
    if abs(b_rate - a_rate) < 1e-6:
        return b_deg, b_rate, speed1, roll1
    c_deg = a_deg - a_rate * (b_deg - a_deg) / (b_rate - a_rate)
    c_deg = float(np.clip(c_deg, -25.0, 25.0))
    c_rate, speed2, roll2 = steady(boat, c_deg, duration)
    return c_deg, c_rate, speed2, roll2


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patterns", nargs="+", default=list(RIG_PATTERNS),
                        choices=list(RIG_PATTERNS))
    parser.add_argument("--duration", type=float, default=30.0,
                        help="seconds of settling per measurement; three "
                             "measurements per pattern")
    parser.add_argument("--rate", type=float, default=28.0)
    args = parser.parse_args(argv)

    rows = []
    bar = progress(total=len(args.patterns), desc="rig patterns", unit="rig")
    for name in args.patterns:
        boat = catalog.eight(rate=args.rate, rig_pattern=name)
        sides = [seat.rigged_side for seat in boat.rig.seats]
        arm = sum(side * seat.station_x
                  for side, seat in zip(sides, boat.rig.seats))
        helm, residual, speed, roll = trim_rudder(boat, args.duration)
        rows.append((name,
                     "".join("P" if s > 0 else "S" for s in sides),
                     arm, helm, residual, speed, roll))
        bar.update(1)
    bar.close()

    fastest = max(r[5] for r in rows)
    print()
    print("%-18s %-10s %9s %10s %9s %9s %9s %10s"
          % ("pattern", "stern->bow", "arm (m)", "helm (deg)", "resid",
             "speed", "roll pp", "vs best"))
    for name, layout, arm, helm, residual, speed, roll in rows:
        delta = RACE_LENGTH / speed - RACE_LENGTH / fastest
        print("%-18s %-10s %+8.2f %+9.1f %+8.2f %8.3f %8.2f %+9.1f s"
              % (name, layout, arm, helm, residual, speed, roll, delta))
    print()
    print("helm is the standing rudder that holds the boat straight; the")
    print("cost column is that rudder's drag over the %d m course." % RACE_LENGTH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
