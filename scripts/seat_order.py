"""Does it matter *which seat* the weak rower sits in?

    python scripts/seat_order.py
    python scripts/seat_order.py --deficit 0.20 --duration 30

Conventional wisdom says yes, and says it firmly: put the weak one in the
middle of the boat, never at stroke, never at bow.  The physics has to be
asked whether it agrees, because the reasoning usually offered for it --
"the ends of the boat have more leverage" -- is a statement about moment
arms, and it is worth checking which moments actually see the seat.

What the mechanism ought to be
------------------------------
A rower pulling ``d`` less hard changes three things, and they do not all
depend on where the rower sits.

**Thrust.**  Each oar pushes the boat along its axis through an oarlock
``0.85 m`` off the centreline.  Weaken one side and the imbalance is a
yaw couple of ``dF x 0.85 m`` -- the *same* couple wherever along the
boat the deficit sits, because every oarlock is the same distance out.
This channel is **seat-blind**.

**Blade side force.**  A sweep blade also pushes sideways, and that force
acts at the rower's longitudinal station: a yaw moment ``F_y x``, which
is zero amidships and largest at the ends.  This channel is **seat-aware**
and is the one the folklore is reaching for.

**Pitch.**  Force applied at station ``x`` pitches the hull about the
centre of mass, and pitch changes the wetted length.  Also seat-aware.

So the honest prior is: seat position matters, but only through the
smaller of the two yaw channels, and the effect could easily be buried.
The experiment settles it -- and note the model has no opinion about crew
morale or the stroke seat's job of setting rhythm, which is a large part
of what the conventional wisdom is really about.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.progress import progress                  # noqa: E402
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402

RACE_LENGTH = 4822.0


def run(boat, duration, dt=0.01):
    """Straight-line trial, rudder centred: speed, drift, roll, pitch."""
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0)
    sim = RowingSimulator(boat, coxswain=cox)
    result = sim.run(duration=duration, dt=dt, surge_speed=4.7)
    t = np.asarray(result.time)
    keep = t >= 0.5 * t[-1]
    velocity = np.asarray(result.velocity)[:2].T[keep]
    attitude = np.asarray(result.attitude)
    return {
        "speed": float(np.hypot(*velocity.T).mean()),
        "yaw_rate": float(np.degrees(np.asarray(result.omega)[2])[keep].mean()),
        "roll": float(np.ptp(np.degrees(attitude[0])[keep])),
        "pitch": float(np.ptp(np.degrees(attitude[1])[keep])),
    }


def deficit_sweep(args):
    """Cost against how weak the weak rower is, at the best and worst seat.

    Two seats only, because the seat experiment already showed which they
    are, and the question here is the shape of the curve rather than the
    ranking.
    """
    reference = catalog.eight(rate=args.rate, rig_pattern=args.rig)
    labels = [seat.label for seat in reference.rig.seats]
    sides = [seat.rigged_side for seat in reference.rig.seats]
    base = run(reference, args.duration)
    port = next(i for i, s in enumerate(sides) if s > 0)
    stbd = next(i for i, s in enumerate(sides) if s < 0)

    print("even crew, %s rig: %.4f m/s" % (args.rig, base["speed"]))
    print()
    print("  %-9s %-10s %10s %10s %10s"
          % ("deficit", "seat", "speed", "cost (s)", "yaw"))
    bar = progress(total=2 * len(args.sweep), desc="deficits", unit="run")
    for deficit in args.sweep:
        for index in (port, stbd):
            boat = catalog.eight(rate=args.rate, rig_pattern=args.rig)
            scales = np.ones(boat.n_seats)
            scales[index] = 1.0 - deficit
            boat.power_scales = scales
            outcome = run(boat, args.duration)
            cost = RACE_LENGTH / outcome["speed"] - RACE_LENGTH / base["speed"]
            print("  %-9.0f%% %-10s %10.4f %+10.2f %+10.3f"
                  % (100 * deficit,
                     "%s (%s)" % (labels[index],
                                  "port" if sides[index] > 0 else "stbd"),
                     outcome["speed"], cost, outcome["yaw_rate"]))
            bar.update(1)
    bar.close()
    print()
    print("if cost were linear in the deficit, doubling the deficit would")
    print("double the cost; departures are the yaw channel showing up.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deficit", type=float, default=0.15,
                        help="how much less the weak rower pulls (0.15 = 15%%)")
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--rate", type=float, default=28.0)
    parser.add_argument("--rig", default="standard")
    parser.add_argument("--sweep", type=float, nargs="+", default=None,
                        help="sweep the deficit instead of the seat, e.g. "
                             "0.05 0.10 0.20 0.30. Answers whether the cost "
                             "of a weak rower is linear in how weak they "
                             "are, which it need not be: thrust enters "
                             "speed through a cube root, but the yaw "
                             "imbalance it creates enters through turning "
                             "drag, which is quadratic")
    args = parser.parse_args(argv)

    if args.sweep:
        return deficit_sweep(args)

    reference = catalog.eight(rate=args.rate, rig_pattern=args.rig)
    labels = [seat.label for seat in reference.rig.seats]
    sides = [seat.rigged_side for seat in reference.rig.seats]
    stations = [seat.station_x for seat in reference.rig.seats]

    base = run(reference, args.duration)
    print("even crew, %s rig: %.4f m/s, yaw %+0.3f deg/s, roll %.2f, "
          "pitch %.3f" % (args.rig, base["speed"], base["yaw_rate"],
                          base["roll"], base["pitch"]))
    print()
    print("one rower %.0f%% down, by seat:" % (100 * args.deficit))
    print("  %-8s %-5s %8s %10s %10s %9s %9s"
          % ("seat", "side", "x (m)", "speed", "cost (s)", "yaw", "roll pp"))

    rows = []
    bar = progress(total=len(labels), desc="seats", unit="seat")
    for index, label in enumerate(labels):
        boat = catalog.eight(rate=args.rate, rig_pattern=args.rig)
        scales = np.ones(boat.n_seats)
        scales[index] = 1.0 - args.deficit
        boat.power_scales = scales
        outcome = run(boat, args.duration)
        cost = RACE_LENGTH / outcome["speed"] - RACE_LENGTH / base["speed"]
        rows.append((label, outcome, cost))
        print("  %-8s %-5s %8.2f %8.4f %+9.2f %+9.3f %9.2f"
              % (label, "port" if sides[index] > 0 else "stbd",
                 stations[index], outcome["speed"], cost,
                 outcome["yaw_rate"], outcome["roll"]))
        bar.update(1)
    bar.close()

    costs = np.array([c for _l, _o, c in rows])
    yaws = np.array([o["yaw_rate"] for _l, o, _c in rows])
    print()
    print("spread across seats: time %.2f s, yaw %.3f deg/s"
          % (costs.max() - costs.min(), yaws.max() - yaws.min()))
    print("best seat %s (%+.2f s), worst %s (%+.2f s)"
          % (rows[int(costs.argmin())][0], costs.min(),
             rows[int(costs.argmax())][0], costs.max()))
    port = [c for (l, o, c), s in zip(rows, sides) if s > 0]
    stbd = [c for (l, o, c), s in zip(rows, sides) if s < 0]
    print("mean cost: port seats %+.2f s, starboard seats %+.2f s"
          % (np.mean(port), np.mean(stbd)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
