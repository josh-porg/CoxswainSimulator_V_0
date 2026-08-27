"""What an imperfect crew does to the boat, by experience level.

    python scripts/crew_scatter.py
    python scripts/crew_scatter.py --draws 8 --duration 20

Every headline number so far came from the mean crew: eight identical
rowers pulling identical strokes.  Real crews scatter, and the scatter is
measured -- 2.3% force variation for an elite sculler, 5.1% for a junior
(Kleshnev), with a club crew between.

This draws crews from those distributions and lets the full 6-DOF boat
row with them: each draw fixes the crew's persistent biases (who is
strong, who is early) plus one stroke's scatter, then holds a straight
course with the rudder centred.  The spread across draws is the honest
error bar under every deterministic result, and the yaw drift is what a
coxswain is actually correcting all day.

What this does not yet do: redraw the scatter every stroke within a run.
That needs a per-stroke hook in the simulator and is the entry to the
full stochastic control problem; the per-crew draw here captures the
persistent asymmetries, which are what drive steering.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.crew.variability import CLUB, ELITE, JUNIOR  # noqa: E402
from coxswain.progress import progress                  # noqa: E402
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402

RACE_LENGTH = 4822.0
LEVELS = {"elite": ELITE, "club": CLUB, "junior": JUNIOR}


def one_run(variability, seed, duration, rate, dt=0.01):
    boat = catalog.eight(rate=rate)
    if variability is not None:
        model = dataclasses.replace(variability, seed=seed)
        model.reset()
        model.apply(boat)
    cox = Coxswain(rudder_override=lambda t, s: 0.0)
    sim = RowingSimulator(boat, coxswain=cox)
    result = sim.run(duration=duration, dt=dt, surge_speed=4.7)
    t = np.asarray(result.time)
    keep = t >= 0.5 * t[-1]
    velocity = np.asarray(result.velocity)[:2].T[keep]
    speed = float(np.hypot(*velocity.T).mean())
    yaw_rate = float(np.degrees(np.asarray(result.omega)[2])[keep].mean())
    roll_swing = float(np.ptp(np.degrees(np.asarray(result.attitude)[0])[keep]))
    return speed, yaw_rate, roll_swing


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=28.0)
    args = parser.parse_args(argv)

    base_speed, base_yaw, base_roll = one_run(None, 0, args.duration,
                                              args.rate)
    print("mean crew: %.3f m/s, yaw %+0.3f deg/s, roll swing %.2f deg"
          % (base_speed, base_yaw, base_roll))
    print()
    print("%-8s %10s %12s %12s %14s"
          % ("crew", "speed", "race spread", "yaw sd", "worst yaw"))

    bar = progress(total=len(LEVELS) * args.draws, desc="crew draws",
                   unit="crew")
    for name, level in LEVELS.items():
        speeds, yaws, rolls = [], [], []
        for seed in range(args.draws):
            s, y, r = one_run(level, 1000 + seed, args.duration, args.rate)
            speeds.append(s), yaws.append(y), rolls.append(r)
            bar.update(1)
        speeds, yaws = np.array(speeds), np.array(yaws)
        spread = RACE_LENGTH / speeds.min() - RACE_LENGTH / speeds.max()
        print("%-8s %7.3f±%.3f %9.1f s %9.3f %+11.3f deg/s"
              % (name, speeds.mean(), speeds.std(), spread,
                 yaws.std(), yaws[np.abs(yaws).argmax()]))
    bar.close()
    print()
    print("race spread is fastest-to-slowest draw over %.0f m; yaw is the"
          % RACE_LENGTH)
    print("standing drift the coxswain must trim out, rudder centred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
