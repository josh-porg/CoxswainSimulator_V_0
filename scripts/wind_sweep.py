"""How much does wind matter on this course?

    python scripts/wind_sweep.py
    python scripts/wind_sweep.py --speeds 4 8 --duration 30

Uniform wind over the whole reach -- the honest first cut.  The real
Charles wind field is anything but uniform (the basin is open, the upper
reach is treed, and every bridge cuts a slot), but a uniform sweep bounds
the sensitivity and says whether the microclimate is worth modelling.

Wind loads are charged as the excess over still air, because the hull
calibration already contains still-air drag; see the note in
``coxswain/sim/simulator.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.wind import UniformWind             # noqa: E402
from coxswain.progress import progress                  # noqa: E402
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402

RACE_LENGTH = 4822.0


def steady(boat, wind, duration=30.0, dt=0.01):
    cox = Coxswain(rudder_override=lambda t, s: 0.0)
    sim = RowingSimulator(boat, coxswain=cox, wind=wind)
    result = sim.run(duration=duration, dt=dt, surge_speed=4.7)
    t = np.asarray(result.time)
    keep = t >= 0.5 * t[-1]
    velocity = np.asarray(result.velocity)[:2].T[keep]
    speed = float(np.hypot(*velocity.T).mean())
    yaw_rate = float(np.degrees(np.asarray(result.omega)[2])[keep].mean())
    roll = float(np.degrees(np.asarray(result.attitude)[0])[keep].mean())
    return speed, yaw_rate, roll


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speeds", type=float, nargs="+", default=[4.0, 8.0],
                        help="wind speeds, m/s (4 is a working breeze, 8 a "
                             "hard day)")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=28.0)
    args = parser.parse_args(argv)

    boat = catalog.eight(rate=args.rate)
    base_speed, base_yaw, base_roll = steady(boat, None, args.duration)
    print("still air: %.3f m/s, race %.1f s"
          % (base_speed, RACE_LENGTH / base_speed))
    print()

    # The boat rows along +x, +y to port.  ``bearing`` is the direction
    # the wind blows TOWARDS (the module is explicit that this is the
    # opposite of the meteorological convention): a headwind blows towards
    # -x, wind from starboard blows towards +y.
    cases = []
    for w in args.speeds:
        cases += [("head, %g m/s" % w, w, np.pi),
                  ("tail, %g m/s" % w, w, 0.0),
                  ("cross from starboard, %g m/s" % w, w, +np.pi / 2),
                  ("cross from port, %g m/s" % w, w, -np.pi / 2)]

    print("%-28s %8s %10s %9s %9s" % ("wind", "speed", "race time",
                                      "yaw bias", "heel"))
    bar = progress(total=len(cases), desc="wind cases", unit="case")
    for label, wind_speed, bearing in cases:
        speed, yaw_rate, roll = steady(
            boat, UniformWind(speed=wind_speed, bearing=bearing),
            args.duration)
        print("%-28s %7.3f  %+8.1f s %+8.3f %+8.2f"
              % (label, speed, RACE_LENGTH / speed - RACE_LENGTH / base_speed,
                 yaw_rate - base_yaw, roll - base_roll))
        bar.update(1)
    bar.close()
    print()
    print("race time is the change over %.0f m at the steady speed; yaw" % RACE_LENGTH)
    print("bias and heel are what the coxswain and crew must hold against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
