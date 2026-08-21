"""Constant rudder versus rudder modulated within the stroke.

The question, from a rower at practice: if the crew has a standing
port/starboard power imbalance, is it better to

* **A -- hold the average.** Set one rudder angle, leave it there, and let
  the boat yaw back and forth within each stroke so long as the heading
  averages out over several strokes.
* **B -- hold the heading.** Correct continuously, using a different rudder
  angle on the drive than on the recovery, so the boat points the same way
  all the time.

There is a real trade-off here, which is why it is worth simulating rather
than arguing about.

Against B: rudder drag goes as the *square* of deflection, so for a given
mean corrective moment, spreading the deflection evenly costs the least
drag.  Concentrating it into part of the cycle needs a bigger peak and
loses more.

For B: rudder force also goes as the square of the *water speed*, and the
boat is much faster during the drive than on the recovery.  Steering while
fast is cheap; steering while slow is expensive.  A controller that knows
this can buy its correction where it is cheapest.

Also for B: a yawing boat is not going where it is pointing.  It carries
leeway, which adds induced drag, and its track through the water is longer
than the straight line.

The metric is what actually wins races: **distance made good along the
intended heading, per second**.  Mean speed alone would reward a boat that
sails fast in the wrong direction.
"""

from __future__ import annotations

import numpy as np

from coxswain.boats import catalog
from coxswain.sim.control import Coxswain, HeadingController
from coxswain.sim.simulator import RowingSimulator

DURATION = 30.0
STEP = 0.005


def _measure(result, settle=0.4):
    """Distance made good, mean speed, and how much the heading wandered."""
    n = int(len(result.time) * settle)
    time = np.asarray(result.time)[n:]
    x = np.asarray(result.surge)[n:]
    y = np.asarray(result.sway)[n:]
    yaw = np.asarray(result.yaw)[n:]

    span = time[-1] - time[0]
    # made good along the intended heading, which is +x
    made_good = (x[-1] - x[0]) / span
    path = np.hypot(np.diff(x), np.diff(y)).sum() / span
    return {
        "made_good": made_good,
        "path_speed": path,
        "yaw_swing": np.degrees(yaw.max() - yaw.min()),
        "yaw_drift": np.degrees(yaw[-1] - yaw[0]),
        "cross_track": y.max() - y.min(),
    }


def run_constant(boat, split, deflection):
    cox = Coxswain(pressure_split=split,
                   rudder_override=lambda t, state: deflection)
    cox.heading = HeadingController(enabled=False)
    return RowingSimulator(boat, coxswain=cox).run(
        duration=DURATION, dt=STEP, surge_speed=4.6)


def run_tracking(boat, split, gain, rate_gain):
    """Strategy B: correct continuously.

    A PD loop on instantaneous heading.  It naturally produces different
    rudder on the drive than on the recovery, because the heading error it
    is chasing is itself different in the two phases -- which is exactly
    what a coxswain steering through the stroke is doing by hand.
    """
    cox = Coxswain(pressure_split=split)
    cox.heading = HeadingController(target=0.0, gain=gain,
                                    rate_gain=rate_gain, enabled=True)
    return RowingSimulator(boat, coxswain=cox).run(
        duration=DURATION, dt=STEP, surge_speed=4.6)


def trim_constant(boat, split, low=-0.25, high=0.25, iterations=18):
    """Find the constant rudder angle that leaves no net heading drift."""
    def drift(deflection):
        return _measure(run_constant(boat, split, deflection))["yaw_drift"]

    lo, hi = low, high
    flo = drift(lo)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        fmid = drift(mid)
        if abs(hi - lo) < 1e-4:
            break
        if (flo < 0) == (fmid < 0):
            lo, flo = mid, fmid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    boat = catalog.eight(rate=32.0)
    split = 0.06          # starboards ~6% up, ports ~6% down: a real bias

    print("Port/starboard power split: %+.0f%% per side" % (100 * split))
    print("Course: hold heading 0, %.0f s at rate 32\n" % DURATION)

    deflection = trim_constant(boat, split)
    print("A  constant rudder, trimmed to %.3f deg" % np.degrees(deflection))
    a = _measure(run_constant(boat, split, deflection))

    print("B  continuous correction (PD on instantaneous heading)")
    b = _measure(run_tracking(boat, split, gain=6.0, rate_gain=2.5))

    print()
    print("%-16s %12s %12s" % ("", "A constant", "B tracking"))
    for key, label, scale, unit in (
            ("made_good", "made good", 1.0, "m/s"),
            ("path_speed", "path speed", 1.0, "m/s"),
            ("yaw_swing", "yaw swing", 1.0, "deg"),
            ("cross_track", "cross track", 1.0, "m"),
    ):
        print("%-16s %12.4f %12.4f  %s"
              % (label, a[key] * scale, b[key] * scale, unit))

    gain = b["made_good"] - a["made_good"]
    print()
    print("B - A made good: %+.4f m/s  (%+.3f%%)"
          % (gain, 100 * gain / a["made_good"]))
    over_5k = gain / a["made_good"] * (5000.0 / a["made_good"])
    print("over a 5000 m Charles course that is %+.1f s" % -over_5k)


if __name__ == "__main__":
    main()
