"""Re-bisect the oarlock forces with aerodynamics switched on.

    python scripts/recalibrate.py

The peak oarlock forces in the catalog were bisected to put each class on
its known speed with **no aerodynamic model running**, so still-air air
drag lived inside the calibrated water resistance.  With aero now always
on, each class rows into its own apparent wind and the old forces leave
every boat slow.  This measures the aero-off speed each class was
calibrated to, then bisects a scale on the peak force until the aero-on
boat matches it, and prints the catalog patch.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.crew.oarlock import OarForceProfile       # noqa: E402
from coxswain.progress import progress                  # noqa: E402
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402

CLASSES = {"8+": (catalog.eight, 32.0), "4+": (catalog.coxed_four, 32.0),
           "1x": (catalog.single_scull, 30.0), "2x": (catalog.double_scull, 30.0)}

#: The speeds each class was originally calibrated to, and the original
#: peak forces that produced them, frozen here as constants.
#:
#: They MUST be constants.  The first version measured its target as "the
#: aero-off speed with the catalog's current force" -- which is correct
#: exactly once.  Run it a second time after applying its own output and
#: the target is measured with the already-raised force, so it rises, and
#: the script cheerfully scales the fleet again: the 8+ went 971 -> 1096
#: -> 1236 N across two runs while believing itself convergent both
#: times.  Calibrating against a baseline your own output mutates is a
#: ratchet, not a calibration.
BASELINE = {"8+": (5.167, 971.0), "4+": (4.688, 1073.0),
            "1x": (4.219, 709.0), "2x": (3.992, 485.0)}


def speed_of(boat, aero, duration=24.0, dt=0.01):
    sim = RowingSimulator(boat, coxswain=Coxswain(
        rudder_override=lambda t, s: 0.0), aero=aero)
    result = sim.run(duration=duration, dt=dt, surge_speed=4.5)
    t = np.asarray(result.time)
    keep = t >= 0.5 * t[-1]
    velocity = np.asarray(result.velocity)[:2].T[keep]
    return float(np.hypot(*velocity.T).mean())


def main():
    rows = []
    bar = progress(total=len(CLASSES) * 8, desc="recalibrating", unit="run")
    for name, (builder, rate) in CLASSES.items():
        target, old_peak = BASELINE[name]
        bar.update(1)

        # The bracket must contain the answer or the bisection silently
        # pins at the cap -- which is exactly what happened to both
        # sculls at [1.0, 1.15]: their scales came back 1.149, the 1x
        # failed validation at 3.86 m/s, and the "converged" number was
        # the ceiling wearing a disguise.  A single's aero share is far
        # larger than an eight's (one hull's windage, one rower's power),
        # and force enters speed at roughly the cube root, so the force
        # scale runs ~3x the speed deficit.
        low, high = 1.0, 1.45
        for _ in range(9):
            scale = 0.5 * (low + high)
            boat = builder(rate=rate,
                           force_profile=OarForceProfile(max_x=old_peak * scale))
            achieved = speed_of(boat, aero=None)
            if achieved < target:
                low = scale
            else:
                high = scale
            bar.update(1)
        scale = 0.5 * (low + high)
        rows.append((name, target, old_peak, old_peak * scale, scale))
    bar.close()

    print()
    print("%-4s %10s %12s %12s %8s" % ("", "target", "old peak", "new peak",
                                       "scale"))
    for name, target, old, new, scale in rows:
        print("%-4s %8.3f m/s %10.1f %12.1f %8.3f"
              % (name, target, old, new, scale))
    print()
    print("PEAK_OARLOCK_FORCE = {" +
          ", ".join('"%s": %.1f' % (n, new) for n, _t, _o, new, _s in rows) +
          "}")


if __name__ == "__main__":
    raise SystemExit(main())
