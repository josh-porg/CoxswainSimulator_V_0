r"""Does rate do anything once power is held fixed?  Holt says yes.

    python scripts/stroke_rate.py
    python scripts/stroke_rate.py --rates 22 26 30 34 --boat 1x

Holt et al. [H20]_ instrumented forty-seven 2000 m races -- singles and
pairs, per-stroke Peach PowerLine gate force and OptimEye GPS -- and asked
which technical variables predict boat velocity.  Their central result is
the one worth testing a simulator against:

**Stroke rate keeps a large positive effect on velocity AFTER adjusting
for power output.**  And mean and peak force turn *negative* after the
same adjustment.  Together those say: at a given power, row at a higher
rate with less force per stroke.  Their explanation is the force-velocity
relation in muscle -- faster oar angular velocity costs less force for the
same power -- but whatever the physiological cause, the *boat* has to
oblige, and whether it does is a question about hydrodynamics that this
model can answer.

The experiment, matched to theirs
---------------------------------
"Adjusted for power" is the whole design.  Raising the rate at a fixed
force profile raises power, and of course a more powerful crew goes
faster; that tells us nothing.  So here the oar force is scaled at every
rate to hold the delivered power constant, and the question is what the
*remaining* speed difference is.

If the model reproduces their sign and rough magnitude, the mechanism is
visible in it: a shorter recovery gives the boat less time to decelerate,
the within-stroke velocity range falls, and the ``<v^n>`` penalty that
:mod:`scripts.unsteady` prices goes down.  That is a hydrodynamic reason
for a result they explain physiologically, and the two are not in
competition.

Their Table 1, which is also a calibration target
--------------------------------------------------
Holt's within-stroke velocity range is the number this project has been
missing.  For a women's single: 2.14 m/s of range on a 4.18 m/s mean, so
**51% peak to peak**.  Men's pairs: 2.71 on 4.97, 55%.  Those are small
boats and an eight is smoother, but it brackets what a real hull does and
it is measured rather than recalled.

References
----------
.. [H20] Holt, A. C., Aughey, R. J., Ball, K., Hopkins, W. G. and
   Siegel, R. (2020) *Technical determinants of on-water rowing
   performance*, Frontiers in Sports and Active Living 2:589013.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.resistance import hull_resistance   # noqa: E402
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402

RACE_LENGTH = 4822.0
MASTERS_POWER = 0.658

#: Holt et al. table 1, per boat class: stroke rate, within-stroke
#: velocity range (m/s), distance per stroke (m), power per rower (W).
HOLT = {
    "M1x": (34.7, 2.27, 7.97, 334.0),
    "W1x": (32.8, 2.14, 7.65, 223.0),
    "M2-": (38.1, 2.71, 7.82, 760.0 / 2),
    "W2-": (35.1, 2.30, 7.38, 481.0 / 2),
}


def build(rate, scale, boat_class="8+"):
    if boat_class == "1x":
        boat = catalog.single_scull(rate=rate, rower_mass=72.0)
    else:
        boat = catalog.eight(rate=rate, rower_mass=72.0, rower_stature=1.72)
    boat.power_scales = np.full(boat.n_seats, float(scale))
    return boat


def steady(boat, guess, duration=26.0, dt=0.01):
    """Mean speed and within-stroke range over whole settled cycles."""
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0)
    sim = RowingSimulator(boat, coxswain=cox)
    result = sim.run(duration=duration, dt=dt, surge_speed=guess)
    t = np.asarray(result.time)
    v = np.hypot(*np.asarray(result.velocity)[:2])
    period = boat.timing.period
    cycles = int((0.5 * t[-1]) // period)
    keep = t >= t[-1] - cycles * period
    v = v[keep]
    return float(v.mean()), float(v.max() - v.min())


def power_of(boat, speed):
    """Propulsive power at steady state: resistance times speed, W."""
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)
    force, _ = hull_resistance(np.array([float(speed), 0.0, 0.0]), submerged,
                               mean_wetted_length=boat.length,
                               water=boat.water, coefficients=boat.resistance)
    return abs(float(force[0])) * float(speed)


def at_fixed_power(rate, target, boat_class, guess, tol=2.0, limit=16):
    """Scale the oar force until the delivered power matches ``target``."""
    low, high = 0.15, 2.2
    for _ in range(limit):
        mid = 0.5 * (low + high)
        boat = build(rate, mid, boat_class)
        speed, swing = steady(boat, guess)
        power = power_of(boat, speed)
        if abs(power - target) < tol:
            return mid, speed, swing, power
        if power < target:
            low = mid
        else:
            high = mid
    return mid, speed, swing, power


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[22.0, 26.0, 30.0, 34.0])
    parser.add_argument("--boat", default="8+", choices=("8+", "1x"))
    parser.add_argument("--race-time", type=float, default=1140.0)
    args = parser.parse_args(argv)

    reference = RACE_LENGTH / args.race_time
    print("Holt et al. table 1 -- what a real racing boat does")
    print("  %-6s %8s %12s %10s %10s"
          % ("class", "rate", "range m/s", "d/stroke", "% of mean"))
    for name, (rate, span, per_stroke, _power) in HOLT.items():
        mean = per_stroke * rate / 60.0
        print("  %-6s %8.1f %12.2f %10.2f %9.0f%%"
              % (name, rate, span, per_stroke, 100 * span / mean))
    print("  Those are singles and pairs.  An eight is smoother, but this")
    print("  brackets what a real hull does and it is measured.")
    print()

    # Anchor: the calibrated masters eight at its own rate and power.
    anchor = build(28.0, MASTERS_POWER, args.boat)
    speed, swing = steady(anchor, reference)
    target = power_of(anchor, speed)
    print("this model at rate 28: %.3f m/s, range %.2f m/s (%.0f%% of mean),"
          % (speed, swing, 100 * swing / speed))
    print("  delivering %.0f W to the water" % target)
    print()

    print("HOLT'S EXPERIMENT: vary the rate, hold the power")
    print("  %-7s %9s %10s %11s %10s %11s"
          % ("rate", "force x", "m/s", "vs rate 28", "range m/s", "% of mean"))
    rows = []
    for rate in args.rates:
        scale, v, span, power = at_fixed_power(rate, target, args.boat,
                                               reference)
        rows.append((rate, v, span))
        print("  %-7.0f %9.3f %10.4f %+10.2f%% %10.2f %10.0f%%"
              % (rate, scale, v, 100 * (v / speed - 1.0), span,
                 100 * span / v))
    print()

    rates = np.array([r for r, _v, _s in rows])
    speeds = np.array([v for _r, v, _s in rows])
    if len(rows) > 1:
        slope = np.polyfit(rates, 100 * speeds / speeds.mean(), 1)[0]
        print("  the model gives %+.2f%% of velocity per stroke per minute,"
              % slope)
        print("  at constant power.")
        print("  Holt: two within-crew SDs of rate (about 3.6 spm in a")
        print("  women's single) moved velocity by roughly +2 to +4%% after")
        print("  adjusting for power, so about +0.6 to +1.1%% per spm.")
        verdict = ("SAME SIGN, comparable magnitude" if 0.2 < slope < 2.0
                   else ("same sign, wrong size" if slope > 0
                         else "WRONG SIGN -- the model disagrees with the data"))
        print("  verdict: %s" % verdict)
    print()
    print("  If the sign agrees, the mechanism is visible here: the range")
    print("  column falls with rate, and scripts/unsteady.py prices exactly")
    print("  that -- a shorter recovery gives the boat less time to slow")
    print("  down, and the <v^n> penalty goes with the square of the swing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
