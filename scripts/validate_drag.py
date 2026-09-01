r"""Test the resistance model against a measured eight.

    python scripts/validate_drag.py

Buckmann and Harris [BH14]_ coasted a lightweight men's 8+ down from
racing speed with the blades feathered clear of the water, logged the
deceleration on GPS, and turned it into drag: ``F = m a``, eighteen trials
across several days and conditions.  It is the only public measurement of
total drag on an eight, and this project has never been compared to it.

What the number is, and what it is not
--------------------------------------
Their "drag coefficient" is a **drag constant** with units of kg/m, from
``C_D = F / v^2``.  They report ``10.5 +/- 1.9``, 95% interval 9.6 to
11.4, evaluated at the start of each coast-down -- that is, at their
highest speed, near 5.5-6 m/s.

Two things have to be right for the comparison to mean anything:

**It is TOTAL drag, not hull drag.**  A coasting shell carries eight
bodies and sixteen feathered oars through the air.  So the model side of
this test is hull resistance plus the calibrated aerodynamic term, not
hull alone.

**Their constant is not constant.**  Their own figure 4 shows ``C_D``
running from about 25 at 3 m/s down to 10 at 6 m/s, and their per-trial
fits give exponents from 1.4 to 2.1 rather than 2.  A single quadratic
constant quoted at one speed will disagree with any model evaluated
somewhere else, and comparing at the wrong speed is the easiest way to
manufacture a discrepancy that is not there.

So the test is run across their speed range, against the exponent band
their trials actually show, rather than against the headline figure.

References
----------
.. [BH14] Buckmann, J. G. and Harris, S. D. (2014) *An experimental
   determination of the drag coefficient of a Mens 8+ racing shell*,
   SpringerPlus 3:512.  Table 2 and figures 4-5.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.resistance import hull_resistance   # noqa: E402
from coxswain.hydro.wind import AeroModel               # noqa: E402

#: Buckmann and Harris table 1: system masses, kg.
BH_MASS = {"rowers": 655.0, "shell": 89.0, "oars": 22.5}
#: Their headline drag constant and 95% interval, kg/m.
BH_CD = 10.5
BH_CD_LOW, BH_CD_HIGH = 9.6, 11.4
#: The speed at which that constant was evaluated -- the start of the
#: coast-down.  Their figure 1 tops out near 7 m/s under power and the
#: deceleration phase begins around 6.
BH_REFERENCE_SPEED = 6.0
#: Per-trial power-law exponents from their table 2, F = a v^b.
BH_EXPONENTS = (1.6, 1.9, 1.9, 1.8, 1.9, 1.7, 1.8, 2.0, 1.8, 2.1, 1.4,
                2.1, 1.8, 1.5, 2.1, 2.0, 2.1, 2.0)


def model_drag(boat, speeds, aero):
    """Total resistance -- hull plus air -- at each speed, N."""
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)
    out = []
    for speed in np.atleast_1d(speeds):
        force, detail = hull_resistance(
            np.array([float(speed), 0.0, 0.0]), submerged,
            mean_wetted_length=boat.length, water=boat.water,
            coefficients=boat.resistance)
        air = 0.5 * 1.225 * aero.total_area * float(speed) ** 2
        out.append((abs(float(force[0])), air, detail))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rower-mass", type=float, default=74.8,
                        help="their subjects averaged 165 lb")
    parser.add_argument("--speeds", type=float, nargs="+",
                        default=[3.0, 4.0, 4.23, 5.0, 5.5, 6.0])
    args = parser.parse_args(argv)

    total_mass = sum(BH_MASS.values())
    print("Buckmann and Harris 2014, lightweight men's 8+ coast-down")
    print("  system mass %.0f kg (%s)"
          % (total_mass, ", ".join("%s %.0f" % kv for kv in BH_MASS.items())))
    print("  C_D = %.1f kg/m (95%% %.1f to %.1f), quoted near %.1f m/s"
          % (BH_CD, BH_CD_LOW, BH_CD_HIGH, BH_REFERENCE_SPEED))
    print("  per-trial exponents span %.1f to %.1f, mean %.2f -- their own"
          % (min(BH_EXPONENTS), max(BH_EXPONENTS), np.mean(BH_EXPONENTS)))
    print("  data is not quadratic, so the constant only applies where it")
    print("  was measured.")
    print()

    boat = catalog.eight(rate=32.0, rower_mass=args.rower_mass,
                         rower_stature=1.86, coxswain_mass=56.7)
    aero = AeroModel.calibrate(boat, reference_speed=5.5)
    rows = model_drag(boat, args.speeds, aero)

    print("model against measurement, across the speed range")
    print("  %6s %9s %8s %9s %10s %10s %9s"
          % ("v m/s", "hull N", "air N", "total N", "measured N",
             "model C_D", "ratio"))
    for speed, (hull, air, _detail) in zip(args.speeds, rows):
        measured = BH_CD * speed ** 2
        model_cd = (hull + air) / speed ** 2
        print("  %6.2f %9.0f %8.0f %9.0f %10.0f %10.1f %9.2f"
              % (speed, hull, air, hull + air, measured, model_cd,
                 (hull + air) / measured))
    print()
    print("  The 'measured N' column applies their quadratic constant at")
    print("  every speed, which their own figure 4 says is wrong away from")
    print("  6 m/s -- it is there to show the shape disagreement, not as a")
    print("  target.  The honest comparison is the single row at %.1f m/s."
          % BH_REFERENCE_SPEED)
    print()

    index = int(np.argmin(np.abs(np.array(args.speeds)
                                 - BH_REFERENCE_SPEED)))
    speed = args.speeds[index]
    hull, air, _ = rows[index]
    model_cd = (hull + air) / speed ** 2
    inside = BH_CD_LOW <= model_cd <= BH_CD_HIGH
    print("AT THEIR MEASUREMENT SPEED, %.1f m/s:" % speed)
    print("  model  C_D = %.1f kg/m  (hull %.0f N + air %.0f N)"
          % (model_cd, hull, air))
    print("  theirs C_D = %.1f kg/m, 95%% interval %.1f to %.1f"
          % (BH_CD, BH_CD_LOW, BH_CD_HIGH))
    print("  ratio %.2f -- %s"
          % (model_cd / BH_CD,
             "INSIDE their confidence interval" if inside
             else "OUTSIDE their confidence interval"))
    print()

    exponent(boat, aero)
    return 0


def exponent(boat, aero):
    """Does the model's drag rise with speed the way theirs does?"""
    speeds = np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    rows = model_drag(boat, speeds, aero)
    total = np.array([h + a for h, a, _ in rows])
    slope = float(np.polyfit(np.log(speeds), np.log(total), 1)[0])
    print("the SHAPE, which is a second and independent test")
    print("  model exponent over 3.5-6.0 m/s: %.2f" % slope)
    print("  their eighteen trials: %.1f to %.1f, mean %.2f"
          % (min(BH_EXPONENTS), max(BH_EXPONENTS), np.mean(BH_EXPONENTS)))
    inside = min(BH_EXPONENTS) <= slope <= max(BH_EXPONENTS)
    print("  %s" % ("inside their spread" if inside
                    else "OUTSIDE their spread"))
    print()
    print("  Exponent and magnitude are separate claims and a model can")
    print("  pass one while failing the other.  Passing both is what would")
    print("  make the resistance model measured rather than assumed.")


if __name__ == "__main__":
    raise SystemExit(main())
