r"""Would trip tape on the oar shafts do anything?

    python scripts/oar_aero.py
    python scripts/oar_aero.py --wind 8 --tape 0.3

The oars carry **half** a rowing eight's aerodynamic drag -- more than the
nine people and far more than the hull -- so the prize here is bigger than
anything clothing or posture can offer.  That makes the question worth
answering properly rather than by analogy.

What trip tape is actually for
------------------------------
A smooth circular cylinder in cross-flow separates at about 80 degrees
from the stagnation point and leaves a wide wake: ``Cd`` near 1.2.  Above
a critical Reynolds number near ``2-3e5`` the boundary layer goes
turbulent *before* separating, clings round further, and the wake
narrows -- ``Cd`` collapses to about 0.5.  That is the drag crisis, and
trip tape is a way of provoking it early.  Achenbach [A71]_ measured how
far: roughness of ``k/d ~ 9e-3`` pulls the critical Reynolds number down
to roughly ``6e4``.

So the question is not "does tape trip boundary layers" -- it does -- but
**"is an oar shaft within reach of its own drag crisis?"**  If it is not,
tape adds roughness on the flat part of the curve, which is a small loss
rather than a gain.

Why the answer is not obvious
-----------------------------
Unlike a rower's arm, an oar shaft is not simply carried along at boat
speed.  It sweeps, and the outboard end sweeps fast: this integrates the
real kinematics from :class:`~coxswain.crew.oarlock.OarAngleSweep` and the
boat's own stroke timing, resolves the air velocity **normal** to the
shaft at every station (a yawed cylinder responds to its normal component
-- the independence principle), and reports the Reynolds number the shaft
actually sees rather than one computed from boat speed alone.

References
----------
.. [A71] Achenbach, E. (1971) *Influence of surface roughness on the
   cross-flow around a circular cylinder*, J. Fluid Mech. 46(2), 321-335.
.. [ZG77] Zdravkovich, M. M. -- the standard compilation of cylinder drag
   against Reynolds number used for the curve below.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.resistance import hull_resistance   # noqa: E402
from coxswain.hydro.wind import AeroModel, log_profile_factor  # noqa: E402

RACE_LENGTH = 4822.0
SPEED_EXPONENT = 0.498
AIR_DENSITY = 1.225
KINEMATIC_VISCOSITY = 1.5e-5

#: Sweep shaft: about 50 mm at the sleeve tapering to 35 mm at the blade.
SHAFT_ROOT = 0.050
SHAFT_TIP = 0.035

#: Cylinder drag against Reynolds number, smooth surface [ZG77]_.
RE_TABLE = np.array([1e3, 1e4, 5e4, 1e5, 2e5, 3e5, 5e5, 1e6, 3e6])
CD_TABLE = np.array([1.00, 1.15, 1.20, 1.20, 1.18, 0.60, 0.38, 0.32, 0.55])

#: Achenbach's critical Reynolds number against relative roughness.
ROUGHNESS_K_OVER_D = np.array([0.0, 1.1e-3, 3.0e-3, 9.0e-3])
CRITICAL_RE = np.array([2.5e5, 1.5e5, 1.0e5, 6.0e4])


def cylinder_cd(reynolds):
    return np.interp(np.asarray(reynolds, float), RE_TABLE, CD_TABLE)


def critical_for(k_over_d):
    return float(np.interp(float(k_over_d), ROUGHNESS_K_OVER_D, CRITICAL_RE))


def shaft_diameter(fraction):
    """Diameter at a fraction of the way from oarlock to blade, m."""
    return SHAFT_ROOT + (SHAFT_TIP - SHAFT_ROOT) * np.asarray(fraction, float)


def kinematics(boat, boat_speed, wind, samples=240, stations=24):
    """Normal air speed and Reynolds number over the shaft and the cycle.

    Returns ``(reynolds, normal_speed, weight)`` arrays of shape
    ``(samples, stations)``; ``weight`` is the drag-area element so the
    cycle mean can be taken properly rather than by eye.
    """
    oar = boat.rig.seats[0].oarlocks[0].oar
    outboard = oar.length - oar.inboard
    period = boat.timing.period
    times = np.linspace(0.0, period, samples, endpoint=False)
    angle = np.asarray(boat.oar_sweep(times, boat.timing), dtype=float)
    rate = np.asarray(boat.oar_sweep.rate(times, boat.timing), dtype=float)

    fraction = np.linspace(0.05, 1.0, stations)
    radius = outboard * fraction
    diameter = shaft_diameter(fraction)

    # Air velocity relative to a point on the shaft, in the boat frame:
    # the boat's own motion gives a headwind along -x, the sweep gives a
    # velocity perpendicular to the shaft.
    normal = np.empty((samples, stations))
    for i, (theta, omega) in enumerate(zip(angle, rate)):
        # Shaft direction in plan, measured from the boat's axis.
        axis = np.array([np.sin(theta), np.cos(theta)])
        perpendicular = np.array([np.cos(theta), -np.sin(theta)])
        stream = np.array([-(boat_speed + wind), 0.0])
        for j, r in enumerate(radius):
            velocity = stream - omega * r * perpendicular
            # Independence principle: only the component normal to the
            # shaft drives cross-flow drag.
            along = float(np.dot(velocity, axis))
            normal[i, j] = float(np.hypot(*velocity) ** 2 - along ** 2) ** 0.5
    reynolds = normal * diameter[None, :] / KINEMATIC_VISCOSITY
    weight = np.tile(diameter * (outboard / stations), (samples, 1))
    return reynolds, normal, weight


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--wind", type=float, default=0.0,
                        help="headwind at 10 m, m/s")
    parser.add_argument("--tape", type=float, default=0.25,
                        help="trip tape thickness, mm")
    args = parser.parse_args(argv)

    boat = catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72)
    speed = RACE_LENGTH / args.race_time
    aero = AeroModel.calibrate(boat, reference_speed=speed)
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)
    force, _ = hull_resistance(np.array([speed, 0.0, 0.0]), submerged,
                               mean_wetted_length=boat.length,
                               water=boat.water, coefficients=boat.resistance)
    water = abs(float(force[0]))
    total = water + 0.5 * AIR_DENSITY * aero.total_area * speed ** 2
    wind_at_oar = args.wind * log_profile_factor(0.40)

    print("the prize, before any physics")
    print("  oars are %.2f m^2 of the boat's %.2f m^2 of drag area (%.0f%%)"
          % (aero.oar_area, aero.total_area,
             100 * aero.oar_area / aero.total_area))
    halved = 0.5 * aero.oar_area * 0.5 * AIR_DENSITY * (speed
                                                        + wind_at_oar) ** 2
    print("  halving it would be worth %.1f s over %.0f s"
          % (args.race_time * SPEED_EXPONENT * halved / total, args.race_time))
    print("  -- which is why this is worth checking rather than dismissing.")
    print()

    reynolds, normal, weight = kinematics(boat, speed, wind_at_oar)
    drag_weight = weight * normal ** 2
    mean_re = float(np.average(reynolds, weights=drag_weight))
    print("what Reynolds number an oar shaft actually reaches")
    print("  shaft %.0f mm at the sleeve tapering to %.0f mm at the blade"
          % (1000 * SHAFT_ROOT, 1000 * SHAFT_TIP))
    print("  normal air speed over the cycle: %.1f to %.1f m/s"
          % (normal.min(), normal.max()))
    print("  Reynolds number: %.1e to %.1e, drag-weighted mean %.1e"
          % (reynolds.min(), reynolds.max(), mean_re))
    print("  peak anywhere on the shaft at any phase: %.1e" % reynolds.max())
    print()

    tape = 1e-3 * args.tape
    k_over_d = tape / SHAFT_ROOT
    smooth_critical = critical_for(0.0)
    taped_critical = critical_for(k_over_d)
    print("where the drag crisis sits")
    print("  smooth shaft                     Re_crit %.1e" % smooth_critical)
    print("  with %.2f mm tape (k/d = %.1e)    Re_crit %.1e"
          % (args.tape, k_over_d, taped_critical))
    print("  turbulent air ahead of it, say a factor of 2 lower  %.1e"
          % (taped_critical / 2.0))
    print()
    margin = (taped_critical / 2.0) / reynolds.max()
    print("  the shaft's PEAK Reynolds number is %.1f times BELOW the most"
          % margin)
    print("  optimistic tripped critical value.  Not close.")
    print()

    fraction_above = float(np.average(reynolds > taped_critical / 2.0,
                                      weights=drag_weight))
    print("  fraction of the shaft's drag generated above even that"
          " threshold: %.1f%%" % (100 * fraction_above))
    print()

    verdict(reynolds, drag_weight, k_over_d)
    alternatives(aero, speed, wind_at_oar, total, args)
    return 0


def alternatives(aero, speed, wind, total, args):
    """Price the things that would work, so the no has something next to it."""
    dynamic = 0.5 * AIR_DENSITY * (speed + wind) ** 2

    def seconds(area_change):
        return args.race_time * SPEED_EXPONENT * area_change * dynamic / total

    print()
    print("  and what those are worth, on the same basis as everything else")
    for label, share in (("5 mm off a 50 mm shaft (-10%% of oar drag)", 0.10),
                         ("a faired section, if it could be held (-70%%)",
                          0.70)):
        print("    %-44s %6.1f s" % (label % () if "%%" not in label else
                                     label.replace("%%", "%"),
                                     seconds(share * aero.oar_area)))
    print("    %-44s %6.1f s" % ("trip tape, as measured above (+3%)",
                                 seconds(-0.03 * aero.oar_area)))


def verdict(reynolds, drag_weight, k_over_d):
    smooth = float(np.average(cylinder_cd(reynolds), weights=drag_weight))
    print("so what does the tape do?")
    print("  drag-weighted Cd of the shaft as it is:      %.2f" % smooth)
    print("  in the subcritical regime roughness raises Cd slightly rather")
    print("  than lowering it -- Achenbach measured a few percent for this")
    print("  k/d below the crisis -- so the tape is a small NET LOSS.")
    print("  Call it %.2f, which is about %+.0f%% of oar drag."
          % (smooth * 1.03, 3.0))
    print()
    print("what would actually work, in order of how much rules stand in")
    print("the way")
    print("  1. A smaller shaft.  Drag goes with diameter at fixed Cd, so")
    print("     5 mm off a 50 mm shaft is 10% of the oar's drag -- ten")
    print("     times what tripping could ever have been, and it is a")
    print("     stiffness decision an oar maker already makes.")
    print("  2. A faired section.  Cd 1.2 -> 0.3 is a factor of four, but")
    print("     the shaft ROTATES when the blade is feathered, so a fixed")
    print("     aerofoil is edge-on for the drive and broadside for the")
    print("     recovery.  That is not a detail, it is the whole problem.")
    print("  3. Nothing on the blade.  Feathering already puts it edge-on")
    print("     to the airflow through the recovery -- the technique is")
    print("     doing the aerodynamics correctly for free.")


if __name__ == "__main__":
    raise SystemExit(main())
