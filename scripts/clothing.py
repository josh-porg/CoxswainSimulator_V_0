"""What the crew wears, and what the coxswain does with their body.

    python scripts/clothing.py
    python scripts/clothing.py --cox-stature 1.73 --cox-mass 68

Two questions that look the same and are not.  The coxswain can change
their *shape* stroke by stroke and cannot change their size; the crew can
change their *surface* and their fit and nothing else.  So they get
different answers, and one of them is a flat no.

The skinsuit trick does not transfer, and the reason is Reynolds number
--------------------------------------------------------------------
Cycling's textured sleeves are not a fabric story, they are a boundary
layer story.  A limb is roughly a cylinder, and a smooth cylinder's drag
coefficient collapses from about 1.2 to about 0.6 when its Reynolds
number crosses the critical value near ``2e5`` -- the golf-ball effect.
Surface texture trips that transition early, which is why a skinsuit with
rough sleeves is faster than a smooth one.

The whole trick depends on sitting near that transition.  A cyclist at
15 m/s with a 0.10 m arm is at ``Re = 1e5`` and within reach of it.  A
rower at 4 m/s is at ``Re = 3e4`` -- **a factor of seven below**, on the
flat part of the curve where roughness does nothing useful and may cost a
little.  Textured racing kit sold on cycling's evidence is not
transferable to a boat moving at a third of the speed, and this script
prints the numbers rather than asserting it.

What is left is fit, and it is not nothing
------------------------------------------
Loose cloth flaps, separates early and adds pressure drag, and none of
that needs a critical Reynolds number to happen.  A jacket over a
unisuit, a loose pinnie, hair out -- those are real area and real
separation, and they are free to remove.

And the oars are half of it
---------------------------
Worth saying before anyone buys kit: on Kleshnev's split the oars carry
**50%** of a shell's aerodynamic drag, the bodies 35%, the hull 15%.  The
largest aerodynamic object in a rowing eight is not a person.
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

#: Critical Reynolds number for a smooth circular cylinder -- where the
#: drag crisis happens and where surface texture starts to pay.
CRITICAL_RE = 2.0e5

#: Typical limb diameters, m.
LIMB = {"forearm": 0.085, "upper arm": 0.100, "thigh": 0.150}

#: How much of the crew's drag area loose kit adds.  Cycling measurements
#: of a fitted skinsuit against a loose jersey span 5-10% of *rider* CdA;
#: rowing kit is closer-fitting to begin with, so this is the low end.
LOOSE_KIT_PENALTY = 0.05

#: Coxswain shielding behind eight rowers, from scripts/cox_fairing.py.
SHIELDING_HEAD_ON = 0.35
CD_BODY = 1.10


def reynolds(speed, diameter):
    return speed * diameter / KINEMATIC_VISCOSITY


def cox_area(stature: float, mass: float, lean: float = 0.0):
    """Frontal area of a seated coxswain above the gunwale, m^2.

    Shoulder breadth from stature by the usual anthropometric ratio
    (about 0.259 of stature), exposed height scaled from a 1.70 m
    reference, and mass entering through breadth because a heavier person
    of the same height is broader.

    ``lean`` is the torso angle from vertical in radians.  Leaning takes
    the chest out of the projection and drops the head, and only the
    first of those is in the area -- the second is in the wind profile
    and in how deep the coxswain sits in the crew's wake.
    """
    breadth = 0.259 * stature * (mass / 62.0) ** (1.0 / 3.0)
    upright_height = 0.50 * (stature / 1.70)
    # The head stays in the flow whatever the torso does; only the trunk
    # foreshortens.  Split the projection so a full lean cannot drive the
    # area to zero, which would be nonsense.
    head = 0.35 * breadth * 0.22
    trunk = breadth * (upright_height - 0.22)
    return float(head + trunk * np.cos(lean))


def cox_height(stature: float, lean: float = 0.0):
    """Height of the coxswain's centre of area above the water, m."""
    return float(0.28 + 0.42 * (stature / 1.70) * np.cos(lean))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--cox-stature", type=float, default=1.727,
                        help="metres (default 1.727 = 5 ft 8 in)")
    parser.add_argument("--cox-mass", type=float, default=68.0,
                        help="kg (default 68 = 150 lb)")
    parser.add_argument("--winds", type=float, nargs="+",
                        default=[0.0, 4.0, 8.0],
                        help="headwind at 10 m, m/s")
    args = parser.parse_args(argv)

    boat = catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72,
                         coxswain_mass=args.cox_mass)
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

    def seconds(drag_change):
        """Seconds off the race for a resistance change in newtons."""
        return float(args.race_time * SPEED_EXPONENT * drag_change / total)

    print("where the air actually pushes (Kleshnev's split, calibrated here)")
    print("  total drag area %.2f m^2 at %.2f m/s" % (aero.total_area, speed))
    for name, area in (("oars", aero.oar_area), ("bodies", aero.crew_area),
                       ("hull and riggers", aero.hull_area)):
        print("  %-18s %5.2f m^2  %3.0f%%"
              % (name, area, 100 * area / aero.total_area))
    print()

    texture(speed, args)
    fit(aero, speed, seconds, args)
    lean(aero, speed, seconds, args)
    return 0


def texture(speed, args):
    """Is a rower's limb anywhere near the drag crisis?  No."""
    print("does textured fabric do anything?  Reynolds number decides")
    print("  %-12s %8s %12s %12s %10s"
          % ("limb", "d (m)", "rower Re", "cyclist Re", "critical"))
    for name, diameter in LIMB.items():
        # A limb's air speed is the boat's, plus the limb's own motion;
        # 1.4x is the rough peak for a hand at the catch.
        rower = reynolds(1.4 * speed, diameter)
        cyclist = reynolds(15.0, diameter)
        print("  %-12s %8.3f %12.0e %12.0e %10.0e"
              % (name, diameter, rower, cyclist, CRITICAL_RE))
    print("  a rower's limbs sit a factor of %.0f below the drag crisis."
          % (CRITICAL_RE / reynolds(1.4 * speed, LIMB["upper arm"])))
    print("  Textured or 'tripped' fabric works by crossing that line early.")
    print("  There is no line to cross here, so it buys nothing -- and the")
    print("  small roughness penalty on the flat part of the curve is real,")
    print("  so it is very slightly worse than smooth.")
    print()


def fit(aero, speed, seconds, args):
    """What loose kit costs, which is the part that does transfer."""
    print("what loose kit costs (%.0f%% of the crew's drag area)"
          % (100 * LOOSE_KIT_PENALTY))
    print("  %-16s %10s %10s %10s"
          % ("headwind", "extra drag", "seconds", "per rower"))
    for wind in args.winds:
        at_crew = wind * log_profile_factor(0.60)
        apparent = speed + at_crew
        extra = (0.5 * AIR_DENSITY * aero.crew_area * LOOSE_KIT_PENALTY
                 * apparent ** 2)
        print("  %8.0f m/s %9.2f N %9.2f s %9.2f s"
              % (wind, extra, seconds(extra), seconds(extra) / 9.0))
    print()
    print("  So: fitted one-piece, nothing over it, nothing flapping, hair")
    print("  tied and tucked.  A jacket left on for a cold start is the whole")
    print("  of this number, and it is larger than the rig, the seating and")
    print("  the wind-adapted line put together.")
    print()
    print("  Note the basis: the crew's %.2f m^2 is an *effective* area, not"
          % aero.crew_area)
    print("  nine isolated bodies -- nine people in clear air would be about")
    print("  3.1 m^2, so the calibration has already discounted them by two")
    print("  thirds for shielding each other.  Taking 5% of the effective")
    print("  figure is therefore the right scale, not an optimistic one.")
    print()


def lean(aero, speed, seconds, args):
    """The coxswain's own lever, at the coxswain's actual size."""
    stature, mass = args.cox_stature, args.cox_mass
    print("the coxswain leaning forward (%.2f m, %.0f kg)" % (stature, mass))
    print("  %-14s %8s %8s %11s %10s"
          % ("posture", "area", "height", "8 m/s head", "4 m/s head"))
    upright_area = cox_area(stature, mass, 0.0)
    rows = []
    for label, angle in (("sitting up", 0.0), ("easy lean, 25", np.radians(25)),
                         ("hard tuck, 45", np.radians(45)),
                         ("flat, 60", np.radians(60))):
        area = cox_area(stature, mass, angle)
        height = cox_height(stature, angle)
        gains = []
        for wind in (8.0, 4.0):
            at_cox = wind * log_profile_factor(height)
            apparent = speed + at_cox
            drag = (0.5 * AIR_DENSITY * area * CD_BODY * SHIELDING_HEAD_ON
                    * apparent ** 2)
            gains.append(drag)
        rows.append((label, area, height, gains))
    reference = rows[0][3]
    for label, area, height, gains in rows:
        print("  %-14s %7.3f %7.2f %10.2f s %9.2f s"
              % (label, area, height,
                 seconds(reference[0] - gains[0]),
                 seconds(reference[1] - gains[1])))
    print()
    print("  Area falls from %.3f to %.3f m^2 across that range, and the head"
          % (upright_area, rows[-1][1]))
    print("  drops into slower air as well -- both effects are in the table.")
    print("  It is free, it is reversible stroke by stroke, and unlike a")
    print("  fairing it costs nothing when the wind is behind you.")
    print("  The catch is that you have to be able to see and to steer, so")
    print("  spend it on the straights into a headwind and sit up for the")
    print("  bridges.")
    print()
    print("  Trim: leaning moves the coxswain's mass from station -6.1 m")
    print("  toward amidships, which unloads a stern that a heavier")
    print("  coxswain already sits deep in.  Second order, and in the")
    print("  helpful direction, so it is not an argument against.")


if __name__ == "__main__":
    raise SystemExit(main())
