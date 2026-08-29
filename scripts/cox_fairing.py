"""Is an aerodynamic coxswain worth anything?

    python scripts/cox_fairing.py
    python scripts/cox_fairing.py --winds 0 4 8 --from 250 340 070

An aero helmet and a tapered tail behind the coxswain's back is a real
proposal -- cyclists have taken exactly that route for forty years -- and
it deserves a number rather than an opinion.  The number turns out to
hinge on one geometric fact that has nothing to do with the fairing.

The coxswain is the last body the air meets
-------------------------------------------
Rowers face the stern, so their backs face the bow.  In a stern-loaded
eight the coxswain sits **aft of everyone**, at station -6.10 m against
stroke's -4.30 m, and sits *lower* than the crew as well: 0.70 m of eye
height against shoulders that pass well above a metre.

So in a headwind the coxswain is the ninth body in a line of nine, deep
inside a turbulent wake that eight people and sixteen oars have already
made.  Fairing the coxswain is fairing the most sheltered person in the
boat, and the honest question is not "how good is the fairing" but "how
much drag is left there to remove".

Where it might actually pay
---------------------------
A crosswind, where nobody is shielding anybody.  Except that the boat's
own speed swings the apparent wind forward -- at 4.2 m/s of boat speed a
4 m/s beam wind arrives 43 degrees off the bow, not 90 -- so even a pure
crosswind is partly a headwind as far as the coxswain is concerned, and
partly shielded.  That is computed here rather than assumed.

And a tail fairing adds side area at the very stern, which is a
weathervane: it puts a yaw moment on the boat in exactly the conditions
it was supposed to help.  That is priced too.

The numbers this rests on
-------------------------
Every one of these is an estimate, and the sensitivity sweep at the end
exists because of it:

* **Coxswain frontal area, 0.20 m^2.**  Head, shoulders and upper chest
  above the gunwale for a 55 kg coxswain sitting normally.
* **Bare drag coefficient, 1.10.**  An upright human head-and-shoulders
  is a bluff body; Hoerner and the cycling literature both put it near
  unity.
* **Faired, 0.55.**  A helmet plus a partial tail.  A *full* teardrop
  reaches 0.1, but nobody is going to row with a two-metre tail; halving
  the coefficient is what a helmet and a modest back shell achieve.
* **Shielding factor, 0.35 head-on.**  Drag of a bluff body deep in the
  wake of a line of others, relative to the same body in clear air.
  In-line bluff-body data spans 0.3-0.6; the coxswain is ninth and lower
  than the rest, so this is the low end of that range.
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

COX_AREA = 0.20          # m^2 above the gunwale
CD_BARE = 1.10
CD_FAIRED = 0.55
SHIELDING_HEAD_ON = 0.35
#: Beyond this angle off the bow the crew stops shielding the coxswain.
SHIELDED_ARC = np.radians(40.0)
#: Height of the coxswain's exposed area above the water, m.
COX_HEIGHT = 0.70
#: Extra side area a tapered back shell adds, m^2, and how far aft of the
#: hull's yaw pivot it sits.  The pivot for a yawing slender hull is about
#: a quarter of the length from the bow, so the stern seat is a long arm.
FAIRING_SIDE_AREA = 0.10
FAIRING_ARM = 10.0


#: How much of a fairing's clean-air benefit survives in a wake.  1.0
#: dead ahead is wrong in the other direction -- head-on IS the wake -- so
#: this is deliberately lowest where the shielding is strongest.
WAKE_EFFECTIVENESS_MIN = 0.45


def wake_effectiveness(angle):
    """Fraction of the fairing's clean-air benefit that survives.

    Worst where the coxswain is deepest in the crew's wake, best on the
    beam where the flow reaching them is clean.  The two effects partly
    cancel -- head-on there is little drag left to save AND the fairing
    saves less of it; on the beam there is more drag and the fairing works
    better -- which is why the answer is flatter across wind angle than
    either factor alone would suggest.
    """
    angle = np.abs(np.asarray(angle, dtype=float))
    ramp = np.clip((angle - SHIELDED_ARC) / np.radians(35.0), 0.0, 1.0)
    return WAKE_EFFECTIVENESS_MIN + (1.0 - WAKE_EFFECTIVENESS_MIN) * ramp


def shielding(angle):
    """How much of the coxswain's clear-air drag survives, 0 to 1.

    One inside the shielded arc means no shielding at all.  Ramped rather
    than switched, because a wake edge is not a wall and a step here would
    put a discontinuity straight into a sweep.
    """
    angle = np.abs(np.asarray(angle, dtype=float))
    ramp = np.clip((angle - SHIELDED_ARC) / np.radians(35.0), 0.0, 1.0)
    return SHIELDING_HEAD_ON + (1.0 - SHIELDING_HEAD_ON) * ramp


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--winds", type=float, nargs="+",
                        default=[0.0, 4.0, 8.0],
                        help="true wind at 10 m, m/s")
    parser.add_argument("--angles", type=float, nargs="+",
                        default=[0.0, 45.0, 90.0, 135.0, 180.0],
                        help="degrees off the bow the TRUE wind comes from")
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
    # Total resistance, water plus air, because that is what the crew's
    # power actually goes against and the denominator has to match.
    total = water + 0.5 * AIR_DENSITY * aero.total_area * speed ** 2

    print("masters eight at %.2f m/s: %.0f N of water resistance" % (speed,
                                                                     water))
    print("  total aerodynamic drag area %.2f m^2 (%.0f%% of resistance in "
          "still air)"
          % (aero.total_area,
             100 * 0.5 * AIR_DENSITY * aero.total_area * speed ** 2 / water))
    print("  the coxswain, in clear air, would be %.2f m^2 of that -- %.0f%%"
          % (COX_AREA * CD_BARE, 100 * COX_AREA * CD_BARE / aero.total_area))
    print("  behind eight rowers, head-on, only %.0f%% of that survives"
          % (100 * SHIELDING_HEAD_ON))
    print()

    print("what the fairing saves, by wind (true wind at 10 m; the wind the")
    print("coxswain sits in at %.2f m is %.0f%% of it)"
          % (COX_HEIGHT, 100 * log_profile_factor(COX_HEIGHT)))
    print("  %-10s" % "true wind" + "".join(
        "%10s" % ("%.0f deg" % a) for a in args.angles))
    best = 0.0
    for wind in args.winds:
        at_cox = wind * log_profile_factor(COX_HEIGHT)
        row = "  %6.0f m/s" % wind
        for angle in args.angles:
            saved = seconds_saved(at_cox, np.radians(angle), speed, total,
                                  args.race_time)
            best = max(best, saved)
            row += "%9.2fs" % saved
        print(row)
    print()
    print("  0 deg is a pure headwind, 180 a pure tailwind.  The best case")
    print("  anywhere in this table is %.2f s." % best)
    print()

    steering(speed, args)
    sensitivity(speed, total, args)
    return 0


def seconds_saved(wind_at_cox, true_angle, boat_speed, total, race_time):
    """Time saved by the fairing at one wind, in seconds.

    ``true_angle`` is measured off the bow to the direction the wind comes
    **from**, so 0 is a headwind.  A wind arriving from the bow moves
    sternward, which is the negative x direction in the boat frame -- the
    first version dropped that negation and produced the diagnostic
    absurdity of a headwind saving less than a tailwind.
    """
    along = -wind_at_cox * np.cos(true_angle) - boat_speed
    across = wind_at_cox * np.sin(true_angle)
    apparent = float(np.hypot(along, across))
    # Angle of the apparent wind off the bow; 0 is dead ahead, where the
    # crew shields the coxswain best.
    apparent_angle = float(np.arctan2(abs(across), -along))

    exposure = float(shielding(apparent_angle))
    dynamic = 0.5 * AIR_DENSITY * COX_AREA * exposure * apparent ** 2
    # A fairing works by keeping flow attached.  The coxswain sits in the
    # separated, unsteady wake of eight rowers, where the flow it needs to
    # organise has already been destroyed -- so a shape that halves Cd in
    # a wind tunnel does much less here.  This is the largest soft spot in
    # the estimate and it is a factor, not a footnote.
    saving = dynamic * (CD_BARE - CD_FAIRED) * wake_effectiveness(
        apparent_angle)
    # Race time responds to a fractional resistance change through the
    # exponent measured for this hull in scripts/time_budget.py: at fixed
    # crew power, v ~ P^0.498, so dt/t = 0.498 dR/R.
    return float(race_time * SPEED_EXPONENT * saving / total)


def steering(boat_speed, args):
    """What the tail costs when the wind is on the beam."""
    print("the other side of a tail fairing: it is a weathervane")
    print("  %-12s %12s %14s" % ("beam wind", "side force", "yaw moment"))
    for wind in args.winds:
        if wind <= 0:
            continue
        at_cox = wind * log_profile_factor(COX_HEIGHT)
        side = 0.5 * AIR_DENSITY * FAIRING_SIDE_AREA * 1.0 * at_cox ** 2
        print("  %8.0f m/s %10.2f N %12.1f N m"
              % (wind, side, side * FAIRING_ARM))
    print("  %.2f m^2 of new side area %.0f m aft of the hull's yaw pivot."
          % (FAIRING_SIDE_AREA, FAIRING_ARM))
    print("  Small against a rudder that makes hundreds of newton-metres,")
    print("  but it acts all race and it pushes the bow INTO the wind,")
    print("  which is a standing rudder load, not a gust.")
    print()


def sensitivity(boat_speed, total, args):
    """How much of the answer is the assumptions?"""
    print("sensitivity, at 8 m/s of true headwind")
    print("  %-28s %10s" % ("if instead", "saving"))
    base = seconds_saved(8.0 * log_profile_factor(COX_HEIGHT), 0.0,
                         boat_speed, total, args.race_time)
    print("  %-28s %9.2fs" % ("(as assumed)", base))
    global COX_AREA, CD_FAIRED, SHIELDING_HEAD_ON
    global WAKE_EFFECTIVENESS_MIN
    for label, name, value in (
            ("coxswain is 0.30 m^2", "COX_AREA", 0.30),
            ("fairing reaches Cd 0.35", "CD_FAIRED", 0.35),
            ("no shielding at all", "SHIELDING_HEAD_ON", 1.0),
            ("shielding is 0.60", "SHIELDING_HEAD_ON", 0.60),
            ("fairing works fully in wake", "WAKE_EFFECTIVENESS_MIN", 1.0),
            ("fairing barely works in wake", "WAKE_EFFECTIVENESS_MIN", 0.2)):
        keep = globals()[name]
        globals()[name] = value
        print("  %-28s %9.2fs"
              % (label, seconds_saved(8.0 * log_profile_factor(COX_HEIGHT),
                                      0.0, boat_speed, total,
                                      args.race_time)))
        globals()[name] = keep
    print()
    print("  the shielding assumption dominates everything else.  If the")
    print("  coxswain is NOT in the crew's wake the case is real; if they")
    print("  are, it is marginal.  That is a measurement -- a tuft or a")
    print("  handheld anemometer in the stern on a windy outing -- and not")
    print("  something this model can settle.")


if __name__ == "__main__":
    raise SystemExit(main())
