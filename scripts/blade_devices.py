r"""Vortex generators on a rowing blade: right physics, wrong sign.

    python scripts/blade_devices.py

The trip-tape question died on Reynolds number.  **This one does not** --
and that is the interesting part.  Water's kinematic viscosity is fifteen
times lower than air's, so a blade of the same size moving at a similar
speed sits two orders of magnitude higher up the Reynolds scale than an
oar shaft in air.  The blade is squarely in the regime where vortex
generators are used on wind turbines and aircraft.

So the idea cannot be dismissed the way the tape could.  It fails for two
other reasons, both of which this script measures rather than asserts.

One: the separation is fixed by the geometry, not the boundary layer
--------------------------------------------------------------------
A vortex generator re-energises a boundary layer so that it stays
attached further round a curved surface.  That requires the separation
point to be *free* to move.  On a thin, sharp-edged plate at a large
angle to the flow, it is not: the flow leaves at the edge because there is
nowhere else for it to go, at any Reynolds number and with any amount of
mixing.  The script reports the flow angle to the blade face through the
drive so it is clear how much of the stroke is spent in that condition.

Two: for most of the drive, more attached flow means LESS force
---------------------------------------------------------------
This is the one that turns the idea around.  An aircraft wing wants lift
and hates pressure drag.  **A rowing blade is a drag device** -- at
mid-drive it is a plate held broadside to the water, and the propulsive
reaction *is* its pressure drag.  A device that narrows the wake and
delays separation reduces exactly the force the crew is trying to make.
Near the catch, where the blade slices at a shallow angle and behaves
more like a foil, the sign flips and the device would help.  So the
question is which phase carries the impulse, which is measurable.

Where the tip devices come in
-----------------------------
Calling them a glorified tip protector is a little unfair.  A rowing
blade is a very low aspect ratio surface, and low aspect ratio surfaces
lose a large share of their force to flow spilling round the ends.  A
fence there is an end plate: it raises the pressure difference across the
blade and therefore raises the force, in the direction the crew wants.
**The tip is the one place on a blade where a device has both the right
mechanism and the right sign.**
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.crew.oarlock import BladeModel            # noqa: E402

RACE_LENGTH = 4822.0
SPEED_EXPONENT = 0.498
WATER_VISCOSITY = 1.0e-6          # m^2/s, fifteen times below air's

#: Chord of the blade in the flow direction, m.  A big blade is about
#: 0.25 m across the face; this is the streamwise dimension that sets the
#: Reynolds number.
BLADE_CHORD = 0.25

#: Beyond this angle from the blade's face normal the flow is shallow
#: enough that the blade behaves like a foil rather than a plate, and a
#: separation-delaying device would work with the crew rather than
#: against them.
FOIL_ANGLE = np.radians(45.0)


def drive_profile(boat, boat_speed, samples=400):
    """Flow angle, Reynolds number and impulse through the drive."""
    blade = BladeModel.sweep(outboard=boat.rig.seats[0].oarlocks[0].oar.length
                             - boat.rig.seats[0].oarlocks[0].oar.inboard)
    timing = boat.timing
    times = np.linspace(0.0, timing.drive_duration, samples)
    angle = np.asarray(boat.oar_sweep(times, timing), dtype=float)
    rate = np.asarray(boat.oar_sweep.rate(times, timing), dtype=float)

    # Blade velocity through the water, resolved on the blade's own axes:
    # normal to the face (this is the model's slip) and along the shaft.
    normal = np.abs(blade.slip_velocity(angle, rate, boat_speed))
    along = np.abs(boat_speed * np.sin(angle))
    speed = np.hypot(normal, along)

    # Angle of the oncoming flow measured from the face normal.  Zero is
    # square-on and bluff; ninety is edgewise and foil-like.
    flow_angle = np.arctan2(along, np.maximum(normal, 1e-6))
    reynolds = speed * BLADE_CHORD / WATER_VISCOSITY
    force = np.abs(blade.normal_force(angle, rate, boat_speed))
    return times, angle, flow_angle, reynolds, force


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    args = parser.parse_args(argv)

    boat = catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72)
    speed = RACE_LENGTH / args.race_time
    times, angle, flow_angle, reynolds, force = drive_profile(boat, speed)

    print("the blade is NOT ruled out by Reynolds number, unlike the shaft")
    print("  water is %.0fx less viscous than air, so the same size and speed"
          % (1.5e-5 / WATER_VISCOSITY))
    print("  sit two decades higher up the scale.")
    print("  blade Reynolds number over the drive: %.1e to %.1e"
          % (reynolds.min(), reynolds.max()))
    print("  force-weighted mean: %.1e"
          % float(np.average(reynolds, weights=force)))
    print("  (an oar shaft in air managed 2e4.  This is the regime where")
    print("   vortex generators are actually used.)")
    print()

    print("but what is the flow doing?  angle from the blade's face normal")
    print("  %-16s %8s %10s %12s %10s"
          % ("phase", "oar deg", "flow deg", "Re", "force N"))
    for label, fraction in (("catch", 0.02), ("early drive", 0.20),
                            ("mid-drive", 0.50), ("late drive", 0.80),
                            ("finish", 0.98)):
        i = int(fraction * (len(times) - 1))
        print("  %-16s %8.0f %10.0f %12.1e %10.0f"
              % (label, np.degrees(angle[i]), np.degrees(flow_angle[i]),
                 reynolds[i], force[i]))
    print("  0 deg of flow angle is square-on to the face -- a bluff plate,")
    print("  separating at its own sharp edges.  90 would be edgewise.")
    print()

    impulse = np.trapezoid(force, times) if hasattr(np, "trapezoid") \
        else np.trapz(force, times)
    foil_like = force * (flow_angle > FOIL_ANGLE)
    foil_impulse = (np.trapezoid(foil_like, times) if hasattr(np, "trapezoid")
                    else np.trapz(foil_like, times))
    share = 100.0 * foil_impulse / impulse
    print("how much of the stroke could a separation-delaying device help?")
    print("  impulse generated at flow angles above %.0f deg (foil-like):"
          % np.degrees(FOIL_ANGLE))
    print("      %.0f%% of the drive's impulse" % share)
    print("  the other %.0f%% is made with the blade close to broadside,"
          % (100 - share))
    print("  where the propulsive force IS the pressure drag and delaying")
    print("  separation would reduce it.  A device at quarter chord would")
    print("  be working against the crew for that majority of the stroke.")
    print()
    print("  And at quarter chord specifically: that placement assumes the")
    print("  flow is still attached over the leading quarter.  On a thin")
    print("  spooned blade meeting the water at %.0f-%.0f degrees off its"
          % (np.degrees(flow_angle.min()), np.degrees(flow_angle.max())))
    print("  face normal, it separates at the edge before it ever reaches")
    print("  the generators.")
    print()

    immersion(boat, speed, args)
    return 0


def immersion(boat, boat_speed, args):
    """What actually moves blade force, on the same basis as everything else.

    Deliberately restricted to covers a racing crew actually rows at.  The
    first version of this table swept down to 25 mm and reported 335
    seconds, which is arithmetically what a 63% force loss does to a race
    and is not a number about rowing: no crew rows 4.8 km with the blades
    barely in.  A model will answer any question it is asked, including a
    silly one.
    """
    blade = BladeModel.sweep()
    optimum = blade.immersion_factor(0.125)
    print("what does move the number, measured with the same model")
    print("  %-30s %10s %11s" % ("blade cover", "force kept", "over a race"))
    for label, cover in (("light, 90 mm -- a tired crew", 0.090),
                         ("good, 110 mm", 0.110),
                         ("Kleshnev optimum, 125 mm", 0.125)):
        factor = blade.immersion_factor(cover)
        seconds = args.race_time * SPEED_EXPONENT * (1.0 - factor / optimum)
        print("  %-30s %9.3f %10.1f s" % (label, factor, seconds))
    print()
    print("  So 35 mm of cover -- one third of a blade width, a difference")
    print("  a coach can see from the launch -- is worth more than every")
    print("  aerodynamic device in this project put together.")
    print()
    print("  Two things this model will not tell you, so do not ask it:")
    print("  it has no penalty for burying the blade too deep (extraction")
    print("  and the drag of getting it in and out), so its advice runs")
    print("  monotonically deeper and should be ignored past the optimum;")
    print("  and it assumes the crew holds the same handle force whatever")
    print("  the blade is doing, which is why sweeping it to absurd covers")
    print("  produces absurd numbers.")
    print()
    print("  The tip fences are the exception worth keeping: a blade this")
    print("  low in aspect ratio spills a lot of flow round its ends, and")
    print("  an end plate raises the pressure difference rather than")
    print("  lowering it.  Right mechanism, right sign, small effect.")


if __name__ == "__main__":
    raise SystemExit(main())
