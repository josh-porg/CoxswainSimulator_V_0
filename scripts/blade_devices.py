r"""Vortex generators on a rowing blade, with the lift term where it belongs.

    python scripts/blade_devices.py

A first version of this script treated the blade as a drag plate through
the whole stroke, took the absolute value of the flow's chordwise
component, and so hid the two things that actually decide the question:
**the blade generates a large share of its force by lift, and the flow
reverses across the face at the perpendicular.**  Both are corrected here.

What the blade is really doing
------------------------------
The flow the blade meets has two components: one normal to the face --
the model's slip -- and one along the chord, ``v_b sin(theta)``.  That
second one carries the sign of the oar angle, so it points one way from
the catch to the perpendicular and **the other way from the perpendicular
to the finish**.  The leading edge and the trailing edge swap places
halfway through the drive.

Between those, the angle of attack sweeps from about 30 degrees at the
catch, through 70-plus at the perpendicular, and back down at the finish.
Low angles are a lifting regime; the perpendicular is a drag regime.  The
blade is both devices at different moments, which is what makes it
interesting and what a ``C2 * slip^2`` model cannot express.

The measurement that settles the device question
------------------------------------------------
Caplan and Gardner [CG07]_ towed scaled Big Blade and Macon blades in a
water flume across the whole sweep range and fitted

.. math::

    C_L = A_L \sin(2\alpha), \qquad C_D = A_D \sin^2(\alpha)

and reported **no significant stall at any angle of attack**.

That sentence is the answer.  Those are the coefficients of a *fully
separated* plate -- the flat-plate crossflow form -- and a surface that
never stalls is a surface whose flow was never attached.  A vortex
generator exists to postpone stall by re-energising an attached boundary
layer.  On a blade there is no attached boundary layer to re-energise and
no stall to postpone, at **any** angle of attack, which is a stronger
statement than anything about Reynolds number.

And the reversal makes the placement self-defeating
---------------------------------------------------
Even granting a mechanism: a generator a quarter chord back from one edge
is three quarters of a chord back from the other, and the two edges trade
roles at the perpendicular.  Whatever it did for the first half of the
drive it would be badly placed to do for the second.

References
----------
.. [CG07] Caplan, N. and Gardner, T. N. (2007) *A fluid dynamic
   investigation of the Big Blade and Macon oar blade designs in rowing
   propulsion*, Journal of Sports Sciences 25(6), 643-650.  Quasi-static
   water-flume tests over the full sweep range; the sine fits and the
   no-stall finding are theirs.
.. [CG10] Coppel, A., Gardner, T., Caplan, N. and Hargreaves, D. (2010)
   *Simulating the fluid dynamic behaviour of oar blades in competition
   rowing*, Proc. IMechE Part P.
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
WATER_DENSITY = 1000.0
WATER_VISCOSITY = 1.0e-6

#: Caplan and Gardner's amplitudes for a Big Blade, representative.
#: ``C_L = A_L sin(2a)``, ``C_D = A_D sin^2(a)``.
A_LIFT = 1.00
A_DRAG = 1.20

#: Big Blade: about 0.55 m along the shaft (the chord the flow crosses)
#: by 0.25 m of vertical extent (the span).
BLADE_CHORD = 0.55
BLADE_AREA = 0.1100


def stroke(boat, boat_speed, samples=600):
    """Signed flow geometry and the lift/drag split through the drive."""
    oar = boat.rig.seats[0].oarlocks[0].oar
    outboard = oar.length - oar.inboard
    blade = BladeModel.sweep(outboard=outboard)
    timing = boat.timing
    times = np.linspace(0.0, timing.drive_duration, samples)
    theta = np.asarray(boat.oar_sweep(times, timing), dtype=float)
    rate = np.asarray(boat.oar_sweep.rate(times, timing), dtype=float)

    # Flow relative to the blade, on the blade's own axes.  ``normal`` is
    # the model's slip; ``chordwise`` carries the sign of the oar angle
    # and is the component that reverses at the perpendicular.
    normal = -blade.slip_velocity(theta, rate, boat_speed)
    chordwise = -boat_speed * np.sin(theta)
    speed = np.hypot(normal, chordwise)

    # Angle of attack from the chord line, signed by which edge leads.
    alpha = np.arctan2(normal, chordwise)
    lift_coefficient = A_LIFT * np.sin(2.0 * alpha)
    drag_coefficient = A_DRAG * np.sin(alpha) ** 2
    dynamic = 0.5 * WATER_DENSITY * BLADE_AREA * speed ** 2

    # Drag acts along the flow; lift perpendicular to it.  Resolve both
    # onto the boat's x axis to get what each actually contributes.
    flow_x = -(normal * np.cos(theta) + chordwise * np.sin(theta))
    flow_y = -(-normal * np.sin(theta) + chordwise * np.cos(theta))
    magnitude = np.maximum(np.hypot(flow_x, flow_y), 1e-9)
    drag_x = dynamic * drag_coefficient * flow_x / magnitude
    lift_x = dynamic * lift_coefficient * (-flow_y) / magnitude

    reynolds = speed * BLADE_CHORD / WATER_VISCOSITY
    return dict(times=times, theta=theta, alpha=alpha, chordwise=chordwise,
                normal=normal, speed=speed, reynolds=reynolds,
                lift_x=lift_x, drag_x=drag_x,
                lift_coefficient=lift_coefficient,
                drag_coefficient=drag_coefficient)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    args = parser.parse_args(argv)

    boat = catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72)
    speed = RACE_LENGTH / args.race_time
    s = stroke(boat, speed)

    print("the flow reverses across the face at the perpendicular")
    print("  %-14s %7s %8s %9s %8s %8s %8s"
          % ("phase", "oar", "alpha", "chordwise", "C_L", "C_D", "Re"))
    for label, fraction in (("catch", 0.03), ("early", 0.22),
                            ("perpendicular", 0.50), ("late", 0.74),
                            ("finish", 0.97)):
        i = int(fraction * (len(s["times"]) - 1))
        print("  %-14s %6.0f %7.0f %8.2f %9.2f %8.2f %8.1e"
              % (label, np.degrees(s["theta"][i]),
                 np.degrees(s["alpha"][i]), s["chordwise"][i],
                 s["lift_coefficient"][i], s["drag_coefficient"][i],
                 s["reynolds"][i]))
    crossings = int(np.sum(np.diff(np.sign(s["chordwise"])) != 0))
    print("  'chordwise' is the flow along the blade's chord, in m/s.  It")
    print("  changes sign %d time(s) in the drive: the leading edge and the"
          % crossings)
    print("  trailing edge trade places at the perpendicular.")
    print()

    integral = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    lift_impulse = float(integral(np.abs(s["lift_x"]), s["times"]))
    drag_impulse = float(integral(np.abs(s["drag_x"]), s["times"]))
    share = 100.0 * lift_impulse / (lift_impulse + drag_impulse)
    print("how the propulsive impulse splits")
    print("  lift  %6.1f N s   %4.1f%%" % (lift_impulse, share))
    print("  drag  %6.1f N s   %4.1f%%" % (drag_impulse, 100 - share))
    print("  So it is genuinely both, and the earlier claim that this is a")
    print("  drag device with a lifting fringe was wrong.")
    print()

    print("what that does NOT rescue")
    print("  Caplan and Gardner measured C_L = %.2f sin(2a) and"
          % A_LIFT)
    print("  C_D = %.2f sin^2(a) with NO SIGNIFICANT STALL at any angle."
          % A_DRAG)
    print("  Those are the coefficients of a fully separated plate.  A")
    print("  surface that never stalls is one whose flow was never")
    print("  attached, and a vortex generator's whole job is to postpone")
    print("  stall on an attached boundary layer.")
    print()
    peak = int(np.argmax(np.abs(s["lift_coefficient"])))
    # Fold onto 0-90: an angle of attack of 135 degrees is 45 degrees
    # measured from the other edge, which is the same flow.
    folded = np.degrees(np.abs(s["alpha"][peak])) % 180.0
    folded = min(folded, 180.0 - folded)
    print("  peak |C_L| in the drive: %.2f, at alpha = %.0f degrees"
          % (np.abs(s["lift_coefficient"]).max(), folded))
    print("  A thin attached foil would peak near alpha = 12-15 degrees and")
    print("  then drop off a cliff.  This peaks at 45 and comes down as a")
    print("  sine -- that is separated-flow lift, from the pressure")
    print("  difference across a wake, not from circulation a generator")
    print("  could thicken.")
    print()
    print("  And the placement defeats itself independently: quarter chord")
    print("  from one edge is three quarters from the other, and the edges")
    print("  swap at the perpendicular.")
    print()

    immersion(args)
    return 0


def immersion(args):
    """What does move blade force, on the same basis as everything else."""
    blade = BladeModel.sweep()
    optimum = blade.immersion_factor(0.125)
    print("what does move the number")
    print("  %-30s %10s %11s" % ("blade cover", "force kept", "over a race"))
    for label, cover in (("light, 90 mm -- a tired crew", 0.090),
                         ("good, 110 mm", 0.110),
                         ("Kleshnev optimum, 125 mm", 0.125)):
        factor = blade.immersion_factor(cover)
        print("  %-30s %9.3f %10.1f s"
              % (label, factor,
                 args.race_time * SPEED_EXPONENT * (1.0 - factor / optimum)))
    print()
    print("  Ventilation -- air drawn down the back of a blade that is not")
    print("  buried -- is the real blade-surface problem, and no device")
    print("  fixes it.  Depth does.")
    print()
    print("  The tip fences remain the exception: a blade this low in")
    print("  aspect ratio spills a lot round its ends, and an end plate")
    print("  raises the pressure difference rather than lowering it.")
    print("  That mechanism does not need attached flow, which is exactly")
    print("  why it is the one that survives.")


if __name__ == "__main__":
    raise SystemExit(main())
