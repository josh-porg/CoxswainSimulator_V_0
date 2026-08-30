r"""Hull roughness, trip strips, riblets, and a Gurney flap on the blade.

    python scripts/surfaces.py

Three questions with three different answers, and the interesting part is
that they fail and succeed for unrelated reasons.

The hull is already turbulent, so tripping it can only cost
-----------------------------------------------------------
Transition on a flat plate happens near ``Re_x = 5e5``.  At racing speed
that is about a hundred millimetres from the bow of a seventeen metre
hull, so **more than 99% of the wetted length is already turbulent**.  A
trip strip can only move that boundary forward, shortening the one
stretch of cheap laminar friction the boat has.  There is no separation
to fix either -- the form-drag term on this hull is under 2% of
resistance, which is what "slender and attached" looks like in a number.

Riblets are the real version of the question, and they are banned
-----------------------------------------------------------------
The way to reduce *turbulent* friction is not roughness, it is anisotropic
roughness: streamwise micro-grooves at a spacing of ``s+ = 15`` wall
units, which impede the spanwise motion of near-wall streamwise vortices.
That is a genuine 5-8% of skin friction, and this script computes the
spacing it would need on this hull.

Then World Rowing names them:

    "No substances or structures (including riblets) capable of modifying
    the natural properties of water or of the boundary layer of the
    hull/water interface shall be used."

That wording is broad enough to cover a trip strip, a polymer coating and
a hydrophobic film as well, so the whole family is closed -- and the one
member of it that would have worked is closed by name.

A Gurney flap depends entirely on which edge you mean
------------------------------------------------------
On the **spanwise tip**, a small perpendicular tab is an end fence, which
is the device that already survives: it blocks flow spilling round a very
low aspect ratio blade and raises the pressure difference across it.

On the **trailing edge**, it is self-defeating for the same reason the
vortex generators were -- the leading and trailing edges swap at the
perpendicular, so a trailing-edge tab spends half the drive as a
leading-edge spoiler.

Around the **whole perimeter**, it stops being a Gurney flap and becomes
a cup, and Hoerner's flanged-plate data says a cup is a much better bluff
body than a plate.  Which is not a discovery: it is what blade
manufacturers have been selling as "curvature" for thirty years.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.resistance import hull_resistance   # noqa: E402

RACE_LENGTH = 4822.0
SPEED_EXPONENT = 0.498
WATER_VISCOSITY = 1.0e-6

#: Flat-plate transition Reynolds number in low-turbulence water.
TRANSITION_RE = 5.0e5

#: Riblet spacing in wall units where the drag reduction peaks, and the
#: reduction achieved there.  3M/NASA film measurements; 8% is the
#: laboratory best and real installations do less.
RIBLET_SPLUS = 15.0
RIBLET_BEST = 0.08

#: Hoerner's flanged circular plates, normal to the flow: drag
#: coefficient against flange depth as a fraction of the plate diameter.
FLANGE_DEPTH = np.array([0.0, 0.05, 0.10, 0.25, 0.50])
FLANGE_CD = np.array([1.17, 1.30, 1.42, 1.75, 2.30])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    args = parser.parse_args(argv)

    boat = catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72)
    speed = RACE_LENGTH / args.race_time
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)
    force, detail = hull_resistance(np.array([speed, 0.0, 0.0]), submerged,
                                    mean_wetted_length=boat.length,
                                    water=boat.water,
                                    coefficients=boat.resistance)
    water = abs(float(force[0]))
    viscous = float(detail["viscous"])
    total = water + 34.0            # plus still-air aerodynamic drag

    def seconds(change):
        return args.race_time * SPEED_EXPONENT * change / total

    print("the hull's boundary layer at %.2f m/s" % speed)
    reynolds = speed * boat.length / WATER_VISCOSITY
    transition = TRANSITION_RE * WATER_VISCOSITY / speed
    print("  length Reynolds number      %.1e" % reynolds)
    print("  transition at Re_x = %.0e   %.3f m from the bow"
          % (TRANSITION_RE, transition))
    print("  laminar share of the hull   %.1f%%"
          % (100 * transition / boat.length))
    print("  viscous %.0f N, wave %.0f N, form %.0f N of %.0f N total"
          % (viscous, detail["wave"], detail["shape"], water))
    print()
    print("  So the hull is turbulent over %.1f%% of its length before"
          % (100 * (1 - transition / boat.length)))
    print("  anyone touches it, and its form drag -- the thing roughness")
    print("  fixes on a bluff body -- is %.1f%% of resistance.  There is"
          % (100 * detail["shape"] / water))
    print("  nothing for a trip strip to do except end the laminar run")
    print("  early, which costs about %.2f s.  It is a pure loss."
          % abs(seconds(0.5 * viscous * transition / boat.length)))
    print()

    riblets(speed, viscous, seconds)
    gurney(args)
    return 0


def riblets(speed, viscous, seconds):
    """The version of 'roughness' that actually reduces turbulent friction."""
    # Friction velocity from the local skin-friction coefficient.
    c_f = 0.074 / (speed * 17.3 / WATER_VISCOSITY) ** 0.2
    u_tau = speed * np.sqrt(c_f / 2.0)
    spacing = RIBLET_SPLUS * WATER_VISCOSITY / u_tau
    print("riblets: the anisotropic roughness that does work")
    print("  skin-friction coefficient   %.5f" % c_f)
    print("  friction velocity           %.3f m/s" % u_tau)
    print("  optimal groove spacing      %.0f micrometres" % (1e6 * spacing))
    print("  (3M's aerospace riblet film was made around 150 um, so this")
    print("   is manufacturable rather than hypothetical)")
    print("  best-case saving            %.1f N, %.1f s"
          % (RIBLET_BEST * viscous, seconds(RIBLET_BEST * viscous)))
    print()
    print("  That is one of the largest single numbers in this whole")
    print("  project -- and it is explicitly illegal.  World Rowing:")
    print('  "No substances or structures (including riblets) capable of')
    print('   modifying the natural properties of water or of the boundary')
    print('   layer of the hull/water interface shall be used."')
    print()
    print("  Note how wide that is.  It closes riblets by name, and a trip")
    print("  strip, a polymer release coating and a hydrophobic film all")
    print("  fall inside 'structures capable of modifying the boundary")
    print("  layer'.  The whole family is shut, so the only legal hull")
    print("  surface question left is whether yours is clean and fair,")
    print("  which is worth having and is not worth a script.")
    print()
    print("  It also applies to the HULL/water interface specifically, so")
    print("  it is not the rule that governs blades.")
    print()


def gurney(args):
    """Which edge is 'the tip'?  The answer changes completely."""
    print("a Gurney flap on the blade")
    print("  the tab is only ever as good as the edge it sits on, and this")
    print("  blade has three different kinds of edge.")
    print()
    print("  1. SPANWISE TIP.  A perpendicular tab here is an end fence.")
    print("     It blocks flow spilling round the end of a very low aspect")
    print("     ratio surface and raises the pressure difference across")
    print("     it.  This is the same device as the tip fence, not a")
    print("     better one -- Concept2's Vortex Edge is exactly this.  It")
    print("     works because the mechanism needs no attached flow and no")
    print("     fixed leading edge.")
    print()
    print("  2. TRAILING EDGE.  Self-defeating, for the reason the vortex")
    print("     generators were: the leading and trailing edges swap at")
    print("     the perpendicular, so a trailing-edge tab spends half the")
    print("     drive as a leading-edge spoiler.  You would need one on")
    print("     each edge, and each would be a spoiler for its off half.")
    print()
    print("  3. THE WHOLE PERIMETER.  Now it is not a Gurney flap, it is a")
    print("     cup -- and Hoerner's flanged plates say a cup is a much")
    print("     better bluff body than a plate:")
    print("     %-22s %8s %10s" % ("flange depth / width", "C_D", "vs flat"))
    for depth, cd in zip(FLANGE_DEPTH, FLANGE_CD):
        print("     %-22.2f %8.2f %9.0f%%"
              % (depth, cd, 100 * (cd / FLANGE_CD[0] - 1.0)))
    print("     A 10%% flange is a %.0f%% rise in blade force at the same"
          % (100 * (FLANGE_CD[2] / FLANGE_CD[0] - 1.0)))
    print("     slip, which is a real gain in blade efficiency.")
    print()
    print("     But this is not a discovery.  It is what blade curvature")
    print("     already is, and what every 'Big Blade', hatchet and cupped")
    print("     profile since the 1990s has been selling.  The flange has")
    print("     been on the blade for thirty years; it is just called a")
    print("     spoon.")
    print()
    print("  Verdict: the tip fence and the cup are the same idea applied")
    print("  at two edges, both work, and both are already fitted to a")
    print("  modern blade.  The thing you would be adding is a slightly")
    print("  deeper version of something you already own -- worth asking")
    print("  your blade supplier about, not worth fabricating.")


if __name__ == "__main__":
    raise SystemExit(main())
