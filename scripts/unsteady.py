r"""What the boat's own surge oscillation costs, and where added mass enters.

    python scripts/unsteady.py
    python scripts/unsteady.py --rate 24 28 32

A rowing shell does not travel at a steady speed.  It surges within every
stroke -- the crew's mass slides one way, the blades push the other -- and
the resistance it meets is not the resistance of its mean speed.  Day et
al. towed a single scull with imposed velocity oscillations to measure
exactly this [D11]_, and it is the one unsteady effect in rowing big
enough to matter and small enough to be worth a closed-form answer.

Two separate things get called "unsteady", and only one of them costs
------------------------------------------------------------------------
**Added mass does not.**  The inertial force ``-m_a dv/dt`` is in
quadrature with the velocity, so over a closed cycle it does exactly zero
net work: the water handed back at the end of the drive whatever it took
at the start.  What added mass does is set *how large the fluctuation is*
for a given force variation -- it is the denominator, not the cost.

**The nonlinearity of resistance does.**  ``R ~ v^n``, and the mean of a
power is not the power of the mean.  For a fluctuation of relative
amplitude ``eps``:

.. math::

    \frac{\langle v^{n+1}\rangle}{\langle v\rangle^{n+1}}
        \approx 1 + \tfrac{1}{2} n (n+1) \varepsilon^2

so the penalty is second order in the swing and cannot be signed away.
This is why a crew that runs the boat smoothly is faster at the same
average speed, and it is measured here on the simulator's own velocity
trace rather than assumed sinusoidal.

Where the panel method comes in
-------------------------------
Added mass is a potential-flow quantity: solve ``d(phi)/dn = n_i`` on the
hull and integrate ``-rho phi n_j`` over the surface.  The panel solver
does that directly and can be checked against a circle, for which the
answer is exactly ``rho pi a^2``.  What it gives here is the added mass
of the **waterplane slice**, which is the horizontal cut -- useful as a
validated number in its own right, and emphatically not the same thing as
the boat's surge added mass, which is a three-dimensional quantity that
strip theory in :mod:`coxswain.hydro.addedmass` handles instead.

References
----------
.. [D11] Day, A. H., Campbell, I., Clelland, D., Cichowicz, J. and
   Doctors, L. J. (2011) *An experimental study of unsteady
   hydrodynamics of a single scull*, Proc. IMechE Part M 225(3),
   282-294.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.panels import (SourcePanelBody,  # noqa: E402
                                   circle_nodes,
                                   waterline_from_offsets)
from coxswain.hydro.resistance import hull_resistance   # noqa: E402
from coxswain.sim.control import Coxswain               # noqa: E402
from coxswain.sim.simulator import RowingSimulator      # noqa: E402

RACE_LENGTH = 4822.0
SPEED_EXPONENT = 0.498
MASTERS_POWER = 0.658


def trace(rate, speed, duration=24.0, dt=0.005):
    """The boat's own surge history over a few settled strokes."""
    boat = catalog.eight(rate=rate, rower_mass=72.0, rower_stature=1.72)
    boat.power_scales = np.full(boat.n_seats, MASTERS_POWER)
    cox = Coxswain(rudder_override=lambda t, s: 0.0, pressure_split=0.0)
    sim = RowingSimulator(boat, coxswain=cox)
    result = sim.run(duration=duration, dt=dt, surge_speed=speed)
    t = np.asarray(result.time)
    v = np.hypot(*np.asarray(result.velocity)[:2])
    period = boat.timing.period
    cycles = int((0.5 * t[-1]) // period)
    keep = t >= t[-1] - cycles * period
    return boat, t[keep], v[keep]


def resistance_curve(boat, speeds):
    """Hull resistance at a set of speeds, from the real hull."""
    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=9.80665, water_level=0.0)
    out = []
    for speed in np.atleast_1d(speeds):
        force, _ = hull_resistance(np.array([float(speed), 0.0, 0.0]),
                                   submerged, mean_wetted_length=boat.length,
                                   water=boat.water,
                                   coefficients=boat.resistance)
        out.append(abs(float(force[0])))
    return np.array(out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--rate", type=float, nargs="+", default=[28.0])
    args = parser.parse_args(argv)

    speed = RACE_LENGTH / args.race_time

    print("added mass from the panels, checked where the answer is known")
    exact = 1000.0 * np.pi
    for count in (60, 160, 400):
        matrix = SourcePanelBody(circle_nodes(1.0, count)).solve(1.0)
        matrix = matrix.added_mass()
        print("  circle, %3d panels: m_xx %8.1f kg/m against %.1f exact "
              "(%.2f%%)" % (count, matrix[0, 0], exact,
                            100 * abs(matrix[0, 0] / exact - 1.0)))
    print()

    boat = catalog.eight(rate=28.0, rower_mass=72.0, rower_stature=1.72)
    body = SourcePanelBody(
        waterline_from_offsets(boat.offsets, panels=200)).solve(speed)
    plane = body.added_mass()
    print("the waterplane slice of this hull, per metre of depth")
    print("  surge  %9.1f kg/m   sway %10.1f kg/m" % (plane[0, 0],
                                                      plane[1, 1]))
    print("  ratio  %9.4f -- a 17.3 m by 0.57 m shape is 30:1 slender, so"
          % (plane[0, 0] / plane[1, 1]))
    print("  it entrains almost nothing moving lengthwise and a great deal")
    print("  moving sideways, which is the whole reason a shell tracks.")
    print("  This is the HORIZONTAL cut and is not the boat's surge added")
    print("  mass; that is a 3-D quantity and strip theory owns it.")
    print()

    print("what the surge oscillation actually costs")
    print("  %-8s %9s %9s %9s %11s %10s"
          % ("rate", "mean m/s", "swing %", "n", "extra power", "over a race"))
    for rate in args.rate:
        boat, times, velocity = trace(rate, speed)
        mean = float(velocity.mean())
        swing = float(velocity.std() / mean)

        # Local resistance exponent, measured rather than assumed.
        probe = mean * np.array([0.97, 1.03])
        low, high = resistance_curve(boat, probe)
        exponent = float(np.log(high / low) / np.log(probe[1] / probe[0]))

        # Power to hold this trace, against power to hold its mean.
        instantaneous = resistance_curve(boat, velocity) * velocity
        penalty = float(instantaneous.mean()
                        / (resistance_curve(boat, mean)[0] * mean) - 1.0)
        print("  %-8.0f %9.4f %9.2f %9.2f %10.2f%% %9.1f s"
              % (rate, mean, 100 * swing, exponent, 100 * penalty,
                 SPEED_EXPONENT * penalty * args.race_time))
    print()
    print("  MEASURED SWING IS NOT VALIDATED, and this number is quadratic")
    print("  in it.  Kleshnev reports about 20%% peak-to-peak boat velocity")
    print("  variation in an eight, which is roughly 7%% as a standard")
    print("  deviation; the simulator produces about twice that.  Scaling")
    print("  the penalty by the square of the ratio:")
    published = 0.07
    for rate in args.rate:
        boat, times, velocity = trace(rate, speed)
        mean = float(velocity.mean())
        swing = float(velocity.std() / mean)
        instantaneous = resistance_curve(boat, velocity) * velocity
        penalty = float(instantaneous.mean()
                        / (resistance_curve(boat, mean)[0] * mean) - 1.0)
        scaled = penalty * (published / swing) ** 2
        print("    rate %2.0f: %.1f s as simulated, %.1f s at a %.0f%% swing"
              % (rate, SPEED_EXPONENT * penalty * args.race_time,
                 SPEED_EXPONENT * scaled * args.race_time, 100 * published))
    print("  Take the second column until the simulator's surge trace has")
    print("  been checked against a real speed-coach log -- which is one")
    print("  outing with a logger, and would settle it.")
    print()
    print("  'n' is the local exponent of R(v), measured on this hull at")
    print("  this speed rather than assumed to be two.  The quadratic")
    print("  estimate 0.5 n (n+1) eps^2 is the closed form; the column")
    print("  above is the integral over the real trace, which is the same")
    print("  thing without the small-swing assumption.")
    print()
    print("  Added mass contributes NOTHING to this number.  Its force is")
    print("  in quadrature with velocity, so around a closed cycle it does")
    print("  zero net work -- it sets the size of the swing, and the swing")
    print("  is what costs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
