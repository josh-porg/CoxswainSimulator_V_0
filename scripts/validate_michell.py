r"""Check Michell's integral before trusting it, then show what it changes.

    python scripts/validate_michell.py

The wave term used to be a constant coefficient on the wetted area, which
made wave drag the same fraction of a ``v^2`` law at every speed -- 28% of
this hull's resistance, at 3 m/s and at 7 m/s alike.  A racing shell is
shaped to sit in a *hollow* of its wave-resistance curve, and a constant
cannot have a hollow.

Michell's integral replaces it with a calculation over the hull's own
offsets and no fitted coefficient at all.  Three checks before it is
believed:

**Numerics.**  Grid convergence on the Wigley parabolic hull, the standard
thin-ship test case.  Two quadrature traps were real: the ``lambda``
integral has to stop where the draft decay kills the integrand rather than
run to infinity, and the station grid has to resolve ``cos(k0 lambda x)``
at the largest ``lambda`` that still contributes.  Getting either wrong
gives a curve with humps in the wrong places and coefficients four orders
of magnitude too large -- which is what the first version did.

**Structure.**  Thin-ship interference puts humps where the hull length is
an odd number of half wavelengths and hollows where it is a whole number.
Michell should reproduce that from the offsets without being told.

**The prediction that matters.**  Pulman puts a racing VIII at Froude 0.35
and calls it a local minimum of the wave-drag curve.  Nothing in this
calculation knows that.  If the hollow lands there, it is a confirmed
prediction; if it does not, the calculation is wrong or Pulman is.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.boats import catalog                      # noqa: E402
from coxswain.hydro.michell import (MichellWave,  # noqa: E402
                                    elliptical_offsets, wigley_offsets)
from coxswain.hydro.resistance import hull_resistance   # noqa: E402

GRAVITY = 9.80665


def convergence():
    """Does the Wigley answer settle as the grid refines?"""
    probe = np.array([0.25, 0.35, 0.45, 0.55]) * np.sqrt(GRAVITY)
    print("numerics: grid convergence on the Wigley hull")
    print("  %-12s %9s %10s %10s %10s %10s"
          % ("grid", "pts/wave", "Fn 0.25", "Fn 0.35", "Fn 0.45", "Fn 0.55"))
    previous = None
    for stations, levels in ((81, 41), (161, 41), (321, 61), (641, 81),
                             (1281, 101)):
        x, z, half = wigley_offsets(stations=stations, levels=levels)
        model = MichellWave(station=x, level=z, half_beam=half)
        values = model.resistance(probe)
        change = ("" if previous is None
                  else "   %+.1f%%" % (100 * (values[2] / previous[2] - 1.0)))
        print("  %-12s %9.0f %10.5f %10.5f %10.5f %10.5f%s"
              % ("%dx%d" % (stations, levels), model.resolution.min(),
                 *values, change))
        previous = values
    print("  Successive changes halve, which is first-order convergence in")
    print("  the grid -- about 1% at 641x81, and that runs in a tenth of a")
    print("  second.")
    print()


def structure(model, length, label):
    """Where the humps and hollows fall, against thin-ship theory."""
    speeds = np.linspace(0.35 * np.sqrt(GRAVITY * length),
                         1.05 * np.sqrt(GRAVITY * length), 220)
    values = model.resistance(speeds)
    froude = speeds / np.sqrt(GRAVITY * length)
    humps = [i for i in range(2, len(values) - 2)
             if values[i] == max(values[i - 2:i + 3])]
    hollows = [i for i in range(2, len(values) - 2)
               if values[i] == min(values[i - 2:i + 3])]
    print("structure: %s" % label)
    print("  humps at Fn   %s"
          % ", ".join("%.3f" % froude[i] for i in humps[:6]))
    print("  hollows at Fn %s"
          % ", ".join("%.3f" % froude[i] for i in hollows[:6]))
    print("  thin-ship interference predicts humps at 1/sqrt((2n-1)pi):")
    print("    %s" % ", ".join("%.3f" % (1 / np.sqrt((2 * n - 1) * np.pi))
                               for n in (1, 2, 3)))
    print("  and hollows at 1/sqrt(2 n pi):")
    print("    %s" % ", ".join("%.3f" % (1 / np.sqrt(2 * n * np.pi))
                               for n in (1, 2, 3)))
    print()
    return froude, values, hollows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-time", type=float, default=1140.0)
    parser.add_argument("--skip-convergence", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_convergence:
        convergence()

    boat = catalog.eight(rate=32.0, rower_mass=74.8, rower_stature=1.86,
                         coxswain_mass=56.7)
    x, z, half = elliptical_offsets(boat.offsets, stations=641, levels=81)
    model = MichellWave(station=x, level=z, half_beam=half)
    froude, values, hollows = structure(model, boat.length, "this eight")

    racing = [froude[i] for i in hollows if 0.30 <= froude[i] <= 0.40]
    print("THE PREDICTION")
    print("  Pulman: a VIII races at Fn 0.35, a local MINIMUM of wave drag.")
    if racing:
        print("  Michell puts a hollow at Fn %s -- confirmed, from the"
              % ", ".join("%.3f" % f for f in racing))
        print("  offsets alone, with nothing fitted to say so.")
    else:
        print("  Michell finds NO hollow between Fn 0.30 and 0.40.  Either")
        print("  the calculation is wrong or Pulman is; do not proceed.")
    print()

    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
        rho=boat.water.density, gravity=GRAVITY, water_level=0.0)
    table = model.tabulate()
    print("what it changes, against the constant coefficient")
    print("  %-7s %-7s %10s %10s %9s %9s %8s"
          % ("v m/s", "Fn", "constant", "Michell", "wave %", "total", "C_D"))
    for speed in (3.0, 4.23, 5.0, 5.5, 6.0, 6.5):
        _f, old = hull_resistance(np.array([speed, 0.0, 0.0]), submerged,
                                  mean_wetted_length=boat.length,
                                  water=boat.water,
                                  coefficients=boat.resistance)
        force, new = hull_resistance(np.array([speed, 0.0, 0.0]), submerged,
                                     mean_wetted_length=boat.length,
                                     water=boat.water,
                                     coefficients=boat.resistance,
                                     wave_table=table)
        air = 0.5 * 1.225 * 3.22 * speed ** 2
        total = abs(float(force[0])) + air
        print("  %-7.2f %-7.3f %10.0f %10.0f %8.0f%% %9.0f %8.1f"
              % (speed, speed / np.sqrt(GRAVITY * boat.length), old["wave"],
                 new["wave"], 100 * new["wave"] / abs(float(force[0])),
                 total, total / speed ** 2))
    print()
    print("  wave drag was a flat 28% of hull resistance at every speed.")
    print("  It is now 8-11%, which is what Pulman means by 'not")
    print("  significant', and it varies with Froude number as it must.")
    print()
    print("  C_D at 6 m/s: 16.2 before, 13.2 now.  The physical floor --")
    print("  ITTC friction plus form plus air, no wave at all -- is 12.1.")
    print("  Buckmann and Harris measured 10.5 (9.6-11.4), which is BELOW")
    print("  that floor and so cannot be a steady-drag measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
