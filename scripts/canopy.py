"""Validate the canopy roughness model, then point it at the Charles.

    python scripts/canopy.py
    python scripts/canopy.py --station 2400 --wind-from 250

The rule this project has followed for every other borrowed model applies
here too: reproduce the published cases first, on ground where the answer
is known, and only then run it on the river.  A roughness length is
particularly easy to get wrong by a factor of three and not notice,
because nothing about the number looks wrong.

Three checks, in order of how much they would hurt to fail
----------------------------------------------------------
**The shape of the curve.**  Raupach's ``z0/h`` rises with frontal area
index, peaks where his ``u*/U`` cap binds, and falls after as elements
begin to shelter one another.  The peak location is fixed by the
constants and can be predicted in closed form, so this check is the
algebra against itself; the peak *value* is checked against the observed
urban range instead, because that is the part that can be wrong.

**The Davenport classes.**  Wieringa's revision of the Davenport scale is
the field's calibration standard: open grass 0.03 m, suburb 0.5 m, city
centre 2 m.  Fed the morphology of each of those, the model has to land
in the right class.  This is the check that catches a wrong ``h``.

**Macdonald, on the dense branch.**  Raupach's falling branch is fitted
furthest from data, so it gets a second opinion.  Disagreement there is
expected and worth seeing rather than hiding.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.hydro.canopy import (C_R, C_S,  # noqa: E402
                                   DAVENPORT, USTAR_MAX,
                                   internal_boundary_layer,
                                   macdonald_roughness,
                                   open_water_equivalent,
                                   raupach_roughness, sheltered_speed)

#: Morphologies for the Davenport classes, from [GO99]_ and the urban
#: climatology literature: mean element height and frontal area index.
#: The expected roughness is Wieringa's, not this model's -- that is the
#: whole point of the comparison.
CASES = (
    #  name                    h (m)  lambda_f   expected z0 (m)  tolerance
    ("open grass, hedges",       1.0,   0.020,   DAVENPORT["open"],        3.0),
    ("roughly open, scattered",  4.0,   0.030,   DAVENPORT["roughly open"], 3.0),
    ("rough, low buildings",     6.0,   0.060,   DAVENPORT["rough"],       2.5),
    ("very rough, suburb",       8.0,   0.110,   DAVENPORT["very rough"],  2.0),
    ("closed, dense suburb",    10.0,   0.180,   DAVENPORT["closed"],      2.0),
    ("mature forest",           18.0,   0.150,   1.4,                      2.0),
    ("city centre",             25.0,   0.250,   DAVENPORT["chaotic"],     2.5),
)


def curve_shape():
    """Does z0/h peak where Raupach's own algebra puts it?

    The first version of this test asserted a peak near lambda_f = 0.2
    because that is where the *observational* reviews put it, failed, and
    looked like a coding error.  It is not: in Raupach's formulation the
    peak is exactly where the u*/U cap binds, which is at

        lambda_f = ((u*/U)_max^2 - C_S) / C_R = 0.29

    and that is a property of the model, not a bug in this file.  The
    test now checks the algebra against itself and the *value* against
    the observed range, which is what can actually be wrong.
    """
    lam = np.linspace(0.005, 0.8, 800)
    ratio = np.array([raupach_roughness(l, 1.0).z0 for l in lam])
    peak = int(np.argmax(ratio))
    predicted = (USTAR_MAX ** 2 - C_S) / C_R
    print("shape of the curve")
    print("  z0/h peaks at %.3f, at lambda_f = %.3f" % (ratio[peak],
                                                        lam[peak]))
    print("  Raupach's cap binds at lambda_f = %.3f, so that is where the"
          % predicted)
    print("  peak belongs; observed urban z0/h scatters 0.05-0.15 "
          "(Grimmond & Oke)")
    good = (abs(lam[peak] - predicted) < 0.02
            and 0.05 <= ratio[peak] <= 0.15)
    print("  %s" % ("PASS" if good else "FAIL"))
    d = [raupach_roughness(l, 1.0).d for l in (0.5, 1.0, 2.0)]
    print("  d/h at lambda_f 0.5 / 1 / 2: %.2f / %.2f / %.2f  "
          "(Raupach fig 1: 0.66 / 0.75 / 0.82)" % tuple(d))
    good = good and abs(d[2] - 0.82) < 0.03
    print()
    return good


def davenport():
    """Does each Davenport class come out in the right class?"""
    print("the Davenport classes (Wieringa 1992)")
    print("  %-26s %6s %8s %10s %10s %8s"
          % ("surface", "h", "lambda", "expected", "model", "ratio"))
    passes = 0
    for name, height, lam, expected, tolerance in CASES:
        z0 = raupach_roughness(lam, height).z0
        ratio = z0 / expected
        ok = 1.0 / tolerance <= ratio <= tolerance
        passes += ok
        print("  %-26s %6.1f %8.3f %10.3f %10.3f %7.2fx%s"
              % (name, height, lam, expected, z0, ratio,
                 "" if ok else "  <-- out"))
    print("  %d of %d inside their tolerance" % (passes, len(CASES)))
    print()
    return passes == len(CASES)


def against_macdonald():
    """Second opinion where Raupach is least constrained."""
    print("Raupach vs Macdonald, where it matters (h = 10 m)")
    print("  %-10s %10s %12s %12s %8s"
          % ("lambda_f", "lambda_p", "Raupach z0", "Macdonald z0", "ratio"))
    for lam_f in (0.05, 0.10, 0.20, 0.30, 0.45):
        lam_p = lam_f                       # cubic elements: the two match
        a = raupach_roughness(lam_f, 10.0).z0
        b = macdonald_roughness(lam_p, lam_f, 10.0).z0
        print("  %-10.2f %10.2f %12.3f %12.3f %8.2f"
              % (lam_f, lam_p, a, b, a / b if b else float("nan")))
    print("  the two are expected to part company above lambda_f ~ 0.2,")
    print("  which is exactly where Raupach's fit runs out of data.")
    print()


def boundary_layer():
    """How deep is the adjusted layer where a boat actually is?"""
    print("internal boundary layer over the water, bank z0 = 0.5 m")
    bank = raupach_roughness(0.11, 8.0)
    print("  (that bank: z0 %.2f m, d %.1f m from h = 8 m, lambda_f = 0.11)"
          % (bank.z0, bank.d))
    control = float(open_water_equivalent(1.5, 6.0, bank))
    print("  the same weather with no bank at all gives %.2f m/s at 1.5 m"
          % control)
    print("  %8s %12s %12s %12s"
          % ("fetch m", "IBL depth m", "at 1.5 m", "% of open"))
    for fetch in (5.0, 10.0, 30.0, 60.0, 100.0, 150.0):
        delta = float(internal_boundary_layer(fetch, bank.z0, 2.0e-4))
        got = float(sheltered_speed(1.5, fetch, 6.0, bank))
        print("  %8.0f %12.1f %12.2f %11.0f%%"
              % (fetch, delta, got, 100 * got / control))
    print("  shelter is a short-fetch effect and it runs out inside a")
    print("  hundred metres.  A rower's chest is at 1.5 m and a forecast")
    print("  anemometer is at 10 m over a different surface entirely, which")
    print("  is why the forecast and the felt wind disagree on this reach.")
    print()


def charles(args):
    """Frontal area index along the reach, from the OSM footprints."""
    from coxswain.river import charles as C
    from coxswain.river.structures import charles_structures

    structures = charles_structures()
    raster = C.charles_channel()
    _, _, race_line, _ = C.hocr_course(raster)
    course = C.charles_course(centreline=race_line, month=10)

    bearing = np.radians(90.0 - (args.wind_from + 180.0))   # met -> maths
    stations = (np.array([args.station], dtype=float) if args.station
                else np.linspace(200.0, course.length - 200.0, 12))
    print("the Charles, wind from %03d degrees, %.0f m upwind sector"
          % (args.wind_from, args.radius))
    print("  %8s %8s %8s %9s %9s %9s"
          % ("station", "n bldg", "h mean", "lambda_f", "z0 m", "at 1.5 m"))
    for station in stations:
        point = course.offset_position(np.array([station]),
                                       np.array([0.0]))[0]
        index = structures.near(point[0], point[1], args.radius)
        if not len(index):
            print("  %8.0f %8d %8s %9s %9s %9s"
                  % (station, 0, "-", "-", "open", "-"))
            continue
        # Frontal area presented to this wind, per unit ground area of
        # the sector the wind crosses.
        frontal = sum(structures.frontal_width(i, bearing)
                      * structures.heights[i] for i in index)
        ground = np.pi * args.radius ** 2 * 0.5      # upwind half only
        lam = float(frontal / ground)
        height = float(np.average(structures.heights[index]))
        rough = raupach_roughness(lam, height)
        speed = float(sheltered_speed(1.5, args.fetch, args.reference, rough))
        print("  %8.0f %8d %8.1f %9.3f %9.3f %9.2f"
              % (station, len(index), height, lam, rough.z0, speed))
    print()
    print("  lambda_f above ~0.2 is outside Raupach's fitted range; those")
    print("  rows are indicative of 'very rough' and not much more.")
    print("  Every height here is 75% inferred from building type, so the")
    print("  z0 column inherits that -- see coxswain/river/structures.py.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--station", type=float, default=None)
    parser.add_argument("--wind-from", type=float, default=250.0,
                        help="meteorological bearing the wind comes FROM")
    parser.add_argument("--radius", type=float, default=250.0)
    parser.add_argument("--fetch", type=float, default=60.0,
                        help="metres of open water upwind of the boat")
    parser.add_argument("--reference", type=float, default=6.0,
                        help="forecast wind at 10 m, m/s")
    parser.add_argument("--skip-charles", action="store_true")
    args = parser.parse_args(argv)

    ok = curve_shape()
    ok = davenport() and ok
    against_macdonald()
    boundary_layer()
    if not ok:
        print("validation failed -- not running this on the river")
        return 1
    if not args.skip_charles:
        charles(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
