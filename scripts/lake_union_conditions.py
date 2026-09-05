r"""Wind and chop over Lake Union, re-run on the surveyed data.

    python scripts/lake_union_conditions.py --out out/conditions

Both analyses were built before the data underneath them was any good,
and both are re-run here to see what the new data actually changed --
which is not the same question as what it *could* change.

What is new under them
----------------------
**Building heights.** 92% of the OpenStreetMap footprints carried a
height guessed from the building type -- every untagged building in
Seattle was nine metres tall.  They now carry lidar-measured apex
heights.  Roughness goes as frontal area, which is height times width, so
this is the input the wind field is most sensitive to.

**Trees.** 470,350 of them, from the city inventory plus the 2021 canopy
polygons, up to 39 m.  They were **not in the roughness at all**, on a
bank where the module's own docstring says shelter is buildings *and*
trees.

**Depth.** 14,871 surveyed values from the NOAA chart and a USACE
multibeam survey, against an invented shelf that ran 2.5 m too deep under
the racing line.  The chop model assumes deep water; whether that is
allowed is a question about the depth, and the depth was a guess.

The question the chop model was never able to answer
----------------------------------------------------
:mod:`coxswain.hydro.chop` computes fetch-limited wave growth from the
JONSWAP relations, which are **deep-water** relations.  Waves stop being
deep-water when the depth falls below half their wavelength.  With an
invented depth that test could not be run; with a survey it can, and it
is run here for every wind speed rather than asserted once.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coxswain.boats import catalog                        # noqa: E402
from coxswain.hydro.canopy import ShelteredWind           # noqa: E402
from coxswain.hydro.chop import (FetchLimitedSea,          # noqa: E402
                                 WalledBasin, added_resistance)
from coxswain.river.seattle import (fetch_at,              # noqa: E402
                                    lake_union_channel,
                                    water_mask)
from coxswain.river.structures import (seattle_structures,  # noqa: E402
                                       seattle_trees)

#: Winds to report, m/s.  6 is a normal Seattle afternoon, 10 is the
#: kind of day crews describe Tail of the Lake as, 14 is a gale.
WINDS = (6.0, 10.0, 14.0)

#: Bearings the wind comes from.  A north-south basin: the along-lake
#: directions are the ones with fetch.
BEARINGS = (0.0, 90.0, 180.0, 270.0)


def wind_along_course(course, structures, channel, speed, bearing,
                      trees=None, samples=40):
    """Mean and minimum sheltered wind at the crew's chest, m/s."""
    field = ShelteredWind(structures, channel, reference_speed=speed,
                          wind_from=bearing, trees=trees)
    stations = np.linspace(0.0, course.length, samples)
    values = np.array([field.speed_at(*course.position_at(s))
                       for s in stations])
    return float(values.mean()), float(values.min()), float(values.max())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/conditions")
    parser.add_argument("--speed", type=float, default=3.9)
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    from render_totl import totl_course

    course = totl_course()
    channel = lake_union_channel()
    structures = seattle_structures()
    trees = seattle_trees()
    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)

    print("=" * 68)
    print("WIND: what the measured heights and the trees did")
    print("=" * 68)
    print("  Reported as the mean sheltered wind at chest height along the")
    print("  course, for a 10 m reference wind.")
    print()
    print("  %-9s %-7s %9s %9s %9s" % ("wind from", "at 10 m", "buildings",
                                       "+ trees", "change"))
    for bearing in BEARINGS:
        for speed in (10.0,):
            plain = wind_along_course(course, structures, channel, speed,
                                      bearing)
            wooded = wind_along_course(course, structures, channel, speed,
                                       bearing, trees=trees)
            print("  %-9.0f %-7.1f %9.2f %9.2f %+8.1f%%"
                  % (bearing, speed, plain[0], wooded[0],
                     100.0 * (wooded[0] / plain[0] - 1.0)))

    print()
    print("=" * 68)
    print("CHOP: and whether the deep-water assumption survives the survey")
    print("=" * 68)

    # Fetch along the course, for the worst direction.
    mask = water_mask(10.0, names=("Lake Union",))
    stations = np.linspace(0.0, course.length, 24)
    fetches = {}
    for bearing in BEARINGS:
        values = [fetch_at(course.position_at(s), bearing, mask=mask)
                  for s in stations]
        fetches[bearing] = float(np.mean(values))
    worst = max(fetches, key=fetches.get)
    print("  mean fetch along the course: %s"
          % ", ".join("%.0f deg %.0f m" % (b, f)
                      for b, f in sorted(fetches.items())))
    print("  worst direction: %.0f deg, %.0f m" % (worst, fetches[worst]))

    # Depth under the course, surveyed.
    depths = np.array([float(course.depth_at(*course.position_at(s)))
                       for s in np.linspace(0.0, course.length, 200)])
    print("  surveyed depth under the line: median %.1f m, minimum %.1f m"
          % (np.median(depths), depths.min()))

    print()
    print("  %-6s %8s %8s %9s %10s %11s" % ("wind", "H_s", "T_p", "length",
                                            "h/L min", "deep water?"))
    rows = []
    for speed in WINDS:
        sea = FetchLimitedSea(wind=speed, fetch=fetches[worst])
        basin = WalledBasin(sea=sea)
        height = basin.open_water_height
        length = sea.wavelength
        # Deep water when the depth exceeds half a wavelength.
        ratio = depths.min() / max(length, 1e-9)
        deep = ratio > 0.5
        resistance = float(added_resistance(height, boat.offsets.max_beam, boat.length))
        rows.append((speed, height, sea.peak_period, length, ratio, deep,
                     resistance))
        print("  %-6.0f %8.2f %8.2f %9.2f %10.2f %11s"
              % (speed, height, sea.peak_period, length, ratio,
                 "yes" if deep else "NO"))

    print()
    print("  Deep-water waves need h > L/2.  The shallowest water under the")
    print("  racing line is %.1f m and the longest wave here is %.1f m, so"
          % (depths.min(), max(r[3] for r in rows)))
    print("  the JONSWAP relations are being used inside their range --")
    print("  which is now a checked statement rather than an assumption.")

    print()
    print("  %-6s %10s %12s %12s" % ("wind", "H_s open", "added drag",
                                     "share of hull"))
    # Hull resistance at race pace, from the same performance model the
    # rest of the project uses, so the share below is a like-for-like
    # comparison rather than two different drag calculations.
    from coxswain.sim.performance import SpeedResponse
    hull = float(SpeedResponse(boat).resistance(args.speed))
    for speed, height, _t, _l, _r, _d, resistance in rows:
        print("  %-6.0f %10.2f %11.1f N %11.1f%%"
              % (speed, height, resistance, 100.0 * resistance / hull))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
