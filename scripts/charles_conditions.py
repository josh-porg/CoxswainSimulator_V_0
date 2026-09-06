r"""Wind and chop over the Charles, on the data the reach now has.

    python scripts/charles_conditions.py --out out/charles_conditions

The analysis Lake Union got (``scripts/lake_union_conditions.py``), run
on the river -- and for the same reason: both inputs it depends on had
just been replaced, and the only way to know what that changed is to run
it both ways.

What is new under it
--------------------
**Building heights.**  7,101 of 9,463 buildings carried a height guessed
from their type -- 9 m for anything untagged -- so three quarters of the
frontal area the roughness integrates was one number.  They now carry
Overture Maps heights: 9,310 of them, spanning 2,269 distinct values
where there were 60 (SOURCES sec. 121).

**Trees.**  2,156 points, every one between 14.0 and 15.0 m, with no
species.  Now 24,392 from Cambridge's street inventory and Boston's
canopy, with growth forms and heights modelled from measured trunk
diameter.

Roughness goes as frontal area, which is height times width, so these
are exactly the inputs the wind field is most sensitive to -- and the
Charles had never been asked what they do to it.

What this does not claim
------------------------
The chop model is fetch-limited JONSWAP, and a river 200 m wide gives a
crosswind fetch of about 200 m.  That is inside the relations' range but
at the bottom of it, where the waves are small enough that the added
resistance is a percent or two of hull drag.  **The Charles' problem is
not chop, it is depth** (SOURCES secs. 66-67), and this run is here to
establish the size of the wind term, not to discover a new one.
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
from coxswain.river import charles                        # noqa: E402
from coxswain.river.seattle import fetch_at               # noqa: E402
from coxswain.river.structures import (charles_structures,  # noqa: E402
                                       charles_trees)

#: Winds to report, m/s.
WINDS = (6.0, 10.0, 14.0)
#: Bearings the wind comes from.  The reach runs roughly east-west, so
#: the along-river directions are the ones with fetch.
BEARINGS = (0.0, 90.0, 180.0, 270.0)


def wind_along_course(course, structures, channel, speed, bearing,
                      trees=None, samples=40):
    """Mean, minimum and maximum sheltered wind at chest height, m/s."""
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
    parser.add_argument("--out", default="out/charles_conditions")
    parser.add_argument("--speed", type=float, default=3.9)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    from render_charles import charles_race_course

    course, geometry, rowable = charles_race_course()
    structures = charles_structures()
    trees = charles_trees()
    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    print("%s: %.0f m" % (course.name, course.length))
    print("  %d buildings, %d trees" % (len(structures.heights),
                                        len(trees.heights)))

    print("=" * 68)
    print("WIND: what the real heights and the real trees do")
    print("=" * 68)
    print("  Mean sheltered wind at chest height along the course, for a")
    print("  10 m/s reference wind.")
    print()
    print("  %-9s %-7s %9s %9s %9s" % ("wind from", "at 10 m", "buildings",
                                       "+ trees", "change"))
    for bearing in BEARINGS:
        plain = wind_along_course(course, structures, rowable, 10.0, bearing)
        wooded = wind_along_course(course, structures, rowable, 10.0, bearing,
                                   trees=trees)
        print("  %-9.0f %-7.1f %9.2f %9.2f %+8.1f%%"
              % (bearing, 10.0, plain[0], wooded[0],
                 100.0 * (wooded[0] / plain[0] - 1.0)))

    print()
    print("=" * 68)
    print("CHOP: and whether the deep-water assumption survives the survey")
    print("=" * 68)
    mask = (rowable.east, rowable.north, rowable.water)
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
        ratio = depths.min() / max(length, 1e-9)
        deep = ratio > 0.5
        resistance = float(added_resistance(height, boat.offsets.max_beam,
                                            boat.length))
        rows.append((speed, height, sea.peak_period, length, ratio, deep,
                     resistance))
        print("  %-6.0f %8.2f %8.2f %9.2f %10.2f %11s"
              % (speed, height, sea.peak_period, length, ratio,
                 "yes" if deep else "NO"))

    print()
    print("  Deep-water waves need h > L/2.  The shallowest water under the")
    print("  racing line is %.1f m and the longest wave here is %.1f m."
          % (depths.min(), max(r[3] for r in rows)))
    failing = [r[0] for r in rows if not r[5]]
    if failing:
        print("  The deep-water relations FAIL for winds of %s m/s; over the"
              % ", ".join("%.0f" % w for w in failing))
        print("  shallows those waves are depth-limited and the H_s above is")
        print("  an upper bound, not an estimate.")
    else:
        print("  The relations hold at every wind reported -- a checked")
        print("  statement, not an assumption.")

    print()
    print("  %-6s %10s %12s %12s" % ("wind", "H_s open", "added drag",
                                     "share of hull"))
    # Hull resistance at race pace, from the same performance model the
    # rest of the project uses, so the share below is like-for-like.
    from coxswain.sim.performance import SpeedResponse
    hull = float(SpeedResponse(boat).resistance(args.speed))
    for speed, height, _t, _l, _r, _d, resistance in rows:
        print("  %-6.0f %10.2f %11.1f N %11.1f%%"
              % (speed, height, resistance, 100.0 * resistance / hull))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
