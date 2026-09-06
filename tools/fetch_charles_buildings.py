r"""Building heights for the Charles reach, from Overture Maps.

    python tools/fetch_charles_buildings.py

What was wrong
--------------
``coxswain/data/charles_structures.npz`` carried 9,463 buildings and
**7,101 of their heights were guessed from the building type** -- the
extractor's own fallback, 9 m for anything untagged.  2,318 came from
``building:levels`` times a storey height and 44 from an actual ``height``
tag.  So three quarters of the skyline, and three quarters of the frontal
area the wind model integrates, was a constant with a shape drawn round
it.  Lake Union's buildings were replaced with lidar-measured apex
heights (SOURCES sec. 106) and the Charles kept the guess.

The source
----------
**Overture Maps**, buildings theme, release 2026-08-19.0.  Overture is
published under CDLA-Permissive-2.0 with ODbL attribution for its
OpenStreetMap-derived parts, and its building heights come from city
open data and from Microsoft's lidar- and imagery-derived estimates
consolidated per footprint.  Over this box it carries 9,587 buildings and
**9,138 of them have a height** -- 95%, against the 25% that had anything
but a guess.

This is not the same claim the Seattle file makes.  Seattle's heights are
a city lidar product: an apex elevation measured off a point cloud, minus
the bare-earth model, per footprint.  Overture's are a consolidation, and
for many buildings the underlying figure is itself modelled.  They are
recorded as ``height_source = 3`` -- neither the measured 0 nor the
guessed 2 -- so a reader can tell the three apart, and so can a test.

Matching
--------
Overture footprints and the OpenStreetMap footprints already stored are
mostly the same geometry from the same upstream, but not always, so
heights are matched by **nearest centroid within 25 m**.  A footprint
that matches nothing keeps whatever it had, and the run reports how many
did.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RELEASE = "2026-08-19.0"
URL = ("s3://overturemaps-us-west-2/release/%s/theme=buildings/"
       "type=building/*" % RELEASE)

#: The racing reach, (south, west, north, east).
BOUNDS = (42.3480, -71.1450, 42.3790, -71.1000)

#: Heights above this are a data error, not a building, m.  The tallest
#: thing near this reach is the Prudential at 228 m and it is well
#: outside the box; nothing here should approach it.
MAX_HEIGHT = 200.0
#: Centroid match radius, m.
MATCH_RADIUS = 25.0
#: Overture's ``height`` is recorded as source 3.
OVERTURE = 3


def overture_buildings():
    """``(latitude, longitude, height)`` for buildings that carry one."""
    import duckdb

    south, west, north, east = BOUNDS
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; "
                "INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
    rows = con.execute(f"""
        SELECT (bbox.ymin + bbox.ymax) / 2 AS latitude,
               (bbox.xmin + bbox.xmax) / 2 AS longitude,
               height
        FROM read_parquet('{URL}', hive_partitioning=1)
        WHERE bbox.xmin BETWEEN {west} AND {east}
          AND bbox.ymin BETWEEN {south} AND {north}
          AND height IS NOT NULL
    """).fetchall()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="charles_structures.npz")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from scipy.spatial import cKDTree

    from coxswain.river.charles import CHARLES_ORIGIN
    from coxswain.river.course import local_tangent_plane

    target = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "coxswain", "data", args.out)
    blob = dict(np.load(target, allow_pickle=False))
    if int(blob["building_height_source"].max()) >= OVERTURE:
        raise SystemExit("this file already carries Overture heights; "
                         "re-run tools/extract_structures.py first")

    print("Overture Maps buildings, release %s" % RELEASE)
    rows = overture_buildings()
    print("  %d buildings with a height over the reach" % len(rows))

    latitude = np.array([r[0] for r in rows], dtype=float)
    longitude = np.array([r[1] for r in rows], dtype=float)
    height = np.array([r[2] for r in rows], dtype=float)
    keep = (height > 1.0) & (height < MAX_HEIGHT)
    if (~keep).any():
        print("  dropped %d with a height outside 1-%.0f m"
              % (int((~keep).sum()), MAX_HEIGHT))
    latitude, longitude, height = latitude[keep], longitude[keep], height[keep]
    east, north = local_tangent_plane(latitude, longitude, CHARLES_ORIGIN)
    theirs = np.column_stack([np.asarray(east), np.asarray(north)])

    # Our footprints, as centroids in the same plane.
    xy = blob["building_xy"]
    offsets = blob["building_offsets"]
    our_east, our_north = local_tangent_plane(xy[:, 0], xy[:, 1],
                                              CHARLES_ORIGIN)
    ours = np.column_stack([np.asarray(our_east), np.asarray(our_north)])
    centroids = np.array([ours[a:b].mean(axis=0)
                          for a, b in zip(offsets[:-1], offsets[1:])])

    tree = cKDTree(theirs)
    gap, index = tree.query(centroids, distance_upper_bound=MATCH_RADIUS)
    matched = np.isfinite(gap)

    heights = np.array(blob["building_height"], dtype=np.float32)
    sources = np.array(blob["building_height_source"], dtype=np.int8)
    was_guessed = sources == 2
    heights[matched] = height[index[matched]].astype(np.float32)
    sources[matched] = OVERTURE

    print("  matched %d of %d footprints within %.0f m"
          % (int(matched.sum()), len(centroids), MATCH_RADIUS))
    print("  of the %d that were guessed from the building type, %d now have "
          "a height" % (int(was_guessed.sum()),
                        int((was_guessed & matched).sum())))
    still = int((sources == 2).sum())
    print("  heights now: %d Overture, %d from levels, %d tagged, %d still "
          "guessed" % (int((sources == OVERTURE).sum()),
                       int((sources == 1).sum()), int((sources == 0).sum()),
                       still))
    print("  tallest %.0f m; median %.1f m (was %.1f m)"
          % (heights.max(), float(np.median(heights)),
             float(np.median(blob["building_height"]))))

    if args.dry_run:
        print("dry run: nothing written")
        return 0
    blob["building_height"] = heights
    blob["building_height_source"] = sources
    np.savez_compressed(target, **blob)
    print("wrote %s (%.1f MB)" % (target, os.path.getsize(target) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
