r"""Fill the canopy the tree inventory does not cover.

    python tools/fetch_seattle_canopy.py

Run after ``tools/fetch_seattle_trees.py``, which it extends.

The gap it closes
-----------------
The city's tree inventory is **street trees and park trees**.  Between
them they miss everything on private land, and on the Eastlake and
Westlake shores of Lake Union that is most of it: the inventory puts 516
trees within 450 m of the west bank and every one of them is inland,
behind the buildings, because the shore itself is private.  A crew rowing
past sees a wooded bank and the render showed bare ground.

``Seattle Tree Canopy 2021`` is the other half: 49,876 polygons over this
box, mapped from imagery and lidar, that say **where canopy is** and
nothing else -- no species, no height, one attribute reading
``gridcode = 1``.

So the two are combined.  The polygons say where; the inventory says how
tall and what shape, taken **locally** -- the median height and the
conifer fraction of the inventoried trees within
:data:`NEIGHBOURHOOD` metres.  A blob of canopy on a hillside of Douglas
fir is filled with Douglas-fir-sized cones; the same blob in a street of
maples is filled with maples.  Where there is no inventory nearby at all
the city-wide medians are used and the tree is marked as such.

What this is not
----------------
It is **not a survey of individual trees**.  Nothing here knows that a
particular tree exists; it knows that a particular 300 m2 of ground is
under canopy, and puts a plausible number of plausible crowns in it.  The
provenance code says so, and anything counting trees rather than drawing
them should filter these out.

Trees already in the inventory are not duplicated: a seed point within a
crown radius of an inventoried tree is dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river.terrain import SEATTLE_DEM_BOUNDS       # noqa: E402

SERVICE = ("https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/"
           "services/TreeCanopy_Seattle_2021/FeatureServer/0/query")
AGENT = "CoxswainSimulator/0.1 (rowing research; Seattle open data)"
PAGE = 400

#: Server-side generalisation, degrees (~1 m).  A canopy blob comes with
#: up to 981 vertices and none of them matter: it is used as a mask.
TOLERANCE = 1e-5

#: How far to look for inventoried trees when deciding how tall the
#: canopy here is, metres.  Large enough to find some on a quiet street,
#: small enough that a stand of firs does not borrow its height from a
#: park of maples half a kilometre away.
NEIGHBOURHOOD = 160.0

#: Smallest canopy polygon worth seeding, m2.  Below this it is a shrub
#: or a sliver of a mapping artefact.
MIN_AREA = 25.0

#: Crown spacing as a multiple of the crown radius.  1.9 leaves the
#: crowns just touching, which is what a closed canopy is.
SPACING = 1.9


def ask(params, tries: int = 4):
    for attempt in range(tries):
        request = urllib.request.Request(
            SERVICE + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": AGENT})
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return json.load(response)
        except Exception as error:                       # noqa: BLE001
            if attempt == tries - 1:
                raise
            print("   %s, retrying" % type(error).__name__)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch(bounds):
    """Canopy outer rings over ``bounds``, as (lat, lon) arrays."""
    south, west, north, east = bounds
    common = {
        "where": "1=1",
        "geometry": "%f,%f,%f,%f" % (west, south, east, north),
        "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID", "returnGeometry": "true",
        "maxAllowableOffset": TOLERANCE,
        "orderByFields": "OBJECTID", "f": "json",
    }
    total = ask(dict(common, returnCountOnly="true"))["count"]
    print("  %d canopy polygons to fetch" % total)
    rings, offset = [], 0
    while offset < total:
        page = ask(dict(common, resultOffset=offset, resultRecordCount=PAGE))
        features = page.get("features", [])
        if not features:
            break
        for feature in features:
            for ring in (feature.get("geometry") or {}).get("rings", []):
                if len(ring) >= 4:
                    # Esri gives (x, y) = (lon, lat); everything here is
                    # (lat, lon), the convention the OSM extract set.
                    rings.append(np.asarray(ring, dtype=float)[:, ::-1])
        offset += len(features)
        if (offset // PAGE) % 20 == 0:
            print("   %d / %d" % (min(offset, total), total),
                  flush=True)
    print("   %d rings                    " % len(rings))
    return rings


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="seattle_trees.npz")
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(root, "coxswain", "data", args.out)
    if not os.path.exists(target):
        raise SystemExit("run tools/fetch_seattle_trees.py first")
    blob = dict(np.load(target, allow_pickle=False))
    if int(blob["tree_height_source"].max()) >= 3:
        raise SystemExit("this file already has canopy-seeded trees; "
                         "re-run tools/fetch_seattle_trees.py first")

    print("Seattle Tree Canopy 2021")
    rings = fetch(SEATTLE_DEM_BOUNDS)

    from matplotlib.path import Path

    from coxswain.river.course import local_tangent_plane
    from coxswain.river.seattle import SEATTLE_ORIGIN

    # Work in metres: seeding on a grid in degrees would space trees
    # 1.5 times further apart east-west than north-south at this latitude.
    known_xy = blob["tree_xy"]
    east, north = local_tangent_plane(known_xy[:, 0], known_xy[:, 1],
                                      SEATTLE_ORIGIN)
    known = np.column_stack([np.asarray(east), np.asarray(north)])
    known_height = blob["tree_height"].astype(float)
    known_conifer = (blob["tree_form"] == 1).astype(float)
    city_height = float(np.median(known_height))
    city_conifer = float(known_conifer.mean())
    print("  inventory: %d trees, median %.1f m, %.0f%% conifer"
          % (len(known), city_height, 100 * city_conifer))

    # A spatial index, because the obvious way does not finish.
    # Scanning all 92,656 inventoried trees for each of 49,876 canopy
    # polygons is 4.6 billion distance computations, and the first
    # version of this script was still running after fifty minutes
    # without printing a line.
    from scipy.spatial import cKDTree
    index = cKDTree(known)

    seeds, heights, forms = [], [], []
    borrowed = local = 0
    for number, ring in enumerate(rings):
        if number % 4000 == 0:
            print("   seeding %d / %d" % (number, len(rings)), flush=True)
        e, n = local_tangent_plane(ring[:, 0], ring[:, 1], SEATTLE_ORIGIN)
        poly = np.column_stack([np.asarray(e), np.asarray(n)])
        low, high = poly.min(axis=0), poly.max(axis=0)
        span = high - low
        # Shoelace area in metres.
        area = 0.5 * abs(float(np.dot(poly[:, 0], np.roll(poly[:, 1], 1))
                               - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))))
        if area < MIN_AREA or span.min() < 1.0:
            continue

        centre = poly.mean(axis=0)
        near = np.asarray(index.query_ball_point(centre, NEIGHBOURHOOD),
                          dtype=int)
        if len(near) >= 3:
            height = float(np.median(known_height[near]))
            conifer = float(known_conifer[near].mean())
            local += 1
        else:
            height, conifer = city_height, city_conifer
            borrowed += 1
        radius = max(0.34 * height, 2.0)
        step = SPACING * radius

        grid_e = np.arange(low[0] + 0.5 * step, high[0], step)
        grid_n = np.arange(low[1] + 0.5 * step, high[1], step)
        if not len(grid_e) or not len(grid_n):
            grid_e, grid_n = np.array([centre[0]]), np.array([centre[1]])
        mesh_e, mesh_n = np.meshgrid(grid_e, grid_n)
        points = np.column_stack([mesh_e.ravel(), mesh_n.ravel()])
        # Deterministic jitter, so a stand is not a plantation and the
        # same polygon seeds identically on every run.
        wobble = np.stack([
            np.sin(points[:, 0] * 12.9898 + points[:, 1] * 78.233),
            np.sin(points[:, 0] * 39.3467 + points[:, 1] * 11.1357)], axis=1)
        points = points + (wobble % 1.0 - 0.5) * 0.55 * step

        inside = Path(poly).contains_points(points)
        points = points[inside]
        if not len(points):
            continue
        # Do not duplicate a tree the city already surveyed.
        gap, _which = index.query(points)
        points = points[gap > radius]
        for point in points:
            seeds.append(point)
            heights.append(height)
            seed = (np.sin(point[0] * 3.71 + point[1] * 7.13) * 4371.1) % 1.0
            forms.append(1 if seed < conifer else 0)

    print("   seeded %d trees from %d polygons        " % (len(seeds),
                                                           len(rings)))
    print("   height taken locally for %d polygons, city-wide for %d"
          % (local, borrowed))
    if not seeds:
        return 0

    # Back to latitude and longitude, which is how the file is stored.
    from coxswain.river.course import inverse_tangent_plane
    seeds = np.asarray(seeds)
    lat, lon = inverse_tangent_plane(seeds[:, 0], seeds[:, 1],
                                     SEATTLE_ORIGIN)
    seeded_xy = np.column_stack([lat, lon])
    check_e, check_n = local_tangent_plane(lat, lon, SEATTLE_ORIGIN)
    error = float(max(np.abs(np.asarray(check_e) - seeds[:, 0]).max(),
                      np.abs(np.asarray(check_n) - seeds[:, 1]).max()))
    print("   round trip back to lat/lon: worst error %.3f m" % error)
    assert error < 0.01, "the inverse projection is not the inverse"

    blob.update(
        tree_xy=np.vstack([blob["tree_xy"], seeded_xy]),
        tree_height=np.concatenate([blob["tree_height"],
                                    np.array(heights, dtype=np.float32)]),
        tree_form=np.concatenate([blob["tree_form"],
                                  np.array(forms, dtype=np.int8)]),
        tree_species=np.concatenate([
            blob["tree_species"],
            np.full(len(seeds), "", dtype="<U48")]),
        # 3 = seeded from a canopy polygon; not an observed tree.
        tree_height_source=np.concatenate([
            blob["tree_height_source"],
            np.full(len(seeds), 3, dtype=np.int8)]),
    )
    np.savez_compressed(target, **blob)
    print("wrote %s (%.1f MB), %d trees total"
          % (target, os.path.getsize(target) / 1e6, len(blob["tree_xy"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
