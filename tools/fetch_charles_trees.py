r"""Trees along the Charles: Cambridge's inventory and Boston's canopy.

    python tools/fetch_charles_trees.py

What was there before
---------------------
2,156 trees, **every one of them between 14.0 and 15.0 m tall**, with no
species -- OpenStreetMap points with a blanket height applied.  The wind
model takes shelter from frontal area, which is height times width, so a
constant height is a constant answer; and the renderer picks a growth
form from the species, so every tree on the Charles was the same ball on
a stick.  Lake Union got 742,517 trees with measured heights and named
species (SOURCES sec. 108) while the Charles kept the constant, and
nothing said so because nothing compared them.

Sources
-------
**Cambridge Street Trees** (``data.cambridgema.gov``, dataset
``82zb-7qc9``, City of Cambridge open data): position, scientific name,
common name and **trunk diameter**.  The north bank of the racing reach
is Cambridge, and Memorial Drive is its treed edge.

**Boston 2019 Tree Canopy Polygons** (BostonMaps ArcGIS, City of Boston
open data): 4,992 canopy polygons over this box.  The south bank is
Boston, and Boston publishes no street-tree inventory here -- so the
canopy is seeded with trees the way Lake Union's was, which puts a stand
where there is a stand instead of leaving the bank bare.

Height: modelled from diameter, and the model is checked
--------------------------------------------------------
Neither city publishes a measured tree height.  Seattle does, and it
publishes trunk diameter beside it, so the relation

    h = 1.3 + a * D^b        (h and D in metres, 1.3 m = breast height)

was fitted to **11,727 Seattle trees that carry both measurements**:

    conifer     h = 1.3 + 23.07 * D^0.793     RMSE 6.1 m, R2 0.38
    broadleaf   h = 1.3 + 19.02 * D^0.723     RMSE 5.7 m, R2 0.45

**Say what that is and is not.**  It is a functional form with real
coefficients, fitted on real paired measurements, and it beats a
constant by about a quarter of the residual spread.  It is not a
measurement, and it was fitted in the Pacific Northwest and applied in
New England, where the same genus grows to a different size.  Heights
from it are marked ``tree_height_source = 4`` so nothing downstream can
mistake them for the measured Seattle ones, and the honest summary is
that this replaces *no* information about height with *weak* information
about height -- plus real information about species and about where the
trees actually are, which is what the growth form and the frontal area
need.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAMBRIDGE = "https://data.cambridgema.gov/resource/82zb-7qc9.json"
BOSTON_CANOPY = ("https://services.arcgis.com/sFnw0xNflSi8J0uh/ArcGIS/rest/"
                 "services/2019_Tree_Canopy_Polygons/FeatureServer/0/query")
AGENT = "CoxswainSimulator/0.1 (rowing course research)"

#: The racing reach, (south, west, north, east) -- the DEM's own box.
BOUNDS = (42.3480, -71.1450, 42.3790, -71.1000)

#: Growth forms, in the order :mod:`coxswain.viz.river3d` expects.
FORMS = ("broadleaf", "conifer", "columnar", "palm")

CONIFER = {
    "abies", "picea", "pinus", "pseudotsuga", "thuja", "tsuga",
    "chamaecyparis", "cupressus", "cupressocyparis", "juniperus",
    "cedrus", "sequoia", "sequoiadendron", "larix", "calocedrus",
    "cryptomeria", "metasequoia", "taxodium", "araucaria", "podocarpus",
}
COLUMNAR_HINT = ("fastigiat", "columnar", "pyramidal", "'stricta'",
                 "sentinel", "lombardy")

#: Fitted on 11,727 Seattle trees carrying both a measured height and a
#: measured diameter; see the module docstring for what that is worth.
ALLOMETRY = {1: (23.07, 0.793), 0: (19.02, 0.723)}

#: Height codes.  0-2 are the Seattle inventory's (measured, species
#: median, genus default) and 3 is a canopy seed; 4 is this file's
#: diameter model, kept distinct so it can never be read as measured.
MODELLED = 4
SEEDED = 3

INCH = 0.0254
#: Largest believable trunk, m.  Two Cambridge records carry a diameter
#: of 945 inches -- 24 m, which is not a tree -- and the power law turned
#: them into 190 m of timber on the bank.  74 inches is the largest of
#: the rest and is a real veteran; the cut is set above it.
MAX_DIAMETER = 80.0 * 0.0254
#: Tallest tree the region grows, m: a white pine in New England tops out
#: near here, and anything above it is the model extrapolating.
MAX_HEIGHT = 45.0
#: Smallest canopy polygon worth seeding, m2.
MIN_CANOPY_AREA = 60.0
#: Crown spacing as a multiple of crown radius.
SPACING = 1.9
#: Crown radius assumed when seeding, m.
SEED_RADIUS = 4.0


def form_of(scientific: str) -> int:
    name = (scientific or "").strip().lower()
    if any(hint in name for hint in COLUMNAR_HINT):
        return 2
    genus = name.split()[0] if name else ""
    return 1 if genus in CONIFER else 0


def height_from_diameter(diameter_m: float, form: int) -> float:
    a, b = ALLOMETRY.get(1 if form == 1 else 0)
    diameter_m = min(max(diameter_m, 0.02), MAX_DIAMETER)
    return float(min(1.3 + a * diameter_m ** b, MAX_HEIGHT))


def get_json(url, params, tries: int = 4):
    """GET and decode JSON.

    ``requests`` rather than ``urllib``: the Cambridge host presents a
    chain ``urllib`` cannot verify from this machine's store, and
    ``requests`` carries certifi's.
    """
    import requests

    for attempt in range(tries):
        try:
            response = requests.get(url, params=params, timeout=180,
                                    headers={"User-Agent": AGENT})
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    return None


def cambridge_trees():
    """``(points, species, heights, forms)`` from the city inventory."""
    south, west, north, east = BOUNDS
    points, species, heights, forms = [], [], [], []
    rejected = []
    offset, page = 0, 5000
    while True:
        rows = get_json(CAMBRIDGE, {
            "$limit": page, "$offset": offset,
            "$where": ("within_box(the_geom, %f, %f, %f, %f)"
                       % (north, west, south, east))})
        if not rows:
            break
        for row in rows:
            geometry = row.get("the_geom") or {}
            if geometry.get("type") != "Point":
                continue
            longitude, latitude = geometry["coordinates"]
            scientific = (row.get("scientific") or "").strip()
            try:
                diameter = float(row.get("diameter") or 0.0) * INCH
            except (TypeError, ValueError):
                diameter = 0.0
            if diameter <= 0.0:
                continue
            if diameter > MAX_DIAMETER:
                rejected.append(diameter / INCH)
                continue
            form = form_of(scientific)
            points.append((latitude, longitude))
            species.append(scientific[:48])
            forms.append(form)
            heights.append(height_from_diameter(diameter, form))
        offset += len(rows)
        if len(rows) < page:
            break
    if rejected:
        print("  rejected %d trees whose recorded diameter is not a tree: "
              "%s inches" % (len(rejected),
                             ", ".join("%.0f" % v for v in sorted(rejected))))
    return points, species, heights, forms


def boston_canopy():
    """Canopy polygon rings over the reach, as arrays of ``(lon, lat)``."""
    south, west, north, east = BOUNDS
    rings, offset = [], 0
    while True:
        page = get_json(BOSTON_CANOPY, {
            "where": "1=1", "outFields": "FID", "returnGeometry": "true",
            "geometry": "%f,%f,%f,%f" % (west, south, east, north),
            "geometryType": "esriGeometryEnvelope", "inSR": "4326",
            "outSR": "4326", "spatialRel": "esriSpatialRelIntersects",
            "f": "json", "resultOffset": offset, "resultRecordCount": 2000})
        features = page.get("features", [])
        for feature in features:
            for ring in (feature.get("geometry") or {}).get("rings", []):
                if len(ring) >= 4:
                    rings.append(np.asarray(ring, dtype=float))
        offset += len(features)
        if len(features) < 2000 or not page.get("exceededTransferLimit"):
            break
    return rings


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="charles_trees.npz")
    parser.add_argument("--no-canopy", action="store_true")
    args = parser.parse_args(argv)

    from coxswain.river.charles import CHARLES_ORIGIN
    from coxswain.river.course import local_tangent_plane

    print("Cambridge Street Trees")
    points, species, heights, forms = cambridge_trees()
    print("  %d trees with a diameter; heights %.1f to %.1f m, median %.1f"
          % (len(points), min(heights), max(heights), float(np.median(heights))))
    counts = np.bincount(np.asarray(forms, dtype=int), minlength=len(FORMS))
    print("  %s" % ", ".join("%s %d" % (FORMS[i], counts[i])
                             for i in range(len(FORMS)) if counts[i]))
    sources = [MODELLED] * len(points)

    if not args.no_canopy:
        print("Boston 2019 Tree Canopy Polygons")
        rings = boston_canopy()
        print("  %d polygons" % len(rings))
        seeded = seed_canopy(rings, points, heights, forms)
        print("  seeded %d trees in canopy with no inventory tree" % seeded[0])
        points += seeded[1]
        heights += seeded[2]
        forms += seeded[3]
        species += [""] * seeded[0]
        sources += [SEEDED] * seeded[0]

    latitude = np.array([p[0] for p in points], dtype=float)
    longitude = np.array([p[1] for p in points], dtype=float)
    target = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "coxswain", "data", args.out)
    np.savez_compressed(
        target,
        tree_xy=np.column_stack([latitude, longitude]),
        tree_height=np.array(heights, dtype=np.float32),
        tree_form=np.array(forms, dtype=np.int8),
        tree_species=np.array(species, dtype="<U48"),
        tree_height_source=np.array(sources, dtype=np.int8),
        bounds=np.array(BOUNDS, dtype=np.float64),
    )
    print("wrote %s: %d trees (%.2f MB)"
          % (target, len(points), os.path.getsize(target) / 1e6))
    # Sanity: the tangent-plane spread should cover the reach, not a point.
    east, north = local_tangent_plane(latitude, longitude, CHARLES_ORIGIN)
    print("  they span east %.0f..%.0f, north %.0f..%.0f m"
          % (np.min(east), np.max(east), np.min(north), np.max(north)))
    return 0


def seed_canopy(rings, points, heights, forms):
    """Trees inside canopy polygons that hold no inventory tree.

    Heights and forms are copied from the nearest inventory tree, so a
    seeded stand looks like the trees actually growing beside it rather
    than like a default.
    """
    from matplotlib.path import Path
    from scipy.spatial import cKDTree

    if not points:
        return 0, [], [], []
    known = np.array(points, dtype=float)              # (lat, lon)
    index = cKDTree(known)
    # Degrees per metre at this latitude, for spacing and radius.
    per_metre_lat = 1.0 / 111_320.0
    per_metre_lon = per_metre_lat / np.cos(np.radians(known[:, 0].mean()))

    seeds, seed_heights, seed_forms = [], [], []
    step_lon = SPACING * SEED_RADIUS * per_metre_lon
    step_lat = SPACING * SEED_RADIUS * per_metre_lat
    for ring in rings:
        low, high = ring.min(axis=0), ring.max(axis=0)
        width = (high[0] - low[0]) / per_metre_lon
        height = (high[1] - low[1]) / per_metre_lat
        if width * height < MIN_CANOPY_AREA:
            continue
        path = Path(ring)
        lons = np.arange(low[0], high[0] + 1e-12, step_lon)
        lats = np.arange(low[1], high[1] + 1e-12, step_lat)
        if not len(lons) or not len(lats):
            continue
        grid_lon, grid_lat = np.meshgrid(lons, lats)
        candidates = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
        inside = path.contains_points(candidates)
        for lon, lat in candidates[inside]:
            # Deterministic jitter, so a stand is not a plantation.
            wobble = (np.sin(lon * 3.71 + lat * 7.13) * 4371.1) % 1.0
            lon += (wobble - 0.5) * step_lon * 0.6
            lat += (wobble - 0.5) * step_lat * 0.6
            gap, near = index.query([lat, lon])
            if gap < SEED_RADIUS * per_metre_lat * 1.5:
                continue                       # a real tree is already here
            seeds.append((lat, lon))
            seed_heights.append(float(heights[int(near)]))
            seed_forms.append(int(forms[int(near)]))
    return len(seeds), seeds, seed_heights, seed_forms


if __name__ == "__main__":
    raise SystemExit(main())
