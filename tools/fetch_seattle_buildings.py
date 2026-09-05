r"""Seattle's lidar building shells, merged over the OpenStreetMap extract.

    python tools/fetch_seattle_buildings.py

Why this replaces most of the OSM heights
-----------------------------------------
The OpenStreetMap extract over this box has 40,386 footprints and knows
the height of **179** of them.  Another 2,889 come from
``building:levels`` times an assumed storey height, and the remaining
37,318 -- 92% -- are a guess from the building type: every untagged
building in Seattle was nine metres tall.  A skyline built from that is a
skyline of one number.

The City of Seattle publishes ``Building Outlines 2015``: roof outlines
derived from 2015/16 lidar, with ``BP99_APEX``, the **measured** apex
elevation of each roof.  73,005 of them cover this box.  Height is that
apex minus the ground under it from the same 3DEP tile the rest of the
scene stands on.

Checked against published heights before being believed:

======================  ==========  ==========  =========
Building                published    measured      error
======================  ==========  ==========  =========
Columbia Center            284 m      280.4 m     -3.6 m
Amazon Doppler             160 m      156.3 m     -3.7 m
======================  ==========  ==========  =========

Units and datum, which is where this sort of thing goes wrong
-------------------------------------------------------------
``BP99_APEX`` is in **feet** and the DEM is in **metres**, and both are
NAVD88.  Nothing in the service says so.  It was settled by taking two
buildings whose heights are published, computing the answer both ways,
and keeping the reading that agreed -- feet gives 280 m for Columbia
Center and metres gives 1013 m.

What OSM is still better at
---------------------------
Names and classes.  Lidar sees a roof; it does not know it is a
boathouse.  So the geometry and the height come from the city and the
``building=*`` class and name are carried across from OSM by finding
which OSM footprint each lidar outline sits inside.  Anything the lidar
missed -- notably the Space Needle, which is a tower and not a building
outline -- is kept from the OSM set rather than dropped.
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

from coxswain.river.terrain import (SEATTLE_DEM_BOUNDS,      # noqa: E402
                                    seattle_terrain)

SERVICE = ("https://services.arcgis.com/ZOyb2t4B0UYuYNYH/arcgis/rest/"
           "services/Building_Outline_2015/FeatureServer/0/query")
AGENT = "CoxswainSimulator/0.1 (rowing research; City of Seattle open data)"
PAGE = 2000                       # the service's own maxRecordCount
FOOT = 0.3048

#: Server-side generalisation, degrees.  About 0.5 m -- a massing model
#: does not need the 64-vertex roof outline the lidar produced, and the
#: vertex count is most of the download.
TOLERANCE = 5e-6

#: Heights outside this band are not believed, in metres.  A roof apex
#: below the ground under it is a registration failure, not a basement.
MIN_HEIGHT, MAX_HEIGHT = 2.0, 350.0


def ask(params, tries: int = 4):
    for attempt in range(tries):
        request = urllib.request.Request(
            SERVICE + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except Exception as error:                       # noqa: BLE001
            if attempt == tries - 1:
                raise
            print("   %s, retrying" % type(error).__name__)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch(bounds):
    """Every outline in ``bounds`` as ``(rings, apex_feet)``."""
    south, west, north, east = bounds
    envelope = "%f,%f,%f,%f" % (west, south, east, north)
    common = {
        "where": "BP99_APEX IS NOT NULL",
        "geometry": envelope, "geometryType": "esriGeometryEnvelope",
        "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "BP99_APEX", "returnGeometry": "true",
        "maxAllowableOffset": TOLERANCE,
        "orderByFields": "OBJECTID", "f": "geojson",
    }
    total = ask(dict(common, returnCountOnly="true", f="json"))["count"]
    print("  %d outlines to fetch, %d at a time" % (total, PAGE))

    rings, apex = [], []
    offset = 0
    while offset < total:
        page = ask(dict(common, resultOffset=offset, resultRecordCount=PAGE))
        features = page.get("features", [])
        if not features:
            break
        for feature in features:
            value = (feature.get("properties") or {}).get("BP99_APEX")
            geometry = feature.get("geometry") or {}
            shells = ([geometry.get("coordinates", [[]])[0]]
                      if geometry.get("type") == "Polygon"
                      else [part[0] for part in
                            geometry.get("coordinates", [])])
            for shell in shells:
                if value is None or len(shell) < 4:
                    continue
                # GeoJSON is (lon, lat); everything downstream is
                # (lat, lon), which is the convention the OSM extract set.
                ring = np.asarray(shell, dtype=float)[:, ::-1]
                rings.append(ring)
                apex.append(float(value))
        offset += len(features)
        print("   %d / %d" % (min(offset, total), total), end="\r")
    print("   %d rings fetched            " % len(rings))
    return rings, np.array(apex, dtype=float)


def attributes_from_osm(rings, blob):
    """Carry OSM class and name onto the lidar outlines.

    Each lidar roof is matched to the OSM footprint that contains its
    centroid.  Bounding-box prefilter first, or this is 73,000 x 40,000
    point-in-polygon tests.
    """
    from matplotlib.path import Path

    offsets = blob["building_offsets"]
    xy = blob["building_xy"]
    polygons = [xy[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    kinds = blob["building_kind"]
    names = blob["building_name"]
    materials = blob["building_material"]
    colours = blob["building_colour"]

    lows = np.array([p.min(axis=0) for p in polygons])
    highs = np.array([p.max(axis=0) for p in polygons])
    centres = np.array([r.mean(axis=0) for r in rings])

    out_kind = np.zeros(len(rings), dtype=np.int8)
    out_name = np.full(len(rings), "", dtype="<U48")
    out_material = np.zeros(len(rings), dtype=np.int8)
    out_colour = np.full((len(rings), 3), -1.0, dtype=np.float32)
    matched = np.zeros(len(polygons), dtype=bool)

    hits = 0
    for index, centre in enumerate(centres):
        candidates = np.nonzero((lows[:, 0] <= centre[0])
                                & (highs[:, 0] >= centre[0])
                                & (lows[:, 1] <= centre[1])
                                & (highs[:, 1] >= centre[1]))[0]
        for candidate in candidates:
            if not Path(polygons[candidate]).contains_point(centre):
                continue
            out_kind[index] = kinds[candidate]
            out_name[index] = names[candidate]
            out_material[index] = materials[candidate]
            out_colour[index] = colours[candidate]
            matched[candidate] = True
            hits += 1
            break
    print("  %d of %d lidar outlines matched to an OSM building (%.0f%%)"
          % (hits, len(rings), 100.0 * hits / max(len(rings), 1)))
    return out_kind, out_name, out_material, out_colour, matched


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="seattle_structures.npz")
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(root, "coxswain", "data", args.out)
    if not os.path.exists(target):
        raise SystemExit("run tools/extract_structures.py first -- this "
                         "merges over its output rather than replacing it")
    blob = dict(np.load(target, allow_pickle=False))

    print("City of Seattle, Building Outlines 2015 (lidar)")
    rings, apex = fetch(SEATTLE_DEM_BOUNDS)

    print(" heights from apex minus 3DEP ground ...")
    terrain = seattle_terrain()
    from coxswain.river.course import local_tangent_plane
    from coxswain.river.seattle import SEATTLE_ORIGIN
    centres = np.array([r.mean(axis=0) for r in rings])
    east, north = local_tangent_plane(centres[:, 0], centres[:, 1],
                                      SEATTLE_ORIGIN)
    ground = terrain.at(np.asarray(east), np.asarray(north))
    heights = apex * FOOT - ground
    good = (heights > MIN_HEIGHT) & (heights < MAX_HEIGHT)
    print("  %d of %d heights believable (%.0f%%); tallest %.0f m"
          % (good.sum(), len(heights), 100.0 * good.mean(),
             heights[good].max()))
    rings = [r for r, keep in zip(rings, good) if keep]
    heights = heights[good]

    kind, name, material, colour, matched = attributes_from_osm(rings, blob)

    # Anything OSM has that the lidar did not see -- towers, and whatever
    # was built after 2016 -- is kept rather than lost.  The Space Needle
    # is the obvious case: it is not a building outline anywhere.
    offsets = blob["building_offsets"]
    xy = blob["building_xy"]
    osm_polygons = [xy[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    extra = np.nonzero(~matched & (blob["building_height"] > 25.0))[0]
    print("  keeping %d unmatched OSM buildings over 25 m (%s)"
          % (len(extra), ", ".join(sorted({str(blob["building_name"][i])
                                           for i in extra
                                           if blob["building_name"][i]})[:3])))

    for index in extra:
        rings.append(osm_polygons[index])
        heights = np.append(heights, blob["building_height"][index])
        kind = np.append(kind, blob["building_kind"][index])
        name = np.append(name, blob["building_name"][index])
        material = np.append(material, blob["building_material"][index])
        colour = np.vstack([colour, blob["building_colour"][index]])

    new_offsets = np.cumsum([0] + [len(r) for r in rings]).astype(np.int32)
    blob.update(
        building_xy=np.concatenate(rings),
        building_offsets=new_offsets,
        building_height=heights.astype(np.float32),
        # 3 = measured from lidar, alongside 0 tagged, 1 levels, 2 type.
        building_height_source=np.where(
            np.arange(len(heights)) < len(heights) - len(extra), 3,
            2).astype(np.int8),
        building_kind=kind.astype(np.int8),
        building_name=name.astype("<U48"),
        building_material=material.astype(np.int8),
        building_colour=colour.astype(np.float32),
        building_roof_shape=np.zeros(len(rings), dtype=np.int8),
        building_roof_height=np.zeros(len(rings), dtype=np.float32),
    )
    np.savez_compressed(target, **blob)
    print("wrote %s (%.1f MB), %d buildings, tallest %.0f m"
          % (target, os.path.getsize(target) / 1e6, len(rings),
             heights.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
