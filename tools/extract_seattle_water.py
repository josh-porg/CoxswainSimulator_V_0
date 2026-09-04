r"""Lake Union, Portage Bay and Union Bay, from OpenStreetMap.

    python tools/extract_seattle_water.py --out data/seattle_water.json

Tail of the Lake circumnavigates Lake Union, and the crew this project is
for trains on these waters. Modelling the course is a **test of the
program before the Charles**: if the line optimiser and the pacing model
say sensible things about water the crew knows by eye, they are worth
trusting on water they do not.

Why geometry and not bathymetry
-------------------------------
The Charles work is dominated by depth, because the racing line there sits
at a depth Froude number near one and a metre of water either way moves
the answer by tens of seconds (SOURCES sec. 66-67). **Lake Union is not
like that.** It is about 15 m deep through the middle, which puts a four
at 3.9 m/s at

    Fr_h = 3.9 / sqrt(9.81 * 15) = 0.32

-- deeply subcritical, on the flat part of the shallow-water curve, where
the correction is 1.00 to three decimals. So the thing that decides a line
here is **distance and turning**, not water depth.

That makes it a cleaner test than the Charles, not a worse one: it
exercises the geometry half of the model with the depth half switched off.

What this does and does not claim
---------------------------------
The **shoreline is real** -- OpenStreetMap water polygons, which for an
urban lake with a federal navigation channel through it are well surveyed.

The **depth is not**. No sounding data is used, and the depth field this
writes is a documented nominal profile, flat in the middle and shelving at
the shore. Anything built on it is marked ``is_survey=False`` and must
stay that way until real soundings are found. NOAA charts the Lake
Washington Ship Canal, so they exist; they are simply not here yet.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import requests

OVERPASS = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "CoxswainSimulator/0.1 (research)"}

#: Tangent-plane origin: the middle of Lake Union.
SEATTLE_ORIGIN = (47.6395, -122.3330)

#: Seattle only.  **The bounding box is not optional.** Without it the
#: name filters match every Lake Union and Portage Bay on earth, and one
#: of them landed 2100 km east of Seattle -- which blew the raster extent
#: up to 2000 km and asked numpy for 42 GiB.
BBOX = "47.60,-122.36,47.68,-122.26"

#: Piers, floating homes and marina structures.  **These, not the
#: shoreline, are what stops a boat getting close in.** Lake Union is
#: ringed with docks and 141 mapped houseboats; the water inside them is
#: on the map and unrowable in fact.
STRUCTURE_QUERY = """
[out:json][timeout:180];
(
  way[man_made=pier](%(bbox)s);
  way["floating"="yes"](%(bbox)s);
  way[leisure=marina](%(bbox)s);
  way[waterway=dock](%(bbox)s);
  way[man_made=breakwater](%(bbox)s);
  way[building=houseboat](%(bbox)s);
);
out geom;
"""

QUERY = """
[out:json][timeout:180];
(
  way[name="Lake Union"][natural=water](%(bbox)s);
  relation[name="Lake Union"][natural=water](%(bbox)s);
  relation[name="Portage Bay"][natural=water](%(bbox)s);
  way[name="Montlake Cut"](%(bbox)s);
  way[name="Union Bay"][natural=water](%(bbox)s);
  relation[name="Union Bay"][natural=water](%(bbox)s);
);
out geom;
""" % {"bbox": BBOX}


def local_plane(lat, lon, origin):
    """Equirectangular tangent plane, metres east and north of ``origin``."""
    earth = 6378137.0
    lat0, lon0 = origin
    east = np.radians(np.asarray(lon) - lon0) * earth * math.cos(
        math.radians(lat0))
    north = np.radians(np.asarray(lat) - lat0) * earth
    return east, north


def fetch(cache):
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as handle:
            return json.load(handle)
    response = requests.post(OVERPASS, data={"data": QUERY}, timeout=300,
                             headers=HEADERS)
    response.raise_for_status()
    payload = response.json()
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    return payload


def rings(payload, origin):
    """Outer rings of every water body, in the local plane."""
    out = []
    for element in payload.get("elements", []):
        name = (element.get("tags") or {}).get("name", "")
        if element.get("type") == "way":
            geometry = element.get("geometry") or []
            if len(geometry) < 4:
                continue
            lat = [p["lat"] for p in geometry]
            lon = [p["lon"] for p in geometry]
            east, north = local_plane(lat, lon, origin)
            out.append({"name": name, "role": "outer",
                        "points": np.column_stack([east, north])})
        else:
            for member in element.get("members") or []:
                if member.get("type") != "way":
                    continue
                geometry = member.get("geometry") or []
                # **Two points is a valid shoreline segment.**  A
                # standalone way needs 4 points to bound an area, but a
                # relation MEMBER is a piece of a boundary, and the
                # shortest pieces are exactly the ones that close the
                # gaps.  Dropping them left Lake Union's ring open in two
                # places, which made stitching cut a chord across the
                # basin and made a flood fill leak to the grid edge.
                if len(geometry) < 2:
                    continue
                lat = [p["lat"] for p in geometry]
                lon = [p["lon"] for p in geometry]
                east, north = local_plane(lat, lon, origin)
                out.append({"name": name,
                            "role": member.get("role") or "outer",
                            "points": np.column_stack([east, north])})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/seattle_water.json")
    parser.add_argument("--cache", default="data/raw/seattle_overpass.json")
    args = parser.parse_args(argv)

    payload = fetch(args.cache)
    pieces = rings(payload, SEATTLE_ORIGIN)
    # Belt and braces: the bounding box should make this a no-op, but a
    # single stray ring is enough to make the raster unusable, so drop
    # anything absurdly far from the origin and say so.
    near = [p for p in pieces
            if np.abs(p["points"]).max() < 20000.0]
    if len(near) != len(pieces):
        print("dropped %d piece(s) further than 20 km from the origin"
              % (len(pieces) - len(near)))
    pieces = near
    print("fetched %d geometry pieces" % len(pieces))
    by_name = {}
    for piece in pieces:
        by_name.setdefault(piece["name"] or "(unnamed)", 0)
        by_name[piece["name"] or "(unnamed)"] += len(piece["points"])
    for name, count in sorted(by_name.items(), key=lambda kv: -kv[1]):
        print("  %-28s %6d vertices" % (name[:28], count))

    data = {
        "origin": list(SEATTLE_ORIGIN),
        "pieces": [{"name": p["name"], "role": p["role"],
                    "points": [[round(float(x), 2), round(float(y), 2)]
                               for x, y in p["points"]]}
                   for p in pieces],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"))
    total = sum(len(p["points"]) for p in pieces)
    print()
    print("wrote %s (%d vertices, %.2f MB)"
          % (args.out, total, os.path.getsize(args.out) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
