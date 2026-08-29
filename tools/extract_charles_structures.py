"""Fetch the buildings and tree canopy either side of the Charles.

    python tools/extract_charles_structures.py

Source
------
OpenStreetMap via the Overpass API, over :data:`DEM_BOUNDS` -- the same
box the 3DEP elevation model covers, so the two datasets register against
each other without any extra bookkeeping.

Why this is needed at all
-------------------------
The stored DEM is **bare earth**.  USGS 3DEP lidar is classified and the
first returns -- the trees and the roofs -- are stripped out to leave the
ground.  That is the right product for a river and the wrong one for a
wind field: what shelters the Powerhouse Stretch is not the shape of the
ground, which is nearly flat, but three storeys of Cambridge and a line
of plane trees on the bank.  Those are exactly the returns 3DEP throws
away, so they have to come from somewhere else.

Heights are the weak part, and honestly so
------------------------------------------
Most OSM buildings here carry no height at all.  In a sample of the
Cambridge bank, 9 of 25 had ``building:levels`` and none had ``height``.
So the height of most of this dataset is *inferred* from the building
type, and every roughness length computed downstream inherits that.  The
provenance is recorded per building in the ``height_source`` array --
``0`` measured, ``1`` from levels, ``2`` a guess from the type -- so any
result can be re-run against the measured subset alone to see how much of
it rests on the guesses.

Storey heights follow the usual survey convention: 3.0 m for dwellings,
3.5 m where the ground floor is commercial or institutional.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river.terrain import DEM_BOUNDS            # noqa: E402

ENDPOINT = "https://overpass-api.de/api/interpreter"
AGENT = "CoxswainSimulator/0.1 (rowing research; OSM data via Overpass)"

#: Metres per storey, by what the ground floor is likely to be.
STOREY = {"residential": 3.0, "institutional": 3.5}

#: Height in metres for buildings carrying neither ``height`` nor
#: ``building:levels``.  Deliberately conservative -- over-tall buildings
#: would exaggerate the shelter, which is the direction that would make
#: the model flatter the crew.
DEFAULT_HEIGHT = {
    "house": 7.0, "detached": 7.0, "semidetached_house": 7.0,
    "residential": 10.0, "apartments": 13.0, "dormitory": 14.0,
    "university": 15.0, "college": 12.0, "school": 10.0,
    "commercial": 11.0, "retail": 8.0, "office": 16.0,
    "industrial": 9.0, "warehouse": 9.0, "hospital": 18.0,
    "church": 12.0, "boathouse": 8.0, "garage": 3.0, "garages": 3.0,
    "roof": 4.0, "shed": 3.0, "hut": 3.0, "greenhouse": 4.0,
    "yes": 9.0,
}
FALLBACK_HEIGHT = 9.0

#: Canopy height for mapped woodland and parks where nothing says
#: otherwise.  Mature street and park trees in Cambridge run 12-18 m;
#: 14 m is the middle of that and is the number Raupach's frontal-area
#: index will be most sensitive to.
CANOPY_HEIGHT = 14.0


def ask(query: str, tries: int = 3):
    """One Overpass query, with the retry its rate limiter expects."""
    for attempt in range(tries):
        request = urllib.request.Request(
            ENDPOINT, data=urllib.parse.urlencode({"data": query}).encode(),
            headers={"User-Agent": AGENT})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except Exception as error:                       # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 10 * (attempt + 1)
            print("   %s, retrying in %d s" % (type(error).__name__, wait))
            time.sleep(wait)
    raise RuntimeError("unreachable")


def parse_height(text):
    """OSM height strings: '12', '12 m', '12.5', "40'".  None if unusable."""
    if not text:
        return None
    text = str(text).strip()
    feet = text.endswith("'") or "ft" in text
    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    return value * 0.3048 if feet else value


def building_height(tags):
    """Height in metres, and where it came from (0 tag, 1 levels, 2 type)."""
    direct = parse_height(tags.get("height"))
    if direct and 2.0 < direct < 200.0:
        return direct, 0
    levels = parse_height(tags.get("building:levels"))
    if levels and 0 < levels < 60:
        kind = tags.get("building", "yes")
        per = STOREY["residential"] if kind in (
            "house", "detached", "residential", "apartments",
            "semidetached_house") else STOREY["institutional"]
        return levels * per, 1
    return DEFAULT_HEIGHT.get(tags.get("building", "yes"),
                              FALLBACK_HEIGHT), 2


def rings(element):
    """Outer ring(s) of a way or relation as (lat, lon) arrays."""
    if element["type"] == "way":
        geometry = element.get("geometry")
        if not geometry or len(geometry) < 4:
            return []
        return [np.array([(p["lat"], p["lon"]) for p in geometry])]
    out = []
    for member in element.get("members", ()):
        if member.get("role") not in ("outer", ""):
            continue
        geometry = member.get("geometry")
        if geometry and len(geometry) >= 4:
            out.append(np.array([(p["lat"], p["lon"]) for p in geometry]))
    return out


def main():
    south, west, north, east = DEM_BOUNDS
    box = "%f,%f,%f,%f" % (south, west, north, east)
    print("Overpass, box %s" % box)

    print(" buildings ...")
    data = ask("[out:json][timeout:300];"
               "(way[building](%s);relation[building](%s););out geom;"
               % (box, box))
    polygons, offsets, heights, sources = [], [0], [], []
    skipped = 0
    for element in data["elements"]:
        height, source = building_height(element.get("tags", {}))
        for ring in rings(element):
            polygons.append(ring)
            offsets.append(offsets[-1] + len(ring))
            heights.append(height)
            sources.append(source)
        if not rings(element):
            skipped += 1
    print("   %d footprints (%d elements had no usable geometry)"
          % (len(heights), skipped))
    counts = np.bincount(np.array(sources, dtype=int), minlength=3)
    print("   heights: %d tagged, %d from levels, %d inferred from type"
          % tuple(counts))

    print(" woodland and parks ...")
    wood = ask("[out:json][timeout:300];"
               "(way[natural=wood](%s);way[landuse=forest](%s);"
               "way[leisure=park](%s);relation[leisure=park](%s););out geom;"
               % (box, box, box, box))
    canopy, canopy_offsets, canopy_height = [], [0], []
    for element in wood["elements"]:
        tags = element.get("tags", {})
        height = parse_height(tags.get("height")) or CANOPY_HEIGHT
        for ring in rings(element):
            canopy.append(ring)
            canopy_offsets.append(canopy_offsets[-1] + len(ring))
            canopy_height.append(height)
    print("   %d canopy polygons" % len(canopy_height))

    print(" individual trees ...")
    trees = ask("[out:json][timeout:300];node[natural=tree](%s);out;" % box)
    points, tree_height = [], []
    for element in trees["elements"]:
        tags = element.get("tags", {})
        points.append((element["lat"], element["lon"]))
        points_height = parse_height(tags.get("height"))
        tree_height.append(points_height or CANOPY_HEIGHT)
    print("   %d mapped trees (OSM tree coverage is partial; the park"
          % len(tree_height))
    print("   polygons above are the more complete record of canopy)")

    target = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "coxswain", "data",
        "charles_structures.npz")
    np.savez_compressed(
        target,
        building_xy=np.concatenate(polygons) if polygons else np.zeros((0, 2)),
        building_offsets=np.array(offsets, dtype=np.int32),
        building_height=np.array(heights, dtype=np.float32),
        building_height_source=np.array(sources, dtype=np.int8),
        canopy_xy=np.concatenate(canopy) if canopy else np.zeros((0, 2)),
        canopy_offsets=np.array(canopy_offsets, dtype=np.int32),
        canopy_height=np.array(canopy_height, dtype=np.float32),
        tree_xy=np.array(points, dtype=np.float64) if points
        else np.zeros((0, 2)),
        tree_height=np.array(tree_height, dtype=np.float32),
        bounds=np.array(DEM_BOUNDS, dtype=np.float64),
    )
    print("wrote %s (%.1f MB)" % (target, os.path.getsize(target) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
