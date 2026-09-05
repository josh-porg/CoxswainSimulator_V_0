"""Fetch the buildings and tree canopy either side of the Charles.

    python tools/extract_structures.py

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

#: Wall material classes, in the order the stored code indexes them.
#:
#: Kept as a *class* rather than resolved to a colour here, because what
#: a brick building should look like is a rendering decision and belongs
#: in the renderer.  What OpenStreetMap knows is that it is brick.
MATERIALS = ("unknown", "brick", "concrete", "glass", "metal", "wood",
             "stone", "plaster", "tile")

#: OSM ``building:material`` and ``material`` values, folded onto those.
MATERIAL_ALIAS = {
    "brick": "brick", "bricks": "brick", "brick_block": "brick",
    "concrete": "concrete", "cement": "concrete",
    "reinforced_concrete": "concrete", "cement_block": "concrete",
    "glass": "glass", "mirror": "glass",
    "metal": "metal", "steel": "metal", "aluminium": "metal",
    "copper": "metal", "zinc": "metal", "corrugated_iron": "metal",
    "wood": "wood", "timber_framing": "wood", "log": "wood",
    "stone": "stone", "sandstone": "stone", "limestone": "stone",
    "granite": "stone", "marble": "stone", "masonry": "stone",
    "plaster": "plaster", "stucco": "plaster", "render": "plaster",
    "tile": "tile", "tiles": "tile", "clay": "tile",
}

#: Roof shapes, in the order the stored code indexes them.
ROOF_SHAPES = ("flat", "gabled", "hipped", "pyramidal", "skillion",
               "dome", "round", "mansard", "gambrel", "half-hipped")

#: Building classes worth telling apart when drawing.  A boathouse, a
#: houseboat and a downtown tower are three different objects to a crew
#: and were three identical grey boxes to the renderer.
KINDS = ("other", "house", "apartments", "commercial", "office",
         "industrial", "civic", "boathouse", "houseboat", "roof",
         "garage", "retail")
KIND_ALIAS = {
    "house": "house", "detached": "house", "semidetached_house": "house",
    "residential": "house", "bungalow": "house", "static_caravan": "house",
    "apartments": "apartments", "dormitory": "apartments",
    "terrace": "apartments", "hotel": "apartments",
    "commercial": "commercial", "retail": "retail", "supermarket": "retail",
    "kiosk": "retail",
    "office": "office",
    "industrial": "industrial", "warehouse": "industrial",
    "manufacture": "industrial", "hangar": "industrial",
    "civic": "civic", "public": "civic", "university": "civic",
    "college": "civic", "school": "civic", "hospital": "civic",
    "church": "civic", "cathedral": "civic", "chapel": "civic",
    "train_station": "civic", "museum": "civic", "stadium": "civic",
    "boathouse": "boathouse", "houseboat": "houseboat",
    "roof": "roof", "garage": "garage", "garages": "garage",
    "carport": "garage", "shed": "garage", "hut": "garage",
}


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
    # 600 m, not 200.  The old cap was chosen where nothing on the
    # Charles clears 60 m, and it silently rejected Columbia Center
    # (284 m) and every other real tower, dropping them back to a
    # 9 m guess from the building type.
    if direct and 2.0 < direct < 600.0:
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


def named_colour(text):
    """An OSM colour tag as RGB in 0-1, or ``None``.

    Accepts ``#rrggbb``, ``#rgb`` and the CSS colour names OSM uses in
    ``building:colour``.  Anything else is left unknown rather than
    guessed: a wrong colour is worse than no colour, because the
    renderer can shade an unknown sensibly and cannot un-paint a
    building somebody tagged ``lightsalmon`` by mistake.
    """
    if not text:
        return None
    text = str(text).strip().lower()
    try:
        from matplotlib.colors import to_rgb
        return tuple(float(v) for v in to_rgb(text))
    except Exception:
        return None


def classify(mapping, tags, *keys):
    """First of ``keys`` present in ``tags``, folded through ``mapping``."""
    for key in keys:
        value = str(tags.get(key, "")).strip().lower()
        if value in mapping:
            return mapping[value]
    return None


def roof_of(tags, wall_height):
    """``(shape index, roof height in metres)``.

    ``roof:height`` is rare, so a gabled or hipped roof with no stated
    height gets a sixth of the wall height, which is the usual 4:12 to
    6:12 pitch on a typical footprint.  Flat roofs get zero, which is
    also what an untagged building gets -- most of this dataset is flat
    by default and saying so is honest.
    """
    shape = str(tags.get("roof:shape", "")).strip().lower()
    index = ROOF_SHAPES.index(shape) if shape in ROOF_SHAPES else 0
    height = parse_height(tags.get("roof:height"))
    if height is None:
        height = 0.0 if index == 0 else min(0.167 * wall_height, 6.0)
    return index, float(max(min(height, 0.6 * wall_height), 0.0))


def tallest_parts(box, ask_fn):
    """``building:part`` heights, as ``(centroid, height)`` pairs.

    OpenStreetMap maps a skyscraper under the Simple 3D Buildings
    schema: a plain ``building`` outline at ground level, plus one
    ``building:part`` polygon per massing step, and **the height lives
    on the parts**.  Reading only the outlines gives Columbia Center as
    a four-storey podium, which is what it literally is -- the other
    280 m are in the parts.

    Returned as centroids rather than polygons because all the caller
    needs is to raise each outline to the tallest thing standing on it.
    """
    data = ask_fn("[out:json][timeout:300];"
                  "(way[\"building:part\"](%s);"
                  "relation[\"building:part\"](%s););out geom;"
                  % (box, box))
    centres, heights = [], []
    for element in data["elements"]:
        tags = element.get("tags", {})
        height = parse_height(tags.get("height"))
        if height is None:
            levels = parse_height(tags.get("building:levels"))
            height = None if levels is None else levels * 3.5
        if height is None or not (2.0 < height < 600.0):
            continue
        for ring in rings(element):
            centres.append(ring.mean(axis=0))
            heights.append(float(height))
    return np.array(centres) if centres else np.zeros((0, 2)), \
        np.array(heights, dtype=float)


def raise_to_parts(polygons, heights, centres, part_heights):
    """Lift each outline to the tallest ``building:part`` standing in it.

    Point-in-polygon for every part against every outline is 40k x 3k
    tests; the bounding-box prefilter takes it to a few per part.
    """
    if not len(centres):
        return heights, 0
    from matplotlib.path import Path

    heights = np.asarray(heights, dtype=float).copy()
    lows = np.array([ring.min(axis=0) for ring in polygons])
    highs = np.array([ring.max(axis=0) for ring in polygons])
    lifted = 0
    for centre, height in zip(centres, part_heights):
        inside = np.nonzero((lows[:, 0] <= centre[0])
                            & (highs[:, 0] >= centre[0])
                            & (lows[:, 1] <= centre[1])
                            & (highs[:, 1] >= centre[1]))[0]
        for index in inside:
            if height <= heights[index]:
                continue
            if Path(polygons[index]).contains_point(centre):
                heights[index] = height
                lifted += 1
                break
    return heights, lifted


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


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch building, canopy and tree footprints from "
                    "OpenStreetMap for a bounding box.")
    parser.add_argument("--bounds", default=None,
                        help="south,west,north,east; defaults to the "
                             "Charles DEM bounds")
    parser.add_argument("--out", default="charles_structures.npz",
                        help="file name under coxswain/data/")
    args = parser.parse_args(argv)

    bounds = (tuple(float(v) for v in args.bounds.split(","))
              if args.bounds else DEM_BOUNDS)
    south, west, north, east = bounds
    box = "%f,%f,%f,%f" % (south, west, north, east)
    print("Overpass, box %s" % box)

    print(" buildings ...")
    data = ask("[out:json][timeout:300];"
               "(way[building](%s);relation[building](%s););out geom;"
               % (box, box))
    polygons, offsets, heights, sources = [], [0], [], []
    materials, colours, kinds, roof_shapes, roof_heights = [], [], [], [], []
    named = []
    skipped = 0
    for element in data["elements"]:
        tags = element.get("tags", {})
        height, source = building_height(tags)
        material = MATERIALS.index(
            classify(MATERIAL_ALIAS, tags, "building:material", "material")
            or "unknown")
        kind = KINDS.index(classify(KIND_ALIAS, tags, "building") or "other")
        colour = named_colour(tags.get("building:colour")
                              or tags.get("colour"))
        shape, roof_height = roof_of(tags, height)
        name = str(tags.get("name", ""))[:48]
        for ring in rings(element):
            polygons.append(ring)
            offsets.append(offsets[-1] + len(ring))
            heights.append(height)
            sources.append(source)
            materials.append(material)
            kinds.append(kind)
            colours.append(colour if colour is not None else (-1.0, -1.0, -1.0))
            roof_shapes.append(shape)
            roof_heights.append(roof_height)
            named.append(name)
        if not rings(element):
            skipped += 1
    print("   %d footprints (%d elements had no usable geometry)"
          % (len(heights), skipped))
    counts = np.bincount(np.array(sources, dtype=int), minlength=3)
    print("   heights: %d tagged, %d from levels, %d inferred from type"
          % tuple(counts))
    known = np.bincount(np.array(materials, dtype=int),
                        minlength=len(MATERIALS))
    listed = ", ".join("%s %d" % (MATERIALS[i], known[i])
                       for i in np.argsort(-known)[:5] if i and known[i])
    print("   materials: %d known of %d (%s)"
          % (len(materials) - known[0], len(materials), listed))
    print("   colours tagged: %d;  pitched roofs: %d;  named: %d"
          % (sum(1 for c in colours if c[0] >= 0.0),
             sum(1 for s in roof_shapes if s), sum(1 for n in named if n)))

    print(" building parts (the real height of every tower) ...")
    centres, part_heights = tallest_parts(box, ask)
    heights, lifted = raise_to_parts(polygons, heights, centres, part_heights)
    print("   %d parts; %d outlines raised, tallest now %.0f m"
          % (len(part_heights), lifted,
             max(heights) if len(heights) else 0.0))

    print(" water bodies, for scenery ...")
    # Scenery water, kept apart from the racing shoreline on purpose.
    # ``data/seattle_water.json`` is a carefully stitched set of named
    # bodies that took several passes to get right, and the optimiser and
    # the depth field both depend on it; this is a wider, cruder net --
    # everything wet in the box -- and it exists only so the renderer can
    # draw Elliott Bay and the ship canal in the distance.
    #
    # It is needed because elevation cannot do the job.  Lidar over water
    # is a specular return, and on Lake Union it comes back so noisy that
    # thresholding the DEM half a metre above the pool calls 48% of the
    # lake dry.  A polygon knows where water is; a DEM knows where high
    # ground is; using each for what it is good at is the same rule that
    # caught the tile being georeferenced 2.2 km out.
    wet = ask("[out:json][timeout:300];"
              "(way[natural=water](%s);relation[natural=water](%s);"
              "way[waterway=riverbank](%s););out geom;" % (box, box, box))
    water_xy, water_offsets = [], [0]
    for element in wet["elements"]:
        for ring in rings(element):
            water_xy.append(ring)
            water_offsets.append(water_offsets[-1] + len(ring))
    print("   %d water polygons" % (len(water_offsets) - 1))

    print(" named bridge decks ...")
    # A bridge is scenery here, not a gate: these are the spans a crew
    # navigates *by* rather than through.  The Aurora Bridge is 47 m over
    # the ship canal and visible from most of Lake Union, and leaving it
    # out of the picture removes the single most recognisable object on
    # the skyline north of downtown.
    #
    # Only named ways, and no footbridges over roads: an unnamed
    # ``bridge=yes`` on a driveway culvert is not a landmark, and there
    # are hundreds of them.
    spans = ask("[out:json][timeout:300];"
                "way[bridge][name][highway~\"^(motorway|trunk|primary|"
                "secondary|tertiary|residential|unclassified)$\"](%s);"
                "out geom;" % box)
    bridge_xy, bridge_offsets, bridge_names, bridge_layer = [], [0], [], []
    for element in spans["elements"]:
        geometry = element.get("geometry") or ()
        if len(geometry) < 2:
            continue
        tags = element.get("tags", {})
        ring = np.array([(point["lat"], point["lon"]) for point in geometry])
        bridge_xy.append(ring)
        bridge_offsets.append(bridge_offsets[-1] + len(ring))
        bridge_names.append(str(tags.get("name", ""))[:48])
        try:
            bridge_layer.append(int(float(tags.get("layer", 1))))
        except (TypeError, ValueError):
            bridge_layer.append(1)
    print("   %d spans (%s)"
          % (len(bridge_names),
             ", ".join(sorted({n for n in bridge_names})[:5]) or "none"))

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
        os.path.abspath(__file__))), "coxswain", "data", args.out)
    np.savez_compressed(
        target,
        building_xy=np.concatenate(polygons) if polygons else np.zeros((0, 2)),
        building_offsets=np.array(offsets, dtype=np.int32),
        building_height=np.array(heights, dtype=np.float32),
        building_height_source=np.array(sources, dtype=np.int8),
        building_material=np.array(materials, dtype=np.int8),
        building_kind=np.array(kinds, dtype=np.int8),
        building_colour=np.array(colours, dtype=np.float32),
        building_roof_shape=np.array(roof_shapes, dtype=np.int8),
        building_roof_height=np.array(roof_heights, dtype=np.float32),
        building_name=np.array(named, dtype="<U48"),
        bridge_xy=np.concatenate(bridge_xy) if bridge_xy
        else np.zeros((0, 2)),
        bridge_offsets=np.array(bridge_offsets, dtype=np.int32),
        bridge_name=np.array(bridge_names, dtype="<U48"),
        bridge_layer=np.array(bridge_layer, dtype=np.int8),
        water_xy=np.concatenate(water_xy) if water_xy else np.zeros((0, 2)),
        water_offsets=np.array(water_offsets, dtype=np.int32),
        canopy_xy=np.concatenate(canopy) if canopy else np.zeros((0, 2)),
        canopy_offsets=np.array(canopy_offsets, dtype=np.int32),
        canopy_height=np.array(canopy_height, dtype=np.float32),
        tree_xy=np.array(points, dtype=np.float64) if points
        else np.zeros((0, 2)),
        tree_height=np.array(tree_height, dtype=np.float32),
        bounds=np.array(bounds, dtype=np.float64),
    )
    print("wrote %s (%.1f MB)" % (target, os.path.getsize(target) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
