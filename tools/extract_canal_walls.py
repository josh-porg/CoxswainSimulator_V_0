r"""The Montlake Cut's walls, walkways and bridge towers, from OpenStreetMap.

    python tools/extract_canal_walls.py --out data/seattle_canal_walls.json

The Cut is not a bank, it is a **walled channel**: concrete retaining
walls both sides, a pedestrian path on top of each, and the Montlake
Bridge's concrete towers standing on the ends of them.  A crew in the Cut
is in a 50 m slot between two walls, and the scene drew it as a grassy
hillside because a hillside is what a bare-earth elevation model shows
when you take the trees off it.

What OpenStreetMap actually has here
------------------------------------
Two long ``barrier=wall`` ways, one each side, running the length of the
Cut -- 590 m on the south side and 763 m on the north.  A row of
``highway=footway`` ways beside them, surfaced ``concrete``, which are
the walkways.  Four short walls flanking the bridge, which are its
approach walls.  And two ``man_made=tower`` polygons at the bridge, one
on each bank: **the towers**.

Heights are not tagged
----------------------
No wall here carries a ``height``.  The wall top is taken from the
elevation model at the wall's own line -- lidar sees the top of a
concrete wall perfectly well, it is the water beside it that lidar
cannot do -- and clamped to a sane range.  The tower height is the one
number here that is neither mapped nor measured: the Montlake Bridge's
towers are given :data:`TOWER_HEIGHT`, and that is an estimate, recorded
as one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OVERPASS = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "CoxswainSimulator/0.1 (rowing course research)"}

#: The Cut and its approaches, (south, west, north, east).
BBOX = "47.6455,-122.3075,47.6495,-122.3010"

#: A wall shorter than this is a garden wall, not a canal wall, m.
MIN_WALL = 15.0
#: Walls and walkways are clamped into this band above the water, m.
MIN_TOP, MAX_TOP = 1.0, 8.0
#: Offsets across a wall at which the elevation model is sampled, m.
OFFSETS = (-6.0, -3.0, 0.0, 3.0, 6.0)


def _offset(east, north, distance):
    """Shift a polyline sideways by ``distance`` along its own normal."""
    line = np.column_stack([east, north])
    step = np.gradient(line, axis=0)
    normal = np.column_stack([-step[:, 1], step[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    moved = line + distance * normal
    return moved[:, 0], moved[:, 1]


#: Height of the Montlake Bridge's towers above the water, m.
#:
#: **Estimated, not measured.**  OpenStreetMap tags no height on either
#: tower and the National Bridge Inventory records the deck and the
#: structure, not the towers over it.  The towers carry the counterweight
#: machinery and stand about two deck heights above a 10.7 m deck; 26 m
#: is that estimate, and it is the only invented number in this file.
TOWER_HEIGHT = 26.0

QUERY = """
[out:json][timeout:180];
(
  way[barrier=wall](%(bbox)s);
  way[man_made=quay](%(bbox)s);
  way[man_made=tower](%(bbox)s);
  way[highway=footway][surface~"concrete|paving_stones"](%(bbox)s);
);
out geom;
""" % {"bbox": BBOX}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/seattle_canal_walls.json")
    args = parser.parse_args(argv)

    from coxswain.river.course import local_tangent_plane
    from coxswain.river.seattle import SEATTLE_ORIGIN
    from coxswain.river.terrain import seattle_terrain

    print("Overpass: the Montlake Cut's walls, walkways and towers")
    response = requests.post(OVERPASS, data={"data": QUERY}, timeout=300,
                             headers=HEADERS)
    response.raise_for_status()
    elements = response.json().get("elements", [])
    terrain = seattle_terrain()

    pieces, counts = [], {}
    for element in elements:
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        tags = element.get("tags", {})
        if tags.get("man_made") == "tower":
            kind = "tower"
        elif tags.get("highway") == "footway":
            kind = "walkway"
        else:
            kind = "wall"

        latitude = np.array([g["lat"] for g in geometry], dtype=float)
        longitude = np.array([g["lon"] for g in geometry], dtype=float)
        east, north = local_tangent_plane(latitude, longitude, SEATTLE_ORIGIN)
        east, north = np.asarray(east), np.asarray(north)
        length = float(np.hypot(*np.diff(np.column_stack([east, north]),
                                         axis=0).T).sum())
        if kind == "wall" and length < MIN_WALL:
            continue
        if kind == "walkway" and length < 60.0:
            continue

        if kind == "tower":
            top = TOWER_HEIGHT
        else:
            # The elevation model **across** the wall, not along it.
            #
            # A wall way is drawn on the water's edge, and sampling the
            # DEM on that line reads the water side of it: the two long
            # Cut walls came out 1.5 and 1.7 m tall, which is a kerb.
            # Sampling a few metres either side and taking the higher
            # finds the ground the wall retains, which is its top.
            above = []
            for offset in OFFSETS:
                shifted_e, shifted_n = _offset(east, north, offset)
                above.append(np.asarray(
                    terrain.height_above_water(shifted_e, shifted_n)))
            top = float(np.clip(np.median(np.max(above, axis=0)),
                                MIN_TOP, MAX_TOP))

        pieces.append({
            "kind": kind,
            "name": str(tags.get("name", ""))[:48],
            "top": round(top, 2),
            "points": [[round(float(x), 2), round(float(y), 2)]
                       for x, y in zip(east, north)],
        })
        counts[kind] = counts.get(kind, 0) + 1

    if not pieces:
        raise SystemExit("nothing came back -- check the bounding box")
    print("  %s" % ", ".join("%d %s" % (n, k) for k, n in sorted(counts.items())))
    for piece in pieces:
        if piece["kind"] in ("wall", "tower"):
            span = np.asarray(piece["points"])
            print("    %-8s top %5.2f m  east %6.0f..%6.0f  north %6.0f..%6.0f"
                  % (piece["kind"], piece["top"], span[:, 0].min(),
                     span[:, 0].max(), span[:, 1].min(), span[:, 1].max()))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"origin": list(SEATTLE_ORIGIN), "pieces": pieces}, handle,
                  separators=(",", ":"))
    print("wrote %s (%.0f kB)" % (args.out, os.path.getsize(args.out) / 1e3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
