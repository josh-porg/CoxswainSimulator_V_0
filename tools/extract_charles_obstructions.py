r"""Boathouse docks and floats on the Charles, from OpenStreetMap.

    python tools/extract_charles_obstructions.py --out data/charles_obstructions.json

**These, not the bank, are what a coxswain steers off.**  The racing
reach is lined with boathouses -- DeWolfe, BU, Riverside, Weld, Newell,
Cambridge Boat Club -- and every one of them has a float sticking out
into the river.  On Lake Union the equivalent structures took 40% of the
lake out of the optimiser's corridor (SOURCES sec. 104); the line there
had been steering through marinas because the shoreline said the water
was wet.  The Charles model has had no obstruction layer at all, so the
same thing has been true of it and nothing has said so.

What this is not
----------------
It is not a claim that OpenStreetMap has every float.  It has 19 piers
over this reach where a coxswain could name more, and the ones it has
are mapped to the boathouse rather than to the float's outer edge in
some cases.  What it is: **strictly better than nothing**, in the one
direction that matters -- every structure it adds is a real structure,
so the corridor can only get more honest, never less.

Coordinates are written in the tangent plane at
:data:`~coxswain.river.charles.CHARLES_ORIGIN`, the same convention
``data/seattle_obstructions.json`` uses, so
:func:`coxswain.river.charles.load_obstructions` is a straight copy of
the Seattle loader.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river.course import local_tangent_plane      # noqa: E402

OVERPASS = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "CoxswainSimulator/0.1 (rowing course research)"}

#: The racing reach and a margin, (south, west, north, east).  Matches
#: ``coxswain.river.terrain.DEM_BOUNDS`` so the obstruction layer cannot
#: reach past the ground it is drawn on.
BBOX = "42.348,-71.145,42.379,-71.100"

#: Structures that keep a shell off the bank.  The same set the Seattle
#: extractor uses, plus ``man_made=quay``: the Charles has walled
#: embankments where Lake Union has floating docks.
QUERY = """
[out:json][timeout:180];
(
  way[man_made=pier](%(bbox)s);
  way[man_made=quay](%(bbox)s);
  way[man_made=breakwater](%(bbox)s);
  way["floating"="yes"](%(bbox)s);
  way[leisure=marina](%(bbox)s);
  way[waterway=dock](%(bbox)s);
);
out geom;
""" % {"bbox": BBOX}


def kind_of(tags) -> str:
    for key, value in (("man_made", "pier"), ("man_made", "quay"),
                       ("man_made", "breakwater"), ("leisure", "marina"),
                       ("waterway", "dock")):
        if tags.get(key) == value:
            return value
    return "floating"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/charles_obstructions.json")
    args = parser.parse_args(argv)

    from coxswain.river.charles import CHARLES_ORIGIN

    print("Overpass: docks and floats over the racing reach")
    response = requests.post(OVERPASS, data={"data": QUERY}, timeout=300,
                             headers=HEADERS)
    response.raise_for_status()
    elements = response.json().get("elements", [])

    obstructions, counts = [], {}
    for element in elements:
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        tags = element.get("tags", {})
        kind = kind_of(tags)
        latitude = np.array([g["lat"] for g in geometry], dtype=float)
        longitude = np.array([g["lon"] for g in geometry], dtype=float)
        east, north = local_tangent_plane(latitude, longitude, CHARLES_ORIGIN)
        obstructions.append({
            "kind": kind,
            "points": [[round(float(x), 2), round(float(y), 2)]
                       for x, y in zip(np.asarray(east), np.asarray(north))],
        })
        counts[kind] = counts.get(kind, 0) + 1

    if not obstructions:
        raise SystemExit("nothing came back -- check the bounding box")
    print("  %s" % ", ".join("%d %s" % (n, k)
                             for k, n in sorted(counts.items())))
    span = np.concatenate([np.asarray(o["points"]) for o in obstructions])
    print("  they span east %.0f..%.0f, north %.0f..%.0f m"
          % (span[:, 0].min(), span[:, 0].max(),
             span[:, 1].min(), span[:, 1].max()))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"origin": list(CHARLES_ORIGIN),
                   "obstructions": obstructions}, handle,
                  separators=(",", ":"))
    print("wrote %s (%.0f kB)" % (args.out, os.path.getsize(args.out) / 1e3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
