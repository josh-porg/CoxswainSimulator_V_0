"""Lake Whatcom's shoreline from OpenStreetMap (relation 415479, osm_lake.json beside this file) as rings.

Joins the relation's member ways into closed outer and inner (island) rings, in (lat, lon).
"""
import json, os

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "whatcom"))   # its data and outputs (gitignored)


def _join(ways):
    """Stitch way coordinate lists into closed rings."""
    ways = [list(w) for w in ways if len(w) > 1]
    rings = []
    while ways:
        ring = ways.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, w in enumerate(ways):
                if w[0] == ring[-1]:
                    ring += w[1:]
                elif w[-1] == ring[-1]:
                    ring += w[::-1][1:]
                elif w[-1] == ring[0]:
                    ring = w[:-1] + ring
                elif w[0] == ring[0]:
                    ring = w[::-1][:-1] + ring
                else:
                    continue
                ways.pop(i)
                changed = True
                break
        rings.append(ring)
    return rings


def rings():
    d = json.load(open(os.path.join(HERE, "osm_lake.json")))
    outer, inner = [], []
    for el in d["elements"]:
        if el["type"] == "relation":
            for m in el["members"]:
                if m.get("type") == "way" and "geometry" in m:
                    pts = [(p["lat"], p["lon"]) for p in m["geometry"]]
                    (inner if m.get("role") == "inner" else outer).append(pts)
        elif el["type"] == "way" and "geometry" in el:
            outer.append([(p["lat"], p["lon"]) for p in el["geometry"]])
    return _join(outer), _join(inner)


if __name__ == "__main__":
    o, i = rings()
    print("outer rings:", [len(r) for r in o], "closed:", [r[0] == r[-1] for r in o])
    print("inner rings (islands):", len(i), [len(r) for r in i][:10])
