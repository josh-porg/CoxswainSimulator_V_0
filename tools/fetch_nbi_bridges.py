r"""What the bridges are actually made of, from the federal inventory.

    python tools/fetch_nbi_bridges.py

The renderer drew every landmark bridge as a slab on round columns, with
a deck height and width typed in by hand.  That is wrong in a way a
coxswain notices, because the Ship Canal Bridge and the Aurora Bridge are
**steel deck trusses** and a truss is the thing you recognise them by.

Rather than find a model of one, ask what they are.  The FHWA
**National Bridge Inventory** records, for every bridge in the country,
the material, the type of design, the span counts, the deck width and the
navigational clearance -- and it is served as a queryable layer by USDOT
BTS.  This project already treats NBI as authoritative for the Charles
(``coxswain.river.bridges.BRIDGE_STRUCTURE``); the difference is that
those numbers were transcribed by hand and these are fetched.

What it says about this water
-----------------------------

==================  ==========  ================  =========  ==========
Crossing            Carries     Type              Max span   Nav clear
==================  ==========  ================  =========  ==========
Lake Wash Ship Can  I-5         steel deck truss    168.2 m     39.0 m
Lake Union          SR 99       steel deck truss    243.8 m     41.1 m
Montlake Cut        SR 513      steel bascule        47.5 m      9.1 m
==================  ==========  ================  =========  ==========

The Aurora Bridge's 243.8 m main span was the longest of its type in the
world when it opened in 1931, which is why it looks the way it does.

The one number that has to be derived
-------------------------------------
NBI gives the **navigational clearance** -- the height of the underside
of the structure over the water, which is the number a mariner needs.
The renderer needs the height of the *deck*, and for a deck truss the
deck sits on top of the truss, so the two differ by the structural depth.

That is not in NBI, so it is estimated from the span at
:data:`TRUSS_DEPTH_RATIO`, and the estimate is checked against the one
bridge whose deck height is published: the Aurora Bridge's roadway is
51 m over the water, against 41.1 + 243.8/20 = 53.3 m here.  Two metres
on fifty, from one stated ratio, and it is recorded as an estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SERVICE = ("https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/"
           "services/NTAD_National_Bridge_Inventory/FeatureServer/0/query")
AGENT = "CoxswainSimulator/0.1 (rowing research; USDOT BTS NTAD)"

#: NBI item 43B, type of design and construction.
STRUCTURE_TYPE = {
    1: "slab", 2: "stringer or girder", 3: "girder and floorbeam",
    4: "tee beam", 5: "box girder, multiple", 6: "box girder, single",
    7: "frame", 8: "orthotropic", 9: "truss, deck", 10: "truss, through",
    11: "arch, deck", 12: "arch, through", 13: "suspension",
    14: "stayed girder", 15: "movable, lift", 16: "movable, bascule",
    17: "movable, swing", 18: "tunnel", 19: "culvert",
    21: "segmental box girder", 22: "channel beam",
}

#: NBI item 43A, kind of material.
STRUCTURE_KIND = {
    1: "concrete", 2: "concrete continuous", 3: "steel",
    4: "steel continuous", 5: "prestressed concrete",
    6: "prestressed concrete continuous", 7: "timber", 8: "masonry",
    9: "aluminium or iron", 0: "other",
}

#: Structural depth of a truss as a fraction of its main span.
#:
#: Deck trusses run about 1/15 to 1/25 of the span; 1/20 is the middle.
#: The Aurora Bridge is the check: 41.1 m clearance plus 243.8/20 gives
#: a 53.3 m deck against a published 51 m.
TRUSS_DEPTH_RATIO = 1.0 / 20.0

FIELDS = ("STRUCTURE_NUMBER_008,FEATURES_DESC_006A,FACILITY_CARRIED_007,"
          "STRUCTURE_KIND_043A,STRUCTURE_TYPE_043B,MAIN_UNIT_SPANS_045,"
          "APPR_SPANS_046,MAX_SPAN_LEN_MT_048,STRUCTURE_LEN_MT_049,"
          "DECK_WIDTH_MT_052,NAV_VERT_CLR_MT_039,NAV_HORR_CLR_MT_040,"
          "YEAR_BUILT_027")


def ask(params, tries: int = 4):
    import time
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
            time.sleep(4 * (attempt + 1))
    raise RuntimeError("unreachable")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bounds", nargs=4, type=float,
                        default=(47.590, -122.375, 47.680, -122.280),
                        metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    parser.add_argument("--out", default="seattle_bridges.npz")
    args = parser.parse_args(argv)

    south, west, north, east = args.bounds
    page = ask({
        "where": "1=1",
        "geometry": "%f,%f,%f,%f" % (west, south, east, north),
        "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": FIELDS, "returnGeometry": "true",
        "resultRecordCount": 400, "f": "json",
    })
    features = page.get("features", [])
    print("National Bridge Inventory: %d structures in the box"
          % len(features))

    rows = []
    for feature in features:
        a = feature.get("attributes") or {}
        geometry = feature.get("geometry") or {}
        clearance = a.get("NAV_VERT_CLR_MT_039") or 0.0
        if clearance <= 3.0 or "x" not in geometry:
            continue                       # not over navigable water
        span = float(a.get("MAX_SPAN_LEN_MT_048") or 0.0)
        kind = int(a.get("STRUCTURE_KIND_043A") or 0)
        design = int(a.get("STRUCTURE_TYPE_043B") or 0)
        trussed = design in (9, 10)
        depth = span * TRUSS_DEPTH_RATIO if trussed else max(span / 30.0, 1.5)
        rows.append((
            str(a.get("FEATURES_DESC_006A") or "")[:40],
            str(a.get("FACILITY_CARRIED_007") or "")[:24],
            float(geometry["y"]), float(geometry["x"]),
            kind, design,
            int(a.get("MAIN_UNIT_SPANS_045") or 0),
            int(a.get("APPR_SPANS_046") or 0),
            span, float(a.get("STRUCTURE_LEN_MT_049") or 0.0),
            float(a.get("DECK_WIDTH_MT_052") or 0.0),
            float(clearance), float(a.get("NAV_HORR_CLR_MT_040") or 0.0),
            int(a.get("YEAR_BUILT_027") or 0),
            float(clearance) + depth, depth,
        ))

    print("  %d cross navigable water:" % len(rows))
    for r in sorted(rows, key=lambda r: -r[9]):
        print("   %-24s %-9s %s / %s"
              % (r[0][:24], r[1][:9],
                 STRUCTURE_KIND.get(r[4], r[4]),
                 STRUCTURE_TYPE.get(r[5], r[5])))
        print("      %d+%d spans, max %.0f m, total %.0f m, deck %.1f m wide"
              % (r[6], r[7], r[8], r[9], r[10]))
        print("      clearance %.1f m -> deck %.1f m (structure %.1f m), %d"
              % (r[11], r[14], r[15], r[13]))

    target = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "coxswain", "data", args.out)
    dtype = [("crosses", "<U40"), ("carries", "<U24"),
             ("latitude", "f8"), ("longitude", "f8"),
             ("kind", "i2"), ("design", "i2"),
             ("main_spans", "i2"), ("approach_spans", "i2"),
             ("max_span", "f4"), ("length", "f4"), ("deck_width", "f4"),
             ("clearance", "f4"), ("horizontal_clearance", "f4"),
             ("year_built", "i4"), ("deck_height", "f4"),
             ("structure_depth", "f4")]
    np.savez_compressed(target, bridges=np.array(rows, dtype=dtype),
                        bounds=np.array(args.bounds, dtype=float))
    print("wrote %s" % target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
