r"""Surveyed depth for Lake Union, from the federal navigation chart.

    python tools/fetch_noaa_soundings.py

What this replaces
------------------
:func:`coxswain.river.seattle.nominal_depth` -- a shelf profile *invented*
so the course object had a depth field at all.  Its own docstring says
so, and the course has been declaring ``is_survey=False`` ever since.

Depth is not a decoration in this model.  The shallow-water resistance
rise goes with the depth Froude number :math:`Fr_h = v/\sqrt{gh}`, and on
the Charles that correction is worth about 82 seconds over the race.  A
guessed depth propagates straight into a race time.

Why this data exists
--------------------
Lake Union is part of the **Lake Washington Ship Canal**, a federal
navigation project, so it is charted: NOAA ENC cell ``US5SEAGL``.  Two
layers matter, and they are complementary.

**Soundings** (``Harbor.Sounding_point``, ``Z`` in metres) are point
measurements -- 560 over the lake.  Exact where they are and nowhere
else.

**Depth areas** (``Harbor.Depth_Area``, ``DRVAL1``/``DRVAL2`` in metres)
are polygons that tile the charted water with a depth *range*: this
region is between 5.4 and 9.1 m.  184 of them over the lake.  Coarser
than a sounding and far better distributed, which is what an interpolator
needs.

Both are used: the soundings as they stand, and each depth area
contributing its **shallower bound** at its own centre.  Taking
``DRVAL1`` rather than the middle of the range is deliberate and it is
the conservative direction -- a shallower depth means a higher Froude
number and a slower predicted boat, so any error from this choice makes
the model pessimistic rather than flattering.

Except for the shore band.  The shallowest areas run "from 0 to 1.8 m",
and their lower bound is the shoreline, not a depth.  Placed at the
centre of a polygon that is 700 m across -- the Union Bay flats, where
Head of the Lake finishes -- a zero declared the whole finish leg
aground: the course optimiser saw a 0.16 m bar in 1.5-2.1 m of charted
water and steered round a shoal that does not exist.  A zero-bound area
now contributes the shallowest sounding inside it, or half its upper
bound if it holds none, and is marked as source 3 so the estimate can
be told from the chart (2 is the USACE multibeam that
``fetch_ehydro_bathymetry.py`` merges in afterwards).

Datum
-----
Chart depths are below the sounding datum, which for the ship canal is
the maintained low-water level of the canal -- the same surface the model
calls the pool.  So these are depths below the water the boat floats on,
which is what :class:`~coxswain.river.course.DepthField` wants, and no
conversion is needed.  ``VERDAT`` is carried through so that claim can be
checked rather than believed.
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

BASE = ("https://gis.charttools.noaa.gov/arcgis/rest/services/encdirect/"
        "enc_harbour/MapServer")
SOUNDINGS = 76
DEPTH_AREAS = 227
AGENT = "CoxswainSimulator/0.1 (rowing research; NOAA ENC, public domain)"

#: Lake Union plus Portage Bay and the cut, a little wider than the lake.
BOUNDS = (47.620, -122.360, 47.670, -122.265)


def ask(layer: int, params, tries: int = 4):
    url = "%s/%d/query?%s" % (BASE, layer, urllib.parse.urlencode(params))
    for attempt in range(tries):
        request = urllib.request.Request(url, headers={"User-Agent": AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except Exception as error:                       # noqa: BLE001
            if attempt == tries - 1:
                raise
            print("   %s, retrying" % type(error).__name__)
            time.sleep(4 * (attempt + 1))
    raise RuntimeError("unreachable")


def envelope(bounds):
    south, west, north, east = bounds
    return {
        "where": "1=1",
        "geometry": "%f,%f,%f,%f" % (west, south, east, north),
        "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects", "f": "json",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="lake_union_depth.npz")
    args = parser.parse_args(argv)

    common = envelope(BOUNDS)
    print("NOAA ENC, Lake Washington Ship Canal")

    points, depths, source = [], [], []

    page = ask(SOUNDINGS, dict(common, outFields="Z,QUASOU,SORDAT",
                               returnGeometry="true", resultRecordCount=4000))
    for feature in page.get("features", []):
        geometry = feature.get("geometry") or {}
        value = (feature.get("attributes") or {}).get("Z")
        if value is None or "x" not in geometry:
            continue
        if not (0.0 < float(value) < 80.0):
            continue
        points.append((geometry["y"], geometry["x"]))
        depths.append(float(value))
        source.append(0)                       # a sounding
    print("  %d soundings, %.1f to %.1f m"
          % (len(depths), min(depths), max(depths)) if depths
          else "  no soundings")

    page = ask(DEPTH_AREAS, dict(common,
                                 outFields="DRVAL1,DRVAL2,VERDAT,QUASOU",
                                 returnGeometry="true",
                                 resultRecordCount=2000))
    areas = 0
    datums = set()
    from matplotlib.path import Path
    soundings_xy = np.array([(lon, lat) for lat, lon in points],
                            dtype=float).reshape(-1, 2)
    soundings_z = np.array(depths, dtype=float)
    shore_bands = 0
    for feature in page.get("features", []):
        attributes = feature.get("attributes") or {}
        shallow = attributes.get("DRVAL1")
        deep = attributes.get("DRVAL2")
        rings = (feature.get("geometry") or {}).get("rings") or []
        if shallow is None or not rings:
            continue
        datums.add(attributes.get("VERDAT"))
        for ring in rings:
            ring = np.asarray(ring, dtype=float)
            if len(ring) < 3:
                continue
            centre = ring.mean(axis=0)
            if float(shallow) > 0.0:
                # The shallower bound, at the polygon's own centre.  See
                # the module docstring: it is the conservative choice.
                value, kind = float(shallow), 1
            else:
                # The shore band: its lower bound is the shoreline.  Use
                # what the chart actually measured inside it, or half
                # the upper bound if it measured nothing.
                inside = (Path(ring).contains_points(soundings_xy)
                          if len(soundings_xy) else np.zeros(0, bool))
                if inside.any():
                    value = float(soundings_z[inside].min())
                elif deep is not None and float(deep) > 0.0:
                    value = 0.5 * float(deep)
                else:
                    continue
                kind = 3        # 2 is the USACE multibeam, merged later
                shore_bands += 1
            points.append((centre[1], centre[0]))
            depths.append(value)
            source.append(kind)
            areas += 1
    print("  %d of them shore bands (0 to x m), given a measured or "
          "estimated depth instead of the zero" % shore_bands)
    print("  %d depth areas; vertical datum codes seen: %s"
          % (areas, sorted(d for d in datums if d is not None) or "none"))

    if not depths:
        raise SystemExit("nothing came back -- check the bounding box")

    depths = np.array(depths, dtype=float)
    print("  %d depth values in total; median %.1f m, 5th percentile %.1f m"
          % (len(depths), np.median(depths), np.quantile(depths, 0.05)))

    target = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "coxswain", "data", args.out)
    np.savez_compressed(
        target,
        depth_xy=np.array(points, dtype=np.float64),
        depth=depths.astype(np.float32),
        depth_source=np.array(source, dtype=np.int8),
        bounds=np.array(BOUNDS, dtype=np.float64),
    )
    print("wrote %s (%.0f kB)" % (target, os.path.getsize(target) / 1e3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
