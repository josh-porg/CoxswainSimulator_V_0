r"""Fetch public-domain aerial imagery to drape over the terrain.

    python tools/fetch_imagery.py --bounds 47.590 -122.375 47.670 -122.300 \
        --out seattle_imagery.jpg

Source
------
USGS ``USGSNAIPPlus`` ImageServer on The National Map: the USDA National
Agriculture Imagery Program plus high-resolution orthoimagery, 0.3 m
pixels, four bands, **public domain**.  Same service family and the same
``exportImage`` call as ``tools/fetch_dem.py``, so the two register
against each other by construction.

Why bother
----------
The scene had two ground colours: one for water and one for land.  That
is a diagram.  A coxswain recognises where they are from the colour and
texture of the shore -- the park, the gasworks, the boatyards, the line
of houseboats -- and none of that survives being painted a single tan.
Draping the photograph over the elevation model puts it all back for the
cost of one image, and it costs nothing in geometry.

The extent trap, again
----------------------
``exportImage`` honours ``size`` exactly and **moves the bounding box**
to match its aspect ratio, saying so only in its JSON reply.  This gets
the same two defences as the DEM fetcher: size derived from the box's
own degree aspect, and the served extent read back and written to a
sidecar.  If imagery and elevation disagreed about where they were, the
photograph would slide across the hills and nothing would raise an
error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = ("https://imagery.nationalmap.gov/arcgis/rest/services/"
            "USGSNAIPPlus/ImageServer/exportImage")
AGENT = "CoxswainSimulator/0.1 (rowing research; USGS/USDA public domain)"

#: Metres per pixel to aim for.  NAIP is 0.3 m and nothing in this scene
#: is seen closer than a hundred metres, so this is about the texture of
#: the shore rather than its detail.
TARGET_METRES = 2.0


def fetch(bounds, out, metres: float = TARGET_METRES, timeout: float = 300.0):
    """Export imagery over ``bounds`` = ``(south, west, north, east)``."""
    south, west, north, east = [float(v) for v in bounds]
    lat_deg, lon_deg = north - south, east - west
    lon_m = lon_deg * 111_320.0 * math.cos(math.radians(
        0.5 * (south + north)))
    width = max(int(round(lon_m / metres)), 16)
    height = max(int(round(width * lat_deg / lon_deg)), 16)
    if max(width, height) > 4000:            # the service's own limit
        scale = max(width, height) / 4000.0
        width, height = int(width / scale), int(height / scale)
    print("  %d x %d px, about %.1f m per pixel"
          % (width, height, lon_m / width))

    fields = {
        "bbox": "%.10f,%.10f,%.10f,%.10f" % (west, south, east, north),
        "bboxSR": 4326, "imageSR": 4326,
        "size": "%d,%d" % (width, height),
        "format": "jpg", "bandIds": "0,1,2",
        "interpolation": "RSP_BilinearInterpolation",
    }

    def call(fmt):
        query = urllib.parse.urlencode(dict(fields, f=fmt))
        request = urllib.request.Request(ENDPOINT + "?" + query,
                                         headers={"User-Agent": AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    served = json.loads(call("json").decode("utf-8"))["extent"]
    extent = (served["ymin"], served["xmin"], served["ymax"], served["xmax"])
    drift = max(abs(a - b) for a, b in zip(extent, (south, west, north, east)))
    print("  served %.5f %.5f %.5f %.5f" % extent)
    if drift > 1e-6:
        print("  NOTE: service moved the box by up to %.5f deg (%.0f m)"
              % (drift, drift * 111_320.0))

    blob = None
    for attempt in range(4):
        try:
            blob = call("image")
            break
        except urllib.error.HTTPError as error:
            if error.code not in (500, 502, 503, 504) or attempt == 3:
                raise
            width, height = int(width / 1.3), int(height / 1.3)
            fields["size"] = "%d,%d" % (width, height)
            print("   %s -- stepping down to %d x %d"
                  % (error.code, width, height))
            time.sleep(3.0 * (attempt + 1))
    if not blob or len(blob) < 1024:
        raise RuntimeError("no image came back")

    with open(out, "wb") as handle:
        handle.write(blob)
    with open(os.path.splitext(out)[0] + ".json", "w") as handle:
        json.dump({"bounds": list(extent), "requested": list(bounds),
                   "size": [width, height], "source": ENDPOINT,
                   "licence": "public domain (USGS/USDA, The National Map)",
                   "note": "bounds are what the service served, not what "
                           "was asked for"}, handle, indent=2)
    print("  wrote %s (%.1f MiB)" % (out, len(blob) / 1048576.0))
    return extent


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bounds", nargs=4, type=float, required=True,
                        metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    parser.add_argument("--out", required=True,
                        help="file name under coxswain/data/")
    parser.add_argument("--metres", type=float, default=TARGET_METRES)
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "coxswain", "data", args.out)
    fetch(args.bounds, out, metres=args.metres)

    try:
        from PIL import Image
        image = Image.open(out)
        print("  %s, %s, %.1f MiB on disk"
              % (image.size, image.mode, os.path.getsize(out) / 1048576.0))
    except Exception as error:                       # pragma: no cover
        print("  (could not summarise: %s)" % error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
