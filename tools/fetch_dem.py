r"""Fetch a bare-earth elevation tile from the USGS 3DEP ImageServer.

    python tools/fetch_dem.py --bounds 47.590 -122.375 47.670 -122.300 \
        --out seattle_dem.tif

Why this exists
---------------
``coxswain/data/charles_dem.tif`` was exported by hand and the recipe
lived only in a docstring, so the second course had no way to get the
same product.  A river's banks are not decoration -- they carry the wind
shelter and they are how a coxswain recognises where they are -- and a
model that can only do that for the course it was written for is a model
with the Charles baked into it.

Source
------
USGS 3D Elevation Program, ``3DEPElevation`` ImageServer, public domain.
The service is **bare earth**: lidar first returns -- roofs and canopy --
are classified out.  That is the right product for ground and the wrong
one for a skyline, which is why the buildings come separately from
OpenStreetMap (``tools/extract_structures.py``).

Elevations are metres NAVD88.  Water bodies come back as whatever the
lidar made of the surface, so the pool level of an impounded lake is a
thing to *measure* from the tile rather than assume; see
:func:`coxswain.river.terrain.pool_level_from`.

The extent trap
---------------
``exportImage`` honours ``size`` exactly and **moves the bounding box**
to match its aspect ratio.  In EPSG:4326 it treats a degree of longitude
as the same width as a degree of latitude, so a size computed from
*metres* -- which is the only sensible way to choose one -- asks for an
aspect the box does not have, and the service quietly widens the box
rather than refusing.  Requesting 47.590-47.670 N at 1407x2226 returns
47.571-47.689: a 2.2 km shift that georeferences the whole tile wrong
and puts Lake Union's shoreline 700 m up a hillside.

Two defences, because this failed silently once already: the size is
derived from the box's own degree aspect, and the extent the service
*actually* returned is read back from its JSON response and written to a
``.json`` sidecar, which is what :mod:`coxswain.river.terrain` reads.
Never trust the requested box.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import urllib.request

ENDPOINT = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
            "3DEPElevation/ImageServer/exportImage")
AGENT = "CoxswainSimulator/0.1 (rowing research; USGS 3DEP public domain)"

#: Metres per pixel to aim for.  The Charles tile is about 4 m and that
#: resolves the levee, the Storrow fill and the bank slope; finer buys
#: nothing a boat can see and costs the download.
TARGET_METRES = 4.0


def fetch(bounds, out, metres: float = TARGET_METRES, timeout: float = 180.0):
    """Export the DEM over ``bounds`` = ``(south, west, north, east)``.

    Returns the extent the service actually served, in the same order.
    """
    import json
    import math

    south, west, north, east = [float(v) for v in bounds]
    # Size the request from ground distance, then square it up against
    # the box's *degree* aspect -- see "The extent trap" above.  Getting
    # this wrong does not raise; it silently returns a different place.
    lat_deg, lon_deg = north - south, east - west
    lat_m = lat_deg * 111_320.0
    lon_m = lon_deg * 111_320.0 * math.cos(math.radians(
        0.5 * (south + north)))
    width = max(int(round(lon_m / metres)), 16)
    height = max(int(round(width * lat_deg / lon_deg)), 16)
    if width > 4000 or height > 4000:            # service limit is 4100
        scale = max(width, height) / 4000.0
        width, height = int(width / scale), int(height / scale)
    print("  %d x %d px, about %.1f x %.1f m per pixel"
          % (width, height, lon_m / width, lat_m / height))

    fields = {
        "bbox": "%.10f,%.10f,%.10f,%.10f" % (west, south, east, north),
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": "%d,%d" % (width, height),
        "format": "tiff",
        "pixelType": "F32",
        "noData": -9999,
        "interpolation": "RSP_BilinearInterpolation",
    }

    def call(fmt):
        query = urllib.parse.urlencode(dict(fields, f=fmt))
        request = urllib.request.Request(ENDPOINT + "?" + query,
                                         headers={"User-Agent": AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def call_image():
        """Fetch the raster, stepping the size down if the service balks.

        The documented limit is 4100 px a side, but the service returns
        502 well below that -- a 2084 x 2223 float tile is refused where
        1407 x 1501 is served.  The ceiling is on payload, not pixels,
        and it is not published, so it has to be found by asking.
        """
        import time
        for attempt in range(5):
            try:
                return call("image")
            except urllib.error.HTTPError as error:
                if error.code not in (500, 502, 503, 504):
                    raise
                width, height = [int(v) for v in fields["size"].split(",")]
                if attempt and min(width, height) > 400:
                    width, height = int(width / 1.3), int(height / 1.3)
                    fields["size"] = "%d,%d" % (width, height)
                    print("  %s -- stepping down to %d x %d"
                          % (error.code, width, height))
                else:
                    print("  %s -- retrying" % error.code)
                time.sleep(2.0 + 3.0 * attempt)
        raise RuntimeError("service would not serve this tile")

    served = json.loads(call("json").decode("utf-8"))["extent"]
    extent = (served["ymin"], served["xmin"], served["ymax"], served["xmax"])
    drift = max(abs(a - b) for a, b in zip(extent, (south, west, north, east)))
    print("  served %.5f %.5f %.5f %.5f" % extent)
    if drift > 1e-6:
        print("  NOTE: service moved the box by up to %.5f deg (%.0f m)"
              % (drift, drift * 111_320.0))

    blob = call_image()
    if len(blob) < 1024 or blob[:2] not in (b"II", b"MM"):
        raise RuntimeError("not a TIFF -- service said: %r" % blob[:400])

    import io as _io

    import numpy as np
    from PIL import Image
    grid = np.array(Image.open(_io.BytesIO(blob)), dtype=np.float32)
    grid = np.where(np.isfinite(grid) & (grid > -50.0), grid, np.nan)

    # Stored as centimetres in int16 rather than float32 GeoTIFF.  The
    # served tile is 19 MiB and this is 6, for a quantisation error of
    # 5 mm against a product whose own vertical accuracy is nearer 10 cm
    # -- so the loss is entirely in digits 3DEP never had.  Repository
    # size is not a cosmetic concern here: this project has already had
    # one push fail on 190 MB of downloaded source material.
    fill = np.int16(-32768)
    quantised = np.where(np.isnan(grid), np.nan, np.round(grid * 100.0))
    quantised = np.where(np.isnan(quantised), float(fill),
                         np.clip(quantised, -32767, 32767)).astype(np.int16)
    np.savez_compressed(
        out, elevation=quantised, bounds=np.asarray(extent, dtype=float),
        scale=np.float64(0.01), nodata=fill,
        requested=np.asarray(bounds, dtype=float))
    with open(os.path.splitext(out)[0] + ".json", "w") as handle:
        json.dump({"bounds": list(extent), "requested": list(bounds),
                   "size": [int(v) for v in fields["size"].split(",")],
                   "source": ENDPOINT, "units": "centimetres, int16",
                   "note": "bounds are what the service served, not what "
                           "was asked for"}, handle, indent=2)
    print("  wrote %s (%.1f MiB, from a %.1f MiB tile)"
          % (out, os.path.getsize(out) / 1048576.0, len(blob) / 1048576.0))
    return extent


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bounds", nargs=4, type=float, required=True,
                        metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    parser.add_argument("--out", required=True,
                        help="file name under coxswain/data/ (.npz)")
    parser.add_argument("--metres", type=float, default=TARGET_METRES)
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "coxswain", "data", args.out)
    fetch(args.bounds, out, metres=args.metres)

    # Report what came back, because a tile of nodata is a plausible
    # looking file and an implausible piece of ground.
    try:
        import numpy as np
        blob = np.load(out)
        grid = blob["elevation"].astype(float) * float(blob["scale"])
        real = grid[blob["elevation"] != blob["nodata"]]
        print("  %d x %d, %.1f%% valid, %.1f to %.1f m"
              % (grid.shape[1], grid.shape[0],
                 100.0 * len(real) / grid.size, real.min(), real.max()))
    except Exception as error:                       # pragma: no cover
        print("  (could not summarise: %s)" % error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
