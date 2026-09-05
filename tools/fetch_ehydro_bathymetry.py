r"""USACE multibeam bathymetry for the Lake Washington Ship Canal.

    python tools/fetch_ehydro_bathymetry.py

Merges into ``coxswain/data/lake_union_depth.npz`` over the NOAA ENC
values that ``tools/fetch_noaa_soundings.py`` puts there.

What this adds
--------------
The chart gives **676** depth values inside Lake Union: point soundings
plus the shallower bound of each charted depth area.  That is enough to
say the invented shelf profile was 4 m too deep, and not enough to draw
the bed.

USACE surveys the ship canal because it is a federal navigation project,
and publishes the surveys through **eHydro** as Esri file geodatabases.
The 30 November 2022 survey carries **398,663 multibeam soundings**, of
which **179,898 fall inside Lake Union** -- 266 times the chart's
coverage, at roughly a metre spacing.

Reading it needs a geodatabase driver: ``pyogrio``, which ships GDAL
wheels and exposes the ``OpenFileGDB`` driver.

Coverage, honestly
------------------
eHydro surveys the **navigation channel**, and the federal channel
crosses only the *north* half of Lake Union, between the Fremont Cut and
Portage Bay: the multibeam runs 47.6425 to 47.6536 N.  The southern basin
is not a federal channel and is not surveyed by USACE at all.  So this
does not replace the chart, it thickens it where it reaches, and the two
are merged with the ENC filling everything the multibeam does not cover.

The datum question, settled by comparison
-----------------------------------------
eHydro labels the whole ship canal project ``MLLW``, which is a tidal
datum and makes no sense above the Ballard Locks.  Rather than reason
about it, the two sources were compared where they overlap: 113 chart
points have a multibeam sounding within 15 m, and the median difference
is **-0.16 m** on medians of 9.8 and 9.9 m.  They are the same
measurement of the same thing, so both are depths below the lake surface
and no conversion is applied.  The rms difference of 2.15 m is the depth
*areas* -- which are conservative shallower bounds, not soundings --
doing what they are supposed to do.

Thinning
--------
180,000 points at metre spacing is far finer than anything downstream
asks for; the racing model queries depth to compute a Froude number over
a hull 13 m long.  Binned to :data:`GRID` metres and reduced to the
**median** per cell, which is robust to the odd bad ping in a way a mean
is not.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The survey that reaches furthest into Lake Union.
SURVEY = "LW_01_LWC_20221130_CS_C_2_4_342"
URL = ("https://ehydroprod.blob.core.usgovcloudapi.net/ehydro-surveys/"
       "CENWS/%s.ZIP" % SURVEY)

#: eHydro publishes in NAD83 / Washington North, US survey feet.
SOURCE_CRS = "EPSG:2285"
FOOT = 0.3048

#: Cell size for thinning, metres.
GRID = 5.0

#: Believable depths for this lake, metres.  Published maximum is near
#: 15 m; anything outside is a bad ping or a point on the bank.
MIN_DEPTH, MAX_DEPTH = 0.3, 20.0


def download(path: str):
    import urllib.request
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print("  using the cached %s" % path)
        return path
    print("  downloading %s ..." % URL)
    request = urllib.request.Request(
        URL, headers={"User-Agent": "CoxswainSimulator/0.1 (rowing research)"})
    with urllib.request.urlopen(request, timeout=900) as response, \
            open(path, "wb") as handle:
        handle.write(response.read())
    print("  %.0f MB" % (os.path.getsize(path) / 1e6))
    return path


def read_points(folder: str):
    """``(latitude, longitude, depth_m)`` from the survey's multibeam."""
    import pyproj
    from pyogrio.raw import read

    names = [n for n in os.listdir(folder) if n.endswith(".gdb")]
    if not names:
        raise SystemExit("no .gdb in %s" % folder)
    gdb = os.path.join(folder, names[0])

    meta, _index, _geometry, fields = read(gdb, layer="SurveyPoint_HD",
                                           read_geometry=False)
    columns = {name: values for name, values in zip(meta["fields"], fields)}
    depth = columns["Z_depth"].astype(float) * FOOT
    transformer = pyproj.Transformer.from_crs(SOURCE_CRS, "EPSG:4326",
                                              always_xy=True)
    longitude, latitude = transformer.transform(
        columns["xLocation"].astype(float),
        columns["yLocation"].astype(float))
    print("  %d soundings, %s, %s" % (len(depth), columns["sourceType"][0],
                                      columns["elevationDatum"][0]))
    return np.asarray(latitude), np.asarray(longitude), depth


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="lake_union_depth.npz")
    parser.add_argument("--grid", type=float, default=GRID)
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(root, "coxswain", "data", args.out)
    if not os.path.exists(target):
        raise SystemExit("run tools/fetch_noaa_soundings.py first -- this "
                         "merges over the chart rather than replacing it")

    raw = os.path.join(root, "data", "raw")
    os.makedirs(raw, exist_ok=True)
    archive = download(os.path.join(raw, "ehydro_%s.zip" % SURVEY))
    folder = os.path.join(raw, "ehydro_%s" % SURVEY)
    if not os.path.isdir(folder):
        print("  extracting ...")
        zipfile.ZipFile(archive).extractall(folder)

    print("USACE eHydro %s" % SURVEY)
    latitude, longitude, depth = read_points(folder)

    from coxswain.river.course import (inverse_tangent_plane,
                                       local_tangent_plane)
    from coxswain.river.seattle import SEATTLE_ORIGIN, water_mask

    east, north = local_tangent_plane(latitude, longitude, SEATTLE_ORIGIN)
    east = np.asarray(east)
    north = np.asarray(north)

    # Only what is actually inside the lake, and only believable depths.
    grid_east, grid_north, wet = water_mask(10.0, names=("Lake Union",))
    rows = np.clip(np.searchsorted(grid_north, north), 0, len(grid_north) - 1)
    columns = np.clip(np.searchsorted(grid_east, east), 0, len(grid_east) - 1)
    keep = (wet[rows, columns] & (depth > MIN_DEPTH) & (depth < MAX_DEPTH))
    print("  %d inside Lake Union and believable" % keep.sum())
    if not keep.sum():
        raise SystemExit("nothing inside the lake -- check the survey")
    east, north, depth = east[keep], north[keep], depth[keep]

    # Thin to a grid, median per cell.
    cell_x = np.floor(east / args.grid).astype(np.int64)
    cell_y = np.floor(north / args.grid).astype(np.int64)
    order = np.lexsort((cell_y, cell_x))
    east, north, depth = east[order], north[order], depth[order]
    cell = np.stack([cell_x[order], cell_y[order]])
    start = np.concatenate([[0], np.nonzero(np.any(np.diff(cell), axis=0))[0]
                            + 1, [len(depth)]])
    thin_e, thin_n, thin_d = [], [], []
    for a, b in zip(start[:-1], start[1:]):
        thin_e.append(east[a:b].mean())
        thin_n.append(north[a:b].mean())
        thin_d.append(np.median(depth[a:b]))
    thin_e = np.array(thin_e)
    thin_n = np.array(thin_n)
    thin_d = np.array(thin_d)
    print("  thinned to %d cells at %.0f m; depth %.1f to %.1f m, median %.1f"
          % (len(thin_d), args.grid, thin_d.min(), thin_d.max(),
             np.median(thin_d)))

    blob = dict(np.load(target, allow_pickle=False))
    chart_xy = blob["depth_xy"]
    chart_e, chart_n = local_tangent_plane(chart_xy[:, 0], chart_xy[:, 1],
                                           SEATTLE_ORIGIN)
    chart_e = np.asarray(chart_e)
    chart_n = np.asarray(chart_n)

    # Drop chart values the multibeam covers: where there is a real
    # survey, a charted depth-area bound is the weaker measurement.
    from scipy.spatial import cKDTree
    gap, _which = cKDTree(np.column_stack([thin_e, thin_n])).query(
        np.column_stack([chart_e, chart_n]))
    superseded = gap < 2.0 * args.grid
    print("  %d of %d chart values superseded by multibeam"
          % (superseded.sum(), len(gap)))

    lat, lon = inverse_tangent_plane(thin_e, thin_n, SEATTLE_ORIGIN)
    blob.update(
        depth_xy=np.vstack([chart_xy[~superseded],
                            np.column_stack([lat, lon])]),
        depth=np.concatenate([blob["depth"][~superseded],
                              thin_d.astype(np.float32)]),
        # 2 = USACE multibeam, alongside 0 sounding and 1 depth-area bound.
        depth_source=np.concatenate([
            blob["depth_source"][~superseded],
            np.full(len(thin_d), 2, dtype=np.int8)]),
    )
    np.savez_compressed(target, **blob)
    print("wrote %s (%.1f MB), %d depth values total"
          % (target, os.path.getsize(target) / 1e6, len(blob["depth"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
