r"""The bed from above: what it actually looks like, and who measured it.

    python scripts/bathymetry_map.py --out out/bathymetry
    python scripts/bathymetry_map.py --race hotl --out out/hotl_bathymetry
    python scripts/bathymetry_map.py --race charles --out out/charles_bathymetry

Three panels, because the interesting thing is not the depth but where
it came from.

**Left** -- the merged depth field over the lake, interpolated onto the
racing raster, with the traced course and the optimised line on it.

**Middle** -- the same field coloured by *provenance*: USACE multibeam,
NOAA point sounding, or NOAA charted depth-area bound.  The multibeam
covers the north half, because that is where the federal navigation
channel runs; the southern basin has only the chart.  A depth map that
does not say which parts are surveyed and which are interpolated is
hiding the only thing a reader needs to judge it.

**Right** -- the merged field against the shelf profile it replaced, as
depth under the racing line, so the size of the correction is visible
rather than asserted.

The Charles is a different case and the panels say so.  Lake Union's
depth is *merged* from a chart and a federal dredging survey, so the
question worth asking of it is which parts came from where.  The Charles
reach was surveyed in one pass -- 12,164 vertices of 1-foot contours from
a side-scan run, processed in ReefMaster and corrected for transducer
depth -- so provenance is not in question and there is no invented shelf
to compare against.  The right panel there plots the critical depth
instead: the depth at which the depth Froude number reaches 1 at race
pace, which is the number that decides whether the bed is slowing the
boat.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SOURCES = ("NOAA sounding", "NOAA depth area", "USACE multibeam",
           "shore band, estimated", "side-scan survey, 1 ft contours")
COLOURS = ("#e0a33e", "#8a6b4a", "#2f7fb5", "#b04a5a", "#4a9d6b")
#: Race pace for the depth Froude number on the right-hand panel, m/s.
RACE_SPEED = 3.9
#: Beyond this distance from a sounding, a cell is not drawn, m.
UNSUPPORTED = 90.0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/bathymetry")
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--race", default="totl",
                        choices=("totl", "hotl", "charles"),
                        help="Lake Union and Tail of the Lake, the whole "
                             "ship canal and Head of the Lake, or the "
                             "surveyed Charles reach")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import NearestNDInterpolator
    from scipy.spatial import cKDTree

    from coxswain.river.course import local_tangent_plane
    from coxswain.river.seattle import (SEATTLE_ORIGIN, SHIP_CANAL,
                                        nominal_depth, water_mask)
    charles_reach = args.race == "charles"
    if charles_reach:
        from coxswain.river import charles as _charles
        race_course = _charles.charles_course
        water_names = None
    elif args.race == "hotl":
        from render_hotl import hotl_course as race_course
        water_names = SHIP_CANAL
    else:
        from render_totl import totl_course as race_course
        water_names = ("Lake Union",)

    os.makedirs(args.out, exist_ok=True)
    ink, panel = "#e6edf2", "#12181d"

    if charles_reach:
        # One survey, one source: the isobath vertices as they stand.
        points, depth = _charles.load_isobaths()
        points = np.asarray(points, dtype=float)
        depth = np.asarray(depth, dtype=float)
        source = np.full(len(depth), 4, dtype=int)
    else:
        blob = np.load(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "coxswain", "data",
            "lake_union_depth.npz"))
        xy = blob["depth_xy"]
        east, north = local_tangent_plane(xy[:, 0], xy[:, 1], SEATTLE_ORIGIN)
        points = np.column_stack([np.asarray(east), np.asarray(north)])
        depth = blob["depth"].astype(float)
        source = blob["depth_source"].astype(int)
    print("%d depth values: %s"
          % (len(depth), ", ".join("%s %d" % (SOURCES[i], (source == i).sum())
                                   for i in range(len(SOURCES))
                                   if (source == i).any())))

    if charles_reach:
        raster = _charles.charles_channel(resolution=args.resolution)
        grid_east, grid_north, wet = raster.east, raster.north, raster.water
    else:
        grid_east, grid_north, wet = water_mask(args.resolution,
                                                names=water_names)
    mesh_e, mesh_n = np.meshgrid(grid_east, grid_north)
    wet_points = np.column_stack([mesh_e[wet], mesh_n[wet]])

    field = np.full(wet.shape, np.nan)
    field[wet] = NearestNDInterpolator(points, depth)(wet_points)
    which = np.full(wet.shape, np.nan)
    gap, index = cKDTree(points).query(wet_points)
    which[wet] = source[index]

    # Do not paint water nobody measured.  A nearest-neighbour field will
    # happily carry a sounding a kilometre across a basin and the picture
    # cannot be told from data; cells further than UNSUPPORTED from any
    # sounding are left blank instead, which is the same principle the
    # provenance panel exists for.
    unsupported = np.full(wet.shape, False)
    unsupported[wet] = gap > UNSUPPORTED
    field[unsupported] = np.nan
    which[unsupported] = np.nan
    print("  %.0f%% of the wet cells are within %.0f m of a sounding"
          % (100.0 * float((gap <= UNSUPPORTED).mean()), UNSUPPORTED))

    course = (race_course() if charles_reach
              else race_course(args.resolution))
    line = course.centreline
    station = np.concatenate([[0.0], np.cumsum(
        np.hypot(*np.diff(line, axis=0).T))])

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 8.0),
                                gridspec_kw={"width_ratios": [1, 1, 1.15]})
    figure.patch.set_facecolor(panel)

    # -- 1. the depth field -------------------------------------------
    axis = axes[0]
    axis.set_facecolor(panel)
    image = axis.pcolormesh(grid_east, grid_north, field, cmap="viridis_r",
                            shading="auto", vmin=0.0, vmax=16.0)
    axis.plot(line[:, 0], line[:, 1], color="#ff9248", linewidth=1.8,
              label="course as drawn")
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Depth below the surface, m", color=ink, fontsize=11)
    bar = figure.colorbar(image, ax=axis, shrink=0.6)
    bar.ax.yaxis.set_tick_params(color=ink, labelcolor=ink)
    bar.outline.set_edgecolor("#2a3640")
    axis.legend(loc="lower left", fontsize=8, facecolor="#111a20",
                edgecolor="#2a3640", labelcolor=ink)

    # -- 2. where each value came from ---------------------------------
    axis = axes[1]
    axis.set_facecolor(panel)
    from matplotlib.colors import BoundaryNorm, ListedColormap
    axis.pcolormesh(grid_east, grid_north, which,
                    cmap=ListedColormap(COLOURS),
                    norm=BoundaryNorm(
                        [i - 0.5 for i in range(len(COLOURS) + 1)],
                        len(COLOURS)),
                    shading="auto")
    axis.plot(line[:, 0], line[:, 1], color="#ffffff", linewidth=1.2,
              alpha=0.8)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Surveyed in one pass; blank is unmeasured"
                   if charles_reach
                   else "Which survey each cell is nearest to",
                   color=ink, fontsize=11)
    from matplotlib.patches import Patch
    present = sorted(set(np.asarray(source, dtype=int).tolist()))
    axis.legend(handles=[Patch(facecolor=COLOURS[i], label=SOURCES[i])
                         for i in present],
                loc="lower left", fontsize=8, facecolor="#111a20",
                edgecolor="#2a3640", labelcolor=ink)

    # -- 3. under the line, against what it replaced --------------------
    axis = axes[2]
    axis.set_facecolor(panel)
    from scipy.ndimage import distance_transform_edt
    reach = distance_transform_edt(wet) * args.resolution
    rows = np.clip(np.searchsorted(grid_north, line[:, 1]), 0,
                   len(grid_north) - 1)
    columns = np.clip(np.searchsorted(grid_east, line[:, 0]), 0,
                      len(grid_east) - 1)
    surveyed = field[rows, columns]
    if charles_reach:
        # Nothing was replaced here -- the reach has only ever had the
        # survey -- so the useful second line is the depth at which the
        # bed starts to matter: Fr_h = 1 at race pace.
        critical = RACE_SPEED ** 2 / 9.81
        axis.axhline(critical, color="#e0517a", linewidth=1.3, ls="--",
                     label="critical depth, $Fr_h$ = 1 at %.1f m/s (%.2f m)"
                           % (RACE_SPEED, critical))
        axis.plot(station, surveyed, color="#4a9d6b", linewidth=1.6,
                  label="surveyed, side-scan 1 ft contours")
    else:
        invented = nominal_depth(reach[rows, columns])
        axis.plot(station, invented, color="#a2382a", linewidth=1.4,
                  label="the shelf profile it replaced")
        axis.plot(station, surveyed, color="#2f7fb5", linewidth=1.6,
                  label="surveyed")
    axis.invert_yaxis()
    axis.set_xlabel("distance along the course, m", color=ink)
    axis.set_ylabel("depth, m", color=ink)
    axis.set_title("Under the racing line", color=ink, fontsize=11)
    axis.tick_params(colors=ink)
    for spine in axis.spines.values():
        spine.set_color("#2a3640")
    axis.grid(True, color="#2a3640", linewidth=0.5)
    axis.legend(fontsize=8, facecolor="#111a20", edgecolor="#2a3640",
                labelcolor=ink)

    if charles_reach:
        title = ("The Charles racing reach: %d surveyed depths, side-scan "
                 "1 ft contours" % len(depth))
    else:
        title = ("%s: %d surveyed depths, NOAA chart and USACE multibeam"
                 % ("The ship canal" if args.race == "hotl" else "Lake Union",
                    len(depth)))
    figure.suptitle(title, color=ink, fontsize=13)
    figure.tight_layout()
    path = os.path.join(args.out, "bathymetry.png")
    figure.savefig(path, dpi=140, facecolor=panel)
    print("wrote", path)

    if charles_reach:
        good = np.isfinite(surveyed)
        froude = RACE_SPEED / np.sqrt(9.81 * np.maximum(surveyed[good], 1e-6))
        print("under the line: median %.1f m, shallowest %.1f m"
              % (np.median(surveyed[good]), surveyed[good].min()))
        print("  depth Froude number at %.1f m/s: median %.2f, worst %.2f"
              % (RACE_SPEED, np.median(froude), froude.max()))
        print("  %.0f%% of the line is shallower than the critical depth"
              % (100.0 * float((froude > 1.0).mean())))
        along = cKDTree(points).query(line)[0]
        print("  the line is a median %.0f m from a sounding, worst %.0f m"
              % (np.median(along), along.max()))
        return 0

    good = np.isfinite(surveyed) & np.isfinite(invented)
    error = invented[good] - surveyed[good]
    print("under the line: surveyed median %.1f m, shelf median %.1f m"
          % (np.median(surveyed[good]), np.median(invented[good])))
    print("  the shelf ran %+.1f m deep on average, %.1f m rms, %+.1f worst"
          % (error.mean(), np.sqrt((error ** 2).mean()),
             error[np.argmax(np.abs(error))]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
