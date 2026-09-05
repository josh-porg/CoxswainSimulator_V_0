r"""Lake Union from above: what the bed actually looks like.

    python scripts/bathymetry_map.py --out out/bathymetry

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
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SOURCES = ("NOAA sounding", "NOAA depth area", "USACE multibeam")
COLOURS = ("#e0a33e", "#8a6b4a", "#2f7fb5")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/bathymetry")
    parser.add_argument("--resolution", type=float, default=10.0)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import NearestNDInterpolator
    from scipy.spatial import cKDTree

    from coxswain.river.course import local_tangent_plane
    from coxswain.river.seattle import (SEATTLE_ORIGIN, nominal_depth,
                                        water_mask)
    from render_totl import totl_course

    os.makedirs(args.out, exist_ok=True)
    ink, panel = "#e6edf2", "#12181d"

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
                                   for i in range(3) if (source == i).any())))

    grid_east, grid_north, wet = water_mask(args.resolution,
                                            names=("Lake Union",))
    mesh_e, mesh_n = np.meshgrid(grid_east, grid_north)
    wet_points = np.column_stack([mesh_e[wet], mesh_n[wet]])

    field = np.full(wet.shape, np.nan)
    field[wet] = NearestNDInterpolator(points, depth)(wet_points)
    which = np.full(wet.shape, np.nan)
    _gap, index = cKDTree(points).query(wet_points)
    which[wet] = source[index]

    course = totl_course(args.resolution)
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
                    norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
                    shading="auto")
    axis.plot(line[:, 0], line[:, 1], color="#ffffff", linewidth=1.2,
              alpha=0.8)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Which survey each cell is nearest to", color=ink,
                   fontsize=11)
    from matplotlib.patches import Patch
    axis.legend(handles=[Patch(facecolor=c, label=s)
                         for c, s in zip(COLOURS, SOURCES)],
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

    figure.suptitle("Lake Union: %d surveyed depths, NOAA chart and USACE "
                    "multibeam" % len(depth), color=ink, fontsize=13)
    figure.tight_layout()
    path = os.path.join(args.out, "bathymetry.png")
    figure.savefig(path, dpi=140, facecolor=panel)
    print("wrote", path)

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
