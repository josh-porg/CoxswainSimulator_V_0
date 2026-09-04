r"""Draw Lake Union so the course can be checked by eye.

    python scripts/render_lake_union.py --out out/lake_union

Every number this project has produced about Lake Union rests on geometry
it extracted itself: a shoreline from OpenStreetMap, 660 dock and
houseboat structures subtracted from it, a lap traced as a contour of
constant clearance. **None of that has been looked at.** A coxswain who
rows there weekly can falsify it in seconds from a picture and cannot
falsify it at all from a table, so this draws the pictures.

Four views, each answering a different question
-----------------------------------------------
**Plan** -- is the water the right shape, and does the line go where a
crew would actually row? The docks are drawn, because they remove 40% of
the lake and are the reason the line sits where it does.

**Rowable water** -- the same lake with the docks taken out, which is the
single most checkable claim here: if the remaining water looks wrong to
someone who rows it, the 40% figure is wrong.

**Oblique** -- the lake with the buildings extruded, including the
downtown towers, because a skyline is how a person recognises a place.

**Cox view** -- from the stern, 0.7 m up, looking down the course. The
only viewpoint from which "is this steerable" means anything.

Deliberately matplotlib rather than the VTK scene in
:mod:`coxswain.viz.scene3d`: that renderer is wired to the Charles through
bridge gates and arch rules that Lake Union does not have, and a picture
that exists beats a picture that is architecturally tidy.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.river.route import (Route, RouteEvaluator,  # noqa: E402
                                  optimise_route)
from coxswain.river.seattle import (SEATTLE_ORIGIN,  # noqa: E402
                                    lake_union_course, load_obstructions,
                                    rowable_mask, water_mask)
from coxswain.river.trajectory import ReducedModel        # noqa: E402

WATER = "#1d3f57"
DRY = "#12181d"
DOCK = "#b8683a"
LINE = "#ff9248"
CENTRE = "#7d8f9c"
BUILDING = "#33414d"


def optimised_line(course, boat):
    evaluator = RouteEvaluator(course, boat=boat,
                               reference_speed=3.895).with_steering(
        ReducedModel())
    best = optimise_route(evaluator, n_control=9, iterations=40)
    stations = np.linspace(0.0, course.length, 400)
    offsets = best.route.offset_at(stations)
    points = []
    centre = np.array([course.position_at(s) for s in stations])
    for index, station in enumerate(stations):
        heading = float(course.heading_at(station))
        normal = np.array([-np.sin(heading), np.cos(heading)])
        points.append(centre[index] + normal * offsets[index])
    return np.asarray(points), best, centre


def draw_plan(axis, east, north, mask, docks=True):
    axis.set_facecolor(DRY)
    axis.pcolormesh(east, north, np.where(mask, 1.0, np.nan),
                    cmap="Blues_r", vmin=0.0, vmax=2.0, shading="auto")
    if docks:
        for _kind, points in load_obstructions():
            if len(points) < 2:
                continue
            axis.plot(points[:, 0], points[:, 1], color=DOCK, linewidth=0.7,
                      alpha=0.9, solid_capstyle="round")
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/lake_union")
    parser.add_argument("--offset", type=float, default=50.0)
    parser.add_argument("--resolution", type=float, default=10.0)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(args.out, exist_ok=True)

    from coxswain.boats import catalog
    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)

    print("building the course")
    course = lake_union_course(resolution=args.resolution,
                               offset=args.offset)
    east, north, water = water_mask(args.resolution, names=("Lake Union",))
    _e, _n, rowable = rowable_mask(args.resolution, names=("Lake Union",))
    line, best, centre = optimised_line(course, boat)
    print("  lap %.0f m; optimised %.1f s, peak yaw %.2f deg/s"
          % (course.length, best.elapsed, best.peak_yaw_rate))

    # -- 1. plan, with docks -------------------------------------------
    figure, axis = plt.subplots(figsize=(7.5, 9.5))
    draw_plan(axis, east, north, water)
    axis.plot(centre[:, 0], centre[:, 1], color=CENTRE, linewidth=1.0,
              linestyle="--", label="lap, %.0f m off shore" % args.offset)
    axis.plot(line[:, 0], line[:, 1], color=LINE, linewidth=2.2,
              label="optimised line")
    axis.plot(line[0, 0], line[0, 1], "o", color=LINE, markersize=7)
    axis.annotate("start / finish\n(Gasworks end)", line[0], color="#cfd8de",
                  fontsize=8, xytext=(12, 12), textcoords="offset points")
    handles = axis.get_legend_handles_labels()[0] + [
        Line2D([], [], color=DOCK, linewidth=1.4,
               label="docks, piers, houseboats")]
    axis.legend(handles=handles, loc="lower left", fontsize=8,
                facecolor="#111a20", edgecolor="#2a3640", labelcolor="#cfd8de")
    axis.set_title("Lake Union: water, docks and the racing line",
                   color="#e6edf2", fontsize=11)
    figure.patch.set_facecolor(DRY)
    figure.tight_layout()
    plan = os.path.join(args.out, "plan.png")
    figure.savefig(plan, dpi=150, facecolor=DRY)
    plt.close(figure)
    print("  wrote %s" % plan)

    # -- 2. what the docks leave --------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 8.0))
    draw_plan(axes[0], east, north, water, docks=False)
    axes[0].set_title("all water\n%.3f km$^2$" % (water.sum()
                                                  * args.resolution ** 2 / 1e6),
                      color="#e6edf2", fontsize=10)
    draw_plan(axes[1], east, north, rowable, docks=True)
    axes[1].plot(line[:, 0], line[:, 1], color=LINE, linewidth=2.0)
    axes[1].set_title("rowable after docks\n%.3f km$^2$  (%.0f%% removed)"
                      % (rowable.sum() * args.resolution ** 2 / 1e6,
                         100 * (1 - rowable.sum() / water.sum())),
                      color="#e6edf2", fontsize=10)
    figure.patch.set_facecolor(DRY)
    figure.tight_layout()
    rowable_png = os.path.join(args.out, "rowable.png")
    figure.savefig(rowable_png, dpi=150, facecolor=DRY)
    plt.close(figure)
    print("  wrote %s" % rowable_png)

    # -- 3. oblique, with the skyline ----------------------------------
    from coxswain.river.structures import seattle_structures
    structures = seattle_structures(SEATTLE_ORIGIN)

    figure = plt.figure(figsize=(12.0, 7.5))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor(DRY)
    figure.patch.set_facecolor(DRY)

    grid_east, grid_north = np.meshgrid(east, north)
    axis.plot_surface(grid_east, grid_north,
                      np.where(water, 0.0, np.nan),
                      color=WATER, alpha=0.85, linewidth=0, shade=False,
                      rcount=120, ccount=120)

    # Buildings, tallest first so the skyline reads.
    order = np.argsort(-structures.heights)
    drawn = 0
    for index in order:
        if drawn >= 900:
            break
        polygon = structures.polygons[index]
        if len(polygon) < 3:
            continue
        centre_xy = polygon.mean(axis=0)
        if abs(centre_xy[0]) > 2600 or abs(centre_xy[1]) > 3400:
            continue
        height = float(structures.heights[index])
        if height < 8.0 and drawn > 400:
            continue
        shade = min(0.25 + height / 200.0, 0.95)
        axis.plot(polygon[:, 0], polygon[:, 1],
                  np.full(len(polygon), height),
                  color=BUILDING, alpha=shade, linewidth=0.5)
        drawn += 1
    print("  drew %d buildings (tallest %.0f m)" % (drawn,
                                                    structures.heights.max()))

    axis.plot(line[:, 0], line[:, 1], np.zeros(len(line)),
              color=LINE, linewidth=2.5)
    axis.set_box_aspect((2.0, 3.0, 0.7))
    axis.view_init(elev=22, azim=-118)
    axis.set_axis_off()
    axis.set_title("Lake Union from the south-west, with the Seattle skyline",
                   color="#e6edf2", fontsize=11)
    oblique = os.path.join(args.out, "oblique.png")
    figure.savefig(oblique, dpi=150, facecolor=DRY, bbox_inches="tight")
    plt.close(figure)
    print("  wrote %s" % oblique)

    # -- 4. from the coxswain's seat -----------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for panel, fraction in zip(axes, (0.05, 0.35, 0.70)):
        index = int(fraction * (len(line) - 1))
        here = line[index]
        ahead = line[min(index + 6, len(line) - 1)]
        heading = np.arctan2(ahead[1] - here[1], ahead[0] - here[0])
        forward = np.array([np.cos(heading), np.sin(heading)])
        side = np.array([-forward[1], forward[0]])

        def project(points):
            delta = np.asarray(points) - here
            along = delta @ forward
            across = delta @ side
            keep = along > 3.0
            return across[keep] / along[keep], 1.0 / along[keep], along[keep]

        panel.set_facecolor("#0a1218")
        wet = np.column_stack([grid_east[water], grid_north[water]])
        x, y, along = project(wet)
        panel.scatter(x, y * 30.0, s=1.4, c=along, cmap="Blues_r",
                      alpha=0.55, linewidths=0)
        for _kind, points in load_obstructions():
            if len(points) < 2:
                continue
            if np.abs(points - here).max() > 900:
                continue
            dx, dy, _a = project(points)
            if len(dx):
                panel.plot(dx, dy * 30.0, color=DOCK, linewidth=1.1,
                           alpha=0.9)
        lx, ly, _a = project(line)
        panel.plot(lx, ly * 30.0, color=LINE, linewidth=2.0)
        panel.set_xlim(-1.1, 1.1)
        panel.set_ylim(0.0, 1.4)
        panel.set_xticks([])
        panel.set_yticks([])
        panel.set_title("%.0f%% of the lap" % (100 * fraction),
                        color="#cfd8de", fontsize=9)
    figure.suptitle("From the coxswain's seat: line in orange, docks in "
                    "rust", color="#e6edf2", fontsize=11)
    figure.patch.set_facecolor(DRY)
    figure.tight_layout()
    cox = os.path.join(args.out, "cox_view.png")
    figure.savefig(cox, dpi=150, facecolor=DRY)
    plt.close(figure)
    print("  wrote %s" % cox)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
