"""Draw the ground the wind crosses, and the wind that reaches the water.

    python scripts/wind_maps.py
    python scripts/wind_maps.py --directions 250 340 70 160 --speed 6

Three maps, because the wind model has three inputs and each of them is
worth being able to look at:

**The corridor.**  Bare-earth elevation with the buildings and canopy that
3DEP strips out drawn back on.  This is the picture that makes the wind
model plausible or not -- if the footprints are in the wrong place, the
roughness is too, and no amount of Raupach fixes it.

**The field.**  Wind speed at a rower's chest over the water, for a given
direction.  The gradient across the channel is the whole tactical content
and it does not survive being averaged into a single number.

**The along-course profile.**  What a crew actually meets, station by
station, for several directions at once.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coxswain.hydro.canopy import (ShelteredWind,  # noqa: E402
                                   open_water_equivalent, raupach_roughness)
from coxswain.river import charles                    # noqa: E402
from coxswain.river.structures import charles_structures  # noqa: E402
from coxswain.river.terrain import POOL_LEVEL, charles_terrain  # noqa: E402

INK, MUTED, RULE = "#16211f", "#5c6968", "#dce2e0"
WATER, BANK, BUILDING, CANOPY = "#9fc4d8", "#e6e0cf", "#8a8079", "#4a6b45"


def geometry(month: int = 10):
    raster = charles.charles_channel()
    _, _, race_line, _ = charles.hocr_course(raster)
    course = charles.charles_course(centreline=race_line, month=month)
    station = np.linspace(0.0, course.length, 900)
    line = course.offset_position(station, np.zeros_like(station))
    return raster, course, station, line


def corridor(raster, line, structures, terrain, out):
    """Elevation, footprints and canopy along the reach."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    east, north = raster.east, raster.north
    low = np.array([line[:, 0].min() - 420.0, line[:, 1].min() - 420.0])
    high = np.array([line[:, 0].max() + 420.0, line[:, 1].max() + 420.0])

    gx, gy = np.meshgrid(np.linspace(low[0], high[0], 700),
                         np.linspace(low[1], high[1], 500))
    height = terrain.at(gx.ravel(), gy.ravel()).reshape(gx.shape) - POOL_LEVEL
    height = np.clip(height, 0.0, 26.0)

    figure, ax = plt.subplots(figsize=(13.0, 7.4))
    ax.pcolormesh(gx, gy, height, cmap="YlOrBr", vmin=0.0, vmax=26.0,
                  shading="auto", alpha=0.85)
    ix = (east >= low[0]) & (east <= high[0])
    iy = (north >= low[1]) & (north <= high[1])
    ax.contourf(east[ix], north[iy],
                raster.water[np.ix_(iy, ix)].astype(float), levels=[0.5, 1.5],
                colors=[WATER])

    inside = [p for p in structures.polygons
              if low[0] <= p[:, 0].mean() <= high[0]
              and low[1] <= p[:, 1].mean() <= high[1]]
    ax.add_collection(PolyCollection(inside, facecolors=BUILDING,
                                     edgecolors="none", alpha=0.9, zorder=3))
    wood = [p for p in structures.canopy
            if low[0] <= p[:, 0].mean() <= high[0]
            and low[1] <= p[:, 1].mean() <= high[1]]
    ax.add_collection(PolyCollection(wood, facecolors=CANOPY,
                                     edgecolors="none", alpha=0.35, zorder=2))
    trees = structures.trees
    keep = ((trees[:, 0] >= low[0]) & (trees[:, 0] <= high[0])
            & (trees[:, 1] >= low[1]) & (trees[:, 1] <= high[1]))
    ax.plot(trees[keep, 0], trees[keep, 1], ".", color=CANOPY, ms=1.6,
            alpha=0.7, zorder=4)
    ax.plot(line[:, 0], line[:, 1], color="#a2382a", lw=2.0, zorder=5,
            label="the race line")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east, m")
    ax.set_ylabel("north, m")
    ax.set_title("What the wind crosses: %d buildings, %d canopy polygons, "
                 "%d trees" % (len(inside), len(wood), int(keep.sum())),
                 fontsize=12, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(0.01, 0.02, "shading is bare-earth height above pool level; "
                        "3DEP strips the roofs and trees, so those are OSM",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    figure.tight_layout()
    path = os.path.join(out, "wind_corridor.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def field_maps(raster, line, structures, args, out):
    """Wind at chest height over the water, by direction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low = np.array([line[:, 0].min() - 120.0, line[:, 1].min() - 120.0])
    high = np.array([line[:, 0].max() + 120.0, line[:, 1].max() + 120.0])
    gx, gy = np.meshgrid(np.arange(low[0], high[0], 30.0),
                         np.arange(low[1], high[1], 30.0))

    control = float(open_water_equivalent(1.5, args.speed))
    rows = int(np.ceil(len(args.directions) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(13.0, 4.4 * rows),
                                squeeze=False)
    for axis, direction in zip(axes.ravel(), args.directions):
        wind = ShelteredWind(structures, raster, args.speed, direction,
                             height=1.5)
        speed = np.full(gx.shape, np.nan)
        for i in range(gx.shape[0]):
            for j in range(gx.shape[1]):
                row, column = raster.index_of(gx[i, j], gy[i, j])
                if raster.water[row, column]:
                    speed[i, j] = wind.speed_at(gx[i, j], gy[i, j])
        mesh = axis.pcolormesh(gx, gy, np.ma.masked_invalid(speed),
                               cmap="RdYlBu_r", shading="auto",
                               vmin=0.55 * control, vmax=1.02 * control)
        figure.colorbar(mesh, ax=axis, pad=0.01, label="m/s at 1.5 m")
        axis.plot(line[:, 0], line[:, 1], color=INK, lw=1.0)
        # The arrow has to POINT somewhere, and the first version drew
        # the same one on all four panels: fixed xy and xytext, with the
        # bearing computed and then thrown away.  A wind rose that does
        # not turn with the wind is worse than no wind rose.
        towards = np.radians(90.0 - (direction + 180.0))
        tail = np.array([0.11, 0.90])
        tip = tail + 0.075 * np.array([np.cos(towards), np.sin(towards)])
        axis.annotate("", xy=tuple(tip), xytext=tuple(tail),
                      xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="-|>", color="#a2382a",
                                      lw=2.2))
        axis.text(tail[0], tail[1] - 0.075, "wind", transform=axis.transAxes,
                  fontsize=8, color="#a2382a", ha="center")
        axis.set_title("wind from %03d deg  (open water would be %.2f m/s)"
                       % (direction, control), fontsize=10, color=INK,
                       loc="left")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes.ravel()[len(args.directions):]:
        axis.axis("off")
    figure.suptitle("Wind reaching a rower's chest, from a %.0f m/s forecast"
                    % args.speed, fontsize=12, color=INK, x=0.02, ha="left")
    figure.tight_layout()
    path = os.path.join(out, "wind_field.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def along_course(raster, course, station, line, structures, args, out):
    """Wind and bank roughness station by station."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = np.linspace(60.0, course.length - 60.0, 90)
    points = course.offset_position(sample, np.zeros_like(sample))
    control = float(open_water_equivalent(1.5, args.speed))

    figure, (ax, bx) = plt.subplots(2, 1, figsize=(12.0, 6.6), sharex=True)
    palette = ["#1f5673", "#2a7f62", "#c8901a", "#a2382a", "#6b3fa0"]
    for colour, direction in zip(palette, args.directions):
        wind = ShelteredWind(structures, raster, args.speed, direction,
                             height=1.5)
        speed = [wind.speed_at(p[0], p[1]) for p in points]
        ax.plot(sample, speed, color=colour, lw=1.6,
                label="from %03d deg" % direction)
        rough = [wind.roughness(p[0], p[1]).z0 for p in points]
        bx.plot(sample, rough, color=colour, lw=1.4)
    ax.axhline(control, color=MUTED, lw=1.0, ls="--")
    ax.text(sample[-1], control, " open water", color=MUTED, fontsize=8,
            va="center")
    ax.set_ylabel("wind at 1.5 m, m/s")
    ax.set_title("Along the course, from a %.0f m/s forecast" % args.speed,
                 fontsize=12, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=len(args.directions))
    bx.set_yscale("log")
    bx.set_ylabel("upwind bank $z_0$, m")
    bx.set_xlabel("station along the course, m")
    for axis in (ax, bx):
        axis.grid(color=RULE, lw=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.tight_layout()
    path = os.path.join(out, "wind_profile.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed", type=float, default=6.0)
    parser.add_argument("--directions", type=float, nargs="+",
                        default=[250.0, 340.0, 70.0, 160.0])
    parser.add_argument("--out", default="out/wind")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    raster, course, station, line = geometry()
    structures = charles_structures()
    terrain = charles_terrain()
    print("%d buildings, %d canopy polygons, %d trees; course %.0f m"
          % (len(structures.polygons), len(structures.canopy),
             len(structures.trees), course.length))

    print("drawing the corridor ...")
    print("  wrote", corridor(raster, line, structures, terrain, args.out))
    print("drawing the along-course profile ...")
    print("  wrote", along_course(raster, course, station, line, structures,
                                  args, args.out))
    print("drawing the field maps (the slow one) ...")
    print("  wrote", field_maps(raster, line, structures, args, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
