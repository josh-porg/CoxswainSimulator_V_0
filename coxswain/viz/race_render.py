r"""Draw any head race, not just the Charles.

`scripts/render3d.py` is wired to the Charles through bridge gates, arch
rules and `hocr_course`, none of which exist on the Oklahoma, on Lake
Union, or on most rivers. This draws whatever it is given: a course, a
water mask, an optional set of structures, and any number of lines to
compare.

The four views each answer a different question
-----------------------------------------------
**Plan** -- is the water the right shape and does the line go where a crew
would row? This is the one that matters, because it is the one that
catches extraction errors. A wrong lake produces plausible lap lengths and
splits (SOURCES sec. 99); it does not survive being drawn.

**Profile** -- depth under the line, with the critical depth for the boat
marked. On the Charles that line is nearly touched (sec. 79); on a lake it
is nowhere near, and seeing which is which is the fastest way to know
whether depth is going to decide anything.

**Oblique** -- the course with structures extruded, for recognising the
place.

**Cox view** -- from the stern, looking down the course.

Deliberately matplotlib. The VTK scene in :mod:`coxswain.viz.scene3d` is
better looking and harder to point at a new river; a picture that exists
beats one that is architecturally tidy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["RaceScene", "TraceLine", "render_all"]

GRAVITY = 9.80665

WATER = "#1d3f57"
DRY = "#12181d"
INK = "#e6edf2"
MUTED = "#7d8f9c"
OBSTRUCTION = "#b8683a"
STRUCTURE = "#33414d"
PALETTE = ("#ff9248", "#4fc3f7", "#a5d76e", "#e05c8a", "#c9a227")


@dataclass
class TraceLine:
    """One line on the water, with a name and a colour."""

    points: np.ndarray                 # (N, 2) east/north
    label: str = ""
    colour: Optional[str] = None
    width: float = 2.0
    style: str = "-"


@dataclass
class RaceScene:
    """Everything needed to draw a race, and nothing course-specific."""

    name: str
    #: ``(east, north, mask)`` from whatever built the water.
    east: np.ndarray
    north: np.ndarray
    water: np.ndarray
    #: Lines to draw, first is treated as the reference.
    lines: List[TraceLine] = field(default_factory=list)
    #: Polylines to draw as barriers -- docks, piers, booms.
    obstructions: Sequence[np.ndarray] = ()
    #: ``(polygons, heights)`` for the oblique view.
    structures: Optional[object] = None
    #: Depth lookup ``(east, north) -> m``; enables the profile view.
    depth_at: Optional[object] = None
    #: Boat speed and length, for the critical-depth marker.
    speed: float = 3.9
    boat_length: float = 13.4
    #: Marks worth labelling: ``(east, north, text)``.
    marks: Sequence[tuple] = ()

    def critical_depth(self) -> float:
        """Depth at which ``Fr_h`` reaches one for this boat's speed."""
        return float(self.speed ** 2 / GRAVITY)


def _style(axis):
    axis.set_facecolor(DRY)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def draw_plan(scene, axis):
    _style(axis)
    axis.pcolormesh(scene.east, scene.north,
                    np.where(scene.water, 1.0, np.nan),
                    cmap="Blues_r", vmin=0.0, vmax=2.0, shading="auto")
    for points in scene.obstructions:
        points = np.asarray(points)
        if len(points) > 1:
            axis.plot(points[:, 0], points[:, 1], color=OBSTRUCTION,
                      linewidth=0.7, alpha=0.9, solid_capstyle="round")
    for index, line in enumerate(scene.lines):
        axis.plot(line.points[:, 0], line.points[:, 1],
                  color=line.colour or PALETTE[index % len(PALETTE)],
                  linewidth=line.width, linestyle=line.style,
                  label=line.label or None)
    for east, north, text in scene.marks:
        axis.plot([east], [north], "o", color=INK, markersize=5)
        axis.annotate(text, (east, north), color=INK, fontsize=8,
                      xytext=(8, 8), textcoords="offset points")
    if any(line.label for line in scene.lines):
        axis.legend(loc="lower left", fontsize=8, facecolor="#111a20",
                    edgecolor="#2a3640", labelcolor=INK)
    axis.set_title("%s -- plan" % scene.name, color=INK, fontsize=11)


def draw_profile(scene, axis):
    """Depth under each line, against the boat's critical depth."""
    axis.set_facecolor(DRY)
    critical = scene.critical_depth()
    for index, line in enumerate(scene.lines):
        step = np.hypot(*np.diff(line.points, axis=0).T)
        along = np.concatenate([[0.0], np.cumsum(step)])
        depth = np.array([float(scene.depth_at(p[0], p[1]))
                          for p in line.points])
        axis.plot(along, depth,
                  color=line.colour or PALETTE[index % len(PALETTE)],
                  linewidth=1.6, label=line.label or None)
    axis.axhline(critical, color="#e05c8a", linestyle="--", linewidth=1.4)
    axis.annotate("critical depth, $Fr_h=1$ at %.1f m/s (%.2f m)"
                  % (scene.speed, critical),
                  (0.01, critical), xycoords=("axes fraction", "data"),
                  color="#e05c8a", fontsize=8,
                  xytext=(0, 5), textcoords="offset points")
    axis.invert_yaxis()
    axis.set_xlabel("distance along the line, m", color=MUTED, fontsize=9)
    axis.set_ylabel("depth, m", color=MUTED, fontsize=9)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#2a3640")
    if any(line.label for line in scene.lines):
        axis.legend(loc="lower right", fontsize=8, facecolor="#111a20",
                    edgecolor="#2a3640", labelcolor=INK)
    axis.set_title("%s -- depth under the line" % scene.name, color=INK,
                   fontsize=11)


def draw_oblique(scene, axis, limit=900):
    axis.set_facecolor(DRY)
    grid_east, grid_north = np.meshgrid(scene.east, scene.north)
    axis.plot_surface(grid_east, grid_north,
                      np.where(scene.water, 0.0, np.nan),
                      color=WATER, alpha=0.85, linewidth=0, shade=False,
                      rcount=110, ccount=110)
    if scene.structures is not None:
        heights = np.asarray(scene.structures.heights, dtype=float)
        order = np.argsort(-heights)
        drawn = 0
        for index in order:
            if drawn >= limit:
                break
            polygon = np.asarray(scene.structures.polygons[index])
            if len(polygon) < 3:
                continue
            height = float(heights[index])
            axis.plot(polygon[:, 0], polygon[:, 1],
                      np.full(len(polygon), height),
                      color=STRUCTURE, linewidth=0.5,
                      alpha=min(0.25 + height / 200.0, 0.95))
            drawn += 1
    for index, line in enumerate(scene.lines):
        axis.plot(line.points[:, 0], line.points[:, 1],
                  np.zeros(len(line.points)),
                  color=line.colour or PALETTE[index % len(PALETTE)],
                  linewidth=2.2)
    span_e = float(scene.east[-1] - scene.east[0])
    span_n = float(scene.north[-1] - scene.north[0])
    longest = max(span_e, span_n)
    axis.set_box_aspect((span_e / longest * 2.4, span_n / longest * 2.4,
                         0.7))
    axis.view_init(elev=22, azim=-118)
    axis.set_axis_off()
    axis.set_title("%s -- oblique" % scene.name, color=INK, fontsize=11)


#: Effective eye height for the forward view, m.  **Deliberately not the
#: real 0.7 m.**  A coxswain's actual eye is so close to the water that
#: everything beyond a boat length sits within four degrees of the
#: horizon: drawn to scale the entire course is one line of pixels.
#: Raising the viewpoint spreads the geometry out enough to read, at the
#: cost of the view being a diagram rather than a photograph.
COX_EYE = 9.0
#: Nearest distance drawn, m.  Must satisfy ``COX_EYE / COX_NEAR <= the
#: top of the frame``, or near geometry projects off the top -- which is
#: exactly what a mismatched pair of these numbers did: a 3 m near clip
#: against a frame that could only show beyond 21 m sent the racing line,
#: which passes under the boat, straight out of the picture.
COX_NEAR = 12.0
#: Half the horizontal field of view, as a tangent.  1.0 is 90 degrees.
COX_FOV = 1.0
#: Farthest water drawn, m.  Beyond this it is a pixel on the horizon.
COX_FAR = 900.0


def _cox_project(points, here, forward, side, close=False):
    """Pinhole projection onto the forward view, with the frame clipped.

    Returns ``(x, y, distance)`` with ``nan`` inserted wherever a polyline
    leaves the view, so matplotlib breaks the line instead of drawing a
    chord across the picture.  Without that, a dock a few metres ahead and
    fifty to the side is joined to whatever came next by a long horizontal
    streak.
    """
    delta = np.asarray(points, dtype=float) - here
    along = delta @ forward
    across = delta @ side
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(along > 0, across / np.maximum(along, 1e-9), np.nan)
        y = np.where(along > 0, COX_EYE / np.maximum(along, 1e-9), np.nan)
    outside = (along < COX_NEAR) | (np.abs(x) > COX_FOV * 1.3)
    x = np.where(outside, np.nan, x)
    y = np.where(outside, np.nan, y)
    return x, y, along


def _cox_water(scene, here, forward, side, top, width=420, height=260):
    """The water as the coxswain sees it, by inverting the projection.

    For each pixel: ``along = COX_EYE / y`` and ``across = x * along``
    give the patch of water that pixel looks at, which is then tested
    against the mask.  Shaded by distance so the eye reads depth.
    """
    xs = np.linspace(-COX_FOV, COX_FOV, width)
    # Row 0 must be the FAR end: with ``origin="upper"`` imshow puts the
    # first row at the top of the extent, which is the horizon.  Running
    # this the other way drew the near water along the skyline and the
    # far water under the bow -- a picture that looked like sky.
    ys = np.linspace(COX_EYE / COX_FAR, top, height)
    gx, gy = np.meshgrid(xs, ys)
    along = COX_EYE / np.maximum(gy, 1e-9)
    across = gx * along
    east = here[0] + forward[0] * along + side[0] * across
    north = here[1] + forward[1] * along + side[1] * across
    column = np.searchsorted(scene.east, east)
    row = np.searchsorted(scene.north, north)
    inside = ((column > 0) & (column < len(scene.east))
              & (row > 0) & (row < len(scene.north)))
    image = np.full(gx.shape, np.nan)
    wet = np.zeros(gx.shape, dtype=bool)
    wet[inside] = scene.water[row[inside] - 1, column[inside] - 1]
    # nearer water darker, so distance reads
    image[wet] = np.clip(np.log10(along[wet]) / 1.6, 0.0, 1.0) * 2.0
    return image


def draw_cox_view(scene, axes, fractions=(0.05, 0.4, 0.75)):
    """Forward view from the stern at several points down the course.

    The viewpoint is the **last** line, which is the one being flown --
    the same convention :func:`write_video` uses.  Sitting on the first
    line instead puts the camera on the drawn course while the optimised
    line runs thirty metres off to one side, so the line the picture is
    about leaves the frame within a boat length of the bow.
    """
    reference = scene.lines[-1].points
    grid_east, grid_north = np.meshgrid(scene.east, scene.north)
    wet = np.column_stack([grid_east[scene.water], grid_north[scene.water]])
    top = COX_EYE / COX_NEAR

    for panel, fraction in zip(axes, fractions):
        index = int(fraction * (len(reference) - 1))
        here = reference[index]
        ahead = reference[min(index + 6, len(reference) - 1)]
        heading = np.arctan2(ahead[1] - here[1], ahead[0] - here[0])
        forward = np.array([np.cos(heading), np.sin(heading)])
        side = np.array([-forward[1], forward[0]])

        panel.set_facecolor("#0a1218")
        # the horizon, so the view has a reference the eye can hold
        panel.axhline(0.0, color="#2a3640", linewidth=0.8)

        # **Sample in SCREEN space, not world space.**  Near water
        # subtends a wide angle, so a world-space grid puts only two or
        # three cells inside the frame and the whole foreground comes out
        # empty while everything distant piles up on the horizon.
        # Inverting the projection instead -- pick a pixel, work out which
        # patch of water it looks at -- fills the view evenly by
        # construction.
        panel.imshow(_cox_water(scene, here, forward, side, top),
                     extent=(-COX_FOV, COX_FOV, top, 0.0),
                     origin="upper", aspect="auto", cmap="Blues_r",
                     vmin=0.0, vmax=2.2, interpolation="nearest")

        for points in scene.obstructions:
            points = np.asarray(points)
            if len(points) < 2 or np.abs(points - here).max() > 900:
                continue
            ox, oy, _d = _cox_project(points, here, forward, side)
            if np.isfinite(ox).any():
                panel.plot(ox, oy, color=OBSTRUCTION, linewidth=1.1,
                           alpha=0.9)

        for order, line in enumerate(scene.lines):
            lx, ly, _d = _cox_project(line.points, here, forward, side)
            panel.plot(lx, ly,
                       color=line.colour or PALETTE[order % len(PALETTE)],
                       linewidth=2.0)

        panel.set_xlim(-COX_FOV, COX_FOV)
        panel.set_ylim(top, 0.0)          # horizon at the top, near water low
        panel.set_xticks([])
        panel.set_yticks([])
        panel.set_title("%.0f%% along" % (100 * fraction), color=MUTED,
                        fontsize=9)


def render_all(scene, out_dir, dpi=150):
    """Write every view this scene can support. Returns the paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written = []
    slug = "".join(c if c.isalnum() else "_" for c in scene.name.lower())

    figure, axis = plt.subplots(figsize=(7.5, 9.0))
    draw_plan(scene, axis)
    figure.patch.set_facecolor(DRY)
    figure.tight_layout()
    path = os.path.join(out_dir, "%s_plan.png" % slug)
    figure.savefig(path, dpi=dpi, facecolor=DRY)
    plt.close(figure)
    written.append(path)

    if scene.depth_at is not None:
        figure, axis = plt.subplots(figsize=(11.0, 4.0))
        draw_profile(scene, axis)
        figure.patch.set_facecolor(DRY)
        figure.tight_layout()
        path = os.path.join(out_dir, "%s_depth.png" % slug)
        figure.savefig(path, dpi=dpi, facecolor=DRY)
        plt.close(figure)
        written.append(path)

    figure = plt.figure(figsize=(12.0, 7.0))
    axis = figure.add_subplot(111, projection="3d")
    draw_oblique(scene, axis)
    figure.patch.set_facecolor(DRY)
    path = os.path.join(out_dir, "%s_oblique.png" % slug)
    figure.savefig(path, dpi=dpi, facecolor=DRY, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    draw_cox_view(scene, axes)
    figure.suptitle("%s -- from the coxswain's seat" % scene.name,
                    color=INK, fontsize=11)
    figure.patch.set_facecolor(DRY)
    figure.tight_layout()
    path = os.path.join(out_dir, "%s_cox.png" % slug)
    figure.savefig(path, dpi=dpi, facecolor=DRY)
    plt.close(figure)
    written.append(path)

    return written


def write_video(scene, path, frames=240, fps=24, dpi=110, trail=90):
    """Fly the boat down the reference line, plan view beside cox view.

    Deliberately matplotlib rather than the VTK scene the Charles uses:
    :mod:`coxswain.viz.scene3d` is wired to bridge gates and arch rules,
    which is exactly the course-specific machinery this renderer exists to
    avoid. A race that has buoys instead of bridges needs a renderer that
    does not know what a bridge is.

    Writes mp4 where ``imageio-ffmpeg`` is available and falls back to GIF
    rather than failing after every frame has been drawn.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    if not scene.lines:
        raise ValueError("nothing to fly down: scene has no lines")
    line = scene.lines[-1].points
    step = np.hypot(*np.diff(line, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(step)])
    marks = np.linspace(0.0, along[-1], frames)
    track = np.column_stack([np.interp(marks, along, line[:, 0]),
                             np.interp(marks, along, line[:, 1])])

    images = []
    for index in range(frames):
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.2))
        _style(axes[0])
        axes[0].pcolormesh(scene.east, scene.north,
                           np.where(scene.water, 1.0, np.nan),
                           cmap="Blues_r", vmin=0.0, vmax=2.0,
                           shading="auto")
        for polygon in scene.obstructions:
            axes[0].plot(polygon[:, 0], polygon[:, 1], color=OBSTRUCTION,
                         linewidth=0.5, alpha=0.85)
        axes[0].plot(line[:, 0], line[:, 1], color=PALETTE[0], linewidth=1.0,
                     alpha=0.35)
        start = max(0, index - trail)
        axes[0].plot(track[start:index + 1, 0], track[start:index + 1, 1],
                     color=PALETTE[0], linewidth=2.4)
        axes[0].plot(track[index, 0], track[index, 1], "o", color="white",
                     markersize=6)
        axes[0].set_title("%s -- %.0f m of %.0f"
                          % (scene.name, marks[index], along[-1]),
                          color=INK, fontsize=9)

        ahead = min(index + 3, frames - 1)
        heading = np.arctan2(track[ahead, 1] - track[index, 1],
                             track[ahead, 0] - track[index, 0])
        forward = np.array([np.cos(heading), np.sin(heading)])
        side = np.array([-forward[1], forward[0]])
        here = track[index]

        def project(points, _here=here, _f=forward, _s=side):
            delta = np.asarray(points, dtype=float) - _here
            depth = delta @ _f
            lateral = delta @ _s
            keep = depth > 4.0
            return lateral[keep] / depth[keep], 1.0 / depth[keep]

        panel = axes[1]
        panel.set_facecolor("#0a1218")
        grid_east, grid_north = np.meshgrid(scene.east, scene.north)
        wet = np.column_stack([grid_east[scene.water],
                               grid_north[scene.water]])
        x, y = project(wet)
        panel.scatter(x, y * 30.0, s=1.2, color="#2a5f80", alpha=0.5,
                      linewidths=0)
        for polygon in scene.obstructions:
            if np.abs(polygon - here).max() > 800:
                continue
            px, py = project(polygon)
            if len(px):
                panel.plot(px, py * 30.0, color=OBSTRUCTION, linewidth=1.0,
                           alpha=0.9)
        lx, ly = project(line)
        panel.plot(lx, ly * 30.0, color=PALETTE[0], linewidth=2.0)
        panel.set_xlim(-1.1, 1.1)
        panel.set_ylim(0.0, 1.4)
        panel.set_xticks([])
        panel.set_yticks([])
        panel.set_title("from the coxswain's seat", color=INK, fontsize=9)

        figure.patch.set_facecolor(DRY)
        figure.tight_layout()
        figure.canvas.draw()
        images.append(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(figure)

    try:
        imageio.mimsave(path, images, fps=fps)
    except Exception:
        path = os.path.splitext(path)[0] + ".gif"
        imageio.mimsave(path, images, duration=1.0 / fps)
    return path
