"""Charts of the Charles model: bathymetry, current, width, and the arches.

These exist so the river model can be *looked at*.  Every number in
:mod:`coxswain.river.charles` and :mod:`coxswain.river.bridges` is derived
from a survey, a gauge record or an inventory, and the quickest way to
catch one that has gone wrong is to draw it on the river and see whether it
lands where the river is.  That is not hypothetical: plotting the landmarks
is what turned up three bridges with wrong coordinates, and a fourth that
was 370 m out along the channel while still sitting only 6 m off it.

Five charts, each answering a different question:

``course_map``
    Where the course goes, over the bathymetry, with the bridges, their
    arches, and the start and finish lines.
``current_map``
    How fast the water is and where the thread of it runs, resolved across
    the width rather than averaged over the section.
``course_profiles``
    The course straightened out: depth across the channel, centreline
    depth, navigable width and current, all against distance from the
    start.  This is the one that shows where the river gets tight.
``arch_chart``
    Every bridge's arches to scale, with the legal ones marked and an
    eight drawn against them for comparison.
``span_map``
    The navigable spans drawn on the water, one panel per bridge, each
    rotated so the boat runs up the page.  The course map has to draw
    4.8 km, at which scale a 20 m arch is a hairline; this is the one to
    look at to see an opening against the river it sits in.

Matplotlib is imported lazily inside each function, so importing this
module costs nothing if you are not drawing.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from . import bridges as _bridges
from . import charles as _charles

__all__ = ["course_map", "current_map", "course_profiles", "arch_chart",
           "span_map", "write_all", "CourseGeometry", "BUILDERS",
           "CHART_FILENAMES"]


#: What :func:`write_all` writes, in the order it writes them.
CHART_FILENAMES = ("charles_course_bathymetry.png",
                   "charles_course_current.png",
                   "charles_course_profiles.png",
                   "charles_bridge_arches.png",
                   "charles_navigable_spans.png")

_WATER = ["#d9edf5", "#a9d6e8", "#6fb4d4", "#3d8cb8", "#22608f", "#123a5c"]
_FLOW = ["#eef3f5", "#cfe3d8", "#a8d3a0", "#e3d270", "#e09a4a", "#b0472c"]
_INK, _MUTED, _RULE = "#16211f", "#5c6968", "#dce2e0"
_ACCENT, _WARN, _GOLD = "#1f5673", "#a2382a", "#c8901a"
_GREEN = "#1f7a4d"


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.edgecolor": _RULE,
        "axes.labelcolor": _INK, "text.color": _INK,
        "xtick.color": _MUTED, "ytick.color": _MUTED,
        "axes.titlesize": 12, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def _cmap(name, colours):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colours)


class CourseGeometry(object):
    """Everything the charts share, worked out once.

    Building the channel raster and the bridge gates is the expensive part
    of drawing any of these, and every chart wants the same ones, so they
    are gathered here and passed around rather than rebuilt per chart.
    """

    def __init__(self, channel=None, month: int = 10):
        self.channel = (_charles.charles_channel() if channel is None
                        else channel)
        (self.start_xy, self.finish_xy, self.line,
         (self.start_station, self.finish_station)) = _charles.hocr_course(
            self.channel)
        self.distance = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(self.line, axis=0), axis=1))])
        self.gates = _bridges.build_gates(channel=self.channel)
        self.month = month
        self.course = _charles.charles_course(month=month)
        self.flow = _charles.ContinuityFlow(
            self.course, discharge=_charles.monthly_discharge(month))

    @property
    def length(self) -> float:
        return float(self.distance[-1])

    def index_at(self, metres) -> int:
        return int(np.argmin(np.abs(self.distance - metres)))

    def normal_at(self, index) -> np.ndarray:
        a = max(index - 1, 0)
        b = min(index + 1, len(self.line) - 1)
        tangent = self.line[b] - self.line[a]
        tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
        return np.array([-tangent[1], tangent[0]])

    def gate_distance(self, gate) -> float:
        """Distance from the start line to where a gate crosses the course."""
        middle = 0.5 * (gate.start + gate.end)
        index = int(np.argmin(np.linalg.norm(self.line - middle, axis=1)))
        return float(self.distance[index])

    def gates_on_course(self):
        """Gates between the start and the finish, in racing order."""
        found = [(gate, self.gate_distance(gate)) for gate in self.gates]
        found = [pair for pair in found if 0.0 <= pair[1] <= self.length]
        found.sort(key=lambda pair: pair[1])
        return found

    def depth_masked(self):
        depth = np.array(self.channel.depth, dtype=float)
        depth[~self.channel.water] = np.nan
        return depth

    def extent(self):
        return [self.channel.east[0], self.channel.east[-1],
                self.channel.north[0], self.channel.north[-1]]

    def frame(self, pad: float = 180.0):
        return (self.line[:, 0].min() - pad, self.line[:, 0].max() + pad,
                self.line[:, 1].min() - pad, self.line[:, 1].max() + pad)


def _draw_gate_line(ax, geometry, xy, colour, label, half=95.0):
    index = int(np.argmin(np.linalg.norm(geometry.line - xy, axis=1)))
    normal = geometry.normal_at(index)
    a, b = xy - normal * half, xy + normal * half
    ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, lw=3.0,
            solid_capstyle="butt", zorder=8)
    ax.annotate(label, xy, textcoords="offset points", xytext=(0, -26),
                ha="center", fontsize=10, fontweight="bold", color=colour,
                zorder=9)


def _draw_furniture(ax, geometry, labels=True, arches=True):
    """Channel edge, centreline, start, finish, bridges and their arches."""
    channel = geometry.channel
    ax.contour(channel.east, channel.north, channel.navigable.astype(float),
               levels=[0.5], colors=[_GOLD], linewidths=1.1, zorder=4)
    ax.plot(geometry.line[:, 0], geometry.line[:, 1], color="white", lw=2.0,
            zorder=5, alpha=0.85)
    _draw_gate_line(ax, geometry, geometry.start_xy, _GREEN, "START")
    _draw_gate_line(ax, geometry, geometry.finish_xy, _WARN, "FINISH")

    for gate, metres in geometry.gates_on_course():
        ax.plot([gate.start[0], gate.end[0]], [gate.start[1], gate.end[1]],
                color=_ACCENT, lw=1.6, alpha=0.55, zorder=6)
        if arches:
            racing = _bridges.racing_arch(gate, channel)
            for arch in _bridges.bridge_arches(gate, channel):
                low, high = arch.interval
                a, b = gate.point_at(low), gate.point_at(high)
                is_racing = racing is not None and arch.index == racing.index
                ax.plot([a[0], b[0]], [a[1], b[1]],
                        color=_GREEN if is_racing else
                        (_ACCENT if arch.legal else _WARN),
                        lw=4.0 if is_racing else 2.4,
                        solid_capstyle="butt", zorder=7)
            for pier in gate.piers:
                point = gate.point_at(pier.centre)
                ax.plot([point[0]], [point[1]], marker="s", ms=3.0,
                        color=_INK, zorder=8)
        if labels:
            middle = geometry.line[geometry.index_at(metres)]
            ax.annotate("%s\n%.0f m" % (gate.name, metres), middle,
                        textcoords="offset points", xytext=(0, 16),
                        ha="center", fontsize=8, color=_ACCENT, zorder=9,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="none", alpha=0.75))


def course_map(geometry=None, path: Optional[str] = None):
    """The course over the bathymetry, with bridges and arches."""
    plt = _style()
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    geometry = CourseGeometry() if geometry is None else geometry
    fig, ax = plt.subplots(figsize=(13.5, 9.0))
    image = ax.imshow(geometry.depth_masked(), origin="lower",
                      extent=geometry.extent(), cmap=_cmap("water", _WATER),
                      norm=Normalize(0.0, 6.0), interpolation="bilinear",
                      zorder=1)
    _draw_furniture(ax, geometry)

    for metres in range(1000, int(geometry.length), 1000):
        point = geometry.line[geometry.index_at(metres)]
        ax.plot([point[0]], [point[1]], marker="o", ms=5, color="white",
                markeredgecolor=_INK, markeredgewidth=0.8, zorder=8)
        ax.annotate("%d km" % (metres // 1000), point,
                    textcoords="offset points", xytext=(11, -11),
                    fontsize=7.5, color="white", zorder=9)

    x0, x1, y0, y1 = geometry.frame()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title("Head of the Charles course — bathymetry and arches\n"
                 "%.0f m from the DeWolfe Boathouse start" % geometry.length,
                 loc="left", pad=12)
    ax.set_xlabel("east of %.4f N, %.4f W  (m)" % _charles.CHARLES_ORIGIN)
    ax.set_ylabel("north (m)")
    bar = fig.colorbar(image, ax=ax, pad=0.01, fraction=0.03)
    bar.set_label("depth (m)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    ax.legend(handles=[
        Line2D([], [], color=_GOLD, lw=1.2, label="navigable boundary"),
        Line2D([], [], color="white", lw=2.0, label="channel centreline"),
        Line2D([], [], color=_GREEN, lw=4.0, label="racing arch"),
        Line2D([], [], color=_ACCENT, lw=2.4, label="legal arch"),
        Line2D([], [], color=_WARN, lw=2.4, label="arch: 60 s penalty"),
        Line2D([], [], color=_GREEN, lw=3.0, label="start line"),
        Line2D([], [], color=_WARN, lw=3.0, label="finish line"),
    ], loc="upper right", fontsize=8, framealpha=0.92, edgecolor=_RULE)
    fig.tight_layout()
    return _save(fig, path, CHART_FILENAMES[0])


def current_map(geometry=None, path: Optional[str] = None, lateral=True):
    """Current over the course, resolved across the width.

    A section-mean speed hides the thing that matters: the flow is not
    uniform across the river, it follows the deep water, and the variation
    across one cross-section is larger than the variation along the whole
    course.
    """
    plt = _style()
    from matplotlib.colors import LogNorm, Normalize

    geometry = CourseGeometry() if geometry is None else geometry
    flow, course = geometry.flow, geometry.course

    fig, ax = plt.subplots(figsize=(13.5, 9.0))
    ax.imshow(geometry.depth_masked(), origin="lower",
              extent=geometry.extent(),
              cmap=_cmap("ground", ["#f4f7f8", "#dfe8ec"]),
              norm=Normalize(0, 6), interpolation="bilinear", zorder=1)

    points, speeds = [], []
    for station in np.linspace(0.0, course.length, 220):
        if lateral and hasattr(flow, "lateral_profile"):
            offsets, _depth, local = flow.lateral_profile(float(station))
            offsets = np.asarray(offsets, dtype=float)
            points.append(course.offset_position(
                np.full(len(offsets), station), offsets))
            speeds.append(np.asarray(local, dtype=float))
        else:
            points.append(course.offset_position(np.array([station]),
                                                 np.array([0.0])))
            speeds.append(np.atleast_1d(flow.speed(station)))
    points = np.concatenate(points, axis=0)
    speeds = np.concatenate(speeds) * 1000.0

    finite = speeds[np.isfinite(speeds) & (speeds > 0)]
    scatter = ax.scatter(points[:, 0], points[:, 1], c=speeds,
                         cmap=_cmap("flow", _FLOW), s=7, zorder=6,
                         norm=LogNorm(max(finite.min(), 0.05), finite.max()))
    _draw_furniture(ax, geometry, labels=False, arches=False)

    x0, x1, y0, y1 = geometry.frame()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title("Head of the Charles course — current\n"
                 "continuity model, %s median discharge %.2f m³/s "
                 "(USGS Waltham)" % (_month_name(geometry.month),
                                     _charles.monthly_discharge(geometry.month)),
                 loc="left", pad=12)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    bar = fig.colorbar(scatter, ax=ax, pad=0.01, fraction=0.03)
    bar.set_label("depth-averaged current (mm/s)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    ax.text(0.015, 0.03,
            "%.1f–%.0f mm/s, up to %.1f× variation across a single "
            "section — against a racing 5 m/s this is slack water"
            % (finite.min(), finite.max(), finite.max() / finite.min()),
            transform=ax.transAxes, fontsize=9, color=_INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=_RULE))
    fig.tight_layout()
    return _save(fig, path, CHART_FILENAMES[1])


def course_profiles(geometry=None, path: Optional[str] = None,
                    half_width: float = 120.0, samples: int = 420):
    """The course straightened out, as four stacked profiles."""
    plt = _style()
    from matplotlib.colors import Normalize

    geometry = CourseGeometry() if geometry is None else geometry
    channel = geometry.channel
    offsets = np.linspace(-half_width, half_width, 121)
    along = np.linspace(0.0, geometry.length, samples)

    depth = np.full((len(offsets), len(along)), np.nan)
    navigable = np.zeros(depth.shape, dtype=bool)
    for j, metres in enumerate(along):
        index = geometry.index_at(metres)
        normal = geometry.normal_at(index)
        points = (geometry.line[index][None, :]
                  + offsets[:, None] * normal[None, :])
        for k, (east, north) in enumerate(points):
            try:
                row, column = channel.index_of(east, north)
            except Exception:
                continue
            if (0 <= row < channel.depth.shape[0]
                    and 0 <= column < channel.depth.shape[1]):
                if channel.water[row, column]:
                    depth[k, j] = channel.depth[row, column]
                navigable[k, j] = bool(channel.navigable[row, column])

    centre_depth = depth[len(offsets) // 2, :]
    width = np.array([np.ptp(offsets[navigable[:, j]])
                      if navigable[:, j].any() else 0.0
                      for j in range(len(along))])
    station = np.clip(geometry.start_station - along, 0.0,
                      geometry.course.length)
    speed = geometry.flow.speed(station) * 1000.0

    fig, axes = plt.subplots(4, 1, figsize=(14.5, 11.0), sharex=True,
                             gridspec_kw={"height_ratios": [3.0, 1.0, 1.0, 1.0],
                                          "hspace": 0.16})
    marks = geometry.gates_on_course()

    ax = axes[0]
    image = ax.pcolormesh(along, offsets, depth, cmap=_cmap("water", _WATER),
                          norm=Normalize(0.0, 5.0), shading="auto")
    ax.contour(along, offsets, navigable.astype(float), levels=[0.5],
               colors=[_GOLD], linewidths=1.2)
    ax.axhline(0.0, color="white", lw=1.4, alpha=0.85)
    for gate, metres in marks:
        ax.axvline(metres, color=_ACCENT, lw=1.6, alpha=0.9)
        ax.annotate(gate.name, (metres, half_width * 0.88), rotation=90,
                    fontsize=8, color=_ACCENT, ha="right", va="top")
    ax.axvline(0.0, color=_GREEN, lw=3.0)
    ax.axvline(geometry.length, color=_WARN, lw=3.0)
    ax.set_ylabel("offset from centreline (m)")
    ax.set_title("Head of the Charles — straightened course\n"
                 "depth across the channel; gold is the navigable boundary, "
                 "white the centreline", loc="left", pad=10)
    bar = fig.colorbar(image, ax=ax, pad=0.008, fraction=0.025)
    bar.set_label("depth (m)", fontsize=8)
    bar.ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.plot(along, centre_depth, color=_ACCENT, lw=1.6)
    ax.axhline(2.0, color=_WARN, ls=":", lw=1.0)
    ax.annotate("2 m", (60, 2.06), fontsize=7.5, color=_WARN)
    ax.set_ylabel("centreline\ndepth (m)")
    ax.set_ylim(0, max(6.0, float(np.nanmax(centre_depth)) * 1.1))

    ax = axes[2]
    ax.fill_between(along, 0, width, color="#8ec6dd", alpha=0.8)
    ax.plot(along, width, color=_ACCENT, lw=1.2)
    ax.axhline(2 * _bridges.EIGHT_ROWED_WIDTH, color=_WARN, ls=":", lw=1.0)
    ax.annotate("two eights abreast",
                (60, 2 * _bridges.EIGHT_ROWED_WIDTH + 3), fontsize=7.5,
                color=_WARN)
    ax.set_ylabel("navigable\nwidth (m)")

    ax = axes[3]
    ax.plot(along, speed, color="#b0472c", lw=1.6)
    ax.set_ylabel("current\n(mm/s)")
    ax.set_xlabel("distance from the start line (m)")
    ax.set_xlim(0, geometry.length)

    for ax in axes[1:]:
        for gate, metres in marks:
            ax.axvline(metres, color=_ACCENT, lw=0.8, alpha=0.4)

    fig.tight_layout()
    return _save(fig, path, CHART_FILENAMES[2])


def arch_chart(geometry=None, path: Optional[str] = None):
    """Every bridge's arches to scale, with an eight drawn for comparison."""
    plt = _style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    geometry = CourseGeometry() if geometry is None else geometry
    channel = geometry.channel
    marks = geometry.gates_on_course()
    boat = _bridges.EIGHT_ROWED_WIDTH

    fig, ax = plt.subplots(figsize=(13.0, 7.5))
    # Drawn as the coxswain meets them: first bridge at the bottom, the
    # boat running up the page, so Cambridge falls on the right of the
    # chart exactly as it falls to starboard in the boat.  Ordering these
    # the other way would put the starboard bank on the left of the page.
    for row, (gate, metres) in enumerate(marks):
        y = row + 1
        low, high = _bridges.waterway(gate, channel)
        middle = 0.5 * (low + high)
        racing = _bridges.racing_arch(gate, channel)
        ax.add_patch(Rectangle((low - middle, y - 0.30), high - low, 0.60,
                               facecolor="#eaf2f6", edgecolor="none",
                               zorder=1))
        for arch in _bridges.bridge_arches(gate, channel):
            a, b = arch.interval
            is_racing = racing is not None and arch.index == racing.index
            ax.add_patch(Rectangle(
                (a - middle, y - 0.30), b - a, 0.60,
                facecolor="#cfe6d8" if is_racing else
                ("#dbe7ee" if arch.legal else "#f2dedb"),
                edgecolor=_GREEN if is_racing else
                (_ACCENT if arch.legal else _WARN),
                lw=1.8 if is_racing else 1.0, zorder=2))
            ax.annotate("%.1f" % arch.width,
                        (0.5 * (a + b) - middle, y + 0.06), ha="center",
                        fontsize=8, color=_INK, zorder=4)
            if is_racing:
                ax.add_patch(Rectangle(
                    (0.5 * (a + b) - middle - 0.5 * boat, y - 0.13), boat,
                    0.26, facecolor=_GREEN, edgecolor="none", alpha=0.85,
                    zorder=3))
        for pier in gate.piers:
            a, b = pier.interval
            ax.add_patch(Rectangle((a - middle, y - 0.34), b - a, 0.68,
                                   facecolor=_INK, edgecolor="none",
                                   zorder=3))
        ax.annotate("%s\n%.0f m" % (gate.name, metres), (-95.0, y),
                    ha="left", va="center", fontsize=9, color=_INK)

    ax.set_xlim(-100.0, 75.0)
    ax.set_ylim(0.3, len(marks) + 0.9)
    ax.set_yticks([])
    ax.set_xlabel("metres from the middle of the opening   "
                  "(port / Boston shore left, starboard / Cambridge shore "
                  "right)")
    ax.set_title("Head of the Charles — the arches, to scale\n"
                 "as the coxswain meets them: first bridge at the bottom, "
                 "Cambridge to starboard. Dark blocks are piers; the green "
                 "bar is a rowed eight (%.2f m tip to tip)" % boat,
                 loc="left", pad=12)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color=_GREEN, lw=6, label="racing arch"),
        Line2D([], [], color=_ACCENT, lw=6, label="legal alternative"),
        Line2D([], [], color=_WARN, lw=6, label="60 s penalty"),
        Line2D([], [], color=_INK, lw=6, label="pier"),
    ], loc="lower right", fontsize=8, framealpha=0.95, edgecolor=_RULE)
    fig.tight_layout()
    return _save(fig, path, CHART_FILENAMES[3])


def span_map(geometry=None, path: Optional[str] = None, half: float = 110.0,
             samples: int = 260):
    """The navigable spans, drawn on the river, one panel per bridge.

    The whole-course map has to draw 4.8 km, at which scale a 20 m arch is
    a hairline and the thing that decides the race is invisible.  This
    draws each crossing at about 200 m across instead, over the real
    bathymetry, so the opening can be seen against the water it sits in.

    Every panel is rotated so the boat runs **up** the page.  That puts
    Cambridge on the right in each one, matching both the starboard hand
    in the boat and the arch chart, and it means the panels can be read
    against each other even though the river faces a different way at
    each bridge.

    The direction of travel is taken from the gate's own normal, signed by
    where the course goes next, rather than from the channel tangent: the
    tangent is unreliable at Eliot for the same reason it cannot tell the
    banks apart there.
    """
    plt = _style()
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    geometry = CourseGeometry() if geometry is None else geometry
    channel = geometry.channel
    marks = geometry.gates_on_course()

    columns = 4
    rows = int(np.ceil(len(marks) / float(columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(15.0, 5.0 * rows),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.16})
    axes = np.atleast_1d(axes).ravel()

    grid = np.linspace(-half, half, samples)
    across, along = np.meshgrid(grid, grid)

    for panel, (gate, metres) in enumerate(marks):
        ax = axes[panel]
        middle = 0.5 * (gate.start + gate.end)
        starboard = gate.direction                 # Boston -> Cambridge
        ahead = _course_heading(geometry, gate, metres)
        # Right-handed panel frame: +x to starboard, +y up the course.
        world = (middle[None, None, :]
                 + across[..., None] * starboard[None, None, :]
                 + along[..., None] * ahead[None, None, :])
        depth = _sample_depth(channel, world[..., 0], world[..., 1])
        navigable = _sample_navigable(channel, world[..., 0], world[..., 1])

        ax.imshow(depth, origin="lower", extent=[-half, half, -half, half],
                  cmap=_cmap("water", _WATER), norm=Normalize(0.0, 6.0),
                  interpolation="bilinear", zorder=1)
        ax.contour(grid, grid, navigable.astype(float), levels=[0.5],
                   colors=[_GOLD], linewidths=1.0, zorder=3)

        racing = _bridges.racing_arch(gate, channel)
        low, high = _bridges.waterway(gate, channel)
        offset = gate.station_of(middle)
        ax.plot([low - offset, high - offset], [0.0, 0.0], color=_INK,
                lw=1.0, alpha=0.35, zorder=4)
        for arch in _bridges.bridge_arches(gate, channel):
            a, b = arch.interval
            is_racing = racing is not None and arch.index == racing.index
            colour = (_GREEN if is_racing
                      else (_ACCENT if arch.legal else _WARN))
            ax.plot([a - offset, b - offset], [0.0, 0.0], color=colour,
                    lw=6.0 if is_racing else 3.5, solid_capstyle="butt",
                    zorder=6)
            # Six arches at the trestle are narrower than their own labels,
            # so the labels are stepped to keep them apart.
            ax.annotate("%.1f" % arch.width,
                        (0.5 * (a + b) - offset,
                         9.0 + 11.0 * (arch.index % 2)),
                        ha="center", fontsize=7.5, color=colour, zorder=8,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.7))
            if is_racing:
                boat = _bridges.EIGHT_ROWED_WIDTH
                ax.add_patch(Rectangle(
                    (0.5 * (a + b) - offset - 0.5 * boat, -half), boat,
                    2 * half, facecolor=_GREEN, alpha=0.13,
                    edgecolor="none", zorder=5))
        for pier in gate.piers:
            a, b = pier.interval
            ax.add_patch(Rectangle((a - offset, -9.0), b - a, 18.0,
                                   facecolor=_INK, edgecolor="none", zorder=6))

        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_title("%s\n%.0f m from the start" % (gate.name, metres),
                     fontsize=9.5, loc="left", pad=6)
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_yticks([])
        if panel // columns == rows - 1:
            ax.set_xlabel("metres across the opening", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.annotate("", xy=(-half + 18, -half + 44),
                    xytext=(-half + 18, -half + 14),
                    arrowprops=dict(arrowstyle="-|>", color=_INK, lw=1.2))
        ax.annotate("row", (-half + 22, -half + 26), fontsize=7,
                    color=_INK, va="center")

    for spare in range(len(marks), len(axes)):
        axes[spare].axis("off")

    handles = [
        Line2D([], [], color=_GREEN, lw=5, label="racing arch"),
        Line2D([], [], color=_ACCENT, lw=4, label="legal alternative"),
        Line2D([], [], color=_WARN, lw=4, label="60 s penalty"),
        Line2D([], [], color=_INK, lw=5, label="pier"),
        Line2D([], [], color=_GOLD, lw=1.2, label="navigable edge"),
    ]
    if len(marks) < len(axes):
        # Put the key in the empty panel rather than over a bridge.
        axes[len(marks)].legend(handles=handles, loc="center", fontsize=9,
                                frameon=False)
    else:
        axes[-1].legend(handles=handles, loc="lower right", fontsize=7,
                        framealpha=0.95, edgecolor=_RULE)

    fig.suptitle("Head of the Charles — the navigable spans on the river\n"
                 "each panel %.0f m across, rotated so the boat runs up the "
                 "page and Cambridge lies to starboard; numbers are clear "
                 "widths in metres" % (2 * half),
                 x=0.008, ha="left", fontsize=12, y=0.985)
    return _save(fig, path, CHART_FILENAMES[4])


def _course_heading(geometry, gate, metres) -> np.ndarray:
    """Unit vector along the course through ``gate``, in the racing sense."""
    ahead = geometry.line[geometry.index_at(min(metres + 120.0,
                                                geometry.length))]
    behind = geometry.line[geometry.index_at(max(metres - 120.0, 0.0))]
    forward = ahead - behind
    normal = gate.normal
    if float(np.dot(forward, normal)) < 0.0:
        normal = -normal
    return normal / max(np.linalg.norm(normal), 1e-9)


def _grid_index(axis, values):
    step = float(axis[1] - axis[0])
    index = np.round((values - float(axis[0])) / step).astype(int)
    return np.clip(index, 0, len(axis) - 1)


def _sample_depth(channel, east, north):
    depth = channel.depth[_grid_index(channel.north, north),
                          _grid_index(channel.east, east)].astype(float)
    wet = channel.water[_grid_index(channel.north, north),
                        _grid_index(channel.east, east)]
    return np.where(wet, depth, np.nan)


def _sample_navigable(channel, east, north):
    return channel.navigable[_grid_index(channel.north, north),
                             _grid_index(channel.east, east)]


#: Chart name to builder, for :func:`write_all` and the command line.
BUILDERS = {"bathymetry": course_map, "current": current_map,
            "profiles": course_profiles, "arches": arch_chart,
            "spans": span_map}


def write_all(directory: str = ".", month: int = 10,
              which: Optional[Sequence[str]] = None):
    """Draw every chart into ``directory``; returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")

    geometry = CourseGeometry(month=month)
    names = list(BUILDERS) if which is None else list(which)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    written = []
    for name in names:
        if name not in BUILDERS:
            raise ValueError("unknown chart %r; choose from %s"
                             % (name, ", ".join(BUILDERS)))
        written.append(BUILDERS[name](geometry, directory))
    return written


def _month_name(month: int) -> str:
    return ("January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November",
            "December")[int(month) - 1]


def _save(fig, path, default_name):
    import matplotlib.pyplot as plt
    if path is None:
        path = default_name
    elif os.path.isdir(path):
        path = os.path.join(path, default_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
