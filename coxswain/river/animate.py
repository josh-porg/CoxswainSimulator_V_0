"""Animate a simulated boat rowing the river.

A coxswain reads a chart differently from an engineer.  A time history of
yaw rate against station is the right way to check a controller and the
wrong way to show someone a race: what a coxswain wants to see is the boat
in the water, with the bank where the bank is, the arches where the arches
are, and the line it is trying to hold drawn ahead of it.

So this draws the boat to scale -- 17.3 m of it, which at the width of the
river is a real object and not a dot -- against the surveyed bank, the
bridge arches with their piers, and the boathouses and Harvard houses that
a crew actually steers by.

Two views, because they answer different questions:

**Chase**
    The camera follows the boat at a fixed scale, so the river comes at
    you the way it does from the stern.  This is the view for judging
    whether a line looks rowable.

**Course**
    The whole reach, with the boat as a moving mark.  This is the view for
    seeing where the line sits overall.

The animation is written with :class:`matplotlib.animation.FuncAnimation`
and saved through whichever writer is available -- ffmpeg for mp4 if it is
installed, Pillow for gif otherwise, which needs nothing extra.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from . import bridges as _bridges
from . import charles as _charles

__all__ = ["BoatArtist", "animate_run", "write_animation"]

_INK, _MUTED, _RULE = "#16211f", "#5c6968", "#dce2e0"
_WATER = ["#dceaf2", "#b9d8e8", "#8ec6dd", "#5aa7cb"]
_HULL, _BLADE = "#f4f1e8", "#c8901a"
_GREEN, _WARN, _ACCENT = "#1f7a4d", "#a2382a", "#1f5673"


class BoatArtist(object):
    """An eight drawn to scale: hull, riggers and eight blades.

    Drawn as a real object rather than a marker.  At 17.3 m long and
    6.8 m across the blades, against a river 50 to 160 m wide, an eight
    takes up a seventh of the channel -- which is the fact a coxswain is
    working with and a dot on a map hides completely.
    """

    LENGTH = 17.3
    BEAM = 0.57
    OAR_REACH = 3.41          # oarlock 0.85 m out, blade tip 2.56 m beyond

    def __init__(self, ax, blades: bool = True):
        from matplotlib.patches import Polygon
        self.ax = ax
        self.blades = blades
        self.hull = Polygon(self._hull_shape(), closed=True,
                            facecolor=_HULL, edgecolor=_INK, lw=0.8,
                            zorder=20)
        ax.add_patch(self.hull)
        self.oars = []
        if blades:
            for _ in range(8):
                line, = ax.plot([], [], color=_BLADE, lw=1.4, zorder=19,
                                solid_capstyle="round")
                self.oars.append(line)

    def _hull_shape(self):
        """A slender hull outline, bow to stern."""
        half = self.LENGTH / 2.0
        x = np.array([half, half * 0.75, half * 0.2, -half * 0.4,
                      -half * 0.85, -half, -half * 0.85, -half * 0.4,
                      half * 0.2, half * 0.75])
        y = np.array([0.0, 0.22, 0.285, 0.27, 0.16, 0.0,
                      -0.16, -0.27, -0.285, -0.22])
        return np.column_stack([x, y])

    def update(self, position, heading, phase: float = 0.0):
        cos, sin = np.cos(heading), np.sin(heading)
        rotation = np.array([[cos, -sin], [sin, cos]])
        self.hull.set_xy(self._hull_shape() @ rotation.T + position[:2])

        if not self.blades:
            return
        # Oar angle through the stroke: catch forward, finish aft.  The
        # exact sweep matters less than that the blades move, which is
        # what makes the animation legible as rowing rather than sliding.
        sweep = np.radians(56.0) * np.cos(2.0 * np.pi * phase)
        seats = np.linspace(-4.3, 4.24, 8)
        for index, (line, station) in enumerate(zip(self.oars, seats)):
            side = 1.0 if index % 2 == 0 else -1.0
            pivot = np.array([station, side * 0.85])
            tip = pivot + np.array([np.sin(sweep) * self.OAR_REACH * -side,
                                    side * np.cos(sweep) * self.OAR_REACH])
            pair = np.stack([pivot, tip]) @ rotation.T + position[:2]
            line.set_data(pair[:, 0], pair[:, 1])


def _draw_scenery(ax, channel, gates, landmarks=True, labels=True):
    """Bank, arches, piers and the buildings a crew steers by."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    depth = np.array(channel.depth, dtype=float)
    depth[~channel.water] = np.nan
    ax.imshow(depth, origin="lower",
              extent=[channel.east[0], channel.east[-1],
                      channel.north[0], channel.north[-1]],
              cmap=LinearSegmentedColormap.from_list("w", _WATER),
              norm=Normalize(0.0, 7.0), interpolation="bilinear", zorder=1)
    ax.contour(channel.east, channel.north, channel.water.astype(float),
               levels=[0.5], colors=["#8a7f6a"], linewidths=1.4, zorder=3)

    for gate, _metres in gates:
        ax.plot([gate.start[0], gate.end[0]], [gate.start[1], gate.end[1]],
                color="#6b6257", lw=5.0, solid_capstyle="butt", zorder=6,
                alpha=0.85)
        for arch in _bridges.bridge_arches(gate, channel):
            a, b = arch.interval
            p, q = gate.point_at(a), gate.point_at(b)
            ax.plot([p[0], q[0]], [p[1], q[1]],
                    color=_GREEN if arch.legal else _WARN, lw=5.0,
                    solid_capstyle="butt", zorder=7)
        for pier in gate.piers:
            a, b = pier.interval
            p, q = gate.point_at(a), gate.point_at(b)
            ax.plot([p[0], q[0]], [p[1], q[1]], color=_INK, lw=7.0,
                    solid_capstyle="butt", zorder=8)

    if landmarks:
        from .course import local_tangent_plane
        for name, latlon in _charles.LANDMARKS:
            east, north = local_tangent_plane(np.array([latlon[0]]),
                                              np.array([latlon[1]]),
                                              _charles.CHARLES_ORIGIN)
            ax.plot([east[0]], [north[0]], marker="s", ms=7,
                    color="#8a7f6a", markeredgecolor=_INK,
                    markeredgewidth=0.7, zorder=9)
            if labels:
                ax.annotate(name, (east[0], north[0]), fontsize=7,
                            color=_INK, zorder=10,
                            textcoords="offset points", xytext=(8, 4))


def animate_run(positions, headings, path, channel, gates, path_length=None,
                view: str = "chase", span: float = 260.0, stride: int = 1,
                period: float = 2.0, times=None, cross_track=None):
    """Build the figure and the frame updater; returns ``(fig, update, n)``."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "axes.edgecolor": _RULE, "text.color": _INK,
                         "xtick.color": _MUTED, "ytick.color": _MUTED,
                         "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    positions = np.asarray(positions, dtype=float)
    headings = np.asarray(headings, dtype=float)
    frames = np.arange(0, len(positions), max(int(stride), 1))

    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=80)
    _draw_scenery(ax, channel, gates, labels=(view != "chase"))
    ax.plot(path[:, 0], path[:, 1], color="#6b3fa0", lw=1.6, ls="--",
            zorder=5, alpha=0.9, label="planned line")
    trail, = ax.plot([], [], color="#6b3fa0", lw=2.4, zorder=15, alpha=0.55,
                     label="rowed")
    boat = BoatArtist(ax, blades=(view == "chase"))
    caption = ax.text(0.015, 0.965, "", transform=ax.transAxes, fontsize=9,
                      va="top", color=_INK, zorder=30,
                      bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                ec=_RULE, alpha=0.93))
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.93,
              edgecolor=_RULE)
    if view != "chase":
        pad = 150.0
        ax.set_xlim(path[:, 0].min() - pad, path[:, 0].max() + pad)
        ax.set_ylim(path[:, 1].min() - pad, path[:, 1].max() + pad)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")

    def update(frame_index):
        i = frames[frame_index]
        point, heading = positions[i], headings[i]
        seconds = float(times[i]) if times is not None else i * 0.01
        boat.update(point, heading, phase=(seconds / period) % 1.0)
        trail.set_data(positions[:i + 1, 0], positions[:i + 1, 1])
        if view == "chase":
            ax.set_xlim(point[0] - span / 2, point[0] + span / 2)
            ax.set_ylim(point[1] - span / 2, point[1] + span / 2)
        text = "%4.0f s" % seconds
        if path_length is not None:
            text += "    %4.0f m to go" % max(path_length - _along(i), 0.0)
        if cross_track is not None:
            text += "    %+.1f m off the line" % cross_track[i]
        caption.set_text(text)
        return ()

    travelled = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(positions, axis=0), axis=1))])

    def _along(i):
        return float(travelled[min(i, len(travelled) - 1)])

    return fig, update, len(frames)


def write_animation(fig, update, frames, path, fps: int = 25):
    """Save with whatever writer is installed; returns the path written."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    movie = animation.FuncAnimation(fig, update, frames=frames, blit=False,
                                    interval=1000.0 / fps)
    root, extension = os.path.splitext(path)
    if animation.writers.is_available("ffmpeg"):
        target = root + ".mp4"
        movie.save(target, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    else:
        target = root + ".gif"
        # A GIF carries every frame whole, so a large figure at many
        # frames runs to tens of megabytes.  Keep it modest.
        movie.save(target, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return target
