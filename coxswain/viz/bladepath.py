r"""The blade's path through the water, in the inertial frame.

Top-down, camera fixed to the water rather than to the boat: the water is
still and the boat runs through it, so the blade traces the characteristic
loop.  Arrows are the blade's normal load at stations through the drive, and
four events are marked -- the catch, where the blade starts to slip, where it
re-anchors, and the finish.

Two sources, one picture
------------------------
:func:`dynamic_trace` draws what a :class:`~coxswain.sim.dynamic_oar.
DynamicOarSimulator` boat actually did: the integrated oar angle and rate,
the hull's own position, heading and surge swing, and the load the water put
on the blade.  That is the figure the programme was asked for, and it can only
be drawn now that the oar angle is a state.

:func:`prescribed_trace` draws the old schedule -- the sweep as a function of
stroke phase at a constant boat speed -- so the two can sit side by side.

What the figure cannot show
---------------------------
**The tangential component.**  [CR06] Model 1 is a pure normal force, so every
arrow is normal to the shaft by construction.  Grift et al. (2021) measured the
tangential part on a real blade and it is not small.  Stated on the figure, not
left to be discovered: it is exactly what tier 2 adds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["BladeTrace", "dynamic_trace", "prescribed_trace", "plot"]


@dataclass
class BladeTrace:
    """One drive's blade path, loads and events, ready to draw."""

    label: str
    #: Blade centre in the inertial frame over the DRIVE, metres, measured
    #: from the catch position.
    x: np.ndarray
    y: np.ndarray
    #: Normal load on the blade, N, signed so positive drives the boat.
    load: np.ndarray
    #: Direction of that load in the inertial frame, unit vectors ``(n, 2)``.
    direction: np.ndarray
    #: Blade slip on its normal, m/s. Negative: the blade is being driven
    #: through the water faster than the boat carries it.
    slip: np.ndarray
    #: Mean boat speed over the stroke, m/s.
    speed: float
    #: Sample indices of the two slip sign changes, catch side first.
    crossings: List[int] = field(default_factory=list)
    #: Whether the oar reached its finish angle within the stroke.  A drive
    #: that runs out of stroke before the finish is drawn to where it got,
    #: and its last point is NOT labelled the finish.
    finished: bool = True

    @property
    def run_back(self) -> float:
        """How far the blade travelled sternward through the water, metres."""
        return float(max(0.0, self.x[0] - self.x.min()))

    @property
    def anchored(self) -> float:
        """Fraction of the drive with the blade not over-driven (slip > 0)."""
        return float(np.mean(self.slip > 0.0)) if self.slip.size else 0.0


def _crossings(slip: np.ndarray) -> List[int]:
    signs = np.sign(slip)
    return [int(i) for i in np.flatnonzero(signs[:-1] * signs[1:] < 0)]


def dynamic_trace(sim, run, label: str, seat_slot: int = 0,
                  lock_index: int = 0) -> BladeTrace:
    """The blade path from a dynamic-oar run's kept last stroke.

    ``run`` is the :class:`~coxswain.sim.dynamic_oar.DynamicRun` returned by
    ``sim.run_strokes``; its ``last_states`` carry the hull and oar states.
    """
    from ..core.frames import hull_to_abs
    from ..core.state import STATE_SIZE, State

    states = getattr(run, "last_states", None)
    if states is None:
        raise ValueError("this run did not keep its last stroke's states")

    n = sim.n_oar_states
    oar = sim._oars[seat_slot]
    seat = sim._seats[seat_slot]
    lock = sim.boat.rig.seats[seat].oarlocks[lock_index]
    side = int(lock.side)
    lock_position = np.asarray(lock.position, dtype=float)

    angles = states[STATE_SIZE + seat_slot]
    rates = states[STATE_SIZE + n + seat_slot]
    drive = np.flatnonzero(angles > oar.finish_angle)
    if drive.size < 3:
        raise ValueError("the kept stroke has no drive to draw")
    finished = bool(np.any(angles <= oar.finish_angle))

    xs, ys, loads, directions, slips = [], [], [], [], []
    for k in drive:
        state = State.from_vector(states[:STATE_SIZE, k])
        angle, rate = float(angles[k]), float(rates[k])
        rotation = hull_to_abs(state.attitude)
        axis = np.array([np.sin(angle), side * np.cos(angle), 0.0])
        normal = np.array([np.cos(angle), -side * np.sin(angle), 0.0])
        blade = np.asarray(state.position, dtype=float) + rotation @ (
            lock_position + oar.outboard * axis)
        speed = sim._lock_speed_on_normal(state, lock, angle)
        direction = (rotation @ normal)[:2]
        xs.append(blade[0])
        ys.append(blade[1])
        loads.append(float(oar.blade.normal_force(angle, rate, speed)))
        directions.append(direction / max(np.linalg.norm(direction), 1e-12))
        slips.append(float(oar.blade.slip_velocity(angle, rate, speed)))

    x = np.asarray(xs) - xs[0]
    y = np.asarray(ys) - ys[0]
    slip = np.asarray(slips)
    return BladeTrace(label=label, x=x, y=y, load=np.asarray(loads),
                      direction=np.asarray(directions), slip=slip,
                      speed=float(np.mean(run.last_speed)),
                      crossings=_crossings(slip), finished=finished)


def prescribed_trace(boat, speed: float, label: str,
                     samples: int = 600, side=None) -> BladeTrace:
    """The old schedule: the prescribed sweep at a constant boat speed.

    ``side`` is the oar's side, +1 port and -1 starboard, and defaults to the
    boat's first oarlock -- the same lock :func:`dynamic_trace` draws.  It
    used to be ignored, so every prescribed trace was a PORT oar: beside a
    starboard seat's dynamic trace the "comparison" panel was its mirror
    image, which is exactly the kind of difference a reader would take for
    physics.
    """
    from ..crew.oarlock import BladeModel

    timing = boat.timing
    lock = boat.rig.seats[0].oarlocks[0]
    outboard = float(lock.oar.outboard)
    side = int(lock.side) if side is None else int(side)
    blade = BladeModel.sweep(outboard=outboard)
    t = np.linspace(0.0, timing.drive_fraction * timing.period, int(samples))
    angle = np.asarray(boat.oar_sweep(t, timing), dtype=float)
    rate = np.asarray(boat.oar_sweep.rate(t, timing), dtype=float)

    x = float(speed) * t + outboard * np.sin(angle)
    y = side * outboard * np.cos(angle)
    slip = np.asarray(blade.slip_velocity(angle, rate, float(speed)))
    load = np.asarray(blade.normal_force(angle, rate, float(speed)))
    direction = np.stack([np.cos(angle), -side * np.sin(angle)], axis=1)
    return BladeTrace(label=label, x=x - x[0], y=y - y[0], load=load,
                      direction=direction, slip=slip, speed=float(speed),
                      crossings=_crossings(slip))


def plot(traces: Sequence[BladeTrace], path: str,
         title: Optional[str] = None, arrows: int = 8) -> str:
    """Draw one panel per trace and write a PNG to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#B45309", "#0F766E", "#4338CA", "#9D174D"]
    fig, axes = plt.subplots(1, len(traces), figsize=(4.5 * len(traces), 5.0),
                             squeeze=False)
    for ax, trace, colour in zip(axes[0], traces, colours * 4):
        ax.plot(trace.x, trace.y, color=colour, lw=3.0, solid_capstyle="round",
                zorder=3)
        peak = max(float(np.abs(trace.load).max()), 1e-9)
        span = max(float(np.ptp(trace.x)), float(np.ptp(trace.y)), 0.5)
        scale = 0.35 * span / peak
        picks = (np.linspace(0.05, 0.95, arrows) * (trace.x.size - 1)).astype(int)
        for k in picks:
            dx, dy = scale * trace.load[k] * trace.direction[k]
            ax.annotate("", xy=(trace.x[k] + dx, trace.y[k] + dy),
                        xytext=(trace.x[k], trace.y[k]),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2,
                                        color="#111827", alpha=0.85,
                                        shrinkA=0, shrinkB=0), zorder=5)

        def mark(k, text, offset):
            ax.plot(trace.x[k], trace.y[k], "o", ms=7, mfc="white",
                    mec=colour, mew=2.0, zorder=6)
            ax.annotate(text, (trace.x[k], trace.y[k]),
                        textcoords="offset points", xytext=offset,
                        fontsize=8.3, color="#111827",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="#D1D5DB", lw=0.7), zorder=7)

        mark(0, "catch", (-38, 10))
        mark(trace.x.size - 1,
             "finish" if trace.finished else "stroke ends before the finish",
             (8, -16))
        names = ["blade starts to slip", "blade re-anchors"]
        for j, k in enumerate(trace.crossings[:2]):
            mark(k, names[j], (9, 14 if j == 0 else -30))

        ax.set_title("%s\n%.2f m/s" % (trace.label, trace.speed), fontsize=10)
        ax.set_xlabel("along the course from the catch (m)", fontsize=8.5)
        ax.annotate("run back through the water %.2f m\nanchored %.0f%% of "
                    "the drive" % (trace.run_back, 100.0 * trace.anchored),
                    (0.5, 0.03), xycoords="axes fraction", ha="center",
                    fontsize=8, color="#374151",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#F9FAFB",
                              ec="#E5E7EB", lw=0.7))
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.22, lw=0.6)
    axes[0][0].set_ylabel("across the course (m)", fontsize=8.5)
    if title:
        fig.suptitle(title, fontsize=11.5)
    fig.text(0.5, 0.005, "Arrows: normal blade load, scaled per panel. The "
             "tangential component is zero by construction under [CR06] "
             "Model 1; Grift et al. (2021) measure it on a real blade.",
             ha="center", fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
