r"""The blade's path through the water, in the inertial frame.

Top-down, camera fixed to the water rather than to the boat: the water is
still and the boat runs through it, so the blade traces the characteristic
loop.  Arrows are the blade's load at stations through the drive, and four
events are marked -- the catch, where the blade starts to slip, where it
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

Both load components
--------------------
A tier 1 run -- [CR06] Model 1, the slip law -- has a normal load only, so its
tangential component is zero by construction and none is drawn.  A tier 2 run
-- lift and drag on the angle of attack -- has both, and both are drawn: the
normal load in the dark arrows, the tangential load along the shaft in red, on
one scale per panel so their relative size is real.  Tier 2 rests on
provisional coefficients from a secondary source ([CG06a] in SOURCES), and the
figure says so wherever it draws a tier 2 panel.  Grift et al. (2021) measured
the tangential load on a real blade; the tier 2 arrows are what to hold against
those traces once they can be obtained.
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
    #: Tangential load along the shaft, N, and its direction in the inertial
    #: frame.  ``None`` for a tier 1 trace, whose tangential load is zero by
    #: construction -- not zero by measurement.
    tangential: Optional[np.ndarray] = None
    tangential_direction: Optional[np.ndarray] = None

    @property
    def run_back(self) -> float:
        """How far the blade travelled sternward through the water, metres."""
        return float(max(0.0, self.x[0] - self.x.min()))

    @property
    def anchored(self) -> float:
        """Fraction of the drive with the blade not over-driven (slip > 0)."""
        return float(np.mean(self.slip > 0.0)) if self.slip.size else 0.0

    @property
    def has_tangential(self) -> bool:
        return self.tangential is not None


def _crossings(slip: np.ndarray) -> List[int]:
    signs = np.sign(slip)
    return [int(i) for i in np.flatnonzero(signs[:-1] * signs[1:] < 0)]


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def dynamic_trace(sim, run, label: str, seat_slot: int = 0,
                  lock_index: int = 0) -> BladeTrace:
    """The blade path from a dynamic-oar run's kept last stroke.

    ``run`` is the :class:`~coxswain.sim.dynamic_oar.DynamicRun` returned by
    ``sim.run_strokes``; its ``last_states`` carry the hull and oar states.
    The loads are the simulator's own, by whichever blade law it runs.
    """
    from ..core.frames import hull_to_abs
    from ..core.state import STATE_SIZE, State

    states = getattr(run, "last_states", None)
    if states is None:
        raise ValueError("this run did not keep its last stroke's states")
    law = getattr(sim, "blade_law", "slip")

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
    tangents, tangent_directions = [], []
    for k in drive:
        state = State.from_vector(states[:STATE_SIZE, k])
        angle, rate = float(angles[k]), float(rates[k])
        rotation = hull_to_abs(state.attitude)
        axis = np.array([np.sin(angle), side * np.cos(angle), 0.0])
        normal = np.array([np.cos(angle), -side * np.sin(angle), 0.0])
        blade = np.asarray(state.position, dtype=float) + rotation @ (
            lock_position + oar.outboard * axis)
        xs.append(blade[0])
        ys.append(blade[1])
        directions.append(_unit((rotation @ normal)[:2]))

        if law == "slip":
            speed = sim._lock_speed_on_normal(state, lock, angle)
            loads.append(float(oar.blade.normal_force(angle, rate, speed)))
            slips.append(float(oar.blade.slip_velocity(angle, rate, speed)))
        else:
            f_n, f_t = sim._blade_loads(seat_slot, angle, rate, state, lock)
            w_n, _w_a = sim._liftdrag[seat_slot].relative_velocity(
                angle, rate, sim._lock_velocity(state, lock)[:2], side)
            loads.append(float(f_n))
            slips.append(float(w_n))
            tangents.append(float(f_t))
            tangent_directions.append(_unit((rotation @ axis)[:2]))

    x = np.asarray(xs) - xs[0]
    y = np.asarray(ys) - ys[0]
    slip = np.asarray(slips)
    return BladeTrace(
        label=label, x=x, y=y, load=np.asarray(loads),
        direction=np.asarray(directions), slip=slip,
        speed=float(np.mean(run.last_speed)),
        crossings=_crossings(slip), finished=finished,
        tangential=np.asarray(tangents) if law != "slip" else None,
        tangential_direction=(np.asarray(tangent_directions)
                              if law != "slip" else None))


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


NORMAL_COLOUR = "#111827"
TANGENTIAL_COLOUR = "#DC2626"


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
        # One scale per panel for BOTH components, so a small tangential load
        # looks small next to the normal one rather than being blown up to fill
        # the panel.
        peak = float(np.abs(trace.load).max())
        if trace.has_tangential and trace.tangential.size:
            peak = max(peak, float(np.abs(trace.tangential).max()))
        peak = max(peak, 1e-9)
        span = max(float(np.ptp(trace.x)), float(np.ptp(trace.y)), 0.5)
        scale = 0.35 * span / peak
        picks = (np.linspace(0.05, 0.95, arrows) * (trace.x.size - 1)).astype(int)

        def arrow(k, magnitude, direction, colour_):
            dx, dy = scale * magnitude * direction
            ax.annotate("", xy=(trace.x[k] + dx, trace.y[k] + dy),
                        xytext=(trace.x[k], trace.y[k]),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2,
                                        color=colour_, alpha=0.85,
                                        shrinkA=0, shrinkB=0), zorder=5)

        for k in picks:
            arrow(k, trace.load[k], trace.direction[k], NORMAL_COLOUR)
            if trace.has_tangential:
                arrow(k, trace.tangential[k], trace.tangential_direction[k],
                      TANGENTIAL_COLOUR)

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
    fig.text(0.5, 0.005, _footnote(traces), ha="center", fontsize=8,
             color="#4B5563")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _footnote(traces: Sequence[BladeTrace]) -> str:
    """What the arrows are, and what the panels without red ones cannot show."""
    if not any(trace.has_tangential for trace in traces):
        return ("Arrows: normal blade load, scaled per panel. The tangential "
                "component is zero by construction under [CR06] Model 1; "
                "Grift et al. (2021) measure it on a real blade.")
    return ("Arrows: normal load (dark) and tangential load along the shaft "
            "(red), one scale per panel. Tier 2 panels rest on provisional "
            "coefficients from a secondary source; panels without red arrows "
            "are tier 1, whose tangential load is zero by construction.")
