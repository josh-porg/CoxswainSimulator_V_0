"""Standard 2-D diagnostic charts for a simulation run.

The set here is chosen to make a wrong result *look* wrong:

``trajectory``          top view of the track, with heading ticks
``speed_history``       hull speed against time, drive phases shaded
``stroke_cycle``        phase-averaged speed over one stroke -- the single
                        most diagnostic plot in rowing, because its shape
                        is well known: the boat is checked at the catch and
                        runs on the recovery
``secondary_motions``   heave, pitch and roll, the motions the paper is
                        specifically about
``rates``               angular velocity in all three axes
``force_breakdown``     every force source against time, summing to the net
``crew_and_hull``       crew centre of mass and hull position, showing them
                        trading momentum

:func:`dashboard` puts them on one figure.

Nothing here caches or mutates; all of it takes a
:class:`~coxswain.sim.results.SimulationResult` and the
:class:`~coxswain.boats.boat.Boat` that produced it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..core.state import State

__all__ = [
    "trajectory",
    "speed_history",
    "stroke_cycle",
    "secondary_motions",
    "rates",
    "force_breakdown",
    "crew_and_hull",
    "dashboard",
    "save_dashboard",
    "phase_average",
    "recompute_breakdown",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def phase_average(time: np.ndarray, signal: np.ndarray, period: float,
                  n_bins: int = 60, skip_cycles: float = 1.0):
    """Average a signal over stroke phase.

    Returns ``(phase_centres, mean, std)``.  ``skip_cycles`` drops the
    opening transient so the average describes settled rowing.
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    keep = time >= time[0] + skip_cycles * period
    if keep.sum() < n_bins:
        keep = np.ones_like(time, dtype=bool)

    phase = np.mod(time[keep], period) / period
    values = signal[keep]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)

    mean = np.full(n_bins, np.nan)
    deviation = np.full(n_bins, np.nan)
    for bin_index in range(n_bins):
        selected = values[index == bin_index]
        if selected.size:
            mean[bin_index] = selected.mean()
            deviation[bin_index] = selected.std()
    return 0.5 * (edges[:-1] + edges[1:]), mean, deviation


def _shade_drive(axis, time, boat, alpha=0.09):
    """Shade the drive phase of every stroke in the window."""
    period = boat.timing.period
    drive = boat.timing.drive_duration
    start = np.floor(time[0] / period) * period
    while start < time[-1]:
        axis.axvspan(max(start, time[0]), min(start + drive, time[-1]),
                     color="tab:blue", alpha=alpha, linewidth=0)
        start += period


def recompute_breakdown(result, simulator):
    """Force and moment history, recomputed from the stored trajectory.

    The integrator does not record forces (they would triple the memory of
    a run), but they are a pure function of ``(t, state)``, so they can be
    replayed exactly.
    """
    sources = ("crew", "oar", "buoyancy", "gravity", "resistance",
               "appendage")
    forces = {name: np.zeros((len(result.time), 3)) for name in sources}
    moments = {name: np.zeros((len(result.time), 3)) for name in sources}
    net = np.zeros((len(result.time), 6))

    for index, t in enumerate(result.time):
        state = State.from_vector(result.states[:, index])
        breakdown = simulator.breakdown(t, state)
        for name in sources:
            forces[name][index] = getattr(breakdown, f"{name}_force")
            moments[name][index] = getattr(breakdown, f"{name}_moment")
        net[index] = breakdown.generalised()

    return forces, moments, net


# --------------------------------------------------------------------------
# individual charts
# --------------------------------------------------------------------------
def trajectory(result, boat=None, axis=None, heading_every: int = 40):
    """Top view of the track, with periodic heading ticks."""
    import matplotlib.pyplot as plt

    axis = axis or plt.subplots(figsize=(6, 6))[1]
    x, y = result.position[0], result.position[1]

    axis.plot(x, y, color="tab:blue", linewidth=1.6, label="track")
    axis.plot(x[0], y[0], "o", color="tab:green", markersize=8, label="start")
    axis.plot(x[-1], y[-1], "o", color="tab:red", markersize=8, label="end")

    if heading_every > 0 and len(x) > heading_every:
        step = slice(None, None, heading_every)
        yaw = result.yaw[step]
        scale = 0.02 * max(np.ptp(x), np.ptp(y), 1.0)
        axis.quiver(x[step], y[step], np.cos(yaw) * scale, np.sin(yaw) * scale,
                    color="0.35", width=0.004, scale_units="xy", scale=1,
                    label="heading")

    axis.set_xlabel("X, along the course [m]")
    axis.set_ylabel("Y, across the course [m]")
    axis.set_title("Track (top view)")
    axis.axis("equal")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    return axis


def speed_history(result, boat=None, axis=None):
    """Hull speed against time, with the drive phases shaded."""
    import matplotlib.pyplot as plt

    axis = axis or plt.subplots(figsize=(8, 3))[1]
    axis.plot(result.time, result.surge_speed, color="tab:blue",
              linewidth=1.2, label="surge (hull x)")
    axis.plot(result.time, result.speed, color="0.5", linewidth=0.9,
              linestyle="--", label="|V|")
    if boat is not None:
        _shade_drive(axis, result.time, boat)
        axis.axhline(result.mean_speed(), color="tab:red", linewidth=0.9,
                     linestyle=":",
                     label=f"mean {result.mean_speed():.2f} m/s")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("speed [m/s]")
    axis.set_title("Hull speed (shaded = drive)")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    return axis


def stroke_cycle(result, boat, axis=None, n_bins: int = 60):
    """Phase-averaged speed over one stroke.

    The shape is the diagnostic: a real shell decelerates through the
    drive as the crew move bow-ward and accelerates on the recovery.  A
    curve peaking in the middle of the drive means the crew reaction has
    the wrong sign somewhere.
    """
    import matplotlib.pyplot as plt

    axis = axis or plt.subplots(figsize=(6, 4))[1]
    phase, mean, deviation = phase_average(
        result.time, result.surge_speed, boat.timing.period, n_bins)

    axis.plot(phase, mean, color="tab:blue", linewidth=2, label="mean")
    axis.fill_between(phase, mean - deviation, mean + deviation,
                      color="tab:blue", alpha=0.2, label="+-1 sd")
    axis.axvspan(0.0, boat.timing.drive_fraction, color="tab:blue",
                 alpha=0.09, linewidth=0, label="drive")

    axis.set_xlabel("stroke phase (0 = catch)")
    axis.set_ylabel("surge speed [m/s]")
    axis.set_title("Speed through the stroke")
    axis.set_xlim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    return axis


def secondary_motions(result, boat=None, axes=None):
    """Heave, pitch and roll -- the motions the paper is about."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes = np.atleast_1d(axes)

    axes[0].plot(result.time, result.heave, color="tab:green")
    axes[0].set_ylabel("heave [m]")
    axes[0].set_title("Secondary motions")

    axes[1].plot(result.time, np.degrees(result.pitch), color="tab:orange")
    axes[1].set_ylabel("pitch [deg]")

    axes[2].plot(result.time, np.degrees(result.roll), color="tab:purple")
    axes[2].set_ylabel("roll [deg]")
    axes[2].set_xlabel("time [s]")

    for axis in axes:
        axis.grid(alpha=0.3)
        axis.axhline(0.0, color="0.7", linewidth=0.7)
        if boat is not None:
            _shade_drive(axis, result.time, boat)
    return axes


def rates(result, boat=None, axis=None):
    """Angular velocity in all three axes."""
    import matplotlib.pyplot as plt

    axis = axis or plt.subplots(figsize=(8, 3))[1]
    omega = np.degrees(result.omega)
    for row, label, colour in ((0, "roll rate", "tab:purple"),
                               (1, "pitch rate", "tab:orange"),
                               (2, "yaw rate", "tab:red")):
        axis.plot(result.time, omega[row], label=label, color=colour,
                  linewidth=1.1)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("rate [deg/s]")
    axis.set_title("Angular velocity")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    return axis


def force_breakdown(result, simulator, axis=None, component: int = 0,
                    forces=None):
    """Every force source against time, for one component.

    ``component`` indexes the absolute frame: 0 = along the course,
    1 = across, 2 = vertical.  Passing a precomputed ``forces`` dict skips
    the replay.
    """
    import matplotlib.pyplot as plt

    axis = axis or plt.subplots(figsize=(8, 4))[1]
    if forces is None:
        forces, _, _ = recompute_breakdown(result, simulator)

    label = "XYZ"[component]
    total = np.zeros(len(result.time))
    for name, series in forces.items():
        axis.plot(result.time, series[:, component], linewidth=1.0, label=name)
        total += series[:, component]
    axis.plot(result.time, total, color="k", linewidth=1.6, linestyle="--",
              label="net")

    axis.set_xlabel("time [s]")
    axis.set_ylabel(f"force {label} [N]")
    axis.set_title(f"Force breakdown ({label}, absolute frame)")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=7, ncol=2)
    return axis


def crew_and_hull(result, boat, axis=None):
    """Crew centre of mass and hull position, trading momentum.

    Both are plotted relative to their own mean and to the system centre
    of mass, so the anti-phase relationship is visible: the crew move
    bow-ward on the drive and the hull is checked, then the reverse.
    """
    import matplotlib.pyplot as plt

    axis = axis or plt.subplots(figsize=(8, 3.5))[1]

    crew_x = np.array([boat.crew_centre_of_mass(t)[0] for t in result.time])
    mean_speed = result.mean_speed()
    hull_detrended = result.position[0] - mean_speed * result.time
    hull_detrended -= hull_detrended.mean()

    axis.plot(result.time, crew_x - crew_x.mean(), color="tab:orange",
              linewidth=1.3, label="crew CoM in hull (detrended)")
    axis.plot(result.time, hull_detrended, color="tab:blue", linewidth=1.3,
              label="hull position (mean speed removed)")
    _shade_drive(axis, result.time, boat)

    axis.set_xlabel("time [s]")
    axis.set_ylabel("displacement [m]")
    axis.set_title("Crew and hull exchange momentum")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    return axis


# --------------------------------------------------------------------------
# assembled dashboard
# --------------------------------------------------------------------------
def dashboard(result, boat, simulator=None, title: Optional[str] = None,
              figsize=(16, 12)):
    """One figure with the full standard set.

    ``simulator`` is optional; without it the force-breakdown panel is
    replaced by the crew/hull momentum exchange, which needs no replay.
    """
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=figsize, constrained_layout=True)
    grid = figure.add_gridspec(4, 3)

    trajectory(result, boat, axis=figure.add_subplot(grid[0:2, 0]))
    speed_history(result, boat, axis=figure.add_subplot(grid[0, 1:]))
    stroke_cycle(result, boat, axis=figure.add_subplot(grid[1, 1]))
    crew_and_hull(result, boat, axis=figure.add_subplot(grid[1, 2]))

    secondary_motions(result, boat, axes=[
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
        figure.add_subplot(grid[2, 2]),
    ])

    rates(result, boat, axis=figure.add_subplot(grid[3, 0:2]))

    axis = figure.add_subplot(grid[3, 2])
    if simulator is not None:
        force_breakdown(result, simulator, axis=axis, component=0)
    else:
        axis.plot(result.time, result.sway, color="tab:red")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("sway [m]")
        axis.set_title("Lateral drift")
        axis.grid(alpha=0.3)

    heading = title or (
        f"{boat.name} at {boat.timing.rate:.0f} spm  --  "
        f"mean {result.mean_speed():.2f} m/s, "
        f"{result.speed_fluctuation_ratio() * 100:.0f}% fluctuation"
    )
    figure.suptitle(heading, fontsize=13)
    return figure


def save_dashboard(result, boat, path, simulator=None, dpi: int = 110,
                   **kwargs):
    """Render :func:`dashboard` straight to a file, without a display."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure = dashboard(result, boat, simulator=simulator, **kwargs)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path
