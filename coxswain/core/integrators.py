"""Time integrators.

Two are provided:

``rk4``       fixed step, deterministic, bit-reproducible across machines.
              Used for regression tests where a golden trajectory must be
              reproduced exactly, and as the default for real-time use.

``adaptive``  wraps :func:`scipy.integrate.solve_ivp`.  Preferred for
              accuracy studies and for anything quoted against the paper.

The legacy code used forward Euler at ``dt = 0.1 s`` against a 2 s stroke
period -- roughly 8 samples across the drive.  Forward Euler has no
stability margin on the lightly damped heave/pitch modes of a shell (the
heave natural period is ~0.5 s), so a good part of the "oscillation" it
showed was the integrator, not the boat.  :func:`rk4` at the same step is
already far better; :func:`estimate_step` picks a step from the stroke
period and the heave frequency.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["rk4_step", "rk4", "adaptive", "estimate_step"]

Derivative = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(derivative: Derivative, t: float, y: np.ndarray,
             dt: float) -> np.ndarray:
    """One classical fourth-order Runge-Kutta step."""
    k1 = derivative(t, y)
    k2 = derivative(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = derivative(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = derivative(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4(derivative: Derivative, t_span, y0: np.ndarray, dt: float):
    """Fixed-step RK4 over ``t_span``.

    Returns ``(t, y)`` where ``y`` has shape ``(len(y0), len(t))`` -- the
    same orientation as ``solve_ivp``.
    """
    t_start, t_end = float(t_span[0]), float(t_span[1])
    if dt <= 0:
        raise ValueError("dt must be positive")
    n_steps = int(np.ceil((t_end - t_start) / dt))

    times = np.empty(n_steps + 1)
    states = np.empty((len(y0), n_steps + 1))

    t = t_start
    # copy: never mutate the caller's initial condition (the legacy
    # integrator did, via `state += ...`)
    y = np.array(y0, dtype=float)
    times[0] = t
    states[:, 0] = y

    for i in range(n_steps):
        step = min(dt, t_end - t)
        y = rk4_step(derivative, t, y, step)
        t += step
        times[i + 1] = t
        states[:, i + 1] = y

    return times, states


def adaptive(derivative: Derivative, t_span, y0: np.ndarray,
             max_step: float = np.inf, rtol: float = 1e-6,
             atol: float = 1e-9, t_eval=None, method: str = "RK45"):
    """Adaptive integration via :func:`scipy.integrate.solve_ivp`.

    Raises ``RuntimeError`` if the solver reports failure, rather than
    returning a partial trajectory that looks like a physical result.
    """
    from scipy.integrate import solve_ivp

    solution = solve_ivp(
        derivative, t_span, np.array(y0, dtype=float), method=method,
        max_step=max_step, rtol=rtol, atol=atol, t_eval=t_eval,
        dense_output=False,
    )
    if not solution.success:
        raise RuntimeError(f"integration failed: {solution.message}")
    return solution.t, solution.y


def estimate_step(stroke_period: float, heave_period: float = 0.5,
                  samples_per_cycle: int = 80) -> float:
    """A safe fixed step for :func:`rk4`.

    Resolves the faster of the stroke forcing and the heave/pitch restoring
    oscillation.  ``heave_period`` defaults to a typical racing-shell heave
    natural period.
    """
    return min(stroke_period, heave_period) / samples_per_cycle
