"""Transcription schemes, and a mesh that knows about the stroke.

Two things live here.

The phase-locked mesh
---------------------
A uniform mesh is wrong for this problem.  The drive and the recovery are
physically different regimes -- the blades are in the water for one and out
for the other, so the crew's steering authority exists in one and not the
other -- and a mesh that straddles the catch forces a single control value
across both.  That is not a resolution question; no amount of refinement
fixes it, because the interval boundary is in the wrong place.

:func:`phase_locked_mesh` places interval boundaries **exactly** at every
catch and every finish, then subdivides within each phase.  Drive and
recovery therefore always carry distinct control values, and each can be
refined independently: a bend needs resolution during the drive, where the
split acts, and much less during the recovery, where only the rudder does
anything.

The mesh is non-uniform, which collocation does not mind -- Hermite-Simpson
and Radau both only need the interval lengths to be known, not equal.

The schemes
-----------
:class:`HermiteSimpson` is the default.  Third order, one interior point,
cheap per node.  For this problem that is the right trade: the dynamics
carry stroke-rate content that has to be resolved in *time* regardless, so
adding nodes is more useful than raising the order of a scheme that would
then be integrating across a catch anyway.

:class:`RadauIIA` is here for when that stops being true.  It is a
collocation scheme on Radau points -- stiffly accurate, A-stable, and of
order ``2s-1`` for ``s`` stages, so three stages give fifth order against
Hermite-Simpson's third.  Where it earns its cost is a *smooth* interval:
if a future version drives the crew from measured joint angles with a
sharper catch, or the shallow-water term steepens near critical, the
stiffness will start to bite and raising the order beats halving the step.

Both present the same interface -- given a dynamics function, a state and
control layout, and a mesh, produce the defect constraints -- so switching
is a one-line change in the caller rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "phase_locked_mesh",
    "MeshInterval",
    "HermiteSimpson",
    "RadauIIA",
    "radau_points",
]


@dataclass(frozen=True)
class MeshInterval:
    """One interval of a transcription mesh.

    ``phase`` is ``"drive"`` or ``"recovery"``; carrying it lets a caller
    bound the controls differently in each -- a split is meaningless on the
    recovery, and constraining it to zero there is both physical and a free
    reduction in the size of the NLP.
    """

    start: float
    end: float
    phase: str
    stroke: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def phase_locked_mesh(timing, n_strokes: int, drive_intervals: int = 6,
                      recovery_intervals: int = 4) -> Sequence[MeshInterval]:
    """A mesh with boundaries exactly at every catch and finish.

    Parameters
    ----------
    timing:
        A :class:`~coxswain.crew.stroke.StrokeTiming`.
    n_strokes:
        How many strokes the horizon covers.
    drive_intervals, recovery_intervals:
        Subdivisions within each phase.  The drive gets more by default
        because that is where both steering controls act; the recovery
        carries only the rudder, at reduced authority.

    Returns intervals in order, contiguous, covering ``n_strokes`` periods.
    """
    if n_strokes < 1:
        raise ValueError("need at least one stroke")
    if drive_intervals < 1 or recovery_intervals < 1:
        raise ValueError("each phase needs at least one interval")

    period = timing.period
    drive = timing.drive_duration
    intervals = []
    for stroke in range(n_strokes):
        base = stroke * period
        edges = np.linspace(base, base + drive, drive_intervals + 1)
        for start, end in zip(edges[:-1], edges[1:]):
            intervals.append(MeshInterval(float(start), float(end),
                                          "drive", stroke))
        edges = np.linspace(base + drive, base + period,
                            recovery_intervals + 1)
        for start, end in zip(edges[:-1], edges[1:]):
            intervals.append(MeshInterval(float(start), float(end),
                                          "recovery", stroke))
    return intervals


# --------------------------------------------------------------------------
# Radau points
# --------------------------------------------------------------------------
def radau_points(n_stages: int):
    """Radau IIA collocation points and differentiation matrix on ``[0, 1]``.

    Points are the roots of ``P_{s-1} + P_s`` shifted to the unit interval,
    with the right endpoint included -- which is what makes the scheme
    stiffly accurate.  Returns ``(points, differentiation_matrix, weights)``.

    CasADi ships these, so they are taken from there rather than
    re-derived: ``collocation_points`` has been checked far more than a
    fresh implementation would be.
    """
    import casadi as ca

    points = np.array([0.0] + list(ca.collocation_points(n_stages, "radau")))
    # Lagrange basis on the collocation points, differentiated at each
    differentiation = np.zeros((n_stages + 1, n_stages + 1))
    weights = np.zeros(n_stages + 1)
    for j in range(n_stages + 1):
        coefficients = np.array([1.0])
        for k in range(n_stages + 1):
            if k == j:
                continue
            coefficients = np.convolve(
                coefficients, np.array([1.0, -points[k]]))
            coefficients /= points[j] - points[k]
        derivative = np.polyder(coefficients)
        for k in range(n_stages + 1):
            differentiation[j, k] = np.polyval(derivative, points[k])
        weights[j] = np.polyval(np.polyint(coefficients), 1.0)
    return points, differentiation, weights


# --------------------------------------------------------------------------
# schemes
# --------------------------------------------------------------------------
class HermiteSimpson:
    """Third-order collocation: cubic Hermite state, Simpson defect.

    One interior control per interval, so a phase-locked mesh gives the
    drive and the recovery genuinely independent inputs.
    """

    name = "hermite-simpson"
    order = 3
    n_interior = 1

    @staticmethod
    def defects(dynamics, state, control, control_mid, times, durations):
        """Defect residuals, one per interval.

        ``dynamics(state, control, time)`` returns the derivative.
        """
        import casadi as ca

        residuals = []
        for k, step in enumerate(durations):
            left, right = state[:, k], state[:, k + 1]
            f_left = dynamics(left, control[:, k], times[k])
            f_right = dynamics(right, control[:, k + 1], times[k + 1])
            middle = 0.5 * (left + right) + step / 8.0 * (f_left - f_right)
            f_middle = dynamics(middle, control_mid[:, k],
                                times[k] + 0.5 * step)
            residuals.append(
                right - left - step / 6.0 * (f_left + 4.0 * f_middle
                                             + f_right))
        return ca.vertcat(*residuals)


class RadauIIA:
    """Radau IIA collocation, order ``2s-1`` for ``s`` stages.

    Not the default.  Hermite-Simpson is cheaper per node, and this problem
    needs resolution in *time* to follow the stroke regardless -- so on a
    phase-locked mesh, adding intervals inside the drive buys more than
    raising the order does.

    It is here because that argument stops holding when an interval becomes
    stiff: a sharper catch profile from measured joint angles, or the
    shallow-water factor near the critical Froude number, would both make
    high order worth its cost.  Being stiffly accurate and A-stable, Radau
    handles those where an explicit-flavoured scheme struggles.
    """

    name = "radau-iia"

    def __init__(self, n_stages: int = 3):
        if n_stages < 1:
            raise ValueError("Radau IIA needs at least one stage")
        self.n_stages = n_stages
        self.points, self.differentiation, self.weights = radau_points(
            n_stages)

    @property
    def order(self) -> int:
        return 2 * self.n_stages - 1

    def defects(self, dynamics, state, stage_states, control, times,
                durations):
        """Defect residuals for the collocation equations plus continuity.

        ``stage_states`` is a list over intervals of ``(n_states, s)``
        matrices holding the state at each collocation point.
        """
        import casadi as ca

        residuals = []
        for k, step in enumerate(durations):
            left = state[:, k]
            stages = stage_states[k]
            end = left * self.differentiation[0, 0] * 0.0 + left  # copy
            # collocation: the polynomial derivative matches the dynamics
            for j in range(1, self.n_stages + 1):
                derivative = self.differentiation[0, j] * left
                for i in range(1, self.n_stages + 1):
                    derivative = derivative + self.differentiation[i, j] \
                        * stages[:, i - 1]
                f = dynamics(stages[:, j - 1], control[:, k],
                             times[k] + self.points[j] * step)
                residuals.append(derivative - step * f)
            # Radau includes the right endpoint, so continuity is exact
            residuals.append(state[:, k + 1] - stages[:, self.n_stages - 1])
        return ca.vertcat(*residuals)
