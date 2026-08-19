"""Crew and coxswain control loops.

A racing shell is not a passively stable vehicle, and the model must say
so honestly rather than hide it in a fudged coefficient.

Roll
----
With the crew's centre of mass roughly 0.35 m above the hull centre of
mass, an eight's crew weight contributes about ``+2580 N m/rad`` of
*upsetting* moment while the bare hull's hydrostatics supply only about
``-1050 N m/rad`` of righting moment.  The net is positive: left alone the
boat capsizes, and the simulator reproduces that in under two seconds.
This is not a modelling error -- it is why a shell tips over the moment
the blades leave the water.

Real crews hold the boat level by trimming handle heights: raising one
hand and lowering the other puts equal and opposite vertical forces on
two oarlocks, which is a pure couple with no net force.
:class:`BalanceController` models that reflex as a saturated PD loop on
roll angle and roll rate.  It is explicitly a *control* model, not a
hydrodynamic one; the saturation limit is what a crew can actually
deliver, so a badly disturbed boat still goes over.

Yaw
---
An alternating sweep rig has a residual yaw couple: summing
``side * x_oarlock`` over an eight's seats gives about ``-4.9 m``, so
every drive applies a moment in the same direction.  Real crews meet this
with the rudder.  :class:`HeadingController` is a PD loop on heading
error, which is also the interface a trajectory optimiser will drive when
the river geometry arrives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..core.frames import wrap_to_pi
from ..core.state import State

__all__ = ["BalanceController", "HeadingController", "Coxswain"]


@dataclass(frozen=True)
class BalanceController:
    """The crew's roll-balancing reflex, as a saturated PD couple.

    Applied as a pure moment about the hull ``x`` axis: the underlying
    handle-height trim is equal and opposite across the boat, so it
    produces no net force.

    Defaults give an eight a net roll stiffness of about
    ``-4500 N m/rad`` -- comfortably stable, settling in roughly a
    second, and holding steady-state roll to a degree or two, which is
    what a competent crew achieves.
    """

    stiffness: float = 6000.0     # N m / rad
    damping: float = 2000.0       # N m / (rad/s)
    max_moment: float = 4000.0    # N m, what a crew can actually apply
    enabled: bool = True

    def moment(self, roll: float, roll_rate: float) -> float:
        if not self.enabled:
            return 0.0
        demand = -(self.stiffness * roll + self.damping * roll_rate)
        return float(np.clip(demand, -self.max_moment, self.max_moment))


@dataclass(frozen=True)
class HeadingController:
    """Coxswain steering: a PD loop on heading error driving the rudder.

    ``target`` may be a constant heading in radians or a callable
    ``t -> heading``, which is the seam a river-following trajectory
    optimiser plugs into.
    """

    target: object = 0.0
    gain: float = 2.5             # rad rudder per rad heading error
    rate_gain: float = 1.2        # rad rudder per rad/s yaw rate
    max_deflection: float = np.radians(25.0)
    enabled: bool = True

    def target_heading(self, t: float) -> float:
        if callable(self.target):
            return float(self.target(t))
        return float(self.target)

    def deflection(self, t: float, state: State) -> float:
        if not self.enabled:
            return 0.0
        error = wrap_to_pi(state.yaw - self.target_heading(t))
        yaw_rate = float(state.omega_hull[2])
        # Positive rudder yaws to starboard, so a positive heading error
        # (drifted to port) calls for positive rudder.
        demand = self.gain * error + self.rate_gain * yaw_rate
        return float(np.clip(demand, -self.max_deflection,
                             self.max_deflection))


@dataclass
class Coxswain:
    """Convenience bundle of the two loops, plus an optional override."""

    balance: BalanceController = None
    heading: HeadingController = None
    rudder_override: Optional[Callable[[float, State], float]] = None

    def __post_init__(self) -> None:
        if self.balance is None:
            self.balance = BalanceController()
        if self.heading is None:
            self.heading = HeadingController()

    def roll_moment(self, state: State) -> float:
        return self.balance.moment(state.roll, float(state.omega_hull[0]))

    def rudder(self, t: float, state: State) -> float:
        if self.rudder_override is not None:
            return float(self.rudder_override(t, state))
        return self.heading.deflection(t, state)
