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
from ..hydro.appendages import MAX_RUDDER_DEFLECTION

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
    max_moment: float = 4000.0    # N m; used only when no authority is set
    enabled: bool = True
    #: Phase-dependent authority.  When set, it replaces ``max_moment``
    #: entirely and the saturation becomes a function of where in the
    #: stroke the crew is.  See :class:`coxswain.crew.balance.PhaseAuthority`
    #: -- the short version is that the blades are the only thing a crew
    #: can push against, so on the recovery there is almost nothing.
    authority: object = None
    timing: object = None
    #: Stroke-to-stroke learned trim (:class:`coxswain.crew.trim.StrokeTrim`).
    #: Feedforward, replayed from the previous stroke -- this is how a crew
    #: actually uses hand heights, and it is a different control law from
    #: the within-stroke reflex that [D96] shows is positive feedback.
    trim: object = None

    def limit(self, t: float = None) -> float:
        if self.authority is None or self.timing is None or t is None:
            return self.max_moment
        return float(self.authority.window(t, self.timing))

    def moment(self, roll: float, roll_rate: float, t: float = None) -> float:
        if not self.enabled:
            return 0.0
        demand = -(self.stiffness * roll + self.damping * roll_rate)
        if self.trim is not None and self.timing is not None and t is not None:
            # Feedforward first: the learned trim is already being applied
            # before the error it corrects would recur.
            demand += self.trim.command(t, self.timing)
        bound = self.limit(t)
        return float(np.clip(demand, -bound, bound))


def balance_for_experience(boat, experience: float = 0.55,
                           learn: bool = True) -> "BalanceController":
    """A crew's balance reflex at a given level, 0 novice to 1 ideal.

    Wires up the three things a real crew balances with and which the
    bare PD default leaves switched off:

    * **Phase authority.**  The blades are the only thing a crew can
      push against, so on the recovery there is almost nothing -- and
      what there is comes mostly from leaning the trunk, not from the
      hands.  :meth:`PhaseAuthority.from_boat` already combines both.
    * **Handle height**, which is how the moment is actually produced.
      That path was already live: the simulator sends the command
      through :class:`BalanceRig`, so it arrives as vertical forces at
      the riggers and carries the pitch couple that comes with them.
    * **Learned trim**, :class:`StrokeTrim`, which is a crew getting the
      boat down over several strokes rather than fighting each one.

    Only two numbers here are calibrated: ``handle_force`` 150 N and a
    2-degree lean, both from :mod:`coxswain.crew.balance`.  How they
    fall off with inexperience is **engineering judgement** -- nobody
    has measured a novice's spare vertical handle force -- and it is
    shaped so an ideal crew gets exactly the calibrated values.
    """
    import numpy as np

    from ..crew.balance import PhaseAuthority
    from ..crew.trim import StrokeTrim

    experience = float(np.clip(experience, 0.0, 1.0))

    # A novice has less to spare while pulling, and less trunk control.
    handle_force = 60.0 + 90.0 * experience          # -> 150 N at ideal
    lean = np.radians(0.7 + 1.3 * experience)        # -> 2 deg at ideal
    authority = PhaseAuthority.from_boat(boat, handle_force=handle_force,
                                         lean_angle=lean)

    # And a slower, softer reflex: the gains are what the crew's nervous
    # system does, the authority is what their bodies can produce.
    reflex = 0.40 + 0.60 * experience

    # Learning is the thing experience most obviously is.  A novice crew
    # does not carry a correction from one stroke to the next; a good one
    # has the boat sat within a few.  Built up front because the
    # controller is frozen -- deliberately, since a control law that can
    # be mutated mid-run is one whose behaviour cannot be reproduced.
    trim = StrokeTrim(learning_gain=0.6 * experience,
                      forgetting=0.70 + 0.22 * experience) if learn else None
    return BalanceController(
        stiffness=6000.0 * reflex,
        damping=2000.0 * reflex,
        authority=authority,
        timing=boat.timing,
        trim=trim,
    )


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
    max_deflection: float = MAX_RUDDER_DEFLECTION
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
    #: Port/starboard pressure split, in ``[-1, 1]``.  Positive means the
    #: port side pulls harder, which yaws the bow to starboard.  Either a
    #: constant or a callable ``(t, state) -> float``.
    #: ``None`` means "defer to the steering law's own pressure call, if
    #: it makes one" -- the old default of ``0.0`` was indistinguishable
    #: from an explicit instruction to row even, which is why the
    #: deferral branch below never fired on its first outing.
    pressure_split: object = None

    def __post_init__(self) -> None:
        if self.balance is None:
            self.balance = BalanceController()
        if self.heading is None:
            self.heading = HeadingController()

    def roll_moment(self, state: State, t: float = None) -> float:
        return self.balance.moment(state.roll, float(state.omega_hull[0]), t)

    def rudder(self, t: float, state: State) -> float:
        if self.rudder_override is not None:
            return float(self.rudder_override(t, state))
        return self.heading.deflection(t, state)

    def split(self, t: float, state: State) -> float:
        """Port/starboard pressure split at this instant.

        The second steering control, and on a river the more important one.
        Measured against the extracted Charles channel: the tightest bends
        demand a 103-146 m turn radius, while full rudder alone holds only
        283 m -- 19% of the reach is tighter than the rudder can manage.
        A coxswain calling for pressure is not a refinement on top of the
        rudder; without it an eight cannot get round the river at all.

        Clamped to ``[-1, 1]``: a split of 1 would mean one side stopped
        dead and the other at full pressure.
        """
        value = self.pressure_split
        # A steering law plugged in through ``rudder_override`` may carry
        # its own pressure call -- both the line-of-sight follower and the
        # MPC set ``.split`` on themselves.  Until this branch existed
        # nothing read it: the MPC solved for two controls every fifth of
        # a second and the boat quietly executed one of them.  The plan
        # said "rudder plus pressure through the bend"; the crew never
        # heard the call.
        if value is None and self.rudder_override is not None:
            value = getattr(self.rudder_override, "split", None)
        if value is None:
            value = 0.0
        if callable(value):
            value = value(t, state)
        return float(np.clip(value, -1.0, 1.0))

    @staticmethod
    def side_gain(split: float, side: int) -> float:
        """Thrust multiplier for one side, given a split.

        Split is applied symmetrically -- half added on one side, half
        removed on the other -- so the *net* thrust is unchanged and the
        split produces a pure yaw couple.  Modelling it any other way would
        let the optimiser accelerate by steering.
        """
        return 1.0 + 0.5 * split * float(side)
