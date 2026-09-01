r"""What a rower can actually produce, and why that makes rate a decision.

The simulator takes oar force as an **input**.  A rower in it can produce
any force at any handle speed, so stroke rate is pure bookkeeping: raise
the rate, scale the force down, and the boat is indifferent.  Tested
against Holt et al. [H20]_ -- who measured a large positive effect of rate
on velocity *after adjusting for power* -- the model returns +0.00% per
stroke per minute against their +0.6 to +1.1%.

That is not a hydrodynamic failure.  It is a missing rower.  Holt's own
explanation is the force-velocity relation in skeletal muscle, and a model
without muscle cannot have the effect they measured, at any level of
hydrodynamic fidelity.

Hill, at the handle
-------------------
Muscle force falls as shortening velocity rises.  Hill's hyperbola
[HILL38]_, written in the normalised form usually used for whole-body
tasks:

.. math::

    \frac{F(v)}{F_0} = \frac{1 - v/v_{max}}{1 + c\,v/v_{max}}

with ``c`` the curvature.  ``c = 0`` is the linear model and gives a power
peak at exactly half the maximum velocity; real muscle curves below that,
with ``c`` around 3-4 for mixed fibre types, moving the power peak down to
roughly a third of ``v_max``.  Mechanical power is ``F(v) v``, and the
existence of an interior maximum is the whole point: **there is a handle
speed that produces the most power, so there is a stroke rate that does.**

Rate, through the geometry
--------------------------
Handle velocity is set by the rate and the stroke length.  Over a drive of
duration ``tau_a`` covering a handle travel ``L``, the mean handle speed is
``L / tau_a``, and the drive duration is a fitted fraction of the stroke
period (:class:`~coxswain.crew.stroke.StrokeTiming`).  So

.. math::

    \bar v(r) = \frac{L}{f(r)\,(60/r)}

rises with rate, and every stroke's available force falls with it.  Work
per stroke falls, stroke count rises, and the product has a maximum.

Two ceilings, not one
---------------------
This is a **mechanical** ceiling and the project already has a
**metabolic** one: critical power plus a finite reserve
(:mod:`coxswain.crew.exertion`).  They constrain different things and both
bind:

* The metabolic ceiling says how much power the crew can average for the
  race duration -- ``CP + W'/T``.
* The mechanical ceiling says which *rates* can deliver that power at all.

A crew racing well below its mechanical peak has a *range* of rates that
deliver the required power, and is then free to choose among them on
hydrodynamic grounds -- which is where the boat's own preference for a
smooth run finally gets a say.  A crew at its mechanical ceiling has no
choice left.

References
----------
.. [HILL38] Hill, A. V. (1938) *The heat of shortening and the dynamic
   constants of muscle*, Proc. Royal Society B 126, 136-195.
.. [H20] Holt, A. C. et al. (2020) *Technical determinants of on-water
   rowing performance*, Frontiers in Sports and Active Living 2:589013.
.. [K16] Kleshnev, V. *Biomechanics of Rowing* -- handle travel and drive
   length figures used for the geometry below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ForceVelocity", "RowerCapacity"]


@dataclass(frozen=True)
class ForceVelocity:
    """Hill's relation, expressed at the handle rather than in a muscle.

    Lumping a whole rowing stroke -- legs, back, arms in sequence -- into
    one force-velocity curve is a real simplification, and the honest
    defence of it is that the *shape* is what matters here: force falling
    with speed and power peaking at an interior velocity.  Nothing below
    depends on the curve being a particular muscle's.
    """

    #: Maximum handle force at zero velocity, N.  Both hands together.
    peak_force: float = 1300.0
    #: Handle velocity at which no force remains, m/s.
    max_velocity: float = 4.6
    #: Hill curvature.  0 is linear; 3-4 is typical of mixed fibre.
    curvature: float = 3.2

    def force(self, velocity):
        """Maximum handle force available at this handle speed, N."""
        ratio = np.clip(np.asarray(velocity, float) / self.max_velocity,
                        0.0, 1.0)
        return self.peak_force * (1.0 - ratio) / (1.0 + self.curvature * ratio)

    def power(self, velocity):
        """Mechanical power at this handle speed, W."""
        return self.force(velocity) * np.asarray(velocity, float)

    @property
    def optimal_velocity(self) -> float:
        """Handle speed of maximum power, m/s."""
        grid = np.linspace(1e-4, self.max_velocity, 4000)
        return float(grid[int(np.argmax(self.power(grid)))])

    @property
    def peak_power(self) -> float:
        return float(self.power(self.optimal_velocity))


@dataclass
class RowerCapacity:
    """What one rower can deliver at a given stroke rate.

    Couples :class:`ForceVelocity` to the boat's own stroke geometry, so
    the answer is in stroke rate -- which is what a coxswain calls -- and
    not in handle metres per second.
    """

    force_velocity: ForceVelocity = None
    #: Handle travel over the drive, m.  A sweep rower's hands move about
    #: 1.55 m through the arc; a sculler's rather less per hand but the
    #: two hands together do comparable work.
    handle_travel: float = 1.55
    #: Fraction of the drive's mean handle speed that the force-velocity
    #: relation should be evaluated at.  The handle is not at its mean
    #: speed throughout -- it accelerates from the catch and slows into
    #: the finish -- and evaluating a concave curve at the mean
    #: overestimates what is available.  Jensen, not fudge.
    speed_factor: float = 1.15

    def __post_init__(self):
        if self.force_velocity is None:
            self.force_velocity = ForceVelocity()

    def handle_speed(self, rate, timing=None) -> float:
        """Effective handle speed at this rate, m/s."""
        from .stroke import StrokeTiming

        timing = timing or StrokeTiming(float(rate))
        return (self.speed_factor * self.handle_travel
                / max(timing.drive_duration, 1e-6))

    def available_power(self, rate, timing=None) -> float:
        """Mechanical power one rower can hold at this rate, W.

        Force from the force-velocity curve at the drive's handle speed,
        times that speed, times the fraction of the cycle spent driving --
        because the recovery produces nothing and the rate is quoted over
        the whole cycle.
        """
        from .stroke import StrokeTiming

        timing = timing or StrokeTiming(float(rate))
        speed = self.handle_speed(rate, timing)
        return float(self.force_velocity.force(speed) * speed
                     * timing.drive_fraction)

    @classmethod
    def calibrated(cls, power: float, rate: float, **kwargs):
        """Scale the curve so a rower delivers ``power`` at ``rate``.

        The Hill parameters are not measurable on a crew, and guessing
        them produced a rower whose mechanical ceiling (287 W) sat BELOW
        the metabolic one the pacing model asks for (313 W) -- a rower who
        cannot produce what they can metabolically afford, at any cadence.
        That is a statement about the guess, not about rowers.

        So the amplitude is calibrated to one observed operating point,
        exactly as :class:`~coxswain.hydro.wind.AeroModel` is, and what
        stays a **prediction** is the SHAPE: where the power optimum sits
        in rate, and how wide the feasible window is.  Those are what the
        rate question actually turns on and neither is fitted.
        """
        probe = cls(**kwargs)
        delivered = probe.available_power(rate)
        scale = float(power) / max(delivered, 1e-9)
        base = probe.force_velocity
        return cls(force_velocity=ForceVelocity(
            peak_force=base.peak_force * scale,
            max_velocity=base.max_velocity,
            curvature=base.curvature), **kwargs)

    def optimal_rate(self, low: float = 14.0, high: float = 48.0,
                     samples: int = 400) -> float:
        """Rate of maximum mechanical power, strokes per minute."""
        rates = np.linspace(low, high, samples)
        power = np.array([self.available_power(r) for r in rates])
        return float(rates[int(np.argmax(power))])

    def rate_window(self, required_power: float, low: float = 14.0,
                    high: float = 48.0, samples: int = 400):
        """Rates that can deliver ``required_power``, strokes per minute.

        Returns ``(lowest, highest)`` or ``None`` if the requirement is
        above the rower's mechanical ceiling at every rate -- which is a
        meaningful answer rather than an error: it says the crew cannot
        hold that power at any cadence, and the pacing model asked for
        too much.
        """
        rates = np.linspace(low, high, samples)
        power = np.array([self.available_power(r) for r in rates])
        feasible = rates[power >= float(required_power)]
        if not len(feasible):
            return None
        return float(feasible.min()), float(feasible.max())
