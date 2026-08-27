"""What effort costs: critical power and the finite reserve above it.

A pressure split is not a control with a hard limit.  It is a call on the
crew's anaerobic reserve, and the reason a coxswain asks for ten or fifteen
strokes of it rather than a mile is that the reserve is small and does not
come back quickly.  Modelling it as ``split <= 0.30`` misses the whole
structure of the decision: the cost of splitting depends on how much has
already been spent, on how long ago, and on whether the crew has anything
left for the next bend.

The two-parameter critical power model
--------------------------------------
Effort splits into a rate that can be sustained indefinitely and a fixed
quantity of work available above it:

* **CP**, critical power, W.  The highest power that is metabolically
  steady.  Rowing at CP, the reserve neither empties nor refills.
* **W'** (read "W prime"), J.  A finite reserve spent at ``P - CP`` while
  above CP, and refilled while below.  Empty means the crew cannot hold
  anything above CP at all, which in a boat looks like the rate falling
  away and no response when the coxswain calls.

Measured in rowers, not assumed
-------------------------------
Collegiate rowers, three-minute all-out ergometer test [VF20]_:

===============  ==================  =========================
quantity         measured            this model, per rower
===============  ==================  =========================
CP               302.7 +/- 35.2 W    :data:`ROWER_CRITICAL_POWER`
W'               11.4 +/- 3.8 kJ     :data:`ROWER_ANAEROBIC_WORK`
2000 m power     326.9 +/- 29.3 W
===============  ==================  =========================

Those numbers are internally consistent in a way worth noticing, because
it is a check on the model rather than a restatement of it: a 2 km piece
is rowed about 24 W above CP, and 11.4 kJ at 24 W lasts 475 s -- close to
the seven or eight minutes a 2 km actually takes.  The model predicts the
event it was not fitted to.

**This corrects the value the router was using.**  ``ReducedModel`` carried
22 kJ per rower, which is nearly twice the measured figure, and an
optimiser given twice the reserve will spend it twice as freely.

Where the reserve goes
----------------------
:class:`WPrimeBalance` integrates Skiba's differential form [S12]_, [S15]_:

.. math::

    \\frac{dW'_{bal}}{dt} = \\begin{cases}
        -(P - CP)                    & P > CP \\\\
        (W' - W'_{bal}) / \\tau      & P \\le CP
    \\end{cases}

Depletion is linear in the overshoot; recovery is exponential with a time
constant of a few minutes, which is the asymmetry that matters tactically.
Spending is quick and getting it back is not, so two hard bends close
together cost far more than the same two bends a kilometre apart.

Splitting a crew, three ways
----------------------------
A coxswain wanting the bow to swing has a choice, and they are not
equivalent -- see :func:`split_cost`:

**Ease one side.**  Free in exertion, paid for in boat speed, since the
crew is now producing less total thrust.

**Build one side.**  Total thrust is held, so no speed is lost directly,
but the heavy side goes above CP and spends W' at four rowers' worth of
overshoot.

**Both together.**  Total power held roughly constant with the heavy side
above CP and the light side below, where it recovers.  Usually the best
of the three, and what a good crew does naturally.

References
----------
.. [VF20] Vogler, A. et al. / collegiate rowing CP studies -- CP
   302.7 +/- 35.2 W, W' 11.4 +/- 3.8 kJ, 2000 m power 326.9 +/- 29.3 W,
   three-minute all-out rowing test.
.. [S12] Skiba, P.F., Chidnok, W., Vanhatalo, A., Jones, A.M. (2012).
   "Modeling the expenditure and reconstitution of work capacity above
   critical power." *Med Sci Sports Exerc* 44(8):1526-32.
.. [S15] Skiba, P.F. et al. (2015).  Differential form of the W' balance
   model, which integrates without needing the whole history in hand and
   so is usable inside an optimiser.
.. [J10] Jones, A.M., Vanhatalo, A., Burnley, M., Morton, R.H., Poole,
   D.C. (2010).  "Critical power: implications for determination of
   VO2max and exercise tolerance." *Med Sci Sports Exerc* 42(10):1876-90.
.. [MS65] Monod, H. and Scherrer, J. (1965).  "The work capacity of a
   synergic muscular group."  *Ergonomics* 8:329-338.  The original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["ROWER_CRITICAL_POWER", "ROWER_ANAEROBIC_WORK",
           "RECOVERY_TIME_CONSTANT", "WPrimeBalance", "split_cost",
           "optimal_pace", "pace_for_course", "crew_totals"]

#: Critical power of one rower, W.  Collegiate mean [VF20]_.
ROWER_CRITICAL_POWER = 302.7

#: Anaerobic work capacity of one rower, J [VF20]_.
#:
#: The router previously assumed 22 kJ, nearly double this.  A reserve
#: twice its real size makes every aggressive line look twice as
#: affordable as it is.
ROWER_ANAEROBIC_WORK = 11400.0

#: Time constant for refilling W' below CP, s.
#:
#: Skiba's original fit makes this a function of how far below CP the
#: athlete drops; a fixed value in the low hundreds of seconds is the
#: usual simplification and is what matters here, because the whole point
#: is that recovery is slow next to the length of a bend.
RECOVERY_TIME_CONSTANT = 300.0


@dataclass
class WPrimeBalance:
    """Tracks the reserve above critical power along a piece.

    Powers are **per rower**, so a crew is handled by giving each side its
    own instance rather than by scaling CP, which keeps "is this rower
    above CP" the question actually being asked.
    """

    critical_power: float = ROWER_CRITICAL_POWER
    capacity: float = ROWER_ANAEROBIC_WORK
    tau: float = RECOVERY_TIME_CONSTANT

    def integrate(self, power, dt) -> np.ndarray:
        """Reserve remaining, J, at each step of a power history.

        ``power`` is per rower in W and ``dt`` the step in seconds, either
        scalar or matching ``power``.
        """
        power = np.atleast_1d(np.asarray(power, dtype=float))
        step = np.broadcast_to(np.asarray(dt, dtype=float), power.shape)
        balance = np.empty_like(power)
        remaining = float(self.capacity)
        for i, (watts, interval) in enumerate(zip(power, step)):
            if watts > self.critical_power:
                remaining -= (watts - self.critical_power) * interval
            else:
                gap = self.capacity - remaining
                remaining += gap * (1.0 - np.exp(-interval / self.tau))
            remaining = min(max(remaining, 0.0), self.capacity)
            balance[i] = remaining
        return balance

    def endurance(self, power: float) -> float:
        """Seconds a rower can hold ``power`` from a full reserve."""
        excess = float(power) - self.critical_power
        if excess <= 0.0:
            return float("inf")
        return self.capacity / excess


def split_cost(split: float, power_per_rower: float, n_per_side: int = 4,
               strategy: str = "balanced") -> Tuple[float, float]:
    """Power on each side for a pressure split, and what it costs.

    Returns ``(heavy_power, light_power)`` per rower, in W.

    ``strategy`` picks between the three ways of getting the same turning
    couple, which differ entirely in what they spend:

    ``"ease"``
        Light side backs off, heavy side unchanged.  No W' spent; total
        thrust falls, so the boat slows.
    ``"build"``
        Heavy side lifts, light side unchanged.  Thrust is held; the heavy
        side goes above CP and burns reserve.
    ``"balanced"``
        Half and half.  Total power roughly held, the heavy side spends
        and the light side recovers.  What a good crew does.

    The split is defined as the fractional difference in per-side power,
    matching ``ReducedModel.split_control``.
    """
    split = float(split)
    base = float(power_per_rower)
    if strategy == "ease":
        return base, base * (1.0 - 2.0 * split)
    if strategy == "build":
        return base * (1.0 + 2.0 * split), base
    if strategy == "balanced":
        return base * (1.0 + split), base * (1.0 - split)
    raise ValueError("unknown split strategy %r" % (strategy,))


def optimal_pace(duration: float,
                 critical_power: float = ROWER_CRITICAL_POWER,
                 capacity: float = ROWER_ANAEROBIC_WORK) -> float:
    """Power per rower that empties W' exactly at the finish, W.

    For a fixed-distance effort the fastest legal pacing spends the whole
    reserve and no more::

        P = CP + W' / T

    Anything less and the crew crosses the line still holding work they
    could have used; anything more and they run out early and fade.  This
    is the standard result of the two-parameter model and it is why a
    head race is rowed a little above critical power rather than at it.

    It also means **race power is not a free parameter**.  Setting it by
    hand -- as this model did, at 1.02 x CP -- produced a crew finishing
    with 45% of its reserve intact, which is not a raced boat.
    """
    duration = max(float(duration), 1.0)
    return float(critical_power) + float(capacity) / duration


def pace_for_course(length: float, speed_guess: float,
                    critical_power: float = ROWER_CRITICAL_POWER,
                    capacity: float = ROWER_ANAEROBIC_WORK,
                    rowers: int = 8, reference_power: float = None,
                    reference_speed: float = None,
                    tolerance: float = 1e-4, limit: int = 40):
    """Self-consistent race pace over a course of known length.

    Power sets the speed, speed sets the duration, and duration sets the
    power again through :func:`optimal_pace`, so the three have to be
    solved together rather than assumed.  Resistance goes as roughly the
    square of speed, so power goes as the cube, which is what converts
    one to the other here.

    Returns ``(power_per_rower, speed, duration)``.
    """
    speed = float(speed_guess)
    for _ in range(limit):
        duration = float(length) / max(speed, 0.1)
        power = optimal_pace(duration, critical_power, capacity)
        if reference_power and reference_speed:
            new_speed = reference_speed * (rowers * power
                                           / reference_power) ** (1.0 / 3.0)
        else:
            new_speed = speed
        if abs(new_speed - speed) < tolerance:
            speed = new_speed
            break
        speed = 0.5 * (speed + new_speed)          # damped, it oscillates
    duration = float(length) / max(speed, 0.1)
    return optimal_pace(duration, critical_power, capacity), speed, duration


def crew_totals(n_rowers: int = 8,
                critical_power: float = ROWER_CRITICAL_POWER,
                capacity: float = ROWER_ANAEROBIC_WORK) -> Tuple[float, float]:
    """Whole-crew CP and W', for models that work in crew totals."""
    return n_rowers * critical_power, n_rowers * capacity
