r"""Where to spend the reserve: pacing a head race against the course.

:mod:`coxswain.crew.exertion` answers *how hard* a crew can go for a given
duration -- ``P = CP + W'/T``, one number for the whole race.  On a
buoyed 2 km in still water that is the whole answer.  On the Charles it is
not, because the course is not the same everywhere: the current runs
harder in the bends, the wind is sheltered behind Harvard and open across
the basin, and the water shoals under the bridges.  A crew that holds one
power everywhere is leaving time on the river.

How much reserve is actually in play
------------------------------------
Worth being blunt about the size of this lever before building on it.  A
masters eight rowing 1140 s has ``W'/T = 11400/1140 = 10 W`` per rower
above a critical power of 303 -- **about 3%**.  The reserve is not a way
to row the race harder; it is a way to row *parts* of it harder.

Spent as surges it is worth more than that sounds.  Thirty watts above CP
for sixty seconds costs 1800 J, roughly a sixth of the reserve, so a crew
has on the order of **six one-minute pushes** in the whole race.  The
coxswain's question is not whether to use them but where, and that is a
question about the course.

The optimality condition, and what it says
------------------------------------------
Minimise ``T = \int dx / v_g`` subject to a fixed total work, with
``v_g = v_w + c`` the ground speed, ``v_w`` the speed through water and
``c`` the along-course current.  Writing ``e = d\ln v/d\ln P``
(:mod:`coxswain.sim.performance` measures it; Young's constant-``C``
algebra gives 1/3), the stationarity condition on segment ``i`` is

.. math::

    \frac{e\,k_i}{P_i} = \lambda \left(1 - e\,k_i\right),
    \qquad k_i = \frac{v_{w,i}}{v_{g,i}}

so that

.. math::

    P_i = \frac{e\,k_i}{\lambda\,(1 - e\,k_i)}

**Every course-dependent term enters through ``k``, the ratio of speed
through water to speed over ground.**  Where the current runs against the
boat ``k > 1`` and the optimal power rises; where it helps, ``k < 1`` and
the optimal power falls.  In still water ``k = 1`` everywhere and the
condition collapses to constant power, which is the classical result and a
useful check that nothing has been smuggled in.

The mechanism is worth stating in words because it is the opposite of the
instinct: **you push hardest where you are slowest**, because that is
where a given number of extra watts buys the most extra *seconds*, not the
most extra metres per second.  Easing in the fast water and pushing in the
slow water shortens the race even though it lowers the average speed at
which the power is applied.

Headwind does the same thing through a different door.  It does not change
``k`` -- air is not water -- so it enters through the resistance curve
instead, and the numerical optimiser below picks it up without being told.
That is the reason this module solves the problem rather than evaluating
the formula: the formula is the check, not the method.

What this does not model
------------------------
Steering and the racing line are :mod:`coxswain.river.route`'s problem and
are held fixed here.  The two do interact -- a crew that has spent its
reserve cannot hold a tight line through Eliot -- and coupling them is the
obvious next step, not something quietly assumed away.

References
----------
.. [S12] Skiba, P.F. et al. (2012) *Modeling the expenditure and
   reconstitution of work capacity above critical power*, Med Sci Sports
   Exerc 44(8):1526-32.
.. [dK99] de Koning, J.J., Bobbert, M.F., Foster, C. (1999) *Determination
   of optimal pacing strategy in track cycling with an energy flow model*,
   J Sci Med Sport 2(3):266-77 -- the variable-course pacing result this
   reproduces from a rowing-specific resistance model.
.. [A07] Atkinson, G., Peacock, O., Passfield, L. (2007) *Variable versus
   constant power strategies during cycling time-trials*, J Sports Sci
   25(9):1001-9 -- the wind and gradient case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .exertion import ROWER_ANAEROBIC_WORK, ROWER_CRITICAL_POWER

__all__ = ["CourseSegment", "PacingPlan", "CoursePacing"]


@dataclass(frozen=True)
class CourseSegment:
    """One stretch of river over which conditions are taken as uniform."""

    #: Along-course length, m.
    length: float
    #: Current along the course, m/s.  **Positive helps the boat.**  On
    #: the Charles the race runs upstream, so this is normally negative.
    current: float = 0.0
    #: Headwind component, m/s.  Positive opposes the boat.
    headwind: float = 0.0
    #: Multiplier on hull resistance from finite depth.  1.0 is deep
    #: water; the shallow-water model supplies the rest.
    depth_factor: float = 1.0
    #: Free-text, so a plan can be read against the river.
    label: str = ""


@dataclass
class PacingPlan:
    """A power schedule and what it costs and buys."""

    powers: np.ndarray            # per rower, per segment, W
    speeds_water: np.ndarray      # m/s
    speeds_ground: np.ndarray     # m/s
    durations: np.ndarray         # s
    reserve: np.ndarray           # W' remaining at each segment end, J
    total_time: float

    @property
    def reserve_spent(self) -> float:
        return float(self.reserve[0] - self.reserve[-1])

    def summary(self) -> dict:
        return {
            "total_time": self.total_time,
            "mean_power": float(np.average(self.powers,
                                           weights=self.durations)),
            "peak_power": float(self.powers.max()),
            "reserve_left": float(self.reserve[-1]),
        }


@dataclass
class CoursePacing:
    """Solve for the power schedule that gets down *this* course fastest.

    ``resistance`` is a callable ``v_water -> newtons`` for the whole boat
    in deep, still air; the segment's ``depth_factor`` multiplies it and
    the headwind is added as an aerodynamic term.  Passing the boat's own
    :func:`~coxswain.hydro.resistance.hull_resistance` keeps this honest;
    passing a power law makes the tests analytic.
    """

    segments: Sequence[CourseSegment]
    resistance: object
    rowers: int = 8
    critical_power: float = ROWER_CRITICAL_POWER
    capacity: float = ROWER_ANAEROBIC_WORK
    #: Recovery time constant below CP, s.  Skiba's differential form.
    recovery_tau: float = 300.0
    #: Frontal area for the headwind term, m^2 -- crew plus hull.
    drag_area: float = 3.22
    air_density: float = 1.225
    #: Efficiency from gate power to power delivered to the water.
    efficiency: float = 0.80
    #: Hard ceiling on per-rower power, W.  Without one the optimiser will
    #: happily ask for a 900 W surge that no masters crew can produce.
    max_power: float = 480.0
    _cache: dict = field(default_factory=dict, repr=False)

    # -- speed from power -------------------------------------------------
    def speed_for_power(self, power: float, segment: CourseSegment) -> float:
        """Speed through water at this per-rower gate power, m/s.

        Bisection on ``R(v) v = P``.  Both sides are monotone in ``v``, so
        the root is unique and bracketing is safe -- the same argument
        :meth:`RouteEvaluator.speed_through_water` makes, and for the same
        reason: a fixed-point sweep oscillates near the shallow-water
        critical region and can return a *higher* speed in shallower water.
        """
        key = (round(float(power), 4), segment.depth_factor, segment.headwind)
        if key in self._cache:
            return self._cache[key]
        delivered = self.efficiency * float(power) * self.rowers

        def excess(speed):
            hull = self.resistance(speed) * segment.depth_factor
            apparent = speed + segment.headwind
            air = (0.5 * self.air_density * self.drag_area
                   * apparent * abs(apparent))
            return (hull + air) * speed - delivered

        low, high = 1e-3, 12.0
        if excess(high) < 0.0:
            return high
        for _ in range(80):
            mid = 0.5 * (low + high)
            if excess(mid) < 0.0:
                low = mid
            else:
                high = mid
        self._cache[key] = 0.5 * (low + high)
        return self._cache[key]

    # -- running a schedule ----------------------------------------------
    def evaluate(self, powers) -> PacingPlan:
        """Time and reserve trace for a given per-segment power schedule."""
        powers = np.asarray(powers, dtype=float).ravel()
        n = len(self.segments)
        if powers.shape != (n,):
            raise ValueError(f"expected {n} powers, got {powers.shape}")

        water = np.empty(n)
        ground = np.empty(n)
        durations = np.empty(n)
        reserve = np.empty(n + 1)
        reserve[0] = self.capacity

        for index, (segment, power) in enumerate(zip(self.segments, powers)):
            speed = self.speed_for_power(power, segment)
            water[index] = speed
            over_ground = speed + segment.current
            if over_ground <= 0.05:
                # A crew that cannot make way against the current takes
                # forever rather than going backwards; the optimiser needs
                # a finite, steep penalty here, not a crash.
                over_ground = 0.05
            ground[index] = over_ground
            durations[index] = segment.length / over_ground

            excess = power - self.critical_power
            if excess > 0.0:
                reserve[index + 1] = (reserve[index]
                                      - excess * durations[index])
            else:
                # Skiba's exponential refill; the asymmetry is the whole
                # tactical point -- spending is fast, getting it back is
                # not.
                #
                # The gap is measured from a reserve floored at zero, not
                # from the raw balance.  A schedule that overspends leaves
                # the balance negative, and refilling from a negative
                # reserve gave a *larger* gap and so a bigger refill: a
                # crew that had blown up recovered faster than one that had
                # not.  That made the reserve non-monotone in power and let
                # the optimiser prefer schedules 15 kJ in deficit precisely
                # because they were infeasible.  The deficit is kept in the
                # trace so infeasibility stays visible; it just no longer
                # earns interest.
                gap = self.capacity - max(reserve[index], 0.0)
                refill = gap * (1.0 - np.exp(-durations[index]
                                             / self.recovery_tau))
                reserve[index + 1] = reserve[index] + refill

        return PacingPlan(powers=powers, speeds_water=water,
                          speeds_ground=ground, durations=durations,
                          reserve=reserve, total_time=float(durations.sum()))

    # -- the two schedules worth comparing --------------------------------
    def flat_power(self, tolerance: float = 1e-3, limit: int = 60) -> float:
        """The one power that just empties the reserve at the finish.

        ``P = CP + W'/T`` with ``T`` solved self-consistently, since the
        power sets the speed which sets the duration which sets the power.
        This is what :func:`~coxswain.crew.exertion.pace_for_course` gives
        and it is the baseline every variable schedule has to beat.
        """
        power = self.critical_power
        for _ in range(limit):
            plan = self.evaluate(np.full(len(self.segments), power))
            updated = self.critical_power + self.capacity / plan.total_time
            if abs(updated - power) < tolerance:
                return float(updated)
            power = 0.5 * (power + updated)
        return float(power)

    def optimise(self, span: float = 60.0, samples: int = 41):
        """Best schedule of the form ``P_i = P0 + span * g(k_i)``.

        The optimality condition makes power a function of ``k =
        v_water/v_ground`` alone, so the schedule has **one** free shape
        parameter once the mean is pinned by the reserve.  Searching that
        one parameter is both far more robust than a free per-segment
        optimisation and far easier to explain to the person who has to
        row it.

        Returns ``(plan, amplitude)``.  The amplitude is how many watts of
        spread the optimum wants between the hardest and easiest water; an
        amplitude near zero says this course does not reward variable
        pacing, which is itself worth knowing.
        """
        baseline = self.flat_power()
        reference = self.evaluate(np.full(len(self.segments), baseline))
        ratio = reference.speeds_water / reference.speeds_ground
        shape = ratio - np.average(ratio, weights=reference.durations)
        if np.allclose(shape, 0.0):
            return reference, 0.0

        shape = shape / np.abs(shape).max()

        def timed(amplitude):
            trial = self._balanced(baseline, shape, float(amplitude))
            return (trial, trial.total_time if trial is not None
                    else float("inf"))

        grid = np.linspace(0.0, span, samples)
        best_plan, best_time, best_amplitude = reference, \
            reference.total_time, 0.0
        for amplitude in grid:
            trial, elapsed = timed(amplitude)
            if elapsed < best_time:
                best_plan, best_time, best_amplitude = trial, elapsed, \
                    float(amplitude)

        # Refine by bisection on the bracket around the grid minimum, so
        # the answer stops depending on ``samples``.  Without this a
        # coarser grid over a wider span reported a *worse* optimum, which
        # is a property of the search and not of the river.
        step = grid[1] - grid[0] if len(grid) > 1 else 0.0
        low = max(0.0, best_amplitude - step)
        high = min(span, best_amplitude + step)
        for _ in range(24):
            if high - low < 0.05:
                break
            left, right = low + (high - low) / 3.0, high - (high - low) / 3.0
            _pl, left_time = timed(left)
            _pr, right_time = timed(right)
            if left_time < right_time:
                high = right
            else:
                low = left
        trial, elapsed = timed(0.5 * (low + high))
        if elapsed < best_time:
            best_plan, best_amplitude = trial, float(0.5 * (low + high))
        return best_plan, best_amplitude

    def _balanced(self, baseline: float, shape: np.ndarray,
                  amplitude: float, tolerance: float = 1.0,
                  limit: int = 60):
        """Shift a shaped schedule until it spends exactly the reserve.

        **Bisection, not Newton.**  The reserve left at the finish falls
        monotonically as the offset rises -- more power everywhere can
        only empty the reserve faster -- so bisection is unconditionally
        convergent.  A Newton step on the same function is not: it
        oscillated across the root and returned schedules overspent by up
        to 10 kJ, which the amplitude search then preferred *because* they
        were illegal and therefore fast.  The search was selecting for
        non-convergence.
        """
        def plan_for(offset):
            powers = np.clip(offset + amplitude * shape,
                             0.5 * self.critical_power, self.max_power)
            return self.evaluate(powers)

        # Bisect on the MINIMUM reserve over the race, not the final one.
        # A crew cannot go into deficit at Eliot and be rescued by an easy
        # last mile, so the final balance is the wrong target: it lets a
        # schedule dip below zero mid-race and still score as legal.  The
        # minimum is also the monotone quantity, which is what makes the
        # bracket safe.
        def worst(offset):
            return float(plan_for(offset).reserve.min())

        low, high = 0.5 * self.critical_power, self.max_power
        if worst(low) < 0.0:
            return None                      # infeasible at any offset
        if worst(high) > 0.0:
            return plan_for(high)            # ceiling binds before the reserve

        plan = None
        for _ in range(limit):
            middle = 0.5 * (low + high)
            plan = plan_for(middle)
            margin = float(plan.reserve.min())
            if abs(margin) < tolerance:
                return plan
            if margin > 0.0:
                low = middle                 # reserve unspent, push harder
            else:
                high = middle
        # Never hand back a schedule that did not converge onto the
        # constraint.  An unconverged plan in deficit is *fast*, and the
        # amplitude search will pick it for that reason -- which is how
        # the 16 s "saving" this module first reported turned out to be a
        # crew 15 kJ overdrawn.
        if plan is None or plan.reserve.min() < -tolerance:
            return None
        return plan
