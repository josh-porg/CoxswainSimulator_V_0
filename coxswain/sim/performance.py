r"""How boat speed responds to a change, with the exponents measured.

Young [Y09]_ derives a set of scaling laws for a racing eight and ranks
the things a crew or a builder could change by how much speed each buys.
Every one of them descends from a single assumption -- that total
resistance is ``R = C v^2 S`` with ``C`` constant -- so that power is
``P = C v^3 S`` and

.. math::

    \frac{v_f}{v_0} = \left(\frac{P_f}{P_0}\right)^{1/3}

with the rest following by feeding a parameter into ``P`` or into ``S``:
stroke rate through eq. (33) at constant work per stroke, stroke length
through eq. (29), and weight through eq. (24) via the wetted area.

**That constant is gone from this project.**  Wave resistance is now
Michell's integral over the hull's own offsets
(:mod:`coxswain.hydro.michell`) and friction follows ITTC-57, so ``C``
varies with speed and ``1/3`` is no longer exact.  Neither is Young's
``1/9`` for weight, which additionally assumes ``S`` grows as ``W^{1/3}``
by geometric similarity rather than measuring it on the hull.

So nothing here hard-codes Young's exponents.  They are **measured** from
whatever resistance model the boat is carrying:

.. math::

    n(v) = \frac{d \ln R}{d \ln v}, \qquad
    \frac{d \ln v}{d \ln P} = \frac{1}{1 + n(v)}

Young's ``1/3`` is the special case ``n = 2``, and any departure from it
is a prediction of the better drag model rather than a discrepancy with
his.

Why the rate law was worth the trouble
--------------------------------------
Varying rate at constant *power* returns **+0.00% per stroke per minute**
in this model, against Holt et al.'s measured +0.6 to +1.1% [H20]_, and
that gap went into the notes as a defect.  Young's eq. (33) says it may
not be one.  His law holds work per stroke fixed, so power rises with
rate, and predicts

.. math::

    \frac{d \ln v}{d\,\mathrm{SR}} = \frac{1}{3\,\mathrm{SR}}

which over Holt's own range of 32.8 to 38.1 spm gives **+0.87 to +1.02%
per spm** -- inside their +0.6 to +1.1% and near the centre of it.  That
is the *power* channel.  If Holt's adjustment for power left any of it
standing, part of their residual rate effect is this and not physiology.

Two readings survive the arithmetic, and this module deliberately does
not choose between them:

* the rate effect is real and mechanical, the model is missing a rower,
  and :mod:`coxswain.crew.muscle` is the candidate; or
* the rate effect is the power channel incompletely removed, and a model
  reporting +0.00% at genuinely constant power is right.

What discriminates is whether the *measured* exponent below reproduces
Young's 1/3 where his assumptions hold.  A model that gets the
constant-work law right has earned the benefit of the doubt on the
constant-power one.

References
----------
.. [Y09] Young, S. F. (2009) *Effects of Various Inefficiencies in Rowing
   on Shell Speed*, BSc thesis, MIT, eqs. (18)-(33).
.. [H20] Holt, A. C. et al. (2020) *Technical determinants of on-water
   rowing performance*, Frontiers in Sports and Active Living 2:589013.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hydro.resistance import hull_resistance

__all__ = ["SpeedResponse", "YOUNG_POWER_EXPONENT", "YOUNG_AREA_EXPONENT",
           "YOUNG_WEIGHT_EXPONENT", "YOUNG_AREA_FROM_WEIGHT",
           "young_rate_slope", "HOLT_RATE_SLOPE", "HOLT_RATE_RANGE"]

GRAVITY = 9.80665

#: Young eqs. (19), (33): ``v ~ P^(1/3)`` when ``R = C v^2 S`` exactly.
YOUNG_POWER_EXPONENT = 1.0 / 3.0
#: Young eq. (19): ``v ~ S^(-1/3)``, all resistance scaling with area.
YOUNG_AREA_EXPONENT = -1.0 / 3.0
#: Young eq. (23): ``S ~ W^(1/3)``, geometric similarity.
YOUNG_AREA_FROM_WEIGHT = 1.0 / 3.0
#: Young eq. (24): ``v ~ W^(-1/9)``, the two above composed.
YOUNG_WEIGHT_EXPONENT = -1.0 / 9.0

#: Holt table 3: fractional speed change per spm, after their adjustment
#: for power.  Reported as 0.6-1.1%.
HOLT_RATE_SLOPE = (0.006, 0.011)
#: The rates their crews actually raced at, singles through pairs.
HOLT_RATE_RANGE = (32.8, 38.1)


def young_rate_slope(rate):
    r"""Fractional speed gain per stroke per minute, Young eq. (33).

    At constant work per stroke ``P \propto SR``, so ``v \propto
    SR^{1/3}`` and ``d ln v / d SR = 1 / (3 SR)``.  A *fraction* per spm;
    multiply by 100 for Holt's percentages.
    """
    return 1.0 / (3.0 * np.asarray(rate, dtype=float))


@dataclass
class SpeedResponse:
    """Measured sensitivity of speed to power, wetted area and weight.

    Wraps one boat and differentiates its own resistance model, so the
    exponents reported are the ones the simulator obeys rather than the
    ones Young's constant-``C`` algebra implies.
    """

    boat: object
    wave_table: object = None
    #: Relative step for the logarithmic derivatives.  Large enough to
    #: clear the wave table's interpolation nodes, small enough that the
    #: curvature of ``R(v)`` does not bias the central difference.
    step: float = 0.02

    def __post_init__(self):
        if self.wave_table is None:
            self.wave_table = getattr(self.boat, "wave_table", None)

    # -- the hull, at a given displacement -------------------------------
    def _submerged(self, mass_scale: float = 1.0):
        boat = self.boat
        heave = boat.mesh.equilibrium_heave(
            float(mass_scale) * boat.total_mass, rho=boat.water.density)
        return boat.mesh.submerged(
            np.array([0.0, 0.0, heave]), np.zeros(3),
            rho=boat.water.density, gravity=GRAVITY, water_level=0.0)

    def breakdown(self, speed: float, mass_scale: float = 1.0) -> dict:
        boat = self.boat
        _force, detail = hull_resistance(
            np.array([float(speed), 0.0, 0.0]), self._submerged(mass_scale),
            mean_wetted_length=boat.length, water=boat.water,
            coefficients=boat.resistance, wave_table=self.wave_table)
        return detail

    def resistance(self, speed: float, mass_scale: float = 1.0) -> float:
        return float(self.breakdown(speed, mass_scale)["total_longitudinal"])

    def wetted_area(self, mass_scale: float = 1.0) -> float:
        return float(self._submerged(mass_scale).wetted_area)

    # -- the exponents ---------------------------------------------------
    def drag_exponent(self, speed: float) -> float:
        """``n = d ln R / d ln v``.  Young assumes exactly 2 everywhere."""
        h = self.step
        low = self.resistance(speed * (1.0 - h))
        high = self.resistance(speed * (1.0 + h))
        return float(np.log(high / low) / np.log((1.0 + h) / (1.0 - h)))

    def power_exponent(self, speed: float) -> float:
        """``d ln v / d ln P``.  Young's 1/3 is the ``n = 2`` case."""
        return 1.0 / (1.0 + self.drag_exponent(speed))

    def rate_slope(self, speed: float, rate: float) -> float:
        """Fractional speed gain per spm at constant work per stroke.

        Young eq. (33) with the measured exponent standing in for 1/3.
        This is the quantity to hold against Holt's +0.6 to +1.1% per spm,
        and it is a **power** effect: work per stroke is fixed, so raising
        the rate raises the power.  Nothing here claims a rate effect at
        constant power; see the module docstring.
        """
        return self.power_exponent(speed) / float(rate)

    def area_exponent(self, speed: float) -> float:
        """``d ln v / d ln S``.  Young's -1/3 assumes all drag scales with S.

        Here only the viscous term does.  The shape term scales with
        transverse area and Michell's wave term with the offsets, so
        wetted area is a lever on a *fraction* of the total and the
        exponent is smaller in magnitude than Young's by exactly that
        fraction.
        """
        detail = self.breakdown(speed)
        share = detail["viscous"] / detail["total_longitudinal"]
        return -float(share) / (1.0 + self.drag_exponent(speed))

    def area_from_weight(self, span: float = 0.10) -> float:
        r"""``d ln S / d ln W`` on this hull.  Similarity says 1/3.

        Young reaches 1/3 by assuming vertical topsides and a rectangular
        waterplane (his eqs. 21-23).  A racing shell has a fine, curved
        waterplane whose area barely grows as it sinks, so the true
        exponent is well below 1/3 and the weight penalty correspondingly
        smaller.  Measured by actually sinking the mesh.
        """
        h = float(span)
        low = self.wetted_area(1.0 - h)
        high = self.wetted_area(1.0 + h)
        return float(np.log(high / low) / np.log((1.0 + h) / (1.0 - h)))

    def weight_exponent(self, speed: float, span: float = 0.10) -> float:
        """``d ln v / d ln W``, both halves measured.  Young's is -1/9.

        The hull is sunk to the heavier displacement and the resulting
        resistance change differentiated, so neither ``dS/dW`` nor
        ``dv/dS`` is assumed.
        """
        h = float(span)
        low = self.resistance(speed, 1.0 - h)
        high = self.resistance(speed, 1.0 + h)
        slope = float(np.log(high / low) / np.log((1.0 + h) / (1.0 - h)))
        return -slope / (1.0 + self.drag_exponent(speed))

    def seconds_per_percent(self, speed: float, race_time: float,
                            exponent: float = None) -> float:
        """Seconds saved over ``race_time`` per 1% more power.

        ``t \\propto v^{-1}`` at fixed distance, so a fractional speed
        gain of ``e * 0.01`` returns ``race_time * e * 0.01`` seconds to
        first order.  This is the headline lever the time budget quotes,
        expressed straight from the exponent.
        """
        if exponent is None:
            exponent = self.power_exponent(speed)
        return float(race_time) * float(exponent) * 0.01
