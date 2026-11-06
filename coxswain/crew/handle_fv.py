"""The crew's handle force-velocity relation: a research option of the dynamic oar (SOURCES sec. 218).

The clock crew pulls with a torque that depends on the oar angle only, ``T_shape(theta)``. A rower's force falls as the
handle speeds up: rowing's handle force-velocity relation is linear, ``F = F0 (1 - v / V0)`` (Sprague et al. 2007, Med.
Sci. Sports Exerc., pull force-velocity r^2 0.99, 11 collegiate men; on water also linear, median R^2 0.96, with V0
about 31 % higher than on the ergometer). At a submaximal effort the rower's activation sets the level and the relation
sets the slope, so about the speed the crew plans at each angle, ``v_ref(theta)`` (the boat's own oar sweep), the pull is

    T(theta, theta_dot) = T_shape(theta) (V0 - v) / (V0 - v_ref(theta)),   v = |theta_dot| * inboard

-- the shape torque when the oar moves as the crew plans, less when it runs ahead, more when it is held back, along the
published slope. Without it the crew has inertia but no damping, and a bending shaft rings on the crew's reflected
inertia at the drive's own period (SOURCES sec. 218).

``v0`` is the handle speed at zero force, m/s. **Sprague et al. 2007 (full text, read 2026-11-05):** 11 collegiate men,
maximal 5-stroke trials on an inertial ergometer at several loads; pull force against pull-averaged handle velocity is
linear, F = 1214 - 197 V N (r^2 0.993; the Hill fit is as good), so V0 = 6.16 m/s and the slope ~197 N s/m; peak
instantaneous force F = 1557 - 184 V. A second figure, V0 3.50 m/s (F0 1322 N, "a Concept2 study of 59 rowers"), is
from an abstract whose source could not be identified: kept only as the low end of a sweep. On water V0 is higher
still (Nordez et al. 2025, Sports Biomech.: the ergometer's 31 % lower; abstract, full text requested).

Caveat: these are maximal relations across trials (each point a different load), with velocity averaged over the pull.
Using one as the within-stroke impedance of a submaximal rower (the slope scaled by the activation, through the
ratio above) is the model's assumption, not something these studies measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Sprague et al. 2007: pull force F0 1214 N, slope -197 N s/m (pull-averaged handle velocity), so V0 = 6.16 m/s.
SPRAGUE2007_F0, SPRAGUE2007_SLOPE = 1214.0, 197.0
SPRAGUE2007_V0 = SPRAGUE2007_F0 / SPRAGUE2007_SLOPE
#: Handle speed at zero force on the ergometer from an unidentified abstract (Concept2, 59 rowers), m/s: a low bound.
ERGOMETER_V0 = 3.50
#: The on-water study's V0 relative to the ergometer's: the ergometer's is 31 % lower (abstract).
ONWATER_OVER_ERGOMETER = 1.0 / 0.69


@dataclass(frozen=True)
class HandleForceVelocity:
    """A linear handle force-velocity relation with zero force at ``v0`` (m/s)."""

    v0: float
    source: str = ""

    def __post_init__(self):
        if not self.v0 > 0.0:
            raise ValueError("v0 must be positive, got %r" % (self.v0,))

    @classmethod
    def sprague2007(cls) -> "HandleForceVelocity":
        """Sprague et al. 2007's pull force-velocity line on an ergometer (full text): V0 6.16 m/s."""
        return cls(SPRAGUE2007_V0, "Sprague et al. 2007: F = 1214 - 197 V (ergometer, pull-averaged)")

    @classmethod
    def onwater(cls) -> "HandleForceVelocity":
        """The ergometer's V0 scaled to water by the on-water study's ratio (SOURCES sec. 218)."""
        return cls(ERGOMETER_V0 * ONWATER_OVER_ERGOMETER, "on-water: Concept2 V0 3.50 m/s / 0.69 (abstracts)")

    @classmethod
    def ergometer(cls) -> "HandleForceVelocity":
        return cls(ERGOMETER_V0, "Concept2 V0 3.50 m/s (abstract)")

    def factor(self, speed: float, planned: float) -> float:
        """``(V0 - v) / (V0 - v_ref)``, never below zero: the pull at handle speed ``speed`` relative to the planned
        speed's. A planned speed at or beyond V0 is refused rather than divided by."""
        if planned >= self.v0:
            raise ValueError("the planned handle speed %.2f m/s reaches V0 %.2f m/s" % (planned, self.v0))
        return max(self.v0 - abs(float(speed)), 0.0) / (self.v0 - abs(float(planned)))

    def key(self) -> tuple:
        return (float(self.v0),)


@dataclass(frozen=True)
class PlannedSpeedHold:
    """The crew's hold on its planned handle speed (SOURCES §220): the pull scaled by ``1 + beta (1 - v / v_ref)``,
    a fractional speed error costing ``beta`` times as much fractional force; never below zero. The same interface as
    :class:`HandleForceVelocity` (``factor``, ``key``), so either drives ``DynamicOarSimulator(handle_fv=)``.

    A linear force-velocity at V0 is this with beta = v_ref / (V0 - v_ref): Sprague et al. 2007's line gives beta
    ~0.35-0.65 at on-water handle speeds (Kleshnev 2005: 1.59 m/s mean, 2.36 peak), a single muscle's Hill curve
    ~0.7-1.0. beta = 1 stops the bending shaft ringing on both athletes in rung 2 (SOURCES §220): the top of the
    force-velocity range, studied rather than sourced."""

    beta: float
    source: str = ""

    def __post_init__(self):
        if not self.beta >= 0.0:
            raise ValueError("beta must be non-negative, got %r" % (self.beta,))

    def factor(self, speed: float, planned: float) -> float:
        if planned <= 1e-9:
            return 1.0
        return max(1.0 + self.beta * (1.0 - abs(float(speed)) / abs(float(planned))), 0.0)

    def key(self) -> tuple:
        return ("hold", float(self.beta))

    @property
    def v0(self) -> float:
        """For labels: the equivalent zero-force speed in units of the planned speed, (1 + beta) / beta."""
        return float("inf") if self.beta == 0.0 else (1.0 + self.beta) / self.beta


class PlannedSpeed:
    """The crew's planned oar rate at each drive angle, from the boat's own sweep: ``rate(theta)`` over the drive."""

    def __init__(self, boat, samples: int = 400):
        sweep, timing = boat.oar_sweep, boat.timing
        tau = np.linspace(0.0, float(timing.drive_fraction) * float(timing.period), samples)
        angle = np.array([float(sweep(t, timing)) for t in tau])
        rate = np.array([abs(float(sweep.rate(t, timing))) for t in tau])
        order = np.argsort(angle)
        self._angle, self._rate = angle[order], rate[order]

    def rate(self, angle: float) -> float:
        return float(np.interp(float(angle), self._angle, self._rate))
