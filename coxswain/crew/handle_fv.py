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

``v0`` is the handle speed at zero force, m/s. The ergometer figure is 3.50 +/- 0.33 m/s (a Concept2 study of 59 rowers,
F0 1322 N; abstract), so on water ~3.5 / 0.69 = 5.1 m/s by the on-water study's ratio (abstract; full texts not yet
read): a range to be swept, not a fitted number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Handle speed at zero force on the ergometer, m/s (Concept2, 59 rowers; abstract).
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
