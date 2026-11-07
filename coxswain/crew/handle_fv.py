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
instantaneous force F = 1557 - 184 V.

**Nordez et al. 2025 (Sports Biomech., full text from the authors, read 2026-11-07):** 14 national-level scullers
(10 men, 4 women), all-out starts in an instrumented double with 266, 287 and 320 cm sculls; Powerline pin force
summed over both gates, handle velocity = gate angular rate x inboard (the model's own definition), both averaged
over the drive (minimum to maximum hand position). Linear, median R^2 0.96: **F0 971 +- 241 N, V0 4.53 +- 0.74 m/s**,
Pmax 1088 W (V0 range 2.9-6.1 m/s; inter-day CV 9 %). The same rowers on a Concept2: F0 1152 N, V0 3.14 m/s -- so
an unidentified abstract's ergometer 3.50 m/s is close, and Sprague's ergometer 6.16 sits above both. As a hold
(PlannedSpeedHold) the on-water line is beta = v / (V0 - v): ~0.5 at 1.6 m/s, ~1.1 at the 2.4 m/s peak.

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
#: Nordez et al. 2025, on water (double scull, drive-averaged handle speed = gate rate x inboard): F0 971 N, V0 4.53 m/s.
NORDEZ2025_ONWATER_F0, NORDEZ2025_ONWATER_V0 = 971.0, 4.53
#: The same study's rowers on a Concept2 ergometer: F0 1152 N, V0 3.14 m/s.
NORDEZ2025_ERGOMETER_F0, NORDEZ2025_ERGOMETER_V0 = 1152.0, 3.14
ERGOMETER_V0 = NORDEZ2025_ERGOMETER_V0


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
        """Nordez et al. 2025's on-water line (full text): V0 4.53 m/s, drive-averaged (SOURCES §222)."""
        return cls(NORDEZ2025_ONWATER_V0, "Nordez et al. 2025: on-water double scull, V0 4.53 m/s (drive-averaged)")

    @classmethod
    def ergometer(cls) -> "HandleForceVelocity":
        """Nordez et al. 2025's rowers on a Concept2: V0 3.14 m/s."""
        return cls(NORDEZ2025_ERGOMETER_V0, "Nordez et al. 2025: Concept2, V0 3.14 m/s")

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
