"""Stroke-to-stroke variation: crews are not metronomes.

Every rower in this model has so far pulled exactly their nominal load at
exactly the nominal time, every stroke.  Real crews do neither, and the
deviation is not noise to be averaged away -- it is the disturbance input
to two things this model has shown to be marginal:

* **roll**, which §15-16 show is an unstable mode held through the recovery
  by about 5% of the drive's authority;
* **heading**, where §21 shows a port/starboard timing split of 65 ms
  spends the entire recovery balance authority, and a side power imbalance
  is a standing yaw bias.

It is also the entry point to the stochastic optimal control problem: a
plan optimised for the mean crew is not the right plan for a crew whose
parameters are drawn afresh every stroke.

Calibration
-----------
Force and power variability are measured.  [K-VAR] reports, comparing an
elite sculler with a junior:

| quantity | elite | junior |
|---|---|---|
| force variation | **2.3%** | **5.1%** |
| work per stroke variation | 1.3% | 4.7% |

and, over a training block, force-curve consistency improving from 10-15%
to 4-6% and work-per-stroke variation from 6.8% to 2.7%.  So the spread
between a novice and an international is roughly a factor of two to five
in force scatter, and it is trainable.

**Timing scatter is not calibrated here.**  [K-VAR] notes that the
coefficient of variation is undefined for timing variables, whose means
pass through zero, so it is reported as standard deviation or range and no
single figure transfers cleanly.  The default below is therefore an
*inference*, not a measurement: §21 shows 65 ms of port/starboard split
exhausts the recovery balance authority, and crews demonstrably do sit
boats, so per-rower timing scatter must be well inside that.  It is
flagged rather than dressed up, and it is the obvious thing to measure
next.

References
----------
[K-VAR] Kleshnev, V. "Rowing Science: New Analysis of Variability of
        Rower's Technique", parts 1-3, row2k.
        https://www.row2k.com/features/6489/
        https://www.row2k.com/features/6503/
        https://www.row2k.com/features/6521/
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CrewVariability", "ELITE", "CLUB", "JUNIOR"]


@dataclass
class CrewVariability:
    """Per-rower, per-stroke variation in power and timing.

    Draws a fresh power multiplier and phase offset for every rower at the
    start of each stroke.  Within a stroke the rower is deterministic --
    they commit to a stroke and execute it, which is both physically right
    and what makes the result usable in a collocation transcription.
    """

    #: Standard deviation of the per-stroke power multiplier, as a
    #: fraction.  [K-VAR]: 0.023 elite, 0.051 junior.
    power_sigma: float = 0.030
    #: Standard deviation of the per-stroke phase offset, in seconds.
    #: See the module docstring -- inferred, not measured.
    timing_sigma: float = 0.015
    #: Persistent per-rower bias, as distinct from stroke-to-stroke
    #: scatter.  A rower who is consistently strong or consistently early
    #: is a different problem from one who is inconsistent: bias can be
    #: rigged or seated around, scatter cannot.
    power_bias_sigma: float = 0.0
    timing_bias_sigma: float = 0.0

    seed: int = 0

    def __post_init__(self):
        for name in ("power_sigma", "timing_sigma", "power_bias_sigma",
                     "timing_bias_sigma"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self._rng = np.random.default_rng(self.seed)
        self._power_bias = None
        self._timing_bias = None

    # -- persistent differences ------------------------------------------
    def biases(self, n_seats: int):
        """Fixed per-rower offsets, drawn once and kept."""
        if self._power_bias is None or len(self._power_bias) != n_seats:
            self._power_bias = self._rng.normal(0.0, self.power_bias_sigma,
                                                n_seats)
            self._timing_bias = self._rng.normal(0.0, self.timing_bias_sigma,
                                                 n_seats)
        return self._power_bias, self._timing_bias

    # -- per-stroke draw ---------------------------------------------------
    def draw(self, n_seats: int):
        """``(power_scales, phase_offsets_seconds)`` for one stroke."""
        power_bias, timing_bias = self.biases(n_seats)
        power = 1.0 + power_bias + self._rng.normal(0.0, self.power_sigma,
                                                    n_seats)
        timing = timing_bias + self._rng.normal(0.0, self.timing_sigma,
                                                n_seats)
        # A rower cannot pull negative; the truncation matters only in the
        # tail of an implausibly scattered crew, but an unbounded Gaussian
        # would eventually produce one.
        return np.maximum(power, 0.0), timing

    def apply(self, boat) -> None:
        """Draw one stroke's worth of variation and set it on ``boat``."""
        power, timing = self.draw(boat.n_seats)
        boat.power_scales = power
        boat.phase_offsets = timing / boat.timing.period

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._power_bias = None
        self._timing_bias = None


#: [K-VAR]: 2.3% force variation for an elite sculler.
ELITE = CrewVariability(power_sigma=0.023, timing_sigma=0.008)

#: Between the two measured points.
CLUB = CrewVariability(power_sigma=0.035, timing_sigma=0.018)

#: [K-VAR]: 5.1% force variation for a junior.
JUNIOR = CrewVariability(power_sigma=0.051, timing_sigma=0.030)
