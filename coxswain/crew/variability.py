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

__all__ = ["CrewVariability", "ELITE", "CLUB", "JUNIOR",
           "for_skill", "skill_label", "SKILL_ANCHORS"]


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

    def apply(self, boat, base=None) -> None:
        """Draw one stroke's worth of variation and set it on ``boat``.

        ``base`` is a per-seat multiplier the draw is applied *on top of*
        -- a crew that is not at the model's reference power, or one with
        a deliberately weakened seat.  It matters because this method
        **assigns** ``power_scales`` rather than scaling them, so without
        it a caller that set the crew to 66% of reference power and then
        applied scatter silently got a full-power crew back.  That is not
        hypothetical: it made a masters eight row at 5.41 m/s instead of
        4.21 and turned the whole consistency comparison into a
        comparison between two different crews.

        Assignment is still the right default, because the per-stroke
        hook calls this every stroke and multiplying into the previous
        draw would compound the scatter without limit.
        """
        power, timing = self.draw(boat.n_seats)
        if base is not None:
            power = power * np.asarray(base, dtype=float)
        boat.power_scales = power
        boat.phase_offsets = timing / boat.timing.period

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._power_bias = None
        self._timing_bias = None


#: [K-VAR]: 2.3% force variation for an elite sculler.
#: Skill anchors: ``(skill, power_sigma, timing_sigma, label)``.
#:
#: Built from the presets below rather than from a fresh curve, so a
#: slider set to "elite" gets exactly the elite numbers and not whatever
#: an interpolation happened to produce there.
#:
#: Note that timing scatter grows FASTER than power scatter across the
#: three presets -- the ratio runs 0.35, 0.51, 0.59 from elite to junior
#: -- so both are interpolated independently.  Tying timing to power by
#: a single ratio would have flattened a real pattern: a less
#: experienced crew is disproportionately worse at going together than
#: at pulling evenly, which is what a coxswain actually hears.
#:
#: Two of the five anchors are measured [K-VAR], one is between them,
#: and two are not measured at all.  "Ideal" is zero by definition: it
#: is the uniformity assumed by every result in this project computed
#: before variability existed.  "Novice" is extrapolation along the
#: elite-to-junior slope and is the one number here nobody has measured.
SKILL_ANCHORS = (
    (0.00, 0.110, 0.075, "novice"),      # extrapolated
    (0.35, 0.051, 0.030, "junior"),      # [K-VAR], measured
    (0.55, 0.035, 0.018, "club"),        # between the measured points
    (0.75, 0.023, 0.008, "elite"),       # [K-VAR], measured
    (1.00, 0.000, 0.000, "ideal"),       # zero by definition
)

#: Persistent per-rower bias as a fraction of the stroke-to-stroke
#: scatter.  A novice crew is not merely inconsistent, it is unmatched:
#: one rower is simply stronger, or habitually early.  That is a
#: different defect -- bias can be rigged or seated around, scatter
#: cannot -- and it is kept well under the scatter because any crew that
#: has rowed together at all has had the worst of its bias seated out.
BIAS_PER_SCATTER = 0.45


def for_skill(skill: float, seed: int = 0) -> "CrewVariability":
    """A crew of a given skill, ``0.0`` novice to ``1.0`` ideal."""
    skill = float(np.clip(skill, 0.0, 1.0))
    points = np.array([a[0] for a in SKILL_ANCHORS])
    power = float(np.interp(skill, points,
                            [a[1] for a in SKILL_ANCHORS]))
    timing = float(np.interp(skill, points,
                             [a[2] for a in SKILL_ANCHORS]))
    return CrewVariability(
        power_sigma=power, timing_sigma=timing,
        power_bias_sigma=power * BIAS_PER_SCATTER,
        timing_bias_sigma=timing * BIAS_PER_SCATTER,
        seed=seed,
    )


def skill_label(skill: float) -> str:
    """The nearest named anchor, for a menu to show."""
    skill = float(np.clip(skill, 0.0, 1.0))
    return min(SKILL_ANCHORS, key=lambda a: abs(a[0] - skill))[3]


ELITE = CrewVariability(power_sigma=0.023, timing_sigma=0.008)

#: Between the two measured points.
CLUB = CrewVariability(power_sigma=0.035, timing_sigma=0.018)

#: [K-VAR]: 5.1% force variation for a junior.
JUNIOR = CrewVariability(power_sigma=0.051, timing_sigma=0.030)
