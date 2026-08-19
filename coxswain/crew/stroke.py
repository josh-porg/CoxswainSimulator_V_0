"""Stroke timing and periodic profile representation.

Timing follows Formaggia et al. (2009) section 5, who take the empirical
fit from Atkinson and Rekers (their refs [2] and [7])::

    tau_a = 0.00015625 (r - 24)^2 - 0.008125 (r - 24) + 0.8      [s]
    T     = 60 / r                                                [s]
    tau_r = T - tau_a                                             [s]

with ``r`` the cadence in strokes per minute.  ``tau_a`` is the active
(drive) phase, ``tau_r`` the recovery.  The fit reproduces the observation
that the recovery is much longer than the drive at low rates and the two
approach parity at racing rates.

Profiles
--------
:class:`FourierProfile` represents any periodic quantity -- a joint angle,
an oarlock force -- as a truncated Fourier series in normalised stroke
phase.  Two reasons:

1. It is exactly periodic and infinitely differentiable, so a quantity
   built from it can never inject the step change in acceleration that the
   legacy piecewise model produced at the catch (measured at 3.4 m/s^2,
   about 1.9 kN of impulsive force on the hull for an eight).
2. It is the natural target for fitting motion-capture data, which is how
   the paper obtained its rower kinematics.  When mocap becomes available,
   :meth:`FourierProfile.fit_samples` consumes it directly and nothing
   downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.taylor import Jet2

__all__ = ["StrokeTiming", "FourierProfile", "DEFAULT_HARMONICS"]

#: Harmonics retained when converting an idealised piecewise profile into a
#: smooth one.  Higher = sharper catch, more high-frequency content in the
#: crew acceleration.
DEFAULT_HARMONICS = 8


@dataclass(frozen=True)
class StrokeTiming:
    """Drive / recovery split as a function of cadence."""

    rate: float  # strokes per minute

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("stroke rate must be positive")
        if self.drive_duration >= self.period:
            raise ValueError(
                f"rate {self.rate} spm gives a drive of "
                f"{self.drive_duration:.3f} s but a stroke period of only "
                f"{self.period:.3f} s; the empirical fit for tau_a is not "
                "valid at this cadence"
            )

    @property
    def period(self) -> float:
        """Stroke period ``T = 60 / r`` in seconds."""
        return 60.0 / self.rate

    @property
    def drive_duration(self) -> float:
        """Active-phase duration ``tau_a`` in seconds."""
        offset = self.rate - 24.0
        return 0.00015625 * offset ** 2 - 0.008125 * offset + 0.8

    @property
    def recovery_duration(self) -> float:
        """Recovery duration ``tau_r = T - tau_a`` in seconds."""
        return self.period - self.drive_duration

    @property
    def drive_fraction(self) -> float:
        """Fraction of the stroke spent on the drive, in ``(0, 1)``."""
        return self.drive_duration / self.period

    @property
    def ratio(self) -> float:
        """Recovery-to-drive time ratio, the number coaches quote."""
        return self.recovery_duration / self.drive_duration

    def phase(self, t):
        """Normalised stroke phase in ``[0, 1)``; 0 is the catch."""
        return np.mod(np.asarray(t, dtype=float), self.period) / self.period

    def is_drive(self, t):
        """Boolean mask: is the blade in the water at time ``t``?"""
        return self.phase(t) < self.drive_fraction


class FourierProfile:
    """A periodic scalar of stroke phase, as a truncated Fourier series.

    ``f(u) = a[0] + sum_k ( a[k] cos(2 pi k u) + b[k] sin(2 pi k u) )``

    where ``u`` is normalised stroke phase.  Evaluation returns a
    :class:`~coxswain.core.taylor.Jet2` carrying the value and its first two
    derivatives *with respect to time*, given the stroke period.
    """

    def __init__(self, cos_coefficients, sin_coefficients, period: float):
        self.cos_coefficients = np.asarray(cos_coefficients, dtype=float)
        self.sin_coefficients = np.asarray(sin_coefficients, dtype=float)
        if self.cos_coefficients.shape != self.sin_coefficients.shape:
            raise ValueError("cosine and sine coefficient arrays must match")
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = float(period)

    @property
    def n_harmonics(self) -> int:
        """Number of harmonics excluding the mean term."""
        return len(self.cos_coefficients) - 1

    @property
    def mean(self) -> float:
        return float(self.cos_coefficients[0])

    # -- evaluation ------------------------------------------------------
    def __call__(self, t) -> Jet2:
        """Evaluate at time ``t`` (seconds), returning value/rate/accel."""
        t = np.asarray(t, dtype=float)
        omega = 2.0 * np.pi / self.period

        # zeros_like(t) broadcasts correctly for 0-d and n-d inputs alike
        value = np.zeros_like(t) + self.cos_coefficients[0]
        first = np.zeros_like(value)
        second = np.zeros_like(value)

        for k in range(1, len(self.cos_coefficients)):
            a, b = self.cos_coefficients[k], self.sin_coefficients[k]
            if a == 0.0 and b == 0.0:
                continue
            w = k * omega
            cos_wt, sin_wt = np.cos(w * t), np.sin(w * t)
            value = value + a * cos_wt + b * sin_wt
            first = first + w * (-a * sin_wt + b * cos_wt)
            second = second + w ** 2 * (-a * cos_wt - b * sin_wt)

        return Jet2(value, first, second)

    def value_at_phase(self, phase):
        """Evaluate as a function of normalised phase rather than time."""
        return self(np.asarray(phase, dtype=float) * self.period).value

    # -- construction ----------------------------------------------------
    @classmethod
    def fit_samples(cls, samples, period: float,
                    n_harmonics: int = DEFAULT_HARMONICS) -> "FourierProfile":
        """Fit to uniformly spaced samples over exactly one period.

        ``samples[i]`` is the value at phase ``i / len(samples)``.  This is
        the entry point for motion-capture data: resample one stroke onto a
        uniform phase grid and hand it here.
        """
        samples = np.asarray(samples, dtype=float)
        n = len(samples)
        if n < 2 * n_harmonics + 1:
            raise ValueError(
                f"need at least {2 * n_harmonics + 1} samples to fit "
                f"{n_harmonics} harmonics, got {n}"
            )

        spectrum = np.fft.rfft(samples) / n
        keep = min(n_harmonics + 1, len(spectrum))

        cos_coefficients = np.zeros(n_harmonics + 1)
        sin_coefficients = np.zeros(n_harmonics + 1)
        cos_coefficients[0] = spectrum[0].real
        for k in range(1, keep):
            cos_coefficients[k] = 2.0 * spectrum[k].real
            sin_coefficients[k] = -2.0 * spectrum[k].imag

        return cls(cos_coefficients, sin_coefficients, period)

    @classmethod
    def from_catch_finish(cls, catch_value: float, finish_value: float,
                          timing: StrokeTiming,
                          n_harmonics: int = DEFAULT_HARMONICS,
                          n_samples: int = 512) -> "FourierProfile":
        """Smooth profile sweeping catch -> finish -> catch.

        The idealised target is a raised cosine on each phase, reaching
        ``finish_value`` exactly at the end of the drive and returning to
        ``catch_value`` at the next catch, with zero slope at both ends.
        Truncating its spectrum to ``n_harmonics`` rounds off the
        discontinuity in curvature at the catch, which is both numerically
        necessary and physically right -- a rower's segment accelerations
        are large at the catch but finite.
        """
        phase = np.arange(n_samples) / n_samples
        drive = timing.drive_fraction

        target = np.empty(n_samples)
        on_drive = phase < drive
        target[on_drive] = catch_value + (finish_value - catch_value) * 0.5 * (
            1.0 - np.cos(np.pi * phase[on_drive] / drive)
        )
        rec = phase[~on_drive]
        target[~on_drive] = finish_value + (catch_value - finish_value) * 0.5 * (
            1.0 - np.cos(np.pi * (rec - drive) / (1.0 - drive))
        )

        return cls.fit_samples(target, timing.period, n_harmonics)

    @classmethod
    def constant(cls, value: float, period: float) -> "FourierProfile":
        return cls([float(value)], [0.0], period)

    def with_period(self, period: float) -> "FourierProfile":
        """Same shape, re-timed to a different stroke period."""
        return FourierProfile(self.cos_coefficients, self.sin_coefficients,
                              period)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"FourierProfile(n_harmonics={self.n_harmonics}, "
                f"mean={self.mean:.4f}, period={self.period:.4f})")
