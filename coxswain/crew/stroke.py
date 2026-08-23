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

__all__ = ["StrokeTiming", "FourierProfile", "FourierTrack",
           "DEFAULT_HARMONICS", "DEFAULT_FLATNESS"]

#: Harmonics retained when converting an idealised piecewise profile into a
#: smooth one.  Higher = sharper catch, more high-frequency content in the
#: crew acceleration.
DEFAULT_HARMONICS = 8

#: How close to constant-rate each phase of the stroke is; see
#: :meth:`FourierProfile._ramp`.  Calibrated so an eight's hull speed
#: fluctuation matches measured boat-mounted accelerometer data.
DEFAULT_FLATNESS = 0.75




def uniform_traverse_warp(phases, com_x, drive_fraction: float,
                          blend: float = 1.0,
                          dwell: float = 0.25) -> np.ndarray:
    """Phase warp that makes the crew traverse at near-constant rate.

    Given the centre-of-mass longitudinal position ``com_x`` sampled at
    ``phases`` (one full stroke, uniform grid), returns warped phases
    ``w(phi)`` -- same grid, same endpoints, catch and finish fixed -- such
    that re-sampling the motion through ``w`` yields ``|d com_x / dt|``
    close to constant *within each phase*.

    This is arc-length reparameterisation, done separately for the drive
    and the recovery so their durations (and the force profile's clock)
    are untouched: ``w' proportional to 1 / max(|x'|, dwell * mean|x'|)``,
    normalised per phase.

    Why: the hull's speed fluctuation is set almost entirely by the peak
    of the crew's centre-of-mass velocity (momentum conservation), and the
    measured fluctuation of real boats sits at the *constant-rate floor* --
    37.3% measured against 36.5% for a perfectly uniform traverse and 58%
    for the keyframe-spline reconstruction.  Real rowers move at
    near-constant slide speed; four keyframes cannot say so, and the
    smooth interpolant between them is humped.  The warp removes the hump
    without touching any measured posture.

    ``blend`` interpolates identity (0) to fully uniform (1).
    ``dwell`` floors the traverse speed used in the reparameterisation, as
    a fraction of the phase's mean speed: the centre of mass reverses at
    the catch and the finish, where ``|x'| -> 0`` and its reciprocal would
    demand the reversal take no time at all.  The floor spends a finite,
    reportable fraction of the phase in the reversal instead.
    """
    phases = np.asarray(phases, dtype=float)
    com_x = np.asarray(com_x, dtype=float)
    rate = np.gradient(com_x, phases)
    warped = np.array(phases)

    for low, high in ((0.0, drive_fraction), (drive_fraction, 1.0)):
        inside = (phases >= low) & (phases <= high)
        if inside.sum() < 4:
            continue
        speed = np.abs(rate[inside])
        floor = dwell * max(speed.mean(), 1e-12)
        speed = np.maximum(speed, floor)
        # cumulative arc length: how much of the traverse is done by each
        # phase.  The floored speed keeps the reversal from taking zero
        # time in the reparameterised motion.
        arc = np.concatenate([[0.0], np.cumsum(
            0.5 * (speed[1:] + speed[:-1]) * np.diff(phases[inside]))])
        if arc[-1] <= 0.0:
            continue
        # constant rate means arc grows linearly with time, so the posture
        # to show at phase ``phi`` is the one whose accumulated arc equals
        # the elapsed fraction: ``w = A^{-1}(linear)``.
        target = arc[-1] * (phases[inside] - low) / (high - low)
        warped[inside] = np.interp(target, arc, phases[inside])

    return (1.0 - float(blend)) * phases + float(blend) * warped


def drive_timing_warp(phases, drive_fraction: float, lag: float):
    """Retime the drive so the crew reaches peak speed later in it.

    ``w(u) = u - lag * f * sin(pi u / f)`` on the drive, identity on the
    recovery.  The catch, the finish and the next catch are fixed points,
    so keyframe postures and the force profile's clock are untouched and
    the crew's travel is preserved exactly -- this is a reparameterisation
    of the same path, not a change to it.

    Positive ``lag`` samples the underlying motion *earlier* in mid-drive,
    which delays the crew's progress and pushes the peak of their
    centre-of-mass speed later in the drive.

    Why this exists
    ---------------
    The model's crew reaches peak centre-of-mass speed at 37% of the
    drive.  Kleshnev measures peak handle velocity at **60%**.  Past its
    peak the crew is decelerating, and a decelerating crew's reaction
    *pushes the hull forward* -- so from 37% onward the model's crew
    reaction adds to blade thrust instead of opposing it.  Measured on
    the water the two very nearly cancel: mean absolute hull acceleration
    through the drive is 0.71 m/s^2, against 3.47 in the model.

    The warp must be applied to the oar sweep as well as to the joint
    angles.  Retiming the body alone moves the shoulders out from under
    the handle, and since the hands are pinned to it by the rig, the
    reach constraint fails at about 0.20 of lag.  See SOURCES sec. 40.

    Monotone for ``|lag| < 1/pi``.
    """
    phases = np.asarray(phases, dtype=float)
    f = float(drive_fraction)
    warped = phases.copy()
    on_drive = phases < f
    warped[on_drive] = (phases[on_drive]
                        - float(lag) * f * np.sin(np.pi * phases[on_drive] / f))
    return np.clip(warped, 0.0, 1.0 - 1e-9)


def recovery_warp(progress, arrival: float):
    """Monotone retiming of the recovery traverse: slow into the catch.

    ``w(s) = s + (1 - arrival)(s^2 - s^3)`` on normalised recovery progress
    ``s`` in [0, 1].  Endpoints are fixed, ``w'(0) = 1`` so the clock is
    continuous at the finish, and ``w'(1) = arrival`` -- the crew arrive at
    the catch at ``arrival`` times the nominal traverse rate.

    ``arrival = 1`` is the identity (uniform clock).  Values below one move
    the traverse earlier in the recovery and soften the reversal at the
    catch, which is the universal coaching instruction ("slow into the
    front") and what Kleshnev's recovery-phase analysis reports: seat speed
    peaks early in the recovery and decays toward the catch.

    Monotone for ``arrival >= 0``: ``w' = 1 + (1-arrival)(2s - 3s^2)`` has
    its minimum ``arrival`` at ``s = 1``.
    """
    arrival = float(arrival)
    return progress + (1.0 - arrival) * (progress ** 2 - progress ** 3)


def recovery_warp_slope(progress, arrival: float):
    """``dw/ds`` of :func:`recovery_warp`."""
    arrival = float(arrival)
    return 1.0 + (1.0 - arrival) * (2.0 * progress - 3.0 * progress ** 2)


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
        """Active-phase duration ``tau_a`` in seconds.

        ``drive_fraction * period``; see :attr:`drive_fraction` for the
        source and for why it replaced Formaggia's ``tau_a`` fit.
        """
        return self.drive_fraction * self.period

    @property
    def recovery_duration(self) -> float:
        """Recovery duration ``tau_r = T - tau_a`` in seconds."""
        return self.period - self.drive_duration

    @property
    def drive_fraction(self) -> float:
        """Fraction of the stroke spent on the drive, in ``(0, 1)``.

        ``0.63067 - 5.20991 / rate``, fitted to Telfer et al. (2023),
        *The effect of foot-stretcher position and stroke rate on
        ergometer rowing kinematics*, PLOS ONE 18(5): e0285676, n = 11
        collegiate rowers, who report the catch as a fraction of the
        stroke cycle at three rates:

        =======  ==========  ==========  ==================
        rate     measured    this fit    Formaggia tau_a
        =======  ==========  ==========  ==================
        22 spm   0.394       0.394       0.300
        26 spm   0.430       0.430       0.340
        32 spm   0.468       0.468       0.397
        =======  ==========  ==========  ==================

        **Formaggia's fit made the drive about 25% too short**, and that
        matters out of proportion to its size: the crew cover the same
        distance in less time, so their acceleration goes as ``1/t^2``,
        and the hull's speed fluctuation follows crew acceleration.

        **Why the reciprocal form.**  A straight line in duration fits
        the three rates but extrapolates badly -- duration falls faster
        than the period, so above 40 spm the fraction turns over and the
        recovery-to-drive ratio stops being monotone.  This form
        saturates, and lands on 1:1.00 at 40 spm, the race-pace ratio
        coaches quote, which it was not given.

        **Independent support from the blade.**  Force-weighted blade
        efficiency over the drive is 0.80-0.85 in Kleshnev's on-water
        measurements.  Under Formaggia's short drive the unfitted sweep
        gives 0.747 -- below the band -- which is why
        :attr:`OarAngleSweep.flatness` carried a fitted value near 0.30
        to patch it.  Under this fit the same unfitted sweep gives
        **0.828**, inside the band, and the patch is unnecessary.  A
        correction that removes a fitted parameter rather than needing a
        new one is the kind worth trusting.

        **Caveat.**  Telfer is ergometer data and their rates span 22-32.
        A boat runs under its crew during the recovery in a way an
        ergometer does not, so the on-water fraction is plausibly a
        little lower, and below 22 spm this is extrapolation: at 20 spm
        it gives a 1:1.70 ratio where coaches quote 1:2.  Racing rates
        are inside the fitted range.  See SOURCES sec. 50.
        """
        return 0.63067 - 5.20991 / self.rate

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


class FourierTrack:
    """A periodic 3-D position track, one :class:`FourierProfile` per axis.

    Used for body points whose motion is set by an external constraint
    rather than by a joint angle -- above all the hands, which must lie on
    the oar handle.  Fitting the *position* directly, instead of fitting
    joint angles and hoping the resulting hand lands on the handle, makes
    the constraint hold by construction, and the Fourier representation
    still gives exact velocities and accelerations.
    """

    def __init__(self, x, y, z):
        self.components = (x, y, z)

    @classmethod
    def fit_samples(cls, samples, period: float,
                    n_harmonics: int = DEFAULT_HARMONICS) -> "FourierTrack":
        """Fit to ``(n_samples, 3)`` positions uniformly spaced over a period."""
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 3:
            raise ValueError(
                f"samples must have shape (n, 3), got {samples.shape}")
        return cls(*[FourierProfile.fit_samples(samples[:, axis], period,
                                                n_harmonics)
                     for axis in range(3)])

    def __call__(self, t):
        """Return ``(x, y, z)`` as a tuple of :class:`Jet2`."""
        return tuple(component(t) for component in self.components)

    def position(self, t) -> np.ndarray:
        """Just the values, shaped ``(..., 3)``."""
        return np.stack([component(t).value for component in self.components],
                        axis=-1)

    @property
    def period(self) -> float:
        return self.components[0].period

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FourierTrack(n_harmonics={self.components[0].n_harmonics})"


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

    @staticmethod
    def _ramp(progress: np.ndarray, flatness: float) -> np.ndarray:
        """Monotone map ``[0,1] -> [0,1]`` blending cosine and linear.

        ``flatness = 0`` gives a raised cosine, whose *derivative* is a
        half-sine peaking at 1.57 times its mean.  ``flatness = 1`` gives a
        straight ramp, i.e. constant rate.

        This parameter is what sets how peaky the crew's velocity is, and
        so how hard the hull surges.  A raised cosine makes an eight's
        speed swing about 2.8 m/s peak to peak against a measured
        1.2-1.5 m/s; real crews move much closer to constant slide speed
        within each phase, which is a large part of what "good sequencing"
        means.  See :data:`DEFAULT_FLATNESS`.
        """
        flatness = float(np.clip(flatness, 0.0, 1.0))
        cosine = 0.5 * (1.0 - np.cos(np.pi * progress))
        return (1.0 - flatness) * cosine + flatness * progress

    @classmethod
    def from_catch_finish(cls, catch_value: float, finish_value: float,
                          timing: StrokeTiming,
                          n_harmonics: int = DEFAULT_HARMONICS,
                          n_samples: int = 512,
                          flatness: float = None) -> "FourierProfile":
        """Smooth profile sweeping catch -> finish -> catch.

        The idealised target ramps from ``catch_value`` to ``finish_value``
        over the drive and back over the recovery, with the rate profile
        set by ``flatness`` (see :meth:`_ramp`).  Truncating its spectrum
        to ``n_harmonics`` rounds off the corner at the catch, which is
        both numerically necessary and physically right -- a rower's
        segment accelerations are large at the catch but finite.
        """
        flatness = DEFAULT_FLATNESS if flatness is None else flatness
        phase = np.arange(n_samples) / n_samples
        drive = timing.drive_fraction

        target = np.empty(n_samples)
        on_drive = phase < drive
        target[on_drive] = catch_value + (finish_value - catch_value) * \
            cls._ramp(phase[on_drive] / drive, flatness)
        rec = (phase[~on_drive] - drive) / (1.0 - drive)
        target[~on_drive] = finish_value + (catch_value - finish_value) * \
            cls._ramp(rec, flatness)

        return cls.fit_samples(target, timing.period, n_harmonics)

    @classmethod
    def from_keyframes(cls, phases, values, timing: StrokeTiming,
                       n_harmonics: int = 3,
                       n_samples: int = 512,
                       flatness: float = None,
                       recovery_arrival: float = 1.0,
                       phase_warp=None,
                       shape_preserving: bool = True) -> "FourierProfile":
        """Smooth periodic profile through measured keyframes.

        This is the entry point for published stroke kinematics: give the
        normalised phase of each measured instant and the value there, and
        get back a profile that passes through them smoothly.

        A periodic cubic spline interpolates the keyframes; its spectrum is
        then truncated to ``n_harmonics``.  The default of 3 harmonics is
        deliberately low -- with only four keyframes per stroke, retaining
        more would fit spline artefacts rather than rower motion, and the
        segment accelerations feed straight into the hull forces.

        **Why ``flatness`` is here.**  A cubic spline through four points
        per cycle cannot help being close to a sinusoid, and a sinusoid is
        the wrong shape: its rate peaks at 1.57 times its mean, whereas
        real rowers traverse between postures at much closer to constant
        speed and reverse sharply at the ends.  Measured on the crew centre
        of mass, the spline gave a peakiness of 1.59 on the drive and 1.86
        on the recovery against a real value near 1.14.

        That is not a cosmetic difference.  The hull's speed fluctuation is
        set almost entirely by the crew's centre-of-mass velocity -- 94% of
        it, measured -- so a 1.6x error in peakiness is a 1.6x error in
        boat speed variation.  It is the reason this model reported 60% of
        intracycle velocity variation against a measured 37.5%.

        ``flatness`` blends the spline towards straight-line traverse
        between keyframes, which is constant rate.  **It defaults to zero**,
        because it does not solve the problem and it costs something real.

        Turning it up to 0.6 with 8 harmonics moves an elite single from
        60% of intracycle velocity variation to 54%, against a measured
        37.5% -- about a third of the way.  It cannot do better, and the
        reason is worth stating: a straight-line traverse has corners at
        the keyframes, and truncating that to a few harmonics rings around
        them, so the reconstruction stops passing through the measured
        angles.  Raising the harmonic count to chase the corners fits
        spline artefacts instead.

        The arithmetic says the target *is* reachable.  Given the measured
        keyframe travel and phase timing, the slowest possible traverse --
        exactly constant rate -- gives a centre-of-mass velocity swing of
        1.66 m/s, and 1.89 m/s is what the measured fluctuation implies.
        Real rowers sit 13% above the constant-rate floor; this
        reconstruction sits 48% above it.

        So this is a **data resolution limit, not a modelling error**.
        Four instants per stroke do not determine the shape of the
        traverse, and the shape is what sets the boat's speed variation.
        Fixing it properly needs a densely sampled seat-position trace,
        not a cleverer interpolation of four points.

        Parameters
        ----------
        phases:
            Normalised stroke phases in ``[0, 1)``, strictly increasing.
        values:
            Value at each phase, in whatever units the caller uses.
        """
        from scipy.interpolate import CubicSpline, PchipInterpolator

        phases = np.asarray(phases, dtype=float)
        values = np.asarray(values, dtype=float)
        if phases.shape != values.shape:
            raise ValueError("phases and values must have the same length")
        if np.any(np.diff(phases) <= 0):
            raise ValueError("phases must be strictly increasing")
        if not (0.0 <= phases[0] and phases[-1] < 1.0):
            raise ValueError("phases must lie in [0, 1)")

        # close the loop: repeat the first keyframe at phase 1
        closed_phases = np.append(phases, 1.0)
        closed_values = np.append(values, values[0])
        if shape_preserving:
            # A periodic cubic spline through four unevenly spaced points
            # **overshoots**, and the overshoot is not small: measured
            # against the Caplan & Gardner keyframes it inflated the trunk
            # link swing from 54.7 to 62.4 degrees and the shank from 76.9
            # to 85.2 -- 11 to 14% on every joint excursion.  Crew
            # centre-of-mass travel is a mass-weighted sum of those
            # excursions, so it inherited the same inflation, and the hull
            # speed fluctuation is proportional to it.  See SOURCES sec. 30.
            #
            # This is distinct from the traverse-shape question discussed
            # above, which is a genuine data-resolution limit.  Reporting a
            # joint swing larger than the data being interpolated is not a
            # resolution limit; it is an artefact of the interpolant.
            #
            # PCHIP is shape-preserving -- it will not exceed the local
            # data range -- and periodicity is obtained by interpolating a
            # three-period tiling and keeping the middle one.
            tiled_phases = np.concatenate([closed_phases[:-1] - 1.0,
                                           closed_phases,
                                           closed_phases[1:] + 1.0])
            tiled_values = np.concatenate([closed_values[:-1],
                                           closed_values,
                                           closed_values[1:]])
            spline = PchipInterpolator(tiled_phases, tiled_values)
        else:
            spline = CubicSpline(closed_phases, closed_values,
                                 bc_type="periodic")

        grid = np.arange(n_samples) / n_samples
        sample_at = grid
        if phase_warp is not None:
            # A full-cycle warp table ``(phases, warped)`` fixing the catch
            # and the finish: the posture shown at phase ``phi`` is the one
            # the unwarped motion had at ``warp(phi)``.  Built by
            # :func:`uniform_traverse_warp`; see SOURCES sec. 25.
            knots, images = phase_warp
            sample_at = np.interp(grid, np.asarray(knots, dtype=float),
                                  np.asarray(images, dtype=float))
        if recovery_arrival != 1.0:
            # Retime the recovery: sample the spline ahead of the uniform
            # clock early in the recovery and at ``recovery_arrival`` of
            # the nominal rate at the catch.  Keyframe *postures* are
            # preserved; only when they occur within the recovery shifts.
            # See :func:`recovery_warp` for the physics and the source.
            drive = timing.drive_fraction
            sample_at = np.array(grid)
            recovering = grid >= drive
            progress = (grid[recovering] - drive) / (1.0 - drive)
            sample_at[recovering] = drive + (1.0 - drive) * recovery_warp(
                progress, recovery_arrival)
        smooth = spline(sample_at)

        # Default OFF.  Turning it up trades keyframe fidelity for profile
        # realism and you cannot have both: the straight-line traverse has
        # corners at the keyframes, and truncating that to a few harmonics
        # rings around them, so the reconstruction stops passing through
        # the measured angles.  See the note below.
        flatness = 0.0 if flatness is None else float(flatness)
        flatness = float(np.clip(flatness, 0.0, 1.0))
        if flatness > 0.0:
            # Straight-line traverse between the same keyframes: constant
            # rate within each interval, sharp reversal at each one.  The
            # harmonic truncation below rounds the corners, which is what
            # a real rower's finite acceleration does too.
            straight = np.interp(grid, closed_phases, closed_values)
            smooth = (1.0 - flatness) * smooth + flatness * straight

        return cls.fit_samples(smooth, timing.period, n_harmonics)

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
