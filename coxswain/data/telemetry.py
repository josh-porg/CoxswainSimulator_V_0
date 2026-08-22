"""Reading real boat telemetry, and fitting the model to it.

Section 23 left the model's largest validation gap in a specific and
unusual state: the cause is known, the fix is known, and neither is a code
change. The crew's centre-of-mass velocity profile is reconstructed at 48%
above the constant-rate floor when real rowers sit at 13%, and four
measured keyframes per stroke cannot determine which. **What is missing is
data, not analysis.**

This module is the other half of that: it ingests a measured trace, pulls
out the quantities the model is wrong about, and fits the model's free
shape parameters to them.

What it needs
-------------
Very little. A single axis of hull acceleration at 25 Hz or better, for a
minute of steady rowing, is enough to determine:

* the stroke period and the drive/recovery split, from the acceleration
  zero crossings;
* the intracycle velocity variation, by integrating within each stroke --
  which is the number section 23 is about;
* the **shape** of the surge profile, and hence the peakiness of the
  crew's centre-of-mass motion, which is the actual unknown.

That is a phone taped to a rigger. A SpeedCoach or Quiske export works
equally well and carries the stroke rate directly.

Per-seat data -- one sensor per rower, or a Quiske seat unit -- additionally
determines the synchronisation coefficients of section 22: the timing
spread, and whether the lag pattern down the boat looks like a directed
chain or a mean field.

Fitting, not just reading
-------------------------
:func:`fit_profile_flatness` searches the one shape parameter the model
leaves free until the simulated intracycle variation matches the measured
one.  :func:`fit_synchronisation` recovers the per-seat phase offsets and
the coupling topology.

Both are validated by **round trip**: generate a trace from the model with
known parameters, fit it back, and check the parameters are recovered.
That proves the pipeline before any real data exists, and it is what
``tests/unit/test_telemetry.py`` asserts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["StrokeTrace", "read_csv", "fit_profile_flatness",
           "fit_synchronisation"]


@dataclass
class StrokeTrace:
    """A measured time series from a boat.

    ``time`` in seconds and one of ``velocity`` (m/s) or ``acceleration``
    (m/s^2), longitudinal, in the hull frame.  Acceleration is what a phone
    gives; velocity is what a GPS-based SpeedCoach gives.  Either is enough.
    """

    time: np.ndarray
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None
    source: str = ""

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        if self.velocity is None and self.acceleration is None:
            raise ValueError("need velocity or acceleration")
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=float)
            if self.velocity.shape != self.time.shape:
                raise ValueError("velocity must match time")
        if self.acceleration is not None:
            self.acceleration = np.asarray(self.acceleration, dtype=float)
            if self.acceleration.shape != self.time.shape:
                raise ValueError("acceleration must match time")

    @property
    def sample_rate(self) -> float:
        return float(1.0 / np.median(np.diff(self.time)))

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])

    # -- deriving one from the other --------------------------------------
    def surge(self) -> np.ndarray:
        """Longitudinal velocity, integrating acceleration if needed.

        Integration drifts, so the result is high-pass filtered by removing
        a slow trend.  That loses the mean speed -- which the accelerometer
        never knew -- but keeps the *variation*, which is the quantity of
        interest here.  A GPS mean can be added back if one is available.
        """
        if self.velocity is not None:
            return self.velocity
        raw = np.concatenate([[0.0], np.cumsum(
            0.5 * (self.acceleration[1:] + self.acceleration[:-1])
            * np.diff(self.time))])
        # remove a slow trend: a moving average over several strokes
        window = max(int(self.sample_rate * 4.0), 3)
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        trend = np.convolve(raw, kernel, mode="same")
        edge = window // 2
        trend[:edge] = trend[edge]
        trend[-edge:] = trend[-edge - 1]
        return raw - trend

    # -- stroke segmentation ----------------------------------------------
    def stroke_period(self, low_spm: float = 12.0,
                      high_spm: float = 50.0) -> float:
        """Dominant stroke period, from the band-limited power spectrum.

        Autocorrelation was the obvious choice and it does not survive real
        data.  A boat-mounted phone picks up hull vibration and slap an
        order of magnitude above the stroke frequency, and the
        autocorrelation peak then pins to whatever the shortest allowed lag
        is -- every trial in the CC0 elite set reported the search floor,
        75 spm, for rowing that was actually 16 to 34.

        The spectrum does not have that failure mode: the stroke
        fundamental is the largest peak *inside the plausible band*,
        whatever is happening above it.  Rating is bounded by the sport --
        nobody rows below 12 or above 50 -- so the band is a real
        constraint rather than a tuned one.
        """
        signal = self.surge()
        signal = signal - signal.mean()
        if len(signal) < 16:
            raise ValueError("trace too short to find a stroke period")
        window = np.hanning(len(signal))
        spectrum = np.abs(np.fft.rfft(signal * window))
        frequency = np.fft.rfftfreq(len(signal), 1.0 / self.sample_rate)

        band = (frequency >= low_spm / 60.0) & (frequency <= high_spm / 60.0)
        if not band.any() or self.duration < 2.0 * 60.0 / low_spm:
            raise ValueError("trace too short to find a stroke period")

        # Harmonic product spectrum.  Band-limiting alone is not enough on
        # real data: a 60 s window of rowing carries low-frequency energy
        # from integration drift, from the boat pitching, and from the crew
        # winding the rate up inside the window, and that can outrank the
        # stroke fundamental.  In the CC0 elite set the 34 spm trial had a
        # clean peak at exactly 34.0 -- ranked sixth, behind 12 spm of
        # drift.
        #
        # A real stroke frequency has harmonics; drift does not.  Boat
        # acceleration in particular has a pronounced second harmonic --
        # the double peak Kleshnev describes during the drive -- so
        # multiplying the spectrum by its own decimations sharpens the true
        # fundamental and suppresses everything that only has energy at one
        # frequency.
        # Multiplying outright would be the textbook harmonic product
        # spectrum, and it zeroes a pure tone -- which has no harmonics at
        # all.  A synthetic sinusoid is exactly that, and so is a very
        # smooth crew.  So harmonic support is used as a *weight* rather
        # than a factor: a peak with harmonics is promoted, a peak without
        # them is left alone rather than destroyed.
        scale = float(spectrum.max()) or 1.0
        support = np.zeros_like(spectrum)
        for harmonic in (2, 3):
            reachable = len(spectrum) // harmonic
            support[:reachable] += spectrum[::harmonic][:reachable] / scale
        score = spectrum * (1.0 + support)
        index = int(np.argmax(np.where(band, score, 0.0)))
        spectrum = score

        # Parabolic refinement: the FFT bin is coarse for a short trace,
        # and the period feeds every downstream quantity.
        if 0 < index < len(spectrum) - 1:
            left, centre, right = spectrum[index - 1:index + 2]
            denominator = left - 2.0 * centre + right
            shift = 0.0 if denominator == 0 else                 0.5 * (left - right) / denominator
        else:
            shift = 0.0
        step = frequency[1] - frequency[0]
        peak = frequency[index] + float(np.clip(shift, -0.5, 0.5)) * step
        if peak <= 0.0:
            raise ValueError("no stroke frequency found in band")
        return float(1.0 / peak)

    def cycles(self, period: Optional[float] = None):
        """Split into whole strokes; yields ``(time, surge)`` per cycle."""
        period = self.stroke_period() if period is None else period
        signal = self.surge()
        start = self.time[0]
        while start + period <= self.time[-1]:
            window = (self.time >= start) & (self.time < start + period)
            if window.sum() > 4:
                yield self.time[window], signal[window]
            start += period

    # -- the numbers section 23 is about ----------------------------------
    def intracycle_variation(self, period: Optional[float] = None) -> float:
        """Mean of ``max - min`` within each stroke, in m/s.

        The IVV of PMC12349136, defined exactly as they define it.
        """
        swings = [float(s.max() - s.min()) for _, s in self.cycles(period)]
        if not swings:
            raise ValueError("no complete strokes in the trace")
        return float(np.mean(swings))

    def coefficient_of_variation(self, mean_speed: float) -> float:
        """SD over mean, the CVV of PMC12349136.

        ``mean_speed`` must be supplied when the trace came from an
        accelerometer, which does not know it.
        """
        return float(self.surge().std() / mean_speed)

    def peakiness(self, period: Optional[float] = None) -> float:
        """Peak rate over mean rate of the surge, phase-averaged.

        **The quantity section 23 needs.**  Peak deviation from the mean
        over mean absolute deviation: ``1.571`` for a sinusoid (peak ``A``
        against ``2A/pi``), ``1.0`` for a square wave, and lower for
        anything flat-topped.  Real rowing is near 1.13; this model
        reconstructs 1.59.

        Deliberately computed on the signal itself, not its derivative.
        Differentiating a measured trace amplifies noise and answers a
        different question -- how sharply the surge *changes*, not how
        peaked it is.
        """
        profile = self.phase_average(period)
        deviation = np.abs(profile - profile.mean())
        mean = deviation.mean()
        return float(deviation.max() / mean) if mean > 0 else 1.0

    def phase_average(self, period: Optional[float] = None,
                      bins: int = 200) -> np.ndarray:
        """Surge averaged over many strokes, on a common phase grid.

        Averaging kills the stroke-to-stroke noise that would otherwise
        dominate a shape estimate.

        ``bins`` sets the timing resolution of anything built on this.  At
        200 bins and a 1.9 s stroke that is 9 ms, against the 65 ms
        port/starboard split that section 21 shows exhausts an eight's
        recovery balance authority -- so the resolution has to be well
        inside the quantity being measured.  At the original 50 bins it was
        37 ms, which is the same order as the effect and would have made
        the synchronisation fit meaningless.
        """
        period = self.stroke_period() if period is None else period
        signal = self.surge()
        phase = np.mod(self.time - self.time[0], period) / period
        index = np.minimum((phase * bins).astype(int), bins - 1)
        out = np.zeros(bins)
        count = np.zeros(bins)
        np.add.at(out, index, signal)
        np.add.at(count, index, 1.0)
        return out / np.maximum(count, 1.0)


def read_csv(path, time_column="time", velocity_column=None,
             acceleration_column=None, source="") -> StrokeTrace:
    """Read a trace from a CSV export.

    Deliberately generic: phone logging apps, SpeedCoach and Quiske all
    export CSV with different column names, and naming them at the call
    site is less brittle than guessing.
    """
    columns = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for name in reader.fieldnames or []:
            columns[name] = []
        for row in reader:
            for name in columns:
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError):
                    columns[name].append(np.nan)

    def pick(name):
        if name is None:
            return None
        if name not in columns:
            raise KeyError(
                f"column {name!r} not in {sorted(columns)}")
        return np.array(columns[name], dtype=float)

    return StrokeTrace(time=pick(time_column),
                       velocity=pick(velocity_column),
                       acceleration=pick(acceleration_column),
                       source=source or str(path))


# ==========================================================================
# Fitting the model to a measured trace
# ==========================================================================
@dataclass
class ProfileFit:
    """Result of fitting the crew motion shape to a measured trace."""

    flatness: float
    peakiness: float
    measured_peakiness: float
    measured_ivv: float
    fitted_ivv: float
    converged: bool

    @property
    def residual(self) -> float:
        return abs(self.fitted_ivv - self.measured_ivv)


def fit_profile_flatness(trace, boat_factory, mean_speed=None,
                         bounds=(0.0, 1.0), tolerance=1e-3,
                         max_iterations: int = 24, **run_kwargs):
    """Find the crew-motion flatness that reproduces a measured trace.

    This is the fit section 23 asks for.  The model's crew centre-of-mass
    velocity profile has one free shape parameter; everything else about
    the kinematics -- travel, timing, mass -- is already pinned by
    measurement and checks out.  So the trace determines it directly.

    ``boat_factory`` takes a flatness and returns a boat, so the caller
    controls which dataset, rate and class are being fitted.

    Bisection on a monotone scalar: flatter crew motion means smaller
    intracycle variation, so there is exactly one crossing.
    """
    from coxswain.sim.simulator import RowingSimulator

    target = trace.intracycle_variation()
    period_guess = trace.stroke_period()

    def simulated(flatness):
        boat = boat_factory(flatness)
        result = RowingSimulator(boat).run(
            duration=run_kwargs.pop("duration", 16.0),
            dt=run_kwargs.pop("dt", 0.006),
            surge_speed=run_kwargs.pop("surge_speed", 4.4))
        speed = np.asarray(result.velocity)[0]
        times = np.asarray(result.time)
        keep = times > 0.5 * times[-1]
        return StrokeTrace(time=times[keep] - times[keep][0],
                           velocity=speed[keep]).intracycle_variation(
            boat.timing.period)

    low, high = bounds
    value_low = simulated(low) - target
    value_high = simulated(high) - target
    if value_low * value_high > 0.0:
        # the target lies outside what the shape parameter can reach --
        # which is the situation section 23 documents for this model
        best = low if abs(value_low) < abs(value_high) else high
        return ProfileFit(flatness=best,
                          peakiness=float("nan"),
                          measured_peakiness=trace.peakiness(period_guess),
                          measured_ivv=target,
                          fitted_ivv=target + (value_low if best == low
                                               else value_high),
                          converged=False)

    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        value = simulated(middle) - target
        if abs(value) < tolerance:
            break
        if value * value_low > 0.0:
            low, value_low = middle, value
        else:
            high = middle
    flatness = 0.5 * (low + high)
    return ProfileFit(flatness=flatness,
                      peakiness=float("nan"),
                      measured_peakiness=trace.peakiness(period_guess),
                      measured_ivv=target,
                      fitted_ivv=simulated(flatness),
                      converged=True)


@dataclass
class SynchronisationFit:
    """Per-seat timing recovered from per-rower telemetry."""

    offsets: np.ndarray          # seconds, relative to the stroke seat
    spread: float                # peak-to-peak, seconds
    coherence: float             # Kuramoto order parameter magnitude
    chain_score: float           # 1 = perfectly monotone lag toward bow

    @property
    def looks_like_a_chain(self) -> bool:
        """Whether the lag pattern is monotone from stroke toward bow.

        Section 22's discriminator: a directed sensory chain accumulates
        lag down the boat, mean-field coupling through the hull does not.
        """
        return self.chain_score > 0.8


def fit_synchronisation(traces: Sequence, period=None) -> SynchronisationFit:
    """Recover per-seat timing from one trace per rower.

    ``traces`` is ordered bow to stroke, matching the seat indexing used
    throughout.  Each is cross-correlated against the stroke seat to find
    its lag; the spread and the shape of the lag pattern then give the two
    numbers section 22 leaves open.
    """
    if len(traces) < 2:
        raise ValueError("need at least two seats to compare")
    reference = traces[-1]
    period = reference.stroke_period() if period is None else period

    offsets = []
    for trace in traces:
        a = reference.phase_average(period)
        b = trace.phase_average(period)
        a = a - a.mean()
        b = b - b.mean()
        correlation = np.correlate(np.tile(b, 2), a, mode="valid")[:len(a)]
        peak = int(np.argmax(correlation))
        # Parabolic interpolation through the peak and its neighbours.
        # Without it the answer is quantised to one bin, and a crew's
        # timing spread is a fraction of a bin -- the fit would report
        # steps rather than lags.
        left = correlation[(peak - 1) % len(correlation)]
        right = correlation[(peak + 1) % len(correlation)]
        centre = correlation[peak]
        denominator = left - 2.0 * centre + right
        offset = 0.0 if denominator == 0 else             0.5 * (left - right) / denominator
        shift = peak + float(np.clip(offset, -0.5, 0.5))
        if shift > len(a) / 2.0:
            shift -= len(a)
        offsets.append(shift / len(a) * period)
    offsets = np.array(offsets) - offsets[-1]

    phases = 2.0 * np.pi * offsets / period
    coherence = float(abs(np.mean(np.exp(1j * phases))))

    steps = np.diff(offsets)
    if len(steps) == 0 or np.all(steps == 0):
        chain = 0.0
    else:
        chain = float(max((steps > 0).mean(), (steps < 0).mean()))
    return SynchronisationFit(offsets=offsets,
                              spread=float(offsets.max() - offsets.min()),
                              coherence=coherence,
                              chain_score=chain)
