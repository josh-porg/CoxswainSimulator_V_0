"""Reading measured telemetry, and fitting the model to it.

Section 23 left the model's largest validation gap needing **data, not
analysis**: the crew's centre-of-mass velocity profile is reconstructed at
48% above the constant-rate floor where real rowers sit at 13%, and four
keyframes per stroke cannot determine which.

There is no real trace yet, so the pipeline is validated by **round trip**
-- generate a trace from the model with known parameters, fit them back,
and check they are recovered.  That is what proves the fitter before the
data exists, and it is what these tests do.
"""

import numpy as np
import pytest

from coxswain.data.telemetry import (StrokeTrace, fit_synchronisation,
                                     read_csv)


def _synthetic(period=1.875, mean=4.5, amplitude=0.8, seconds=24.0,
               rate=50.0, peakiness="sine", phase=0.0, noise=0.0, seed=0):
    """A trace with known properties, for round-trip testing."""
    time = np.arange(0.0, seconds, 1.0 / rate)
    angle = 2.0 * np.pi * (time / period) + phase
    if peakiness == "sine":
        shape = np.sin(angle)
    else:                                   # flat-topped: closer to real
        shape = np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** 0.35
    signal = mean + amplitude * shape
    if noise:
        signal = signal + np.random.default_rng(seed).normal(
            0.0, noise, signal.shape)
    return StrokeTrace(time=time, velocity=signal, source="synthetic")


# --------------------------------------------------------------------------
# the container
# --------------------------------------------------------------------------
def test_a_trace_needs_something_to_work_with():
    with pytest.raises(ValueError, match="velocity or acceleration"):
        StrokeTrace(time=np.arange(10.0))


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="must match time"):
        StrokeTrace(time=np.arange(10.0), velocity=np.arange(5.0))


def test_sample_rate_and_duration():
    trace = _synthetic(seconds=10.0, rate=40.0)
    assert trace.sample_rate == pytest.approx(40.0, rel=1e-6)
    assert trace.duration == pytest.approx(10.0, abs=0.05)


# --------------------------------------------------------------------------
# stroke segmentation
# --------------------------------------------------------------------------
def test_the_stroke_period_is_recovered():
    """Round trip on the most basic quantity."""
    for period in (1.6, 1.875, 2.2):
        trace = _synthetic(period=period, seconds=40.0)
        assert trace.stroke_period() == pytest.approx(period, rel=0.03)


def test_the_period_survives_noise():
    trace = _synthetic(period=1.875, seconds=40.0, noise=0.15, seed=3)
    assert trace.stroke_period() == pytest.approx(1.875, rel=0.05)


def test_autocorrelation_does_not_double_the_rate():
    """Peak counting reports twice the rate on a rough trace with two
    maxima per cycle; the autocorrelation does not."""
    time = np.arange(0.0, 40.0, 0.02)
    angle = 2.0 * np.pi * time / 1.875
    signal = 4.5 + 0.6 * np.sin(angle) + 0.5 * np.sin(2.0 * angle)
    trace = StrokeTrace(time=time, velocity=signal)
    assert trace.stroke_period() == pytest.approx(1.875, rel=0.05)


def test_a_short_trace_is_refused():
    trace = _synthetic(seconds=0.5, rate=50.0)
    with pytest.raises(ValueError, match="too short"):
        trace.stroke_period()


def test_cycles_cover_whole_strokes():
    trace = _synthetic(period=1.875, seconds=20.0)
    cycles = list(trace.cycles(1.875))
    assert len(cycles) >= 9
    for times, _ in cycles:
        assert times[-1] - times[0] < 1.875


# --------------------------------------------------------------------------
# the measured quantities, round tripped
# --------------------------------------------------------------------------
def test_intracycle_variation_recovers_a_known_amplitude():
    """IVV is peak-to-peak within a cycle, so a sinusoid of amplitude A
    must give 2A."""
    trace = _synthetic(amplitude=0.8, seconds=30.0)
    assert trace.intracycle_variation(1.875) == pytest.approx(1.6, rel=0.05)


def test_the_paper_definition_is_reproduced():
    """PMC12349136: males, IVV 5.78 km/h on a mean of 15.40 -> 37.5%."""
    trace = _synthetic(mean=15.40 / 3.6, amplitude=0.5 * 5.78 / 3.6,
                       seconds=40.0)
    ratio = trace.intracycle_variation(1.875) / 4.278
    assert ratio == pytest.approx(0.375, abs=0.02)


def test_coefficient_of_variation_matches_the_definition():
    """SD over mean.  For a sinusoid the SD is A/sqrt(2)."""
    trace = _synthetic(mean=4.5, amplitude=0.8, seconds=40.0)
    expected = 0.8 / np.sqrt(2.0) / 4.5
    assert trace.coefficient_of_variation(4.5) == pytest.approx(
        expected, rel=0.05)


def test_peakiness_distinguishes_a_sinusoid_from_a_flat_profile():
    """The quantity section 23 actually needs.

    A sinusoid and a flat-topped traverse can have the same amplitude and
    the same period and still load the hull completely differently.
    """
    sine = _synthetic(peakiness="sine", seconds=40.0).peakiness(1.875)
    flat = _synthetic(peakiness="flat", seconds=40.0).peakiness(1.875)
    assert flat < sine


def test_phase_average_suppresses_noise():
    """And by how much, which is a trade against timing resolution.

    Finer bins resolve smaller lags but average fewer samples each.  At
    200 bins, 50 Hz and 40 s there are 10 samples a bin, so noise of SD
    ``s`` survives at about ``s/sqrt(10)`` and the worst bin of 200 sits
    near three of those.  Resolution was chosen first -- 9 ms against the
    65 ms effect being measured -- and the averaging follows from it.
    """
    clean = _synthetic(seconds=40.0).phase_average(1.875)
    noisy = _synthetic(seconds=40.0, noise=0.2, seed=1).phase_average(1.875)
    worst = np.abs(noisy - clean).max()
    assert worst < 3.5 * 0.2 / np.sqrt(10.0)


def test_a_longer_trace_averages_better():
    """Which is the answer if a trace is noisy: row for longer."""
    short = _synthetic(seconds=20.0, noise=0.2, seed=2)
    long = _synthetic(seconds=80.0, noise=0.2, seed=2)
    clean = _synthetic(seconds=80.0).phase_average(1.875)
    short_error = np.abs(short.phase_average(1.875) - clean).max()
    long_error = np.abs(long.phase_average(1.875) - clean).max()
    assert long_error < short_error


# --------------------------------------------------------------------------
# accelerometer input -- what a phone gives
# --------------------------------------------------------------------------
def test_acceleration_integrates_to_the_right_shape():
    """A phone taped to a rigger is the cheapest way to close this gap, and
    it measures acceleration, not speed."""
    period, amplitude = 1.875, 0.8
    time = np.arange(0.0, 40.0, 0.02)
    omega = 2.0 * np.pi / period
    acceleration = amplitude * omega * np.cos(omega * time)
    trace = StrokeTrace(time=time, acceleration=acceleration)
    assert trace.stroke_period() == pytest.approx(period, rel=0.05)
    assert trace.intracycle_variation(period) == pytest.approx(
        2.0 * amplitude, rel=0.25)


def test_integration_drift_is_removed():
    """Raw integration walks away; the variation must not."""
    time = np.arange(0.0, 40.0, 0.02)
    omega = 2.0 * np.pi / 1.875
    acceleration = 0.8 * omega * np.cos(omega * time) + 0.05
    trace = StrokeTrace(time=time, acceleration=acceleration)
    surge = trace.surge()
    assert abs(surge[-100:].mean() - surge[:100].mean()) < 0.5


# --------------------------------------------------------------------------
# reading files
# --------------------------------------------------------------------------
def test_csv_round_trip(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("t,v\n0.0,4.1\n0.02,4.3\n0.04,4.6\n", encoding="utf-8")
    trace = read_csv(path, time_column="t", velocity_column="v")
    np.testing.assert_allclose(trace.time, [0.0, 0.02, 0.04])
    np.testing.assert_allclose(trace.velocity, [4.1, 4.3, 4.6])


def test_a_missing_column_says_what_is_there(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("t,v\n0.0,4.1\n", encoding="utf-8")
    with pytest.raises(KeyError, match="speed"):
        read_csv(path, time_column="t", velocity_column="speed")


# --------------------------------------------------------------------------
# synchronisation, round tripped
# --------------------------------------------------------------------------
def test_per_seat_lags_are_recovered():
    """Section 22's coefficients, from one sensor per rower.

    Known offsets are imposed and must come back out.
    """
    period = 1.875
    lags = np.array([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.015, 0.0])
    traces = [_synthetic(period=period, seconds=40.0,
                         phase=-2.0 * np.pi * lag / period) for lag in lags]
    fit = fit_synchronisation(traces, period)
    np.testing.assert_allclose(fit.offsets, lags, atol=0.02)


def test_the_spread_is_recovered():
    period = 1.875
    lags = np.array([0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.0])
    traces = [_synthetic(period=period, seconds=40.0,
                         phase=-2.0 * np.pi * lag / period) for lag in lags]
    fit = fit_synchronisation(traces, period)
    assert fit.spread == pytest.approx(0.06, abs=0.02)


def test_a_together_crew_reads_as_coherent():
    period = 1.875
    traces = [_synthetic(period=period, seconds=40.0) for _ in range(8)]
    fit = fit_synchronisation(traces, period)
    assert fit.coherence > 0.99
    assert fit.spread < 0.02


def test_a_chain_pattern_is_identified():
    """Section 22's discriminator: monotone lag from stroke toward bow is
    a directed sensory chain; mean-field coupling is not."""
    period = 1.875
    lags = np.linspace(0.07, 0.0, 8)
    traces = [_synthetic(period=period, seconds=40.0,
                         phase=-2.0 * np.pi * lag / period) for lag in lags]
    assert fit_synchronisation(traces, period).looks_like_a_chain


def test_scattered_timing_does_not_read_as_a_chain():
    period = 1.875
    rng = np.random.default_rng(4)
    lags = np.abs(rng.normal(0.0, 0.03, 8))
    lags[-1] = 0.0
    traces = [_synthetic(period=period, seconds=40.0,
                         phase=-2.0 * np.pi * lag / period) for lag in lags]
    assert not fit_synchronisation(traces, period).looks_like_a_chain


def test_one_seat_is_not_enough():
    with pytest.raises(ValueError, match="at least two"):
        fit_synchronisation([_synthetic()])
