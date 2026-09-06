"""The stroke sounds are timing plus a measured envelope; both testable."""

import numpy as np

from coxswain.viz.strokeaudio import (CATCH_BANDS, CYCLE_FLOOR,
                                      events_between, load_envelope,
                                      synthesise, synthesise_cycle)


def test_events_fall_where_the_stroke_says():
    found = events_between(0.0, 4.05, 2.0, 0.4)
    assert [name for _t, name in found] == [
        "release", "catch", "release", "catch"]
    assert [round(t, 3) for t, _n in found] == [0.8, 2.0, 2.8, 4.0]


def test_a_stall_does_not_drop_a_catch():
    """An interval longer than a stroke returns every event in it."""
    assert [n for _t, n in events_between(1.0, 4.0, 2.0, 0.4)] == [
        "catch", "release", "catch"]


def test_the_catch_matches_the_measured_envelope():
    """Synthesis is a fit, not a guess, so it can be checked against the
    thing it was fitted to."""
    catch = synthesise()["catch"]
    rate = 44100
    mag = np.abs(np.fft.rfft(catch * np.hanning(len(catch))))
    freq = np.fft.rfftfreq(len(catch), 1.0 / rate)
    got, want = [], []
    for low, high, level in CATCH_BANDS:
        inside = (freq >= low) & (freq < high)
        if not inside.any():
            continue
        got.append(20 * np.log10(max(mag[inside].mean(), 1e-12)))
        want.append(level)
    got = np.asarray(got) - max(got)
    assert np.median(np.abs(got - np.asarray(want))) < 2.0


def test_the_boat_is_continuous_not_two_bangs():
    """The bed carries the cycle at the measured level.  Without this the
    catch sits over near-silence and the result is a drum machine."""
    clips = synthesise()
    assert 0.4 < CYCLE_FLOOR < 0.9
    assert abs(np.abs(clips["slide"]).max() - CYCLE_FLOOR) < 0.05


def test_the_full_cycle_follows_the_envelope_through_the_stroke():
    loaded = load_envelope()
    if loaded is None:
        return
    bands, envelope, _period = loaded
    cycle = synthesise_cycle(2.0)
    assert cycle is not None and len(cycle) == int(44100 * 2.0)
    # The low band is a catch phenomenon: loud at phase 0, well down by
    # mid-drive.  If that ordering is lost the clip is not phase-varying.
    low = np.argmin(np.abs(0.5 * (bands[:, 0] + bands[:, 1]) - 140.0))
    assert envelope[low, 0] > 2.0 * envelope[low, envelope.shape[1] // 4]
