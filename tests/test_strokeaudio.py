"""The stroke sounds are measured events at measured phases."""

import numpy as np

from coxswain.viz.strokeaudio import (events_between, load_envelope,
                                      shell_of, synthesise, synthesise_cycle)


class _Boat:
    def __init__(self, seats):
        self.n_seats = seats


def test_events_fall_at_the_phases_they_are_given():
    found = events_between(0.0, 4.05, 2.0, [0.0, 0.4])
    assert [round(t, 3) for t, _i in found] == [0.8, 2.0, 2.8, 4.0]
    assert [i for _t, i in found] == [1, 0, 1, 0]


def test_a_stall_does_not_drop_an_event():
    """An interval longer than a stroke returns every event in it."""
    found = events_between(1.0, 4.0, 2.0, [0.0, 0.4])
    assert [i for _t, i in found] == [0, 1, 0]


def test_a_four_and_an_eight_are_different_boats():
    assert shell_of(_Boat(4)) == "four"
    assert shell_of(_Boat(8)) == "eight"
    four, eight = load_envelope("four"), load_envelope("eight")
    if four is None or eight is None:
        return
    # The eight is measurably the brighter boat: more of its energy sits
    # above 1 kHz at the catch.  If these ever come out the same, the two
    # envelopes have been built from the same recording.
    def high_share(env):
        bands, spectrum = env["bands"], env["envelope"][:, 0]
        high = 0.5 * (bands[:, 0] + bands[:, 1]) > 1000.0
        return float(spectrum[high].sum() / spectrum.sum())
    assert high_share(eight) > high_share(four)


def test_every_shell_has_a_catch_and_a_finish_event():
    for shell in ("four", "eight"):
        built = synthesise(shell)
        if built is None:
            continue
        assert len(built["events"]) >= 2
        assert float(built["phases"][0]) == 0.0        # the catch
        # A later event, in the second half of the drive or beyond: the
        # blades coming out and being feathered.
        assert any(p > 0.1 for p in built["phases"][1:])
        # and it is a different sound from the catch, not a copy
        catch, later = built["events"][0], built["events"][1]
        assert len(later) < len(catch)


def test_the_catch_is_the_loudest_thing_in_the_cycle():
    built = synthesise("four")
    if built is None:
        return
    peak = max(np.abs(clip).max() for clip in built["events"][1:])
    assert np.abs(built["events"][0]).max() > peak
    assert np.abs(built["bed"]).max() < np.abs(built["events"][0]).max()


def test_the_full_cycle_clip_is_one_stroke_long():
    cycle = synthesise_cycle(2.0, "four")
    if cycle is None:
        return
    assert len(cycle) == int(44100 * 2.0)
