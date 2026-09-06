"""Buoys as one-sided limits, and a line of buoys as a line."""

import numpy as np

from coxswain.river.buoys import one_sided_limits, place


def straight(length=1000.0, step=10.0):
    x = np.arange(0.0, length + step, step)
    return np.column_stack([x, np.zeros_like(x)])


def test_a_port_mark_narrows_only_the_port_side():
    line = straight()
    half = np.full(len(line), 100.0)
    # keep to port, 30 m to port (north) of the lane at x=500
    port, starboard, marks = one_sided_limits(line, half, [(1, 500.0, 30.0)],
                                              reach=40.0, margin=6.0)
    at = np.argmin(np.abs(line[:, 0] - 500.0))
    assert port[at] == 24.0
    assert starboard[at] == 100.0
    assert port[np.argmin(np.abs(line[:, 0] - 300.0))] == 100.0
    assert not marks[0].ignored


def test_a_starboard_mark_narrows_only_the_starboard_side():
    line = straight()
    half = np.full(len(line), 100.0)
    port, starboard, _ = one_sided_limits(line, half, [(0, 500.0, -30.0)],
                                          reach=40.0, margin=6.0)
    at = np.argmin(np.abs(line[:, 0] - 500.0))
    assert starboard[at] == 24.0
    assert port[at] == 100.0


def test_two_marks_of_one_colour_bind_between_them():
    """The gap between two marks 150 m apart used to be wide open."""
    line = straight()
    half = np.full(len(line), 100.0)
    port, _, _ = one_sided_limits(line, half,
                                  [(1, 400.0, 30.0), (1, 550.0, 50.0)],
                                  reach=40.0, margin=6.0, chain=250.0)
    mid = np.argmin(np.abs(line[:, 0] - 500.0))     # outside either reach
    expected = 30.0 + 20.0 * (100.0 / 150.0) - 6.0   # interpolated
    assert abs(port[mid] - expected) < 1e-9
    # and beyond the chain distance they stay separate
    port, _, _ = one_sided_limits(line, half,
                                  [(1, 200.0, 30.0), (1, 800.0, 50.0)],
                                  reach=40.0, margin=6.0, chain=250.0)
    assert port[np.argmin(np.abs(line[:, 0] - 500.0))] == 100.0


def test_marks_that_are_not_lane_limits_are_named_as_such():
    line = straight()
    station = line[:, 0]
    marks = place(line, [(1, 500.0, 30.0),      # a limit
                         (1, 500.0, 300.0),     # marshalling, far off
                         (1, 20.0, 30.0),       # in the start chute
                         (1, 500.0, -30.0)],    # wrong side of the lane
                  station, limit_reach=80.0, end_zone=100.0)
    assert [m.ignored for m in marks] == [
        "", "off the lane", "start chute or finish gate",
        "on the wrong side of the drawn lane"]


def test_a_limit_never_closes_the_lane():
    line = straight()
    half = np.full(len(line), 100.0)
    port, _, _ = one_sided_limits(line, half, [(1, 500.0, 3.0)],
                                  reach=40.0, margin=6.0)
    assert port.min() == 2.0
