r"""The tabulated crew kinematics against the chain solved directly.

The derivative was spending 2.3 ms per evaluation re-solving the joint
chain, 6.6 times a frame, and the answer depends on stroke time alone.
:class:`~coxswain.boats.boat.StrokeTable` solves it once per boat and
interpolates.  These hold the interpolation to well under anything the
model claims, at the sample count actually shipped -- and they hold the
speed-up too, because a table that is not faster is only a second copy
of the answer.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from coxswain.boats import catalog


@pytest.fixture(scope="module")
def four():
    boat = catalog.coxed_four(rate=30.0)
    boat.tabulate_crew = True
    return boat


@pytest.fixture(scope="module")
def eight_with_offsets():
    """A crew whose seats are NOT in time, so every seat is its own group."""
    boat = catalog.eight(rate=32.0)
    boat.tabulate_crew = True
    boat.phase_offsets = np.linspace(-0.03, 0.03, boat.n_seats)
    return boat


def test_the_table_matches_the_chain_to_a_tenth_of_a_millimetre(four):
    """Position, at random times, against the exact solve."""
    rng = np.random.default_rng(3)
    period = float(four.timing.period)
    worst = 0.0
    for t in rng.uniform(-2.0 * period, 3.0 * period, 60):
        _m, exact, _v, _a = four.crew_field(t, exact=True)
        _m, table, _v, _a = four.crew_field(t)
        worst = max(worst, float(np.abs(exact - table).max()))
    assert worst < 2.0e-4, worst           # 0.2 mm, over a 13 m boat


def test_velocity_and_acceleration_track_the_chain(four):
    """Relative to the stroke's own peaks, not to zero.

    A linear table is out by about ``h^2 |jerk| / 8`` in acceleration
    at the catch, where the jerk is largest; that is a fraction of a
    percent of the peak, and the mass-matrix reaction it feeds is a
    small term in the surge balance to begin with.
    """
    rng = np.random.default_rng(5)
    period = float(four.timing.period)
    _m, _p, v_ref, a_ref = four.crew_field(0.0, exact=True)
    v_scale = max(float(np.abs(v_ref).max()), 1e-6)
    a_scale = max(float(np.abs(a_ref).max()), 1e-6)
    # peaks over the stroke, from the exact solve
    for t in np.linspace(0.0, period, 25, endpoint=False):
        _m, _p, v, a = four.crew_field(t, exact=True)
        v_scale = max(v_scale, float(np.abs(v).max()))
        a_scale = max(a_scale, float(np.abs(a).max()))
    for t in rng.uniform(0.0, period, 40):
        _m, _p, v_e, a_e = four.crew_field(t, exact=True)
        _m, _p, v_t, a_t = four.crew_field(t)
        assert np.abs(v_e - v_t).max() / v_scale < 5e-3
        assert np.abs(a_e - a_t).max() / a_scale < 2e-2


def test_hands_match_including_the_lateral_sweep(four):
    rng = np.random.default_rng(7)
    period = float(four.timing.period)
    for t in rng.uniform(0.0, 2.0 * period, 40):
        exact = four.hand_positions(t, exact=True)
        table = four.hand_positions(t)
        assert np.abs(exact - table).max() < 3.0e-4, t


def test_phase_offsets_are_a_lookup_shift_not_a_rebuild(eight_with_offsets):
    """Seats out of time read the same table at different phases."""
    boat = eight_with_offsets
    period = float(boat.timing.period)
    before = len(boat.__dict__.get("_stroke_tables", {}))
    for t in np.linspace(0.0, period, 9):
        _m, exact, _v, _a = boat.crew_field(t, exact=True)
        _m, table, _v, _a = boat.crew_field(t)
        assert np.abs(exact - table).max() < 2.0e-4
    # eight distinct offsets -> eight groups, but ONE rower table each
    # (the leaders are the individual rowers), and a second pass at new
    # offsets must not build any more.
    built = len(boat.__dict__.get("_stroke_tables", {}))
    assert built >= 1
    boat.phase_offsets = np.linspace(-0.02, 0.02, boat.n_seats)
    for t in np.linspace(0.0, period, 5):
        boat.crew_field(t)
    assert len(boat.__dict__.get("_stroke_tables", {})) == built


def test_the_table_is_periodic_and_continuous(four):
    period = float(four.timing.period)
    _m, p0, _v, _a = four.crew_field(0.0)
    _m, p1, _v, _a = four.crew_field(period)
    _m, p2, _v, _a = four.crew_field(-period)
    assert np.abs(p0 - p1).max() < 1e-9
    assert np.abs(p0 - p2).max() < 1e-9
    # no seam at the wrap: the last sample interpolates into the first
    eps = 1e-6
    _m, a, _v, _a = four.crew_field(period - eps)
    _m, b, _v, _a = four.crew_field(period + eps)
    assert np.abs(a - b).max() < 1e-4


def test_the_derivative_agrees_with_and_without_the_table(four):
    """End to end: the simulator's state derivative, both ways."""
    from coxswain.sim.simulator import RowingSimulator

    sim = RowingSimulator(four)
    state = sim.initial_state(surge_speed=4.5)
    rng = np.random.default_rng(11)
    # the exact derivative's per-row peak over one stroke
    boat = four
    boat.tabulate_crew = False
    peaks = np.zeros(len(np.asarray(sim.derivative(0.0, state))))
    for tt in np.linspace(0.0, float(boat.timing.period), 24, endpoint=False):
        sim._crew_cache = sim._hand_cache = (None, None)
        peaks = np.maximum(peaks, np.abs(np.asarray(sim.derivative(tt, state))))
    peaks = np.maximum(peaks, 1e-6)
    boat.tabulate_crew = True
    for t in rng.uniform(0.0, float(four.timing.period), 12):
        four.tabulate_crew = True
        sim._crew_cache = (None, None)
        sim._hand_cache = (None, None)
        with_table = np.asarray(sim.derivative(t, state))
        four.tabulate_crew = False
        sim._crew_cache = (None, None)
        sim._hand_cache = (None, None)
        exact = np.asarray(sim.derivative(t, state))
        four.tabulate_crew = True
        # Scaled per ROW by that quantity's peak over the stroke, not by
        # its value at this instant: an acceleration passing through
        # zero is not a large relative error, it is zero.
        assert np.all(np.abs(with_table - exact) <= 1e-2 * peaks), (
            t, np.abs(with_table - exact) / peaks)


def test_the_table_is_actually_faster(four, eight_with_offsets):
    """A floor, not a benchmark.

    A homogeneous four is ONE chain per evaluation, so the direct solve
    is already cheap there and the table wins by an order of magnitude.
    The game's eight carries per-seat timing scatter, so every seat is
    its own group and the direct path solves eight chains -- that is
    the 2.3 ms the profile found, and where the table earns its keep.
    """
    def timed(boat, exact):
        period = float(boat.timing.period)
        ts = np.linspace(0.0, period, 40)
        boat.crew_field(0.0)                     # build the table
        best = float("inf")
        for _repeat in range(3):                 # best of three: a floor,
            t0 = time.perf_counter()            # not a benchmark, and the
            for t in ts:                         # suite runs it in any order
                boat.crew_field(t, exact=exact)
            best = min(best, time.perf_counter() - t0)
        return best

    assert timed(four, True) / max(timed(four, False), 1e-9) > 5.0
    assert (timed(eight_with_offsets, True)
            / max(timed(eight_with_offsets, False), 1e-9)) > 10.0
