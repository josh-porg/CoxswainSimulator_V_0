r"""The tabulated oar force against ``oar_force`` evaluated directly.

Same argument as the stroke table: the oarlock force is a function of
stroke time, the profile and the sweep, evaluated eight times per
derivative through Python; it is periodic; it is solved once.  Held to
the direct function on both sides, and the sweep rate with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.oarlock import oar_force


@pytest.fixture(scope="module")
def eight():
    boat = catalog.eight(rate=32.0)
    boat.tabulate_crew = True
    return boat


def test_the_table_matches_oar_force_on_both_sides(eight):
    rng = np.random.default_rng(2)
    period = float(eight.timing.period)
    peak = max(float(np.abs(eight.oar_force_at(t, +1, exact=True)).max())
               for t in np.linspace(0.0, period, 50, endpoint=False))
    for t in rng.uniform(-period, 2.0 * period, 60):
        for side in (+1, -1):
            exact = oar_force(t, eight.timing, side, eight.force_profile,
                              eight.oar_sweep)
            table = eight.oar_force_at(t, side)
            assert np.abs(exact - table).max() < 2e-3 * peak, (t, side)


def test_starboard_is_port_with_the_lateral_reversed(eight):
    """The one thing the table assumes about ``oar_force``."""
    for t in np.linspace(0.0, float(eight.timing.period), 17):
        port = eight.oar_force_at(t, +1)
        stbd = eight.oar_force_at(t, -1)
        assert port[0] == pytest.approx(stbd[0])
        assert port[2] == pytest.approx(stbd[2])
        assert port[1] == pytest.approx(-stbd[1])


def test_the_sweep_rate_matches(eight):
    rng = np.random.default_rng(4)
    period = float(eight.timing.period)
    scale = max(abs(float(eight.oar_sweep.rate(t, eight.timing)))
                for t in np.linspace(0.0, period, 50, endpoint=False))
    for t in rng.uniform(0.0, period, 40):
        exact = float(eight.oar_sweep.rate(t, eight.timing))
        table = eight.oar_rate_at(t)
        assert abs(exact - table) < 1e-2 * scale, t


def test_the_force_is_zero_through_the_recovery_in_the_table(eight):
    """A table must not smear the drive into the recovery."""
    period = float(eight.timing.period)
    drive = float(eight.timing.drive_duration)
    for t in np.linspace(drive + 0.02, period - 0.005, 12):
        assert np.abs(eight.oar_force_at(t, +1)).max() < 1e-9, t


def test_the_derivative_agrees_with_and_without_the_oar_table(eight):
    from coxswain.sim.simulator import RowingSimulator

    sim = RowingSimulator(eight)
    state = sim.initial_state(surge_speed=5.0)
    rng = np.random.default_rng(9)
    # the exact derivative's per-row peak over one stroke
    boat = eight
    boat.tabulate_crew = False
    peaks = np.zeros(len(np.asarray(sim.derivative(0.0, state))))
    for tt in np.linspace(0.0, float(boat.timing.period), 24, endpoint=False):
        sim._crew_cache = sim._hand_cache = (None, None)
        peaks = np.maximum(peaks, np.abs(np.asarray(sim.derivative(tt, state))))
    peaks = np.maximum(peaks, 1e-6)
    boat.tabulate_crew = True
    for t in rng.uniform(0.0, float(eight.timing.period), 10):
        eight.tabulate_crew = True
        sim._crew_cache = sim._hand_cache = (None, None)
        with_table = np.asarray(sim.derivative(t, state))
        eight.tabulate_crew = False
        sim._crew_cache = sim._hand_cache = (None, None)
        exact = np.asarray(sim.derivative(t, state))
        eight.tabulate_crew = True
        # Scaled per ROW by that quantity's peak over the stroke, not by
        # its value at this instant: an acceleration passing through
        # zero is not a large relative error, it is zero.
        assert np.all(np.abs(with_table - exact) <= 1e-2 * peaks), (
            t, np.abs(with_table - exact) / peaks)
