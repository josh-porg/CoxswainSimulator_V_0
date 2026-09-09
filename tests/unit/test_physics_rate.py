r"""The trainer's physics rate, held to the reference rate.

The real-time loop integrated at 100 Hz -- 6.6 derivative evaluations
per 60 Hz frame with RK4 -- and each evaluation is the whole 6-DOF
model.  Measured: 60 Hz reproduces 100 Hz to 0.2 mm of position and
2 mm/s of speed over 24 s of an eight with timing scatter, and roll and
yaw to better than a thousandth of a degree.  So the trainer runs at
60 Hz (50 on the lowest tier), and this is what stops that becoming
"the trainer runs at 20 Hz" one convenient commit at a time.

The studies are untouched: ``run()`` takes whatever ``dt`` it is given.
"""

from __future__ import annotations

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.blade_contact import BladeContact
from coxswain.sim.control import Coxswain
from coxswain.sim.simulator import RowingSimulator


def _trajectory(rate_hz: float, seconds: float = 20.0):
    boat = catalog.eight(rate=32.0)
    boat.tabulate_crew = True
    boat.phase_offsets = np.linspace(-0.02, 0.02, boat.n_seats)
    sim = RowingSimulator(boat, coxswain=Coxswain(), fast=True,
                          blade_contact=BladeContact.from_boat(boat))
    result = sim.run(duration=seconds, dt=1.0 / rate_hz, surge_speed=4.6)
    return np.asarray(result.time), np.asarray(result.states)


@pytest.fixture(scope="module")
def reference():
    return _trajectory(100.0)


@pytest.mark.parametrize("rate_hz", [60.0, 50.0])
def test_a_lower_physics_rate_is_the_same_boat(reference, rate_hz):
    t_ref, s_ref = reference
    t_low, s_low = _trajectory(rate_hz)
    on_ref = np.vstack([np.interp(t_ref, t_low, s_low[k])
                        for k in range(s_low.shape[0])])
    speed_ref = np.hypot(s_ref[6], s_ref[7])
    speed_low = np.hypot(on_ref[6], on_ref[7])
    assert np.abs(on_ref[0] - s_ref[0]).max() < 0.01, "position, m"
    assert np.abs(speed_low - speed_ref).max() < 0.02, "speed, m/s"
    assert np.degrees(np.abs(on_ref[3] - s_ref[3]).max()) < 0.02, "roll"
    assert np.degrees(np.abs(on_ref[5] - s_ref[5]).max()) < 0.02, "yaw"


def test_the_tiers_do_not_go_below_fifty_hertz():
    """Fifty is the measured floor; nothing may set it lower quietly."""
    from coxswain.viz.menu import QUALITY_TIERS

    for tier in QUALITY_TIERS:
        assert tier.physics_hz >= 50.0, (tier.key, tier.physics_hz)
        assert tier.physics_hz <= 100.0, (tier.key, tier.physics_hz)
