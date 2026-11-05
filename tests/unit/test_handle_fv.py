"""The crew's handle force-velocity relation (coxswain/crew/handle_fv.py; SOURCES §218)."""
import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.handle_fv import ERGOMETER_V0, HandleForceVelocity, PlannedSpeed
from coxswain.sim.dynamic_oar import DynamicOarSimulator, _match_key


def test_the_pull_is_unchanged_at_the_planned_speed_and_falls_along_the_slope():
    fv = HandleForceVelocity(5.0)
    assert fv.factor(2.0, 2.0) == pytest.approx(1.0)
    assert fv.factor(3.0, 2.0) == pytest.approx(2.0 / 3.0)        # running ahead pulls less
    assert fv.factor(1.0, 2.0) == pytest.approx(4.0 / 3.0)        # held back pulls more
    assert fv.factor(6.0, 2.0) == 0.0                             # never negative
    with pytest.raises(ValueError):
        fv.factor(1.0, 5.0)
    assert HandleForceVelocity.onwater().v0 == pytest.approx(ERGOMETER_V0 / 0.69)


def test_planned_speed_is_the_boats_own_sweep():
    boat = catalog.single_scull(rate=30.0)
    planned = PlannedSpeed(boat)
    timing = boat.timing
    tau = 0.4 * timing.drive_fraction * timing.period
    angle = float(boat.oar_sweep(tau, timing))
    assert planned.rate(angle) == pytest.approx(abs(float(boat.oar_sweep.rate(tau, timing))), rel=0.02)


def test_only_the_clock_crew_and_the_cache_key_sees_it():
    boat = catalog.single_scull(rate=30.0)
    with pytest.raises(ValueError):
        DynamicOarSimulator(boat, peak_torque=150.0, crew="follows", handle_fv=HandleForceVelocity(5.0))
    sim = DynamicOarSimulator(boat, peak_torque=150.0, handle_fv=HandleForceVelocity(5.0))
    assert sim.handle_fv.v0 == 5.0
    a = _match_key(boat, 300.0, "sweep", "slip")
    b = _match_key(boat, 300.0, "sweep", "slip", handle_fv=HandleForceVelocity(5.0))
    c = _match_key(boat, 300.0, "sweep", "slip", handle_fv=HandleForceVelocity(4.0))
    assert len({a, b, c}) == 3


def test_the_torque_reads_the_oar_rate():
    boat = catalog.single_scull(rate=30.0)
    sim = DynamicOarSimulator(boat, peak_torque=150.0, handle_fv=HandleForceVelocity(5.0))
    plain = DynamicOarSimulator(boat, peak_torque=150.0)
    from coxswain.core.state import State
    state = State.from_vector(np.zeros(12))
    angle = np.radians(10.0)
    planned = sim._planned.rate(angle)
    assert sim._torque(0, angle, state, 0.1, planned) == pytest.approx(plain._torque(0, angle, state, 0.1))
    assert sim._torque(0, angle, state, 0.1, planned * 1.5) < plain._torque(0, angle, state, 0.1)
    assert sim._torque(0, angle, state, 0.1) == pytest.approx(plain._torque(0, angle, state, 0.1))   # no rate given
