"""The blade going in at the catch: BioRow's 4 deg burial, and its entrained water."""
import numpy as np
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.crew.blade_immersion import BIOROW_BURY_DEG, EntryImmersion
from coxswain.sim.control import Coxswain
from coxswain.sim.dynamic_oar import DynamicOarSimulator


def test_wetted_fraction_runs_zero_to_one_over_the_burial():
    law = EntryImmersion()
    bury = np.radians(BIOROW_BURY_DEG)
    assert law.wetted(0.0) == 0.0
    assert law.wetted(0.5 * bury) == pytest.approx(0.5)
    assert law.wetted(bury) == 1.0 and law.wetted(3.0 * bury) == 1.0
    assert law.wetted(-0.01) == 0.0


def test_mass_fraction_is_the_wetted_height_squared_and_its_slope_is_consistent():
    law = EntryImmersion(4.0, 0.5)
    bury = law.bury
    for x in (0.1 * bury, 0.4 * bury, 0.9 * bury):
        assert law.mass_fraction(x) == pytest.approx(0.5 * (x / bury) ** 2)
        h = 1e-7
        numeric = (law.mass_fraction(x + h) - law.mass_fraction(x - h)) / (2 * h)
        assert law.mass_fraction_slope(x) == pytest.approx(numeric, rel=1e-5)
    assert law.mass_fraction_slope(2.0 * bury) == 0.0


def test_immersion_is_refused_off_the_handle():
    boat = physics.resolve("research").apply(catalog.single_scull(rate=30.0))
    with pytest.raises(ValueError, match="crew='handle'"):
        DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", blade_law="liftdrag",
                            blade_immersion=EntryImmersion())


@pytest.fixture(scope="module")
def immersed():
    boat = physics.resolve("research").apply(catalog.single_scull(rate=30.0))
    boat.power_scales = np.ones(boat.n_seats)
    sim = DynamicOarSimulator(boat, peak_torque=0.0, catch="sweep", crew="handle",
                              blade_law="liftdrag", blade_added_mass="patton",
                              blade_immersion=EntryImmersion(4.0, 1.0),
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    return sim, sim.run_strokes(3, surge_speed=4.2)


def test_the_blade_is_dry_at_entry_and_buried_four_degrees_later(immersed):
    sim, result = immersed
    entry_angle = sim.entries[-1][2]
    assert sim._wetted(0, entry_angle) == 0.0
    assert sim._wetted(0, entry_angle - np.radians(2.0)) == pytest.approx(0.5)
    assert sim._wetted(0, entry_angle - np.radians(4.0)) == 1.0


def test_the_entrained_water_returns_the_momentum_it_takes(immersed):
    """F_a = -d(m_a w_n)/dt with m_a = 0 at entry and w_n ~ 0 at release: its impulse over
    the wet interval vanishes, which checks the pickup term m_a' w_n with the rest."""
    sim, result = immersed
    t, y = result.last_time, result.last_states
    air = sim._air_mask
    with_mass = np.array([sim.handle_torques(t[k], y[:, k], air[:, k])[0, 0] for k in range(t.size)])
    sim.blade_added_mass = "none"
    try:
        without = np.array([sim.handle_torques(t[k], y[:, k], air[:, k])[0, 0] for k in range(t.size)])
    finally:
        sim.blade_added_mass = "patton"
    arm = float(sim._oars[0].outboard)
    force = (with_mass - without) / arm                    # -F_a along the normal, N
    wet = ~air[0]
    impulse = np.trapezoid(np.where(wet, force, 0.0), t)
    scale = np.trapezoid(np.where(wet, np.abs(force), 0.0), t)
    assert scale > 1.0
    assert abs(impulse) < 0.05 * scale
