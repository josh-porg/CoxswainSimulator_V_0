"""Research profiles row [CR06]'s *fitted* sculling blade coefficient.

[CR06] computed C2 = (1/2) rho C0 A0 = 58.7 for a scull.  The value that
minimised the error against their own singles data was about 2.4x that, and
with it their oar-angle error halved (their p. 208, Fig. 7).  Checked three
ways here before adoption (TRACKING): their athlete's traces alone imply
2.3-3.4x, her oar angle needs 2.4x, and on Holt's singles the speed gap closes
3.0-3.5 points with blade efficiency landing in Kleshnev's band.

The fit lumps transient added mass into a steady coefficient, so it is
exclusive of the Patton blade added mass, and it was fitted on singles, so
sweep blades keep 84.5.  The shipped game keeps 58.7.
"""

import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.crew.oardynamics import OarDynamics
from coxswain.crew.oarlock import BladeModel
from coxswain.sim.dynamic_oar import DynamicOarSimulator

NOMINAL, FITTED = 58.7, 140.88


def test_shipped_keeps_the_computed_value_and_research_carries_the_fit():
    assert physics.resolve(physics.SHIPPED).scull_c2 is None
    assert physics.resolve("research").scull_c2 == FITTED
    assert physics.resolve("learned").scull_c2 == FITTED
    assert FITTED == pytest.approx(2.4 * NOMINAL)


def test_the_blade_models_own_defaults_are_untouched():
    """The fit lives in the profile, not in the blade model: the shipped
    game and every un-profiled caller still get [CR06]'s computed values."""
    assert BladeModel.sculling().c2 == NOMINAL
    assert BladeModel.sweep().c2 == 84.5


def test_a_research_single_builds_its_dynamic_oar_with_the_fitted_value():
    boat = physics.resolve("research").apply(catalog.build("1x", rate=30.0))
    assert boat.scull_c2 == FITTED
    assert OarDynamics.from_boat(boat).blade.c2 == pytest.approx(FITTED)


def test_an_unprofiled_single_keeps_the_computed_value():
    boat = catalog.build("1x", rate=30.0)
    assert getattr(boat, "scull_c2", None) is None
    assert OarDynamics.from_boat(boat).blade.c2 == pytest.approx(NOMINAL)


def test_a_shipped_single_keeps_the_computed_value():
    boat = physics.resolve(physics.SHIPPED).apply(catalog.build("1x", rate=30.0))
    assert getattr(boat, "scull_c2", None) is None
    assert OarDynamics.from_boat(boat).blade.c2 == pytest.approx(NOMINAL)


def test_sweep_rigs_are_untouched_because_the_fit_is_from_singles():
    boat = physics.resolve("research").apply(catalog.build("8+", rate=32.0))
    assert getattr(boat, "scull_c2", None) is None
    assert OarDynamics.from_boat(boat).blade.c2 == pytest.approx(84.5)


def test_an_explicit_blade_still_wins():
    """``from_boat(blade=...)`` is how a study overrides the physics; the
    profile must not reach past it."""
    boat = physics.resolve("research").apply(catalog.build("1x", rate=30.0))
    mine = BladeModel.sculling(outboard=2.0)
    assert OarDynamics.from_boat(boat, blade=mine).blade is mine


def test_the_efficiency_wiring_sees_the_same_value():
    boat = physics.resolve("research").apply(catalog.build("1x", rate=30.0))
    assert physics.resolve("research").blade_model(boat).c2 == pytest.approx(FITTED)


def test_the_fitted_c2_and_patton_added_mass_are_refused_together():
    """Her stroke admits one or the other, not both.

    [CR06] attribute their 2.4x to transient added mass, but fitting ``C2``
    and an added mass jointly to the blade load their own athlete's oar
    balance implies gives ``m_a = -1.5 +- 0.15 kg`` against Patton's 13.1,
    with ``C2`` unmoved at 140.3 (TRACKING, 2026-09-18).  So the exclusion
    rests on the measurement, not on their explanation of it.
    """
    boat = physics.resolve("research").apply(catalog.build("1x", rate=30.0))
    with pytest.raises(ValueError, match="exclusive"):
        DynamicOarSimulator(boat, peak_torque=120.0, blade_added_mass="patton")


def test_added_mass_still_runs_on_a_boat_without_the_fit():
    boat = catalog.build("1x", rate=30.0)
    DynamicOarSimulator(boat, peak_torque=120.0, blade_added_mass="patton")


def test_a_non_positive_scull_c2_is_refused():
    with pytest.raises(ValueError, match="scull_c2"):
        physics.PhysicsProfile(name="x", summary="x", blade_tier=1,
                               rower="prescribed", oar="dynamic", scull_c2=0.0)


def test_a_blade_coefficient_without_a_blade_is_refused():
    with pytest.raises(ValueError, match="scull_c2"):
        physics.PhysicsProfile(name="x", summary="x", blade_tier=0,
                               rower="prescribed", scull_c2=140.88)
