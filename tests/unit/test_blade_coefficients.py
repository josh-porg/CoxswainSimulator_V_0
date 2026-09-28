"""Tier 2's full-size corrections: sourced factors on the flume amplitudes, off by default."""
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.crew.liftdrag import LiftDragBlade
from coxswain.sim.dynamic_oar import DynamicOarSimulator, _match_key


def test_the_full_scale_amplitudes_are_the_flume_ones_times_the_sourced_factors():
    flume = LiftDragBlade.big_blade(outboard=1.8, area=0.083)
    coppel = LiftDragBlade.big_blade_full_scale(outboard=1.8, area=0.083, source="coppel")
    plates = LiftDragBlade.big_blade_full_scale(outboard=1.8, area=0.083, source="sliasas_tullis")
    assert (coppel.lift_amplitude, coppel.drag_amplitude) == pytest.approx(
        (flume.lift_amplitude, 0.65 * flume.drag_amplitude))
    assert (plates.lift_amplitude, plates.drag_amplitude) == pytest.approx(
        (0.8 * flume.lift_amplitude, 0.7 * flume.drag_amplitude))
    with pytest.raises(ValueError):
        LiftDragBlade.big_blade_full_scale(outboard=1.8, source="guess")


@pytest.fixture(scope="module")
def single():
    return physics.resolve("research").apply(catalog.single_scull(rate=32.0))


def test_the_simulator_default_is_the_flume(single):
    sim = DynamicOarSimulator(single, peak_torque=100.0, blade_law="liftdrag")
    assert sim.blade_coefficients == "cg07"
    assert sim._liftdrag[0].drag_amplitude == pytest.approx(2.07)


def test_the_simulator_takes_a_full_size_correction(single):
    sim = DynamicOarSimulator(single, peak_torque=100.0, blade_law="liftdrag",
                              blade_coefficients="coppel")
    assert sim._liftdrag[0].drag_amplitude == pytest.approx(0.65 * 2.07)
    assert sim._liftdrag[0].outboard == pytest.approx(float(sim._oars[0].outboard))


def test_coefficients_are_refused_off_tier_two_or_unknown(single):
    with pytest.raises(ValueError, match="tier 2"):
        DynamicOarSimulator(single, peak_torque=100.0, blade_coefficients="coppel")
    with pytest.raises(ValueError, match="unknown blade coefficients"):
        DynamicOarSimulator(single, peak_torque=100.0, blade_law="liftdrag",
                            blade_coefficients="fitted")


def test_matched_torques_do_not_share_a_cache_entry_across_coefficients(single):
    assert (_match_key(single, 300.0, "sweep", "liftdrag")
            != _match_key(single, 300.0, "sweep", "liftdrag", "none", None, "coppel"))
