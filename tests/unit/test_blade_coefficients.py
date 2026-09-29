"""Tier 2's full-size corrections: sourced factors on the flume amplitudes, off by default."""
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.crew.liftdrag import LiftDragBlade
from coxswain.sim.dynamic_oar import DynamicOarSimulator, _match_key


def test_the_full_scale_corrections_are_the_sourced_ratios():
    import numpy as np
    flume = LiftDragBlade.big_blade(outboard=1.8, area=0.083)
    coppel = LiftDragBlade.big_blade_full_scale(outboard=1.8, area=0.083, source="coppel")
    plates = LiftDragBlade.big_blade_full_scale(outboard=1.8, area=0.083, source="sliasas_tullis")
    for deg, lift_ratio, drag_ratio in ((20, 0.57 / 0.78, 0.31 / 0.44), (45, 1.12 / 1.20, 0.82 / 1.11),
                                        (90, 1.0, 1.36 / 1.85)):
        a = np.radians(deg)
        (fl, fd), (cl, cd) = flume.lift_drag(a), coppel.lift_drag(a)
        assert cd == pytest.approx(drag_ratio * fd, rel=1e-9)
        if deg < 90:
            assert cl == pytest.approx(lift_ratio * fl, rel=1e-9)
    a = np.radians(33.0)
    assert plates.lift_drag(a)[0] == pytest.approx(0.8 * flume.lift_drag(a)[0])
    assert plates.lift_drag(a)[1] == pytest.approx(0.7 * flume.lift_drag(a)[1])
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
    assert sim._liftdrag[0].correction == LiftDragBlade.FULL_SCALE["coppel"]
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


# -- the clashes the compatibility map found, now refused --------------------------------

def _deep():
    from coxswain.crew.blade_depth import BladeDepth
    return BladeDepth.constant_cover(0.0, lever=1.795, width=0.193, reference="deep")


def _mean():
    from coxswain.crew.blade_depth import BladeDepth
    return BladeDepth.constant_cover(0.0, lever=1.795, width=0.193, reference="mean")


def test_a_deep_depth_is_refused_where_the_coefficients_already_carry_a_surface(single):
    with pytest.raises(ValueError, match="fitted sculling C2"):
        DynamicOarSimulator(single, peak_torque=100.0, blade_depth=_deep())
    with pytest.raises(ValueError, match="CG07"):
        DynamicOarSimulator(single, peak_torque=100.0, blade_law="liftdrag", blade_depth=_deep())
    with pytest.raises(ValueError, match="ST09"):
        DynamicOarSimulator(single, peak_torque=100.0, blade_law="liftdrag",
                            blade_coefficients="sliasas_tullis", blade_depth=_deep())
    sim = DynamicOarSimulator(single, peak_torque=100.0, blade_law="liftdrag")
    with pytest.raises(ValueError, match="CG07"):
        sim.blade_depth = _deep()                    # assignment is checked too


def test_a_rigid_lid_set_takes_the_depth_and_a_shape_only_depth_goes_anywhere(single):
    DynamicOarSimulator(single, peak_torque=100.0, blade_law="liftdrag",
                        blade_coefficients="coppel", blade_depth=_deep())
    DynamicOarSimulator(single, peak_torque=100.0, blade_depth=_mean())
    DynamicOarSimulator(single, peak_torque=100.0, blade_law="liftdrag", blade_depth=_mean())


def test_strips_are_refused_with_the_fitted_scull_c2(single):
    with pytest.raises(ValueError, match="strip integration"):
        DynamicOarSimulator(single, peak_torque=100.0, blade_span=0.43)
    DynamicOarSimulator(single, peak_torque=100.0, blade_law="liftdrag", blade_span=0.43)
    sweep = physics.resolve("research").apply(catalog.pair(rate=32.0)) if hasattr(catalog, "pair") else None
    if sweep is not None:
        DynamicOarSimulator(sweep, peak_torque=100.0, blade_span=0.52)   # sweep oars: nominal C2
