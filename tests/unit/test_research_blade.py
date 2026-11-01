"""The research profile's blade: the fins' cross-flow law on the blade, normal load only, in strips (2026-10-31).

SOURCES sec. 199-202. [CG07]'s measured points resolve to a force normal to a stalled blade within ~2 %, and [H10]
measured ~3 % on the water mid-drive, so the cross-flow law's tangential term (an artefact of resolving an
attached-flow lift with a separated-flow drag) is off. Closed loop it passes rung 2's gate on both athletes,
nothing fitted. These tests pin that the profile names it, that every profile-driven path builds it, and that the
law itself carries no tangential load.
"""
import numpy as np
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.crew.liftdrag import CrossflowBlade, LiftDragBlade
from coxswain.sim import dynamic_oar
from coxswain.sim.dynamic_oar import DynamicOarSimulator, simulator_for


def test_research_names_the_cross_flow_normal_blade():
    research = physics.resolve("research")
    assert research.blade_law == "liftdrag"
    assert research.blade_coefficients == "crossflow-normal@2.2"
    assert research.blade_strips is True
    assert research.scull_c2 is None


def test_shipped_has_no_blade_options_to_move():
    shipped = physics.resolve(physics.SHIPPED)
    assert shipped.blade_law == "slip" and shipped.blade_coefficients == "cg07" and not shipped.blade_strips


@pytest.mark.parametrize("name, span", [("1x", 0.43), ("4+", 0.52), ("8+", 0.52)])
def test_blade_options_integrate_over_the_rigs_own_blade(name, span):
    boat = catalog.build(name, rate=30.0)
    opts = physics.resolve("research").blade_options(boat)
    assert opts["blade_law"] == "liftdrag"
    assert opts["blade_coefficients"] == "crossflow-normal@2.2"
    assert opts["blade_span"] == pytest.approx(span)


def test_the_normal_only_law_carries_no_tangential_load():
    full = LiftDragBlade.crossflow(outboard=1.8, area=0.083, length=0.43, face_on=2.2)
    normal = LiftDragBlade.crossflow(outboard=1.8, area=0.083, length=0.43, face_on=2.2, tangential=False)
    assert isinstance(normal, CrossflowBlade)
    for alpha in np.radians([10.0, 30.0, 45.0, 70.0, 90.0]):
        cn_f, _ct_f = full.coefficients(alpha)
        cn_n, ct_n = normal.coefficients(alpha)
        assert cn_n == pytest.approx(cn_f)
        assert ct_n == 0.0
    assert full.coefficients(np.radians(45.0))[1] > 0.1          # the artefact the switch removes


def test_the_simulator_builds_the_normal_only_blade_from_its_name():
    boat = physics.resolve("research").apply(catalog.build("1x", rate=30.0))
    sim = DynamicOarSimulator(boat, peak_torque=100.0, blade_law="liftdrag",
                              blade_coefficients="crossflow-normal@2.2", blade_span=0.43)
    assert all(isinstance(b, CrossflowBlade) and b.tangential is False for b in sim._liftdrag)
    sim = DynamicOarSimulator(boat, peak_torque=100.0, blade_law="liftdrag", blade_coefficients="crossflow@2.2")
    assert all(b.tangential is True for b in sim._liftdrag)


def test_simulator_for_takes_the_profiles_blade(monkeypatch):
    seen = {}

    def fake_torque(cls, boat, watts, **kw):
        seen.update(kw)
        return 100.0
    monkeypatch.setattr(DynamicOarSimulator, "torque_for_power", classmethod(fake_torque))
    boat = physics.resolve("research").apply(catalog.build("1x", rate=30.0))
    boat.handle_watts = 250.0
    sim = simulator_for(boat)
    assert sim.blade_law == "liftdrag" and sim.blade_coefficients == "crossflow-normal@2.2"
    assert sim.blade_span == pytest.approx(0.43)
    assert seen["blade_law"] == "liftdrag" and seen["blade_coefficients"] == "crossflow-normal@2.2"
    assert seen["blade_span"] == pytest.approx(0.43)


def test_the_scorecard_settles_on_the_profiles_blade(monkeypatch):
    from coxswain.validation import scorecard
    built = {}

    class Stop(Exception):
        pass

    def fake_torque(cls, boat, watts, **kw):
        built.update(kw)
        raise Stop
    monkeypatch.setattr(DynamicOarSimulator, "torque_for_power", classmethod(fake_torque))
    boat = physics.resolve("research").apply(catalog.build("4+", rate=30.0))
    with pytest.raises(Stop):
        scorecard.settle_dynamic(boat, 250.0, 4.0)
    assert built["blade_law"] == "liftdrag" and built["blade_coefficients"] == "crossflow-normal@2.2"
    assert built["blade_span"] == pytest.approx(0.52)


def test_a_bad_blade_configuration_is_refused():
    with pytest.raises(ValueError, match="blade law"):
        physics.PhysicsProfile("x", "", blade_tier=1, rower="prescribed", oar="dynamic", blade_law="magic")
    with pytest.raises(ValueError, match="dynamic oar"):
        physics.PhysicsProfile("x", "", blade_tier=0, rower="prescribed", blade_law="liftdrag")
    with pytest.raises(ValueError, match="lift-drag"):
        physics.PhysicsProfile("x", "", blade_tier=1, rower="prescribed", oar="dynamic",
                               blade_coefficients="crossflow@2.2")


def test_the_crossflow_names_parse():
    assert dynamic_oar._crossflow_face_on("crossflow-normal@2.2") == 2.2
    assert dynamic_oar._crossflow_face_on("crossflow@1.3") == 1.3
    assert dynamic_oar._crossflow_tangential("crossflow-normal@2.2") is False
    assert dynamic_oar._crossflow_tangential("crossflow@2.2") is True
