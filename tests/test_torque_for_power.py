"""Rower power, matched under either catch.

The rest catch's torque is closed-form and must stay exactly that.  The sweep
catch's rower work includes the energy the sweep carries into the water, which
depends on the entry speed the run decides, so its torque is matched on a
settle and cached.  Profiles name their catch; every profile today names
"rest", so nothing that runs a profile changes yet.
"""

import dataclasses

import numpy as np
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.sim.dynamic_oar import (DynamicOarSimulator, _match_key,
                                      simulator_for)


def _research(name="1x", rate=30.0, watts=300.0):
    boat = physics.resolve("research").apply(catalog.build(name, rate=rate))
    boat.handle_watts = watts
    return boat


def test_the_research_profiles_catch_by_the_sweep_and_shipped_does_not():
    """``research`` and ``learned`` moved to [CR06]'s catch on 2026-09-13;
    ``shipped`` has no dynamic oar and keeps the rest default."""
    assert physics.resolve(physics.SHIPPED).catch == "rest"
    assert physics.resolve("research").catch == "sweep"
    assert physics.resolve("learned").catch == "sweep"


def _rest_profile(monkeypatch):
    rest = dataclasses.replace(physics.resolve("research"),
                               name="research-rest", catch="rest")
    monkeypatch.setitem(physics.PROFILES, "research-rest", rest)
    boat = rest.apply(catalog.build("1x", rate=30.0))
    boat.handle_watts = 300.0
    return boat


def test_an_unknown_catch_rule_is_refused():
    with pytest.raises(ValueError, match="catch rule"):
        physics.PhysicsProfile(name="x", summary="x", blade_tier=1,
                               rower="prescribed", oar="dynamic",
                               catch="early")


def test_the_sweep_catch_needs_the_dynamic_oar():
    with pytest.raises(ValueError, match="dynamic oar"):
        physics.PhysicsProfile(name="x", summary="x", blade_tier=0,
                               rower="prescribed", catch="sweep")


def test_a_different_hull_shape_is_a_different_cache_entry():
    """Two boats identical but for their offsets row different strokes, so
    they must not share a matched torque."""
    from coxswain.hydro.hull import parametric_offsets

    a = _research()
    b = _research()
    b.offsets = parametric_offsets(7.925, 0.274, 0.101, fullness=2.4,
                                   freeboard=0.20)
    assert _match_key(a, 300.0, "sweep", "slip") != \
        _match_key(b, 300.0, "sweep", "slip")
    assert _match_key(a, 300.0, "sweep", "slip") == \
        _match_key(_research(), 300.0, "sweep", "slip")


def test_a_different_pull_shape_is_a_different_cache_entry():
    """The shape decides how much work a peak torque does; a study shape
    must not be handed the default shape's matched torque."""
    from coxswain.crew.oarlock import OarForceProfile

    default = _research()
    study = physics.resolve("research").apply(catalog.build(
        "1x", rate=30.0, force_profile=OarForceProfile(shape="cr06")))
    study.handle_watts = 300.0
    assert _match_key(default, 300.0, "sweep", "slip") != \
        _match_key(study, 300.0, "sweep", "slip")


def test_the_rest_catch_torque_is_the_closed_form_exactly():
    boat = _research("8+", rate=28.0, watts=380.0)
    assert DynamicOarSimulator.torque_for_power(boat, 380.0) == \
        DynamicOarSimulator.peak_torque_for_power(boat, 380.0)


def test_the_factory_passes_a_rest_profiles_catch_through(monkeypatch):
    boat = _rest_profile(monkeypatch)
    sim = simulator_for(boat)
    assert sim.catch == "rest"
    assert sim.peak_torque == DynamicOarSimulator.peak_torque_for_power(
        boat, 300.0)


def test_the_factory_passes_the_research_profiles_sweep_catch_through(
        monkeypatch):
    """A planted cache entry stands in for the settle, which is slow; it is
    removed again after the test so no later test can be handed it."""
    boat = _research()
    monkeypatch.setitem(DynamicOarSimulator._MATCHED,
                        _match_key(boat, 300.0, "sweep", "slip"), 123.0)
    sim = simulator_for(boat)
    assert sim.catch == "sweep" and sim.peak_torque == 123.0


def test_the_factory_lets_an_explicit_catch_override_the_profile():
    sim = simulator_for(_research(), catch="rest")
    assert sim.catch == "rest"
    assert sim.peak_torque == DynamicOarSimulator.peak_torque_for_power(
        _research(), 300.0)


@pytest.mark.slow
def test_the_sweep_catch_torque_is_matched_and_cached():
    boat = _research()
    DynamicOarSimulator._MATCHED.clear()
    torque = DynamicOarSimulator.torque_for_power(boat, 300.0, catch="sweep",
                                                  start=4.2)
    assert DynamicOarSimulator.torque_for_power(boat, 300.0,
                                                catch="sweep") == torque
    # the rower pays for the energy carried in, so the pull is lighter
    assert torque < DynamicOarSimulator.peak_torque_for_power(boat, 300.0)
    sim = DynamicOarSimulator(boat, peak_torque=torque, catch="sweep")
    run = sim.run_strokes(12, surge_speed=4.2)
    assert run.settled_power() == pytest.approx(300.0, rel=0.015)


@pytest.mark.slow
def test_a_sweep_catch_profile_builds_its_crew_at_matched_power(monkeypatch):
    sweep = dataclasses.replace(physics.resolve("research"),
                                name="research-sweep", catch="sweep")
    monkeypatch.setitem(physics.PROFILES, "research-sweep", sweep)
    boat = sweep.apply(catalog.build("1x", rate=30.0))
    boat.handle_watts = 300.0
    sim = simulator_for(boat)
    assert sim.catch == "sweep"
    assert sim.peak_torque == DynamicOarSimulator.torque_for_power(
        boat, 300.0, catch="sweep")
    assert sim.peak_torque != DynamicOarSimulator.peak_torque_for_power(
        boat, 300.0)
