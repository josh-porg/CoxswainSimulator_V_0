"""Oar-shaft bending as a quasi-static hinge (coxswain/crew/oar_bending.py; DynamicOarSimulator(oar_bending=...)).

SOURCES §186, §193. A research option: off by default, refused outside the sweep catch, carried in the matched-torque
cache key, and stiff enough a shaft reproduces the rigid oar.
"""
import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.crew.oar_bending import OarBending, beam_hinge_ratio
from coxswain.sim.control import Coxswain
from coxswain.sim.dynamic_oar import DynamicOarSimulator, _match_key


def test_robert_bench_constants_and_the_beam_check():
    b = OarBending.robert2019_sweep()
    assert b.k_ang == pytest.approx(2948.0)
    assert b.hinge_inboard_of_root == pytest.approx(2948.0 / 3424.1)
    # a uniform cantilever puts the hinge within 10% of their bench on their oar (root 1.558 m, blade 0.52 m)
    assert beam_hinge_ratio(1.558, 2.078) == pytest.approx(b.hinge_inboard_of_root, rel=0.10)


def test_scull_from_laschowskis_measured_shaft():
    """[L14] Table 3's medium shaft slope (4.63 deg at 201 N, 1.299 m out) gives an EI that matches the thesis's own
    Table 2 near the blade (1992-2169 N m^2); on a 1.805 m scull it bends ~10 deg at 400 N m at the pin. The old law
    read the abstract's 1.18 deg (a difference between shafts) as the slope and bent ~1 deg (SOURCES sec. 217)."""
    from coxswain.crew.oar_bending import L14_EXTRA_SOFT_EI, L14_MEDIUM_EI
    assert 1992.0 * 0.95 < L14_MEDIUM_EI < 2169.0 * 1.05
    assert L14_EXTRA_SOFT_EI < L14_MEDIUM_EI
    s = OarBending.laschowski2014_scull(1.805, 0.43)
    assert np.degrees(400.0 / s.k_ang) == pytest.approx(9.7, abs=0.3)


def _single(bending, strokes=4):
    boat = catalog.single_scull(rate=30.0)
    boat.power_scales = np.ones(boat.n_seats)
    sim = DynamicOarSimulator(boat, peak_torque=150.0, catch="sweep", oar_bending=bending,
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    return sim, sim.run_strokes(strokes, surge_speed=4.0)


def test_a_very_stiff_shaft_is_the_rigid_oar():
    _sim, rigid = _single(None)
    sim, stiff = _single(OarBending(1e9, 0.9))
    assert stiff.settled_speed() == pytest.approx(rigid.settled_speed(), rel=1e-4)
    assert np.max(np.abs(sim._bend)) < 1e-4


def test_a_soft_shaft_bends_and_changes_the_stroke():
    sim, soft = _single(OarBending.robert2019_sweep())
    _rigid_sim, rigid = _single(None)
    assert np.isfinite(soft.settled_speed())
    assert sim.bend_solve_failures == 0          # every step's bend balanced (STRUCTURAL_ISSUES S13)
    assert abs(soft.settled_speed() - rigid.settled_speed()) > 1e-4


def test_bending_is_refused_outside_the_sweep_catch():
    boat = catalog.single_scull(rate=30.0)
    with pytest.raises(ValueError):
        DynamicOarSimulator(boat, peak_torque=150.0, catch="rest", oar_bending=OarBending.robert2019_sweep())


def test_bending_is_in_the_matched_torque_key():
    boat = catalog.single_scull(rate=30.0)
    a = _match_key(boat, 300.0, "sweep", "slip")
    b = _match_key(boat, 300.0, "sweep", "slip", bending=OarBending.robert2019_sweep())
    assert a != b


def test_one_load_path_slip_law_unchanged_by_routing():
    # STRUCTURAL_ISSUES S1: the plain slip law now goes through _blade_loads_arm like every other option
    boat = catalog.single_scull(rate=30.0)
    sim = DynamicOarSimulator(boat, peak_torque=150.0, catch="sweep",
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    run = sim.run_strokes(2, surge_speed=4.0)
    assert np.isfinite(run.settled_speed())


def test_liftdrag_blade_parameter_and_its_cache_key():
    # STRUCTURAL_ISSUES S3: a study blade enters through the constructor, and the matched-torque key carries it
    from coxswain.crew.liftdrag import LiftDragBlade
    boat = catalog.single_scull(rate=30.0)
    blade = LiftDragBlade.crossflow(outboard=1.8, area=0.083, length=0.43, face_on=2.2)
    sim = DynamicOarSimulator(boat, peak_torque=150.0, catch="sweep", blade_law="liftdrag", liftdrag_blade=blade,
                              coxswain=Coxswain(rudder_override=lambda t, s: 0.0), fast=True)
    assert all(b is blade for b in sim._liftdrag)
    assert _match_key(boat, 300.0, "sweep", "liftdrag") != _match_key(boat, 300.0, "sweep", "liftdrag", blade=blade)
    with pytest.raises(ValueError):
        DynamicOarSimulator(boat, peak_torque=150.0, catch="sweep", blade_law="slip", liftdrag_blade=blade)


def test_tip_outboard_is_the_tip_and_the_centre_is_named():
    oar = catalog.single_scull(rate=30.0).rig.seats[0].oarlocks[0].oar
    assert oar.tip_outboard == oar.outboard
    assert oar.blade_centre_outboard == pytest.approx(oar.tip_outboard - 0.5 * oar.blade_length)


def test_a_bending_profile_bends_sculls_and_the_simulator_uses_the_boats_own():
    """SOURCES §217: a profile with ``oar_bending="sourced"`` bends sculls by [L14] and leaves sweep oars rigid; no
    shipped or research profile sets it (withdrawn the day it was made a default); False forces the rigid oar."""
    import dataclasses
    import pytest
    from coxswain import physics
    from coxswain.boats import catalog
    from coxswain.sim.dynamic_oar import _resolve_bending
    assert physics.resolve("research").oar_bending is None
    bending = dataclasses.replace(physics.resolve("research"), oar_bending="sourced")
    scull = catalog.single_scull(rate=32.0, rower_mass=85.0, rower_stature=1.90)
    bending.apply(scull)
    assert "L14" in scull.oar_bending.source
    sweep = catalog.eight(rate=32.0)
    bending.apply(sweep)
    assert getattr(sweep, "oar_bending", None) is None
    plain = catalog.single_scull(rate=32.0, rower_mass=85.0, rower_stature=1.90)
    physics.resolve("shipped").apply(plain)
    assert getattr(plain, "oar_bending", None) is None
    assert _resolve_bending(scull, None, "sweep") is scull.oar_bending
    assert _resolve_bending(scull, False, "sweep") is None            # a study's rigid oar
    assert _resolve_bending(scull, None, "rest") is None              # only stepped under the sweep catch
    with pytest.raises(ValueError):
        _resolve_bending(scull, scull.oar_bending, "rest")


def test_kleshnevs_energy_efficiency_barely_sees_the_bend():
    """Kleshnev's eb = (Ph - Pwb) / Ph is an energy ratio on the gate-angle track. A quasi-static shaft returns what it
    stores by the release, so the gate track and the bent blade's own track give nearly the same figure, which is why
    it can be measured on bending oars; the load-weighted slip ratio is a different quantity (SOURCES §217)."""
    _sim, rigid = _single(None)
    for record in rigid.strokes:
        assert record.blade_efficiency_k07 == pytest.approx(record.blade_efficiency_k07_blade, abs=1e-12)
    _sim, soft = _single(OarBending.robert2019_sweep())
    gate = soft.settled_blade_efficiency(track="k07")
    own = soft.settled_blade_efficiency(track="k07_blade")
    assert 0.0 < gate < 1.0
    assert gate == pytest.approx(own, abs=0.02)
    with pytest.raises(ValueError):
        soft.settled_blade_efficiency(track="tip")


def test_the_blade_added_mass_is_refused_on_a_bending_oar():
    """The coupled added-mass solve takes the blade's acceleration from the shaft alone; until it carries the bend it
    runs rigid only (SOURCES §217). The boat's own bending is refused too, with the way out named."""
    import dataclasses
    from coxswain import physics
    bending = dataclasses.replace(physics.resolve("research"), oar_bending="sourced")
    boat = bending.apply(catalog.single_scull(rate=30.0))
    assert boat.oar_bending is not None
    kw = dict(peak_torque=0.0, catch="sweep", crew="handle", blade_law="liftdrag", blade_added_mass="patton")
    with pytest.raises(ValueError, match="oar_bending=False"):
        DynamicOarSimulator(boat, **kw)
    assert DynamicOarSimulator(boat, oar_bending=False, **kw).oar_bending is None
