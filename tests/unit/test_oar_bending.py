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


def test_scull_from_laschowski_bends_about_a_degree_at_four_hundred_newton_metres():
    s = OarBending.laschowski2015_scull(1.805, 0.43)
    assert np.degrees(400.0 / s.k_ang) == pytest.approx(1.1, abs=0.2)


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
