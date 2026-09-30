"""The research profile's fins on [WF58] eq. [1], with its own Munk factor (SOURCES sec. 169).

The shipped game must not move: its fins stay on the legacy law and its simulators on the
default Munk factor.  Both switches are in the matched-torque key.
"""
import dataclasses

import numpy as np
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.hydro import appendages as A
from coxswain.hydro.addedmass import DEFAULT_MUNK_FACTOR
from coxswain.sim.dynamic_oar import _match_key
from coxswain.sim.simulator import RowingSimulator


def test_the_shipped_game_keeps_its_fins_and_munk_factor():
    shipped = physics.resolve("shipped")
    assert shipped.fin_law == "legacy" and shipped.munk_factor is None
    boat = shipped.apply(catalog.build("8+", rate=28.0))
    assert all(s.lift_model == "legacy" for s in boat.appendages)
    assert not hasattr(boat, "munk_factor")
    assert RowingSimulator(boat).munk_factor == DEFAULT_MUNK_FACTOR


def test_the_research_profile_puts_its_fins_on_eq_1():
    research = physics.resolve("research")
    assert research.fin_law == "whicker_fehlner"
    assert research.fin_crossflow == 0.80                 # [WF58], square tips
    boat = research.apply(catalog.build("8+", rate=28.0))
    for surface in boat.appendages:
        assert surface.lift_model == "whicker_fehlner"
        assert surface.reflection == research.fin_reflection
        assert surface.crossflow_coefficient == 0.80
    assert boat.munk_factor == research.munk_factor
    assert physics.resolve("learned").fin_law == research.fin_law
    assert physics.resolve("learned").munk_factor == research.munk_factor


def test_an_explicit_munk_factor_still_wins():
    plain = catalog.build("8+", rate=28.0)
    plain.munk_factor = 0.2
    assert RowingSimulator(plain).munk_factor == 0.2
    assert RowingSimulator(plain, munk_factor=0.0).munk_factor == 0.0


def test_the_study_s_choices_are_the_profile_s():
    """SOURCES sec. 169: reflection 2 ([WF58]'s definition), Munk refitted to 0.49-0.50."""
    research = physics.resolve("research")
    assert research.fin_reflection == 2.0
    assert research.munk_factor == 0.50


def test_omega_is_the_quarter_chord_sweep():
    """[WF58] p. 20: Omega is the sweep of the quarter-chord line."""
    assert A.SKEG_EIGHT.quarter_chord_sweep == pytest.approx(A.SKEG_EIGHT.sweep)
    fin = catalog.build("8+", rate=28.0).appendages[0]
    b, cr, ct = fin.span, fin.chord, fin.chord * fin.taper_ratio
    tip_qc = b * np.tan(fin.sweep) + ct / 4.0         # leading edge runs straight to the tip
    assert np.tan(fin.quarter_chord_sweep) == pytest.approx((tip_qc - cr / 4.0) / b)
    assert 20.0 < np.degrees(fin.quarter_chord_sweep) < 28.0 < np.degrees(fin.sweep)


@pytest.mark.parametrize("reflection", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("v, yaw_rate, deflection", [(0.2, 0.04, 0.0), (-0.1, 0.0, 0.3),
                                                     (0.0, -0.05, -0.7)])
def test_the_casadi_fins_follow_the_law(reflection, v, yaw_rate, deflection):
    ca = pytest.importorskip("casadi")
    from coxswain.river import hydro_casadi

    fin = dataclasses.replace(catalog.build("8+", rate=28.0).appendages[0],
                              lift_model="whicker_fehlner", reflection=reflection,
                              crossflow_coefficient=0.80)
    want_f, want_m = A.surface_load(fin, np.array([5.0, v, 0.0]), yaw_rate, deflection)
    got_f, got_m = hydro_casadi.surface_load(fin, 5.0, v, yaw_rate, deflection,
                                                A.FRESH_WATER.density)
    np.testing.assert_allclose(np.array(ca.DM(got_f)).ravel(), want_f, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(np.array(ca.DM(got_m)).ravel(), want_m, rtol=1e-8, atol=1e-9)


def _profiled(**changes):
    profile = dataclasses.replace(physics.resolve("research"), **changes)
    return profile.apply(catalog.build("8+", rate=28.0))


def test_fin_law_reflection_munk_and_roll_friction_are_in_the_torque_key():
    base = _match_key(_profiled(), 260.0, "sweep", "slip")
    assert base == _match_key(_profiled(), 260.0, "sweep", "slip")
    for change in ({"fin_law": "legacy"}, {"fin_reflection": 1.3},
                   {"munk_factor": 0.123}, {"roll_friction": "legacy"}):
        assert _match_key(_profiled(**change), 260.0, "sweep", "slip") != base, change


def test_bad_switches_are_refused():
    research = physics.resolve("research")
    for change in ({"fin_law": "guess"}, {"fin_reflection": 0.5}, {"munk_factor": 1.5}):
        with pytest.raises(ValueError):
            dataclasses.replace(research, **change)
