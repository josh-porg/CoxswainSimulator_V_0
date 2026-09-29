"""The blade's sources, reproduced: each test checks a number the source itself prints or plots.

Data files are in data/literature, each digitised from the source with its reading accuracy
stated in the file's header.
"""
import csv
import os

import numpy as np
import pytest

from coxswain.crew.blade_depth import grift_curve
from coxswain.crew.liftdrag import LiftDragBlade

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIT = os.path.join(ROOT, "data", "literature")


def _rows(name):
    with open(os.path.join(LIT, name), encoding="utf-8") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


# -- [CG07], via Coppel's replot: the tier 2 blade against the measured Big Blade -------------

def test_tier_two_reproduces_caplan_and_gardners_big_blade_through_the_drive_range():
    """0-90 deg, the range the model's attack angle spans: rms 0.07 in C_L and 0.10 in C_D,
    about the spread between Coppel's own CFD models of the same blade."""
    blade = LiftDragBlade.big_blade(outboard=1.8, area=0.083)
    lift_err, drag_err = [], []
    for r in _rows("cg07_bigblade_via_coppel2010.csv"):
        deg = float(r["angle_deg"])
        if deg > 90:
            continue
        cl, cd = blade.lift_drag(np.radians(deg))
        lift_err.append(float(cl) - float(r["cl"]))
        if r["cd"]:
            drag_err.append(float(cd) - float(r["cd"]))
    assert np.sqrt(np.mean(np.square(lift_err))) < 0.08
    assert np.sqrt(np.mean(np.square(drag_err))) < 0.12
    assert max(abs(e) for e in lift_err + drag_err) < 0.2


def test_the_amplitudes_are_the_measured_peaks():
    rows = {float(r["angle_deg"]): r for r in _rows("cg07_bigblade_via_coppel2010.csv")}
    assert float(rows[90.0]["cd"]) == pytest.approx(2.07, abs=0.03)     # A_d
    assert float(rows[45.0]["cl"]) == pytest.approx(1.25, abs=0.03)     # A_l


# -- [CO10] section 3.5: the full-size correction ------------------------------------------

def test_the_digitised_curves_reproduce_coppels_table_3_7():
    for r in _rows("coppel2010_fullsize_vs_quarterscale.csv"):
        assert abs(float(r["cl_quarter"]) - float(r["cl_full"])) == pytest.approx(
            float(r["table37_dcl"]), abs=0.03)
        assert abs(float(r["cd_quarter"]) - float(r["cd_full"])) == pytest.approx(
            float(r["table37_dcd"]), abs=0.03)


def test_the_35_percent_is_relative_to_the_full_size_value():
    r = {float(x["angle_deg"]): x for x in _rows("coppel2010_fullsize_vs_quarterscale.csv")}[90.0]
    assert float(r["table37_dcd"]) / float(r["cd_full"]) == pytest.approx(0.35, abs=0.015)   # +-0.02 reading


def test_the_model_uses_the_digitised_ratios():
    angles, lift_ratio, drag_ratio = LiftDragBlade.FULL_SCALE["coppel"]
    rows = {float(x["angle_deg"]): x for x in _rows("coppel2010_fullsize_vs_quarterscale.csv")}
    for deg, lr, dr in zip(angles, lift_ratio, drag_ratio):
        r = rows[deg]
        assert dr == pytest.approx(float(r["cd_full"]) / float(r["cd_quarter"]), rel=1e-9)
        if float(r["cl_quarter"]):
            assert lr == pytest.approx(float(r["cl_full"]) / float(r["cl_quarter"]), rel=1e-9)


# -- [G19] / [G20]: the numbers Grift prints ------------------------------------------------

def test_grifts_printed_depth_numbers():
    """Abstract: drag at 1/5 plate height of cover is 45% above the top edge at the surface;
    the figure's reference lines are the fence (1.10) and the deep AR-2 plate (1.30)."""
    depth, cd = grift_curve()
    at = dict(zip(np.round(depth, 3), cd))
    assert at[0.2] / at[0.0] == pytest.approx(1.45, abs=0.03)
    assert at[0.0] == pytest.approx(1.10, abs=0.03)
    assert cd[-1] == pytest.approx(1.30, abs=0.03)


# -- Patton (1965), as [G19]/[G20] quote it, and [LB19] ------------------------------------

def test_patton_reproduces_grifts_printed_value_for_his_plate():
    """[G20] eq. (2.4): 0.84 (pi rho / 4) l_a l_b^2 = 1.3 kg for his 0.2 x 0.1 m plate."""
    from coxswain.crew.blade_added_mass import patton_added_mass
    assert patton_added_mass(0.2, 0.1, 1000.0) == pytest.approx(1.3, abs=0.05)


def test_labbe_is_their_eq_3b_on_their_own_blade():
    """[LB19] print no added-mass value to compare with, only the definition: rho C_m Omega,
    Omega = pi S l_b / 4, S = l_b h_b, on their 7.0 x 4.7 cm blade."""
    from coxswain.crew.blade_added_mass import labbe_added_mass
    S = 0.070 * 0.047
    assert labbe_added_mass(0.070, 0.047, 1000.0) == pytest.approx(1000.0 * 0.7 * np.pi * S * 0.070 / 4)
