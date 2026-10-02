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


# -- Sretenskii (finite depth) against Doctors, Day & Clelland (2010) -------------------------

@pytest.mark.parametrize("d_over_l", [0.7767, 0.25])
def test_finite_depth_michell_reproduces_doctors_steady_wigley(d_over_l):
    """Their Wigley model at Fr 0.3 in their tank's depth (depth Froude 0.34 and 0.60): the
    steady wave resistance their linear theory prints, within the reading (+-0.1e-3 of 5.15e-3)."""
    from coxswain.hydro.finite_depth_michell import FiniteDepthMichell
    from coxswain.hydro.michell import wigley_offsets
    row = {float(r["d_over_L"]): r for r in _rows("doctors2010_steady_wigley.csv")}[d_over_l]
    L, B, T = 3.0, 0.3, 0.1875
    x, z, half = wigley_offsets(L, B, T, stations=161, levels=41)
    model = FiniteDepthMichell(station=x, level=z, half_beam=half, depth=d_over_l * L,
                               density=1000.0, quadrature="trapezoid")
    speed = 0.3 * np.sqrt(9.81 * L)
    weight = 1000.0 * 9.81 * (4.0 / 9.0) * L * B * T
    assert model.resistance([speed])[0] / weight == pytest.approx(float(row["theory_rw_over_w"]), abs=0.12e-3)


# -- [K05] Fig. 1: the digitised on-water patterns reproduce [K05]'s own table ---------------

def test_k05_fig1_digitisation_reproduces_their_segment_travels():
    """Travel = integral of v_segment / v_handle over the 1.59 m drive: the curves, digitised,
    give [K05]'s tabulated legs 0.51, trunk 0.48, arms 0.62 m -- never fitted to them."""
    rows = _rows("k05_fig1_onwater.csv")
    pct = np.array([float(r["length_pct"]) for r in rows])
    h = np.array([float(r["handle_speed"]) for r in rows])
    path = pct / 100.0 * 1.59
    for column, printed, tol in (("legs_velocity", 0.51, 0.03), ("trunk_velocity", 0.48, 0.04),
                                 ("arms_velocity", 0.62, 0.03)):
        v = np.array([float(r[column]) for r in rows])
        assert np.trapezoid(v / h, path) == pytest.approx(printed, abs=tol)


def test_k05_fig1_peak_handle_speed_is_their_table():
    rows = _rows("k05_fig1_onwater.csv")
    h = np.array([float(r["handle_speed"]) for r in rows])
    pct = np.array([float(r["length_pct"]) for r in rows])
    assert h.max() == pytest.approx(2.36, abs=0.06)                     # racing rate, table row 12
    assert pct[np.argmax(h)] == pytest.approx(65.2, abs=6.0)             # table row 14


def test_k05_fig1_handle_force_is_their_table():
    """The on-water force panel, digitised the same way: max 602 N at 34.7% of the drive length
    and average / max 56.9% over the drive's time (table rows 6, 8, 9), never fitted to them."""
    rows = _rows("k05_fig1_onwater.csv")
    pct = np.array([float(r["length_pct"]) for r in rows])
    h = np.array([float(r["handle_speed"]) for r in rows])
    f = np.array([float(r["handle_force_N"]) for r in rows])
    assert f.max() == pytest.approx(602.0, rel=0.03)
    assert pct[np.argmax(f)] == pytest.approx(34.7, abs=2.0)
    ok = h > 0.2                                                         # dt = ds / v, ends unresolved
    dt = np.where(ok, 1.0 / np.where(ok, h, 1.0), 0.0)
    assert np.sum(f * dt) / np.sum(dt) / f.max() == pytest.approx(0.569, abs=0.025)


def test_k05_fig1_recovery_returns_the_segments_by_their_table_travel():
    """The recovery branch, digitised separately: legs and trunk return by [K05]'s 0.51 and 0.48 m.
    (The arms come back 0.56 m against 0.62: their return is at the finish, where the handle's
    turning point is unresolved; the body uses legs and trunk only.)"""
    rows = _rows("k05_fig1_recovery.csv")
    pct = np.array([float(r["length_pct"]) for r in rows])
    h = np.array([float(r["handle_speed"]) for r in rows])
    ok = h < -0.2
    path = pct / 100.0 * 1.59
    for column, printed in (("legs_velocity", 0.51), ("trunk_velocity", 0.48)):
        v = np.array([float(r[column]) for r in rows])
        travel = np.trapezoid(np.where(ok, v / np.where(ok, h, -1.0), 0.0), path)
        assert travel == pytest.approx(printed, abs=0.03)


def test_k05_fig1_boat_acceleration_extremes_are_their_table():
    rows = _rows("k05_fig1_boat_acceleration.csv")
    a = np.array([float(r["acceleration_mps2"]) for r in rows])
    assert a.min() == pytest.approx(-7.92, abs=0.3)                      # table row 15, racing
    assert a.max() == pytest.approx(3.39, abs=0.3)                       # table row 16
