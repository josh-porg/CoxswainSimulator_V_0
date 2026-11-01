"""The joint muscles and lower control layers against their sources' own statements (SOURCES §207-208)."""
from __future__ import annotations

import numpy as np
import pytest

from coxswain.crew import joint_muscles as J


def test_anderson_velocity_constants_mean_what_the_paper_says():
    """C4 and C5 are the speeds where torque is 75 % and 50 % of isometric; C5/C4 is the paper's per-exertion ratio."""
    ratios = {"HE": 2.020, "HF": 2.037, "KE": 2.606, "KF": 2.605, "PF": 3.602, "DF": 2.777}
    for ex, r in ratios.items():
        jt = J.anderson(ex, "M18-25", 72.8, 1.748, sides=1)
        assert jt.C5 / jt.C4 == pytest.approx(r, abs=0.01)
        assert jt.velocity_factor(0.0) == pytest.approx(1.0)
        assert jt.velocity_factor(jt.C4) == pytest.approx(0.75, abs=1e-9)
        assert jt.velocity_factor(jt.C5) == pytest.approx(0.50, abs=1e-9)


def test_anderson_peak_at_c3_and_scaled_by_weight_times_height():
    jt = J.anderson("KE", "M18-25", 72.8, 1.748, sides=1)
    assert jt.C1 == pytest.approx(0.163 * 72.8 * 9.81 * 1.748)        # ~203 N.m per leg, the isometric peak
    th = np.linspace(0.0, 1.8, 181)
    assert th[np.argmax(jt.max_active(th, 0.0))] == pytest.approx(jt.C3, abs=0.01)
    both = J.anderson("KE", "M18-25", 72.8, 1.748)
    assert both.max_active(1.0, 0.0) == pytest.approx(2 * jt.max_active(1.0, 0.0))


def test_eccentric_is_concentric_scaled_by_c6():
    jt = J.anderson("HE", "M18-25", 72.8, 1.748, sides=1)
    assert jt.velocity_factor(-1.0) == pytest.approx(jt.velocity_factor(1.0) * (1 + jt.C6))
    assert jt.velocity_factor(-1.0) > 1.0                                 # HE: C6 above the concentric slope


def test_damping_is_the_force_velocity_slope():
    jt = J.anderson("HE", "M18-25", 72.8, 1.748)
    b = jt.damping(0.5, 1.0, 1.0)
    num = -0.5 * (jt.max_active(1.0, 1.01) - jt.max_active(1.0, 0.99)) / 0.02
    assert b == pytest.approx(num, rel=1e-3) and b > 0


def test_activation_needed_counts_passive_torque_and_flags_infeasible():
    jt = J.anderson("HE", "M18-25", 72.8, 1.748)
    cap = jt.max_active(1.0, 2.0)
    assert jt.activation_needed(cap + jt.passive(1.0), 1.0, 2.0) == pytest.approx(1.0)
    off = J.anderson("HE", "M18-25", 72.8, 1.748, passive=False)
    assert off.passive(1.0) == 0.0
    assert off.activation_needed(2 * cap, 1.0, 2.0) == pytest.approx(2.0)
    # the hip's passive term was measured with the knee nearly straight: large at 1 rad (why rowing drops it)
    assert J.anderson("HE", "M18-25", 72.8, 1.748, sides=1).passive(1.0) > 150.0


def test_elbow_table_reproduces_cells_and_fills_unreported_speeds():
    el = J.elbow("flexion", "M", sides=1)
    assert el.max_active(np.radians(65.0), 0.0) == pytest.approx(63.0)
    assert el.max_active(np.radians(90.0), np.radians(120.0)) == pytest.approx(36.8)
    fast = el.max_active(np.radians(15.0), np.radians(300.0))            # unreported cell: extrapolated, not NaN
    assert np.isfinite(fast) and 0.0 <= fast < 32.1


def test_activation_dynamics_time_constants():
    ad = J.ActivationDynamics.of("zajac")
    assert ad.rate(1.0, 0.0) == pytest.approx(1.0 / 0.010)
    assert ad.rate(0.01, 1.0) == pytest.approx((0.01 - 1.0) / 0.040)
    assert J.ActivationDynamics.of("thelen").tau_deact == 0.050


def test_trunk_impedance_ratios_to_gravity():
    imp = J.TrunkImpedance(gravity_stiffness=100.0)
    assert imp.intrinsic_stiffness(0.20) == pytest.approx(1280 / 1695 * 100)
    assert imp.intrinsic_stiffness(0.35) == pytest.approx(1738 / 1695 * 100)
    assert imp.intrinsic_stiffness(0.90) == pytest.approx(2116 / 1695 * 100)   # held flat beyond 50 % (flagged)
    assert imp.delay == 0.040
    z, wn = imp.reflex_filter(0.35)
    assert z == pytest.approx(0.55) and wn == pytest.approx(2 * np.pi * 9.54)


def test_reflex_responds_to_stretch_only_after_its_delay_with_unit_dc_gain():
    imp = J.TrunkImpedance(gravity_stiffness=100.0)
    rf = J.ReflexState(imp)
    t = np.arange(0.0, 0.6, 1e-3)
    for x in t:
        rf.record(x, 1.0 if x >= 0.1 else 0.0)                          # a stretch velocity step at 0.1 s
    outs = [rf.output(x, 0.35) for x in t]
    k = np.searchsorted(t, 0.1 + imp.delay)
    assert max(abs(o) for o in outs[:k]) < 1e-12                          # nothing before the delay
    assert outs[-1] == pytest.approx(imp.reflex_gain(0.35), rel=1e-2)     # settles at G_R x velocity
    rf2 = J.ReflexState(imp)
    for x in t:
        rf2.record(x, -1.0)                                               # shortening: rectified away
    assert abs(rf2.output(0.6, 0.35)) < 1e-12
