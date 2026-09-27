"""Blade depth through the drive (sprint 1 #2): geometry, the Grift curve, and the
dynamic oar hook, which must leave a depth-blind run exactly as it was."""
import numpy as np
import pytest

from coxswain.crew.blade_depth import BladeDepth, grift_curve


def _flat(v_deg, reference="deep", **kw):
    return BladeDepth.from_samples([0.0, 1.0], [v_deg, v_deg], lever=1.8, width=0.2,
                                   reference=reference, **kw)


def test_grift_curve_is_the_digitised_figure():
    depth, cd = grift_curve()
    assert cd[np.argmin(np.abs(depth - 0.0))] == pytest.approx(1.10)
    assert cd.max() == pytest.approx(1.60)
    assert depth[np.argmax(cd)] == pytest.approx(0.20)
    assert cd[-1] == pytest.approx(1.30)


def test_a_blade_clear_of_the_water_carries_nothing():
    up = _flat(10.0)                                   # centre 0.31 m up: fully out
    assert up.wetted(0.5) == 0.0
    assert up.factor(0.5) == 0.0


def test_a_deep_blade_matches_the_deep_plate():
    deep = _flat(-20.0)                                # centre 0.62 m down
    assert deep.wetted(0.5) == 1.0
    assert deep.factor(0.5) == pytest.approx(1.0)       # 1.30 / deep reference 1.30


def test_the_optimum_is_a_fifth_of_a_blade_down():
    # choose the vertical angle that puts the top edge 0.2 widths below the surface
    width, lever = 0.2, 1.8
    centre = -(0.2 * width + 0.5 * width)
    v = np.degrees(np.arcsin(centre / lever))
    best = _flat(v)
    assert best.cover(0.5) == pytest.approx(0.2 * width)
    assert best.factor(0.5) == pytest.approx(1.60 / 1.30)


def test_the_zero_offset_moves_the_blade():
    assert _flat(0.0, zero_offset=0.05).centre_height(0.5) == pytest.approx(0.05)


def test_mean_reference_keeps_the_fitted_level():
    prof = BladeDepth.from_samples([0.0, 0.3, 0.7, 1.0], [2.0, -6.0, -6.0, 1.0],
                                   lever=1.8, width=0.2, reference="mean")
    u = np.linspace(0, 1, 201)
    f = prof.factor(u)
    assert np.mean(f[f > 0]) == pytest.approx(1.0, rel=1e-9)


def test_the_dynamic_oar_is_unchanged_without_a_depth_profile():
    from coxswain.boats import catalog
    from coxswain.sim.dynamic_oar import DynamicOarSimulator

    boat = catalog.single_scull(rate=30.0)
    sim = DynamicOarSimulator(boat, peak_torque=100.0)
    assert sim.blade_depth is None
    y = sim.augmented_initial_state(4.0)
    from coxswain.core.state import STATE_SIZE, State
    lock = boat.rig.seats[sim._seats[0]].oarlocks[0]
    state = State.from_vector(y[:STATE_SIZE])
    base = sim._blade_loads(0, np.radians(10.0), -2.0, state, lock)
    sim.blade_depth = _flat(-20.0)                      # deep: factor exactly 1
    assert sim._blade_loads(0, np.radians(10.0), -2.0, state, lock) == pytest.approx(base)
    sim.blade_depth = _flat(10.0)                       # out of the water: no load
    assert sim._blade_loads(0, np.radians(10.0), -2.0, state, lock) == (0.0, 0.0)


def test_a_depth_profile_reaches_the_hull_and_the_oar():
    """The fast slip-law paths must not bypass the depth factor: an out-of-water profile
    leaves the hull unloaded and the oar with no blade torque."""
    from coxswain.boats import catalog
    from coxswain.core.state import STATE_SIZE, State
    from coxswain.sim.dynamic_oar import DynamicOarSimulator

    boat = catalog.single_scull(rate=30.0)
    sim = DynamicOarSimulator(boat, peak_torque=0.0)
    y = sim.augmented_initial_state(4.0)
    n = sim.n_oar_states
    angles = np.full(n, np.radians(10.0))
    rates = np.full(n, -2.0)
    sim._oar_state = (angles, rates)
    state = State.from_vector(y[:STATE_SIZE])
    wet, _ = sim._oar_loads(0.1, state)
    sim.blade_depth = _flat(10.0)
    dry, _ = sim._oar_loads(0.1, state)
    sim._oar_state = None
    assert np.linalg.norm(wet) > 1.0
    assert np.allclose(dry, 0.0)
    _, acc_dry = sim._oar_rates(0.1, y[:STATE_SIZE], angles, rates)
    sim.blade_depth = None
    _, acc_wet = sim._oar_rates(0.1, y[:STATE_SIZE], angles, rates)
    assert not np.allclose(acc_dry, acc_wet)
