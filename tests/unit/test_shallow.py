"""Unit tests for the shallow-water wave-resistance correction.

Checked against the regimes and numbers Day et al. (2011) state -- see
``docs/SOURCES.md`` section 6.
"""

import numpy as np
import pytest

from coxswain.hydro.shallow import (
    DEEP_WATER_FROUDE,
    DEFAULT_MAX_AMPLIFICATION,
    ShallowWaterModel,
    critical_speed,
    depth_froude,
    matched_deep_water_speed,
    wave_resistance_factor,
)


# --------------------------------------------------------------------------
# depth Froude number
# --------------------------------------------------------------------------
def test_depth_froude_definition():
    """Fr_h = U / sqrt(g h)."""
    assert depth_froude(5.0, 4.0) == pytest.approx(5.0 / np.sqrt(9.81 * 4.0))


def test_depth_froude_is_zero_in_infinite_depth():
    assert depth_froude(5.0, np.inf) == pytest.approx(0.0)


def test_depth_froude_uses_speed_magnitude():
    assert depth_froude(-5.0, 4.0) == pytest.approx(depth_froude(5.0, 4.0))


def test_depth_froude_vectorises():
    froude = depth_froude(5.0, np.array([2.0, 4.0, 8.0]))
    assert froude.shape == (3,)
    assert froude[0] > froude[1] > froude[2]


def test_critical_speed_matches_the_published_value():
    """Day et al.: "on a rowing lake with depth of 3.0 m, the critical
    speed is around 5.4 m/s"."""
    assert critical_speed(3.0) == pytest.approx(5.4, abs=0.05)


def test_critical_speed_is_where_the_froude_number_is_one():
    for depth in (1.5, 3.0, 7.0):
        assert depth_froude(critical_speed(depth), depth) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Schlichting's matched speed
# --------------------------------------------------------------------------
def test_matched_deep_water_speed_satisfies_its_defining_equation():
    speed, depth = np.array(4.0), np.array(5.0)
    matched = matched_deep_water_speed(speed, depth)
    recovered = matched * np.sqrt(np.tanh(9.81 * depth / matched ** 2))
    assert recovered == pytest.approx(float(speed), rel=1e-6)


def test_matched_speed_always_exceeds_the_shallow_speed():
    """Shallow water makes the waves of a faster deep-water hull."""
    for depth in (3.0, 5.0, 12.0):
        speed = np.array(4.0)
        assert matched_deep_water_speed(speed, np.array(depth)) >= 4.0


def test_matched_speed_approaches_the_actual_speed_in_deep_water():
    matched = matched_deep_water_speed(np.array(4.0), np.array(500.0))
    assert matched == pytest.approx(4.0, rel=1e-3)


def test_matched_speed_grows_as_depth_shrinks():
    speeds = [float(matched_deep_water_speed(np.array(4.5), np.array(h)))
              for h in (20.0, 10.0, 6.0, 4.0)]
    assert speeds == sorted(speeds)


# --------------------------------------------------------------------------
# the amplification factor
# --------------------------------------------------------------------------
def test_factor_is_unity_in_deep_water():
    assert wave_resistance_factor(5.0, 1000.0) == pytest.approx(1.0)


@pytest.mark.parametrize("froude", [0.1, 0.25, 0.4, 0.49])
def test_factor_is_unity_below_the_deep_water_threshold(froude):
    """Day et al.: "if Fr_h <= 0.5, results are similar to deep water"."""
    depth = 4.0
    speed = froude * np.sqrt(9.81 * depth)
    assert wave_resistance_factor(speed, depth) == pytest.approx(1.0)


def test_factor_exceeds_unity_approaching_critical():
    depth = 3.0
    speed = 0.85 * np.sqrt(9.81 * depth)
    assert wave_resistance_factor(speed, depth) > 1.05


def test_factor_increases_monotonically_up_to_critical():
    depth = 3.0
    froudes = np.linspace(0.5, 1.0, 25)
    factors = [float(wave_resistance_factor(f * np.sqrt(9.81 * depth), depth))
               for f in froudes]
    assert np.all(np.diff(factors) >= -1e-9), "factor must not dip below critical"


def test_factor_peaks_near_critical():
    depth = 3.0
    froudes = np.linspace(0.3, 1.6, 60)
    factors = np.array([float(wave_resistance_factor(
        f * np.sqrt(9.81 * depth), depth)) for f in froudes])
    peak_froude = froudes[int(np.argmax(factors))]
    assert 0.95 <= peak_froude <= 1.15


def test_factor_relaxes_supercritically():
    """Day et al.: above critical the transverse wave system disappears."""
    depth = 3.0
    critical = np.sqrt(9.81 * depth)
    at_critical = float(wave_resistance_factor(critical, depth))
    well_above = float(wave_resistance_factor(1.8 * critical, depth))
    assert well_above < at_critical


def test_factor_never_exceeds_the_cap():
    depth = 3.0
    for froude in np.linspace(0.1, 3.0, 80):
        factor = float(wave_resistance_factor(froude * np.sqrt(9.81 * depth),
                                              depth))
        assert 1.0 <= factor <= DEFAULT_MAX_AMPLIFICATION + 1e-9


def test_factor_respects_a_custom_cap():
    depth = 3.0
    factor = float(wave_resistance_factor(np.sqrt(9.81 * depth), depth,
                                          max_amplification=1.5))
    assert factor == pytest.approx(1.5, abs=1e-6)


def test_factor_depends_only_on_the_depth_froude_number():
    """The correction is scale free, so equal Fr_h must give equal factors."""
    for froude in (0.6, 0.75, 0.88):
        a = float(wave_resistance_factor(froude * np.sqrt(9.81 * 2.0), 2.0))
        b = float(wave_resistance_factor(froude * np.sqrt(9.81 * 8.0), 8.0))
        assert a == pytest.approx(b, rel=1e-6)


def test_factor_is_continuous_across_the_regime_boundaries():
    depth = 3.0
    froudes = np.linspace(0.4, 1.7, 400)
    factors = np.array([float(wave_resistance_factor(
        f * np.sqrt(9.81 * depth), depth)) for f in froudes])
    assert np.abs(np.diff(factors)).max() < 0.1, "no step at a handover"


def test_a_shell_on_a_three_metre_course_sweeps_through_critical():
    """Day et al.: at 3.0 m a pair's Fr_h runs 0.65 to 1.09 within a stroke."""
    assert depth_froude(3.5, 3.0) == pytest.approx(0.65, abs=0.02)
    assert depth_froude(5.9, 3.0) == pytest.approx(1.09, abs=0.02)


# --------------------------------------------------------------------------
# the configuration object
# --------------------------------------------------------------------------
def test_default_model_is_deep_water_and_inert():
    model = ShallowWaterModel()
    assert not model.enabled
    assert float(model.factor(5.5)) == pytest.approx(1.0)


def test_model_reports_its_critical_speed():
    assert ShallowWaterModel(depth=3.0).critical_speed == pytest.approx(
        5.42, abs=0.02)


def test_model_factor_matches_the_free_function():
    model = ShallowWaterModel(depth=3.0)
    assert float(model.factor(4.8)) == pytest.approx(
        float(wave_resistance_factor(4.8, 3.0)))


@pytest.mark.parametrize("kwargs,match", [
    ({"depth": 0.0}, "depth must be positive"),
    ({"depth": -2.0}, "depth must be positive"),
    ({"max_amplification": 0.5}, "at least 1"),
    ({"subcritical_limit": 1.2}, "must lie in"),
    ({"subcritical_limit": 0.3}, "must lie in"),
    ({"supercritical_relax": 0.9}, "must exceed 1"),
])
def test_invalid_configuration_is_rejected(kwargs, match):
    base = {"depth": 3.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ShallowWaterModel(**base)


# --------------------------------------------------------------------------
# integration with the resistance model
# --------------------------------------------------------------------------
def test_resistance_reports_the_depth_terms(eight):
    from coxswain.hydro.resistance import hull_resistance

    heave = eight.mesh.equilibrium_heave(eight.total_mass,
                                         rho=eight.water.density)
    props = eight.mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3),
                                 rho=eight.water.density)
    _, detail = hull_resistance(np.array([5.3, 0.0, 0.0]), props,
                                eight.length, eight.water, eight.resistance,
                                ShallowWaterModel(depth=3.0))
    assert detail["depth_froude"] == pytest.approx(0.977, abs=0.01)
    assert detail["depth_factor"] > 1.5


def test_shallow_water_raises_only_the_wave_term(eight):
    from coxswain.hydro.resistance import hull_resistance

    heave = eight.mesh.equilibrium_heave(eight.total_mass,
                                         rho=eight.water.density)
    props = eight.mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3),
                                 rho=eight.water.density)
    velocity = np.array([5.3, 0.0, 0.0])

    _, deep = hull_resistance(velocity, props, eight.length, eight.water,
                              eight.resistance)
    _, shallow = hull_resistance(velocity, props, eight.length, eight.water,
                                 eight.resistance,
                                 ShallowWaterModel(depth=3.0))

    assert shallow["shape"] == pytest.approx(deep["shape"])
    assert shallow["viscous"] == pytest.approx(deep["viscous"])
    assert shallow["wave"] > deep["wave"]
    assert shallow["total_longitudinal"] > deep["total_longitudinal"]


def test_deep_water_default_leaves_resistance_unchanged(eight):
    from coxswain.hydro.resistance import hull_resistance

    heave = eight.mesh.equilibrium_heave(eight.total_mass,
                                         rho=eight.water.density)
    props = eight.mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3),
                                 rho=eight.water.density)
    velocity = np.array([5.3, 0.0, 0.0])

    _, implicit = hull_resistance(velocity, props, eight.length, eight.water,
                                  eight.resistance)
    _, explicit = hull_resistance(velocity, props, eight.length, eight.water,
                                  eight.resistance, ShallowWaterModel())
    assert implicit["total_longitudinal"] == pytest.approx(
        explicit["total_longitudinal"])
    assert implicit["depth_factor"] == pytest.approx(1.0)
