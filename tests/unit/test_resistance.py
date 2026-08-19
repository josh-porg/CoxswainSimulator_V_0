"""Unit tests for hull resistance.

Each of Formaggia et al.'s three longitudinal terms is checked against the
published formula, and the two legacy defects (natural log for ITTC, and a
kinematic viscosity ten times too large) are pinned with dedicated tests.
"""

import numpy as np
import pytest

from coxswain.hydro.hull import HullMesh, parametric_offsets
from coxswain.hydro.resistance import (
    FRESH_WATER,
    PAPER_LITERAL,
    SEA_WATER,
    ResistanceCoefficients,
    WaterProperties,
    friction_coefficient,
    hull_resistance,
)


@pytest.fixture
def properties():
    offsets = parametric_offsets(17.3, 0.57, 0.165, fullness=2.6)
    mesh = HullMesh(offsets)
    heave = mesh.equilibrium_heave(855.0, rho=FRESH_WATER.density)
    return mesh.submerged(np.array([0.0, 0.0, heave]), np.zeros(3),
                          rho=FRESH_WATER.density)


# --------------------------------------------------------------------------
# water properties
# --------------------------------------------------------------------------
def test_fresh_water_viscosity_is_physical():
    """The legacy value implied nu = 9.6e-6, ten times too large."""
    assert 0.9e-6 < FRESH_WATER.kinematic_viscosity < 1.5e-6


def test_fresh_water_is_less_dense_than_sea_water():
    assert FRESH_WATER.density < SEA_WATER.density


def test_water_properties_validate():
    with pytest.raises(ValueError, match="must be positive"):
        WaterProperties(density=-1.0, kinematic_viscosity=1e-6)


# --------------------------------------------------------------------------
# ITTC 1957 friction line
# --------------------------------------------------------------------------
def test_friction_coefficient_matches_the_ittc_formula():
    reynolds = 1.0e7
    expected = 0.075 / (np.log10(reynolds) - 2.0) ** 2
    assert friction_coefficient(reynolds) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("reynolds,expected", [
    (1.0e6, 0.075 / 16.0),      # log10 = 6
    (1.0e8, 0.075 / 36.0),      # log10 = 8
])
def test_friction_coefficient_at_round_reynolds_numbers(reynolds, expected):
    assert friction_coefficient(reynolds) == pytest.approx(expected, rel=1e-12)


def test_friction_uses_log10_not_natural_log():
    """The legacy bug: np.log understates C_f by about a factor of eight."""
    reynolds = 8.35e7
    natural = 0.075 / (np.log(reynolds) - 2.0) ** 2
    assert friction_coefficient(reynolds) > 6.0 * natural


def test_friction_coefficient_decreases_with_reynolds():
    values = [friction_coefficient(r) for r in (1e6, 1e7, 1e8, 1e9)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_friction_coefficient_is_clamped_at_low_reynolds():
    """log10(Re) -> 2 would divide by zero; the line is not valid there."""
    assert np.isfinite(friction_coefficient(1.0))
    assert friction_coefficient(1.0) == friction_coefficient(1.0e5)


def test_friction_coefficient_is_in_the_expected_band_for_a_shell():
    reynolds = 5.5 * 17.3 / FRESH_WATER.kinematic_viscosity
    assert 0.0018 < friction_coefficient(reynolds) < 0.0025


# --------------------------------------------------------------------------
# the three longitudinal terms
# --------------------------------------------------------------------------
def test_shape_term_matches_the_published_formula(properties):
    speed = 5.0
    _, detail = hull_resistance(np.array([speed, 0.0, 0.0]), properties, 17.3)
    expected = (0.5 * FRESH_WATER.density * speed ** 2
                * properties.transverse_area * 0.01)
    assert detail["shape"] == pytest.approx(expected, rel=1e-12)


def test_viscous_term_matches_the_published_formula(properties):
    speed = 5.0
    _, detail = hull_resistance(np.array([speed, 0.0, 0.0]), properties, 17.3)
    reynolds = speed * 17.3 / FRESH_WATER.kinematic_viscosity
    expected = (0.5 * FRESH_WATER.density * speed ** 2
                * properties.wetted_area * friction_coefficient(reynolds))
    assert detail["viscous"] == pytest.approx(expected, rel=1e-12)


def test_wave_term_uses_the_configured_reference_area(properties):
    speed = 5.0
    coefficients = ResistanceCoefficients(wave=0.001,
                                          wave_reference="transverse")
    _, detail = hull_resistance(np.array([speed, 0.0, 0.0]), properties, 17.3,
                                coefficients=coefficients)
    expected = (0.5 * FRESH_WATER.density * speed ** 2
                * properties.transverse_area * 0.001)
    assert detail["wave"] == pytest.approx(expected, rel=1e-12)


def test_reynolds_number_uses_the_mean_wetted_length(properties):
    speed = 4.0
    _, detail = hull_resistance(np.array([speed, 0.0, 0.0]), properties, 12.0)
    assert detail["reynolds"] == pytest.approx(
        speed * 12.0 / FRESH_WATER.kinematic_viscosity, rel=1e-12)


def test_total_is_the_sum_of_the_three_terms(properties):
    _, detail = hull_resistance(np.array([5.0, 0.0, 0.0]), properties, 17.3)
    assert detail["total_longitudinal"] == pytest.approx(
        detail["shape"] + detail["viscous"] + detail["wave"], rel=1e-12)


def test_viscous_drag_dominates_for_a_racing_shell(properties):
    _, detail = hull_resistance(np.array([5.5, 0.0, 0.0]), properties, 17.3)
    assert detail["viscous"] > detail["wave"] > detail["shape"]
    assert detail["viscous"] / detail["total_longitudinal"] > 0.55


# --------------------------------------------------------------------------
# magnitude sanity -- an eight at race pace
# --------------------------------------------------------------------------
def test_total_resistance_of_an_eight_matches_towing_data(properties):
    """~400-550 N at 5.5 m/s, i.e. 2.2-3.0 kW of hull power."""
    _, detail = hull_resistance(np.array([5.5, 0.0, 0.0]), properties, 17.3)
    assert 350.0 < detail["total_longitudinal"] < 600.0


def test_hull_power_is_consistent_with_crew_output(properties):
    speed = 5.5
    _, detail = hull_resistance(np.array([speed, 0.0, 0.0]), properties, 17.3)
    power = detail["total_longitudinal"] * speed
    assert 1800.0 < power < 3400.0


def test_the_papers_literal_wave_coefficient_is_not_physical(properties):
    """Documents why PAPER_LITERAL is not the default.

    C_dw = 0.02 on the waterplane area predicts more wave drag alone than
    an eight's entire measured resistance.
    """
    _, literal = hull_resistance(np.array([5.5, 0.0, 0.0]), properties, 17.3,
                                 coefficients=PAPER_LITERAL)
    _, default = hull_resistance(np.array([5.5, 0.0, 0.0]), properties, 17.3)
    assert literal["wave"] > 2000.0
    assert literal["total_longitudinal"] > 4.0 * default["total_longitudinal"]


# --------------------------------------------------------------------------
# direction and symmetry
# --------------------------------------------------------------------------
def test_resistance_opposes_forward_motion(properties):
    force, _ = hull_resistance(np.array([5.0, 0.0, 0.0]), properties, 17.3)
    assert force[0] < 0.0


def test_resistance_opposes_reverse_motion(properties):
    force, _ = hull_resistance(np.array([-5.0, 0.0, 0.0]), properties, 17.3)
    assert force[0] > 0.0


def test_resistance_vanishes_at_rest(properties):
    force, _ = hull_resistance(np.zeros(3), properties, 17.3)
    np.testing.assert_allclose(force, np.zeros(3), atol=1e-12)


def test_longitudinal_resistance_scales_close_to_the_square_of_speed(
        properties):
    _, slow = hull_resistance(np.array([3.0, 0.0, 0.0]), properties, 17.3)
    _, fast = hull_resistance(np.array([6.0, 0.0, 0.0]), properties, 17.3)
    ratio = fast["total_longitudinal"] / slow["total_longitudinal"]
    # slightly below 4 because C_f falls with Reynolds number
    assert 3.5 < ratio < 4.0


def test_lateral_resistance_opposes_sideslip(properties):
    port, _ = hull_resistance(np.array([5.0, 0.5, 0.0]), properties, 17.3)
    starboard, _ = hull_resistance(np.array([5.0, -0.5, 0.0]), properties,
                                   17.3)
    assert port[1] < 0.0 < starboard[1]
    assert port[1] == pytest.approx(-starboard[1], rel=1e-12)


def test_vertical_resistance_opposes_heave_velocity(properties):
    down, _ = hull_resistance(np.array([5.0, 0.0, -0.3]), properties, 17.3)
    assert down[2] > 0.0


def test_cross_flow_terms_are_quadratic(properties):
    small, _ = hull_resistance(np.array([5.0, 0.2, 0.0]), properties, 17.3)
    large, _ = hull_resistance(np.array([5.0, 0.4, 0.0]), properties, 17.3)
    assert large[1] == pytest.approx(4.0 * small[1], rel=1e-12)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def test_coefficients_reject_an_unknown_wave_reference():
    with pytest.raises(ValueError, match="wave_reference"):
        ResistanceCoefficients(wave_reference="banana")


def test_default_coefficients_use_the_published_shape_and_friction_values():
    coefficients = ResistanceCoefficients()
    assert coefficients.shape == pytest.approx(0.01)
    assert coefficients.friction_zero == pytest.approx(0.075)


def test_form_factor_scales_only_the_viscous_term(properties):
    plain = ResistanceCoefficients()
    formed = ResistanceCoefficients(form_factor=1.2)
    _, a = hull_resistance(np.array([5.0, 0.0, 0.0]), properties, 17.3,
                           coefficients=plain)
    _, b = hull_resistance(np.array([5.0, 0.0, 0.0]), properties, 17.3,
                           coefficients=formed)
    assert b["viscous"] == pytest.approx(1.2 * a["viscous"], rel=1e-12)
    assert b["wave"] == pytest.approx(a["wave"], rel=1e-12)


def test_sea_water_gives_more_resistance_than_fresh(properties):
    _, fresh = hull_resistance(np.array([5.0, 0.0, 0.0]), properties, 17.3,
                               water=FRESH_WATER)
    _, salt = hull_resistance(np.array([5.0, 0.0, 0.0]), properties, 17.3,
                              water=SEA_WATER)
    assert salt["total_longitudinal"] > fresh["total_longitudinal"]
