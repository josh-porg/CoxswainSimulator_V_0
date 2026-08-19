"""Unit tests for the 2-jet automatic differentiation used by the kinematics.

Every rule is checked against central finite differences of the same
expression, so a wrong propagation rule cannot pass.
"""

import numpy as np
import pytest

from coxswain.core import taylor
from coxswain.core.taylor import Jet2, constant, variable

STEP = 1e-5


def numerical_derivatives(func, t):
    """Central-difference value, first and second derivative of ``func``."""
    f0 = func(t)
    fp, fm = func(t + STEP), func(t - STEP)
    return f0, (fp - fm) / (2 * STEP), (fp - 2 * f0 + fm) / STEP ** 2


def check_against_finite_difference(jet_func, plain_func, t, atol=1e-6):
    jet = jet_func(variable(t))
    value, first, second = numerical_derivatives(plain_func, t)
    assert jet.value == pytest.approx(value, abs=atol)
    assert jet.first == pytest.approx(first, abs=atol)
    assert jet.second == pytest.approx(second, abs=1e-3)


# --------------------------------------------------------------------------
# constructors
# --------------------------------------------------------------------------
def test_variable_has_unit_first_derivative():
    v = variable(2.0)
    assert v.value == pytest.approx(2.0)
    assert v.first == pytest.approx(1.0)
    assert v.second == pytest.approx(0.0)


def test_constant_has_zero_derivatives():
    c = constant(3.5)
    assert c.value == pytest.approx(3.5)
    assert c.first == pytest.approx(0.0)
    assert c.second == pytest.approx(0.0)


# --------------------------------------------------------------------------
# arithmetic
# --------------------------------------------------------------------------
def test_addition():
    check_against_finite_difference(lambda t: t + 3.0, lambda t: t + 3.0, 1.3)


def test_subtraction_both_directions():
    check_against_finite_difference(lambda t: t - 2.0, lambda t: t - 2.0, 1.3)
    check_against_finite_difference(lambda t: 2.0 - t, lambda t: 2.0 - t, 1.3)


def test_product_rule():
    check_against_finite_difference(lambda t: t * t.sin(),
                                    lambda t: t * np.sin(t), 1.3)


def test_quotient_rule():
    check_against_finite_difference(lambda t: t.sin() / (t + 2.0),
                                    lambda t: np.sin(t) / (t + 2.0), 1.3)


def test_reciprocal():
    check_against_finite_difference(lambda t: 1.0 / (t + 2.0),
                                    lambda t: 1.0 / (t + 2.0), 1.3)


@pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0, 3.0, -1.0, -2.5])
def test_power_rule(exponent):
    check_against_finite_difference(lambda t: (t + 3.0) ** exponent,
                                    lambda t: (t + 3.0) ** exponent, 1.3)


def test_negation():
    jet = -variable(1.3)
    assert jet.value == pytest.approx(-1.3)
    assert jet.first == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# elementary functions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t", [-1.7, -0.4, 0.0, 0.9, 2.6])
def test_sin(t):
    check_against_finite_difference(lambda j: j.sin(), np.sin, t)


@pytest.mark.parametrize("t", [-1.7, -0.4, 0.0, 0.9, 2.6])
def test_cos(t):
    check_against_finite_difference(lambda j: j.cos(), np.cos, t)


def test_sqrt():
    check_against_finite_difference(lambda t: (t + 4.0).sqrt(),
                                    lambda t: np.sqrt(t + 4.0), 1.3)


@pytest.mark.parametrize("t", [-1.2, 0.0, 0.7, 2.0])
def test_tanh(t):
    check_against_finite_difference(lambda j: j.tanh(), np.tanh, t)


def test_exp():
    check_against_finite_difference(lambda j: j.exp(), np.exp, 0.8)


# --------------------------------------------------------------------------
# composition -- the case the legacy code hand-differentiated
# --------------------------------------------------------------------------
def test_tanh_of_cos_composition():
    """amplitude * tanh(k cos(w t) + c) -- the legacy slide model."""
    amplitude, k, c, omega = 0.4, 1.5, -0.75, 3.9

    def jet_form(t):
        return amplitude * ((omega * t).cos() * k + c).tanh()

    def plain_form(t):
        return amplitude * np.tanh(k * np.cos(omega * t) + c)

    for t in (0.05, 0.31, 0.77, 1.4):
        check_against_finite_difference(jet_form, plain_form, t)


def test_deeply_nested_composition():
    def jet_form(t):
        return ((t.sin() * 2.0 + 1.5) ** 1.5 / (t.cos() + 3.0)).tanh()

    def plain_form(t):
        return np.tanh((2 * np.sin(t) + 1.5) ** 1.5 / (np.cos(t) + 3.0))

    check_against_finite_difference(jet_form, plain_form, 0.6)


# --------------------------------------------------------------------------
# vectorisation
# --------------------------------------------------------------------------
def test_jets_broadcast_over_arrays():
    t = np.array([0.2, 0.9, 1.6])
    jet = variable(t).sin()
    np.testing.assert_allclose(jet.value, np.sin(t), atol=1e-14)
    np.testing.assert_allclose(jet.first, np.cos(t), atol=1e-14)
    np.testing.assert_allclose(jet.second, -np.sin(t), atol=1e-14)


def test_array_jet_arithmetic_matches_scalar_jets():
    t = np.array([0.3, 1.1])
    vector = (variable(t) * variable(t).cos())
    for index, scalar_t in enumerate(t):
        scalar = variable(scalar_t) * variable(scalar_t).cos()
        assert vector.value[index] == pytest.approx(scalar.value)
        assert vector.first[index] == pytest.approx(scalar.first)
        assert vector.second[index] == pytest.approx(scalar.second)


def test_as_tuple_round_trip():
    value, first, second = variable(0.5).sin().as_tuple()
    assert value == pytest.approx(np.sin(0.5))
    assert first == pytest.approx(np.cos(0.5))
    assert second == pytest.approx(-np.sin(0.5))


def test_module_exports():
    assert set(taylor.__all__) == {"Jet2", "variable", "constant"}
    assert isinstance(variable(0.0), Jet2)
