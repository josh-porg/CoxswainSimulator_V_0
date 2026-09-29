"""Whicker & Fehlner (1958), DTMB Report 933, eq. [1], as printed on p. 28."""
import dataclasses

import numpy as np
import pytest

from coxswain.hydro import appendages as A


def printed(a_e, sweep_deg, alpha_deg, c_dc):
    """Eq. [1] in its own units: alpha in degrees, a0 per degree."""
    a0 = 0.9 * 2 * np.pi / 57.3
    om = np.radians(sweep_deg)
    slope = a0 * a_e / (np.cos(om) * np.sqrt(a_e ** 2 / np.cos(om) ** 4 + 4) + 57.3 * a0 / np.pi)
    return slope * alpha_deg + c_dc / a_e * (alpha_deg / 57.3) ** 2


@pytest.mark.parametrize("alpha", [1.0, 8.0, 15.0, 25.0])
def test_the_option_is_eq_1(alpha):
    fin = dataclasses.replace(A.SKEG_EIGHT, lift_model="whicker_fehlner", crossflow_coefficient=0.80)
    a_e = 2.0 * fin.aspect_ratio
    assert float(A.lift_coefficient_at(fin, np.radians(alpha))) == pytest.approx(
        printed(a_e, np.degrees(fin.sweep), alpha, 0.80), rel=1e-3)   # they round 180/pi to 57.3
    assert float(A.lift_coefficient_at(fin, -np.radians(alpha))) == pytest.approx(
        -printed(a_e, np.degrees(fin.sweep), alpha, 0.80), rel=1e-3)


def test_their_own_models_slope_at_aspect_ratio_one():
    """a_e = 1, no sweep: 0.9 * 2 pi / (sqrt(5) + 1.8) = 1.401 per radian."""
    fin = A.LiftingSurface(span=1.0, chord=2.0, position=np.zeros(3), lift_model="whicker_fehlner",
                           reflection=2.0)
    assert fin.aspect_ratio == pytest.approx(0.5)
    small = float(A.lift_coefficient_at(fin, 1e-6)) / 1e-6
    assert small == pytest.approx(0.9 * 2 * np.pi / (np.sqrt(5.0) + 1.8), rel=1e-6)


def test_the_default_is_the_shipped_law_and_bad_inputs_are_refused():
    assert A.SKEG_EIGHT.lift_model == "legacy"
    with pytest.raises(ValueError):
        dataclasses.replace(A.SKEG_EIGHT, lift_model="guess")
    with pytest.raises(ValueError):
        dataclasses.replace(A.SKEG_EIGHT, reflection=3.0)
