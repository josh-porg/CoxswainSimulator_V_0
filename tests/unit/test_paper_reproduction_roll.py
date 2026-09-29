"""Roll damping against the formulas as printed: Ikeda's lift and Kato's friction, from
Falzarano, Somayajula & Seah (2015), Ocean Systems Engineering 5(2), eqs. (6)-(13).

The review prints Ikeda's figures, not their inputs, so this is the formulas reproduced, not
a worked example.
"""
import numpy as np
import pytest

from coxswain import physics
from coxswain.boats import catalog
from coxswain.hydro.radiation import StripDamping


@pytest.fixture(scope="module")
def strip():
    return StripDamping(catalog.single_scull(rate=30.0).offsets)


@pytest.mark.parametrize("og", [0.0, 0.05, -0.1, -0.3])
def test_ikeda_lift_is_eq_12(strip, og):
    """B_L = 0.075 rho U L D^3 k_N (1 + 2.8 OG/D + 4.667 (OG/D)^2), k_N = 2 pi D / L (C_M < 0.92)."""
    rho, U, L, D = 1000.0, 4.2, strip.length, strip.max_draft
    k_n = 2 * np.pi * D / L
    printed = 0.075 * rho * U * L * D ** 3 * k_n * (1 + 2.8 * og / D + (0.7 / 0.15) * (og / D) ** 2)
    assert strip.roll_lift(U, rho, vertical_centre=og) == pytest.approx(printed, rel=1e-12)


def test_kato_friction_is_eqs_6_to_11(strip):
    rho, nu, w, R0, U = 1000.0, 1.0e-6, 3.0, np.radians(3.0), 4.0
    L, B, D, cb = strip.length, strip.max_beam, strip.max_draft, strip.block_coefficient
    S = L * (1.7 * D + cb * B)
    re = (0.887 + 0.145 * cb) * (S / L) / np.pi
    cf = 1.328 * (3.22 * re ** 2 * R0 ** 2 * w / nu) ** -0.5
    bf0 = 4 / (3 * np.pi) * rho * S * re ** 3 * R0 * w * cf
    assert strip.roll_friction_kato(w, R0, rho, nu) == pytest.approx(bf0, rel=1e-12)
    assert strip.roll_friction_kato(w, R0, rho, nu, speed=U) == pytest.approx(
        bf0 * (1 + 4.1 * U / (w * L)), rel=1e-12)


def test_the_block_coefficient_of_a_shell_is_plausible(strip):
    assert 0.3 < strip.block_coefficient < 0.6


def test_kato_is_research_only():
    assert physics.resolve("shipped").roll_friction == "legacy"
    assert physics.resolve("research").roll_friction == "kato"
    boat = physics.resolve("research").apply(catalog.single_scull(rate=30.0))
    assert boat.roll_friction == "kato"
    plain = catalog.single_scull(rate=30.0)
    assert getattr(plain, "roll_friction", "legacy") == "legacy"
    assert StripDamping(plain.offsets).roll_friction_form == "legacy"
    with pytest.raises(ValueError):
        StripDamping(plain.offsets, roll_friction="guess")
