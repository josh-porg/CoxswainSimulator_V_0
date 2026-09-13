"""Sretenskii's thin-ship wave resistance in finite depth, W&L (1960) eq. (20.69).

Three checks the formula has to pass before it may replace the chosen
shallow-water factor: its deep limit is Michell's integral; at depth Froude
0.5 and below it is indistinguishable from deep water [D11]; and resistance
peaks just below the critical speed sqrt(g h) (Havelock 1922).  Run on the
Wigley hull, where they are cheap.
"""

import numpy as np
import pytest

from coxswain.hydro.finite_depth_michell import (
    FiniteDepthMichell,
    depth_weight,
    stationary_root,
)
from coxswain.hydro.michell import GRAVITY, MichellWave, wigley_offsets


def wigley(depth, **kwargs):
    x, z, half = wigley_offsets()
    return FiniteDepthMichell(station=x, level=z, half_beam=half,
                              quadrature="trapezoid", depth=depth, **kwargs)


def test_stationary_root_satisfies_the_dispersion_relation():
    nu, depth = 4.0, 1.0
    mu = stationary_root(nu, depth)
    assert 0.0 < mu < nu
    assert mu - nu * np.tanh(mu * depth) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("nu_h", [0.5, 1.0])
def test_no_root_at_or_above_the_critical_speed(nu_h):
    """``U^2 / g h >= 1``: the transverse waves cannot keep up."""
    assert stationary_root(nu_h / 2.0, 2.0) == 0.0


def test_depth_weight_is_one_at_the_surface_and_exponential_when_deep():
    mu = np.array([0.5, 5.0, 50.0])
    level = np.array([-0.2, 0.0])
    shallow = depth_weight(mu, level, 1.0)
    assert np.allclose(shallow[:, 1], 1.0)
    exact = np.cosh(mu[:, None] * (level[None, :] + 1.0)) / np.cosh(mu[:, None])
    assert np.allclose(shallow, exact)
    deep = depth_weight(mu, level, 1.0e4)
    assert np.allclose(deep, np.exp(mu[:, None] * level[None, :]))


def test_refuses_water_shallower_than_the_draft():
    with pytest.raises(ValueError, match="draft"):
        wigley(0.05)


@pytest.mark.parametrize("froude", [0.2, 0.3, 0.5, 1.0])
def test_deep_limit_is_michells_integral(froude):
    """W&L: 'As h -> infinity ... one obtains one of the forms of Michell's
    integral.'  Same hull, same weights, a different variable of integration."""
    x, z, half = wigley_offsets()
    speed = froude * np.sqrt(GRAVITY)
    deep = MichellWave(station=x, level=z, half_beam=half,
                       quadrature="trapezoid").resistance(speed)[0]
    assert wigley(50.0).resistance(speed)[0] == pytest.approx(deep, rel=1e-4)


def test_low_depth_froude_matches_deep_water():
    """[D11]: at ``Fr_h <= 0.5`` 'results are similar to deep water'."""
    x, z, half = wigley_offsets()
    speed = 0.2 * np.sqrt(GRAVITY)                  # Fr_L 0.2
    depth = 0.5                                     # Fr_h 0.28
    deep = MichellWave(station=x, level=z, half_beam=half,
                       quadrature="trapezoid").resistance(speed)[0]
    assert wigley(depth).resistance(speed)[0] == pytest.approx(deep, rel=5e-3)


def test_the_tables_deep_lookup_is_the_integral_off_its_samples():
    """The deep table samples 301 speeds, because 64 read up to 14% off the
    integral at 2-3 m/s and 1.7% at 3-4 on the eight."""
    from coxswain.boats import catalog
    from coxswain.hydro.finite_depth_michell import FiniteDepthWaveTable
    from coxswain.hydro.michell import elliptical_offsets

    boat = catalog.build("8+", rate=32.0)
    x, z, half = elliptical_offsets(boat.offsets, stations=161, levels=21)
    table = FiniteDepthWaveTable(station=x, level=z, half_beam=half)
    direct = MichellWave(station=x, level=z, half_beam=half,
                         quadrature="trapezoid")
    speeds = np.arange(3.0071, 7.5, 0.0313)
    exact = direct.resistance(speeds)
    looked_up = np.array([float(table(s)) for s in speeds])
    assert np.max(np.abs(looked_up - exact) / exact) < 2.5e-3


def test_resistance_peaks_just_below_the_critical_speed():
    """Havelock (1922): the peak sits just below ``sqrt(g h)`` and resistance
    falls past it, as the transverse waves drop out."""
    depth = 0.25
    froude_h = np.linspace(0.6, 1.4, 81)
    resistance = wigley(depth).resistance(froude_h * np.sqrt(GRAVITY * depth))
    peak = froude_h[int(np.argmax(resistance))]
    assert 0.9 <= peak <= 1.0
    assert resistance[-1] < resistance.max()
