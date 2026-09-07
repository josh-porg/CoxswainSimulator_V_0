"""The hull's wave pattern from the free-surface Green's function.

Everything here is a property of the *shape* of the pattern, which the
integral fixes and the closure constant cannot touch -- except the last
test, which checks the closure does what it says.
"""

import numpy as np
import pytest

from coxswain.boats import catalog
from coxswain.hydro.havelock import (GRAVITY, HavelockField, bake_wave,
                                     closure_scale)

TAN_KELVIN = np.tan(np.arcsin(1.0 / 3.0))


@pytest.fixture(scope="module")
def boat():
    return catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)


@pytest.fixture(scope="module")
def field(boat):
    return HavelockField(boat.offsets)


def test_the_transverse_wavelength_is_two_pi_u_squared_over_g(field, boat):
    """The lambda -> 1 end of the integral is the transverse system, and
    its wavelength is a constant of deep water, not of the hull."""
    half = 0.5 * boat.length
    for speed in (3.5, 4.5, 5.5):
        expected = 2.0 * np.pi * speed ** 2 / GRAVITY
        # A window that holds several wavelengths at every speed, well
        # clear of the hull; an 80 m window held four at 5.5 m/s and the
        # autocorrelation found a 2 m artefact instead.
        xs = np.linspace(-half - 175.0, -half - 25.0, 1501)
        line = field.elevation(xs, np.array([0.0]), speed)[0]
        line = line - line.mean()
        ac = np.correlate(line, line, "full")[len(line) - 1:]
        dx = xs[1] - xs[0]
        lo, hi = int(0.5 * expected / dx), int(2.0 * expected / dx)
        peak = (np.argmax(ac[lo:hi]) + lo) * dx
        assert abs(peak - expected) < 0.04 * expected


def test_the_pattern_is_confined_to_the_kelvin_wedge(field, boat):
    """Outside 19.47 degrees the free waves cancel.

    This is the stationary-phase result and it falls out of the integral
    rather than being put in: there is no wedge anywhere in the code.
    It is also the test that failed before the quadrature was made dense
    enough to resolve the phase, when the pattern filled 48 degrees.
    """
    half = 0.5 * boat.length
    for speed in (3.5, 4.5, 5.5):
        astern = 45.0
        ys = np.linspace(0.0, 2.6 * TAN_KELVIN * astern, 400)
        cut = np.abs(field.elevation(np.array([-half - astern]), ys,
                                     speed)[:, 0])
        inside = cut[ys < 0.8 * TAN_KELVIN * astern].max()
        outside = cut[ys > 1.7 * TAN_KELVIN * astern].max()
        assert outside < 0.06 * inside


def test_nothing_is_radiated_ahead_of_the_bow(field, boat):
    """The radiation condition: a source leaves waves behind it only."""
    half = 0.5 * boat.length
    ahead = field.elevation(np.linspace(half + 0.5, half + 20.0, 50),
                            np.array([0.0, 2.0, 8.0]), 4.5)
    assert float(np.abs(ahead).max()) == 0.0


def test_the_separable_far_field_matches_the_explicit_sum(field, boat):
    """Astern of the hull the sum collapses to one exponential per
    lambda; that shortcut must be exact, not approximate."""
    half = 0.5 * boat.length
    speed = 4.5
    xs = np.array([-half - 15.0, -half - 60.0])
    ys = np.linspace(-25.0, 25.0, 101)
    fast = field.elevation(xs, ys, speed)
    k0, lam, weight = field._quadrature(speed)
    decay = np.exp(np.clip(k0 * lam[:, None] ** 2 * field.level[None, :],
                           -700.0, 0.0))
    strength = np.einsum("lz,xz->lx", decay, field.slope) \
        * field._dz * field._dx
    across = np.cos(k0 * lam[:, None]
                    * np.sqrt(np.maximum(lam[:, None] ** 2 - 1.0, 0.0))
                    * ys[None, :])
    slow = np.zeros_like(fast)
    for j, xe in enumerate(xs):
        phase = np.sin(k0 * lam[:, None] * (field.station - xe)[None, :])
        slow[:, j] = (weight * np.einsum("lx,lx->l", strength, phase)) @ across
    assert np.abs(fast - slow).max() < 1e-12 * max(np.abs(fast).max(), 1e-9)


def test_the_closure_puts_the_wave_resistance_into_the_wake(field, boat):
    """Energy across the wedge per metre of track equals R_w, by
    construction -- so check the construction: rescaling the field by the
    closure and re-measuring must give the resistance back."""
    from coxswain.hydro.michell import MichellWave

    speed = 4.5
    resistance = float(MichellWave.from_offsets(boat.offsets)
                       .resistance(np.array([speed]))[0])
    scale = closure_scale(field, speed, resistance=resistance)
    assert scale > 0.0
    wavelength = 2.0 * np.pi * speed ** 2 / GRAVITY
    astern = 45.0
    xs = -astern - 0.5 * boat.length - np.linspace(0.0, wavelength, 9,
                                                   endpoint=False)
    width = 1.6 * TAN_KELVIN * astern + 6.0
    ys = np.linspace(-width, width, 241)
    eta = field.elevation(xs, ys, speed, scale=scale)
    energy = 0.5 * 1000.0 * GRAVITY * float(
        np.mean(eta ** 2, axis=1).sum() * (ys[1] - ys[0]))
    assert abs(energy - resistance) < 0.02 * resistance


def test_the_bake_covers_the_speeds_the_boat_is_rowed_at(boat):
    speeds, east, north, waves, scales = bake_wave(
        boat, speeds=np.array([3.0, 4.5, 6.0]), nx=61, ny=31)
    assert waves.shape == (3, 31, 61)
    assert (scales > 0.0).all()
    assert np.isfinite(waves).all()
    # A faster boat leaves a longer wake: energy sits further astern.
    half = 0.5 * boat.length
    behind = east < -half
    weight = np.abs(waves[:, :, behind]).sum(axis=1)     # (speeds, X)
    centroid = (weight * east[None, behind]).sum(axis=1) / weight.sum(axis=1)
    assert centroid[2] < centroid[0]
