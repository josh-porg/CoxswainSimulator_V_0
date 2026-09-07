"""The water is the sea state the analysis reports, plus the boat's wake."""

import numpy as np

from coxswain.viz.water import (KELVIN_HALF_ANGLE, PuddleTrail,
                                kelvin_height, kelvin_wavelength, sea_for)


def test_the_surface_has_the_significant_height_it_was_asked_for():
    """A sum of components is only the right sea if its statistics are.

    ``H_s = 4 sqrt(m0)`` for a narrow-band sea, so four times the
    standard deviation of the surface must come back as the height
    ``FetchLimitedSea`` predicted.  Nothing here is tuned to make that
    true -- the component amplitudes are set from the spectrum.
    """
    for wind in (4.0, 8.0, 12.0):
        field = sea_for(wind, 1000.0)
        rng = np.random.default_rng(0)
        east = rng.uniform(-400.0, 400.0, 30000)
        north = rng.uniform(-400.0, 400.0, 30000)
        surface = field.height_at(east, north, 0.0)
        assert abs(4.0 * surface.std()
                   - field.significant_height) < 0.02 * field.significant_height


def test_a_flat_calm_is_flat():
    field = sea_for(0.0, 1000.0)
    assert field.significant_height == 0.0
    assert float(np.abs(field.height_at(np.zeros(5), np.zeros(5), 0.0)).max()) == 0.0


def test_the_kelvin_wedge_is_a_constant_of_the_water():
    """19.47 degrees, and the same at every speed -- which is why every
    boat's wake is the same shape."""
    assert abs(np.degrees(KELVIN_HALF_ANGLE) - 19.471) < 0.01
    for speed in (2.0, 4.0, 6.0):
        along = np.full(40, 25.0)
        across = np.linspace(0.0, 25.0, 40)
        height = kelvin_height(along, across, speed)
        edge = across[np.nonzero(height)[0].max()]
        assert abs(edge / 25.0 - np.tan(KELVIN_HALF_ANGLE)) < 0.05


def test_nothing_outside_the_wedge():
    across = np.tan(KELVIN_HALF_ANGLE) * 30.0 * 1.5
    assert kelvin_height(np.array([30.0]), np.array([across]), 4.0)[0] == 0.0
    # and nothing ahead of the boat, either
    assert kelvin_height(np.array([-5.0]), np.array([0.0]), 4.0)[0] == 0.0


def test_the_wake_lengthens_with_the_square_of_the_speed():
    """``2 pi V^2 / g`` exactly, which is why a faster boat leaves a
    longer-waved wake."""
    assert abs(kelvin_wavelength(4.0) / kelvin_wavelength(2.0) - 4.0) < 1e-6
    assert abs(kelvin_wavelength(5.0) - 2 * np.pi * 25.0 / 9.80665) < 1e-6


def test_puddles_fade_and_the_buffer_does_not_grow():
    trail = PuddleTrail(capacity=4, lifetime=6.0)
    for k in range(10):
        trail.drop(float(k), 0.0, float(k))
    assert len(trail.points) == 4
    rows = trail.as_uniform(now=9.5)
    # The one dropped at t=9 is nearly fresh; the one at t=6 is half gone.
    assert rows[:, 2].max() > 0.9
    assert (rows[:, 2] > 0.0).all()
    # Long after, they are all gone.
    assert float(PuddleTrail.as_uniform(trail, now=100.0)[:, 2].max()) == 0.0
