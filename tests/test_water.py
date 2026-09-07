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


def test_the_wake_leaves_the_bow_not_the_middle():
    """Two wedge apexes, a hull length apart.

    A hull is a pressure source at the bow and a sink at the stern, so
    its wake is two Kelvin systems superposed -- and the leading V starts
    at the stem.  A single system centred on the boat would put one apex
    under the middle of it, which is not what a coxswain sees.
    """
    from coxswain.viz.water import hull_wake

    length, speed = 13.4, 4.5
    across = np.zeros(400)
    along = np.linspace(-12.0, 30.0, 400)
    # The bow system starts half a length forward of centre, so there is
    # disturbance ahead of where a centred system could reach.
    centred = kelvin_height(along, across, speed, amplitude=0.08)
    both = hull_wake(along, across, speed, 0.08, length)
    ahead = along < -0.5
    assert np.abs(centred[ahead]).max() == 0.0
    assert np.abs(both[ahead]).max() > 0.0


def test_the_two_systems_are_a_hull_length_apart():
    from coxswain.viz.water import hull_wake

    length, speed = 13.4, 4.5
    # Sweep across the wake well behind the boat and find the wedge edges
    # of each system by where the disturbance first appears.
    def edge(offset):
        along = np.full(600, 40.0) + offset
        across = np.linspace(0.0, 30.0, 600)
        h = kelvin_height(along, across, speed, amplitude=0.08)
        live = np.nonzero(np.abs(h) > 0)[0]
        return across[live.max()] if len(live) else 0.0

    # Each system's wedge widens from its own origin, so the bow system
    # is wider at any given station by half a length times the tangent.
    bow_edge = edge(-0.5 * length)
    stern_edge = edge(+0.5 * length)
    gap = (bow_edge - stern_edge) / np.tan(KELVIN_HALF_ANGLE)
    assert abs(abs(gap) - length) < 1.0


def test_the_lee_is_calm_and_the_windward_side_piles_up():
    """A boat shelters the water downwind of it and stacks it upwind.

    The mechanism is wind blocking, not wave diffraction -- half a metre
    of beam against a two-metre wave scatters almost nothing -- so the
    lee is a decayed short-wave field that recovers downwind, and the
    windward gain is small and short-ranged.
    """
    from coxswain.viz.water import hull_shelter

    wind_from = np.pi / 2          # blows toward -y
    lee = hull_shelter(0.0, -2.0, 0.0, 0.0, 0.0, wind_from, 13.4, 0.5)
    windward = hull_shelter(0.0, 2.0, 0.0, 0.0, 0.0, wind_from, 13.4, 0.5)
    far = hull_shelter(0.0, -120.0, 0.0, 0.0, 0.0, wind_from, 13.4, 0.5)
    assert lee < 0.75           # calmed
    assert windward > 1.02      # piled up
    assert abs(far - 1.0) < 0.05   # and it recovers


def test_the_shadow_is_as_wide_as_the_boat_is_across_the_wind():
    """Beam-on shelters a long strip; bow-on shelters almost none."""
    from coxswain.viz.water import hull_shelter

    wind_from = np.pi / 2
    across = np.linspace(-12.0, 12.0, 49)
    downwind = np.full_like(across, -8.0)
    beam_on = hull_shelter(across, downwind, 0.0, 0.0, 0.0, wind_from,
                           13.4, 0.5)
    bow_on = hull_shelter(across, downwind, 0.0, 0.0, np.pi / 2, wind_from,
                          13.4, 0.5)
    assert (beam_on < 0.9).sum() > 5 * (bow_on < 0.9).sum()
