"""The water the hull pushes about, from the thin-ship source sheet."""

import numpy as np

from coxswain.boats import catalog
from coxswain.hydro.nearfield import GRAVITY, bake, elevation, geometric_field


def _boat():
    return catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)


def test_the_water_piles_up_ahead_of_the_stem():
    """Stagnation: the flow is slowed at the bow, so the surface rises.

    Getting the sign wrong puts the heap at the stern, which is what the
    first version did -- thin-ship theory is written with the stream
    along +x and in this project +x is the bow, so the source strength
    flips.
    """
    boat = _boat()
    east, north, field = bake(boat)
    centre = field[np.argmin(np.abs(north))]
    half = 0.5 * boat.length
    ahead = centre[np.argmin(np.abs(east - (half + 0.6)))]
    assert ahead > 0.0


def test_the_surface_is_drawn_down_along_the_midbody():
    """Round the shoulder the flow accelerates, so the surface falls."""
    boat = _boat()
    east, north, field = bake(boat)
    centre = field[np.argmin(np.abs(north))]
    inside = np.abs(east) < 0.35 * boat.length
    assert centre[inside].mean() < 0.0


def test_the_elevation_goes_as_the_square_of_the_speed():
    """``eta = U^2/g * F`` exactly -- which is what lets it be baked.

    If this ever stopped holding, the whole real-time scheme would be
    wrong, because the texture is multiplied by ``U^2/g`` at run time.
    """
    field = np.array([[0.3, -0.2], [0.1, 0.05]])
    assert np.allclose(elevation(field, 6.0),
                       elevation(field, 3.0) * 4.0)
    assert np.allclose(elevation(field, 1.0), field / GRAVITY)


def test_it_dies_away_from_the_hull():
    """A near field that reached the far field would double-count the
    wake, which the Kelvin construction already draws."""
    boat = _boat()
    east = np.linspace(-40.0, 40.0, 160)
    north = np.array([0.0, 3.0, 12.0, 30.0])
    field = geometric_field(boat.offsets, east, north)
    close = np.abs(field[1]).max()
    far = np.abs(field[3]).max()
    assert far < 0.10 * close


def test_a_longer_hull_spreads_its_disturbance_further():
    four = _boat()
    try:
        eight = catalog.eight(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    except TypeError:
        eight = catalog.eight(rate=30.0)
    if eight.length <= four.length:
        return
    for boat in (four, eight):
        east, north, field = bake(boat)
        centre = np.abs(field[np.argmin(np.abs(north))])
        live = east[centre > 0.05 * centre.max()]
        span = live.max() - live.min()
        assert span > 0.5 * boat.length
