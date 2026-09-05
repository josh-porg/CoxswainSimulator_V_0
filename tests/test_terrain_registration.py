"""The DEM has to sit where the shoreline says it does.

This is a regression test for a failure that produced no error and no
implausible number.  ``exportImage`` honours its ``size`` argument
exactly and **moves the bounding box** to match that size's aspect ratio,
announcing the change only in a JSON field nobody was reading.  Asking
for 47.590-47.670 N at a size computed from metres returned
47.571-47.689, and the resulting tile georeferenced Lake Union 2.2 km
from itself -- putting the racing water on the side of a hill while every
summary statistic stayed entirely reasonable.

The check that catches it is a *cross-dataset* one: OpenStreetMap says
where the shore is and 3DEP says where the low ground is, and those two
have to be the same place.  Neither dataset can confirm it alone, which
is why nothing caught it until the two were drawn on top of each other.
"""

import numpy as np
import pytest

from coxswain.river.terrain import (SEATTLE_POOL_LEVEL, charles_terrain,
                                    pool_level_from, seattle_terrain)


@pytest.fixture(scope="module")
def lake():
    from coxswain.river.seattle import water_mask
    east, north, water = water_mask(10.0, names=("Lake Union",))
    grid_east, grid_north = np.meshgrid(east, north)
    return grid_east[water], grid_north[water], water


def test_the_dem_is_low_everywhere_the_shoreline_says_water(lake):
    """The decisive check: OSM water must sit on 3DEP low ground.

    A 2.2 km georeferencing error moves 43% of the lake onto ground
    above 10 m.  Correctly registered it is under 1%, and the residue is
    the shoreline cells themselves, where a 10 m raster straddles the
    bank.
    """
    east, north, _mask = lake
    elevation = seattle_terrain().at(east, north)
    assert (elevation > 10.0).mean() < 0.01


def test_the_lake_surface_matches_the_stated_pool_level(lake):
    """And it must be low at the *right* height, not merely low.

    Elliott Bay is also low ground, three kilometres away and five
    metres down; a tile shifted onto it would pass the test above.
    """
    east, north, _mask = lake
    elevation = seattle_terrain().at(east, north)
    assert abs(np.quantile(elevation, 0.05) - SEATTLE_POOL_LEVEL) < 0.5


def test_landmark_hills_stand_where_they_are(lake):
    """Queen Anne and Capitol Hill, against their published summits.

    Elevation alone would survive a pure translation along the lake, so
    this pins the tile in both axes using ground that is nowhere near
    the water.
    """
    from coxswain.river.course import local_tangent_plane
    from coxswain.river.seattle import SEATTLE_ORIGIN

    terrain = seattle_terrain()
    # (latitude, longitude, published summit in metres)
    summits = [(47.6370, -122.3570, 139.0),    # Queen Anne
               (47.6300, -122.3120, 133.0)]    # Capitol Hill
    for latitude, longitude, published in summits:
        east, north = local_tangent_plane(np.array([latitude]),
                                          np.array([longitude]),
                                          SEATTLE_ORIGIN)
        # Take the local maximum: a summit is a point and the tile is a
        # 3 m raster, so demanding the exact cell is a test of the
        # gazetteer's rounding rather than of the registration.
        offsets = np.linspace(-120.0, 120.0, 9)
        block = np.array([[terrain.at(east[0] + dx, north[0] + dy)[0]
                           for dx in offsets] for dy in offsets])
        assert abs(block.max() - published) < 15.0


def test_pool_level_from_finds_a_flat_low_surface():
    grid = np.concatenate([np.full(400, 5.2), np.linspace(6.0, 90.0, 600)])
    assert abs(pool_level_from(grid) - 5.2) < 0.05


def test_each_course_carries_its_own_waterline():
    """The Charles basin and Lake Union are 4.5 m apart vertically.

    ``POOL_LEVEL`` used to be a module constant, so the second course
    would have been drawn nearly five metres under its own banks.
    """
    assert abs(charles_terrain().pool - 0.6) < 1e-6
    assert abs(seattle_terrain().pool - SEATTLE_POOL_LEVEL) < 1e-6
    assert seattle_terrain().pool - charles_terrain().pool > 4.0
