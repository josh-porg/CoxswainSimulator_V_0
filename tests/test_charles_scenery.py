"""The Charles reach has the data the Seattle courses have.

These are the checks that would have caught what was wrong before: three
quarters of the building heights guessed from the building type, every
tree between 14.0 and 15.0 m, no docks at all, and no photograph.  None
of it raised an error; the model simply answered questions about a river
that had one building height and one tree.
"""

import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "coxswain", "data")


@pytest.fixture(scope="module")
def structures():
    from coxswain.river.structures import charles_structures
    return charles_structures()


# -- buildings ------------------------------------------------------------

def test_most_building_heights_are_not_guessed(structures):
    """Source 2 is "guessed from the building type" -- 9 m for anything
    untagged.  It was 7,101 of 9,463; Overture leaves 126."""
    blob = np.load(os.path.join(DATA, "charles_structures.npz"))
    source = blob["building_height_source"]
    guessed = float((source == 2).mean())
    assert guessed < 0.05, "%.0f%% of heights are still type guesses" % (
        100 * guessed)


def test_building_heights_actually_vary(structures):
    """A guess gives one number per building type.  The check is not the
    mean -- which barely moved -- but the spread: 60 distinct heights
    across 9,463 buildings is a lookup table, not a survey."""
    blob = np.load(os.path.join(DATA, "charles_structures.npz"))
    heights = blob["building_height"]
    assert len(np.unique(heights)) > 1000
    assert heights.max() > 50.0


def test_no_building_is_taller_than_the_region_builds(structures):
    blob = np.load(os.path.join(DATA, "charles_structures.npz"))
    assert blob["building_height"].max() < 200.0


# -- trees ----------------------------------------------------------------

def test_the_trees_are_not_all_the_same_height():
    """Every tree used to be between 14.0 and 15.0 m."""
    from coxswain.river.structures import charles_trees
    trees = charles_trees()
    assert len(trees.heights) > 20000
    assert trees.heights.std() > 2.0
    inside_the_old_band = float(((trees.heights >= 14.0)
                                 & (trees.heights <= 15.0)).mean())
    assert inside_the_old_band < 0.5


def test_the_trees_have_species_and_forms():
    from coxswain.river.structures import charles_trees
    trees = charles_trees()
    named = np.char.str_len(np.asarray(trees.species, dtype=str)) > 0
    assert named.sum() > 5000
    forms = np.asarray(trees.form).astype(int)
    assert (forms == 1).sum() > 100, "no conifers were identified"


def test_tree_heights_are_marked_as_modelled():
    """Modelled from trunk diameter, and never mistakable for measured."""
    blob = np.load(os.path.join(DATA, "charles_trees.npz"))
    source = blob["tree_height_source"]
    assert set(np.unique(source).tolist()) <= {3, 4}
    assert (source == 4).sum() > 5000


# -- docks ----------------------------------------------------------------

def test_the_docks_exist_and_are_on_the_reach():
    from coxswain.river import charles
    docks = charles.load_obstructions()
    assert len(docks) >= 15
    points = np.concatenate([p for _k, p in docks])
    assert -3000 < points[:, 0].min() and points[:, 0].max() < 3000
    assert -3000 < points[:, 1].min() and points[:, 1].max() < 3000


def test_the_docks_narrow_the_corridor():
    """The whole point of extracting them.  If removing the docks changes
    no station's clearance, the layer is not reaching the water."""
    from coxswain.river import charles
    plain = charles.charles_channel()
    rowable = charles.rowable_channel(plain)
    _s, _f, line, _st = charles.hocr_course(plain)
    before = plain.half_width_along(line)
    after = rowable.half_width_along(line)
    assert (after <= before + 1e-6).all()
    assert (before - after > 1.0).sum() > 20


# -- the corridor ---------------------------------------------------------

def test_the_corridor_leaves_room_for_the_blades():
    """Clearance is measured to the hull centreline, so a corridor equal
    to it lets a shell with 3.5 m of blade sit half a metre off a dock."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from render_charles import BOAT_HALF_SPAN, charles_race_course
    course, _geometry, rowable = charles_race_course()
    bare = rowable.half_width_along(course.centreline)
    assert (course.half_width <= bare - BOAT_HALF_SPAN + 1e-6).all()


def test_charles_course_accepts_a_centreline_and_a_half_width():
    """The regression that found the scoping bug.

    ``water_half_width`` was only assigned inside the branch that derived
    the line and the corridor, so the first caller to supply both -- a
    course whose corridor has the docks and the boat's beam taken out --
    raised ``UnboundLocalError``.
    """
    from coxswain.river import charles
    raster = charles.charles_channel()
    line = raster.centreline()
    half = np.full(len(line), 20.0)
    course = charles.charles_course(centreline=line, half_width=half)
    assert course.water_half_width is not None
    assert len(course.water_half_width) == len(line)


# -- imagery --------------------------------------------------------------

def test_the_orthophoto_registers_with_the_elevation_model():
    """Photograph and DEM must cover the same ground, or the picture
    slides across the hills and nothing raises."""
    from coxswain.river.terrain import charles_imagery, charles_terrain
    imagery = charles_imagery()
    terrain = charles_terrain()
    assert abs(imagery.east[0] - terrain.east.min()) < 5.0
    assert abs(imagery.east[-1] - terrain.east.max()) < 5.0
    assert abs(imagery.north[0] - terrain.north.min()) < 5.0
    assert abs(imagery.north[-1] - terrain.north.max()) < 5.0


def test_a_scene_with_no_imagery_still_finds_the_charles_photo():
    """It defaults, the way terrain and structures do."""
    from coxswain.boats import catalog
    from coxswain.viz.river3d import RiverScene
    boat = catalog.coxed_four(rate=30.0, rower_mass=68.0,
                              rower_stature=1.70, coxswain_mass=68.0)
    assert RiverScene(boat).imagery() is not None
