"""Data files must resolve wherever the program is started from.

A packaged build unpacks itself to a temporary directory and is launched
from wherever the user double-clicked, so anything found by a path
relative to the working directory is found only on the machine it was
developed on.
"""

import os

import numpy as np

from coxswain.core.resources import data_path, frozen, project_root


def test_data_paths_are_absolute_and_independent_of_the_cwd(tmp_path,
                                                            monkeypatch):
    first = data_path("data", "totl_course.npy")
    monkeypatch.chdir(tmp_path)
    assert data_path("data", "totl_course.npy") == first
    assert os.path.isabs(first)


def test_the_traced_courses_load_from_anywhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    import sys

    sys.path.insert(0, os.path.join(project_root(), "scripts"))
    for module, key in (("render_totl", "totl"), ("render_hotl", "hotl")):
        loaded = importlib.import_module(module)
        line = np.load(loaded.COURSE_PATH)
        assert len(line) > 10
        assert np.load(loaded.BUOY_PATH).shape[1] == 3


def test_the_packaged_payload_lists_everything_that_is_needed():
    """The build refuses rather than shipping a broken executable."""
    import sys

    sys.path.insert(0, os.path.join(project_root(), "tools"))
    import build_exe

    present, missing = build_exe.check_payload()
    assert not missing, "not built yet: %s" % ", ".join(missing)
    assert build_exe.payload_size(present) > 1.0


def test_not_frozen_in_a_checkout():
    assert frozen() is False


def test_the_payload_covers_every_course_not_just_one():
    """The hand-written list missed charles_isobaths.csv, so the build
    ran Seattle fine and died on the Charles.  Each course's own data
    must be in the payload, and this names them per course so a passing
    test cannot again mean "one course works"."""
    import sys

    sys.path.insert(0, os.path.join(project_root(), "tools"))
    import build_exe

    payload = set(os.path.basename(name)
                  for name in build_exe.discover_payload())
    per_course = {
        "charles": ("charles_dem.tif", "charles_structures.npz",
                    "charles_isobaths.csv", "charles_obstructions.json"),
        "totl": ("seattle_dem.npz", "seattle_structures.npz",
                 "seattle_water.json", "totl_course.npy"),
        "hotl": ("hotl_course.npy", "hotl_buoys.npy",
                 "lake_union_depth.npz", "seattle_bridge_outlines.json"),
    }
    for course, names in per_course.items():
        for name in names:
            assert name in payload, "%s needs %s" % (course, name)


def test_the_payload_is_derived_rather_than_typed():
    """If this ever becomes a literal list again it will drift from the
    code, which is how the last one went wrong."""
    import sys

    sys.path.insert(0, os.path.join(project_root(), "tools"))
    import build_exe

    found = build_exe.discover_payload()
    assert len(found) >= 20
    assert all(name.startswith(("coxswain/data/", "data/")) for name in found)
