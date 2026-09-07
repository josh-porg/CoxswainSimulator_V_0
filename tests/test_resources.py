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
