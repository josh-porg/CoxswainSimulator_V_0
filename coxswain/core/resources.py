r"""Finding the project's data files, in a checkout or inside an executable.

Most of what the trainer needs at run time is opened by file name rather
than imported: the elevation model, the orthophoto, the building and tree
extracts, the traced courses and their buoys, the baked water textures.
In a checkout those live at the repository root under ``data/`` and
inside the package under ``coxswain/data/``, and code found them with a
path relative to the working directory.

That works until the program is not started from the repository root --
which is exactly what happens when it is packaged as an executable and
handed to somebody else.  PyInstaller unpacks the bundle to a temporary
directory and sets :data:`sys._MEIPASS` to it; the working directory is
wherever the user happened to double-click.  So a relative ``data/...``
resolves to nothing, and the failure appears only on the other person's
machine, which is the worst place to discover it.

:func:`data_path` asks, in order: the unpacked bundle, then the
repository root. Both answers are absolute, so neither depends on where
the program was started.
"""

from __future__ import annotations

import os
import sys

__all__ = ["bundle_root", "project_root", "data_path", "frozen"]


def frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False)
                and hasattr(sys, "_MEIPASS"))


def bundle_root():
    """Where the bundle was unpacked, or ``None`` in a checkout."""
    return getattr(sys, "_MEIPASS", None) if frozen() else None


def project_root() -> str:
    """The repository root: two directories above this file."""
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def data_path(*parts: str) -> str:
    """Absolute path to a data file, wherever it is being run from.

    ``data_path("data", "totl_course.npy")`` in a checkout gives the file
    in the repository; in a packaged build it gives the copy inside the
    unpacked bundle.  Returns the repository path when neither exists, so
    that a missing file reports the place a developer would look.
    """
    relative = os.path.join(*parts)
    root = bundle_root()
    if root is not None:
        candidate = os.path.join(root, relative)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(project_root(), relative)
