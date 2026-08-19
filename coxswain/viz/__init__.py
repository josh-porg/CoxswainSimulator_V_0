"""Diagnostic plotting and 3-D visualisation.

:mod:`~coxswain.viz.plots` needs only matplotlib.
:mod:`~coxswain.viz.scene3d` additionally needs PyVista, and is imported
lazily so that the rest of the package does not depend on VTK.

    python -m coxswain.viz --boat 8+ --rate 32 --show-3d
"""

from . import plots

__all__ = ["plots", "scene3d"]


def __getattr__(name):
    if name == "scene3d":
        from . import scene3d
        return scene3d
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
