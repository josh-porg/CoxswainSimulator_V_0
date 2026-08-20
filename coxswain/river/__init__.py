"""Rivers as spatial fields: depth, current and navigable channel.

The long-term goal of this project is trajectory optimisation over a real
reach of the Charles.  This subpackage holds the spatial description that
makes that possible: where the water is, how deep it is, and how fast it
is moving.

The field machinery is complete; Charles bathymetry is not loaded.  See
:func:`~coxswain.river.course.charles_river_sketch`.
"""

from .course import (
    Course,
    CurrentField,
    DepthField,
    charles_river_sketch,
    local_tangent_plane,
)

__all__ = [
    "Course",
    "CurrentField",
    "DepthField",
    "charles_river_sketch",
    "local_tangent_plane",
]
