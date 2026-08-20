"""Rivers as spatial fields, and optimal lines through them.

The long-term goal of this project is trajectory optimisation over a real
reach of the Charles.  This subpackage holds everything spatial that makes
that possible, in four layers:

:mod:`~coxswain.river.course`
    The generic description -- a centreline with a width, a depth field and
    a current field, queried by position.

:mod:`~coxswain.river.charles`
    Real data: the CRAB / MIT Sea Grant one-foot depth contours and a
    continuity flow model driven by USGS discharge at Waltham.

:mod:`~coxswain.river.channel`
    Where the water actually is.  Reconstructs the water body from the
    contours as an alpha shape and derives the navigable channel,
    centreline and half-width from a distance transform -- rather than
    assuming any of them.

:mod:`~coxswain.river.route` and :mod:`~coxswain.river.trajectory`
    Two ways to ask for the best line.  ``route`` scores an offset profile
    quasi-steadily and is fast enough to search; ``trajectory`` solves a
    minimum-time optimal control problem with rudder and crew pressure as
    controls, by Hermite-Simpson collocation.
"""

from .channel import ChannelRaster, build_channel
from .course import (
    Course,
    CurrentField,
    DepthField,
    charles_river_sketch,
    local_tangent_plane,
)
from .route import Route, RouteEvaluator, optimise_route

__all__ = [
    "ChannelRaster",
    "Course",
    "CurrentField",
    "DepthField",
    "Route",
    "RouteEvaluator",
    "build_channel",
    "charles_river_sketch",
    "local_tangent_plane",
    "optimise_route",
]
