"""Rivers as spatial fields: depth, current and navigable channel.

Everything before this module treats the water as a uniform half-space with
one depth and no flow.  That is right for a buoyed 2 km course and wrong for
the Charles, where the depth runs from about 2 m to 8 m, there is a real
downstream current, and the navigable width narrows to a few boat lengths
at the bridges.

Three fields, all queried by absolute-frame position:

:class:`DepthField`
    ``h(x, y)`` in metres.  Feeds
    :class:`~coxswain.hydro.shallow.ShallowWaterModel`, which is where
    finite depth changes the wave resistance.  The effect is not small: an
    eight at 5.5 m/s in 3 m of water loses about 13% of its speed.

:class:`CurrentField`
    Depth-averaged water velocity ``(u, v)`` in m/s.  Enters as a *relative
    velocity*: hydrodynamic forces depend on the boat's motion through the
    water, not over the ground, while the crew's inertial reactions and the
    trajectory itself are in the ground frame.  Getting that split right is
    the whole point of separating them here.

:class:`Course`
    A centreline with a width, so a route can be expressed as an offset
    from the centre rather than in absolute coordinates, and so a candidate
    trajectory can be checked for staying in the water.

Coordinates
-----------
Absolute frame, matching :mod:`coxswain.core.frames`: ``X`` and ``Y``
horizontal in metres, ``Z`` up with the undisturbed surface at
``water_level``.  A real river needs a projection from latitude/longitude
onto a local tangent plane; :func:`local_tangent_plane` does that, so
survey data can be loaded in its native coordinates.

Status
------
The field machinery here is complete and tested.  What is *not* included is
Charles River bathymetry: no open dataset has been loaded, so the built-in
:func:`charles_river_sketch` is an explicitly labelled placeholder with
plausible but invented numbers.  It is there so the plumbing can be
exercised end to end, and it must not be used for a real route decision --
:attr:`Course.is_survey` is ``False`` for it, and
:meth:`Course.require_survey` raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EARTH_RADIUS",
    "DepthField",
    "CurrentField",
    "Course",
    "local_tangent_plane",
    "charles_river_sketch",
]

#: Mean Earth radius, for the tangent-plane projection.
EARTH_RADIUS = 6_371_000.0


def local_tangent_plane(latitude, longitude, origin: Tuple[float, float]):
    """Project geographic coordinates onto a local east-north plane.

    Adequate over the ~10 km of a river reach, where the error from
    ignoring Earth curvature is well under a metre.

    Returns ``(east, north)`` in metres relative to ``origin``, which is
    ``(latitude, longitude)`` in degrees.
    """
    lat0, lon0 = np.radians(origin[0]), np.radians(origin[1])
    lat = np.radians(np.asarray(latitude, dtype=float))
    lon = np.radians(np.asarray(longitude, dtype=float))
    east = EARTH_RADIUS * (lon - lon0) * np.cos(lat0)
    north = EARTH_RADIUS * (lat - lat0)
    return east, north


class DepthField:
    """Water depth as a function of absolute-frame position.

    Built either from a uniform depth or from scattered survey soundings.
    Queries outside the surveyed region return the nearest known value
    rather than extrapolating, because extrapolated bathymetry is worse
    than no bathymetry.
    """

    def __init__(self, depth: float = None, points: np.ndarray = None,
                 depths: np.ndarray = None, minimum: float = 0.5,
                 is_survey: bool = False):
        if depth is None and points is None:
            raise ValueError("give either a uniform depth or survey points")
        self.uniform = None if depth is None else float(depth)
        self.minimum = float(minimum)
        self.is_survey = bool(is_survey)
        self._interpolator = None

        if points is not None:
            points = np.asarray(points, dtype=float)
            depths = np.asarray(depths, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("points must have shape (n, 2)")
            if depths.shape != (points.shape[0],):
                raise ValueError("depths must have one entry per point")
            if np.any(depths <= 0):
                raise ValueError("depths must be positive")
            self.points = points
            self.depths = depths
            self._build_interpolator()
        else:
            self.points = None
            self.depths = None

    def _build_interpolator(self) -> None:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        self._interpolator = LinearNDInterpolator(self.points, self.depths)
        self._fallback = NearestNDInterpolator(self.points, self.depths)

    @classmethod
    def uniform_depth(cls, depth: float) -> "DepthField":
        return cls(depth=depth)

    def __call__(self, x, y):
        """Depth in metres at one or many positions."""
        if self.uniform is not None:
            return np.full(np.shape(x), self.uniform, dtype=float) \
                if np.shape(x) else self.uniform

        query = np.column_stack([np.ravel(x), np.ravel(y)])
        value = self._interpolator(query)
        outside = ~np.isfinite(value)
        if np.any(outside):
            value[outside] = self._fallback(query[outside])
        value = np.maximum(value, self.minimum)
        return value.reshape(np.shape(x)) if np.shape(x) else float(value[0])

    def shallow_model(self, x, y, **kwargs):
        """A :class:`ShallowWaterModel` for the depth at this position."""
        from ..hydro.shallow import ShallowWaterModel

        return ShallowWaterModel(depth=float(self(x, y)), **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.uniform is not None:
            return f"DepthField(uniform={self.uniform:.2f} m)"
        return (f"DepthField({len(self.points)} soundings, "
                f"{self.depths.min():.1f}-{self.depths.max():.1f} m, "
                f"survey={self.is_survey})")


class CurrentField:
    """Depth-averaged water velocity as a function of position.

    Sign convention: the returned vector is the velocity of the *water*
    in the absolute frame.  A boat's velocity through the water is
    ``boat_velocity - current``, which is the quantity the hydrodynamics
    must see.  A crew rowing upstream at 5 m/s over the ground in a 0.4 m/s
    current is doing the hydrodynamic work of 5.4 m/s.
    """

    def __init__(self, velocity=(0.0, 0.0),
                 function: Optional[Callable] = None):
        self.uniform = None if function is not None else \
            np.asarray(velocity, dtype=float)
        if self.uniform is not None and self.uniform.shape != (2,):
            raise ValueError("uniform current must be a 2-vector (east, north)")
        self.function = function

    @classmethod
    def still(cls) -> "CurrentField":
        return cls(velocity=(0.0, 0.0))

    def __call__(self, x, y) -> np.ndarray:
        """Water velocity ``(vx, vy)`` in the absolute frame."""
        if self.function is not None:
            return np.asarray(self.function(x, y), dtype=float)
        return self.uniform.copy()

    def velocity_3d(self, x, y) -> np.ndarray:
        """The same, as a 3-vector with zero vertical component."""
        planar = self(x, y)
        return np.array([planar[0], planar[1], 0.0])

    @property
    def is_still(self) -> bool:
        return self.uniform is not None and not np.any(self.uniform)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.function is not None:
            return "CurrentField(<callable>)"
        return (f"CurrentField(({self.uniform[0]:.2f}, "
                f"{self.uniform[1]:.2f}) m/s)")


@dataclass
class Course:
    """A navigable reach: centreline, width, depth and current.

    The centreline is a polyline of absolute-frame ``(x, y)`` points.
    ``half_width`` may be a scalar or one value per centreline point, and
    bounds the navigable channel either side.
    """

    centreline: np.ndarray
    half_width: np.ndarray
    depth: DepthField
    current: CurrentField = field(default_factory=CurrentField.still)
    name: str = "course"
    is_survey: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        self.centreline = np.asarray(self.centreline, dtype=float)
        if self.centreline.ndim != 2 or self.centreline.shape[1] != 2:
            raise ValueError("centreline must have shape (n, 2)")
        if len(self.centreline) < 2:
            raise ValueError("centreline needs at least two points")

        self.half_width = np.broadcast_to(
            np.asarray(self.half_width, dtype=float),
            (len(self.centreline),)).astype(float)
        if np.any(self.half_width <= 0):
            raise ValueError("half_width must be positive")

        segments = np.diff(self.centreline, axis=0)
        lengths = np.hypot(segments[:, 0], segments[:, 1])
        if np.any(lengths <= 0):
            raise ValueError("centreline has repeated points")
        self.station = np.concatenate([[0.0], np.cumsum(lengths)])

    # -- geometry ---------------------------------------------------------
    @property
    def length(self) -> float:
        """Total centreline length in metres."""
        return float(self.station[-1])

    def position_at(self, station) -> np.ndarray:
        """Centreline position at a distance along the course."""
        station = np.asarray(station, dtype=float)
        x = np.interp(station, self.station, self.centreline[:, 0])
        y = np.interp(station, self.station, self.centreline[:, 1])
        return np.stack([x, y], axis=-1)

    def heading_at(self, station) -> np.ndarray:
        """Centreline heading in radians at a distance along the course."""
        delta = 1e-3 * max(self.length, 1.0)
        ahead = self.position_at(np.minimum(station + delta, self.length))
        behind = self.position_at(np.maximum(station - delta, 0.0))
        step = ahead - behind
        return np.arctan2(step[..., 1], step[..., 0])

    def half_width_at(self, station) -> np.ndarray:
        return np.interp(station, self.station, self.half_width)

    def offset_position(self, station, offset) -> np.ndarray:
        """Position at ``station`` displaced ``offset`` metres to port.

        This is the natural parameterisation for route optimisation: a
        candidate line is a function ``offset(station)``, bounded by the
        channel width, rather than a free curve in the plane.
        """
        centre = self.position_at(station)
        heading = self.heading_at(station)
        normal = np.stack([-np.sin(heading), np.cos(heading)], axis=-1)
        return centre + np.asarray(offset, dtype=float)[..., None] * normal

    def nearest_station(self, x, y) -> float:
        """Distance along the course of the closest centreline point.

        Exact against the polyline segments, not just the vertices.
        """
        point = np.array([x, y], dtype=float)
        starts = self.centreline[:-1]
        segments = np.diff(self.centreline, axis=0)
        lengths_sq = np.einsum("ij,ij->i", segments, segments)
        t = np.clip(
            np.einsum("ij,ij->i", point - starts, segments) / lengths_sq,
            0.0, 1.0)
        projected = starts + t[:, None] * segments
        distances = np.hypot(*(point - projected).T)
        best = int(np.argmin(distances))
        return float(self.station[best]
                     + t[best] * np.sqrt(lengths_sq[best]))

    def is_inside(self, x, y) -> bool:
        """Whether a position lies within the navigable channel."""
        station = self.nearest_station(x, y)
        centre = self.position_at(np.array(station))
        return bool(np.hypot(x - centre[0], y - centre[1])
                    <= self.half_width_at(station))

    # -- fields -----------------------------------------------------------
    def depth_at(self, x, y) -> float:
        return float(self.depth(x, y))

    def current_at(self, x, y) -> np.ndarray:
        return self.current.velocity_3d(x, y)

    def depth_profile(self, n: int = 200):
        """Depth along the centreline, as ``(station, depth)``."""
        station = np.linspace(0.0, self.length, n)
        centre = self.position_at(station)
        return station, self.depth(centre[:, 0], centre[:, 1])

    # -- guard ------------------------------------------------------------
    def require_survey(self) -> None:
        """Raise unless this course is backed by real survey data.

        Call this before quoting a number that a crew would act on.  A
        sketched course reproduces the *shape* of the problem but none of
        its values, and the difference is invisible once results are in a
        table.
        """
        if not self.is_survey:
            raise ValueError(
                f"course {self.name!r} is a sketch, not survey data"
                + (f": {self.notes}" if self.notes else "")
                + ". Load real bathymetry before using it for a routing "
                  "decision."
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Course({self.name!r}, {self.length:.0f} m, "
                f"survey={self.is_survey})")


def charles_river_sketch() -> Course:
    """A PLACEHOLDER Charles River reach -- invented numbers.

    Approximates the Head of the Charles course from the start below BU
    Bridge up to the finish near Eliot Bridge: roughly 4800 m, a channel
    narrowing at the bridges, a nominal downstream current, and a depth
    that shoals on the insides of the bends.

    **None of these values are surveyed.** The geometry is a smooth
    caricature and the depths are plausible guesses in the 2-8 m band the
    river is generally quoted at. It exists to exercise the field
    machinery; :meth:`Course.require_survey` will refuse it.

    To replace it with real data, build a :class:`DepthField` from NOAA or
    USGS soundings projected through :func:`local_tangent_plane`, and pass
    ``is_survey=True``.
    """
    station = np.linspace(0.0, 4800.0, 60)
    # a meandering centreline: two long bends, as the reach actually has
    x = station
    y = 220.0 * np.sin(2.0 * np.pi * station / 3600.0) \
        + 70.0 * np.sin(2.0 * np.pi * station / 1250.0)
    centreline = np.column_stack([x, y])

    # channel pinches at three bridge-like stations
    half_width = np.full_like(station, 45.0)
    for bridge in (900.0, 2400.0, 3900.0):
        half_width -= 22.0 * np.exp(-((station - bridge) / 160.0) ** 2)

    # depth: deeper in the channel, shoaling on the inside of each bend
    rng = np.random.default_rng(20260819)
    sounding_station = np.repeat(np.linspace(0.0, 4800.0, 120), 7)
    sounding_offset = np.tile(np.linspace(-40.0, 40.0, 7), 120)
    centre = np.column_stack([
        sounding_station,
        220.0 * np.sin(2.0 * np.pi * sounding_station / 3600.0)
        + 70.0 * np.sin(2.0 * np.pi * sounding_station / 1250.0),
    ])
    points = centre + np.column_stack([np.zeros_like(sounding_offset),
                                       sounding_offset])
    depth = (5.2
             - 2.3 * (sounding_offset / 45.0) ** 2
             + 1.1 * np.sin(2.0 * np.pi * sounding_station / 1700.0)
             + rng.normal(0.0, 0.12, size=points.shape[0]))
    depth = np.clip(depth, 1.6, 8.0)

    return Course(
        centreline=centreline,
        half_width=half_width,
        depth=DepthField(points=points, depths=depth, is_survey=False),
        current=CurrentField(velocity=(-0.25, 0.0)),
        name="Charles River (SKETCH)",
        is_survey=False,
        notes="invented bathymetry and geometry; not surveyed",
    )
