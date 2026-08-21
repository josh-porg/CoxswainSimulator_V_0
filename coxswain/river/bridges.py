"""Bridges as navigation constraints, not just landmarks.

Six bridges cross the Head of the Charles course, and they are the
tightest constraint on it -- hitting a pier is the classic way to lose the
race.  Until now ``charles.BRIDGES`` held only their centre coordinates,
which is enough to label a plot and nothing else: an optimised trajectory
was free to pass straight through a pier.

What is real here and what is not
---------------------------------
The **deck geometry is real**.  Each bridge's carriageway centreline comes
from OpenStreetMap, projected to the same local tangent plane as the depth
survey, so the line a bridge draws across the river is measured rather
than guessed.

The **pier positions are not surveyed**.  No published source for the arch
spans of these particular bridges was found.  Rather than invent them, the
navigable opening is derived from data that does exist: a gate is open
exactly where the bridge line crosses water deep enough to row, according
to the same channel raster the rest of the model uses.  That gives a hard
constraint that is honest about what it knows -- it will keep a boat off
the abutments and out of the shallows, and it will not pretend to know
where a mid-river pier stands.

:attr:`BridgeGate.piers` is there for when survey data turns up.  Any
piers listed are subtracted from the opening.

Why a gate and not an obstacle
------------------------------
A bridge is a line the boat must cross exactly once, at a point that has to
be inside an opening with clearance.  Writing it that way gives the
optimiser a single scalar constraint per bridge evaluated at one crossing,
instead of a keep-out region it has to be excluded from along the whole
trajectory.  That is both cheaper and better conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["Pier", "BridgeGate", "OSM_BRIDGE_DECKS", "build_gates"]


#: Bridge deck centrelines from OpenStreetMap, as ``(lat, lon)`` endpoints
#: of the span crossing the river.  Queried from the Overpass API over the
#: racing reach; see the module docstring for provenance.
OSM_BRIDGE_DECKS = {
    "Eliot Bridge": ((42.3714040, -71.1335355), (42.3719846, -71.1322580)),
    "Larz Anderson": ((42.3686436, -71.1234695), (42.3692860, -71.1229182)),
    "Weeks Footbridge": ((42.3680971, -71.1185501), (42.3688631, -71.1177282)),
    "Western Avenue": ((42.3642964, -71.1162617), (42.3641615, -71.1181022)),
    "River Street": ((42.3609465, -71.1177158), (42.3614261, -71.1162116)),
}


@dataclass(frozen=True)
class Pier:
    """A pier, as a keep-out interval along the bridge span.

    ``centre`` is the distance in metres from the first span endpoint;
    ``width`` is the pier's full width across the channel.
    """

    centre: float
    width: float

    @property
    def interval(self) -> Tuple[float, float]:
        return (self.centre - 0.5 * self.width,
                self.centre + 0.5 * self.width)


@dataclass
class BridgeGate:
    """A line the boat must cross, and the part of it that is passable."""

    name: str
    start: np.ndarray              # (2,) metres east/north
    end: np.ndarray                # (2,) metres east/north
    piers: Sequence[Pier] = field(default_factory=tuple)
    _openings: Optional[Sequence[Tuple[float, float]]] = None

    # -- geometry ---------------------------------------------------------
    @property
    def span(self) -> float:
        return float(np.hypot(*(self.end - self.start)))

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from ``start`` to ``end``, across the river."""
        return (self.end - self.start) / self.span

    @property
    def normal(self) -> np.ndarray:
        """Unit vector along the river, perpendicular to the span."""
        along = self.direction
        return np.array([-along[1], along[0]])

    def point_at(self, distance) -> np.ndarray:
        """Position on the span, ``distance`` metres from ``start``."""
        return self.start + np.asarray(distance)[..., None] * self.direction

    def station_of(self, point) -> float:
        """Distance along the span of the projection of ``point``."""
        offset = np.asarray(point, dtype=float)[..., :2] - self.start
        return float(np.dot(offset, self.direction))

    def signed_distance(self, point) -> float:
        """Signed distance from the gate line, along the river direction.

        Changes sign as the boat passes the bridge, which is what makes it
        usable as the event that locates the crossing.
        """
        offset = np.asarray(point, dtype=float)[..., :2] - self.start
        return float(np.dot(offset, self.normal))

    # -- what is passable -------------------------------------------------
    def open_intervals(self, raster=None, min_depth: float = 0.6,
                       samples: int = 400):
        """Stretches of the span a boat can actually pass through.

        Derived from the channel raster: open where the water is at least
        ``min_depth`` deep, minus any known piers.  A shell draws well
        under 0.3 m, but a blade needs water on the recovery and a bank is
        not a place to be, so the default is deliberately conservative.
        """
        if self._openings is not None:
            return self._openings
        if raster is None:
            intervals = [(0.0, self.span)]
        else:
            distance = np.linspace(0.0, self.span, samples)
            points = self.point_at(distance)
            if hasattr(raster, "is_navigable"):
                # A ChannelRaster already encodes the depth threshold it
                # was built with, so re-thresholding here would apply two
                # different definitions of navigable to the same course.
                passable = np.array([bool(raster.is_navigable(p[0], p[1]))
                                     for p in points])
            else:
                passable = np.array([float(raster.depth_at(p[0], p[1]))
                                     >= min_depth for p in points])
            intervals = _runs(distance, passable)

        for pier in self.piers:
            intervals = _subtract(intervals, pier.interval)
        self._openings = tuple(intervals)
        return self._openings

    def clearance(self, point, raster=None, min_depth: float = 0.6) -> float:
        """Metres from ``point`` to the nearest edge of its opening.

        Positive inside an opening, negative outside.  This is the quantity
        to constrain: ``clearance >= half_beam + margin``.
        """
        station = self.station_of(point)
        best = -np.inf
        for low, high in self.open_intervals(raster, min_depth):
            if low <= station <= high:
                best = max(best, min(station - low, high - station))
            else:
                best = max(best,
                           -min(abs(station - low), abs(station - high)))
        return float(best)

    def widest_opening(self, raster=None, min_depth: float = 0.6):
        intervals = self.open_intervals(raster, min_depth)
        return max(intervals, key=lambda pair: pair[1] - pair[0])


def _runs(distance, mask):
    """Contiguous ``True`` runs of ``mask``, as distance intervals."""
    intervals = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = distance[i]
        elif not flag and start is not None:
            intervals.append((float(start), float(distance[i - 1])))
            start = None
    if start is not None:
        intervals.append((float(start), float(distance[-1])))
    return intervals


def _subtract(intervals, hole):
    """Remove ``hole`` from a list of intervals."""
    low, high = hole
    result = []
    for a, b in intervals:
        if high <= a or low >= b:
            result.append((a, b))
            continue
        if a < low:
            result.append((a, low))
        if high < b:
            result.append((high, b))
    return result


def build_gates(origin=None, names=None):
    """Every bridge on the reach, projected to the local tangent plane."""
    from .charles import CHARLES_ORIGIN
    from .course import local_tangent_plane

    origin = CHARLES_ORIGIN if origin is None else origin
    gates = []
    for name, ends in OSM_BRIDGE_DECKS.items():
        if names is not None and name not in names:
            continue
        (lat0, lon0), (lat1, lon1) = ends
        east, north = local_tangent_plane(np.array([lat0, lat1]),
                                          np.array([lon0, lon1]), origin)
        gates.append(BridgeGate(name=name,
                                start=np.array([east[0], north[0]]),
                                end=np.array([east[1], north[1]])))
    return gates
