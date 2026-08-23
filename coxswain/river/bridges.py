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
    # The seven bridges of the race course, in upstream (racing) order,
    # are BU, Grand Junction, River Street, Western Avenue, Weeks, Larz
    # Anderson and Eliot.  The first two sit almost on top of each other
    # just above the start and are the reason the opening 500 m is the
    # most congested water on the course.
    "BU Bridge": ((42.3513013, -71.1108090), (42.3534798, -71.1105131)),
    "Grand Junction RR": ((42.3521494, -71.1109938),
                          (42.3530544, -71.1096620)),
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
    start: np.ndarray              # (2,) metres east/north, the Boston shore
    end: np.ndarray                # (2,) metres east/north, the Cambridge shore
    piers: Sequence[Pier] = field(default_factory=tuple)
    structure: Optional["BridgeStructure"] = None
    legal_arches: Tuple[str, ...] = ()
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


# ---------------------------------------------------------------------------
# Structure: how many arches, how wide, and which one you are allowed to use
# ---------------------------------------------------------------------------
#
# The docstring above was written when no source for the arch spans had been
# found, so the gates knew only "somewhere in the wet part".  Sources have
# since been found and the piers below are no longer guesses.
#
# **Span counts and lengths** come from the Federal Highway Administration's
# National Bridge Inventory, 2024 Massachusetts delimited file, matched to
# these structures by position (every match lands within 8 m of the surveyed
# bridge coordinate).  NBI item 45 gives the number of spans in the main
# unit, item 48 the length of the longest, items 39/40 the permitted
# navigation clearances.
#
# **Pier thickness is measured**, not assumed: OpenStreetMap carries five
# ``bridge:support=pier`` polygons for the Grand Junction trestle, and they
# measure 3.14-3.45 m thick, mean 3.32 m, each 18 m long in the flow
# direction.  That is the only direct pier survey on the reach and it is
# what :data:`PIER_THICKNESS` records.
#
# The Weeks Footbridge is a footbridge, so it is absent from the National
# Bridge Inventory; its three arches are documented instead by Simpson
# Gumpertz & Heger, who carried out its restoration, and its deck length
# comes from the same OpenStreetMap survey as the rest.
#
# One independent check says the method works.  Eliot's centre span is
# 33.5 m centre-to-centre; take one pier thickness off and the clear opening
# is 30.2 m, against the 30.5 m navigation clearance NBI states for it.  The
# two numbers come from different columns and agree to 1%.
#
# Where they disagree, they disagree in a way that is worth knowing: NBI
# gives Anderson and Western Avenue the *same* 25.9 m clearance despite
# different span lengths, and gives Anderson a clearance 2.4 m wider than
# its own longest span, which is impossible for a physical opening.  Item 40
# is a permitted channel width, not a measured one.  The physical opening
# derived from the spans is used for geometry; the NBI figure is kept beside
# it as :attr:`BridgeStructure.permitted_width` so the two never get mixed up.

#: Pier thickness across the channel, metres.  Measured; see above.
PIER_THICKNESS = 3.32

#: Bow-to-blade-tip width of a rowed eight, metres -- the width that
#: actually has to fit through an arch.  Oarlock 0.85 m off centre, oar
#: 3.70 m with 1.14 m inboard, so a blade tip sits 3.41 m out and two of
#: them span 6.82 m.  Derived from the rig in :mod:`coxswain.boats.catalog`.
EIGHT_ROWED_WIDTH = 6.82


@dataclass(frozen=True)
class BridgeStructure:
    """What the bridge is made of, as far as the record shows."""

    main_spans: int
    max_span: Optional[float] = None        # NBI 48, centre to centre
    permitted_width: Optional[float] = None  # NBI 40, see the note above
    vertical_clearance: Optional[float] = None   # NBI 39
    year_built: Optional[int] = None
    source: str = ""


#: Structure of every bridge on the racing reach.
BRIDGE_STRUCTURE = {
    "River Street": BridgeStructure(3, 22.9, 21.3, 4.9, 1925, "NBI 2024"),
    "Western Avenue": BridgeStructure(3, 26.8, 25.9, 3.7, 1924, "NBI 2024"),
    "Weeks Footbridge": BridgeStructure(3, None, None, None, 1926, "SGH; OSM deck"),
    "Larz Anderson": BridgeStructure(3, 23.5, 25.9, 3.7, 1912, "NBI 2024"),
    "Eliot Bridge": BridgeStructure(3, 33.5, 30.5, 4.3, 1950, "NBI 2024"),
    "BU Bridge": BridgeStructure(7, 51.8, None, None, 1928, "NBI 2024"),
    "Grand Junction RR": BridgeStructure(6, None, None, None, 1900,
                                         "OSM pier survey"),
}

#: Piers of the Grand Junction railroad trestle, ``(lat, lon)`` of each
#: pier centre, measured from OpenStreetMap ``bridge:support=pier``
#: polygons.  These are surveyed positions, not derived ones.
MEASURED_PIERS = {
    "Grand Junction RR": (
        (42.3523313, -71.1107551),
        (42.3524827, -71.1105298),
        (42.3526290, -71.1103000),
        (42.3527396, -71.1101340),
        (42.3528863, -71.1099136),
    ),
}

#: Which arch a Head of the Charles entry may use, by name.
#:
#: The regatta's rules are written as prohibitions and they are asymmetric.
#: The **left (Boston) arch of every bridge is out of bounds**, and the
#: **right (Cambridge) arch is additionally out of bounds at the BU railroad
#: trestle, Anderson and Eliot**.  Either is a 60 second penalty, on top of
#: any buoy penalty.  Everywhere else the centre arch is the preferred route
#: and the Cambridge arch is available when the centre is congested.
#:
#: ``"centre"`` and ``"cambridge"`` are resolved against the arches actually
#: found in the opening.  BU Bridge is the exception that needs naming
#: rather than counting: it is a seven span bridge whose shore arch is
#: shallow, and the Charles River Rowing Committee's traffic pattern puts
#: upstream traffic through *the second arch from the Cambridge shore*.
HOCR_ARCH_RULE = {
    "BU Bridge": ("second_from_cambridge",),
    "Grand Junction RR": ("centre",),
    "River Street": ("centre", "cambridge"),
    "Western Avenue": ("centre", "cambridge"),
    "Weeks Footbridge": ("centre", "cambridge"),
    "Larz Anderson": ("centre",),
    "Eliot Bridge": ("centre",),
}

#: Penalty for using a prohibited arch, seconds.
WRONG_ARCH_PENALTY = 60.0


@dataclass(frozen=True)
class Arch:
    """One opening under a bridge."""

    index: int              # 0 is the Boston shore arch
    centre: float           # metres along the span from the Boston end
    width: float            # clear opening between pier faces
    legal: bool             # may a racing crew use it
    label: str = ""

    @property
    def interval(self) -> Tuple[float, float]:
        return (self.centre - 0.5 * self.width, self.centre + 0.5 * self.width)

    def fits(self, boat_width: float = EIGHT_ROWED_WIDTH) -> float:
        """How many boats of ``boat_width`` fit abreast in this arch."""
        return self.width / float(boat_width)
