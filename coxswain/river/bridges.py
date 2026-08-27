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

The **piers are now known too**, which they were not when this module was
first written.  That earlier version said no published source for the arch
spans had been found and left :attr:`BridgeGate.piers` empty "for when
survey data turns up"; it has since turned up, from two directions.  The
Federal Highway Administration's National Bridge Inventory carries the span
count and span lengths of every road bridge on the reach, and OpenStreetMap
carries surveyed pier polygons for the Grand Junction trestle.  See the
note above :data:`PIER_THICKNESS` for what each source does and does not
support.

The distinction the old text drew is still the right one, it has just
moved: the trestle's piers are *measured*, everything else's are *derived*
from measured span lengths under a stated symmetry assumption.  Anything
that turns out to depend on which of those it is should say so.

The **water is the other half of the constraint**.  An arch is open only
where it also crosses water deep enough to row, taken from the same
channel raster as the rest of the model, so shallow shore arches close
themselves without needing to be listed.

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

__all__ = ["Pier", "BridgeGate", "OSM_BRIDGE_DECKS", "build_gates",
           "Arch", "BridgeStructure", "BRIDGE_STRUCTURE", "MEASURED_PIERS",
           "PIER_THICKNESS", "EIGHT_ROWED_WIDTH", "HOCR_ARCH_RULE",
           "WRONG_ARCH_PENALTY", "waterway", "derive_piers", "bridge_arches",
           "candidate_arches", "racing_arch", "crossing_angle"]


#: Bridge deck centrelines from OpenStreetMap, as ``(lat, lon)`` endpoints
#: of the span crossing the river.  Queried from the Overpass API over the
#: racing reach; see the module docstring for provenance.
#:
#: **Ordered (Boston end, Cambridge end)** at every bridge, which is what
#: makes arch numbering mean the same thing everywhere and lets the
#: regatta's left/right rules be written down once.
#:
#: This ordering is stated rather than inferred, and that is deliberate.
#: Working it out from the river -- Cambridge is the bank to starboard of a
#: crew rowing up the course -- is right in principle and fails in practice
#: at Eliot, because the course turns through the big northward loop there
#: and the centreline's local heading at the bridge is not the direction of
#: travel through it.  No choice of baseline fixes that: short baselines
#: pick up the wiggle, long ones pick up the loop, and the two disagree by
#: more than ninety degrees.  Latitude alone will not do it either, since
#: Western Avenue crosses close enough to east-west that its ends differ by
#: 15 m in latitude and 150 m in longitude.  Which bank is Cambridge is a
#: fact about the river, so it is recorded as one.
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
    # Listed Boston end first like the rest, which for this one means the
    # western end: Western Avenue crosses almost east-west, so its banks
    # are east and west rather than north and south.
    "Western Avenue": ((42.3641615, -71.1181022), (42.3642964, -71.1162617)),
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


def build_gates(origin=None, names=None, channel=None, piers=True):
    """Every bridge on the reach, projected to the local tangent plane.

    Gates come back oriented so that ``start`` is the **Boston** shore and
    ``end`` is the **Cambridge** shore, taken from the order the endpoints
    are listed in :data:`OSM_BRIDGE_DECKS`; see the note there for why that
    ordering is recorded rather than worked out from the river.

    ``channel`` is the raster the piers and openings are read against; it
    defaults to the cached Charles channel.  Pass ``piers=False`` for the
    bare deck lines with nothing subtracted.
    """
    from .charles import CHARLES_ORIGIN, charles_channel
    from .course import local_tangent_plane

    origin = CHARLES_ORIGIN if origin is None else origin
    if channel is None and piers:
        channel = charles_channel(origin)

    centreline = None if channel is None else channel.centreline()

    gates = []
    for name, ends in OSM_BRIDGE_DECKS.items():
        if names is not None and name not in names:
            continue
        (lat0, lon0), (lat1, lon1) = ends
        east, north = local_tangent_plane(np.array([lat0, lat1]),
                                          np.array([lon0, lon1]), origin)
        a = np.array([east[0], north[0]])
        b = np.array([east[1], north[1]])
        gate = BridgeGate(name=name, start=a, end=b,
                          structure=BRIDGE_STRUCTURE.get(name),
                          legal_arches=HOCR_ARCH_RULE.get(name, ()))
        if piers and channel is not None:
            gate.piers = derive_piers(gate, channel)
        gates.append(gate)

    if centreline is not None:
        gates.sort(key=lambda g: -_station_on(0.5 * (g.start + g.end),
                                              centreline))
    return gates


def _station_on(point, centreline) -> float:
    """Distance along ``centreline`` of the point nearest ``point``."""
    index = int(np.argmin(np.linalg.norm(centreline - point, axis=1)))
    step = np.linalg.norm(np.diff(centreline[:index + 1], axis=0), axis=1)
    return float(step.sum())


#: Baseline for the river tangent used to tell the two banks apart, metres.
#:
#: A tangent taken over a couple of centreline points is far too short at
#: Eliot: the course turns through the big northward loop there, and a
#: short baseline picks up the curve instead of the direction of travel,
#: which put Eliot's Cambridge end on the Boston bank while the other six
#: bridges came out right.  Fifty metres is long against the bend and
#: short against the reach.
_TANGENT_BASELINE = 50.0


def starboard_end_is_first(a, b, centreline) -> bool:
    """Is ``a`` the end to starboard of a crew rowing up the course?

    Kept as a **cross-check on** :data:`OSM_BRIDGE_DECKS`, not as the way
    the gates are built.  Cambridge is the starboard bank, so this should
    agree with the recorded ordering at every bridge where the river runs
    straight enough through the crossing for a local tangent to mean
    anything -- which is all of them except Eliot.
    """
    middle = 0.5 * (a + b)
    index = int(np.argmin(np.linalg.norm(centreline - middle, axis=1)))
    step = np.linalg.norm(np.diff(centreline, axis=0), axis=1).mean()
    reach = max(int(round(_TANGENT_BASELINE / max(step, 1e-6))), 2)
    lo = max(index - reach, 0)
    hi = min(index + reach, len(centreline) - 1)
    downstream = centreline[hi] - centreline[lo]
    upstream = -downstream / max(np.linalg.norm(downstream), 1e-9)
    starboard = np.array([upstream[1], -upstream[0]])
    return float(np.dot(a - middle, starboard)) > 0.0


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
    #: NBI 49, the whole structure end to end.  Used to stop the opening
    #: running past the abutments: the depth raster only knows about water
    #: and will happily report it where a bridge's wing wall stands, which
    #: at River Street made the waterway come out 78 m wide against a 64 m
    #: bridge and turned each abutment into a phantom 26 m shore arch.
    structure_length: Optional[float] = None


#: Structure of every bridge on the racing reach.
BRIDGE_STRUCTURE = {
    "River Street": BridgeStructure(
        3, 22.9, 21.3, 4.9, 1925, "NBI 2024", structure_length=64.0),
    "Western Avenue": BridgeStructure(
        3, 26.8, 25.9, 3.7, 1924, "NBI 2024", structure_length=85.3),
    "Weeks Footbridge": BridgeStructure(
        3, None, None, None, 1926, "SGH; OSM deck"),
    "Larz Anderson": BridgeStructure(
        3, 23.5, 25.9, 3.7, 1912, "NBI 2024", structure_length=70.7),
    "Eliot Bridge": BridgeStructure(
        3, 33.5, 30.5, 4.3, 1950, "NBI 2024", structure_length=112.2),
    "BU Bridge": BridgeStructure(
        7, 51.8, None, None, 1928, "NBI 2024", structure_length=220.7),
    "Grand Junction RR": BridgeStructure(
        6, None, None, None, 1900, "OSM pier survey"),
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
    """One opening under a bridge.

    Two widths, because a skewed bridge has two.  :attr:`width` is the
    structural opening, measured along the deck between pier faces.
    :attr:`effective_width` is what a boat rowing along the river actually
    has to fit through, which is smaller whenever the bridge does not
    cross square.
    """

    index: int              # 0 is the Boston shore arch
    centre: float           # metres along the span from the Boston end
    width: float            # clear opening between pier faces, along the deck
    legal: bool             # may a racing crew use it
    label: str = ""
    #: Sine of the angle between the deck and the river, so 1.0 for a
    #: square crossing.  See :func:`crossing_angle`.
    squareness: float = 1.0

    @property
    def interval(self) -> Tuple[float, float]:
        return (self.centre - 0.5 * self.width, self.centre + 0.5 * self.width)

    @property
    def effective_width(self) -> float:
        """Clear width across the boat's path rather than across the deck.

        A corridor heading down the river between two pier faces ``w``
        apart along a deck meeting the river at angle ``phi`` is only
        ``w sin(phi)`` wide.  Square crossings are unaffected; the Grand
        Junction trestle crosses at 41 degrees off square and loses a
        quarter of every opening.
        """
        return self.width * float(self.squareness)

    def fits(self, boat_width: float = EIGHT_ROWED_WIDTH) -> float:
        """How many boats of ``boat_width`` fit abreast in this arch.

        Uses :attr:`effective_width`, since that is the one the boat meets.
        """
        return self.effective_width / float(boat_width)


def waterway(gate: BridgeGate, raster, min_depth: float = 0.6,
             samples: int = 400) -> Tuple[float, float]:
    """Cached; see :func:`_waterway`."""
    cache = getattr(gate, "_waterway_cache", None)
    if cache is None:
        cache = gate._waterway_cache = {}
    key = (id(raster), float(min_depth), int(samples))
    if key not in cache:
        cache[key] = _waterway(gate, raster, min_depth, samples)
    return cache[key]


def _waterway(gate: BridgeGate, raster, min_depth: float = 0.6,
              samples: int = 400) -> Tuple[float, float]:
    """The wet part of the span, before any pier is taken out of it.

    This is what the arches have to be laid out inside, so it has to be
    measured before the piers exist rather than after.
    """
    distance = np.linspace(0.0, gate.span, samples)
    points = gate.point_at(distance)
    if hasattr(raster, "is_navigable"):
        wet = np.array([bool(raster.is_navigable(p[0], p[1])) for p in points])
    else:
        wet = np.array([float(raster.depth_at(p[0], p[1])) >= min_depth
                        for p in points])
    runs = _runs(distance, wet)
    if not runs:
        return (0.0, gate.span)
    low, high = max(runs, key=lambda pair: pair[1] - pair[0])

    # A bridge cannot open wider than it is long.  Where the raster says
    # water past the abutments, trust the bridge.
    structure = gate.structure or BRIDGE_STRUCTURE.get(gate.name)
    limit = None if structure is None else structure.structure_length
    if limit is not None and (high - low) > limit:
        middle = 0.5 * (low + high)
        low, high = middle - 0.5 * limit, middle + 0.5 * limit
    return (low, high)


def derive_piers(gate: BridgeGate, raster, min_depth: float = 0.6):
    """Piers of one bridge, as keep-out intervals along its span.

    Surveyed positions are used where they exist -- only the Grand Junction
    trestle has them.  Everywhere else the piers are placed from the span
    lengths in the National Bridge Inventory: the centre span is laid
    symmetrically about the middle of the wet opening and the piers sit at
    its ends, one :data:`PIER_THICKNESS` wide.

    That construction assumes the bridge is symmetric about the channel,
    which is what a three arch bridge over a single channel normally is,
    and which the near-equal side spans in the inventory support --
    Anderson's two side spans come out at 23.6 m against a 23.5 m centre.
    """
    measured = MEASURED_PIERS.get(gate.name)
    if measured is not None:
        from .charles import CHARLES_ORIGIN
        from .course import local_tangent_plane
        lats = np.array([p[0] for p in measured])
        lons = np.array([p[1] for p in measured])
        east, north = local_tangent_plane(lats, lons, CHARLES_ORIGIN)
        return tuple(sorted(
            (Pier(gate.station_of(np.array([e, n])), PIER_THICKNESS)
             for e, n in zip(east, north)),
            key=lambda p: p.centre))

    structure = gate.structure or BRIDGE_STRUCTURE.get(gate.name)
    if structure is None:
        return ()

    low, high = waterway(gate, raster, min_depth)
    middle = 0.5 * (low + high)

    if structure.max_span is not None:
        half = 0.5 * float(structure.max_span)
        return (Pier(middle - half, PIER_THICKNESS),
                Pier(middle + half, PIER_THICKNESS))

    # No inventory record -- the Weeks footbridge is the case, since a
    # footbridge is not in the National Bridge Inventory at all.  Its arch
    # count is documented even though its spans are not, so divide the
    # opening into that many equal arches.  Equal spacing is what the three
    # bridges with measured spans come out at anyway: Anderson's side spans
    # are 23.6 m against a 23.5 m centre.
    n = int(structure.main_spans)
    if n < 2:
        return ()
    pitch = (high - low) / n
    return tuple(Pier(low + pitch * (i + 1), PIER_THICKNESS)
                 for i in range(n - 1))


def bridge_arches(gate: BridgeGate, raster, min_depth: float = 0.6):
    """Cached; see :func:`_bridge_arches` for the computation.

    A gate's arches depend only on the gate and the raster, neither of
    which moves during an optimisation, so computing them per candidate
    line was pure repetition.
    """
    cache = getattr(gate, "_arch_cache", None)
    if cache is None:
        cache = gate._arch_cache = {}
    key = (id(raster), float(min_depth))
    if key not in cache:
        cache[key] = _bridge_arches(gate, raster, min_depth)
    return cache[key]


def _bridge_arches(gate: BridgeGate, raster, min_depth: float = 0.6):
    """Every opening under ``gate``, numbered from the Boston shore.

    Openings narrower than a rowed eight are dropped: an arch a boat
    cannot fit through is not an arch as far as this model is concerned,
    and keeping them would shift the numbering the rules depend on.
    """
    low, high = waterway(gate, raster, min_depth)
    intervals = []
    for a, b in gate.open_intervals(raster, min_depth):
        a, b = max(a, low), min(b, high)
        if b - a >= EIGHT_ROWED_WIDTH:
            intervals.append((a, b))
    if not intervals:
        return ()

    legal = _resolve_arch_rule(gate.legal_arches or
                               HOCR_ARCH_RULE.get(gate.name, ()),
                               intervals)
    squareness = crossing_angle(gate, raster)
    arches = []
    for i, (low, high) in enumerate(intervals):
        if i == 0:
            label = "Boston shore"
        elif i == len(intervals) - 1:
            label = "Cambridge shore"
        else:
            label = "centre" if len(intervals) == 3 else "arch %d" % (i + 1)
        arches.append(Arch(index=i, centre=0.5 * (low + high),
                           width=high - low, legal=i in legal, label=label,
                           squareness=squareness))
    return tuple(arches)


def crossing_angle(gate: BridgeGate, raster) -> float:
    """How square the bridge meets the river: ``sin`` of the angle between.

    ``1.0`` is a square crossing.  Six of the seven bridges on the reach
    are within 20 degrees of square and one is not: the Grand Junction
    trestle carries the railway diagonally across the river at 47 degrees
    while the river runs at 96, a **41 degree skew**, so every opening
    under it is a quarter narrower to a boat than it is to the bridge.

    Returns ``1.0`` when there is no channel to measure against, which
    leaves the structural width unchanged rather than inventing a skew.
    """
    if raster is None or not hasattr(raster, "centreline"):
        return 1.0
    centreline = raster.centreline()
    middle = 0.5 * (gate.start + gate.end)
    index = int(np.argmin(np.linalg.norm(centreline - middle, axis=1)))
    low = max(index - 8, 0)
    high = min(index + 8, len(centreline) - 1)
    along = centreline[high] - centreline[low]
    norm = float(np.linalg.norm(along))
    if norm < 1e-9:
        return 1.0
    along = along / norm
    # |sin| of the angle between the deck and the river is the magnitude of
    # their 2-D cross product, both being unit vectors.
    deck = gate.direction
    return float(abs(deck[0] * along[1] - deck[1] * along[0]))


def _resolve_arch_rule(names, intervals):
    """Turn rule names into arch indices for the openings actually found.

    ``"centre"`` is resolved as the arch nearest the middle of the whole
    opening rather than by counting, so it still means something when a
    bridge has an even number of arches, as the trestle does.
    """
    if not intervals:
        return set()
    n = len(intervals)
    middle = 0.5 * (intervals[0][0] + intervals[-1][1])
    nearest_middle = min(range(n),
                         key=lambda i: abs(0.5 * (intervals[i][0]
                                                  + intervals[i][1]) - middle))
    chosen = set()
    for name in names:
        if name == "centre":
            chosen.add(nearest_middle)
        elif name == "cambridge":
            chosen.add(n - 1)
        elif name == "boston":
            chosen.add(0)
        elif name == "second_from_cambridge":
            chosen.add(max(n - 2, 0))
        else:
            raise ValueError("unknown arch rule %r" % (name,))
    return chosen


def candidate_arches(gate: BridgeGate, raster, min_depth: float = 0.6):
    """Every arch a racing crew is *allowed* to use.

    This, not :func:`racing_arch`, is what a route search or an optimiser
    should be given.  An arch is only removed from the set when the rules
    remove it, and at River Street and Western Avenue the rules remove
    only the Boston arch -- the Cambridge arch stays in, and it is the
    wider of the two openings at both bridges (25.9 m against 19.6 m at
    River Street, 27.0 m against 23.5 m at Western Avenue).  Whether the
    line through it is quicker depends on where it puts the boat for the
    next bend, which is a question for the trajectory solver and not one to
    be settled in advance by this function.
    """
    return tuple(a for a in bridge_arches(gate, raster, min_depth) if a.legal)


def racing_arch(gate: BridgeGate, raster, min_depth: float = 0.6):
    """The conventional line: the centre arch where the rules allow one.

    A **default for drawing and reporting, not a constraint.**  The centre
    arch is the regatta's stated preferred route, but it is not always the
    widest and it is not always the fastest -- see
    :func:`candidate_arches`, which is the function to use when the
    question is where the boat is permitted to go.  Falls back to the
    widest legal arch where no centre arch is permitted.
    """
    arches = bridge_arches(gate, raster, min_depth)
    legal = [a for a in arches if a.legal]
    if not legal:
        return None
    centre = _resolve_arch_rule(("centre",), [a.interval for a in arches])
    for arch in legal:
        if arch.index in centre:
            return arch
    return max(legal, key=lambda a: a.width)
