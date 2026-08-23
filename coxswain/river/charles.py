"""The Charles River: surveyed bathymetry and a calibrated flow model.

This replaces the placeholder sketch in :mod:`coxswain.river.course` with
real data.

Bathymetry
----------
Charles River Alliance of Boaters and MIT Sea Grant, 2016-17 sonar survey
of the Lower Charles from the New Charles River Dam to the Watertown Dam
(~14.5 km), the first detailed chart of the river since 1902.  Surveyed
with a Lowrance HDS-7 broadband sonar and Point-1 GPS on track lines 9-18 m
apart, processed in ReefMaster and corrected for transducer depth.

    C. Zimba, M. J. Sacarny, M. Yoder, B. Bray and C. Chryssostomidis,
    *Changes in the Depth of the Lower Charles River Basin*, CRAB / MIT Sea
    Grant (2018).
    Chart: http://www.charlesriverallianceofboaters.org/chart/charles.kmz

``data/charles_isobaths.csv`` holds the 1-foot contour vertices extracted
from that KMZ, 0.30 m to 10.36 m, converted to metres.  Depths are below
the basin's normal pool, which the New Charles River Dam holds nearly
constant -- so they are already depth below the surface the boat sits on,
with no tidal reduction needed.  That is a genuine simplification the
Charles allows and a tidal estuary would not.

Flow
----
The lower Charles is an **impoundment**, not a free-flowing river: the New
Charles River Dam sets the level and the water is close to slack.  Flow
speed therefore comes from continuity rather than from a slope-driven
resistance law --

    U(s) = Q / A(s)

for discharge ``Q`` and wetted cross-sectional area ``A(s)`` at station
``s``, with ``A`` integrated from the surveyed bathymetry.  This is the
mechanism CRAB themselves invoke for the shoaling areas: "As water over a
given cross section becomes shallower, water flow velocity must increase."

``Q`` comes from USGS 01104500 CHARLES RIVER AT WALTHAM, the long-record
gauge immediately above the reach, condensed to monthly statistics over
1931-2026 in ``data/charles_discharge_waltham.csv``.  Waltham is above the
Watertown Dam, so it misses the small ungauged inflow between there and
the basin; for the Charles that is a few percent, and it is the best
available proxy.

Manning's equation is deliberately **not** used.  It needs an energy slope,
and the slope across an impounded basin is neither measured here nor
meaningfully constant.  Continuity needs only discharge and geometry, both
of which are measured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from .course import Course, CurrentField, DepthField, local_tangent_plane

__all__ = [
    "thalweg",
    "CHARLES_ORIGIN",
    "WATERTOWN_DAM",
    "ELIOT_BRIDGE",
    "BU_BRIDGE",
    "DISCHARGE_GAUGE",
    "isobath_path",
    "discharge_path",
    "load_isobaths",
    "load_discharge",
    "monthly_discharge",
    "charles_depth_field",
    "ContinuityFlow",
    "charles_course",
    "landmark_station",
    "test_section",
    "BRIDGES",
    "WEEKS_FOOTBRIDGE",
    "LARZ_ANDERSON_BRIDGE",
]

#: Tangent-plane origin: roughly the middle of the surveyed reach.
CHARLES_ORIGIN = (42.3625, -71.1200)

#: Landmarks along the rowing reach, ``(latitude, longitude)``.
#:
#: Listed downstream to upstream, which is the order a Head of the Charles
#: entry meets them: the course runs *up* the river from below BU Bridge to
#: past Eliot.
#:
#: **Corrected against published bridge coordinates.**  Three of these were
#: wrong, and the error showed up as a large distance between the landmark
#: and the extracted channel centreline -- Eliot was 361 m off, Western
#: Avenue 167 m, BU 136 m.  The channel was not at fault: replacing the
#: coordinates with the surveyed positions brings every bridge onto the
#: centreline.
#:
#: ==================  ========  =========  =========  =========
#: bridge              was       now        gap        sinuosity
#: ==================  ========  =========  =========  =========
#: BU Bridge           136 m     **30 m**
#: River Street        6 m       6 m        1281 m     1.16
#: Western Avenue      167 m     **3 m**    336 m      1.01
#: Weeks Footbridge    6 m*      **15 m**   498 m      1.02
#: Anderson Memorial   21 m      **5 m**    426 m      1.01
#: Eliot Bridge        361 m     **6 m**    1268 m     1.49
#: ==================  ========  =========  =========  =========
#:
#: \* Weeks was the subtle one.  Its old coordinate sat 6 m from the
#: centreline, so the offset check passed -- but 370 m too far *upstream*,
#: almost on top of Anderson, leaving only 259 m between two bridges that
#: are really 426 m apart.  **A small offset proves a landmark is in the
#: channel, not that it is at the right place along it**; the gap and
#: sinuosity columns are what catch a landmark that has slid along the
#: river.  Both are now checked.
#:
#: Sinuosity is channel distance over straight-line distance to the
#: previous bridge; 1.49 through the Anderson-to-Eliot bend is the large
#: northward loop and everything else is nearly straight, which is what
#: the reach looks like.
WATERTOWN_DAM = (42.36482, -71.18978)
ELIOT_BRIDGE = (42.37180, -71.13280)
LARZ_ANDERSON_BRIDGE = (42.36890, -71.12320)
WEEKS_FOOTBRIDGE = (42.36853, -71.11807)
WESTERN_AVE_BRIDGE = (42.36422, -71.11690)
RIVER_ST_BRIDGE = (42.36123, -71.11670)
BU_BRIDGE = (42.35238, -71.11066)

#: Boston University's DeWolfe Boathouse, at the foot of the BU Bridge.
#: The Head of the Charles start line lies just off its front, about 160 m
#: downstream of the bridge -- the race does **not** start at the bridge.
DEWOLFE_BOATHOUSE = (42.35420, -71.10850)

#: Official Head of the Charles course length, metres.
HOCR_COURSE_LENGTH = 4828.0

#: Every bridge on the racing reach, in upstream order.
BRIDGES = (
    ("BU Bridge", BU_BRIDGE),
    ("River Street", RIVER_ST_BRIDGE),
    ("Western Avenue", WESTERN_AVE_BRIDGE),
    ("Weeks Footbridge", WEEKS_FOOTBRIDGE),
    ("Anderson Memorial", LARZ_ANDERSON_BRIDGE),
    ("Eliot Bridge", ELIOT_BRIDGE),
)

#: The gauge the flow model is driven from.
DISCHARGE_GAUGE = {
    "site": "01104500",
    "name": "CHARLES RIVER AT WALTHAM, MA",
    "latitude": 42.37231857,
    "longitude": -71.2336668,
    "period_of_record": "1931-2026",
    "url": "https://waterdata.usgs.gov/monitoring-location/USGS-01104500/",
}


def _data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, os.pardir, os.pardir, "data")


def isobath_path() -> str:
    return os.path.normpath(os.path.join(_data_dir(),
                                         "charles_isobaths.csv"))


def discharge_path() -> str:
    return os.path.normpath(os.path.join(_data_dir(),
                                         "charles_discharge_waltham.csv"))


@lru_cache(maxsize=2)
def load_isobaths(origin: Tuple[float, float] = CHARLES_ORIGIN):
    """Surveyed depth soundings, projected to the local tangent plane.

    Returns ``(points, depths)`` with ``points`` of shape ``(n, 2)`` in
    metres east/north of ``origin`` and ``depths`` in metres.
    """
    path = isobath_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Regenerate it from the CRAB chart with "
            "_extract_bathy.py, or see this module's docstring for the "
            "source URL."
        )
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    east, north = local_tangent_plane(raw[:, 1], raw[:, 0], origin)
    return np.column_stack([east, north]), raw[:, 2]


@lru_cache(maxsize=1)
def load_discharge() -> np.ndarray:
    """Monthly discharge statistics, shape ``(12, 6)`` in m3/s.

    Columns are mean, p10, median, p90, min, max; rows are months 1-12.
    """
    path = discharge_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("month"):
            continue
        rows.append([float(v) for v in line.split(",")])
    return np.array(rows)[:, 1:]


def monthly_discharge(month: int, statistic: str = "median") -> float:
    """Discharge at Waltham for a calendar month, in m3/s.

    ``statistic`` is one of ``mean``, ``p10``, ``median``, ``p90``, ``min``,
    ``max``.  The Head of the Charles is rowed in October, whose median is
    2.8 m3/s against a March median of 15.3 -- a factor of five, which is
    why quoting a single "the current on the Charles" figure is useless.
    """
    order = ("mean", "p10", "median", "p90", "min", "max")
    if statistic not in order:
        raise ValueError(f"statistic must be one of {order}")
    if not 1 <= month <= 12:
        raise ValueError("month must be 1-12")
    return float(load_discharge()[month - 1, order.index(statistic)])


def charles_depth_field(origin: Tuple[float, float] = CHARLES_ORIGIN,
                        minimum: float = 0.5) -> DepthField:
    """A :class:`DepthField` built from the surveyed isobaths."""
    points, depths = load_isobaths(origin)
    return DepthField(points=points, depths=depths, minimum=minimum,
                      is_survey=True)


@dataclass
class ContinuityFlow:
    """Depth-averaged flow speed from continuity, ``U = Q / A``.

    The cross-sectional area at each station is integrated from the
    surveyed depth field across the channel, so the model has exactly two
    inputs -- measured bathymetry and a measured discharge -- and no fitted
    parameters.

    The result for the Charles is worth stating plainly: at typical
    discharge the basin is **nearly slack**.  October's median 2.8 m3/s
    over a cross-section of several hundred square metres gives millimetres
    per second.  The current only becomes worth modelling in flood, and
    then it concentrates wherever the channel is narrow or shoaled --
    which is precisely where CRAB report the sedimentation problems.
    """

    course: "Course"
    discharge: float = 2.84          # m3/s, October median at Waltham
    n_transect: int = 41             # samples across the channel
    #: Exponent relating local depth-averaged velocity to depth in the
    #: lateral distribution.  2/3 is Manning; 1/2 would be Chezy.  Only the
    #: *shape* depends on it -- the magnitude is fixed by continuity.
    velocity_exponent: float = 2.0 / 3.0
    _area_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def cross_section_area(self, station) -> np.ndarray:
        """Wetted area at one or more stations, in m^2.

        Integrates depth across the navigable width by the trapezium rule.
        """
        station = np.atleast_1d(np.asarray(station, dtype=float))
        areas = np.empty(station.shape)
        for i, s in enumerate(station):
            half = float(self.course.water_half_width_at(s))
            offsets = np.linspace(-half, half, self.n_transect)
            points = self.course.offset_position(
                np.full(self.n_transect, s), offsets)
            depths = self.course.depth(points[:, 0], points[:, 1])
            areas[i] = np.trapezoid(depths, offsets)
        return areas

    def speed(self, station) -> np.ndarray:
        """Depth-averaged flow speed at a station, in m/s."""
        area = self.cross_section_area(station)
        return self.discharge / np.maximum(area, 1e-6)

    def profile(self, n: int = 60):
        """``(station, area, speed)`` along the course."""
        station = np.linspace(0.0, self.course.length, n)
        area = self.cross_section_area(station)
        return station, area, self.discharge / np.maximum(area, 1e-6)

    def lateral_profile(self, station: float):
        """Flow speed across one cross-section: ``(offsets, depth, speed)``.

        The section-mean speed ``Q/A`` is not what a boat feels.  Water
        moves fastest in the deep channel and slowest over the shoals, and
        that lateral spread is the entire reason a line choice exists:
        going upstream a crew wants the slack water near the bank, coming
        down they want the thread of the current.

        The distribution follows the standard conveyance argument.  Locally,
        Manning gives ``u = h^(2/3) S^(1/2) / n``, so across a wide section

            u(y) = Q h(y)^(2/3) / integral h(y)^(5/3) dy

        The slope ``S`` and roughness ``n`` **cancel**.  That is what makes
        this usable here: the *shape* of the distribution needs only the
        surveyed bathymetry, and the *magnitude* is still pinned by the
        measured discharge.  Manning is used for the lateral shape, where
        its unknowns drop out, and still not for the magnitude, where they
        would not.

        By construction ``integral u h dy == Q`` exactly, so this refines
        :meth:`speed` without contradicting it.
        """
        # integrate over the WATER width, not the navigable one
        half = float(self.course.water_half_width_at(station))
        offsets = np.linspace(-half, half, self.n_transect)
        points = self.course.offset_position(
            np.full(self.n_transect, float(station)), offsets)
        depth = np.asarray(self.course.depth(points[:, 0], points[:, 1]),
                           dtype=float)

        conveyance = np.trapezoid(depth ** (1.0 + self.velocity_exponent),
                                  offsets)
        if conveyance <= 0.0:
            return offsets, depth, np.zeros_like(depth)
        speed = self.discharge * depth ** self.velocity_exponent / conveyance
        return offsets, depth, speed

    def _speed_grid(self, n_station: int):
        """Precompute speed on a (station, offset-fraction) grid.

        The field is evaluated every derivative call, so rebuilding a
        cross-section each time would dominate the run.  Offsets are stored
        as a fraction of the local half-width so the grid stays rectangular
        even where the channel narrows.
        """
        stations = np.linspace(0.0, self.course.length, n_station)
        fractions = np.linspace(-1.0, 1.0, self.n_transect)
        grid = np.empty((n_station, self.n_transect))
        for i, s in enumerate(stations):
            _, _, speed = self.lateral_profile(s)
            grid[i] = speed
        return stations, fractions, grid

    def as_current_field(self, n: int = 80,
                         lateral: bool = True) -> CurrentField:
        """A :class:`CurrentField` pointing downstream at the local speed.

        Downstream is towards decreasing station, since the course is laid
        out bow-first up the river the way a crew rows it.

        With ``lateral=True`` (the default) the speed varies **across** the
        channel as well as along it, from :meth:`lateral_profile`.  Setting
        it ``False`` falls back to the section mean ``Q/A`` everywhere,
        which is what a route optimiser would see as a river with no line
        in it.
        """
        course = self.course
        stations, fractions, grid = self._speed_grid(n)
        mean_speed = self.speed(stations)

        def flow(x, y):
            s = course.nearest_station(float(x), float(y))
            heading = float(course.heading_at(np.array(s)))
            if lateral:
                centre = course.position_at(np.array(s))
                # signed offset: positive to port of the centreline
                normal = np.array([-np.sin(heading), np.cos(heading)])
                offset = float(np.dot([x - centre[0], y - centre[1]], normal))
                half = max(float(course.half_width_at(s)), 1e-6)
                row = np.array([np.interp(s, stations, grid[:, j])
                                for j in range(grid.shape[1])])
                magnitude = float(np.interp(np.clip(offset / half, -1.0, 1.0),
                                            fractions, row))
            else:
                magnitude = float(np.interp(s, stations, mean_speed))
            # water flows towards the start of the course (downstream)
            return (-magnitude * np.cos(heading),
                    -magnitude * np.sin(heading))

        return CurrentField(function=flow)


def thalweg(origin: Tuple[float, float] = CHARLES_ORIGIN,
            depth: DepthField = None, n_bins: int = 46,
            n_probe: int = 90, smooth: int = 3) -> np.ndarray:
    """The deep channel, traced through the surveyed bathymetry.

    Successive east-west bins are probed across their north extent and the
    deepest position in each is kept, then the result is smoothed.  This
    follows the navigable channel rather than the geometric middle of the
    water, which matters here: the Charles has shoals well inside its
    banks, and a centreline drawn down the middle runs over the Magazine
    Beach and Sunset Bay deposits CRAB document.

    It is still **not** a surveyed navigation channel or a race line -- it
    is the deepest water, which is a defensible default and nothing more.
    """
    depth = charles_depth_field(origin) if depth is None else depth
    points, _ = load_isobaths(origin)
    east = points[:, 0]
    edges = np.linspace(east.min(), east.max(), n_bins + 1)

    spine = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = points[(east >= lo) & (east < hi)]
        if len(band) < 30:
            continue
        centre_east = 0.5 * (lo + hi)
        north = np.linspace(np.percentile(band[:, 1], 2),
                            np.percentile(band[:, 1], 98), n_probe)
        probe = depth(np.full(n_probe, centre_east), north)
        spine.append([centre_east, float(north[int(np.argmax(probe))])])

    spine = np.array(spine)
    if smooth > 1 and len(spine) > 2 * smooth:
        # Smoothing the north positions alone walks the line *out* of the
        # channel wherever the river bends -- averaging across a bend cuts
        # the corner onto the bank.  Measured: it halved the median depth
        # along the spine, from 3.49 m to 1.77 m.  So smooth for continuity,
        # then re-snap each point to the deepest water near it.
        kernel = np.ones(smooth) / smooth
        smoothed = np.convolve(spine[:, 1], kernel, mode="same")
        smoothed[:smooth] = spine[:smooth, 1]
        smoothed[-smooth:] = spine[-smooth:, 1]

        snap = 0.6 * np.abs(np.diff(spine[:, 1])).max() + 25.0
        for i, (x, y) in enumerate(zip(spine[:, 0], smoothed)):
            north = np.linspace(y - snap, y + snap, 41)
            probe = depth(np.full(north.shape, x), north)
            spine[i, 1] = north[int(np.argmax(probe))]

    return spine


def refine_thalweg(spine: np.ndarray, depth: DepthField,
                   passes: int = 4, reach: float = 70.0,
                   n_probe: int = 61, smooth: int = 3) -> np.ndarray:
    """Re-snap a channel spine to the deepest water on the local normal.

    **Not used by default: measured worse than the plain trace.**

    The reasoning for trying it was sound.  :func:`thalweg` bins east-west
    and probes north, which only searches *across* the channel where the
    river runs east-west; through the north-south meanders the probe runs
    along the channel instead and can return a point that is not in deep
    water at all.  Probing the local perpendicular should fix that.

    It does not.  Fraction of the spine in water shallower than the given
    depth, on the CRAB survey:

    ==========  ==============  ==================
    threshold   plain trace     perpendicular pass
    ==========  ==============  ==================
    0.6 m       13.0%           7.0%
    1.0 m       24.0%           28.3%
    1.2 m       26.3%           35.3%
    2.0 m       34.7%           43.7%
    ==========  ==============  ==================

    It clears the worst pinning at the 0.5 m floor and loses everywhere
    else, and the mean depth along the line drops from 3.39 m to 3.09 m.
    The likely reason is that the perpendicular estimated from a coarse,
    smoothed spine is itself unreliable through the bends, so the probe
    still often runs partly along the channel -- and now it also drags the
    line laterally where the plain trace at least stayed put.

    Kept because the approach is right in principle and would work given a
    better initial spine or a finer survey; the ordering matters too, and
    is recorded here: smoothing *after* snapping ends a pass with the line
    walked out of the channel, which measured 44% under 1.2 m.
    """
    spine = np.asarray(spine, dtype=float).copy()
    if len(spine) < 3:
        return spine

    step = np.linspace(-reach, reach, n_probe)

    for _ in range(passes):
        # Smooth FIRST, snap SECOND.  Smoothing averages across bends and
        # so cuts corners onto the bank; ending a pass on a smoothing step
        # leaves the line there.  Doing it the other way round -- snapping
        # last -- measured 44% of the spine in water under 1.2 m against
        # 15% for this order, on the same survey.
        if smooth > 1 and len(spine) > 2 * smooth:
            kernel = np.ones(smooth) / smooth
            for axis in (0, 1):
                blurred = np.convolve(spine[:, axis], kernel, mode="same")
                blurred[:smooth] = spine[:smooth, axis]
                blurred[-smooth:] = spine[-smooth:, axis]
                spine[:, axis] = blurred

        tangent = np.gradient(spine, axis=0)
        length = np.hypot(tangent[:, 0], tangent[:, 1])
        length[length == 0.0] = 1.0
        normal = np.stack([-tangent[:, 1] / length,
                           tangent[:, 0] / length], axis=-1)

        for i, (point, unit) in enumerate(zip(spine, normal)):
            candidates = point + step[:, None] * unit
            probe = depth(candidates[:, 0], candidates[:, 1])
            spine[i] = candidates[int(np.argmax(probe))]

    return spine


_CHANNEL_CACHE = {}


def charles_channel(origin: Tuple[float, float] = CHARLES_ORIGIN,
                    resolution: float = 6.0, **kwargs):
    """The navigable channel raster, extracted from the depth contours.

    Cached: the extraction triangulates 12k vertices and runs a distance
    transform over a 1.7 M cell grid, which is a second or two, and every
    caller wants the same answer.

    See :mod:`coxswain.river.channel` for what it does and why the survey
    has to be treated as contours rather than soundings.
    """
    from .channel import build_channel

    key = (origin, resolution, tuple(sorted(kwargs.items())))
    if key not in _CHANNEL_CACHE:
        points, depths = load_isobaths(origin)
        _CHANNEL_CACHE[key] = build_channel(points, depths,
                                            resolution=resolution, **kwargs)
    return _CHANNEL_CACHE[key]


def landmark_station(latlon, channel=None, origin=CHARLES_ORIGIN):
    """Where a ``(lat, lon)`` landmark falls along the channel centreline.

    Returns ``(station, offset)`` in metres: distance along the line, and
    how far the landmark sits from it.  A large offset means the landmark
    coordinate and the extracted channel disagree, which is worth knowing
    before quoting anything about that spot.
    """
    channel = charles_channel(origin) if channel is None else channel
    line = channel.centreline()
    east, north = local_tangent_plane(latlon[0], latlon[1], origin)
    point = np.array([float(east), float(north)])
    station = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(line, axis=0), axis=1))])
    distance = np.linalg.norm(line - point, axis=1)
    index = int(np.argmin(distance))
    return float(station[index]), float(distance[index])


def test_section(channel=None, origin=CHARLES_ORIGIN,
                 before: float = 400.0, after: float = 200.0):
    """The Weeks-to-Anderson reach: a development section, not the race.

    Roughly 800 m of channel running up from below Weeks Footbridge,
    through the Weeks turn, and out past Larz Anderson.  Chosen because it
    is short enough to iterate on and contains the tightest steering on the
    course -- so a model that can fly this can probably fly the rest, and
    one that cannot has failed on the part that decides races.

    Both landmarks sit within about 20 m of the extracted centreline, so
    unlike the ends of the reach the geometry here is well determined.

    Returns ``(start_xy, goal_xy, line)`` where ``line`` is the centreline
    between them, usable directly as an initial guess.
    """
    channel = charles_channel(origin) if channel is None else channel
    line = channel.centreline()
    station = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(line, axis=0), axis=1))])

    weeks, _ = landmark_station(WEEKS_FOOTBRIDGE, channel, origin)
    anderson, _ = landmark_station(LARZ_ANDERSON_BRIDGE, channel, origin)

    # the course runs upstream, i.e. towards decreasing station
    high = weeks + before
    low = anderson - after
    inside = (station >= low) & (station <= high)
    segment = line[inside][::-1]      # ordered in the direction of travel
    return segment[0], segment[-1], segment


def hocr_course(channel=None, origin: Tuple[float, float] = CHARLES_ORIGIN,
                length: float = HOCR_COURSE_LENGTH):
    """The Head of the Charles racing course, start line to finish line.

    The start line lies off the front of BU's DeWolfe Boathouse, which
    sits at the foot of the BU Bridge about 160 m downstream of it -- the
    race does not start at the bridge.  The finish is
    :data:`HOCR_COURSE_LENGTH` upstream along the channel, 4828 m, the
    official three miles.  The race runs *up* the river, so the finish is
    at a **lower** station than the start.

    The finish is derived from the course length rather than from a
    surveyed coordinate: unlike the bridges, the finish line has no fixed
    structure to take a position from, and placing it by measuring the
    real distance along the real channel is the honest construction.  It
    lands above Eliot Bridge, which is where the finish is.

    Returns ``(start_xy, finish_xy, line, stations)``: the two line
    positions in the local tangent plane, the centreline between them
    ordered in the direction of travel, and ``(start_station,
    finish_station)``.
    """
    channel = charles_channel(origin) if channel is None else channel
    line = channel.centreline()
    station = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(line, axis=0), axis=1))])

    start, _ = landmark_station(DEWOLFE_BOATHOUSE, channel, origin)
    finish = start - float(length)

    inside = (station >= finish) & (station <= start)
    segment = line[inside][::-1]          # ordered bow-first up the river
    return segment[0], segment[-1], segment, (start, finish)


def charles_course(centreline: np.ndarray = None,
                   half_width: float = None,
                   month: int = 10, statistic: str = "median",
                   origin: Tuple[float, float] = CHARLES_ORIGIN) -> Course:
    """The surveyed Charles reach, with a continuity-derived current.

    Unlike :func:`~coxswain.river.course.charles_river_sketch` this is
    marked ``is_survey=True`` and passes :meth:`Course.require_survey`,
    because the depths are measured.

    The centreline is still supplied by the caller or defaulted to a coarse
    polyline through the surveyed extent -- extracting a true navigation
    channel from the isobaths is a separate job.  ``Course.is_survey``
    refers to the **bathymetry**; a caller wanting a specific race line
    should pass its own centreline.
    """
    depth = charles_depth_field(origin)

    if centreline is None or half_width is None:
        # Derive both from the survey rather than assuming either.  The
        # contours are the only statement in the data about where the water
        # is; a centreline and a width invented independently of them is how
        # a "channel" ends up 26% aground.
        raster = charles_channel(origin)
        line = raster.centreline()
        if centreline is None:
            centreline = line
        if half_width is None:
            half_width = raster.half_width_along(centreline)
        water_half_width = raster.water_half_width_along(centreline)

    course = Course(
        centreline=centreline,
        half_width=half_width,
        water_half_width=water_half_width,
        depth=depth,
        name="Charles River (CRAB/MIT Sea Grant 2016-17 survey)",
        is_survey=True,
        notes=("bathymetry surveyed; centreline is a coarse spine, not a "
               "surveyed navigation channel"),
    )
    flow = ContinuityFlow(course,
                          discharge=monthly_discharge(month, statistic))
    course.current = flow.as_current_field()
    return course
