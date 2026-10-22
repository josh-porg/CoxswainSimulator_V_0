"""Candidate racing lines, so an optimised one has something to beat.

An optimiser that reports "16 seconds faster" is reporting a number
against an implied baseline, and if that baseline is the channel
centreline the number flatters itself: nobody rows the centreline.  Real
coxswains steer a recognisable line -- point at the arch, hold it, take
the next one -- and the honest question is whether an optimiser beats
*that*, not whether it beats a strawman.

So this builds a handful of lines a competent crew might actually row:

:func:`centreline_route`
    Down the middle of the surveyed channel.  Not a racing line, but the
    reference everything else is quoted against.
:func:`arch_route`
    Point at the middle of each legal arch and hold it.  This is what a
    coxswain is taught to do, and on this course it is a strong line.
:func:`shortest_route`
    Least distance that still stays in the channel and goes through legal
    arches.  The "row the shortest race" instinct, taken seriously.
:func:`inside_bend_route`
    Cut every corner, proportional to how hard it bends.  The naive
    aggressive line -- included because it is what an optimiser without a
    turn-rate constraint converges to, and it is instructive to see it
    scored honestly.

Every one of them is scored by the same
:class:`~coxswain.river.route.RouteEvaluator`, with the same turn-rate
limit and the same arch penalties, so the comparison is like for like.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .route import Route

__all__ = ["to_offset", "arch_point", "crossing_along", "illegal_bridges", "centreline_route", "arch_route", "shortest_route",
           "inside_bend_route", "legalise", "pinned_arch_route",
           "candidate_lines"]


def to_offset(course, point) -> tuple:
    """``(station, offset)`` of a world point in the course frame.

    ``offset`` is positive to port, matching
    :meth:`~coxswain.river.course.Course.offset_position`.
    """
    point = np.asarray(point, dtype=float)[:2]
    station = course.nearest_station(point[0], point[1])
    centre = course.position_at(station)
    heading = course.heading_at(station)
    normal = np.array([-np.sin(heading), np.cos(heading)])
    return float(station), float(np.dot(point - centre, normal))


def arch_point(course, gate, along: float, margin: float = 4.0) -> tuple:
    """``(station, offset)`` of the point ``along`` metres down ``gate``'s deck, offset clipped to the
    corridor. A control point placed **here** puts the line through that point of the deck.

    Placing it at the bridge's own station with this offset is right only for a square crossing: on
    the Grand Junction trestle (41 degrees skew) the line then crossed the deck metres away, outside
    the trestle's narrow second-from-Cambridge arch (2026-10-22)."""
    station, offset = to_offset(course, gate.point_at(along))
    limit = max(float(course.half_width_at(station)) - margin, 0.0)
    return float(station), float(np.clip(offset, -limit, limit))


def crossing_along(route, course, gate, metres: float, reach: float = 300.0):
    """Where ``route`` actually crosses ``gate``'s deck, metres along the deck, or ``None``.

    The same construction as the evaluator's arch check (``RouteEvaluator._arch_violations``): the
    sign change of the gate's signed distance nearest the bridge, interpolated."""
    s = np.arange(max(metres - reach, 0.0), min(metres + reach, course.length), 1.0)
    pts = course.offset_position(s, route.offset_at(s))
    d = (pts - gate.start) @ gate.normal
    idx = np.nonzero(np.diff(np.sign(d)) != 0)[0]
    if not len(idx):
        return None
    middle = 0.5 * (gate.start + gate.end)
    i = int(min(idx, key=lambda j: np.linalg.norm(pts[j] - middle)))
    if np.linalg.norm(pts[i] - middle) > gate.span:
        return None
    t = 0.0 if d[i + 1] == d[i] else float(-d[i] / (d[i + 1] - d[i]))
    return float(gate.station_of(pts[i] + t * (pts[i + 1] - pts[i])))


def illegal_bridges(route, course, raster, gates) -> list:
    """Names of the bridges ``route`` passes outside every legal arch."""
    from . import bridges as _bridges
    out = []
    for gate, metres in gates:
        legal = _bridges.candidate_arches(gate, raster)
        if not legal:
            continue
        along = crossing_along(route, course, gate, metres)
        if along is None or not any(a.interval[0] <= along <= a.interval[1] for a in legal):
            out.append(gate.name)
    return out


def _with_point(stations, offsets, station, offset, merge: float = 15.0):
    """Add a control point, or move the nearest one within ``merge`` metres (not the ends)."""
    inner = [i for i in range(1, len(stations) - 1) if abs(stations[i] - station) < merge]         if len(stations) > 2 else []
    if inner:
        i = min(inner, key=lambda j: abs(stations[j] - station))
        stations[i], offsets[i] = station, offset
    else:
        stations.append(station); offsets.append(offset)


def _ordered(stations, offsets, name):
    order = np.argsort(stations)
    st = np.asarray(stations, float)[order]
    off = np.asarray(offsets, float)[order]
    for i in range(1, len(st)):                    # two bridges' arch points can land on one station
        if st[i] <= st[i - 1] + 0.5:
            st[i] = st[i - 1] + 0.5
    return Route(st, off, name=name)


def centreline_route(course) -> Route:
    """Straight down the middle of the channel."""
    return Route(np.array([0.0, course.length]), np.array([0.0, 0.0]),
                 name="centreline")


def arch_route(course, raster, gates, margin: float = 4.0,
               widest: bool = False) -> Route:
    """Point at each legal arch and hold it -- the taught line.

    Control points sit at each bridge, on the middle of the arch the boat
    should use, with the ends pinned to the centreline so the line starts
    and finishes down the middle.  ``widest=True`` aims at the widest legal
    arch instead of the conventional centre one, which differs at River
    Street and Western Avenue where the Cambridge arch is the larger
    opening.
    """
    from . import bridges as _bridges

    stations, offsets = [0.0], [0.0]
    for gate, metres in gates:
        arches = _bridges.candidate_arches(gate, raster)
        if not arches:
            continue
        if widest:
            arch = max(arches, key=lambda a: a.width)
        else:
            arch = _bridges.racing_arch(gate, raster) or arches[0]
        st, offset = arch_point(course, gate, arch.centre, margin)
        stations.append(st)
        offsets.append(offset)
    stations.append(course.length)
    offsets.append(0.0)

    order = np.argsort(stations)
    name = "widest legal arch" if widest else "through the arches"
    return Route(np.asarray(stations)[order], np.asarray(offsets)[order],
                 name=name)


def inside_bend_route(course, aggression: float = 0.75,
                      margin: float = 4.0, n_control: int = 25) -> Route:
    """Cut every corner, in proportion to how hard it bends.

    The inside of a bend is the short way round and the naive fast line.
    It is also the line with the *highest* curvature, so it demands the
    most of the rudder -- which is exactly the trade a turn-rate
    constraint exists to price.
    """
    stations = np.linspace(0.0, course.length, n_control)
    heading = np.unwrap(course.heading_at(stations))
    curvature = np.gradient(heading, stations)
    scale = np.max(np.abs(curvature))
    if scale <= 0:
        return centreline_route(course)
    limits = np.array([max(course.half_width_at(s) - margin, 0.0)
                       for s in stations])
    # positive offset is to port; a port-hand bend has heading increasing,
    # so the inside of it is to port.
    offsets = np.sign(curvature) * np.abs(curvature) / scale
    offsets = offsets * limits * float(aggression)
    offsets[0] = offsets[-1] = 0.0
    return Route(stations, offsets, name="inside the bends")


def shortest_route(course, evaluator=None, n_control: int = 15,
                   iterations: int = 60, margin: float = 4.0,
                   gates=None, raster=None) -> Route:
    """Least distance that stays in the channel and uses legal arches.

    Coordinate descent on the same offsets the time optimiser uses, but
    minimising path length rather than time, so the two differ only in
    what they are asked for.  Where ``gates`` and ``raster`` are supplied,
    a line through a forbidden arch is charged the same way the evaluator
    charges it -- a short line that takes three 60 second penalties is not
    a line anyone would row.
    """
    from . import bridges as _bridges

    stations = np.linspace(0.0, course.length, n_control)
    limits = np.array([max(course.half_width_at(s) - margin, 0.0)
                       for s in stations])
    offsets = np.zeros(n_control)

    def score(values):
        route = Route(stations, values, name="shortest")
        sample = np.linspace(0.0, course.length, 900)
        points = course.offset_position(sample, route.offset_at(sample))
        length = float(np.hypot(*np.diff(points, axis=0).T).sum())
        if gates is not None and raster is not None:
            for gate, metres in gates:
                index = int(np.argmin(np.abs(sample - metres)))
                where = gate.station_of(points[index])
                arches = _bridges.bridge_arches(gate, raster)
                inside = [a for a in arches
                          if a.interval[0] <= where <= a.interval[1]]
                if not inside or not inside[0].legal:
                    length += 300.0     # metres, as a stand-in for 60 s
        return length

    best = score(offsets)
    step = float(np.median(limits)) * 0.6
    for _ in range(iterations):
        improved = False
        for index in range(1, n_control - 1):
            for delta in (step, -step):
                trial = offsets.copy()
                trial[index] = np.clip(trial[index] + delta,
                                       -limits[index], limits[index])
                value = score(trial)
                if value < best - 1e-6:
                    offsets, best, improved = trial, value, True
        if not improved:
            step *= 0.5
            if step < 0.25:
                break
    return Route(stations, offsets, name="shortest legal")




def legalise(route, course, raster, gates, margin: float = 4.0):
    """Pull a line through a legal arch at every bridge.

    A comparison between lines is only worth reading if they are all
    lines a crew could actually row, and several of these are not:
    the channel centreline goes straight through a pier at the Grand
    Junction trestle, and cutting the bends takes four forbidden arches.
    Scoring those against legal lines compares a race against a
    disqualification.

    Each bridge gets a control point on the nearest legal arch to wherever
    the line was already going -- nearest, so the line keeps its own
    character and is only moved as far as the rules require.  Points away
    from the bridges are left alone.
    """
    from . import bridges as _bridges

    stations = list(np.atleast_1d(route.stations).astype(float))
    offsets = list(np.atleast_1d(route.offsets).astype(float))

    # A point added at one bridge bends the spline at its neighbours, so a bridge that was legal on
    # the first pass can stop being so: repeat until a pass changes nothing.
    for _ in range(4):
        before = (list(stations), list(offsets))
        _legalise_pass(stations, offsets, route.name, course, raster, gates, margin)
        if (stations, offsets) == before:
            break
    return _ordered(stations, offsets, route.name)


def _legalise_pass(stations, offsets, name, course, raster, gates, margin):
    from . import bridges as _bridges

    for gate, metres in gates:
        legal = _bridges.candidate_arches(gate, raster)
        if not legal:
            continue
        # where the line really crosses the deck (the evaluator's test), not its offset at the
        # bridge's station, which on a skewed bridge is a different place (2026-10-22)
        current = _ordered(list(stations), list(offsets), name)
        along = crossing_along(current, course, gate, metres)
        if along is not None and any(a.interval[0] <= along <= a.interval[1] for a in legal):
            continue                      # already legal, leave it be
        ref = along if along is not None else gate.station_of(course.offset_position(
            np.array([metres]), current.offset_at(np.array([metres])))[0])
        arch = min(legal, key=lambda a: abs(a.centre - ref))
        st, offset = arch_point(course, gate, arch.centre, margin)
        _with_point(stations, offsets, st, offset)


def pinned_arch_route(course, raster, gates, pins, margin: float = 4.0,
                      name: str = "pinned"):
    """A line forced through named arches at named bridges.

    ``pins`` maps a bridge name to an arch label, e.g.
    ``{"River Street": "Cambridge shore"}``.  Used as the starting point
    for an optimiser told to keep that choice, so a strategy can be
    costed rather than assumed.
    """
    from . import bridges as _bridges

    stations, offsets = [0.0], [0.0]
    for gate, metres in gates:
        arches = _bridges.candidate_arches(gate, raster)
        if not arches:
            continue
        wanted = pins.get(gate.name)
        if wanted is not None:
            match = [a for a in arches if a.label == wanted]
            arch = match[0] if match else arches[-1]
        else:
            arch = _bridges.racing_arch(gate, raster) or arches[0]
        st, offset = arch_point(course, gate, arch.centre, margin)
        stations.append(st)
        offsets.append(offset)
    stations.append(course.length)
    offsets.append(0.0)
    return _ordered(stations, offsets, name)


def candidate_lines(course, raster, gates, margin: float = 4.0,
                    legal: bool = True) -> list:
    """The full set of baselines, in the order worth reading them.

    With ``legal=True`` every one is pulled through a permitted arch
    first, so the table compares races rather than disqualifications.
    """
    routes = [
        centreline_route(course),
        arch_route(course, raster, gates, margin=margin),
        arch_route(course, raster, gates, margin=margin, widest=True),
        shortest_route(course, margin=margin, gates=gates, raster=raster),
        inside_bend_route(course, margin=margin),
    ]
    if legal:
        routes = [legalise(r, course, raster, gates, margin) for r in routes]
    return routes
