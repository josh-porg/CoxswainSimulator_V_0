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

__all__ = ["to_offset", "centreline_route", "arch_route", "shortest_route",
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
        _, offset = to_offset(course, gate.point_at(arch.centre))
        limit = max(course.half_width_at(metres) - margin, 0.0)
        stations.append(float(metres))
        offsets.append(float(np.clip(offset, -limit, limit)))
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

    for gate, metres in gates:
        legal = _bridges.candidate_arches(gate, raster)
        if not legal:
            continue
        here = float(route.offset_at(np.array([metres]))[0])
        point = course.offset_position(np.array([metres]),
                                       np.array([here]))[0]
        along = gate.station_of(point)
        inside = [a for a in legal
                  if a.interval[0] <= along <= a.interval[1]]
        if inside:
            continue                      # already legal, leave it be
        arch = min(legal, key=lambda a: abs(a.centre - along))
        _, offset = to_offset(course, gate.point_at(arch.centre))
        limit = max(course.half_width_at(metres) - margin, 0.0)
        stations.append(float(metres))
        offsets.append(float(np.clip(offset, -limit, limit)))

    order = np.argsort(stations)
    return Route(np.asarray(stations)[order], np.asarray(offsets)[order],
                 name=route.name)


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
        _, offset = to_offset(course, gate.point_at(arch.centre))
        limit = max(course.half_width_at(metres) - margin, 0.0)
        stations.append(float(metres))
        offsets.append(float(np.clip(offset, -limit, limit)))
    stations.append(course.length)
    offsets.append(0.0)
    order = np.argsort(stations)
    return Route(np.asarray(stations)[order], np.asarray(offsets)[order],
                 name=name)


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
