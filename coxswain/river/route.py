"""Evaluating and optimising a line down the river.

This is what the bathymetry and the flow field were built for.  A coxswain
steering the Charles is trading three things against each other, and they
do not point the same way:

* **distance** -- the inside of a bend is shorter;
* **depth** -- the inside of a bend is also where it shoals, and shallow
  water costs speed through the wave-resistance rise;
* **current** -- the deep channel carries the fastest water, which helps
  going downstream and hurts coming up.

A line that is short is usually shallow and slack; a line that is deep is
usually long and, going upstream, against the strongest flow.  There is no
a-priori answer, which is exactly why it is worth computing.

A surrogate, deliberately
-------------------------
Running the full 6-DOF model down a 4 km course takes minutes, and an
optimiser needs hundreds of candidates.  :class:`RouteEvaluator` is a
quasi-steady surrogate instead: at each station it asks the *real*
resistance model what the local depth does to drag, converts that to a
speed through the water at constant crew power, adds the local current, and
integrates ``ds / v_ground``.

What it keeps: the actual :func:`~coxswain.hydro.resistance.hull_resistance`
and :class:`~coxswain.hydro.shallow.ShallowWaterModel`, the surveyed
bathymetry, and the conveyance-weighted lateral flow.

What it drops: everything unsteady.  The crew's surge oscillation, the
transient of accelerating into a bend, and the yaw dynamics of actually
holding a line are all absent.  So this ranks routes; it does not predict
race times.  :func:`verify_with_simulator` re-runs a chosen line through
the full model to check the ranking is not an artefact of the surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Route",
    "RouteEvaluation",
    "RouteEvaluator",
    "optimise_route",
]


@dataclass(frozen=True)
class Route:
    """A line down the course, as an offset from the centreline.

    ``offsets`` are metres to port at each of ``stations``; the line
    between them is a periodic-free cubic interpolation.  Positive is port,
    matching the hull ``y`` axis.
    """

    stations: np.ndarray
    offsets: np.ndarray
    name: str = "route"

    def __post_init__(self) -> None:
        stations = np.asarray(self.stations, dtype=float)
        offsets = np.asarray(self.offsets, dtype=float)
        if stations.shape != offsets.shape:
            raise ValueError("stations and offsets must have the same length")
        if len(stations) < 2:
            raise ValueError("a route needs at least two control points")
        if np.any(np.diff(stations) <= 0):
            raise ValueError("stations must be strictly increasing")
        object.__setattr__(self, "stations", stations)
        object.__setattr__(self, "offsets", offsets)

    @classmethod
    def centreline(cls, course, n: int = 12) -> "Route":
        stations = np.linspace(0.0, course.length, n)
        return cls(stations, np.zeros(n), name="centreline")

    @classmethod
    def constant_offset(cls, course, offset: float, n: int = 12) -> "Route":
        stations = np.linspace(0.0, course.length, n)
        return cls(stations, np.full(n, float(offset)),
                   name=f"offset {offset:+.0f} m")

    def offset_at(self, station):
        """Offset in metres at one or many stations, monotone cubic."""
        from scipy.interpolate import PchipInterpolator

        interpolator = PchipInterpolator(self.stations, self.offsets,
                                         extrapolate=True)
        return interpolator(np.asarray(station, dtype=float))

    def clip_to_channel(self, course, margin: float = 0.0) -> "Route":
        """Pull the control points inside the navigable channel."""
        limit = np.array([course.half_width_at(s) for s in self.stations])
        limit = np.maximum(limit - margin, 0.0)
        return Route(self.stations, np.clip(self.offsets, -limit, limit),
                     name=self.name)

    def path(self, course, n: int = 400) -> np.ndarray:
        """The line as absolute-frame ``(x, y)`` points."""
        station = np.linspace(0.0, course.length, n)
        return course.offset_position(station, self.offset_at(station))


@dataclass
class RouteEvaluation:
    """Result of scoring one route."""

    route: Route
    station: np.ndarray
    path_length: float
    elapsed: float
    depth: np.ndarray
    current_along: np.ndarray
    speed_water: np.ndarray
    speed_ground: np.ndarray
    fraction_aground: float = 0.0
    #: Time before the grounding penalty is applied.  This is the physical
    #: estimate; :attr:`elapsed` is the *score* the optimiser minimises and
    #: is inflated wherever the line crosses unnavigable water.  Report
    #: this one as a time, and :attr:`elapsed` only as a ranking.
    elapsed_clean: float = 0.0

    @property
    def mean_ground_speed(self) -> float:
        """Physical mean speed, penalty excluded."""
        return self.path_length / self.elapsed_clean

    def summary(self) -> dict:
        return {
            "name": self.route.name,
            "path_length": self.path_length,
            "elapsed": self.elapsed,
            "elapsed_clean": self.elapsed_clean,
            "mean_ground_speed": self.mean_ground_speed,
            "min_depth": float(self.depth.min()),
            "mean_depth": float(self.depth.mean()),
            "mean_current_along": float(self.current_along.mean()),
            "fraction_aground": self.fraction_aground,
        }


class RouteEvaluator:
    """Scores routes on a course, at fixed crew power.

    The reference speed is the deep-water speed the crew holds; everything
    else is expressed relative to it, so the evaluator never has to know
    the crew's absolute power.
    """

    def __init__(self, course, flow=None, boat=None,
                 reference_speed: float = 5.2, upstream: bool = True,
                 n_samples: int = 400, margin: float = 3.0,
                 minimum_depth: float = 1.2):
        self.course = course
        self.flow = flow
        self.boat = boat
        self.reference_speed = float(reference_speed)
        #: Rowing up the course (towards increasing station) or down it.
        self.upstream = bool(upstream)
        self.n_samples = int(n_samples)
        #: Keep this far off each bank; a shell is 17 m long and does not
        #: turn quickly, and the surveyed edge is not a hard wall.
        self.margin = float(margin)
        #: Water this shallow is not navigable.  A racing eight draws only
        #: ~0.2 m, but blades reach well below the hull and the surveyed
        #: depth near a bank is both uncertain and often weedy.  Without
        #: this the optimiser will route through the shallowest water it
        #: can find, since that is where the wave-resistance model is
        #: weakest and most easily exploited.
        self.minimum_depth = float(minimum_depth)
        self._speed_table = None
        self._flow_grid = None

    # -- physics ----------------------------------------------------------
    def speed_through_water(self, depth) -> np.ndarray:
        """Speed a crew of fixed power holds at this depth.

        At steady state the crew's power equals the resistance power, so
        ``R(v) v = P``.  Resistance scales close to ``v^2``, and shallow
        water multiplies it by the wave factor ``F(v, h)``, so

            P = k F(v, h) v^3  =>  F(v, h) v^3 = F(v_ref, inf) v_ref^3

        which is solved here for ``v``.  The factor comes from the real
        :class:`~coxswain.hydro.shallow.ShallowWaterModel`, not a fitted
        curve, so the shallow-water calibration in ``docs/SOURCES.md`` §6
        carries over unchanged.

        Solved by bisection, not fixed-point iteration.  ``F`` rises very
        steeply through the critical region, and a fixed-point sweep there
        does not converge -- it oscillates between branches and can return
        a *higher* speed in shallower water, which an optimiser will then
        happily steer into.  ``F(v) v^3`` is strictly increasing in ``v``
        (both factors are), so the root is unique and bracketing is safe.
        """
        scalar = np.ndim(depth) == 0
        depth = np.atleast_1d(np.asarray(depth, dtype=float))

        # The relation depends only on depth once the reference speed is
        # fixed, and an optimiser asks for it hundreds of thousands of
        # times.  Solve it on a grid once, then interpolate: the curve is
        # smooth and monotone, so interpolation error is far below the
        # uncertainty in the bathymetry it is fed.
        if self._speed_table is None:
            self._speed_table = self._build_speed_table()
        grid_depth, grid_speed = self._speed_table
        speed = np.interp(depth, grid_depth, grid_speed)
        return float(speed[0]) if scalar else speed

    def _build_speed_table(self, n: int = 240,
                           max_depth: float = 60.0):
        """Solve the power balance on a depth grid, once."""
        from scipy.optimize import brentq

        from ..hydro.shallow import ShallowWaterModel

        grid = np.concatenate([
            np.linspace(0.2, 6.0, n // 2, endpoint=False),
            np.geomspace(6.0, max_depth, n - n // 2),
        ])
        speed = np.empty_like(grid)
        template = self.boat.shallow if self.boat is not None \
            else ShallowWaterModel()
        target = self.reference_speed ** 3

        for index, h in enumerate(grid):
            if not np.isfinite(h) or h <= 0:
                speed[index] = self.reference_speed
                continue
            model = ShallowWaterModel(
                depth=float(h),
                max_amplification=template.max_amplification,
                subcritical_limit=template.subcritical_limit,
                supercritical_relax=template.supercritical_relax,
                gravity=template.gravity,
            )

            def excess(v, _model=model):
                return float(_model.factor(v)) * v ** 3 - target

            low = 1e-3
            if excess(self.reference_speed) <= 0.0:
                # deep enough that the factor is 1: the reference stands
                speed[index] = self.reference_speed
                continue
            speed[index] = brentq(excess, low, self.reference_speed,
                                  xtol=1e-9, rtol=1e-12)

        # Enforce monotonicity in depth.  Above about Fr_h = 1.3 the
        # shallow-water model's supercritical branch relaxes the wave
        # resistance below its deep-water value -- real physics for a
        # planing hull outrunning its own wave system, and nonsense for a
        # rowing eight, which at that Froude number would be in a metre of
        # water dragging its skeg.  Left alone, the optimiser finds it and
        # routes through the shallows.
        #
        # This is a guard, not physics: it says the model is not valid
        # there rather than claiming to know what happens.  The clamp only
        # ever binds below the critical depth, which `minimum_depth` should
        # already have excluded from any route worth considering.
        speed = np.minimum.accumulate(speed[::-1])[::-1]
        return grid, speed

    def current_along_path(self, station: np.ndarray,
                           offset: np.ndarray) -> np.ndarray:
        """Component of the current along the direction of travel, m/s.

        Positive helps.  A crew rowing upstream meets the flow head on, so
        this is negative there and positive coming down.

        Takes ``(station, offset)`` rather than ``(x, y)``: the caller
        already knows them, and recovering them from absolute coordinates
        would mean a nearest-point search against every centreline segment
        for every sample of every candidate route.
        """
        if self.flow is None:
            return np.zeros(len(station))

        if self._flow_grid is None:
            self._flow_grid = self.flow._speed_grid(120)
        stations, fractions, grid = self._flow_grid

        half = np.array([max(self.course.half_width_at(s), 1e-6)
                         for s in station])
        fraction = np.clip(offset / half, -1.0, 1.0)

        # bilinear interpolation on the precomputed (station, fraction) grid
        column = np.clip(
            np.searchsorted(fractions, fraction) - 1, 0, len(fractions) - 2)
        weight = ((fraction - fractions[column])
                  / (fractions[column + 1] - fractions[column]))
        low = np.array([np.interp(s, stations, grid[:, c])
                        for s, c in zip(station, column)])
        high = np.array([np.interp(s, stations, grid[:, c + 1])
                         for s, c in zip(station, column)])
        speed = low + weight * (high - low)

        # the water runs downstream, i.e. towards decreasing station
        return -speed if self.upstream else speed

    # -- scoring ----------------------------------------------------------
    def evaluate(self, route: Route) -> RouteEvaluation:
        """Time to travel this line, and the fields along it."""
        route = route.clip_to_channel(self.course, self.margin)
        station = np.linspace(0.0, self.course.length, self.n_samples)
        offset = route.offset_at(station)
        points = self.course.offset_position(station, offset)

        # true path length: the line, not the centreline
        step = np.diff(points, axis=0)
        segment = np.hypot(step[:, 0], step[:, 1])
        path_length = float(segment.sum())

        depth = np.asarray(self.course.depth(points[:, 0], points[:, 1]),
                           dtype=float)
        speed_water = self.speed_through_water(depth)
        current_along = self.current_along_path(station, offset)
        speed_ground = np.maximum(speed_water + current_along, 0.05)

        # trapezoidal integration of dt = ds / v
        inverse = 1.0 / speed_ground
        elapsed = float(np.sum(0.5 * (inverse[:-1] + inverse[1:]) * segment))

        aground = depth < self.minimum_depth
        fraction_aground = float(aground.mean())
        elapsed_clean = elapsed
        if fraction_aground > 0.0:
            # Penalise rather than reject: the optimiser needs a gradient
            # out of an infeasible region, not a wall.  Scaled so that any
            # grounding is decisively worse than any routing gain.
            elapsed *= 1.0 + 10.0 * fraction_aground

        return RouteEvaluation(
            route=route, station=station, path_length=path_length,
            elapsed=elapsed, depth=depth, current_along=current_along,
            speed_water=speed_water, speed_ground=speed_ground,
            fraction_aground=fraction_aground, elapsed_clean=elapsed_clean,
        )

    def compare(self, routes: Sequence[Route]):
        """Score several routes, fastest first."""
        results = [self.evaluate(r) for r in routes]
        return sorted(results, key=lambda r: r.elapsed)


def optimise_route(evaluator: RouteEvaluator, n_control: int = 9,
                   iterations: int = 60, seed: int = 0,
                   initial: Optional[Route] = None) -> RouteEvaluation:
    """Search for the quickest line by coordinate descent on the offsets.

    Deliberately simple: the objective is cheap, smooth in each offset, and
    box-constrained by the channel, so a few sweeps of coordinate descent
    with a shrinking step find the same answer a fancier method would --
    and it is obvious what it did, which matters more here than speed.
    """
    course = evaluator.course
    stations = np.linspace(0.0, course.length, n_control)
    limits = np.array([max(course.half_width_at(s) - evaluator.margin, 0.0)
                       for s in stations])

    offsets = np.zeros(n_control) if initial is None \
        else initial.offset_at(stations)
    offsets = np.clip(offsets, -limits, limits)

    best = evaluator.evaluate(Route(stations, offsets, name="optimised"))
    step = float(np.median(limits)) * 0.6

    for _ in range(iterations):
        improved = False
        for index in range(n_control):
            for direction in (+1.0, -1.0):
                trial = offsets.copy()
                trial[index] = np.clip(trial[index] + direction * step,
                                       -limits[index], limits[index])
                if trial[index] == offsets[index]:
                    continue
                candidate = evaluator.evaluate(
                    Route(stations, trial, name="optimised"))
                if candidate.elapsed < best.elapsed - 1e-9:
                    best, offsets, improved = candidate, trial, True
        if not improved:
            step *= 0.5
            if step < 0.25:
                break

    return best
