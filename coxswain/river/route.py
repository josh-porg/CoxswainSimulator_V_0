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
    #: Steepest yaw rate the line asks for, deg/s.
    peak_yaw_rate: float = 0.0
    #: Bridges passed through a forbidden arch, or through a pier.
    illegal_arches: int = 0
    #: Largest port/starboard pressure split the line asks for -- the
    #: value **wanted**, which may exceed what the crew can give.
    peak_split: float = 0.0
    #: Strokes over which the line asks for any split at all.
    split_strokes: float = 0.0
    #: Anaerobic reserve per rower at the finish, J.
    w_prime_left: float = float("nan")
    #: Lowest the reserve gets anywhere on the course, J.  Zero means the
    #: crew ran out and the line is not rowable.
    w_prime_low: float = float("nan")

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
            "peak_yaw_rate": self.peak_yaw_rate,
            "illegal_arches": self.illegal_arches,
            "peak_split": self.peak_split,
            "split_strokes": self.split_strokes,
            "w_prime_left": self.w_prime_left,
            "w_prime_low": self.w_prime_low,
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
        #: Fastest the boat can turn, deg/s, or ``None`` to ignore it.
        #:
        #: Without this the optimiser cuts corners for free.  It is not a
        #: small effect: on the Charles the unconstrained line came back
        #: demanding **5.75 deg/s** where the boat makes about 1.5 and the
        #: centreline itself only asks 3.30 -- so "optimising" made the
        #: line harder to steer than doing nothing, and the time it claimed
        #: to save was time the boat could never have taken.
        self.max_yaw_rate = None
        #: Bridge gates from :mod:`coxswain.river.bridges`, or ``None``.
        #: A line that saves ten seconds and takes a 60 second penalty at
        #: three bridges is not a faster line, and the evaluator cannot
        #: know that unless it is told where the arches are.
        self.gates = None
        #: Seconds charged for using an arch the rules forbid.
        self.arch_penalty = 60.0
        #: Channel raster the arches are read against; set with
        #: :meth:`with_bridges`.
        self._raster = None
        #: A :class:`~coxswain.river.trajectory.ReducedModel`, once
        #: :meth:`with_steering` has been called.  Supersedes
        #: :attr:`max_yaw_rate`: instead of a hard cap on how fast the
        #: boat may turn, the line is charged what steering it *costs*.
        self.steering = None
        #: Critical-power model for the crew, once :meth:`with_exertion`
        #: has been called.  Without it a pressure split costs only drag,
        #: which is the smaller half of what it really costs.
        self.exertion = None
        #: Power the crew holds relative to critical power.  A head race is
        #: rowed a shade above CP -- high enough that W' drains slowly all
        #: race, which is what makes a split expensive late.
        self.race_intensity = 1.02
        self.rowers = 8
        #: Force the line through a named arch at named bridges, e.g.
        #: ``{"Western Avenue": "Cambridge shore"}``.  The rules leave both
        #: the centre and the Cambridge arch open at River Street, Western
        #: Avenue and Weeks, and which to take is a real strategic choice:
        #: the Cambridge arch is the wider opening at two of them, but it
        #: puts the boat on the outside of what follows.  Pinning it lets
        #: the two strategies be optimised separately and compared, instead
        #: of the optimiser silently picking one.
        self.required_arches = {}
        #: Strokes of port/starboard split a crew can be asked for over a
        #: race.  A coxswain calls pressure in bursts of 10-15 strokes and
        #: 25 at the outside, so a line that needs it continuously is not a
        #: line anyone can row, however good its arithmetic looks.
        self.split_stroke_budget = 25.0
        #: Stroke rate the budget is counted at.
        self.stroke_rate = 32.0
        self._speed_table = None
        self._flow_grid = None

    def with_steering(self, model, raster=None, gates=None):
        """Score lines against both of the coxswain's controls.

        A hard turn-rate cap is the wrong shape for this problem.  The
        rudder is not the only way to turn an eight: the coxswain can call
        for a **port/starboard pressure split**, and on a river that is
        the decisive control -- full rudder alone holds a 259 m radius
        where the tightest bends here demand 103-146 m, while rudder plus
        a 30% split reaches 130 m.

        So a bend is not forbidden, it is *expensive*.  What it costs is
        speed, through two terms the reduced model already carries: drag
        rising with the square of the yaw rate, and the thrust a split
        crew spends on a couple instead of on going forwards.  The line
        that wins is the one that balances distance against the steering
        it has to buy, which is the trade a coxswain actually makes.

        A line is only *infeasible* when rudder and split together at
        their limits cannot hold it.
        """
        self.steering = model
        if raster is not None:
            self.with_bridges(raster, gates=gates)
        return self

    def with_exertion(self, model=None, race_intensity: float = None,
                      rowers: int = 8):
        """Charge a pressure split against the crew's anaerobic reserve.

        A split is not free and it is not merely draggy.  Holding one side
        above its critical power spends W', a reserve of about 11.4 kJ per
        rower that refills over minutes, not seconds
        (:mod:`coxswain.crew.exertion`).  That is the real reason a
        coxswain calls for ten or fifteen strokes of pressure and not a
        mile of it, and until this is in the objective the optimiser has no
        reason to ration it.

        It also makes the cost *positional*.  Weeks and Anderson are 433 m
        apart, about 85 s at racing speed, well inside the recovery time
        constant -- so pressure spent at Weeks is not back for Anderson,
        and the model can now say so.
        """
        from ..crew.exertion import WPrimeBalance, optimal_pace
        self.exertion = WPrimeBalance() if model is None else model
        self.rowers = int(rowers)
        if race_intensity is None:
            # **Race power is not a free parameter.**  A fixed-distance
            # effort is paced to spend the whole reserve and no more, so
            # P = CP + W'/T with T the race duration; anything less and the
            # crew crosses the line still holding work they could have
            # used.  This was set by hand at 1.02 x CP and produced a crew
            # finishing with 45% of W' intact, which is not a raced boat.
            duration = self.course.length / max(self.reference_speed, 0.1)
            power = optimal_pace(duration, self.exertion.critical_power,
                                 self.exertion.capacity)
            race_intensity = power / self.exertion.critical_power
        self.race_intensity = float(race_intensity)
        return self

    def with_wind(self, field, boat=None, drag_area: float = None,
                  height: float = 0.43):
        """Charge the line for the wind it actually meets.

        Until now wind was a scenario the whole race shared: a headwind
        cost 86 s and there was nothing a coxswain could do about it.
        With a spatially varying field there *is* something to do -- the
        sheltered side of a 150 m river carries about 80% of the wind the
        open side does -- and pricing it is what lets the optimiser trade
        a longer line for a quieter one.

        ``drag_area`` is the crew's total ``C_d A``; left out, it is
        calibrated off ``boat`` the same way
        :class:`~coxswain.hydro.wind.AeroModel` does, so this shares the
        13%-of-resistance calibration rather than inventing a second one.

        ``height`` is the area-weighted height of the aerodynamic
        components, 0.43 m, **not** 10 m and not the 1.5 m a rower's chest
        sits at.  The field must therefore be built to report wind at that
        height; asking a 10 m field and then applying a log profile as
        well is the double-correction this signature exists to prevent.
        """
        from ..hydro.wind import AeroModel

        self.wind = field
        if drag_area is None:
            if boat is None:
                boat = self.boat
            if boat is None:
                raise ValueError("with_wind needs a boat or a drag_area")
            drag_area = AeroModel.calibrate(
                boat, reference_speed=self.reference_speed).total_area
        self.wind_area = float(drag_area)
        self.wind_height = float(height)
        return self

    #: Wind field, or ``None`` for still air.  Set by :meth:`with_wind`.
    wind = None
    wind_area = 0.66
    wind_height = 0.43
    #: Air density, kg/m^3.
    air_density = 1.225

    def wind_penalty(self, points, tangent, speed_ground):
        """Fractional speed change from the wind along this line.

        The relative wind is the true wind minus the boat's own velocity,
        so a headwind is charged the square of the *sum* and a tailwind
        the square of the difference -- which is where the published
        asymmetry (12.2% lost to a 5 m/s headwind, 5.1% gained from the
        same tailwind) comes from without being put in by hand.

        Applied through the same relation steering drag uses: at fixed
        crew power an added drag ``dR`` slows the boat by
        ``dv/v = -dR / (3R + dR)``.
        """
        if self.wind is None:
            return np.zeros(len(points))
        vectors = np.array([self.wind.at(px, py)[:2] for px, py in points])
        relative = vectors - speed_ground[:, None] * tangent
        along = np.einsum("ij,ij->i", relative, tangent)
        magnitude = np.hypot(relative[:, 0], relative[:, 1])
        # ``along`` is the component of the relative wind along the boat's
        # own heading, so a headwind makes it negative and the force it
        # produces is negative too -- a resistance.  The added resistance
        # is therefore minus that force, and the sign has to survive into
        # the speed change or the model helpfully rows the crew home in a
        # gale: the first version dropped one negation and reported a
        # 6 m/s headwind as 77 seconds FASTER than still air.
        force = 0.5 * self.air_density * self.wind_area * magnitude * along
        # **Relative to still air, not to a vacuum.**  The reference speed
        # already includes the aerodynamic drag of rowing through calm air
        # -- that is 13% of a shell's resistance and it is inside the
        # calibration.  Charging the full windy force on top of it
        # double-counts the still-air part, which made a 3.6 m/s tailwind
        # come out as a small *loss* instead of a gain and a headwind come
        # out a third too expensive.
        still = -0.5 * self.air_density * self.wind_area * speed_ground ** 2
        added = -(force - still)
        base = self.hydrodynamic_drag(speed_ground)
        return -added / np.maximum(3.0 * base + added, 0.2 * base)

    def hydrodynamic_drag(self, speed) -> np.ndarray:
        """Water resistance at these speeds, N -- the base for a drag ratio.

        Taken from the steering model's critical power where one is
        attached, so the two added-drag terms are charged against the same
        denominator rather than against two different ideas of how hard
        the crew is working.
        """
        speed = np.maximum(np.asarray(speed, dtype=float), 0.05)
        if self.steering is not None:
            return np.full_like(
                speed, self.steering.critical_power
                / max(self.reference_speed, 1e-6))
        return np.full_like(speed, 260.0)

    def with_bridges(self, raster, gates=None, max_yaw_rate=None):
        """Give the evaluator the arches and the boat's turning limit.

        Both are off by default, which is why the first optimised line on
        the Charles came back 16 s "faster" while demanding 5.75 deg/s of
        a boat that makes 1.5, and passing three bridges through arches
        that carry a 60 second penalty each.
        """
        from .charts import CourseGeometry
        self._raster = raster
        if gates is None:
            gates = CourseGeometry(channel=raster).gates_on_course()
        self.gates = gates
        self.max_yaw_rate = max_yaw_rate
        return self

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

        # -- wind, before the clock starts --------------------------------
        # One fixed-point pass: the relative wind depends on boat speed,
        # which the wind then changes.  A second pass moves the answer by
        # under a millisecond per kilometre, which is well inside the
        # field's own uncertainty.
        if self.wind is not None:
            step_full = np.vstack([step, step[-1:]])
            length = np.hypot(step_full[:, 0], step_full[:, 1])
            tangent = step_full / np.maximum(length, 1e-9)[:, None]
            for _ in range(2):
                speed_ground = np.maximum(
                    speed_ground * (1.0 + self.wind_penalty(
                        points, tangent, speed_ground)), 0.05)

        # trapezoidal integration of dt = ds / v
        inverse = 1.0 / speed_ground
        elapsed = float(np.sum(0.5 * (inverse[:-1] + inverse[1:]) * segment))

        # -- can the boat actually steer this line, and at what cost? ----
        # Always measured, so a line can never be quoted as fast without
        # the turn rate it demands sitting next to the time.
        required = self._required_yaw(points, speed_ground)
        peak_turn = float(required.max())
        excess_turn = 0.0
        peak_split = 0.0
        split_strokes = 0.0
        w_prime_left = float("nan")
        w_prime_low = float("nan")

        if self.steering is not None:
            model = self.steering
            rate = np.radians(required)
            moment = model.yaw_damping * speed_ground * rate
            by_rudder = (model.yaw_control * speed_ground ** 2
                         * model.rudder_limit)
            wanted = (np.maximum(moment - by_rudder, 0.0)
                      / model.split_control)
            # Report what the line ASKS FOR, not what it is allowed.  The
            # clipped value saturates at the limit wherever the line is
            # infeasible and so reads the same for a line needing 31% as
            # for one needing 81%, which is exactly the case that matters.
            peak_split = float(wanted.max())
            over_split = np.maximum(wanted - model.split_limit, 0.0)
            split = np.minimum(wanted, model.split_limit)
            excess_turn = float(over_split.mean()) + 4.0 * float(over_split.max())

            # How long is the crew actually being asked to split?  A
            # coxswain calls for pressure in bursts of 10-15 strokes, 25 at
            # the outside -- not continuously for 4.8 km.  Counted at the
            # stroke rate over the distance the split is on.
            on = wanted > 0.05
            if on.any():
                metres_on = float(segment[on[:-1]].sum())
                seconds_on = metres_on / max(float(speed_ground.mean()), 0.05)
                split_strokes = seconds_on * self.stroke_rate / 60.0
            else:
                split_strokes = 0.0

            # Steering costs thrust.  At fixed crew power P = R v with
            # R ~ k v^2, an added constant drag dR slows the boat by
            # dv/v = -dR / (3 R + dR).
            added = (model.turn_drag * rate ** 2
                     + model.split_drag * split ** 2)
            base = model.critical_power / max(self.reference_speed, 1e-6)
            speed_ground = speed_ground * (1.0 - added / (3.0 * base + added))
            speed_ground = np.maximum(speed_ground, 0.05)
            over_budget = max(split_strokes - self.split_stroke_budget, 0.0)
            excess_turn += 0.02 * over_budget

            # -- what the split costs the crew, not just the boat --------
            if self.exertion is not None:
                cp = self.exertion.critical_power
                base = cp * self.race_intensity
                # A balanced split lifts one side and eases the other, so
                # the heavy side carries base * (1 + s).
                heavy = base * (1.0 + split)
                dt = np.concatenate([[0.0], segment / np.maximum(
                    0.5 * (speed_ground[:-1] + speed_ground[1:]), 0.05)])
                balance = self.exertion.integrate(heavy, dt)
                w_prime_left = float(balance[-1])
                w_prime_low = float(balance.min())
                if w_prime_low <= 0.0:
                    # The reserve ran out: the crew cannot hold this line,
                    # whatever the stopwatch says about it.
                    excess_turn += 2.0
            inverse = 1.0 / speed_ground
            elapsed = float(np.sum(0.5 * (inverse[:-1] + inverse[1:])
                                   * segment))
            elapsed_clean_steer = elapsed
        elif self.max_yaw_rate is not None:
            over = np.maximum(required - float(self.max_yaw_rate), 0.0)
            # Charged on the **peak** as well as the mean.  A mean-only
            # penalty is nearly free for a brief excursion -- 1200 samples
            # dilute a short corner to nothing -- and the first version of
            # this returned lines peaking at 4.89 deg/s against a stated
            # limit of 1.50.  A line the boat cannot follow is infeasible,
            # not slightly slow, so the peak has to dominate.
            excess_turn = float(over.mean()) + 4.0 * float(over.max())

        # -- is it legal at the bridges? ---------------------------------
        illegal = self._arch_violations(points) if self.gates else 0

        aground = depth < self.minimum_depth
        fraction_aground = float(aground.mean())
        elapsed_clean = elapsed
        elapsed += self.arch_penalty * illegal
        if excess_turn > 0.0:
            # Same shape as the grounding penalty: a gradient out of the
            # infeasible region rather than a wall, and steep enough that
            # no routing gain can pay for it.
            elapsed *= 1.0 + 20.0 * excess_turn
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
            peak_yaw_rate=peak_turn, illegal_arches=illegal,
            peak_split=peak_split, split_strokes=split_strokes,
            w_prime_left=w_prime_left, w_prime_low=w_prime_low,
        )

    @staticmethod
    def _required_yaw(points, speed_ground, smooth_metres: float = 17.3):
        """Yaw rate the line demands, deg/s, from its own curvature.

        The smoothing length is in **metres**, not in samples, and that
        distinction was a real bug rather than a tidiness point.  Fixed at
        nine samples, the window scaled with ``n_samples``: at the 900
        samples this evaluator uses over 4822 m it smoothed curvature over
        **48 m**, and reported a peak demand of 4.2 deg/s on a line the
        boat actually meets at **10.4 deg/s** -- twice what full rudder
        gives.  A knot narrower than the window is invisible to the
        penalty, so the optimiser was free to emit one, and did: 0.120 per
        metre at the Magazine Beach bend, an 8 m radius.  The controller
        then missed the corner and took 500 m to recover, which read as a
        controller failure for three separate experiments.

        One boat length is the right window and it is the same argument
        :class:`~coxswain.sim.mpc.PathMPC` already makes for itself: a
        17.3 m hull cannot respond to curvature structure shorter than
        itself, so smoothing over its own length discards nothing that was
        ever steerable -- while smoothing over three lengths discards the
        evidence that the line is unsteerable.
        """
        points = np.asarray(points, dtype=float)
        spacing = float(np.median(np.hypot(*np.diff(points, axis=0).T)))
        smooth = max(int(round(float(smooth_metres) / max(spacing, 1e-6))), 3)
        kernel = np.ones(smooth) / smooth
        x = np.convolve(points[:, 0], kernel, mode="same")
        y = np.convolve(points[:, 1], kernel, mode="same")
        ds = np.hypot(np.gradient(x), np.gradient(y))
        ds = np.maximum(ds, 1e-9)
        dx, dy = np.gradient(x) / ds, np.gradient(y) / ds
        ddx, ddy = np.gradient(dx) / ds, np.gradient(dy) / ds
        curvature = np.abs(dx * ddy - dy * ddx)
        edge = smooth * 3
        curvature[:edge] = curvature[edge]
        curvature[-edge:] = curvature[-edge - 1]
        return np.degrees(curvature * np.asarray(speed_ground))

    def _arch_violations(self, points) -> int:
        """How many bridges this line passes on the wrong side of.

        The crossing is found where the path actually **crosses the gate
        line** -- the point at which the gate's signed distance changes
        sign -- and not by matching station numbers.  Station matching
        looks equivalent and is not: the course centreline is resampled
        when the :class:`~coxswain.river.course.Course` is built, so a
        gate's station and the route's station drift apart by a few
        metres, which at the Grand Junction trestle is the difference
        between an arch and a pier.  That drift had two lines reported
        illegal here while a direct check said they were fine.
        """
        from . import bridges as _bridges
        count = 0
        for gate, _metres in self.gates:
            offsets = (points - gate.start) @ gate.normal
            sign = np.sign(offsets)
            crossings = np.nonzero(np.diff(sign) != 0)[0]
            if len(crossings) == 0:
                count += 1                      # never crosses: not a route
                continue
            # A gate is a finite span, but its *line* runs on forever, and
            # a river that bends back on itself crosses that line more than
            # once -- the Charles crosses some of these gate lines a
            # kilometre from the bridge they belong to.  Taking the first
            # sign change therefore scored the wrong crossing and reported
            # every line illegal.  Take the crossing nearest the bridge.
            middle = 0.5 * (gate.start + gate.end)
            i = int(min(crossings,
                        key=lambda j: np.linalg.norm(points[j] - middle)))
            if np.linalg.norm(points[i] - middle) > gate.span:
                count += 1                      # never reaches this bridge
                continue
            a, b = offsets[i], offsets[i + 1]
            t = 0.0 if b == a else float(-a / (b - a))
            point = points[i] + t * (points[i + 1] - points[i])
            where = gate.station_of(point)
            arches = _bridges.bridge_arches(gate, self._raster)
            inside = [arch for arch in arches
                      if arch.interval[0] <= where <= arch.interval[1]]
            if not inside or not inside[0].legal:
                count += 1
                continue
            wanted = self.required_arches.get(gate.name)
            if wanted is not None and inside[0].label != wanted:
                count += 1
        return count


    def loss_breakdown(self, route, reference_length=None) -> dict:
        """Where the seconds go on this line, in an additive account.

        A single race time says which line is quicker and nothing about
        why.  This peels the effects off one at a time, each difference
        being the cost of adding that effect to the one before:

        ``distance``
            Rowing further than the shortest line under comparison.  The
            only term that is purely geometric.
        ``depth``
            Shallow water raising resistance.  The inside of a bend is
            where it shoals, so this often opposes ``distance``.
        ``current``
            Slack on this river in October -- a few tenths of a second
            over the whole course -- but it belongs in the account so its
            smallness is visible rather than assumed.
        ``steering``
            Drag from turning: yaw rate squared, plus the thrust a split
            crew spends on a couple instead of on going forwards.
        ``penalty``
            Sixty seconds per forbidden arch.

        The terms sum to the race time by construction, so nothing hides
        in a residual.
        """
        result = self.evaluate(route)
        station = result.station
        offset = route.offset_at(station)
        points = self.course.offset_position(station, offset)
        segment = np.hypot(*np.diff(points, axis=0).T)
        length = float(segment.sum())

        reference = length if reference_length is None else float(reference_length)
        ideal = reference / self.reference_speed

        def travel(speed):
            inverse = 1.0 / np.maximum(speed, 0.05)
            return float(np.sum(0.5 * (inverse[:-1] + inverse[1:]) * segment))

        flat = travel(np.full_like(station, self.reference_speed))
        with_depth = travel(result.speed_water)
        with_current = travel(np.maximum(result.speed_water
                                         + result.current_along, 0.05))
        with_steering = result.elapsed_clean
        race = with_steering + self.arch_penalty * result.illegal_arches

        return {
            "ideal": ideal,
            "distance": flat - ideal,
            "depth": with_depth - flat,
            "current": with_current - with_depth,
            "steering": with_steering - with_current,
            "penalty": race - with_steering,
            "race": race,
            "path_length": length,
        }

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

    from ..progress import progress
    sweeps = progress(range(iterations), desc="  optimising the line",
                      unit="sweep")
    for _ in sweeps:
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
