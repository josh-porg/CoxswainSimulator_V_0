"""Stochastic trajectory optimisation: one plan, many crews and days.

The deterministic solve answers "what is the fastest line for *this* crew
in *these* conditions". That is not the question a coxswain has on race
day. The crew's power varies stroke to stroke, their timing scatters, the
wind is whatever it is, and the current is only known approximately. A
line that is optimal at the nominal parameters can be fragile in a way
that a line a little slower on paper is not.

This is the standard **sample-average approximation** of a stochastic
programme: draw a set of scenarios, transcribe the full dynamics once per
scenario, and optimise a single control policy across all of them at
once. The controls are shared -- the coxswain steers one boat and cannot
condition on which scenario they are in -- while the states are free per
scenario, because the boat really does end up somewhere different in each.

Nothing is reduced. Each scenario carries the same 6-DOF dynamics, the
same phase-locked mesh and the same channel and clearance constraints as
the deterministic solve.

What is sampled, and from what
------------------------------
Every distribution here is taken from measurement, not chosen:

``power``
    Per-rower, per-stroke multiplier on handle force. Kleshnev's
    variability series measures **2.3% for an elite sculler and 5.1% for a
    junior**; work per stroke 1.3% and 4.7%.
``timing``
    Per-rower phase scatter. Cuijpers, Zaal & de Poel (2015) measure the
    SD of crew relative phase at **2.2 deg in-phase**, rising to 4.8 deg
    at 36 spm -- 11 to 25 ms at racing rate.
``wind``
    Speed and bearing. Kleshnev reports a 5 m/s headwind costs an eight
    12.2% of its speed and the same tailwind gains 5.1%, so the asymmetry
    matters and the distribution must cover both.
``current``
    Multiplier on the modelled discharge. The Charles flow model is fitted
    to a gauge some distance upstream, so the scale is the least certain
    part of it.

Objective
---------
The default is a **risk-averse** blend: mean progress penalised by the
spread across scenarios,

    maximise  E[progress] - kappa * sd[progress]

with ``kappa = 0`` recovering the plain sample average. This is a
mean-standard-deviation objective, which is the usual first step before
committing to a full CVaR formulation, and it is what a coxswain
describes when they say they would rather be reliably quick than
occasionally brilliant.

References
----------
See ``docs/SOURCES.md`` section 25 for the full corpus; the specific
sources for each distribution are named above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = ["Scenario", "ScenarioSet", "StochasticTrajectory",
           "StochasticPlan"]


@dataclass(frozen=True)
class Scenario:
    """One draw of the uncertain parameters."""

    power_scales: np.ndarray        # per seat, multiplier
    phase_offsets: np.ndarray       # per seat, fraction of a stroke
    wind_speed: float = 0.0         # m/s, at 10 m
    wind_bearing: float = 0.0       # rad, direction blown towards
    current_scale: float = 1.0      # multiplier on modelled discharge
    weight: float = 1.0             # probability weight

    def describe(self) -> str:
        return ("power %.3f+-%.3f  timing sd %.1f ms  wind %.1f m/s @ %3.0f "
                "deg  current x%.2f"
                % (self.power_scales.mean(), self.power_scales.std(),
                   1000.0 * self.phase_offsets.std(),
                   self.wind_speed, np.degrees(self.wind_bearing),
                   self.current_scale))


@dataclass
class ScenarioSet:
    """A sampled set of scenarios plus the distributions they came from."""

    scenarios: Sequence[Scenario]
    seed: int = 0

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        return iter(self.scenarios)

    @property
    def weights(self) -> np.ndarray:
        raw = np.array([s.weight for s in self.scenarios], dtype=float)
        return raw / raw.sum()

    @classmethod
    def sample(cls, n_seats: int, n_scenarios: int = 8,
               power_sigma: float = 0.023,
               timing_sigma: float = 0.012,
               wind_sigma: float = 2.5,
               current_sigma: float = 0.20,
               period: float = 1.875,
               seed: int = 0,
               include_nominal: bool = True) -> "ScenarioSet":
        """Draw scenarios from the measured distributions.

        Defaults are the **elite** end of the measured ranges: 2.3% force
        variability (Kleshnev) and 12 ms of timing scatter (Cuijpers et
        al., 2.2 deg at 32 spm).  Raise them for a club crew.

        ``include_nominal`` puts the mean case in the set, which makes the
        stochastic optimum directly comparable with the deterministic one
        and guarantees the set is never pathologically unrepresentative at
        small sample sizes.
        """
        rng = np.random.default_rng(seed)
        drawn = []
        if include_nominal:
            drawn.append(Scenario(power_scales=np.ones(n_seats),
                                  phase_offsets=np.zeros(n_seats),
                                  wind_speed=0.0, wind_bearing=0.0,
                                  current_scale=1.0))
        while len(drawn) < n_scenarios:
            power = 1.0 + rng.normal(0.0, power_sigma, n_seats)
            timing = rng.normal(0.0, timing_sigma, n_seats) / period
            timing -= timing.mean()
            drawn.append(Scenario(
                power_scales=np.maximum(power, 0.0),
                phase_offsets=timing,
                wind_speed=abs(rng.normal(0.0, wind_sigma)),
                wind_bearing=rng.uniform(0.0, 2.0 * np.pi),
                current_scale=float(np.clip(
                    1.0 + rng.normal(0.0, current_sigma), 0.3, 2.0)),
            ))
        return cls(scenarios=tuple(drawn), seed=seed)

    def summary(self) -> str:
        lines = ["%d scenarios" % len(self)]
        for i, s in enumerate(self):
            lines.append("  %2d  %s" % (i, s.describe()))
        return "\n".join(lines)


@dataclass
class StochasticPlan:
    """One control policy, and how it performed across the scenarios."""

    time: np.ndarray
    controls: np.ndarray                 # shared across scenarios
    states: Sequence[np.ndarray]         # one per scenario
    progress: np.ndarray                 # one per scenario
    scenarios: ScenarioSet
    success: bool
    stats: dict

    @property
    def expected(self) -> float:
        return float(np.dot(self.scenarios.weights, self.progress))

    @property
    def spread(self) -> float:
        mean = self.expected
        var = float(np.dot(self.scenarios.weights,
                           (self.progress - mean) ** 2))
        return float(np.sqrt(max(var, 0.0)))

    @property
    def worst(self) -> float:
        return float(self.progress.min())


class StochasticTrajectory:
    """Sample-average approximation over the full 6-DOF dynamics.

    Wraps :class:`~coxswain.river.sixdof_trajectory.SixDofTrajectory`: one
    scenario becomes one copy of the dynamics with its own crew draw and
    its own wind, sharing a single control trajectory.
    """

    def __init__(self, boat, raster, scenarios: ScenarioSet, **kwargs):
        from .sixdof_trajectory import SixDofTrajectory

        self.scenarios = scenarios
        self.boat = boat
        # One SixDofTrajectory per scenario.  They share the raster and the
        # progress field -- the river is the same in every scenario -- but
        # each carries its own crew draw, so each has its own oar fits.
        self.legs = []
        base = SixDofTrajectory(boat, raster, **kwargs)
        self.base = base
        for scenario in scenarios:
            self.legs.append(self._leg_for(scenario, base, kwargs))

    def _leg_for(self, scenario, base, kwargs):
        """A dynamics copy carrying this scenario's crew and weather."""
        from ..crew.oarlock import BladeModel
        from .sixdof import SixDofModel
        from .sixdof_trajectory import SixDofTrajectory

        boat = self.boat
        # The crew draw changes the oar loads, so the periodic fits must be
        # rebuilt; the hull surrogate does not depend on the crew and is
        # shared, which is most of the build cost.
        previous_power = boat.power_scales.copy()
        previous_phase = boat.phase_offsets.copy()
        boat.power_scales = scenario.power_scales
        boat.phase_offsets = scenario.phase_offsets
        try:
            model = SixDofModel(boat, surrogate=base.model.surrogate,
                                blade=base.model.blade)
            leg = SixDofTrajectory(
                boat, base.raster, model=model,
                progress_field=None if base is None else None,
                margin=base.margin)
            # reuse the base's interpolants: same river, same everything
            leg.depth = base.depth
            leg.clearance = base.clearance
            leg.progress = base.progress
            leg.has_current = base.has_current
            if base.has_current:
                leg.current_east = base.current_east
                leg.current_north = base.current_north
            leg.scenario = scenario
        finally:
            boat.power_scales = previous_power
            boat.phase_offsets = previous_phase
        return leg

    def solve(self, start_state, n_strokes: int = 2,
              drive_intervals: int = 4, recovery_intervals: int = 3,
              kappa: float = 0.0, max_iter: int = 400,
              comfort: float = 1.2, comfort_weight: float = 6e-2,
              roll_weight: float = 2e-2,
              rudder_limit: float = np.radians(25.0),
              split_limit: float = 0.15,
              power_bounds=(0.70, 1.15),
              smoothing_weight: float = 1e-2,
              print_level: int = 0):
        """Optimise one control policy across every scenario.

        ``kappa`` is the risk aversion: the objective is
        ``E[progress] - kappa * sd[progress]``, so zero is the plain sample
        average and larger values buy consistency at the cost of expected
        speed.
        """
        import casadi as ca

        from .collocation import HermiteSimpson, phase_locked_mesh
        from .scaling import ProblemScaling

        mesh = phase_locked_mesh(self.boat.timing, n_strokes,
                                 drive_intervals, recovery_intervals)
        n = len(mesh)
        times = np.array([i.start for i in mesh] + [mesh[-1].end])
        durations = np.array([i.duration for i in mesh])
        n_states = self.base.model.n_states
        n_controls = self.base.model.n_controls

        speed = max(float(np.hypot(*np.asarray(start_state)[6:8])), 1.0)
        scaling = ProblemScaling.for_six_dof(
            self.base.model, leg_length=max(speed * times[-1], 1.0),
            speed=speed, rudder_limit=rudder_limit,
            split_limit=split_limit)
        state_scale = ca.DM(scaling.state.reshape(-1, 1))
        control_scale = ca.DM(scaling.control.reshape(-1, 1))

        # ---- shared controls, per-scenario states --------------------
        control = ca.MX.sym("U", n_controls, n + 1)
        control_mid = ca.MX.sym("Um", n_controls, n)
        states = [ca.MX.sym("X%d" % k, n_states, n + 1)
                  for k in range(len(self.scenarios))]

        constraints, lower, upper = [], [], []
        progress_terms = []
        room_terms = []
        roll_terms = []
        clearance_scale = self.base.blade_reach + self.base.margin

        for leg, X in zip(self.legs, states):
            dynamics = scaling.scaled_dynamics(
                leg.dynamics_function(wind=getattr(leg, "scenario", None)),
                ca)
            constraints.append(HermiteSimpson.defects(
                dynamics, X, control, control_mid, times, durations))
            lower.append(np.zeros(n_states * n))
            upper.append(np.zeros(n_states * n))

            constraints.append(
                X[:, 0] - ca.DM(scaling.to_scaled_state(start_state)))
            lower.append(np.zeros(n_states))
            upper.append(np.zeros(n_states))

            room = []
            for k in range(n + 1):
                where = ca.vertcat(X[0, k] * state_scale[0],
                                   X[1, k] * state_scale[1])
                room.append((self.base.clearance(where)
                             - clearance_scale) / clearance_scale)
            constraints.append(ca.vertcat(*room))
            lower.append(np.zeros(n + 1))
            upper.append(np.full(n + 1, ca.inf))
            room_terms.append(room)
            roll_terms.append(X[3, :])

            constraints.append(ca.reshape(X[12, :], n + 1, 1))
            lower.append(np.zeros(n + 1))
            upper.append(np.full(n + 1, ca.inf))

            final = ca.vertcat(X[0, n] * state_scale[0],
                               X[1, n] * state_scale[1])
            first = ca.vertcat(X[0, 0] * state_scale[0],
                               X[1, 0] * state_scale[1])
            progress_terms.append(
                (self.base.progress(final) - self.base.progress(first))
                / max(float(scaling.state[0]), 1.0))

        weights = self.scenarios.weights
        expected = sum(w * p for w, p in zip(weights, progress_terms))
        objective = -expected
        if kappa != 0.0:
            variance = sum(w * (p - expected) ** 2
                           for w, p in zip(weights, progress_terms))
            objective = objective + kappa * ca.sqrt(variance + 1e-12)

        rate = control[0, 1:] - control[0, :-1]
        objective = objective + smoothing_weight * ca.sumsqr(rate)

        # The same running clearance-comfort barrier and roll penalty the
        # deterministic solver uses.  They are not options: without them
        # the deterministic run cut the inside of the wide reaches and
        # arrived at the station-450 pinch with 8.5 m of room, and roll sat
        # on the surrogate's tabulation bound rather than anywhere
        # physical.  A stochastic solve that behaved differently from the
        # deterministic one for reasons unrelated to uncertainty would not
        # be measuring uncertainty.  See SOURCES sec. 30.
        for weight, entries in zip(weights, room_terms):
            deficit = 0
            for entry in entries:
                slack = comfort - entry
                positive = 0.5 * (slack + ca.sqrt(slack * slack + 1e-6))
                deficit = deficit + positive * positive
            objective = objective + weight * comfort_weight * deficit / (n + 1)
        for weight, roll in zip(weights, roll_terms):
            objective = objective + weight * roll_weight * ca.sumsqr(roll)

        # ---- bounds ---------------------------------------------------
        surrogate = self.base.model.surrogate
        state_lo = np.full((n_states, n + 1), -ca.inf)
        state_hi = np.full((n_states, n + 1), ca.inf)
        state_lo[3, :] = float(surrogate.roll[0]) * 0.95
        state_hi[3, :] = float(surrogate.roll[-1]) * 0.95
        state_lo[4, :] = float(surrogate.pitch[0]) * 0.95
        state_hi[4, :] = float(surrogate.pitch[-1]) * 0.95
        state_lo[2, :] = float(surrogate.heave[0]) * 0.95
        state_hi[2, :] = float(surrogate.heave[-1]) * 0.95
        state_lo = state_lo / scaling.state[:, None]
        state_hi = state_hi / scaling.state[:, None]

        control_lo = np.zeros((n_controls, n + 1))
        control_hi = np.zeros((n_controls, n + 1))
        mid_lo = np.zeros((n_controls, n))
        mid_hi = np.zeros((n_controls, n))

        def fill(lo, hi, column, phase):
            lo[0, column], hi[0, column] = -rudder_limit, rudder_limit
            if phase == "drive":
                lo[1, column], hi[1, column] = -split_limit, split_limit
            else:
                lo[1, column] = hi[1, column] = 0.0
            lo[2, column], hi[2, column] = power_bounds

        for k in range(n):
            fill(control_lo, control_hi, k, mesh[k].phase)
            fill(mid_lo, mid_hi, k, mesh[k].phase)
        fill(control_lo, control_hi, n, mesh[-1].phase)
        control_lo = control_lo / scaling.control[:, None]
        control_hi = control_hi / scaling.control[:, None]
        mid_lo = mid_lo / scaling.control[:, None]
        mid_hi = mid_hi / scaling.control[:, None]

        variables = ca.vertcat(*[ca.vec(X) for X in states],
                               ca.vec(control), ca.vec(control_mid))
        problem = {"x": variables, "f": objective,
                   "g": ca.vertcat(*constraints)}
        solver = ca.nlpsol("stochastic", "ipopt", problem, {
            "ipopt.max_iter": max_iter,
            "ipopt.print_level": print_level,
            "ipopt.sb": "yes", "print_time": False,
            "ipopt.tol": 1e-5, "ipopt.acceptable_tol": 1e-4,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.hessian_approximation": "limited-memory",
        })

        guesses = []
        for leg in self.legs:
            guesses.append(leg._initial_guess(
                start_state, times, n, n_states, n_controls, None, scaling))
        # each guess is [states; controls; mid]; take states from each and
        # controls from the first, since the controls are shared
        cut = n_states * (n + 1)
        x0 = ca.vertcat(*[g[:cut] for g in guesses], guesses[0][cut:])

        solution = solver(
            x0=x0,
            lbx=ca.vertcat(*[ca.vec(ca.DM(state_lo))
                             for _ in states],
                           ca.vec(ca.DM(control_lo)),
                           ca.vec(ca.DM(mid_lo))),
            ubx=ca.vertcat(*[ca.vec(ca.DM(state_hi))
                             for _ in states],
                           ca.vec(ca.DM(control_hi)),
                           ca.vec(ca.DM(mid_hi))),
            lbg=ca.vertcat(*[ca.DM(np.asarray(b, dtype=float).reshape(-1, 1))
                             for b in lower]),
            ubg=ca.vertcat(*[ca.DM(np.asarray(b, dtype=float).reshape(-1, 1))
                             for b in upper]),
        )
        stats = solver.stats()
        values = np.array(solution["x"]).ravel()

        out_states, offset = [], 0
        for _ in states:
            block = values[offset:offset + cut].reshape(
                n_states, n + 1, order="F") * scaling.state[:, None]
            out_states.append(block)
            offset += cut
        controls = values[offset:offset + n_controls * (n + 1)].reshape(
            n_controls, n + 1, order="F") * scaling.control[:, None]

        progress = np.array([
            float(ca.DM(self.base.progress(ca.DM([s[0, -1], s[1, -1]])))[0]
                  - ca.DM(self.base.progress(ca.DM([s[0, 0], s[1, 0]])))[0])
            for s in out_states])

        return StochasticPlan(
            time=times, controls=controls, states=out_states,
            progress=progress, scenarios=self.scenarios,
            success=bool(stats.get("success", False)), stats=stats)
