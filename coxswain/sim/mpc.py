"""Model predictive control for steering the boat down a line.

:class:`~coxswain.sim.guidance.PathFollower` is reactive: it looks at where
the boat is now and corrects.  A coxswain is not reactive.  They put the
rudder on *before* the bend, because they can see the bend coming and they
know how slowly the boat answers.  That difference showed up directly --
the reactive law held 2.7 m of cross-track on the straight and 4.5 m
through the Weeks turn, losing exactly where anticipation matters.

Model predictive control is the principled version of anticipating.  At
each step it solves a small optimal control problem over the next few
seconds:

* roll the boat forward under a model of its dynamics,
* score the predicted path against the line it should be on,
* pick the rudder and pressure that score best,
* apply only the first of them, then throw the rest away and solve again.

The last point is what makes it robust.  The plan is always wrong -- the
model is a reduced one and the river is not exactly where the survey says
-- but only the first control is ever used, so the error never accumulates.

Why a reduced model inside the controller
-----------------------------------------
The prediction runs on :class:`~coxswain.river.trajectory.ReducedModel`,
not the 6-DOF simulator, and that is deliberate on two counts.  A real
coxswain steers with an internal model far cruder than the boat itself;
and a controller that needs a 6-DOF solve per step is not one anybody
could run.  The mismatch between the two is then a *feature of the
experiment*: if MPC on a five-state model steers the full boat well, that
says the reduced model captures what matters for steering.

The cost
--------
Four terms, and the weights are the whole design:

``cross-track``
    Distance from the line.  The thing being asked for.
``heading``
    Alignment with the line's direction.  Without it the boat can sit on
    the line while pointing across it, which is worse than being off it.
``rudder``
    Effort, and its rate.  Rudder is drag; a controller that saws the
    blade about is slower than one that does not, even when both hold the
    line.
``split``
    Pressure, weighted hard.  A coxswain calls for it in bursts, so it
    should be expensive enough that the optimiser reaches for the rudder
    first and the crew second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["PathMPC"]


@dataclass
class PathMPC:
    """Receding-horizon steering, solved with CasADi and IPOPT.

    Usable as a ``rudder_override`` on
    :class:`~coxswain.sim.control.Coxswain`, so it drops into the same
    seam the reactive follower uses.
    """

    path: np.ndarray
    #: The model the controller predicts with.  **Fit this to the boat.**
    #: Left at the documented defaults it carries ``yaw_inertia`` 22000
    #: against a real 9489 and ``yaw_control`` 539 against 411 -- a
    #: controller believing the boat 2.3x more sluggish than it is
    #: over-commands, the boat answers quicker than predicted, and it
    #: overshoots.  That produced a steady weave of about +/-8 m about the
    #: line, which reads as the boat simply not being on it.
    model: object = None
    horizon: float = 6.0                 # seconds to look ahead
    steps: int = 12                      # collocation steps over it
    interval: float = 0.20               # seconds between re-solves

    weight_cross: float = 4.0
    weight_heading: float = 12.0
    weight_rudder: float = 0.8
    weight_rudder_rate: float = 2.5
    weight_split: float = 40.0

    max_rudder: Optional[float] = None
    max_split: float = 0.30

    # -- state carried between calls --------------------------------------
    split: float = field(default=0.0, init=False)
    cross_track: float = field(default=0.0, init=False)
    station: float = field(default=0.0, init=False)
    solves: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        from ..river.trajectory import ReducedModel

        self.path = np.asarray(self.path, dtype=float)[:, :2]
        step = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.distance = np.concatenate([[0.0], np.cumsum(step)])
        heading = np.arctan2(np.gradient(self.path[:, 1]),
                             np.gradient(self.path[:, 0]))
        self.heading = np.unwrap(heading)
        if self.model is None:
            self.model = ReducedModel()
        if self.max_rudder is None:
            self.max_rudder = float(self.model.rudder_limit)
        self._last = 0
        self._held = 0.0
        self._plan = None
        self._plan_time = 0.0
        self._next_solve = -1.0
        self._filtered_rate = 0.0
        self._build()

    # -- the optimal control problem --------------------------------------
    def _build(self) -> None:
        """Assemble the CasADi problem once; re-solved with new parameters."""
        import casadi as ca

        n, dt = self.steps, self.horizon / self.steps
        model = self.model
        opti = ca.Opti()

        state = opti.variable(3, n + 1)      # cross-track, heading error, r
        rudder = opti.variable(1, n)
        split = opti.variable(1, n)

        start = opti.parameter(3)
        speed = opti.parameter()
        curvature = opti.parameter(n)        # of the line, over the horizon
        previous = opti.parameter()          # last rudder, for the rate term

        opti.subject_to(state[:, 0] == start)

        cost = 0.0
        for k in range(n):
            error, psi, r = state[0, k], state[1, k], state[2, k]
            # Frenet kinematics along the line: the boat drifts off at the
            # rate its heading differs from the line's, and the heading
            # error grows with yaw rate less the line's own turning.
            derror = speed * ca.sin(psi)
            dpsi = r - speed * curvature[k]
            # **Positive rudder yaws to starboard**, which is a *negative*
            # yaw rate in this frame -- the convention
            # :class:`~coxswain.sim.control.HeadingController` uses and the
            # one the 6-DOF simulator implements.  Getting this backwards
            # does not merely steer the wrong way: the boat runs off, the
            # cross-track term grows without bound, the problem becomes
            # badly conditioned and the solver starts failing, which holds
            # the last command and makes it worse.  84% of solves failed
            # and the boat ended 691 m off the line.
            moment = (-model.yaw_control * speed ** 2 * rudder[0, k]
                      - model.split_control * split[0, k]
                      - model.yaw_damping * speed * r)
            dr = moment / model.yaw_inertia

            opti.subject_to(state[0, k + 1] == error + dt * derror)
            opti.subject_to(state[1, k + 1] == psi + dt * dpsi)
            opti.subject_to(state[2, k + 1] == r + dt * dr)

            previous_rudder = previous if k == 0 else rudder[0, k - 1]
            cost += (self.weight_cross * error ** 2
                     + self.weight_heading * psi ** 2
                     + self.weight_rudder * rudder[0, k] ** 2
                     + self.weight_rudder_rate
                     * (rudder[0, k] - previous_rudder) ** 2
                     + self.weight_split * split[0, k] ** 2)

        # a terminal term, so the horizon does not end mid-correction
        cost += 4.0 * self.weight_cross * state[0, n] ** 2
        cost += 4.0 * self.weight_heading * state[1, n] ** 2

        opti.subject_to(opti.bounded(-self.max_rudder, rudder,
                                     self.max_rudder))
        opti.subject_to(opti.bounded(-self.max_split, split, self.max_split))
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "max_iter": 120, "sb": "yes",
                     "acceptable_tol": 1e-4, "acceptable_iter": 8})

        self._opti = opti
        self._vars = (state, rudder, split)
        self._params = (start, speed, curvature, previous)

    # -- geometry ---------------------------------------------------------
    def nearest(self, point) -> int:
        lo = self._last
        hi = min(lo + 600, len(self.path))
        index = lo + int(np.argmin(
            np.linalg.norm(self.path[lo:hi] - point[:2], axis=1)))
        self._last = min(index, len(self.path) - 2)
        return self._last

    def _line_curvature(self, index, speed) -> np.ndarray:
        """Curvature of the line at each horizon step, ahead of the boat."""
        dt = self.horizon / self.steps
        out = np.zeros(self.steps)
        for k in range(self.steps):
            ahead = self.distance[index] + speed * dt * k
            j = int(np.searchsorted(self.distance, ahead))
            j = int(np.clip(j, 1, len(self.path) - 2))
            span = self.distance[j + 1] - self.distance[j - 1]
            if span > 1e-6:
                out[k] = (self.heading[j + 1] - self.heading[j - 1]) / span
        return out

    # -- the control law --------------------------------------------------
    def __call__(self, t: float, state) -> float:
        import casadi as ca
        from ..core.frames import wrap_to_pi

        position = np.asarray(state.position, dtype=float)[:2]
        index = self.nearest(position)
        self.station = float(self.distance[index])

        tangent = np.array([np.cos(self.heading[index]),
                            np.sin(self.heading[index])])
        across = np.array([-tangent[1], tangent[0]])
        self.cross_track = float(np.dot(position - self.path[index], across))
        error = wrap_to_pi(state.yaw - float(self.heading[index]))
        speed = max(float(np.linalg.norm(np.asarray(state.velocity)[:2])), 0.5)

        # The boat's yaw time constant is I / (Nr u), about 0.06 s -- far
        # shorter than the re-solve interval.  Freezing the first command
        # between solves therefore throws away most of the boat's
        # bandwidth, and cost the first version of this controller more
        # than its anticipation gained: 5.54 m of cross-track against the
        # reactive law's 4.51 m.  Between solves, *fly the plan* -- the
        # optimiser already computed a rudder trajectory, so follow it.
        if self._plan is not None and t < self._next_solve:
            elapsed = t - self._plan_time
            step = self.horizon / self.steps
            self._held = float(np.interp(elapsed,
                                         np.arange(self.steps) * step,
                                         self._plan))
            return float(np.clip(self._held, -self.max_rudder,
                                 self.max_rudder))
        self._next_solve = t + self.interval

        opti = self._opti
        start, speed_p, curvature_p, previous_p = self._params
        # Yaw rate swings hard within each stroke -- the rig's own couple
        # and the roll coupling -- so the raw value makes the plan jump
        # every solve.  A coxswain does not react to within-stroke wobble
        # either; they feel the average.
        raw = float(state.omega_hull[2])
        # Short enough to take the within-stroke swing off without adding
        # phase lag the loop then has to fight -- but not so short that the
        # stroke's own yaw swing reaches the solver, which at 0.25 s put
        # the failure rate above 50%.  The plan is re-solved five times a
        # second; it does not need to chase a signal that oscillates twice
        # that fast.
        blend = np.exp(-self.interval / 0.45)
        self._filtered_rate = blend * self._filtered_rate + (1 - blend) * raw
        opti.set_value(start, [self.cross_track, error, self._filtered_rate])
        opti.set_value(speed_p, speed)
        opti.set_value(curvature_p, self._line_curvature(index, speed))
        opti.set_value(previous_p, self._held)

        # Warm start from a forward roll of the model.  Without a guess
        # IPOPT begins at zero, which violates every dynamics equality at
        # once; with one it converges in about a dozen iterations.
        guess = np.zeros((3, self.steps + 1))
        guess[:, 0] = [self.cross_track, error, float(state.omega_hull[2])]
        step = self.horizon / self.steps
        for k in range(self.steps):
            e_k, psi_k, r_k = guess[:, k]
            guess[0, k + 1] = e_k + step * speed * np.sin(psi_k)
            guess[1, k + 1] = psi_k + step * r_k
            guess[2, k + 1] = r_k - step * (self.model.yaw_damping * speed
                                            * r_k) / self.model.yaw_inertia
        opti.set_initial(self._vars[0], guess)

        try:
            solution = opti.solve()
            _state, rudder, split = self._vars
            plan = np.atleast_1d(solution.value(rudder)).ravel()
            self._plan = plan
            self._plan_time = t
            self._held = float(plan[0])
            self.split = float(np.atleast_1d(solution.value(split)).ravel()[0])
            self.solves += 1
        except RuntimeError:
            # Keep flying the last plan rather than freezing: it was
            # optimal a fifth of a second ago and still nearly is.  This
            # is a fallback, not a mode of operation -- if it fires more
            # than a few per cent of the time the problem is conditioned
            # badly and the controller is no longer doing what it says.
            # A failed solve is not a crisis: hold the last command.  It
            # was optimal a quarter of a second ago and the boat has not
            # moved far since.
            self.failures += 1
        return float(np.clip(self._held, -self.max_rudder, self.max_rudder))
