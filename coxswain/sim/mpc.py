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

The cost, and what the weights are actually for
-----------------------------------------------
Four terms.  The weights were tuned for a long time against cross-track
error, which is the wrong objective and was quietly deciding things:
scored on tracking, tighter is always better, so the tuning pressure runs
one way and never stops.

Scored on the clock -- ``scripts/mpc_tune.py`` times the boat between two
fixed gates on the water -- **it runs the other way.**  Over the Weeks
turn, raising ``weight_cross`` from 2 to 120 improves tracking from 1.41 m
to 1.05 m rms and costs **1.96 seconds**, which over a full course is
about 12.7 s.  The distance column is what makes that legible: the tightly
tracked boat travels *one metre less* and still loses two seconds, so
every bit of the loss is rudder and yaw drag bought with helm.

A controller that holds the line beautifully is not the goal.  The line is
already the fast way round; holding it to the last centimetre costs more
than the centimetres are worth.  The weights below are therefore a
compromise the clock chose, not the tightest tracking available.

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

    #: ``"qp"`` linearises the one nonlinearity and solves an exact
    #: quadratic program; ``"nlp"`` is the original IPOPT formulation,
    #: kept so the two can be compared rather than asserted.
    #:
    #: Everything in this problem is already linear-quadratic except
    #: ``de/dt = u sin(psi)``.  Linearising *that* about the measured
    #: heading error -- not about zero, so it stays accurate at large
    #: errors -- turns a nonconvex program solved by an interior-point
    #: method with an iteration limit into a convex one solved exactly.
    #: An active-set QP either returns the optimum or reports genuine
    #: infeasibility; it cannot quietly run out of iterations, which is
    #: what the fallback path was absorbing.
    solver: str = "qp"

    #: Estimate the unmodelled yaw moment and feed it forward.
    #:
    #: The reduced model has no rig bias, no crosswind weathercocking and
    #: no current shear, and the boat has all three -- a standard-rigged
    #: eight carries a standing yaw of about 0.19 deg/s with the rudder
    #: centred.  Without an estimate the controller can only ever chase
    #: that with cross-track feedback, which means living with a steady
    #: offset.  A one-state disturbance observer lets it hold a standing
    #: rudder instead, which is what a coxswain does without thinking.
    estimate_bias: bool = True
    bias_gain: float = 0.25

    # -- state carried between calls --------------------------------------
    split: float = field(default=0.0, init=False)
    cross_track: float = field(default=0.0, init=False)
    station: float = field(default=0.0, init=False)
    solves: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    last_error: str = field(default="", init=False)
    #: Estimated unmodelled yaw moment, N m.  Diagnostic as well as
    #: control: a large steady value means the boat has a bias the model
    #: does not, which is a rigging or a wind finding, not a controller one.
    bias: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        from ..river.trajectory import ReducedModel

        self.path = np.asarray(self.path, dtype=float)[:, :2]
        step = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.distance = np.concatenate([[0.0], np.cumsum(step)])
        heading = np.arctan2(np.gradient(self.path[:, 1]),
                             np.gradient(self.path[:, 0]))
        self.heading = np.unwrap(heading)

        # Curvature, smoothed to boat scale and clipped to what a shell
        # can do.  Raw two-point differences on a path sampled every
        # metre turn the route's piecewise-linear knots into spikes of
        # 0.057 1/m -- a 17 m radius, a 15 deg/s demand -- and feeding
        # that to the solver as feedforward hands it a constraint no
        # rudder satisfies: two thirds of all solves died at maximum
        # iterations, on every real path and never on a synthetic
        # straight.  A 17.3 m hull cannot respond to curvature structure
        # shorter than itself, so smoothing over two boat lengths loses
        # nothing that was ever steerable.
        raw = np.gradient(self.heading, np.maximum(self.distance, 1e-9))
        window = max(int(round(35.0 / max(np.median(np.diff(
            self.distance)), 1e-6))), 1)
        kernel = np.ones(window) / window
        smooth = np.convolve(raw, kernel, mode="same")
        self.curvature = np.clip(smooth, -0.015, 0.015)
        if self.model is None:
            self.model = ReducedModel()
        if self.max_rudder is None:
            self.max_rudder = float(self.model.rudder_limit)
        self._last = 0
        self._held = 0.0
        self._plan = None
        self._split_plan = None
        self._plan_time = 0.0
        self._prev_time = None
        self._prev_rate = 0.0
        self._prev_moment = 0.0
        self.terminal_weight = None
        self._next_solve = -1.0
        self._filtered_rate = 0.0
        self._build()

    # -- the optimal control problem --------------------------------------
    def _terminal_weight(self, speed: float, decay: float, dt: float):
        """Terminal cost from the infinite-horizon solution, not a guess.

        The original terminal term was ``4x`` the running weights, chosen
        because a horizon that simply stops mid-correction lets the
        optimiser park the boat off the line at step N and pay nothing
        for it.  Four is a reasonable guess; the Riccati solution is the
        right answer.  Weight the terminal state by ``P`` from the
        discrete algebraic Riccati equation and the finite horizon
        inherits the stability of the infinite one.

        Returns ``None`` if the DARE will not solve, and the caller falls
        back to the old heuristic.
        """
        import numpy as np

        try:
            from scipy.linalg import solve_discrete_are
        except Exception:                                # noqa: BLE001
            return None
        model = self.model
        gain = ((1.0 - decay) * model.yaw_control * speed ** 2
                / (model.yaw_damping * speed))
        a = np.array([[1.0, dt * speed, 0.0],
                      [0.0, 1.0, dt],
                      [0.0, 0.0, decay]])
        b = np.array([[0.0], [0.0], [-gain]])
        q = np.diag([self.weight_cross, self.weight_heading, 0.0])
        r = np.array([[self.weight_rudder + self.weight_rudder_rate]])
        try:
            return solve_discrete_are(a, b, q, r)
        except Exception:                                # noqa: BLE001
            return None

    def _build(self) -> None:
        """Assemble the problem once; re-solved with new parameters."""
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
        # Linearisation point for the one nonlinear term, and the
        # estimated unmodelled yaw moment.  Both are parameters, so the
        # program stays quadratic and is re-linearised every solve.
        psi_ref = opti.parameter()
        bias = opti.parameter()

        opti.subject_to(state[:, 0] == start)

        decay = ca.exp(-dt * model.yaw_damping * speed / model.yaw_inertia)
        cost = 0.0
        for k in range(n):
            error, psi, r = state[0, k], state[1, k], state[2, k]
            # Frenet kinematics along the line: the boat drifts off at the
            # rate its heading differs from the line, and the heading error
            # grows with yaw rate less the line own turning.
            #
            # ``sin(psi)`` is the ONLY nonlinearity in the whole problem.
            # In "qp" mode it is replaced by its tangent at the measured
            # heading error -- not at zero, which would be a small-angle
            # approximation good only near the line, but at wherever the
            # boat actually is.  The program becomes convex and exactly
            # solvable while staying accurate at large errors.
            if self.solver == "qp":
                derror = speed * (ca.sin(psi_ref)
                                  + ca.cos(psi_ref) * (psi - psi_ref))
            else:
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
            control_moment = (-model.yaw_control * speed ** 2 * rudder[0, k]
                              - model.split_control * split[0, k]
                              + bias)

            opti.subject_to(state[0, k + 1] == error + dt * derror)
            opti.subject_to(state[1, k + 1] == psi + dt * dpsi)
            # The yaw-rate dynamics are discretised EXACTLY, not by Euler.
            # Their time constant is I/(N_r u) -- about 0.06 s -- against a
            # 0.5 s transcription step, so forward Euler amplification is
            # 1 - dt/tau = roughly MINUS EIGHT: the transcribed recursion
            # explodes even though the boat it describes is the most docile
            # system imaginable.  The dynamics are linear in r with the
            # control constant over the step, so the zero-order-hold form
            #     r+ = phi r + (1 - phi) M / (N_r u),  phi = exp(-dt/tau)
            # is exact and unconditionally stable at any step length.
            opti.subject_to(state[2, k + 1] == decay * r
                            + (1.0 - decay) * control_moment
                            / (model.yaw_damping * speed))

            previous_rudder = previous if k == 0 else rudder[0, k - 1]
            cost += (self.weight_cross * error ** 2
                     + self.weight_heading * psi ** 2
                     + self.weight_rudder * rudder[0, k] ** 2
                     + self.weight_rudder_rate
                     * (rudder[0, k] - previous_rudder) ** 2
                     + self.weight_split * split[0, k] ** 2)

        # A terminal term, so the horizon does not end mid-correction.
        reference = float(getattr(self.model, "reference_speed", 0.0) or 4.5)
        phi = float(np.exp(-dt * model.yaw_damping * reference
                           / model.yaw_inertia))
        terminal = self._terminal_weight(reference, phi, dt)
        if terminal is None:
            cost += 4.0 * self.weight_cross * state[0, n] ** 2
            cost += 4.0 * self.weight_heading * state[1, n] ** 2
        else:
            final = state[:, n]
            cost += ca.mtimes([final.T, ca.DM(terminal), final])
        self.terminal_weight = terminal

        opti.subject_to(opti.bounded(-self.max_rudder, rudder,
                                     self.max_rudder))
        opti.subject_to(opti.bounded(-self.max_split, split, self.max_split))
        opti.minimize(cost)
        if self.solver == "qp":
            # A problem that is already quadratic converges in two SQP
            # steps -- one to the optimum, one to notice the step is zero
            # -- so the iteration budget exists only to let it *say* so.
            # Set to 1 it solved the problem correctly every time and
            # reported Maximum_Iterations_Exceeded anyway, which the
            # caller counted as a failure: 99.6% "failures" that were
            # each holding a perfectly good answer at arm's length.
            # An active-set method returns the optimum or reports genuine
            # infeasibility; it has no iteration limit to quietly hit,
            # which is what the fallback path was absorbing.
            opti.solver("sqpmethod",
                        {"print_time": False, "qpsol": "qrqp",
                         "print_iteration": False, "print_header": False,
                         "print_status": False, "max_iter": 12,
                         "qpsol_options": {"print_iter": False,
                                           "print_header": False,
                                           "error_on_fail": False}})
        else:
            opti.solver("ipopt", {"print_time": False},
                        {"print_level": 0, "max_iter": 600, "sb": "yes",
                         "acceptable_tol": 1e-3, "acceptable_iter": 4})

        self._opti = opti
        self._vars = (state, rudder, split)
        self._params = (start, speed, curvature, previous, psi_ref, bias)

    # -- geometry ---------------------------------------------------------
    def nearest(self, point) -> int:
        lo = self._last
        hi = min(lo + 600, len(self.path))
        index = lo + int(np.argmin(
            np.linalg.norm(self.path[lo:hi] - point[:2], axis=1)))
        self._last = min(index, len(self.path) - 2)
        return self._last

    def _line_curvature(self, index, speed) -> np.ndarray:
        """Curvature of the line at each horizon step, ahead of the boat.

        Sampled from the precomputed smoothed-and-clipped array; see
        ``__post_init__`` for why the raw differences must never reach
        the solver.
        """
        dt = self.horizon / self.steps
        out = np.zeros(self.steps)
        for k in range(self.steps):
            ahead = self.distance[index] + speed * dt * k
            j = int(np.clip(np.searchsorted(self.distance, ahead),
                            0, len(self.curvature) - 1))
            out[k] = self.curvature[j]
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
        (start, speed_p, curvature_p, previous_p, psi_ref_p,
         bias_p) = self._params
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

        # -- what the model did not predict, fed back as a moment --------
        # A one-state disturbance observer.  Roll the model forward from
        # the last solve under the control that was actually applied, and
        # charge the difference between that and the measured yaw rate to
        # an unmodelled moment.  It picks up the rig couple, a crosswind
        # weathervane and current shear -- none of which the reduced model
        # contains and all of which the boat has.
        #
        # Without it the only way to hold a line against a standing yaw
        # bias is a standing cross-track error, because cross-track
        # feedback is the sole path to a non-zero rudder.  With it the
        # controller carries the trim on the feedforward, which is what a
        # coxswain does with a thumb on the strings and no thought at all.
        if self.estimate_bias and self._prev_time is not None:
            elapsed = max(t - self._prev_time, 1e-3)
            tau = self.model.yaw_inertia / max(
                self.model.yaw_damping * speed, 1e-6)
            phi = float(np.exp(-elapsed / tau))
            steady = ((self._prev_moment + self.bias)
                      / max(self.model.yaw_damping * speed, 1e-6))
            predicted = phi * self._prev_rate + (1.0 - phi) * steady
            innovation = self._filtered_rate - predicted
            self.bias += (self.bias_gain * innovation
                          * self.model.yaw_damping * speed)
            # Never let the estimate exceed what the rudder could trim; a
            # bias beyond that is not a bias the controller can use, and
            # an unbounded integrator is how observers turn a transient
            # into a runaway.
            authority = (self.model.yaw_control * speed ** 2
                         * self.max_rudder)
            self.bias = float(np.clip(self.bias, -authority, authority))

        opti.set_value(start, [self.cross_track, error, self._filtered_rate])
        opti.set_value(speed_p, speed)
        opti.set_value(curvature_p, self._line_curvature(index, speed))
        opti.set_value(previous_p, self._held)
        # Linearise the one nonlinear term about where the boat actually
        # is, so the quadratic program stays faithful at large heading
        # errors instead of only near the line.
        opti.set_value(psi_ref_p, error if self.solver == "qp" else 0.0)
        opti.set_value(bias_p, self.bias if self.estimate_bias else 0.0)

        # Warm start from a forward roll of the model.  Without a guess
        # IPOPT begins at zero, which violates every dynamics equality at
        # once; with one it converges in about a dozen iterations.
        # The guess must satisfy the SAME dynamics the constraints impose,
        # from the SAME initial state.  This one started from the raw yaw
        # rate while the constraint started from the filtered one, and it
        # propagated heading without the curvature feedforward the
        # constraint contains -- so the guess violated the k=0 equality by
        # construction and misjudged every curved horizon.
        curvature_values = self._line_curvature(index, speed)
        guess = np.zeros((3, self.steps + 1))
        guess[:, 0] = [self.cross_track, error, self._filtered_rate]
        step = self.horizon / self.steps
        for k in range(self.steps):
            e_k, psi_k, r_k = guess[:, k]
            guess[0, k + 1] = e_k + step * speed * np.sin(psi_k)
            guess[1, k + 1] = psi_k + step * (r_k - speed
                                              * curvature_values[k])
            guess[2, k + 1] = r_k - step * (self.model.yaw_damping * speed
                                            * r_k) / self.model.yaw_inertia
        opti.set_initial(self._vars[0], guess)

        # Controls must be warm-started too, and from something sane.
        # ``set_initial`` was only ever given the states, so rudder and
        # split began each solve wherever the PREVIOUS solve's last
        # iterate left them -- after a clean solve that is a good start,
        # but after a failed one it is the failed solve's garbage, and the
        # next solve inherits it.  One bad solve then seeds the next,
        # which is how a controller ends up failing every other time and
        # calling it fifty per cent.  The receding-horizon warm start is
        # the previous plan shifted by the time that has passed, padded
        # with its own tail.
        if self._plan is not None and len(self._plan) == self.steps:
            shift = max(int(round((t - self._plan_time)
                                  / (self.horizon / self.steps))), 0)
            rolled = np.concatenate([
                self._plan[shift:],
                np.full(min(shift, self.steps), self._plan[-1])])
            opti.set_initial(self._vars[1], rolled[:self.steps][None, :])
        else:
            opti.set_initial(self._vars[1],
                             np.full((1, self.steps), self._held))
        # The split was warm-started from zeros every single solve, which
        # threw away a perfectly good previous answer and asked the solver
        # to rediscover it.  Shift it the same way the rudder is shifted.
        if self._split_plan is not None and len(self._split_plan) == self.steps:
            shift = max(int(round((t - self._plan_time)
                                  / (self.horizon / self.steps))), 0)
            rolled = np.concatenate([
                self._split_plan[shift:],
                np.full(min(shift, self.steps), self._split_plan[-1])])
            opti.set_initial(self._vars[2], rolled[:self.steps][None, :])
        else:
            opti.set_initial(self._vars[2], np.zeros((1, self.steps)))

        try:
            solution = opti.solve()
            _state, rudder, split = self._vars
            plan = np.atleast_1d(solution.value(rudder)).ravel()
            self._plan = plan
            self._plan_time = t
            self._held = float(plan[0])
            split_plan = np.atleast_1d(solution.value(split)).ravel()
            self._split_plan = split_plan
            self.split = float(split_plan[0])
            self.solves += 1
        except RuntimeError as error:
            self.last_error = str(error)[:600]
            # Keep flying the last plan rather than freezing: it was
            # optimal a fifth of a second ago and still nearly is.  This
            # is a fallback, not a mode of operation -- if it fires more
            # than a few per cent of the time the problem is conditioned
            # badly and the controller is no longer doing what it says.
            # A failed solve is not a crisis: hold the last command.  It
            # was optimal a quarter of a second ago and the boat has not
            # moved far since.
            self.failures += 1
        # Whatever happened above, this is the command the boat will now
        # fly, so this is the moment the observer must predict from.
        self._prev_time = t
        self._prev_rate = self._filtered_rate
        self._prev_moment = (-self.model.yaw_control * speed ** 2 * self._held
                             - self.model.split_control * self.split)
        return float(np.clip(self._held, -self.max_rudder, self.max_rudder))
