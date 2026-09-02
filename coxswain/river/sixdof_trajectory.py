"""Trajectory optimisation over the full six-degree-of-freedom dynamics.

This is the one that does not reduce anything.

``trajectory.solve_trajectory`` optimises a :class:`~.trajectory.ReducedModel`
-- a stroke-averaged planar surrogate fitted to the real boat.  It is fast
and it was useful for finding a route, but the thing it optimises is not
the thing this project simulates.  A stroke-averaged model cannot express
the question the whole exercise is about: whether the rudder should be
doing something different on the drive than on the recovery.

What is optimised here
----------------------
:meth:`~.sixdof.SixDofModel.derivative` exactly, at every collocation
point:

* all thirteen states -- three positions, three Euler angles, three linear
  and three angular velocities, and the crew's anaerobic reserve;
* the hull's submerged geometry as a function of heave, pitch **and** roll,
  from the tabulated exact mesh;
* nonlinear hydrodynamics, including the shallow-water increment evaluated
  at the depth under the boat *at that point on the river*;
* oar loads from the real rig, including the sweep rig's own asymmetry and
  the crew's per-side power split;
* the slip-dependent blade model, whose efficiency depends on the boat's
  instantaneous speed and so cannot be precomputed;
* the crew's balance effort applied through the riggers, with the pitch
  coupling that a sweep rig imposes on it.

The only approximations are the ones IPOPT needs: bspline interpolants
standing in for the branch-laden wetted-surface clip and for the raster
lookups, and Fourier series for the crew's prescribed motion. Both are
refined against a stated error bound rather than assumed -- see
:meth:`~.hullsurrogate.HullSurrogate.validate` and
:meth:`~.strokemodel.StrokePeriodicFit.fit_to_tolerance`.

Why the horizon is a number of strokes
--------------------------------------
The stroke rate is prescribed, so a horizon of ``n_strokes`` is a fixed
duration and minimum-time has nothing to vary. The natural objective is
therefore the other one: **cover as much of the course as possible in a
given number of strokes.** That is also what a crew actually does.

Progress is measured along the channel, not as straight-line distance --
the reach bends, and rewarding displacement would pay the boat to cut
across land.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .collocation import HermiteSimpson, phase_locked_mesh
from .scaling import ProblemScaling
from ..hydro.appendages import MAX_RUDDER_DEFLECTION

__all__ = ["SixDofPlan", "SixDofTrajectory", "solve_sixdof_trajectory"]


@dataclass
class SixDofPlan:
    """Everything the solve produced."""

    time: np.ndarray              # (n_nodes,)
    states: np.ndarray            # (13, n_nodes)
    controls: np.ndarray          # (3, n_nodes)
    mesh: Sequence
    progress: float               # metres of course covered
    success: bool
    stats: dict
    #: Raw scaled NLP solution vector.  Kept so a receding-horizon driver
    #: can warm start the next block from this one: consecutive blocks
    #: share a mesh shape, and the boat is doing much the same thing a
    #: few strokes later, so the previous solution is a far better guess
    #: than a fresh dynamics rollout.  Without it every block re-solved
    #: from scratch, which is what made the corrected -- and correctly
    #: harder to steer -- boat stall at the station-450 pinch.
    solution: np.ndarray = None

    @property
    def position(self):
        return self.states[0:3]

    @property
    def attitude(self):
        return self.states[3:6]

    @property
    def rudder(self):
        return self.controls[0]

    @property
    def split(self):
        return self.controls[1]

    @property
    def power(self):
        return self.controls[2]

    def phase_of(self, index: int) -> str:
        return self.mesh[min(index, len(self.mesh) - 1)].phase

    def rudder_by_phase(self):
        """Mean rudder on the drive and on the recovery.

        The number the whole phase-locked mesh exists to produce: if these
        differ, steering through the stroke is doing something a constant
        angle cannot.
        """
        drive, recovery = [], []
        for k, interval in enumerate(self.mesh):
            (drive if interval.phase == "drive" else recovery).append(
                self.controls[0, k])
        return float(np.mean(drive)), float(np.mean(recovery))

    @property
    def anaerobic(self):
        return self.states[12]


class SixDofTrajectory:
    """Builds and solves the NLP.  Kept as a class so the expensive parts --
    the hull surrogate, the Fourier fits, the raster interpolants -- are
    built once and reused across solves with different horizons."""

    def __init__(self, boat, raster, model=None, blade=None,
                 progress_field=None, margin: float = 2.0,
                 guess_step: float = 0.004):
        import casadi as ca

        from .sixdof import SixDofModel

        self.boat = boat
        self.raster = raster
        self.model = model if model is not None else SixDofModel(
            boat, blade=blade)
        self.margin = float(margin)
        #: Explicit step for the initial-guess rollout.  Set by the stiffest
        #: mode in the model, which is the crew's balance loop, not by the
        #: accuracy the guess needs.
        self.guess_step = float(guess_step)

        #: How far the blades reach from the centreline.  The hull is
        #: 0.57 m in the beam and irrelevant; what hits the bank is a blade.
        lock = boat.rig.seats[0].oarlocks[0]
        self.blade_reach = float(abs(lock.position[1]) + lock.oar.outboard)

        self._ca = ca
        self._build_lookups(progress_field)

    # -- the river, as differentiable functions ---------------------------
    def _build_lookups(self, progress_field):
        ca = self._ca
        raster = self.raster
        grid = [raster.east.tolist(), raster.north.tolist()]

        def interpolant(name, values, fill):
            filled = np.array(values, dtype=float)
            filled[~np.isfinite(filled)] = fill
            return ca.interpolant(name, "bspline", grid,
                                  filled.T.ravel(order="F").tolist())

        # Depth drives the shallow-water increment and the blade blockage.
        # Outside the water it is floored rather than left nan: the
        # clearance constraint keeps the boat inside, and a nan there would
        # poison the derivative everywhere.
        self.depth = interpolant("depth", raster.depth, 0.3)
        self.clearance = interpolant("clearance", raster.clearance, 0.0)
        if progress_field is None:
            progress_field = build_progress_field(raster)
        self.progress = interpolant("progress", progress_field, 0.0)

        self.has_current = raster.current_east is not None
        if self.has_current:
            self.current_east = interpolant("cur_e", raster.current_east, 0.0)
            self.current_north = interpolant("cur_n", raster.current_north,
                                             0.0)

    def dynamics_function(self, wind=None):
        """The full 6-DOF derivative, with the river read from the state.

        ``wind`` is an optional object carrying ``wind_speed`` and
        ``wind_bearing``; when given, the aerodynamic loads of
        :mod:`coxswain.hydro.wind` are added.  Used by the stochastic
        solver, where each scenario has its own weather.
        """
        ca = self._ca
        state = ca.MX.sym("state", self.model.n_states)
        control = ca.MX.sym("control", self.model.n_controls)
        time = ca.MX.sym("time")

        where = ca.vertcat(state[0], state[1])
        depth = ca.fmax(self.depth(where), 0.3)
        # The wind is set on the model rather than passed down the call,
        # because it is a constant of the scenario and not a function of
        # the state.  Previously this argument was accepted and discarded.
        previous = getattr(self.model, "wind_abs", None)
        self.model.wind_abs = self.wind_vector(wind)
        try:
            derivative = self.model.derivative(state, control, time, depth)
        finally:
            if previous is not None:
                self.model.wind_abs = previous

        if self.has_current:
            # The boat moves through water that is itself moving, so the
            # ground track is the water-relative motion plus the current.
            # The dynamics are written relative to the water; only the
            # kinematics need correcting.
            drift = ca.vertcat(self.current_east(where),
                               self.current_north(where), 0.0)
            derivative = ca.vertcat(derivative[0:3] + drift,
                                    derivative[3:])
        return ca.Function("sixdof_river", [state, control, time],
                           [derivative])

    @staticmethod
    def wind_vector(wind):
        """``(east, north)`` m/s the wind blows TOWARDS, from a scenario.

        Scenarios carry ``wind_speed`` and ``wind_bearing``, the bearing
        being the direction blown towards in the model's maths frame (see
        :class:`~coxswain.river.stochastic.Scenario`).  Returns zeros for
        ``None`` so a windless solve is the default rather than an error.

        This replaces ``wind_note``, which returned its argument unchanged
        and claimed in its docstring that the wind was applied elsewhere.
        It was not applied anywhere.  See SOURCES sec. 69.
        """
        if wind is None:
            return np.zeros(2)
        speed = float(getattr(wind, "wind_speed", 0.0))
        bearing = float(getattr(wind, "wind_bearing", 0.0))
        return np.array([speed * np.cos(bearing), speed * np.sin(bearing)])

    def initial_state(self, position, heading, speed):
        """A start state that is actually consistent.

        The velocity states are in the **absolute** frame, so a boat
        pointing along ``heading`` and moving forwards has
        ``(u cos h, u sin h)`` -- not ``(u, 0)``.  Setting the latter with
        a nonzero heading asks the model for a boat crabbing sideways at
        racing speed, which produces enormous lateral load and rolls it
        over.  That is the model behaving correctly on a nonsense input,
        and it is an easy input to write by accident.
        """
        state = np.zeros(self.model.n_states)
        state[0], state[1] = float(position[0]), float(position[1])
        state[2] = float(self.boat.equilibrium_heave())
        state[5] = float(heading)
        state[6] = float(speed) * np.cos(float(heading))
        state[7] = float(speed) * np.sin(float(heading))
        state[12] = float(self.model.anaerobic_capacity)
        return state

    # -- transcription ----------------------------------------------------
    def solve(self, start_state, n_strokes: int = 6,
              drive_intervals: int = 6, recovery_intervals: int = 4,
              scheme=None, guess=None, max_iter: int = 400,
              rudder_limit: float = MAX_RUDDER_DEFLECTION,
              split_limit: float = 0.15,
              power_bounds=(0.70, 1.15), print_level: int = 0,
              exact_hessian: bool = False, scaling=None,
              smoothing_weight: float = 1e-2,
              terminal_weight: float = 5e-2,
              roll_weight: float = 2e-2,
              comfort: float = 1.2,
              comfort_weight: float = 6e-2):
        """Maximise progress along the channel over ``n_strokes``.

        ``scheme`` defaults to Hermite-Simpson.  Pass ``RadauIIA(s)`` to
        transcribe at order ``2s-1`` instead; the mesh and every constraint
        are unchanged, which is the point of keeping both behind one
        interface.
        """
        ca = self._ca
        scheme = HermiteSimpson if scheme is None else scheme
        mesh = phase_locked_mesh(self.boat.timing, n_strokes,
                                 drive_intervals, recovery_intervals)
        n = len(mesh)
        times = np.array([interval.start for interval in mesh]
                         + [mesh[-1].end])
        durations = np.array([interval.duration for interval in mesh])

        raw_dynamics = self.dynamics_function()
        n_states = self.model.n_states
        n_controls = self.model.n_controls

        # Non-dimensionalise.  The unscaled transcription spans six orders
        # of magnitude -- 1000 m of position against 0.02 rad of roll --
        # and an interior-point method inherits that conditioning directly.
        # This is a change of units, not of model; see coxswain.river.scaling.
        if scaling is None:
            travel = max(float(np.hypot(*(np.asarray(start_state)[6:8])))
                         * float(times[-1]), 1.0)
            scaling = ProblemScaling.for_six_dof(
                self.model, leg_length=travel,
                speed=max(float(np.hypot(*np.asarray(start_state)[6:8])), 1.0),
                rudder_limit=rudder_limit, split_limit=split_limit)
        self.scaling = scaling
        dynamics = scaling.scaled_dynamics(raw_dynamics, ca)
        state_scale = ca.DM(scaling.state.reshape(-1, 1))
        control_scale = ca.DM(scaling.control.reshape(-1, 1))

        state = ca.MX.sym("X", n_states, n + 1)
        control = ca.MX.sym("U", n_controls, n + 1)
        control_mid = ca.MX.sym("Um", n_controls, n)

        constraints = [scheme.defects(dynamics, state, control, control_mid,
                                      times, durations)]
        lower = [np.zeros(n_states * n)]
        upper = [np.zeros(n_states * n)]

        constraints.append(
            state[:, 0] - ca.DM(scaling.to_scaled_state(start_state)))
        lower.append(np.zeros(n_states))
        upper.append(np.zeros(n_states))

        # -- path constraints ---------------------------------------------
        # The blades must stay inside the navigable channel.  Minimum
        # distance from the edge, not tracking a centreline: the boat may
        # be anywhere it likes so long as it clears.
        clearance_scale = self.blade_reach + self.margin
        room = []
        for k in range(n + 1):
            where = ca.vertcat(state[0, k] * state_scale[0],
                               state[1, k] * state_scale[1])
            # Divided by the clearance the boat needs, so the constraint
            # reads "how many boat-widths of room is there, minus one" and
            # is O(1) like the defects and the reserve.  Left in metres it
            # was ~30 against their ~1, which is the same conditioning
            # mistake as leaving the variables unscaled.
            room.append((self.clearance(where)
                         - (self.blade_reach + self.margin))
                        / clearance_scale)
        constraints.append(ca.vertcat(*room))
        lower.append(np.zeros(n + 1))
        upper.append(np.full(n + 1, ca.inf))

        # the crew cannot spend reserve it does not have
        constraints.append(ca.reshape(state[12, :], n + 1, 1))
        lower.append(np.zeros(n + 1))
        upper.append(np.full(n + 1, ca.inf))

        # -- bounds --------------------------------------------------------
        state_lo = np.full((n_states, n + 1), -ca.inf)
        state_hi = np.full((n_states, n + 1), ca.inf)
        # Attitude stays inside the range the hull surrogate was sampled
        # over.  Outside it the bspline is extrapolating, and the buoyancy
        # it returns is not this hull.
        surrogate = self.model.surrogate
        state_lo[3, :] = float(surrogate.roll[0]) * 0.95
        state_hi[3, :] = float(surrogate.roll[-1]) * 0.95
        state_lo[4, :] = float(surrogate.pitch[0]) * 0.95
        state_hi[4, :] = float(surrogate.pitch[-1]) * 0.95
        state_lo[2, :] = float(surrogate.heave[0]) * 0.95
        state_hi[2, :] = float(surrogate.heave[-1]) * 0.95

        control_lo = np.zeros((n_controls, n + 1))
        control_hi = np.zeros((n_controls, n + 1))
        mid_lo = np.zeros((n_controls, n))
        mid_hi = np.zeros((n_controls, n))

        def fill(lo, hi, column, phase):
            # The model orders its controls (rudder, split, power).  Getting
            # this backwards silently fed the split into ``power ** 1.5``,
            # which is NaN for a negative split -- and a NaN in the Hessian
            # rather than in the residual, so IPOPT reported it from a
            # place unrelated to the cause.
            lo[0, column], hi[0, column] = -rudder_limit, rudder_limit
            if phase == "drive":
                lo[1, column], hi[1, column] = -split_limit, split_limit
            else:
                # Blades are out of the water on the recovery, so a
                # pressure split there is not a control but a fiction.
                lo[1, column] = hi[1, column] = 0.0
            lo[2, column], hi[2, column] = power_bounds

        for k in range(n):
            fill(control_lo, control_hi, k, mesh[k].phase)
            fill(mid_lo, mid_hi, k, mesh[k].phase)
        fill(control_lo, control_hi, n, mesh[-1].phase)

        # -- objective -----------------------------------------------------
        final = ca.vertcat(state[0, n] * state_scale[0],
                           state[1, n] * state_scale[1])
        first = ca.vertcat(state[0, 0] * state_scale[0],
                           state[1, 0] * state_scale[1])
        # Progress in metres, divided by the distance the boat would cover
        # anyway.  The objective is then O(1) and, more usefully, it means
        # the same thing on a four-stroke test and a full-course solve --
        # an unscaled objective would be twenty on one and five hundred on
        # the other, and IPOPT's convergence tolerances are absolute.
        objective_scale = max(float(scaling.state[0]), 1.0)
        made = (self.progress(final) - self.progress(first))             / objective_scale

        # A light penalty on rudder rate.  Without it the solver may
        # chatter between adjacent nodes, which is not a steering input a
        # coxswain could produce and would make the drive-versus-recovery
        # comparison meaningless.
        #
        # ``control`` is scaled, so ``rate`` is a fraction of full rudder
        # per interval and the weight is dimensionless.  Both terms of the
        # objective are now non-dimensional, which they were not when the
        # variables were scaled and this was left alone -- the weight then
        # meant something different depending on the rudder limit.
        #
        # This weight matters more than a regulariser usually does, because
        # on a straight reach with no power bias the rudder's split between
        # drive and recovery lies in the **null space of the objective**.
        # Progress is 20.12 m at every weight from 1e-4 to 1e0; only the
        # chatter changes, and the apparent drive-versus-recovery
        # difference tracks the chatter:
        #
        #     weight   drive-rec   chatter
        #     1e-4     -2.72 deg   0.680 deg
        #     1e-3     -2.34       0.259
        #     1e-2     -0.81       0.069
        #     1e-1     -0.11       0.009
        #     1e0      -0.01       0.002
        #
        # So a large reported difference is not a finding, it is an
        # under-determined direction being filled with noise.  Read
        # ``rudder_by_phase`` only when the chatter is well below the
        # difference being claimed.  The interesting case -- where the
        # split is genuinely determined -- is a bend, or a standing
        # port/starboard power bias, not a straight reach.
        rate = control[0, 1:] - control[0, :-1]
        smoothing = smoothing_weight * ca.sumsqr(rate)

        # Terminal room.  A receding horizon that only maximises progress
        # over its own block is myopic: it cuts the inside of every bend,
        # and the clearance it spends is not paid back because the next
        # block inherits the position.  Running the Weeks-Anderson section
        # that way, clearance fell 30 -> 17 -> 6 m over three blocks and
        # the solve went infeasible with the boat against the bank.
        #
        # Rewarding room at the end of the block is the standard fix: it is
        # a terminal cost standing in for the value of the states the block
        # cannot see.
        end = ca.vertcat(state[0, n] * state_scale[0],
                         state[1, n] * state_scale[1])
        terminal_room = self.clearance(end) / clearance_scale
        smoothing = smoothing - terminal_weight * terminal_room

        # Running clearance comfort.
        #
        # A terminal reward alone is not enough, and the Weeks-Anderson
        # section shows why.  Its centreline clearance runs 60 m through
        # stations 200-350 and pinches to 30 m at station 450.  Through
        # the wide part a linear terminal reward barely competes with
        # progress, so the boat corner-cuts; it then arrives at the pinch
        # some 22 m off the centreline and the solve fails with 8.5 m of
        # room at station 476.
        #
        # This term costs nothing while the boat is comfortably clear and
        # rises quadratically once it is not, which is the usual soft
        # treatment of a state constraint whose hard version is already
        # imposed above.  It shapes the approach without distorting the
        # optimum where there is room to spare -- the boat is still free
        # to be anywhere it likes, it just stops spending clearance it
        # will need later.
        #
        # ``room`` is already non-dimensional: clearance beyond what the
        # boat needs, in units of what it needs.  ``comfort`` is therefore
        # read as "keep this many boat-clearances in reserve".
        deficit = 0
        for entry in room:
            slack = comfort - entry
            # smooth max(0, slack); the corner at slack = 0 is exactly
            # where the term switches on, so it must not be a corner
            positive = 0.5 * (slack + ca.sqrt(slack * slack + 1e-6))
            deficit = deficit + positive * positive
        smoothing = smoothing + comfort_weight * deficit / (n + 1)

        # Roll is bounded by the hull surrogate's sampled range, and the
        # optimiser will sit on that bound if nothing else stops it -- 7.6
        # deg peak to peak, against the 1-2 deg sections 15-16 show a crew
        # actually holds.  A bound that comes from how the surrogate was
        # tabulated is not a physical statement, so roll is penalised
        # rather than left to rest on it.
        smoothing = smoothing + roll_weight * ca.sumsqr(state[3, :])

        # Every bound is stated in physical units above; divide through so
        # the solver sees O(1) boxes to match its O(1) variables.
        state_lo = state_lo / scaling.state[:, None]
        state_hi = state_hi / scaling.state[:, None]
        control_lo = control_lo / scaling.control[:, None]
        control_hi = control_hi / scaling.control[:, None]
        mid_lo = mid_lo / scaling.control[:, None]
        mid_hi = mid_hi / scaling.control[:, None]

        variables = ca.vertcat(ca.vec(state), ca.vec(control),
                               ca.vec(control_mid))
        problem = {"x": variables,
                   "f": -made + smoothing,
                   "g": ca.vertcat(*constraints)}
        options = {
            "ipopt.max_iter": max_iter,
            "ipopt.print_level": print_level,
            "ipopt.sb": "yes",
            "print_time": False,
            "ipopt.tol": 1e-5,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.mu_strategy": "adaptive",
        }
        if not exact_hessian:
            # A quasi-Newton Hessian.  This is a *solver* choice, not a
            # change to the model: the objective, the constraints and the
            # dynamics are identical either way, and IPOPT still converges
            # to a point satisfying the same KKT conditions.  The exact
            # Hessian of this dynamics -- through the hull surrogate, the
            # Fourier fits and the blade model -- is enormous to form, and
            # for a collocation problem of this size L-BFGS is the standard
            # trade.  Pass ``exact_hessian=True`` to compare.
            options["ipopt.hessian_approximation"] = "limited-memory"
        solver = ca.nlpsol("sixdof", "ipopt", problem, options)

        x0 = self._initial_guess(start_state, times, n, n_states, n_controls,
                                 guess, scaling)
        solution = solver(
            x0=x0,
            lbx=ca.vertcat(ca.vec(ca.DM(state_lo)), ca.vec(ca.DM(control_lo)),
                           ca.vec(ca.DM(mid_lo))),
            ubx=ca.vertcat(ca.vec(ca.DM(state_hi)), ca.vec(ca.DM(control_hi)),
                           ca.vec(ca.DM(mid_hi))),
            lbg=ca.vertcat(*[ca.DM(np.asarray(b, dtype=float).reshape(-1, 1))
                             for b in lower]),
            ubg=ca.vertcat(*[ca.DM(np.asarray(b, dtype=float).reshape(-1, 1))
                             for b in upper]),
        )
        stats = solver.stats()
        values = np.array(solution["x"]).ravel()
        cut = n_states * (n + 1)
        states = values[:cut].reshape(n_states, n + 1, order="F")
        controls = values[cut:cut + n_controls * (n + 1)].reshape(
            n_controls, n + 1, order="F")
        states = states * scaling.state[:, None]
        controls = controls * scaling.control[:, None]

        start_progress = float(ca.DM(
            self.progress(ca.DM([states[0, 0], states[1, 0]])))[0])
        end_progress = float(ca.DM(
            self.progress(ca.DM([states[0, -1], states[1, -1]])))[0])
        return SixDofPlan(time=times, states=states, controls=controls,
                          mesh=mesh, progress=end_progress - start_progress,
                          success=bool(stats.get("success", False)),
                          stats=stats, solution=values)

    def _initial_guess(self, start_state, times, n, n_states, n_controls,
                       guess, scaling=None):
        ca = self._ca
        if guess is not None:
            return guess
        # Roll the real dynamics forward instead of guessing a straight
        # coast.  A straight-line guess leaves every defect large, and the
        # solver spends its whole budget making the trajectory dynamically
        # feasible before it can start improving it.  Integrating first
        # makes the defects small at iteration zero, so IPOPT starts from
        # a trajectory the boat could actually fly.
        dynamics = self.dynamics_function()
        states = np.zeros((n_states, n + 1))
        states[:, 0] = np.asarray(start_state, dtype=float)
        nominal = np.zeros(n_controls)
        nominal[2] = 1.0
        # Substep.  One RK4 step per mesh interval is not enough: the
        # crew's balance loop is stiff -- a 6000 N m/rad spring against the
        # hull's roll inertia -- and at the ~0.27 s of a mesh interval an
        # explicit step of it diverges, oscillating to +/-70 degrees of roll
        # within four nodes and overflowing shortly after.  The collocation
        # itself is implicit and handles this; only the explicit guess
        # needs the smaller step.
        substeps = max(1, int(np.ceil(float(np.max(np.diff(times)))
                                      / self.guess_step)))
        for k in range(n):
            x = states[:, k]
            step = float(times[k + 1] - times[k]) / substeps
            t = float(times[k])
            for _ in range(substeps):
                k1 = np.array(dynamics(x, nominal, t)).ravel()
                k2 = np.array(dynamics(x + 0.5 * step * k1, nominal,
                                       t + 0.5 * step)).ravel()
                k3 = np.array(dynamics(x + 0.5 * step * k2, nominal,
                                       t + 0.5 * step)).ravel()
                k4 = np.array(dynamics(x + step * k3, nominal,
                                       t + step)).ravel()
                x = x + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += step
            states[:, k + 1] = x

        controls = np.tile(nominal[:, None], (1, n + 1))
        mid = np.tile(nominal[:, None], (1, n))
        if scaling is not None:
            # The rollout runs in physical units -- it has to, it calls the
            # real dynamics -- so it is scaled on the way out, not on the
            # way in.
            states = states / scaling.state[:, None]
            controls = controls / scaling.control[:, None]
            mid = mid / scaling.control[:, None]
        return ca.vertcat(ca.DM(states.reshape(-1, 1, order="F")),
                          ca.DM(controls.reshape(-1, 1, order="F")),
                          ca.DM(mid.reshape(-1, 1, order="F")))


def build_progress_field(raster, centreline=None):
    """Arc length along the channel, as a raster.

    Rewarding straight-line displacement would pay the boat to cut across
    land on a bend.  This tabulates, for every cell, the arc length of the
    nearest point on the channel centreline, so the objective measures
    progress along the river.
    """
    from scipy.spatial import cKDTree

    if centreline is None:
        centreline = raster.centreline()
    line = np.asarray(centreline, dtype=float)
    step = np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1]))
    arc = np.concatenate([[0.0], np.cumsum(step)])

    east, north = np.meshgrid(raster.east, raster.north)
    query = np.column_stack([east.ravel(), north.ravel()])
    _, index = cKDTree(line).query(query)
    return arc[index].reshape(east.shape)
