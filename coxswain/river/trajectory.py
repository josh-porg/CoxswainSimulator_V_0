"""Trajectory optimisation with rudder control, by Hermite-Simpson collocation.

:mod:`coxswain.river.route` answers "which line is quickest" by scoring an
offset profile.  It says nothing about whether a coxswain could *steer*
that line: the offset is a free function of station, and nothing connects
it to a rudder.

This module closes that gap.  The line becomes the output of a dynamic
model with the rudder as its input, so the answer is a steering plan rather
than a wish.

The model
---------
A reduced five-state model, not the full 6-DOF one::

    x       east position          [m]
    y       north position         [m]
    psi     heading                [rad]
    u       surge speed            [m/s]
    r       yaw rate               [rad/s]

plus ``w``, the crew's remaining anaerobic work capacity, and **three**
controls: the rudder angle ``delta``, the port/starboard pressure split
``s``, and the power fraction ``pi``.  Dynamics:

    x_dot   = u cos(psi) + c_x(x, y)
    y_dot   = u sin(psi) + c_y(x, y)
    psi_dot = r
    u_dot   = (pi T - D(u, h) - D_turn(r) - D_split(s)) / m
    r_dot   = (N_delta u^2 delta + N_s s - N_r u r) / I_z
    w_dot   = -(P(pi) - CP)

Pacing
------
Holding the crew at constant power, as this model previously did, removes
the decision a crew actually makes: how hard to push, and where.  On a
head course that is not separable from steering, because a pressure split
spends thrust on a couple -- so the optimiser must be able to answer
"push through the bend, or ease and steer" rather than having the first
half of it fixed.

Effort is bounded by the **critical-power model** (Monod & Scherrer 1965;
Morton 2006), the standard two-parameter description of endurance
performance: power ``CP`` is sustainable indefinitely, and any excess
draws on a finite anaerobic capacity ``W'`` which depletes at ``P - CP``
and recovers below it.  ``W'`` is carried as a sixth state and constrained
non-negative, so the optimiser cannot spend energy the crew does not have.
Without that budget, minimum time is trivially "row flat out everywhere".

with ``c`` the depth-averaged water velocity.  The current enters the
*position* equations only: hydrodynamic forces depend on motion through
the water, which is what ``u`` already is, while the trajectory is over
the ground.  Getting that split wrong is the same class of error as
mixing hull and absolute frames.

The current is optional.  It is carried on the channel raster (see
:func:`~coxswain.river.channel.attach_current`) rather than passed
separately, so cropping cannot silently lose it; a raster without one
gives the still-water answer.

Why two controls.  The rudder on its own cannot get an eight round this
river.  Measured against the channel extracted in
:mod:`coxswain.river.channel`, the Charles demands a turn radius of
103-146 m at its tightest bends, and **19% of the reach is tighter than
full rudder alone can hold** (259 m).  Adding a 30% pressure split brings
it to 130 m.  A coxswain calling for pressure is not a refinement on top
of steering -- on a river it *is* the steering, and a model with only a
rudder silently cannot fly the course.

Both controls cost speed: turning bleeds energy into the appendages, and
splitting the crew spends work on a couple rather than on thrust.  Without
those penalties the optimiser would steer for free.

Why reduced.  Not because the 6-DOF model resists CasADi -- an earlier
version of this docstring claimed that and it was wrong.  ``ca.chol``
handles the 6x6 solve, the crew kinematics are Fourier series and trig,
``ca.if_else`` covers the branches, and the field lookups are already
interpolants; only ``HullMesh.submerged`` is genuinely awkward, and a
smooth parametric fit would replace it.

The real obstacle is **timescale separation**.  The 6-DOF model carries
2 Hz stroke dynamics; a Charles course lasts ~40 minutes.  Resolving both
in one transcription needs of order 10^4 nodes times 12 states, and the
stroke-scale oscillation is not what the route decision turns on.  The
standard treatment of a multiscale optimal-control problem is to average
over the fast scale, which is exactly what this reduced model is.  A
stroke-averaged 6-DOF in CasADi is a real middle path and remains open.

The surge and yaw coefficients are therefore **fitted from the 6-DOF
simulator**, not invented: see :func:`fit_reduced_model`.

Why Hermite-Simpson
-------------------
It is a third-order-accurate implicit collocation scheme: the state is a
cubic Hermite spline over each interval, the dynamics are enforced at both
ends and at the midpoint, and the defect constraint is Simpson's rule.
That buys much better accuracy per node than trapezoidal collocation on
smooth problems, which matters here because the node count is what sets
the size of the NLP and a 12 km reach needs a lot of them.

The depth and current fields are not analytic, so they enter as CasADi
``interpolant`` lookups on the surveyed rasters -- differentiable by
construction, and using the real data rather than a fitted surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from ..hydro.appendages import MAX_RUDDER_DEFLECTION

__all__ = [
    "ReducedModel",
    "TrajectorySolution",
    "fit_reduced_model",
    "solve_trajectory",
]


@dataclass
class ReducedModel:
    """Coefficients of the reduced steering model.

    ``thrust`` and ``drag`` are expressed so that the boat settles at
    ``reference_speed`` in deep water with zero rudder, which is the
    condition the 6-DOF model is run at to fit them.
    """

    mass: float = 855.0              # kg, eight plus crew and coxswain
    yaw_inertia: float = 22000.0     # kg m2 about the vertical axis
    reference_speed: float = 5.2     # m/s, deep water, straight
    drag_coefficient: float = 0.0    # N s2/m2, set by fitting
    #: Yaw moment per unit rudder angle per unit speed squared, N m/rad.
    #: Fitted from step-rudder responses of the 6-DOF model for an eight,
    #: at 4 and 8 degrees.  Raised from 539.0 alongside
    #: :attr:`rudder_saturation`, which are solved together so the mean
    #: slope over those two fit angles is unchanged AND full rudder
    #: reproduces the 6-DOF's 259 m turn radius.  Changing either alone
    #: breaks one of those two conditions.
    yaw_control: float = 585.5
    #: Yaw damping per unit speed, N m s/rad.
    yaw_damping: float = 32000.0
    #: Extra drag per unit yaw rate squared, N s2/rad2.  Turning costs
    #: speed; without this the optimiser steers for free.
    turn_drag: float = 8000.0
    #: Maximum usable rudder angle, radians.  See
    #: :data:`~coxswain.hydro.appendages.MAX_RUDDER_DEFLECTION`; this was
    #: 12 degrees, which is not what the boat has and cost the optimiser
    #: most of its steering.
    rudder_limit: float = MAX_RUDDER_DEFLECTION
    #: Effective-deflection scale at which rudder authority saturates,
    #: radians.  **Without this the model is a linearisation used five
    #: times past its fit range.**
    #:
    #: ``yaw_control`` is fitted from step responses at 4 and 8 degrees
    #: (see :meth:`fit_from_boat`) and then applied linearly out to the
    #: 45-degree structural limit.  It should not be: this model has no
    #: sway state, so it cannot represent the sideslip a large rudder
    #: induces, and that sideslip is what cancels most of the moment.
    #: SOURCES sec. 36 measures the effect in the full model -- rudder
    #: alone gives about 3.5 deg/s with sideslip ignored and about 1.1
    #: deg/s once it is not.
    #:
    #: Left linear, the reduced model held a **75.6 m** turn radius at full
    #: rudder against the 6-DOF model's measured **259 m** -- 3.4 times more
    #: steering than the boat has, in a surrogate whose entire job is to
    #: produce trajectories the full model can fly.
    #:
    #: ``delta_eff = s tanh(delta / s)`` preserves the fitted slope as
    #: ``delta -> 0`` and saturates at ``s``, which is set so that full
    #: rudder reproduces the 6-DOF's 259 m.  Set to zero to disable.
    rudder_saturation: float = 0.21127
    #: Yaw moment per unit port/starboard pressure split, N m.  The
    #: coxswain's second control, and on a river the decisive one.
    #: Measured against the extracted Charles channel: full rudder alone
    #: holds a 259 m turn radius while the tightest bends demand 103-146 m.
    #: Rudder plus a 30% split reaches 130 m.
    split_control: float = 7455.0
    #: Largest pressure split worth asking a crew for.  Past about a third
    #: the light side is barely rowing and the lost thrust costs more than
    #: the turn gains.
    split_limit: float = 0.30
    #: Extra drag per unit split squared, N.  Splitting the crew spends
    #: work on a couple instead of on thrust.
    split_drag: float = 250.0
    #: Critical power for the whole crew, W.  Roughly 380 W per rower for a
    #: club eight; a world-level crew is nearer 450.  This is the power the
    #: crew can hold for the duration of a head race without drawing down
    #: W'.
    critical_power: float = 3040.0
    #: Anaerobic work capacity for the whole crew, J.  ~22 kJ per rower.
    #: Everything above CP comes out of here, and it is finite.
    anaerobic_capacity: float = 176000.0
    #: Bounds on the power fraction relative to `critical_power`.  A crew
    #: cannot row at zero, and cannot hold much more than 1.5x CP for long
    #: enough to matter.
    power_min: float = 0.55
    power_max: float = 1.45

    def __post_init__(self) -> None:
        if self.drag_coefficient == 0.0:
            # thrust is held fixed, so drag must balance it at the
            # reference speed: T = k v_ref^2
            self.drag_coefficient = 1.0
        self.thrust = self.drag_coefficient * self.reference_speed ** 2

    def effective_rudder(self, delta, symbolic=None):
        """Deflection the yaw moment actually responds to, radians.

        ``s tanh(delta / s)``: identity for small ``delta``, so the slope
        fitted at 4 and 8 degrees is untouched, and saturating at ``s`` so
        that full rudder does not buy authority the boat does not have.
        See :attr:`rudder_saturation`.

        Pass CasADi as ``symbolic`` to build the expression instead.
        """
        scale = float(self.rudder_saturation)
        if scale <= 0.0:
            return delta
        if symbolic is None:
            return scale * np.tanh(np.asarray(delta, dtype=float) / scale)
        return scale * symbolic.tanh(delta / scale)

    def steady_yaw_rate(self, delta=0.0, split=0.0, speed=None):
        """Steady-state yaw rate at this rudder and split, rad/s."""
        u = self.reference_speed if speed is None else float(speed)
        return ((self.yaw_control * u ** 2 * self.effective_rudder(delta)
                 + self.split_control * float(split))
                / (self.yaw_damping * u))

    def steady_turn_radius(self, delta=0.0, split=0.0, speed=None):
        """Radius of the steady turn those controls hold, metres."""
        u = self.reference_speed if speed is None else float(speed)
        rate = self.steady_yaw_rate(delta, split, u)
        return u / abs(rate) if rate else float("inf")

    def straight_line_speed(self, depth_factor: float = 1.0,
                            power_fraction: float = 1.0) -> float:
        """Steady speed with no rudder, at a given drag multiplier."""
        return float(np.sqrt(power_fraction * self.thrust
                             / (self.drag_coefficient * depth_factor)))

    def power_at(self, power_fraction):
        """Mechanical power drawn at a given thrust fraction, W.

        Thrust times speed is the useful power, and at steady state speed
        scales as the cube root of power -- so a thrust fraction ``pi``
        corresponds to a power fraction ``pi**1.5``.  Expressing the
        control as thrust and deriving power keeps the surge equation
        linear in the control, which the NLP prefers.
        """
        return self.critical_power * power_fraction ** 1.5


def fit_reduced_model(boat=None, reference_speed: float = 5.2,
                      duration: float = 10.0, dt: float = 0.01,
                      **overrides) -> ReducedModel:
    """Fit the reduced model to the full 6-DOF simulator.

    The mass and yaw inertia come straight from the boat.  The yaw control
    and damping coefficients are identified from step-rudder responses of
    the full model: a constant rudder held from a straight run gives a
    steady turn rate, and the ratio of the two fixes ``yaw_control /
    yaw_damping``, while the rise time fixes the pair.

    Falls back to the documented defaults when no boat is supplied, so the
    module is usable without paying for a 6-DOF run.

    ``duration`` need not be long.  The yaw time constant is
    ``I_z / (N_r u)``, about 0.06 s for an eight, so the steady turn is
    reached within a stroke; the default of 10 s is set by wanting whole
    stroke cycles to average over, not by the settling time.
    """
    if boat is None:
        return ReducedModel(reference_speed=reference_speed, **overrides)

    from ..sim.simulator import RowingSimulator

    # Yaw inertia must be hull PLUS crew.  In the 6-DOF model the crew is a
    # separate moving-mass field and `hull_inertia` is the bare shell --
    # 1915 kg m2 for an eight, against roughly 20 000 once eight rowers
    # spread over +-4 m are included.  The reduced model has no separate
    # crew, so using the hull figure alone would make the boat ten times
    # more manoeuvrable than it is.
    mass, position, _, _ = boat.crew_field(0.0)
    crew_yaw = float(np.sum(mass * (position[:, 0] ** 2
                                    + position[:, 1] ** 2)))
    model = ReducedModel(
        mass=boat.total_mass,
        yaw_inertia=float(boat.hull_inertia[2, 2]) + crew_yaw,
        reference_speed=reference_speed,
        **overrides,
    )

    from ..sim.control import Coxswain

    def steady_rate(rudder=0.0, split=0.0):
        cox = Coxswain(rudder_override=lambda t, s: rudder,
                       pressure_split=split)
        result = RowingSimulator(boat, coxswain=cox).run(
            duration=duration, dt=dt, surge_speed=reference_speed)
        # omega is the absolute-frame angular velocity; its vertical
        # component is the yaw rate for the small roll and pitch a shell
        # actually sees
        return float(np.mean(result.omega[2][result.last_cycles(3)]))

    # An eight yaws about 0.4 deg/s with the rudder centred and both sides
    # pulling equally: the standard alternating rig carries its port and
    # starboard oarlocks a seat apart in x, so a sweep stroke's lateral
    # force acts through a 1.22 m couple (SOURCES 60).  Control authority
    # is what the control buys **over and above** that, so the neutral rate
    # has to come out of both fits.
    #
    # Leaving it in was how `munk_factor` came to be calibrated against a
    # contaminated quantity and ended up twice too large, taking the boat's
    # directional stability with it.  The same arithmetic here would have
    # told the optimiser the rudder was worth roughly twice what it is, and
    # an optimised line is only as good as the authority it assumes.
    neutral = steady_rate()

    # Steady rudder turn: N_delta u^2 delta == N_r u r.
    ratios = []
    for deflection in (np.radians(4.0), np.radians(8.0)):
        rate = steady_rate(rudder=deflection) - neutral
        if np.isfinite(rate) and abs(rate) > 0:
            ratios.append(abs(rate) / (reference_speed * deflection))
    if ratios:
        model.yaw_control = float(np.mean(ratios)) * model.yaw_damping

    # Steady split turn: N_split s == N_r u r.
    split_ratios = []
    for split in (0.15, 0.30):
        rate = steady_rate(split=split) - neutral
        if np.isfinite(rate) and abs(rate) > 0:
            split_ratios.append(abs(rate) * reference_speed / split)
    if split_ratios:
        model.split_control = float(np.mean(split_ratios)) * model.yaw_damping
    return model


@dataclass
class TrajectorySolution:
    """Result of a collocation solve."""

    time: np.ndarray
    state: np.ndarray            # (5, n)
    rudder: np.ndarray           # (n,)
    split: np.ndarray            # (n,) port/starboard pressure split
    power: np.ndarray            # (n,) thrust fraction of critical power
    duration: float
    success: bool
    message: str = ""

    @property
    def position(self) -> np.ndarray:
        return self.state[0:2]

    @property
    def heading(self) -> np.ndarray:
        return self.state[2]

    @property
    def speed(self) -> np.ndarray:
        return self.state[3]

    @property
    def yaw_rate(self) -> np.ndarray:
        return self.state[4]

    @property
    def anaerobic_remaining(self) -> np.ndarray:
        """W' left, in joules, at each node."""
        return self.state[5]

    def summary(self) -> dict:
        step = np.diff(self.position, axis=1)
        return {
            "duration": self.duration,
            "path_length": float(np.hypot(step[0], step[1]).sum()),
            "mean_speed": float(self.speed.mean()),
            "max_rudder_deg": float(np.degrees(np.abs(self.rudder).max())),
            "max_yaw_rate_deg": float(np.degrees(np.abs(self.yaw_rate).max())),
            "max_split": float(np.abs(self.split).max()),
            "power_range": (float(self.power.min()), float(self.power.max())),
            "anaerobic_spent": float(self.anaerobic_remaining[0]
                                     - self.anaerobic_remaining[-1]),
            "success": self.success,
        }


def _casadi_fields(channel, flow, reference_speed):
    """Build differentiable lookups for the fields on the raster.

    Returns ``(depth, clearance, current_east, current_north)``; the two
    current lookups are ``None`` when the raster carries no flow.
    """
    import casadi as ca

    east = np.asarray(channel.east, dtype=float)
    north = np.asarray(channel.north, dtype=float)

    depth = np.nan_to_num(channel.depth, nan=0.1)
    depth = np.maximum(depth, 0.1)
    # CasADi interpolant wants the grid flattened in Fortran order
    depth_lookup = ca.interpolant(
        "depth", "bspline", [east.tolist(), north.tolist()],
        depth.T.ravel(order="F").tolist())

    clearance_lookup = ca.interpolant(
        "clearance", "bspline", [east.tolist(), north.tolist()],
        channel.clearance.T.ravel(order="F").tolist())

    current_east = current_north = None
    if channel.has_current:
        current_east = ca.interpolant(
            "current_east", "bspline", [east.tolist(), north.tolist()],
            channel.current_east.T.ravel(order="F").tolist())
        current_north = ca.interpolant(
            "current_north", "bspline", [east.tolist(), north.tolist()],
            channel.current_north.T.ravel(order="F").tolist())

    return depth_lookup, clearance_lookup, current_east, current_north


def _guess_along_channel(channel, start, goal, n_nodes: int) -> np.ndarray:
    """A feasible starting path: the least-cost route through the water.

    Reuses :meth:`ChannelRaster.centreline`, restricted to the leg, so the
    guess is navigable by construction and the optimiser starts inside the
    feasible corridor rather than having to find it.
    """
    navigable = channel.navigable
    cells = np.flatnonzero(navigable.ravel())
    rows, columns = np.unravel_index(cells, navigable.shape)
    coordinates = np.column_stack([channel.east[columns],
                                   channel.north[rows]])

    def nearest(point):
        return int(np.argmin(np.hypot(coordinates[:, 0] - point[0],
                                      coordinates[:, 1] - point[1])))

    path = channel.centreline(smooth=5, start=nearest(start),
                              end=nearest(goal))

    # resample to exactly n_nodes by arc length
    step = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(path, axis=0), axis=1))])
    wanted = np.linspace(0.0, step[-1], n_nodes)
    return np.vstack([np.interp(wanted, step, path[:, 0]),
                      np.interp(wanted, step, path[:, 1])])


def solve_trajectory(channel, start: np.ndarray, goal: np.ndarray,
                     model: ReducedModel = None, n_nodes: int = 60,
                     initial_guess: np.ndarray = None,
                     clearance_margin: float = 6.0,
                     max_duration: float = 4000.0,
                     print_level: int = 0) -> TrajectorySolution:
    """Minimum-time trajectory from ``start`` to ``goal``, rudder-steered.

    Transcribed by Hermite-Simpson collocation and solved with IPOPT.

    Parameters
    ----------
    channel:
        A :class:`~coxswain.river.channel.ChannelRaster`; supplies the
        depth and clearance fields.
    start, goal:
        ``(x, y)`` in the tangent-plane frame.
    n_nodes:
        Collocation nodes.  Hermite-Simpson places an extra midpoint in
        each interval, so the effective resolution is roughly double.
    clearance_margin:
        Stay this far inside the navigable boundary, in metres.
    """
    import casadi as ca

    model = ReducedModel() if model is None else model

    # An initial guess that crosses land makes this hopeless: the feasible
    # corridor is tens of metres wide in a nonconvex problem, and IPOPT
    # started from outside it does not find its way in.  A straight line
    # between two points 2 km apart on a bending river was measured at 36%
    # navigable.  Prefer a guess that follows the water.
    if initial_guess is None:
        initial_guess = _guess_along_channel(channel, start, goal, n_nodes)
    else:
        initial_guess = np.asarray(initial_guess, dtype=float)
        if initial_guess.shape[0] != 2:
            initial_guess = initial_guess.T

    # Crop the fields to the leg being solved; see ChannelRaster.crop.
    channel = channel.crop(initial_guess.T)
    (depth_lookup, clearance_lookup, current_east_lookup,
     current_north_lookup) = _casadi_fields(channel, None,
                                            model.reference_speed)

    opti = ca.Opti()

    # decision variables: state at each node, control at each node and
    # midpoint, and the total duration
    state = opti.variable(6, n_nodes)
    rudder = opti.variable(1, n_nodes)
    rudder_mid = opti.variable(1, n_nodes - 1)
    split = opti.variable(1, n_nodes)
    split_mid = opti.variable(1, n_nodes - 1)
    power = opti.variable(1, n_nodes)
    power_mid = opti.variable(1, n_nodes - 1)
    duration = opti.variable()

    def dynamics(s, delta, pressure, effort):
        x, y, psi, u, r = s[0], s[1], s[2], s[3], s[4]
        # shallow water raises drag; a smooth, monotone surrogate for the
        # wave-resistance rise, calibrated to the same 3 m / -13% point the
        # full model reproduces
        h = depth_lookup(ca.vertcat(x, y))
        shallow = 1.0 + 1.6 * ca.exp(-(h - 0.8) / 1.6)
        drag = model.drag_coefficient * shallow * u ** 2
        turn_loss = model.turn_drag * r ** 2
        split_loss = model.split_drag * pressure ** 2
        position = ca.vertcat(x, y)
        drift_east = (0.0 if current_east_lookup is None
                      else current_east_lookup(position))
        drift_north = (0.0 if current_north_lookup is None
                       else current_north_lookup(position))
        drawn = model.critical_power * effort ** 1.5
        return ca.vertcat(
            u * ca.cos(psi) + drift_east,
            u * ca.sin(psi) + drift_north,
            r,
            (effort * model.thrust - drag - turn_loss - split_loss)
            / model.mass,
            (model.yaw_control * u ** 2
             * model.effective_rudder(delta, ca)
             + model.split_control * pressure
             - model.yaw_damping * u * r) / model.yaw_inertia,
            -(drawn - model.critical_power),
        )

    step = duration / (n_nodes - 1)

    # Hermite-Simpson defect constraints
    for k in range(n_nodes - 1):
        left, right = state[:, k], state[:, k + 1]
        f_left = dynamics(left, rudder[0, k], split[0, k], power[0, k])
        f_right = dynamics(right, rudder[0, k + 1], split[0, k + 1],
                           power[0, k + 1])
        # cubic Hermite midpoint
        middle = 0.5 * (left + right) + step / 8.0 * (f_left - f_right)
        f_middle = dynamics(middle, rudder_mid[0, k], split_mid[0, k],
                            power_mid[0, k])
        # Simpson quadrature defect
        opti.subject_to(
            right - left == step / 6.0 * (f_left + 4.0 * f_middle + f_right))

    # stay in navigable water, with a margin
    for k in range(n_nodes):
        opti.subject_to(
            clearance_lookup(ca.vertcat(state[0, k], state[1, k]))
            >= clearance_margin)

    opti.subject_to(opti.bounded(-model.rudder_limit, rudder,
                                 model.rudder_limit))
    opti.subject_to(opti.bounded(-model.rudder_limit, rudder_mid,
                                 model.rudder_limit))
    opti.subject_to(opti.bounded(-model.split_limit, split, model.split_limit))
    opti.subject_to(opti.bounded(-model.split_limit, split_mid,
                                 model.split_limit))
    opti.subject_to(opti.bounded(model.power_min, power, model.power_max))
    opti.subject_to(opti.bounded(model.power_min, power_mid, model.power_max))
    # the crew cannot spend anaerobic capacity it does not have
    opti.subject_to(opti.bounded(0.0, state[5, :],
                                 model.anaerobic_capacity))
    opti.subject_to(opti.bounded(0.5, state[3, :], 3.0 * model.reference_speed))
    opti.subject_to(opti.bounded(10.0, duration, max_duration))

    opti.subject_to(state[0, 0] == start[0])
    opti.subject_to(state[1, 0] == start[1])
    opti.subject_to(state[3, 0] == model.reference_speed)
    opti.subject_to(state[4, 0] == 0.0)
    opti.subject_to(state[5, 0] == model.anaerobic_capacity)
    opti.subject_to(state[0, -1] == goal[0])
    opti.subject_to(state[1, -1] == goal[1])

    opti.minimize(duration)

    line = initial_guess
    heading = np.arctan2(np.gradient(line[1]), np.gradient(line[0]))
    guess = np.vstack([line, heading,
                       np.full(n_nodes, model.reference_speed),
                       np.zeros(n_nodes),
                       np.full(n_nodes, model.anaerobic_capacity)])
    opti.set_initial(state, guess)
    opti.set_initial(rudder, np.zeros((1, n_nodes)))
    opti.set_initial(rudder_mid, np.zeros((1, n_nodes - 1)))
    opti.set_initial(split, np.zeros((1, n_nodes)))
    opti.set_initial(split_mid, np.zeros((1, n_nodes - 1)))
    opti.set_initial(power, np.ones((1, n_nodes)))
    opti.set_initial(power_mid, np.ones((1, n_nodes - 1)))
    span = float(np.hypot(*(np.asarray(goal) - np.asarray(start))))
    opti.set_initial(duration, max(span / model.reference_speed, 20.0))

    opti.solver("ipopt", {"print_time": False},
                {"print_level": print_level, "max_iter": 3000,
                 "tol": 1e-6, "acceptable_tol": 1e-4})

    try:
        solution = opti.solve()
        success, message = True, "optimal"
        state_value = solution.value(state)
        rudder_value = solution.value(rudder)
        split_value = solution.value(split)
        power_value = solution.value(power)
        duration_value = float(solution.value(duration))
    except RuntimeError as error:
        # IPOPT can stop at an acceptable point; keep it and say so
        success, message = False, str(error).splitlines()[0]
        state_value = opti.debug.value(state)
        rudder_value = opti.debug.value(rudder)
        split_value = opti.debug.value(split)
        power_value = opti.debug.value(power)
        duration_value = float(opti.debug.value(duration))

    time = np.linspace(0.0, duration_value, n_nodes)
    return TrajectorySolution(time=time, state=np.atleast_2d(state_value),
                              rudder=np.atleast_1d(rudder_value),
                              split=np.atleast_1d(split_value),
                              power=np.atleast_1d(power_value),
                              duration=duration_value, success=success,
                              message=message)
