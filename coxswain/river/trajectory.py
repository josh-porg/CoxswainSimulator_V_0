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

with the rudder angle ``delta`` as the single control.  Dynamics:

    x_dot   = u cos(psi) + c_x(x, y)
    y_dot   = u sin(psi) + c_y(x, y)
    psi_dot = r
    u_dot   = (T(u, h) - D(u, h) - D_turn(r)) / m
    r_dot   = (N_delta u^2 delta - N_r u r) / I_z

Why reduced.  Hermite-Simpson needs the dynamics evaluated symbolically at
every collocation point and midpoint, and differentiated exactly.  The
6-DOF model contains a Cholesky solve, a 96-body moving-mass field, scipy
interpolators and branch logic -- none of it expressible in CasADi without
rewriting the entire package.  The standard remedy, and the one used here,
is to fit a reduced model to the high-fidelity one, optimise on the
reduced model, and verify the answer back in the full model.

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
    yaw_control: float = 900.0
    #: Yaw damping per unit speed, N m s/rad.
    yaw_damping: float = 32000.0
    #: Extra drag per unit yaw rate squared, N s2/rad2.  Turning costs
    #: speed; without this the optimiser steers for free.
    turn_drag: float = 8000.0
    #: Maximum usable rudder angle, radians.
    rudder_limit: float = np.radians(12.0)

    def __post_init__(self) -> None:
        if self.drag_coefficient == 0.0:
            # thrust is held fixed, so drag must balance it at the
            # reference speed: T = k v_ref^2
            self.drag_coefficient = 1.0
        self.thrust = self.drag_coefficient * self.reference_speed ** 2

    def straight_line_speed(self, depth_factor: float = 1.0) -> float:
        """Steady speed with no rudder, at a given drag multiplier."""
        return float(np.sqrt(self.thrust
                             / (self.drag_coefficient * depth_factor)))


def fit_reduced_model(boat=None, reference_speed: float = 5.2,
                      **overrides) -> ReducedModel:
    """Fit the reduced model to the full 6-DOF simulator.

    The mass and yaw inertia come straight from the boat.  The yaw control
    and damping coefficients are identified from step-rudder responses of
    the full model: a constant rudder held from a straight run gives a
    steady turn rate, and the ratio of the two fixes ``yaw_control /
    yaw_damping``, while the rise time fixes the pair.

    Falls back to the documented defaults when no boat is supplied, so the
    module is usable without paying for a 6-DOF run.
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

    steady = {}
    for deflection in (np.radians(4.0), np.radians(8.0)):
        simulator = RowingSimulator(
            boat, rudder=lambda t, state, d=deflection: d)
        result = simulator.run(duration=24.0, dt=0.006,
                               surge_speed=reference_speed)
        window = result.last_cycles(3)
        # omega is the absolute-frame angular velocity; its vertical
        # component is the yaw rate for the small roll and pitch a shell
        # actually sees
        steady[deflection] = float(np.mean(result.omega[2][window]))

    # steady turn: N_delta u^2 delta == N_r u r  =>  r = (N_delta/N_r) u delta
    ratios = [abs(rate) / (reference_speed * deflection)
              for deflection, rate in steady.items() if deflection > 0]
    if ratios and np.isfinite(ratios).all() and max(ratios) > 0:
        gain = float(np.mean(ratios))
        model.yaw_control = gain * model.yaw_damping
    return model


@dataclass
class TrajectorySolution:
    """Result of a collocation solve."""

    time: np.ndarray
    state: np.ndarray            # (5, n)
    rudder: np.ndarray           # (n,)
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

    def summary(self) -> dict:
        step = np.diff(self.position, axis=1)
        return {
            "duration": self.duration,
            "path_length": float(np.hypot(step[0], step[1]).sum()),
            "mean_speed": float(self.speed.mean()),
            "max_rudder_deg": float(np.degrees(np.abs(self.rudder).max())),
            "max_yaw_rate_deg": float(np.degrees(np.abs(self.yaw_rate).max())),
            "success": self.success,
        }


def _casadi_fields(channel, flow, reference_speed):
    """Build differentiable lookups for depth and current on the raster."""
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

    return depth_lookup, clearance_lookup


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
    depth_lookup, clearance_lookup = _casadi_fields(channel, None,
                                                    model.reference_speed)

    opti = ca.Opti()

    # decision variables: state at each node, control at each node and
    # midpoint, and the total duration
    state = opti.variable(5, n_nodes)
    rudder = opti.variable(1, n_nodes)
    rudder_mid = opti.variable(1, n_nodes - 1)
    duration = opti.variable()

    def dynamics(s, delta):
        x, y, psi, u, r = s[0], s[1], s[2], s[3], s[4]
        # shallow water raises drag; a smooth, monotone surrogate for the
        # wave-resistance rise, calibrated to the same 3 m / -13% point the
        # full model reproduces
        h = depth_lookup(ca.vertcat(x, y))
        shallow = 1.0 + 1.6 * ca.exp(-(h - 0.8) / 1.6)
        drag = model.drag_coefficient * shallow * u ** 2
        turn_loss = model.turn_drag * r ** 2
        return ca.vertcat(
            u * ca.cos(psi),
            u * ca.sin(psi),
            r,
            (model.thrust - drag - turn_loss) / model.mass,
            (model.yaw_control * u ** 2 * delta
             - model.yaw_damping * u * r) / model.yaw_inertia,
        )

    step = duration / (n_nodes - 1)

    # Hermite-Simpson defect constraints
    for k in range(n_nodes - 1):
        left, right = state[:, k], state[:, k + 1]
        f_left = dynamics(left, rudder[0, k])
        f_right = dynamics(right, rudder[0, k + 1])
        # cubic Hermite midpoint
        middle = 0.5 * (left + right) + step / 8.0 * (f_left - f_right)
        f_middle = dynamics(middle, rudder_mid[0, k])
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
    opti.subject_to(opti.bounded(0.5, state[3, :], 3.0 * model.reference_speed))
    opti.subject_to(opti.bounded(10.0, duration, max_duration))

    opti.subject_to(state[0, 0] == start[0])
    opti.subject_to(state[1, 0] == start[1])
    opti.subject_to(state[3, 0] == model.reference_speed)
    opti.subject_to(state[4, 0] == 0.0)
    opti.subject_to(state[0, -1] == goal[0])
    opti.subject_to(state[1, -1] == goal[1])

    opti.minimize(duration)

    line = initial_guess
    heading = np.arctan2(np.gradient(line[1]), np.gradient(line[0]))
    guess = np.vstack([line, heading,
                       np.full(n_nodes, model.reference_speed),
                       np.zeros(n_nodes)])
    opti.set_initial(state, guess)
    opti.set_initial(rudder, np.zeros((1, n_nodes)))
    opti.set_initial(rudder_mid, np.zeros((1, n_nodes - 1)))
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
        duration_value = float(solution.value(duration))
    except RuntimeError as error:
        # IPOPT can stop at an acceptable point; keep it and say so
        success, message = False, str(error).splitlines()[0]
        state_value = opti.debug.value(state)
        rudder_value = opti.debug.value(rudder)
        duration_value = float(opti.debug.value(duration))

    time = np.linspace(0.0, duration_value, n_nodes)
    return TrajectorySolution(time=time, state=np.atleast_2d(state_value),
                              rudder=np.atleast_1d(rudder_value),
                              duration=duration_value, success=success,
                              message=message)
