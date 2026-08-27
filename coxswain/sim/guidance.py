"""Steering a real boat down a planned line, in a moving fluid.

The route optimiser produces a *path*.  A coxswain cannot fly a path; they
can only put the rudder somewhere and see what the boat does.  Closing that
gap is what this module is for, and it is also the honest test of
everything upstream: a line that looks quick in a quasi-steady evaluator
has to be checked against a boat with mass, yaw inertia, and a rudder that
only works when water is moving past it.

The guidance law
----------------
This implements the nonlinear path-following law of Furieri, Stastny,
Marconi, Siegwart and Gilitschenski [GWW]_, which was written for small
fixed-wing aircraft in strong wind and explicitly covers the same problem
for **underwater currents**.  A boat in a river is the identical geometry:
the hull is a non-holonomic unicycle whose velocity through the *water* is
what the rudder controls, while the path is fixed to the *ground*, and the
two differ by the flow.

The law builds a look-ahead unit vector that blends heading straight at the
path with running along it, according to how far off the boat is::

    L = cos(theta) * N + sin(theta) * T
    theta(|d|) = (pi/2) * sqrt(1 - sat(|d| / delta))

with ``N`` the unit vector from the boat toward the nearest path point and
``T`` the path tangent there.  On the line, ``theta = pi/2`` and ``L = T``:
steer along the path.  Far off it, ``theta = 0`` and ``L = N``: steer
straight at the path.  Between, it turns on smoothly.

That single expression is what a fixed look-ahead distance approximates
badly.  A look-ahead point is a crude sample of this blend, and its
character changes with how far off the line you are -- which is why such
controllers weave when close and cut corners when far.  Here the blend is
explicit and the transition width ``delta`` is the only knob.

Correcting for the current
--------------------------
``L`` is the desired direction of travel **over the ground**.  What the
rudder actually sets is the boat's heading through the water, and the two
differ by the current.  Solving the flow triangle

    w + v * L_water  is parallel to  L_ground

gives the crab angle the boat must carry.  On the Charles in October the
current is a few centimetres per second against five metres per second of
boat, so the correction is a fraction of a degree -- but it is written the
general way, because it costs nothing to do properly and the same code
should hold in flood.

The paper's real contribution is the case where the flow is *faster* than
the vehicle, where no heading achieves the desired ground track and the
law degrades to a stable safety objective instead of demanding the
impossible.  That case cannot arise on this river; the feasibility test is
kept anyway so the failure is explicit rather than a silent NaN.

.. [GWW] Furieri, L., Stastny, T., Marconi, L., Siegwart, R. and
   Gilitschenski, I. (2017).  "Gone with the Wind: Nonlinear Guidance for
   Small Fixed-Wing Aircraft in Arbitrarily Strong Windfields."  American
   Control Conference.  arXiv:1609.07577.
.. [P04] Park, S., Deyst, J. and How, J. (2004).  "A New Nonlinear
   Guidance Logic for Trajectory Tracking."  AIAA GNC.  The no-wind
   ancestor this builds on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core.frames import wrap_to_pi

__all__ = ["PathFollower"]


def _rotate(vector, angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([cos * vector[0] - sin * vector[1],
                     sin * vector[0] + cos * vector[1]])


@dataclass
class PathFollower:
    """Nonlinear path following with a rudder loop underneath.

    Usable directly as a ``rudder_override`` on
    :class:`~coxswain.sim.control.Coxswain`.
    """

    #: The line to follow, ``(n, 2)`` in the tangent-plane frame.
    path: np.ndarray
    #: Transition width, metres.  Inside this the law swings from steering
    #: at the path to steering along it.  Roughly a boat length and a half
    #: for an eight: shorter and the rudder is never still, longer and the
    #: boat rounds off the bends the line was drawn to hold.
    boundary_layer: float = 25.0
    #: Rudder per radian of heading error.  Matches the sign convention of
    #: :class:`~coxswain.sim.control.HeadingController`: positive rudder
    #: yaws to starboard, so a positive error calls for positive rudder.
    gain: float = 2.5
    #: Rudder per radian per second of yaw rate.
    rate_gain: float = 1.2
    #: Deflection limit, radians.
    max_rudder: float = np.radians(45.0)
    #: Current as a function of position, ``(x, y) -> (vx, vy)`` in the
    #: ground frame.  ``None`` for still water.
    current: Optional[Callable] = None
    #: Largest pressure split the coxswain will call for when the rudder
    #: alone has run out.
    max_split: float = 0.30

    # -- recorded each call, for inspection afterwards --------------------
    cross_track: float = field(default=0.0, init=False)
    station: float = field(default=0.0, init=False)
    split: float = field(default=0.0, init=False)
    crab: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.path = np.asarray(self.path, dtype=float)[:, :2]
        step = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.distance = np.concatenate([[0.0], np.cumsum(step)])
        self._last = 0

    def nearest(self, point) -> int:
        """Closest path point, searched forward only.

        Searching the whole path each step is both slow and wrong: where
        the river doubles back on itself the globally nearest point can be
        on another reach entirely, and the boat would be steered at it.
        Progress along the line is monotone, so the search starts where it
        left off.
        """
        lo = self._last
        hi = min(lo + 600, len(self.path))
        window = self.path[lo:hi]
        index = lo + int(np.argmin(np.linalg.norm(window - point[:2], axis=1)))
        self._last = min(index, len(self.path) - 2)
        return self._last

    def frame_at(self, index):
        """Unit tangent and signed curvature of the path at ``index``."""
        a = max(index - 2, 0)
        b = min(index + 2, len(self.path) - 1)
        tangent = self.path[b] - self.path[a]
        norm = np.linalg.norm(tangent)
        tangent = tangent / norm if norm > 1e-9 else np.array([1.0, 0.0])
        # curvature from three points, for the radial shift
        if 0 < index < len(self.path) - 1:
            p0, p1, p2 = self.path[index - 1:index + 2]
            v1, v2 = p1 - p0, p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(p2 - p0)
            curvature = 2.0 * cross / denom if denom > 1e-9 else 0.0
        else:
            curvature = 0.0
        return tangent, float(curvature)

    def look_ahead(self, position) -> np.ndarray:
        """Desired direction of travel over the ground."""
        index = self.nearest(position)
        self.station = float(self.distance[index])
        tangent, curvature = self.frame_at(index)

        error = self.path[index] - position[:2]        # toward the path
        magnitude = float(np.linalg.norm(error))
        if magnitude < 1e-6:
            return tangent
        normal = error / magnitude

        # Radial shift: a curved path needs the vehicle biased slightly
        # inside it, or a steady turn is flown with a standing offset.
        limit = max(abs(curvature), 1e-9)
        ratio = min(limit / max(1.0 / self.boundary_layer, limit), 1.0)
        shift = (1.0 - (2.0 / np.pi * np.arccos(np.clip(ratio, 0.0, 1.0))) ** 2)
        shift *= self.boundary_layer
        saturated = min((magnitude + shift) / self.boundary_layer, 1.0)

        theta = 0.5 * np.pi * np.sqrt(max(1.0 - saturated, 0.0))
        return np.cos(theta) * normal + np.sin(theta) * tangent

    def water_heading(self, ground_direction, position, speed) -> float:
        """Heading through the water that makes good ``ground_direction``.

        The flow triangle: the component of the current across the desired
        track has to be cancelled by pointing the boat up into it.
        """
        self.crab = 0.0
        if self.current is None or speed < 1e-6:
            return float(np.arctan2(ground_direction[1], ground_direction[0]))
        flow = np.asarray(self.current(position[0], position[1]),
                          dtype=float)[:2]
        across = np.array([-ground_direction[1], ground_direction[0]])
        sideways = float(np.dot(flow, across))
        ratio = sideways / speed
        if abs(ratio) >= 1.0:
            # The current across the track is faster than the boat: no
            # heading makes this track good.  Point straight into it and
            # take what is left, rather than returning a NaN.
            self.crab = float(np.sign(ratio) * 0.5 * np.pi)
        else:
            self.crab = float(np.arcsin(-ratio))
        aimed = _rotate(ground_direction, self.crab)
        return float(np.arctan2(aimed[1], aimed[0]))

    def __call__(self, t: float, state) -> float:
        position = np.asarray(state.position, dtype=float)[:2]
        direction = self.look_ahead(position)

        index = self._last
        tangent, _ = self.frame_at(index)
        across = np.array([-tangent[1], tangent[0]])
        self.cross_track = float(np.dot(position - self.path[index], across))

        speed = float(np.linalg.norm(np.asarray(state.velocity)[:2]))
        desired = self.water_heading(direction, position, speed)

        error = wrap_to_pi(state.yaw - desired)
        yaw_rate = float(state.omega_hull[2])
        demand = self.gain * error + self.rate_gain * yaw_rate

        self.split = 0.0
        if abs(demand) > self.max_rudder and self.max_split > 0.0:
            overflow = (abs(demand) - self.max_rudder) / self.max_rudder
            self.split = float(np.sign(demand) * self.max_split
                               * np.clip(overflow, 0.0, 1.0))
        return float(np.clip(demand, -self.max_rudder, self.max_rudder))
