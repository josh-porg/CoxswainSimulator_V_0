"""The 6-DOF state vector and its named views.

The state is a flat length-12 vector so that it can be handed straight to
``scipy.integrate.solve_ivp`` (and, later, to a trajectory-optimisation
transcription).  Working with raw indices is what let the legacy code
disagree with itself about Euler-angle ordering, so every access goes
through :class:`State`, which is a thin named view over the flat array.

Layout::

    [0:3]   position   G_h in the absolute frame           [m]
    [3:6]   attitude   [roll, pitch, yaw]                  [rad]
    [6:9]   velocity   d(G_h)/dt in the absolute frame     [m/s]
    [9:12]  omega      angular velocity, absolute frame    [rad/s]

Both ``velocity`` and ``omega`` are expressed in the **absolute** frame,
matching the formulation of Formaggia et al. eq. (14).  Body-frame views are
available as properties for the hydrodynamics, which are naturally
body-frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import frames

__all__ = ["STATE_SIZE", "SLICES", "State", "pack", "unpack"]

STATE_SIZE = 12

SLICES = {
    "position": slice(0, 3),
    "attitude": slice(3, 6),
    "velocity": slice(6, 9),
    "omega": slice(9, 12),
}


@dataclass(frozen=True)
class State:
    """Named, read-only view of a 6-DOF state vector.

    ``State`` does not copy: :meth:`from_vector` wraps views onto the
    supplied array where possible, so constructing one inside a derivative
    evaluation is cheap.
    """

    position: np.ndarray
    attitude: np.ndarray
    velocity: np.ndarray
    omega: np.ndarray

    # -- construction ----------------------------------------------------
    @classmethod
    def from_vector(cls, y: np.ndarray) -> "State":
        y = np.asarray(y, dtype=float)
        if y.shape != (STATE_SIZE,):
            raise ValueError(
                f"state vector must have shape ({STATE_SIZE},), got {y.shape}"
            )
        return cls(
            position=y[SLICES["position"]],
            attitude=y[SLICES["attitude"]],
            velocity=y[SLICES["velocity"]],
            omega=y[SLICES["omega"]],
        )

    @classmethod
    def zeros(cls) -> "State":
        return cls.from_vector(np.zeros(STATE_SIZE))

    @classmethod
    def create(cls, position=(0.0, 0.0, 0.0), attitude=(0.0, 0.0, 0.0),
               velocity=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)) -> "State":
        return cls(
            position=np.asarray(position, dtype=float),
            attitude=np.asarray(attitude, dtype=float),
            velocity=np.asarray(velocity, dtype=float),
            omega=np.asarray(omega, dtype=float),
        )

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.position, self.attitude, self.velocity, self.omega]
        )

    # -- named attitude components --------------------------------------
    @property
    def roll(self) -> float:
        return float(self.attitude[frames.ROLL])

    @property
    def pitch(self) -> float:
        return float(self.attitude[frames.PITCH])

    @property
    def yaw(self) -> float:
        return float(self.attitude[frames.YAW])

    # -- derived frames --------------------------------------------------
    @property
    def rot_hull_to_abs(self) -> np.ndarray:
        return frames.hull_to_abs(self.attitude)

    @property
    def velocity_hull(self) -> np.ndarray:
        """Translational velocity resolved in the hull frame ``(u, v, w)``."""
        return frames.abs_to_hull(self.attitude) @ self.velocity

    @property
    def omega_hull(self) -> np.ndarray:
        """Angular velocity resolved in the hull frame ``(p, q, r)``."""
        return frames.abs_to_hull(self.attitude) @ self.omega

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def surge_speed(self) -> float:
        """Forward speed along the hull ``x`` axis."""
        return float(self.velocity_hull[0])

    @property
    def sideslip(self) -> float:
        """Sideslip angle ``beta = atan2(v, u)`` in radians, hull frame."""
        u, v, _ = self.velocity_hull
        if abs(u) < 1e-9 and abs(v) < 1e-9:
            return 0.0
        return float(np.arctan2(v, u))

    def replace(self, **kwargs) -> "State":
        """Return a copy with the named components replaced."""
        fields = {
            "position": self.position,
            "attitude": self.attitude,
            "velocity": self.velocity,
            "omega": self.omega,
        }
        fields.update({k: np.asarray(v, dtype=float) for k, v in kwargs.items()})
        return State(**fields)


def pack(position, attitude, velocity, omega) -> np.ndarray:
    """Assemble a flat state vector from its four components."""
    return np.concatenate([
        np.asarray(position, dtype=float),
        np.asarray(attitude, dtype=float),
        np.asarray(velocity, dtype=float),
        np.asarray(omega, dtype=float),
    ])


def unpack(y: np.ndarray):
    """Split a flat state vector into ``(position, attitude, velocity, omega)``."""
    y = np.asarray(y, dtype=float)
    return (y[SLICES["position"]], y[SLICES["attitude"]],
            y[SLICES["velocity"]], y[SLICES["omega"]])
