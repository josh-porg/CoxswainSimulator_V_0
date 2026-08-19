"""Simulation output container and derived metrics.

Keeps the raw state history plus the named views and cycle-averaged
metrics that the validation and regression tests assert on, so those
tests read as statements about rowing rather than about array indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.frames import PITCH, ROLL, YAW, abs_to_hull
from ..core.state import SLICES

__all__ = ["SimulationResult"]


@dataclass
class SimulationResult:
    """Time history of a run."""

    time: np.ndarray          # (n,)
    states: np.ndarray        # (12, n)
    boat: object

    # -- raw views --------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.states[SLICES["position"]]

    @property
    def attitude(self) -> np.ndarray:
        return self.states[SLICES["attitude"]]

    @property
    def velocity(self) -> np.ndarray:
        return self.states[SLICES["velocity"]]

    @property
    def omega(self) -> np.ndarray:
        return self.states[SLICES["omega"]]

    # -- named channels ---------------------------------------------------
    @property
    def surge(self) -> np.ndarray:
        return self.position[0]

    @property
    def sway(self) -> np.ndarray:
        return self.position[1]

    @property
    def heave(self) -> np.ndarray:
        return self.position[2]

    @property
    def roll(self) -> np.ndarray:
        return self.attitude[ROLL]

    @property
    def pitch(self) -> np.ndarray:
        return self.attitude[PITCH]

    @property
    def yaw(self) -> np.ndarray:
        return self.attitude[YAW]

    @property
    def speed(self) -> np.ndarray:
        """Absolute-frame speed magnitude."""
        return np.linalg.norm(self.velocity, axis=0)

    @property
    def surge_speed(self) -> np.ndarray:
        """Forward speed along the hull ``x`` axis, ``u``."""
        return np.array([
            abs_to_hull(self.attitude[:, i]) @ self.velocity[:, i]
            for i in range(len(self.time))
        ])[:, 0]

    @property
    def is_finite(self) -> bool:
        return bool(np.isfinite(self.states).all())

    # -- cycle metrics ----------------------------------------------------
    def last_cycles(self, n_cycles: int = 2) -> slice:
        """Index slice covering the last ``n_cycles`` strokes.

        Metrics should be taken here, after the start-up transient.
        """
        window = n_cycles * self.boat.timing.period
        start = np.searchsorted(self.time, self.time[-1] - window)
        return slice(int(start), len(self.time))

    def mean_speed(self, n_cycles: int = 2) -> float:
        return float(self.surge_speed[self.last_cycles(n_cycles)].mean())

    def speed_fluctuation(self, n_cycles: int = 2) -> float:
        """Peak-to-peak surge speed over the last whole strokes."""
        return float(np.ptp(self.surge_speed[self.last_cycles(n_cycles)]))

    def speed_fluctuation_ratio(self, n_cycles: int = 2) -> float:
        """Peak-to-peak surge speed as a fraction of the mean."""
        return self.speed_fluctuation(n_cycles) / self.mean_speed(n_cycles)

    def heave_amplitude(self, n_cycles: int = 2) -> float:
        return float(np.ptp(self.heave[self.last_cycles(n_cycles)]))

    def pitch_amplitude(self, n_cycles: int = 2) -> float:
        """Peak-to-peak pitch in radians."""
        return float(np.ptp(self.pitch[self.last_cycles(n_cycles)]))

    def roll_amplitude(self, n_cycles: int = 2) -> float:
        return float(np.ptp(self.roll[self.last_cycles(n_cycles)]))

    def yaw_amplitude(self, n_cycles: int = 2) -> float:
        return float(np.ptp(self.yaw[self.last_cycles(n_cycles)]))

    def distance(self) -> float:
        return float(np.hypot(self.surge[-1] - self.surge[0],
                              self.sway[-1] - self.sway[0]))

    def summary(self) -> dict:
        """Everything the regression tests compare, in one dict."""
        return {
            "mean_speed": self.mean_speed(),
            "speed_fluctuation": self.speed_fluctuation(),
            "speed_fluctuation_ratio": self.speed_fluctuation_ratio(),
            "heave_amplitude": self.heave_amplitude(),
            "pitch_amplitude_deg": np.degrees(self.pitch_amplitude()),
            "roll_amplitude_deg": np.degrees(self.roll_amplitude()),
            "yaw_amplitude_deg": np.degrees(self.yaw_amplitude()),
            "distance": self.distance(),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"SimulationResult({self.boat.name!r}, "
                f"{len(self.time)} steps, t={self.time[-1]:.1f} s)")
