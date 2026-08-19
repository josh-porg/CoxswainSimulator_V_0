"""Oarlock forces and the force/moment they deliver to the hull.

Force profile
-------------
Formaggia et al. eq. (15) fit measured oarlock loads with a half sine over
the drive::

    f_o_x = F_max_x sin(pi t / tau_a)
    f_o_z = F_max_z sin(pi t / tau_a)      0 <= t <= tau_a
    f_o   = 0                              during the recovery

with typical peaks ``F_max_x = 1200 N`` and ``F_max_z = 200 N``.  The
profile is continuous (it vanishes at both the catch and the finish), so
it introduces no impulsive load.

Lateral component
-----------------
The paper works in the boat's symmetry plane and so has no ``y``
component.  Six degrees of freedom need one, and it matters: it is what
makes a sweep boat wag its stern.  The oar sweeps through an angle
``phi(t)`` measured from the boat's transverse axis, and the horizontal
oarlock load acts perpendicular to the shaft, giving

    f_o_y = side * f_o_x * tan(phi)

For a sculler the two oars carry opposite ``side`` and the lateral force
cancels exactly, recovering the paper's planar model.  For a sweep rig it
cancels only in the *sum*; the port and starboard oars act at different
stations, leaving a residual yaw moment each stroke.

Transmission to the hull
------------------------
An ideal lever (eq. 12) puts ``F_h = -(L - r_h)/L F_o`` at the handle, so
the hull receives

    force  = (r_h / L) F_o
    moment = (x_o - x_h + (r_h / L) x_h) x F_o

per eq. (14a) and (14b), where ``x_o`` and ``x_h`` are the oarlock and
hand positions relative to ``G_h``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stroke import StrokeTiming

__all__ = ["OarForceProfile", "OarAngleSweep", "oar_force", "hull_load"]


@dataclass(frozen=True)
class OarForceProfile:
    """Peak oarlock loads for one athlete."""

    max_x: float = 1200.0  # N, longitudinal
    max_z: float = 200.0   # N, vertical

    def magnitude(self, t, timing: StrokeTiming):
        """Half-sine shape factor in ``[0, 1]``; zero during the recovery."""
        t = np.asarray(t, dtype=float)
        phase_time = np.mod(t, timing.period)
        active = phase_time <= timing.drive_duration
        shape = np.sin(np.pi * phase_time / timing.drive_duration)
        return np.where(active, np.maximum(shape, 0.0), 0.0)


@dataclass(frozen=True)
class OarAngleSweep:
    """Oar shaft angle from the boat's transverse axis, positive to bow.

    The blade starts bow-ward at the catch and finishes stern-ward, which
    is what pushes water aft and the boat forward.
    """

    catch_angle: float = np.radians(55.0)
    finish_angle: float = np.radians(-35.0)

    def __call__(self, t, timing: StrokeTiming):
        """Oar angle at time ``t``.

        Sweeps linearly in stroke phase across the drive and returns over
        the recovery, matching the blade being out of the water.
        """
        t = np.asarray(t, dtype=float)
        phase = timing.phase(t)
        drive = timing.drive_fraction

        on_drive = phase < drive
        drive_progress = np.clip(phase / drive, 0.0, 1.0)
        recovery_progress = np.clip((phase - drive) / (1.0 - drive), 0.0, 1.0)

        span = self.finish_angle - self.catch_angle
        during_drive = self.catch_angle + span * drive_progress
        during_recovery = self.finish_angle - span * recovery_progress

        return np.where(on_drive, during_drive, during_recovery)

    @property
    def total_sweep(self) -> float:
        return abs(self.catch_angle - self.finish_angle)


def oar_force(t, timing: StrokeTiming, side: int,
              profile: OarForceProfile = None,
              sweep: OarAngleSweep = None) -> np.ndarray:
    """Force applied by one rower at one oarlock, in the hull frame.

    Returns a ``(3,)`` array ``[f_x, f_y, f_z]``.
    """
    profile = profile or OarForceProfile()
    sweep = sweep or OarAngleSweep()

    shape = float(profile.magnitude(t, timing))
    f_x = profile.max_x * shape
    f_z = profile.max_z * shape
    f_y = side * f_x * np.tan(float(sweep(t, timing)))

    return np.array([f_x, f_y, f_z])


def hull_load(force: np.ndarray, oarlock_position: np.ndarray,
              hand_position: np.ndarray, gearing: float):
    """Force and moment delivered to the hull by one oar.

    Parameters
    ----------
    force:
        Oarlock force ``F_o`` in the hull frame.
    oarlock_position, hand_position:
        ``x_o`` and ``x_h`` relative to ``G_h``, hull frame.
    gearing:
        ``r_h / L`` from :attr:`coxswain.boats.rig.Oar.gearing`.

    Returns
    -------
    ``(force, moment)`` in the hull frame.
    """
    net_force = gearing * np.asarray(force, dtype=float)
    lever = (np.asarray(oarlock_position, dtype=float)
             - np.asarray(hand_position, dtype=float)
             + gearing * np.asarray(hand_position, dtype=float))
    return net_force, np.cross(lever, force)
