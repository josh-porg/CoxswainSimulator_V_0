r"""The oar angle as a state, not a schedule.

What this replaces
------------------
:class:`~coxswain.crew.oarlock.OarAngleSweep` maps stroke phase to an oar
angle.  It is a *schedule*: the oar is where the clock says, whatever the
water is doing.  That is the reason the blade force cannot be made to
depend on speed -- substituting [CR06]'s ``F = C2 slip^2`` while leaving
the angle prescribed gives net propulsive impulse of about zero at racing
speed, because ``v cos(theta)`` overwhelms ``l theta_dot`` through
mid-drive, slip changes sign and the blade generates drag.

In [CR06] the oar angle is a *consequence*.  Here it is an *input*.  They
are the same equation solved in opposite directions, and this module
solves it the way the paper does::

    I phi_ddot = tau_handle(t) + l F_n(phi, phi_dot, v)

The rower pulls the handle, the water resists the blade, and the angle is
whatever the balance produces.

Sign conventions
----------------
``phi`` is measured from the boat's transverse axis, positive towards the
bow, exactly as :class:`OarAngleSweep` defines it: the drive runs from a
positive catch angle to a negative finish angle, so **``phi_dot`` is
negative through the drive** and a driving handle torque is negative.

``slip = l phi_dot + v cos(phi)`` is the normal component of blade
velocity relative to the water, identical to
:meth:`~coxswain.crew.oarlock.BladeModel.slip_velocity`.  The blade force
is normal to the shaft and opposes the slip, so its torque about the pin
opposes ``phi_dot`` -- which is what makes this a resistance and not a
second engine.

What the inertia has to be
--------------------------
**Not the oar's.**  A composite sweep oar has ``inertia_about_lock`` of
about 4.4 kg m^2, and a mid-drive blade torque of 1660 N m on that gives
``phi_ddot = 374 rad/s^2`` where the sweep it has to produce peaks near
11.  The inertia that matters is the rower's body reflected through the
inboard: 60 kg at 1.14 m is 78 kg m^2, twenty times the oar's own.

So :attr:`OarDynamics.inertia` is a **required argument with no default**,
and deliberately so.  There are two defensible ways to supply it -- a
lumped effective inertia per seat, or the reflected inertia of a
forward-dynamic rower -- and this module refuses to choose, because a
default here would silently become a fitted parameter that nobody
remembered fitting.  See ``docs/PHYSICS_PROGRAMME.md``.

What becomes an output
----------------------
Two things that are currently inputs:

* **The shape of the sweep.**  ``OarAngleSweep.flatness`` exists because
  the raised-cosine default over-drives the blade and somebody has to
  choose how much to flatten it.  Here nobody does.
* **The drive duration.**  With the angle dynamic you may prescribe *when*
  the rower extracts or *at what angle*, not both.  A rower extracts at a
  body position, so the finish angle is the natural input and the drive
  duration is a prediction -- testable against [HF09]'s on-water pairs,
  which the current ergometer-fitted formula misses by 18-28%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

__all__ = ["OarDynamics", "DriveResult"]


@dataclass(frozen=True)
class DriveResult:
    """One drive, integrated from the catch to the finish angle."""

    time: np.ndarray
    angle: np.ndarray
    rate: np.ndarray
    #: Normal blade force, N, signed along ``e_theta``.
    blade_force: np.ndarray
    #: Component of the blade force along the boat's x axis, N.
    propulsive: np.ndarray
    #: ``l phi_dot + v cos(phi)`` at each sample, m/s.
    slip: np.ndarray
    #: True when the finish angle was reached before ``max_time``.
    finished: bool

    @property
    def duration(self) -> float:
        """Predicted drive duration, s. An output, not a formula."""
        return float(self.time[-1])

    @property
    def propulsive_impulse(self) -> float:
        """``\\int F_x dt`` over the drive, N s.

        The quantity that came out at about zero when the same force model
        was substituted under a prescribed angle. It must be positive, and
        comfortably so, or the boat is not being driven.
        """
        return float(np.trapezoid(self.propulsive, self.time))

    @property
    def swept(self) -> float:
        """Total angle swept, radians."""
        return float(abs(self.angle[0] - self.angle[-1]))


@dataclass(frozen=True)
class OarDynamics:
    """One oar, pivoted at the pin, with the water on the other end.

    ``inertia`` is about the pin and **must include the rower's reflected
    mass**; see the module docstring for why it has no default.
    """

    blade: object                 # coxswain.crew.oarlock.BladeModel
    inboard: float                # r_h, pin to handle centre, m
    outboard: float               # l, pin to blade centre of pressure, m
    inertia: float                # I about the pin, kg m^2
    catch_angle: float = np.radians(55.0)
    finish_angle: float = np.radians(-35.0)

    def __post_init__(self) -> None:
        if self.inertia <= 0.0:
            raise ValueError("inertia about the pin must be positive")
        if self.outboard <= 0.0 or self.inboard <= 0.0:
            raise ValueError("inboard and outboard must be positive")
        if self.finish_angle >= self.catch_angle:
            raise ValueError(
                "the drive runs from a bow-ward catch to a stern-ward "
                f"finish, so finish_angle ({self.finish_angle}) must be "
                f"below catch_angle ({self.catch_angle})")

    # -- the water -------------------------------------------------------
    def slip(self, angle, rate, boat_speed):
        """``l phi_dot + v cos(phi)``, m/s."""
        return (self.outboard * np.asarray(rate, dtype=float)
                + np.asarray(boat_speed, dtype=float)
                * np.cos(np.asarray(angle, dtype=float)))

    def blade_torque(self, angle, rate, boat_speed,
                     water_depth=None, cover=None):
        """Torque about the pin from the blade, N m.

        The force is normal to the shaft and acts at ``outboard``, so the
        torque is ``l F_n``. It opposes ``phi_dot`` by construction --
        ``F_n`` carries the sign of ``-slip`` -- which is the restoring
        term the efficiency-only wiring lacked.
        """
        force = self.blade.normal_force(angle, rate, boat_speed,
                                        water_depth, cover)
        return self.outboard * np.asarray(force, dtype=float)

    # -- the balance -----------------------------------------------------
    def acceleration(self, angle, rate, handle_torque, boat_speed,
                     water_depth=None, cover=None):
        """``phi_ddot`` from the torque balance about the pin."""
        blade = self.blade_torque(angle, rate, boat_speed,
                                  water_depth, cover)
        return (np.asarray(handle_torque, dtype=float) + blade) / self.inertia

    def drive(self, handle_torque: Callable[[float], float],
              boat_speed, dt: float = 0.002, max_time: float = 4.0,
              water_depth: Optional[float] = None,
              cover: Optional[float] = None) -> DriveResult:
        """Integrate one drive, from the catch to the finish angle.

        ``handle_torque(t)`` returns the rower's torque about the pin in
        N m, **negative through the drive** (it drives ``phi`` down). It
        is a function of time and not of state: this stage keeps the
        rower open-loop and lets only the *angle* respond, which is what
        separates it from the forward-dynamic rower that comes later.

        ``boat_speed`` may be a float or a callable of time, so the same
        routine serves a fixed-speed study and a coupled run.

        Heun, because the right-hand side is quadratic in ``phi_dot`` and
        explicit Euler on a quadratic drag term drifts in a direction
        that flatters the model.
        """
        speed_at = (boat_speed if callable(boat_speed)
                    else (lambda _t: boat_speed))

        times = [0.0]
        angles = [float(self.catch_angle)]
        rates = [0.0]                 # the oar reverses at the catch

        t, angle, rate = 0.0, float(self.catch_angle), 0.0
        while t < max_time and angle > self.finish_angle:
            speed = float(speed_at(t))
            accel = float(self.acceleration(angle, rate, handle_torque(t),
                                            speed, water_depth, cover))
            predict_angle = angle + dt * rate
            predict_rate = rate + dt * accel

            speed_next = float(speed_at(t + dt))
            accel_next = float(self.acceleration(
                predict_angle, predict_rate, handle_torque(t + dt),
                speed_next, water_depth, cover))

            angle = angle + 0.5 * dt * (rate + predict_rate)
            rate = rate + 0.5 * dt * (accel + accel_next)
            t += dt

            times.append(t)
            angles.append(angle)
            rates.append(rate)

        time = np.asarray(times, dtype=float)
        angle_a = np.asarray(angles, dtype=float)
        rate_a = np.asarray(rates, dtype=float)
        speeds = np.asarray([float(speed_at(each)) for each in time],
                            dtype=float)

        slip = self.slip(angle_a, rate_a, speeds)
        force = np.asarray(self.blade.normal_force(
            angle_a, rate_a, speeds, water_depth, cover), dtype=float)
        return DriveResult(
            time=time, angle=angle_a, rate=rate_a,
            blade_force=force,
            propulsive=force * np.cos(angle_a),
            slip=slip,
            finished=bool(angle_a[-1] <= self.finish_angle),
        )

    # -- construction ----------------------------------------------------
    @classmethod
    def from_boat(cls, boat, inertia: float, seat: int = 0,
                  blade=None) -> "OarDynamics":
        """Build from a boat's own rig geometry and sweep limits.

        ``inertia`` stays required. Everything else -- inboard, outboard,
        catch and finish angles, and the blade coefficient -- comes from
        the boat, so a study cannot accidentally mix one boat's geometry
        with another's coefficients.
        """
        from .oarlock import BladeModel

        lock = boat.rig.seats[seat].oarlocks[0]
        if blade is None:
            sculling = not bool(boat.rig.is_sweep)
            maker = BladeModel.sculling if sculling else BladeModel.sweep
            blade = maker(outboard=float(lock.oar.outboard))
        return cls(
            blade=blade,
            inboard=float(lock.oar.inboard),
            outboard=float(lock.oar.outboard),
            inertia=float(inertia),
            catch_angle=float(boat.oar_sweep.catch_angle),
            finish_angle=float(boat.oar_sweep.finish_angle),
        )
