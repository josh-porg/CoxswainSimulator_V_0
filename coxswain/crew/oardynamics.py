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
inboard, which is twenty times the oar's own.

The first version of this module made :attr:`OarDynamics.inertia` a
required argument with no default, on the grounds that the alternative was
a lumped number chosen to make the answer come out -- and a fitted
parameter is the class of thing the physics programme exists to remove.

**It does not have to be fitted.**  The generalised inertia for the
coordinate ``phi`` is ``sum_i m_i |d x_i / d phi|^2``, and every term is
already in the model: de Leva segment masses, and segment velocities from
the joint chain.  :func:`reflected_inertia` computes it, and it is what
``from_boat`` uses by default.

It is not a constant.  On the catalogue eight it runs about 93 kg m^2
early in the drive down to 13 at the finish -- the legs move a lot of mass
per radian of oar and the arms very little -- so the balance carries the
term a varying inertia requires::

    I(phi) phi_ddot + (1/2) (dI/dphi) phi_dot^2 = tau_handle + l F_n

See ``docs/PHYSICS_PROGRAMME.md``.

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

__all__ = ["OarDynamics", "DriveResult", "reflected_inertia",
           "InertiaProfile"]


#: Fraction of the peak sweep rate below which the reduction to a single
#: coordinate is not trustworthy; see :func:`reflected_inertia`.
RATE_FLOOR = 0.25


def reflected_inertia(boat, seat: int = 0, samples: int = 61,
                      rate_floor: float = RATE_FLOOR):
    """Crew inertia reflected onto the oar angle, ``(angle, inertia)``.

    The generalised inertia for a coordinate ``phi`` is
    ``sum_i m_i |d x_i / d phi|^2``, and every term of that is already in
    the model: de Leva segment masses, and segment velocities in the hull
    frame from the joint chain.  ``d x_i / d phi`` is ``v_i / phi_dot``.

    **So this is derived, not fitted**, which matters: the alternative was
    a lumped effective inertia chosen to make the answer come out, and a
    fitted parameter is the class of thing the physics programme exists to
    remove.

    It is not a constant.  Measured on the catalogue eight it runs about
    93 kg m^2 early in the drive down to 13 at the finish, because the
    legs move a great deal of mass per radian of oar early on and the arms
    very little late.  A rower's handle speeding up through the second
    half of the drive is that, not a change of effort.

    Two limits on believing it, both real:

    * **It diverges at both ends.**  The prescribed crew motion does not
      stop when the prescribed oar sweep does -- the sweep is a raised
      cosine with exactly zero rate at the catch and the finish, and the
      body is still moving there -- so ``v_i / phi_dot`` blows up.  That
      is an inconsistency in the *current* kinematics, not a property of
      rowing, and it means the reduction to one coordinate is only valid
      over the interior of the drive.  Samples below ``rate_floor`` times
      the peak sweep rate are therefore dropped, and the caller is
      expected to clamp outside the range returned.
    * **It inherits the ergometer.**  The joint angles behind it are
      fitted from stationary-ergometer motion capture, which is the
      second defect the programme is fixing.  When the rower becomes
      forward-dynamic this quantity stops being computed and starts being
      a consequence of the multibody chain.

    Returns ``(angle, inertia)``, both 1-D and ordered from the catch
    towards the finish, i.e. by **decreasing** angle.
    """
    timing = boat.timing
    rower = boat.crew[seat].rower
    mass = np.asarray(rower.segment_masses, dtype=float)

    drive = timing.drive_fraction * timing.period
    t = np.linspace(0.0, drive, int(samples))
    rate = np.asarray(boat.oar_sweep.rate(t, timing), dtype=float)
    angle = np.asarray(boat.oar_sweep(t, timing), dtype=float)

    keep = np.abs(rate) >= float(rate_floor) * np.abs(rate).max()
    if not keep.any():
        raise ValueError("no part of the drive is above the rate floor")

    values = []
    for each in t[keep]:
        _position, velocity, _acceleration = rower.segment_state(float(each))
        speed2 = np.sum(np.asarray(velocity, dtype=float) ** 2, axis=1)
        values.append(float((mass * speed2).sum()))
    kinetic = np.asarray(values, dtype=float)
    return angle[keep], kinetic / rate[keep] ** 2


@dataclass(frozen=True)
class InertiaProfile:
    """Inertia about the pin as a function of oar angle.

    Wraps the table from :func:`reflected_inertia`, adds the oar's own
    inertia, and clamps outside the range where the reduction is valid.

    Configuration-dependent inertia is not a detail that can be dropped:
    with ``I`` a function of ``phi`` the equation of motion is

        I(phi) phi_ddot + (1/2) (dI/dphi) phi_dot^2 = tau

    and leaving out the second term would be wrong by the amount the
    inertia changes -- which here is a factor of seven across the drive.
    """

    angle: np.ndarray
    inertia: np.ndarray
    #: Added to every value: the oar's own second moment about the pin.
    oar_inertia: float = 0.0

    def __post_init__(self) -> None:
        angle = np.asarray(self.angle, dtype=float)
        inertia = np.asarray(self.inertia, dtype=float)
        if angle.shape != inertia.shape or angle.size < 2:
            raise ValueError("angle and inertia must be matching 1-D arrays "
                             "with at least two samples")
        if np.any(inertia <= 0.0):
            raise ValueError("reflected inertia must be positive")
        # np.interp needs increasing x; the table runs catch to finish.
        order = np.argsort(angle)
        object.__setattr__(self, "angle", angle[order])
        object.__setattr__(self, "inertia", inertia[order])

    def __call__(self, phi):
        """Total inertia about the pin at ``phi``, clamped at the ends."""
        return (np.interp(np.asarray(phi, dtype=float), self.angle,
                          self.inertia) + float(self.oar_inertia))

    @classmethod
    def of(cls, boat, seat: int = 0, **kwargs) -> "InertiaProfile":
        angle, inertia = reflected_inertia(boat, seat=seat, **kwargs)
        oar = boat.rig.seats[seat].oarlocks[0].oar
        return cls(angle=angle, inertia=inertia,
                   oar_inertia=float(oar.inertia_about_lock))


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
    #: I about the pin, kg m^2. A float, or a callable of the oar angle --
    #: see :class:`InertiaProfile`, which is what it should usually be,
    #: because the reflected crew inertia varies sevenfold over the drive.
    inertia: object
    catch_angle: float = np.radians(55.0)
    finish_angle: float = np.radians(-35.0)

    def __post_init__(self) -> None:
        if callable(self.inertia):
            probe = float(self.inertia(0.5 * (self.catch_angle
                                              + self.finish_angle)))
            if probe <= 0.0:
                raise ValueError("inertia about the pin must be positive")
        elif float(self.inertia) <= 0.0:
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
    def inertia_at(self, angle):
        """``(I, dI/dphi)`` at ``angle``.

        A scalar inertia gives a zero slope; a callable is differenced
        centrally, which is cheap and adequate because the profile it
        wraps is a linear interpolant of a smooth quantity.
        """
        if not callable(self.inertia):
            return float(self.inertia), 0.0
        step = 1e-4
        here = float(self.inertia(angle))
        ahead = float(self.inertia(angle + step))
        behind = float(self.inertia(angle - step))
        return here, (ahead - behind) / (2.0 * step)

    def acceleration(self, angle, rate, handle_torque, boat_speed,
                     water_depth=None, cover=None):
        """``phi_ddot`` from the torque balance about the pin.

        With a configuration-dependent inertia the balance is not
        ``I phi_ddot = tau``.  Lagrange gives

            I(phi) phi_ddot + (1/2) (dI/dphi) phi_dot^2 = tau

        and the second term is not small here: the reflected crew inertia
        changes by a factor of seven across the drive, so dropping it
        would be a first-order error, not a refinement.  Physically it is
        why a handle speeds up through the second half of the drive
        without the rower pulling any harder -- the effective mass being
        accelerated falls away as the legs finish and the arms take over.
        """
        blade = self.blade_torque(angle, rate, boat_speed,
                                  water_depth, cover)
        moment, slope = self.inertia_at(angle)
        rate = np.asarray(rate, dtype=float)
        return ((np.asarray(handle_torque, dtype=float) + blade
                 - 0.5 * slope * rate ** 2) / moment)

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
    def from_boat(cls, boat, inertia=None, seat: int = 0,
                  blade=None) -> "OarDynamics":
        """Build from a boat's own rig geometry and sweep limits.

        ``inertia`` defaults to :meth:`InertiaProfile.of` -- the crew's
        own inertia reflected onto the oar angle, computed from de Leva
        segment masses and the joint chain, plus the oar's own second
        moment. Derived, not fitted. Pass a float to override it with a
        constant, which is a study and not the default for a reason.
        """
        if inertia is None:
            inertia = InertiaProfile.of(boat, seat=seat)
        from .oarlock import BladeModel

        lock = boat.rig.seats[seat].oarlocks[0]
        if blade is None:
            sculling = not bool(boat.rig.is_sweep)
            maker = BladeModel.sculling if sculling else BladeModel.sweep
            # the blade force acts at the blade's centre, not its tip
            blade = maker(outboard=float(lock.oar.blade_centre_outboard))
        return cls(
            blade=blade,
            inboard=float(lock.oar.inboard),
            outboard=float(lock.oar.blade_centre_outboard),
            inertia=inertia if callable(inertia) else float(inertia),
            catch_angle=float(boat.oar_sweep.catch_angle),
            finish_angle=float(boat.oar_sweep.finish_angle),
        )
