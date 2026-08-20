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

__all__ = [
    "BladeModel","OarForceProfile", "OarAngleSweep", "oar_force", "hull_load",
           "oar_axis", "blade_position", "handle_position"]


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
    #: Blend towards a linear ramp.  The default of 0 is a pure raised
    #: cosine, which is the physically correct choice: the oar angle has a
    #: genuine turning point at the catch and at the finish, so its rate is
    #: zero there.  A raised cosine peaks at 1.57 times the mean angular
    #: rate mid-drive, which matches measured oar-angle traces.  Values
    #: above 0 flatten the mid-drive rate at the cost of reintroducing a
    #: corner at the ends.
    flatness: float = 0.0

    def __call__(self, t, timing: StrokeTiming):
        """Oar angle at time ``t``.

        The oar **reverses direction** at the catch and again at the
        finish, so its angular rate must pass through zero there.  A
        sweep that is linear in stroke phase instead steps the rate
        discontinuously, which puts a corner in the handle path -- and
        since the rower's hands are constrained to the handle, that corner
        propagates straight into the crew's segment accelerations and
        hence into the hull forces.

        Each phase therefore uses a raised-cosine ramp blended towards a
        linear one by ``flatness``: zero rate at both ends, near-constant
        rate through the middle of the drive, which is what measured oar
        angle traces show.
        """
        t = np.asarray(t, dtype=float)
        phase = timing.phase(t)
        drive = timing.drive_fraction

        on_drive = phase < drive
        drive_progress = np.clip(phase / drive, 0.0, 1.0)
        recovery_progress = np.clip((phase - drive) / (1.0 - drive), 0.0, 1.0)

        span = self.finish_angle - self.catch_angle
        during_drive = self.catch_angle + span * self._ramp(drive_progress)
        during_recovery = self.finish_angle - span * self._ramp(
            recovery_progress)

        return np.where(on_drive, during_drive, during_recovery)

    def _ramp(self, progress):
        """Monotone 0 to 1 map with zero slope at both ends.

        ``flatness = 0`` is a pure raised cosine; ``flatness = 1`` is the
        linear ramp, which reintroduces the corner.
        """
        progress = np.clip(np.asarray(progress, dtype=float), 0.0, 1.0)
        cosine = 0.5 * (1.0 - np.cos(np.pi * progress))
        flatness = float(np.clip(self.flatness, 0.0, 1.0))
        return (1.0 - flatness) * cosine + flatness * progress

    @property
    def total_sweep(self) -> float:
        return abs(self.catch_angle - self.finish_angle)


@dataclass(frozen=True)
class BladeModel:
    """Slip-dependent blade force, after Cabrera, Ruina & Kleshnev (2006).

    The prescribed half-sine of :class:`OarForceProfile` is an *open loop*:
    the crew pushes the same force whatever the boat is doing.  A real
    blade does not.  Its force comes from pushing water, so it depends on
    how fast the blade is actually moving through that water -- the slip
    velocity -- which depends on boat speed.  That feedback is what makes
    a blade lose grip as the boat runs away from it near the finish, and
    it is absent from a prescribed profile.

    [CR06] eq. (11), their "Model 1", due to Pope and to Alexander (1925):
    the resultant blade force is normal to the blade and proportional to
    the square of the normal component of blade slip velocity,

        v_O    = v_b sin(theta) e_r + (l theta_dot + v_b cos(theta)) e_theta
        F_oar  = C2 (l theta_dot + v_b cos(theta))^2

    with ``theta`` the oar angle from the boat's transverse axis, ``l``
    the outboard length, ``v_b`` the boat speed, and

        C2 = 0.5 rho C0 A0

    for blade face area ``A0`` and a shape constant ``C0``.  [CR06] fit
    ``C2`` to on-water force and kinematic data: **58.7** for a single
    scull and **84.5** for sweep.  Their sensitivity analysis found the fit
    quality more sensitive to the blade and hull drag coefficients than to
    any other parameter, and that allowing slip at all was a *necessary*
    ingredient -- a non-slipping blade (their C_D = 1) does not reproduce
    the data.

    Their Model 2 resolves lift and drag against angle of attack, after
    Wang, Birch & Dickinson's hovering-insect-wing treatment.  It is not
    implemented here; [H10] separately shows the pure normal-force
    assumption used here understates blade losses by about 18%, which
    bounds the error this carries.

    This is offered alongside the prescribed profile rather than replacing
    it: the prescribed path is what the whole regression suite is pinned
    to, and swapping the force model changes every number in it.
    """

    #: Equivalent blade force coefficient, N s^2/m^2.  [CR06] Table 3.
    c2: float = 84.5
    #: Outboard length, oarlock to blade centre of pressure, in metres.
    outboard: float = 2.28

    @classmethod
    def sculling(cls, outboard: float = 2.28) -> "BladeModel":
        """[CR06]'s single-scull fit."""
        return cls(c2=58.7, outboard=outboard)

    @classmethod
    def sweep(cls, outboard: float = 2.28) -> "BladeModel":
        """[CR06]'s sweep fit."""
        return cls(c2=84.5, outboard=outboard)

    def slip_velocity(self, angle, angular_rate, boat_speed):
        """Normal component of blade velocity relative to the water.

        ``l theta_dot + v_b cos(theta)``.  Negative during the drive with
        the sign convention of :class:`OarAngleSweep`, where the angle
        *decreases* from catch to finish; the blade is then sweeping
        sternward through the water and pushing the boat forward.
        """
        return (self.outboard * np.asarray(angular_rate, dtype=float)
                + np.asarray(boat_speed, dtype=float)
                * np.cos(np.asarray(angle, dtype=float)))

    def normal_force(self, angle, angular_rate, boat_speed):
        """Blade force magnitude, signed to oppose the slip.

        ``C2 * slip^2`` with the sign of ``-slip``, so the water always
        resists the blade rather than driving it.  Squaring alone would
        lose that and make the blade produce thrust on the recovery.
        """
        slip = self.slip_velocity(angle, angular_rate, boat_speed)
        return -np.sign(slip) * self.c2 * slip ** 2

    def propulsive_force(self, angle, angular_rate, boat_speed):
        """Component of the blade force along the boat's ``x`` axis.

        The blade force is normal to the shaft, so only ``cos(theta)`` of
        it drives the boat.  Near the catch the oar is at a large angle and
        most of the blade load is lateral -- which is exactly the effect
        [B09] exploits in arguing for a less catch-heavy stroke.
        """
        normal = self.normal_force(angle, angular_rate, boat_speed)
        return normal * np.cos(np.asarray(angle, dtype=float))

    def efficiency(self, angle, angular_rate, boat_speed):
        """Fraction of blade power that goes into moving the boat.

        ``1 - |slip| / |blade speed|``: the classic definition of blade
        efficiency, and the quantity the fixed ``blade_efficiency = 0.78``
        elsewhere in this module stands in for.  Returns 0 when the blade
        is not moving relative to the boat.
        """
        slip = np.abs(self.slip_velocity(angle, angular_rate, boat_speed))
        blade_speed = np.abs(self.outboard
                             * np.asarray(angular_rate, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            value = 1.0 - slip / blade_speed
        return np.where(blade_speed > 1e-9, np.clip(value, 0.0, 1.0), 0.0)


def oar_force(t, timing: StrokeTiming, side: int,
              profile: OarForceProfile = None,
              sweep: OarAngleSweep = None) -> np.ndarray:
    """Force applied by one rower at one oarlock, in the hull frame.

    The half-sine of eq. (15) sets the *magnitude* of the horizontal load;
    the oar angle sets its direction, the load acting perpendicular to the
    shaft::

        f_x = |F| cos(phi)          f_y = side |F| sin(phi)

    so the propulsive component tapers away near the catch and finish,
    where the oar is at a large angle and most of the load goes sideways.
    Resolving the magnitude onto ``x`` only, and deriving ``f_y`` from
    ``f_x tan(phi)``, gets this backwards: it makes the total load
    *largest* at the catch.

    Returns a ``(3,)`` array ``[f_x, f_y, f_z]`` in the hull frame.
    """
    profile = profile or OarForceProfile()
    sweep = sweep or OarAngleSweep()

    shape = float(profile.magnitude(t, timing))
    angle = float(sweep(t, timing))

    horizontal = profile.max_x * shape
    return np.array([
        horizontal * np.cos(angle),
        side * horizontal * np.sin(angle),
        profile.max_z * shape,
    ])


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


def oar_axis(t, timing: StrokeTiming, side: int,
             sweep: "OarAngleSweep" = None) -> np.ndarray:
    """Unit vector along the oar shaft, oarlock -> blade, in the hull frame.

    The shaft lies in the horizontal plane of the hull.  ``angle`` is
    measured from the transverse axis, positive towards the bow, so at the
    catch (positive angle) the blade is bow-ward of its oarlock and at the
    finish it is stern-ward -- the sweep that drives the boat forward.

    ``side`` selects which way the shaft points laterally: a port oarlock
    carries an oar reaching out to port.
    """
    sweep = sweep or OarAngleSweep()
    angle = np.asarray(sweep(t, timing), dtype=float)
    return np.stack([
        np.sin(angle),
        side * np.cos(angle),
        np.zeros_like(angle),
    ], axis=-1)


def blade_position(t, timing: StrokeTiming, oarlock,
                   sweep: "OarAngleSweep" = None) -> np.ndarray:
    """Blade centre in the hull frame, ``outboard`` out along the shaft."""
    axis = oar_axis(t, timing, oarlock.side, sweep)
    return np.asarray(oarlock.position, dtype=float) + oarlock.oar.outboard * axis


def handle_position(t, timing: StrokeTiming, oarlock,
                    sweep: "OarAngleSweep" = None,
                    grip_offset: float = 0.0) -> np.ndarray:
    """Grip point in the hull frame, back along the shaft from the oarlock.

    This is where a hand *must* be if the rower is holding the oar.

    ``grip_offset`` moves the grip towards the oarlock from the end of the
    handle.  It is zero for the outside hand, which takes the very end, and
    :attr:`~coxswain.boats.rig.Oar.grip_separation` for the inside hand of
    a sweep rower.  A sculler has one hand per oar and uses zero for both.
    """
    axis = oar_axis(t, timing, oarlock.side, sweep)
    reach = oarlock.oar.inboard - float(grip_offset)
    return np.asarray(oarlock.position, dtype=float) - reach * axis
