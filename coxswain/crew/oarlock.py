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


#: Exponents of the drive force curve, ``u**a (1-u)**b`` on the drive.
#: Fitted to two of Kleshnev's published figures at once: peak force at
#: **40% of the drive length**, and force already down to **74% of peak
#: by 60%** of the drive.  Both are reproduced to three figures.
DRIVE_SHAPE = (1.4852, 2.2278)

#: Mean of that shape over the drive, against ``2/pi`` for a half-sine.
#: The ratio is what keeps peak force and total impulse consistent when
#: switching between them.
DRIVE_SHAPE_MEAN = 0.53853

#: ``shift_per_spm`` that reproduces McBride's measurement: peak oar force
#: arrives **3.4% of the stroke cycle (3 degrees of oar angle) earlier** at
#: race pace than at 20 spm [W18]_.
#:
#: **The default model moves it the wrong way.**  With no shift the peak
#: arrives 4.63% of the cycle *later* at 36 spm than at 20, because
#: :attr:`StrokeTiming.drive_fraction` grows with rate and the peak sits at
#: a fixed fraction *of the drive*.  Against McBride's 3.4% earlier that is
#: an 8.0% discrepancy in the wrong direction -- a real defect that only
#: showed up because the review reports the observable in cycle terms.
#:
#: Not switched on by default: every speed calibration in the catalogue
#: predates it, exactly as with ``shape``.  Pass it explicitly.
#:
#: .. [W18] Warmenhoven, J., Cobley, S., Draper, C., Smith, R. (2018)
#:    *Over 50 Years of Researching Force Profiles in Rowing: What Do We
#:    Know?*, Sports Medicine 48:2703-2714, sec. 6.3.1, citing McBride.
MCBRIDE_SHIFT_PER_SPM = 0.010329


@dataclass(frozen=True)
class OarForceProfile:
    """Peak oarlock loads for one athlete, and the shape of the drive.

    **Why the shape is not a half-sine.**  It was, and a half-sine peaks
    at the middle of the drive.  Real force curves are front-loaded:
    Kleshnev puts the peak at 40% of the drive length, and by 60% the
    force has already fallen to 74% of peak, where a half-sine is still
    at 95%.

    That is not a cosmetic difference, and the way it showed up is worth
    recording.  Differencing the boat-mounted and rower-mounted IMUs from
    the club 2x session gives the crew's acceleration *relative to the
    hull* directly.  Measured, the hull's acceleration swing is 0.785 of
    the crew's -- **below** the crew mass fraction of 0.86, meaning the
    blade thrust partly cancels the crew's reaction.  The model had that
    ratio at 1.18, **above** the mass fraction: thrust was amplifying the
    crew reaction instead of opposing it, because a symmetric half-sine
    puts peak thrust at the same instant as the crew's own peak.

    Moving the peak to where Kleshnev measures it is what fixes the
    phasing.  See SOURCES sec. 38.
    """

    max_x: float = 1200.0  # N, longitudinal, at the peak of the drive
    max_z: float = 200.0   # N, vertical
    #: ``"kleshnev"`` for the fitted front-loaded curve, ``"half_sine"``
    #: for the previous symmetric one.  Kept switchable because every
    #: speed calibration in the catalogue predates the change.
    shape: str = "kleshnev"
    #: Extra front-loading, as a shift of the peak within the drive.
    #: Positive moves the peak EARLIER.  Zero reproduces the fitted
    #: Kleshnev shape exactly, so nothing changes unless it is asked for.
    #:
    #: This exists because the peak's position is not a constant of the
    #: rower: McBride reports peak oar force occurring **3.4% of the
    #: stroke cycle (3 degrees of oar angle) earlier** when the rate rises
    #: from 20 spm to race pace [W18]_.  A profile pinned at one position
    #: cannot represent that, and the rate question is exactly what this
    #: project keeps running into.
    peak_shift: float = 0.0
    #: Rate at which ``peak_shift`` is quoted, spm.  Above it the peak
    #: moves earlier and below it later, at ``shift_per_spm``.
    reference_rate: float = 20.0
    #: Fraction of the DRIVE that the peak moves earlier per spm.
    #: Calibrated so that 20 spm to 36 spm reproduces McBride's 3.4% of
    #: the cycle; see ``scripts/force_profile.py`` for the check, and note
    #: that the drive fraction itself grows with rate, so the cycle-level
    #: shift is not simply this times the drive.
    shift_per_spm: float = 0.0

    def peak_position(self, timing: StrokeTiming = None) -> float:
        """Where the peak sits within the drive, as a fraction in (0, 1)."""
        if self.shape == "half_sine":
            base = 0.5
        else:
            a, b = DRIVE_SHAPE
            base = a / (a + b)
        shift = float(self.peak_shift)
        if timing is not None and self.shift_per_spm:
            shift += self.shift_per_spm * (float(timing.rate)
                                           - self.reference_rate)
        return float(np.clip(base - shift, 0.05, 0.95))

    @staticmethod
    def _exponents(peak: float, total: float):
        """Beta exponents putting the mode at ``peak`` with fixed ``a+b``.

        Holding ``a + b`` fixed keeps the curve's *width* roughly constant
        while the mode moves, so shifting the peak does not silently also
        change how long the rower is near maximum load.
        """
        a = peak * total
        return a, total - a

    def magnitude(self, t, timing: StrokeTiming):
        """Shape factor in ``[0, 1]``; zero through the recovery."""
        t = np.asarray(t, dtype=float)
        phase_time = np.mod(t, timing.period)
        active = phase_time <= timing.drive_duration
        u = np.clip(phase_time / timing.drive_duration, 0.0, 1.0)
        if self.shape == "half_sine" and not (self.peak_shift
                                              or self.shift_per_spm):
            curve = np.sin(np.pi * u)
        else:
            base_a, base_b = DRIVE_SHAPE
            peak = self.peak_position(timing)
            a, b = self._exponents(peak, base_a + base_b)
            norm = peak ** a * (1.0 - peak) ** b
            curve = u ** a * (1.0 - u) ** b / norm
        return np.where(active, np.maximum(curve, 0.0), 0.0)

    def mean_to_peak(self, timing: StrokeTiming, samples: int = 400) -> float:
        """Mean force over the drive divided by peak force, ``MPFR``.

        A discriminating measure in the review literature: elite rowers
        show a **significantly higher** ratio than sub-elite [W18]_,
        because impulse is the area under the curve and a rectangular
        profile carries more of it for the same peak.  Chasing peak force
        alone is the classic mistake -- two rowers with identical peaks can
        differ substantially in what they deliver.

        This is a diagnostic on the profile, not an input to it.
        """
        u = (np.arange(samples) + 0.5) / samples
        curve = self.magnitude(u * timing.drive_duration, timing)
        return float(curve.mean() / max(curve.max(), 1e-12))


@dataclass(frozen=True)
class OarAngleSweep:
    """Oar shaft angle from the boat's transverse axis, positive to bow.

    The blade starts bow-ward at the catch and finishes stern-ward, which
    is what pushes water aft and the boat forward.
    """

    catch_angle: float = np.radians(55.0)
    finish_angle: float = np.radians(-35.0)
    #: Blend towards a linear ramp.  0 is a pure raised cosine, which has
    #: the right *end* behaviour: the oar angle has a genuine turning point
    #: at the catch and at the finish, so its rate is zero there.  Values
    #: above 0 flatten the mid-drive rate at the cost of reintroducing a
    #: corner at the ends.
    #:
    #: **The default of 0 is probably too peaked.**  A raised cosine peaks
    #: at 1.57 times the mean angular rate, which drives the blade through
    #: the water faster than the boat runs and so makes it slip hard.  Fed
    #: through :class:`BladeModel`, that gives a force-weighted blade
    #: efficiency of 0.73 over the drive, against the 0.80-0.85 Kleshnev
    #: reports for good crews.  A flatness near **0.30** reproduces the
    #: measured figure (0.82).  This is an independent constraint on the
    #: sweep shape that the blade model supplies and nothing else here
    #: does; the default is left at 0 only because the regression suite is
    #: pinned to it.  See ``docs/SOURCES.md`` section 7.
    flatness: float = 0.0
    #: Retiming of the recovery sweep: the oar, like the body it is
    #: attached to, moves back early in the recovery and arrives at the
    #: catch at this fraction of the nominal rate.  Must match the crew's
    #: ``recovery_arrival`` or the hands leave the handle.  1.0 = uniform.
    recovery_arrival: float = 1.0
    #: Full-cycle phase warp ``((phases...), (warped...))`` shared with the
    #: crew's joint drivers -- the hands are on the handle, so the oar must
    #: run on the same clock as the body.  ``None`` is the identity.
    warp_knots: tuple = None

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
        if self.warp_knots is not None:
            knots, images = self.warp_knots
            phase = np.interp(phase, np.asarray(knots, dtype=float),
                              np.asarray(images, dtype=float))
        drive = timing.drive_fraction

        on_drive = phase < drive
        drive_progress = np.clip(phase / drive, 0.0, 1.0)
        recovery_progress = np.clip((phase - drive) / (1.0 - drive), 0.0, 1.0)
        if self.recovery_arrival != 1.0:
            from .stroke import recovery_warp
            recovery_progress = recovery_warp(recovery_progress,
                                              self.recovery_arrival)

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

    def rate(self, t, timing: StrokeTiming):
        """Angular rate ``d(angle)/dt`` in rad/s.

        With a phase warp in place the analytic form no longer applies, so
        the rate is a central difference of the warped angle -- the result
        feeds Fourier fits and diagnostics, which smooth it anyway.

        The blade model needs this, and finite-differencing the angle would
        be both slower and noisier inside a derivative evaluation.  Signed
        the same way as the angle: negative through the drive, since the
        oar sweeps from a bow-ward catch to a stern-ward finish.
        """
        t = np.asarray(t, dtype=float)
        if self.warp_knots is not None:
            step = 1e-5 * timing.period
            return (np.asarray(self(t + step, timing))
                    - np.asarray(self(t - step, timing))) / (2.0 * step)
        phase = timing.phase(t)
        drive = timing.drive_fraction
        span = self.finish_angle - self.catch_angle

        on_drive = phase < drive
        drive_progress = np.clip(phase / drive, 0.0, 1.0)
        recovery_progress = np.clip((phase - drive) / (1.0 - drive), 0.0, 1.0)
        warp_slope = 1.0
        if self.recovery_arrival != 1.0:
            from .stroke import recovery_warp, recovery_warp_slope
            warp_slope = recovery_warp_slope(recovery_progress,
                                             self.recovery_arrival)
            recovery_progress = recovery_warp(recovery_progress,
                                              self.recovery_arrival)

        # d(phase)/dt is 1/period; each phase then rescales by its own span
        during_drive = (span * self._ramp_slope(drive_progress)
                        / (drive * timing.period))
        during_recovery = (-span * self._ramp_slope(recovery_progress)
                           * warp_slope / ((1.0 - drive) * timing.period))
        return np.where(on_drive, during_drive, during_recovery)

    def _ramp_slope(self, progress):
        """Derivative of :meth:`_ramp` with respect to its argument."""
        progress = np.clip(np.asarray(progress, dtype=float), 0.0, 1.0)
        cosine = 0.5 * np.pi * np.sin(np.pi * progress)
        flatness = float(np.clip(self.flatness, 0.0, 1.0))
        return (1.0 - flatness) * cosine + flatness

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
    #: Vertical extent of the blade face, in metres.  A World Rowing "big
    #: blade" (hatchet) is roughly 0.25 m across the widest part.
    blade_width: float = 0.25
    #: Depth of water covering the top edge of the blade, in metres.  The
    #: default is half a blade width, which is where Kleshnev puts the
    #: optimum -- see :meth:`immersion_factor`.
    cover: float = 0.125
    #: Steepness of the ventilation roll-off, dimensionless.  Set so that a
    #: cover of half a blade width recovers 90% of the deep-blade force.
    ventilation_k: float = 4.605
    #: Maskell bluff-body blockage constant.  2.5 is the standard value for
    #: bluff bodies; see :meth:`blockage_factor`.
    blockage_m: float = 2.5
    #: Free-stream drag coefficient of the blade face, used only by the
    #: blockage correction.  About 1.1 for a flat plate normal to the flow.
    blade_cd: float = 1.1

    @classmethod
    def sculling(cls, outboard: float = 2.28) -> "BladeModel":
        """[CR06]'s single-scull fit."""
        return cls(c2=58.7, outboard=outboard)

    @classmethod
    def sweep(cls, outboard: float = 2.28) -> "BladeModel":
        """[CR06]'s sweep fit."""
        return cls(c2=84.5, outboard=outboard)

    # -- depth of water around the blade ---------------------------------
    def immersion_factor(self, cover: float = None) -> float:
        """Force retained as a function of how deeply the blade is buried.

        A blade whose top edge sits at the surface **ventilates**: air is
        drawn down the low-pressure face, which collapses the pressure
        difference the blade works by.  This is the loss visible as surface
        tearing and vortices behind a washed-out blade.  Burying the blade
        suppresses it.

        Kleshnev puts the optimum at about **half a blade width of water
        over the blade**, and reports that modelling shows deeper immersion
        beats holding the blade at the surface.  Atkins argues the same
        qualitatively: at constant propulsive force, deeper immersion needs
        *less* slip, which is the definition of a more efficient blade.

        Modelled as ``1 - exp(-k c / W)`` in the cover ``c``, with ``k`` set
        so that ``c = W/2`` returns 0.90.  This is a *shape* chosen to
        respect the two things the sources agree on -- monotonic in cover,
        saturating near the reported optimum -- and not a fitted law.  No
        published force-versus-immersion curve for a rowing blade was
        found; see ``docs/SOURCES.md`` section 7.

        Over-immersion is deliberately **not** penalised here.  Kleshnev's
        3.5% speed loss for six degrees of extra blade depth is borne by
        the shaft and the vertical handle force, not by the blade face, so
        it does not belong in this factor.
        """
        cover = self.cover if cover is None else cover
        if cover <= 0.0:
            return 0.0
        return float(1.0 - np.exp(-self.ventilation_k * cover
                                  / self.blade_width))

    def blockage_factor(self, water_depth: float = None) -> float:
        """Force amplification from finite depth of water around the blade.

        The blade has to push water *around* itself.  With the free surface
        above and the bed below, that flow is confined to the water column,
        and the shallower the column the harder it is to get out of the way
        -- so the effective drag coefficient rises.  This is why the same
        crew at the same rate is not doing the same thing on the Charles as
        on a deep lake, quite apart from what the hull is doing.

        Maskell's bluff-body blockage correction::

            C_D(confined) = C_D(free) (1 + m sigma C_D(free))

        with vertical blockage ratio ``sigma = W / h`` for blade width ``W``
        and water depth ``h``, and ``m = 2.5`` the standard bluff-body
        constant.

        **This is an upper bound, not a calibrated fit.**  At
        ``sigma = 0.05`` it returns 1.14, against the "under 10% change in
        drag coefficient" reported for confined bluff bodies at that
        blockage.  Two reasons to read the model as conservative: that
        figure comes from ducts confined on all sides, whereas a blade is
        confined only vertically, which is weaker; and ``m = 2.5`` is the
        generic bluff-body value, not one measured for a blade.  Matching
        the cited datum exactly would need ``m`` near 1.8.  No
        blade-specific blockage measurement was found, so the standard
        constant is kept and the discrepancy stated rather than tuned away.

        Only vertical confinement is counted.  A river is laterally open at
        the scale of a blade, so there is no side-wall term.

        Returns 1.0 for infinite depth.
        """
        if water_depth is None or not np.isfinite(water_depth):
            return 1.0
        if water_depth <= self.blade_width:
            raise ValueError(
                f"water depth {water_depth:.3f} m is not deeper than the "
                f"blade width {self.blade_width:.3f} m; the blade would not "
                "fit in the water column"
            )
        sigma = self.blade_width / water_depth
        return float(1.0 + self.blockage_m * sigma * self.blade_cd)

    def depth_factor(self, water_depth: float = None,
                     cover: float = None) -> float:
        """Combined effect of blade cover and water depth on blade force.

        The two act in opposite directions: shallow burial *loses* force to
        ventilation, shallow water *gains* it to confinement.  On the
        Charles at 2-3 m both are active at once.
        """
        return self.immersion_factor(cover) * self.blockage_factor(water_depth)

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

    def normal_force(self, angle, angular_rate, boat_speed,
                     water_depth: float = None, cover: float = None):
        """Blade force magnitude, signed to oppose the slip.

        ``C2 * slip^2`` with the sign of ``-slip``, so the water always
        resists the blade rather than driving it.  Squaring alone would
        lose that and make the blade produce thrust on the recovery.

        ``water_depth`` and ``cover`` scale the result by
        :meth:`depth_factor`; leaving both ``None`` gives the deep-water,
        nominally-buried blade of [CR06].
        """
        slip = self.slip_velocity(angle, angular_rate, boat_speed)
        scale = (1.0 if (water_depth is None and cover is None)
                 else self.depth_factor(water_depth, cover))
        return -np.sign(slip) * self.c2 * scale * slip ** 2

    def propulsive_force(self, angle, angular_rate, boat_speed,
                         water_depth: float = None, cover: float = None):
        """Component of the blade force along the boat's ``x`` axis.

        The blade force is normal to the shaft, so only ``cos(theta)`` of
        it drives the boat.  Near the catch the oar is at a large angle and
        most of the blade load is lateral -- which is exactly the effect
        [B09] exploits in arguing for a less catch-heavy stroke.
        """
        normal = self.normal_force(angle, angular_rate, boat_speed,
                                   water_depth, cover)
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
