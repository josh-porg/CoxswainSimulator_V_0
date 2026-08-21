"""How a crew actually applies a balance moment, and what else it does.

The balance reflex used to be a pure couple about the hull ``x`` axis: a
saturated PD loop on roll, added straight to the moment vector.  That is a
reduction, and in a sweep boat it is not a harmless one.

A crew has no way to apply a pure roll couple.  What they can do is change
the height of their hands, which loads the oar as a lever about the
oarlock and puts a **vertical force at the rigger**.  The riggers are at
fixed points on the hull, so the moment the crew can produce is whatever
that set of point forces happens to make -- and in a sweep boat rigged the
normal alternating way, the port and starboard riggers are not at the same
longitudinal stations.

For the eight in this catalogue the port oarlocks average ``x = -0.34 m``
and the starboard ``x = +0.88 m``: a 1.22 m offset, exactly one seat
spacing.  A balance effort that pushes one side down and lifts the other
therefore applies four downward forces that sit 1.22 m from the four
upward ones, which is a couple in the *vertical-longitudinal* plane as
well as the intended one in the transverse plane.

**Balancing a sweep eight pitches it.**  The coupling is not small: for
this rig the pitch moment is about 0.72 of the roll moment.  Nothing about
that is exotic -- it is a direct consequence of alternating rigging, and
it is invisible to any model that adds balance as a pure ``x`` couple.

The net *vertical force* does cancel, so the old docstring's claim was
right as far as it went.  It was the pitch term it was silent about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BalanceRig", "PhaseAuthority", "static_roll_stiffness",
           "roll_inertia", "roll_divergence_time",
           "trunk_lean_authority"]


@dataclass(frozen=True)
class BalanceRig:
    """Geometry linking a crew's balance effort to the load on the hull.

    The crew's effort is parameterised by a single scalar ``f``: the
    vertical force one oarlock carries, positive up on the port side and
    negative on starboard (or the reverse, which just flips the sign of
    every output).  Everything the hull feels is linear in ``f``, so the
    whole mechanism reduces to three exact coefficients -- no smoothing,
    no fit, and CasADi differentiates it trivially.
    """

    roll_per_unit: float      # N m of roll moment per newton of rigger load
    pitch_per_unit: float     # N m of pitch moment, the coupling term
    heave_per_unit: float     # N of net vertical force; zero for even rigs

    @classmethod
    def from_boat(cls, boat) -> "BalanceRig":
        roll = pitch = heave = 0.0
        for seat in boat.rig.seats:
            for lock in seat.oarlocks:
                x, y, _ = np.asarray(lock.position, dtype=float)
                sign = float(lock.side)
                # a vertical force F_z at (x, y) gives moment (y F_z, -x F_z, 0)
                roll += y * sign
                pitch += -x * sign
                heave += sign
        return cls(float(roll), float(pitch), float(heave))

    @property
    def pitch_coupling(self) -> float:
        """Pitch moment produced per unit of roll moment demanded.

        Zero for a sculling boat and for any rig symmetric fore-and-aft;
        about 0.72 for a conventionally rigged eight.
        """
        if self.roll_per_unit == 0.0:
            return 0.0
        return self.pitch_per_unit / self.roll_per_unit

    def loads(self, roll_moment):
        """``(force, moment)`` in the hull frame for a demanded roll moment.

        Works with floats or CasADi expressions.  The demanded roll moment
        is delivered exactly; the pitch moment and any net heave come along
        with it because the rig's geometry says they must.
        """
        if self.roll_per_unit == 0.0:
            zero = roll_moment * 0.0
            return (zero, zero, zero), (zero, zero, zero)
        unit = roll_moment / self.roll_per_unit
        zero = roll_moment * 0.0
        force = (zero, zero, unit * self.heave_per_unit)
        moment = (roll_moment, unit * self.pitch_per_unit, zero)
        return force, moment


# ==========================================================================
# Phase-dependent balance authority
# ==========================================================================
@dataclass(frozen=True)
class PhaseAuthority:
    """How much roll moment a crew can actually apply, and when.

    A constant limit across the stroke is not a small simplification.  The
    two phases work by completely different mechanisms:

    **Drive.**  The blades are buried.  A rower who changes handle height
    loads the oar as a lever against water that pushes back, so the rigger
    carries ``(1 + inboard/outboard)`` times the handle force.  [D96]:
    "The oars can be used to force the hull flat during the drive."

    **Recovery.**  The blades are in the air.  There is nothing to push
    against, so the *only* reaction available is the oar's own inertia --
    the rower can angularly accelerate it about the oarlock, and the
    reaction appears at the rigger.  [D96] is explicit that hand-height
    adjustment "can only produce transient forces", and that the
    alternative mechanism, skimming the blades on the water, is
    unavailable to a crew boat: "crew boats require you to get the spoons
    right off the water to clear the puddles coming down from rowers
    behind you, hence this strategy is not available."

    Worse, [D96] finds the obvious correction is *destabilising*:
    "Lifting or dropping the hands during the recovery to keep the blades
    at a constant height off the water with a non-flat boat tends to make
    it even less flat.  The system has positive feedback."

    References
    ----------
    [D96] "Balance of Racing Rowing Boats", Furnivall Sculling Club, 1996
          (2013 PDF revision), hosted at
          https://eodg.atm.ox.ac.uk/user/dudhia/rowing/physics/Balance_of_Racing_Rowing_Boats_v3.pdf
    """

    drive: float                  # N m
    recovery: float               # N m
    hand_travel: float            # m, vertical hand movement assumed
    handle_force: float           # N, vertical handle force on the drive

    @property
    def ratio(self) -> float:
        """Recovery authority as a fraction of drive authority."""
        return self.recovery / self.drive if self.drive else 0.0

    def limit(self, phase: str) -> float:
        return self.drive if phase == "drive" else self.recovery

    def window(self, t, timing, ca=None, sharpness: float = 40.0):
        """Authority limit at time ``t``, smoothed for IPOPT.

        The true limit is a square wave: full authority while the blades
        are buried, almost none while they are in the air.  A square wave
        has no derivative at the catch and the finish, and the phase-locked
        mesh puts nodes exactly there, so it is smoothed with a pair of
        logistic edges.

        ``sharpness`` is in units of inverse stroke phase.  At the default
        the transition occupies about ``4/sharpness = 0.10`` of the stroke,
        roughly 0.19 s at rate 32 -- comparable to the time a blade
        actually takes to enter and leave the water, so the smoothing is
        not merely numerical convenience.  :meth:`window_error` reports how
        far the smoothed limit departs from the square wave, which is the
        bound this approximation has to be judged against.
        """
        if ca is None:
            import numpy as backend

            phase = backend.mod(t, timing.period) / timing.period

            def logistic(z):
                return 1.0 / (1.0 + backend.exp(-z))
        else:
            phase = ca.fmod(t, timing.period) / timing.period

            def logistic(z):
                return 1.0 / (1.0 + ca.exp(-z))

        drive = timing.drive_fraction
        buried = (logistic(sharpness * phase)
                  * logistic(sharpness * (drive - phase)))
        return self.recovery + (self.drive - self.recovery) * buried

    def window_error(self, timing, samples: int = 2001,
                     sharpness: float = 40.0) -> float:
        """Worst departure of the smoothed limit from the square wave.

        Reported as a fraction of the drive authority, excluding a band of
        one transition width either side of each edge -- inside that band
        the two disagree by construction, and the question is whether they
        agree everywhere else.
        """
        import numpy as np

        phase = np.linspace(0.0, 1.0, samples, endpoint=False)
        smooth = self.window(phase * timing.period, timing,
                             sharpness=sharpness)
        exact = np.where(phase < timing.drive_fraction, self.drive,
                         self.recovery)
        width = 4.0 / sharpness
        edges = ((phase < width)
                 | (np.abs(phase - timing.drive_fraction) < width)
                 | (phase > 1.0 - width))
        return float(np.abs(smooth - exact)[~edges].max() / self.drive)

    @classmethod
    def from_boat(cls, boat, handle_force: float = 150.0,
                  hand_travel: float = 0.15, gravity: float = 9.80665,
                  lean_angle: float = np.radians(2.0)):
        """Derive both limits from the rig.

        ``handle_force`` is the vertical force a rower can apply at the
        handle *while also pulling*.  It is the one physiological number
        here and it is deliberately conservative: instrumented-oar studies
        put blade forces in the 0-150 N range for the calibration of
        three-dimensional oarlock transducers, and the vertical component
        a rower can spare mid-drive is at the low end of what they produce
        horizontally.

        ``hand_travel`` is how far the hands can move vertically during the
        recovery.  Together with the recovery duration it bounds the
        angular acceleration that can be imposed on a free oar, and hence
        the reaction at the rigger.  Nothing else on the recovery is
        available to a crew boat.
        """
        rig = BalanceRig.from_boat(boat)
        recovery_time = float(boat.timing.recovery_duration)

        drive = recovery = 0.0
        for seat in boat.rig.seats:
            for lock in seat.oarlocks:
                oar = lock.oar
                arm = abs(float(lock.position[1]))
                outboard = oar.length - oar.inboard

                # -- drive: blade anchored, oar is a lever on the water --
                lever = 1.0 + oar.inboard / outboard
                drive += handle_force * lever * arm

                # -- recovery: only the oar's own inertia reacts ---------
                # Bang-bang vertical hand motion over the recovery gives a
                # handle displacement of alpha * inboard * (T/2)^2 / 2, so
                # the travel available bounds the angular acceleration.
                span = 0.5 * recovery_time
                alpha = (2.0 * hand_travel
                         / (oar.inboard * span ** 2)) if span > 0 else 0.0
                handle = alpha * oar.inertia_about_lock / oar.inboard
                # reaction at the lock: the handle force plus the inertial
                # reaction of the oar's centre of mass
                reaction = handle + oar.mass * alpha * oar.centre_of_mass_offset
                recovery += reaction * arm

        # Trunk lean acts in both phases, but it is only *decisive* on the
        # recovery, where the oars offer almost nothing.  See
        # :func:`trunk_lean_authority` for why it is not free.
        lean = trunk_lean_authority(boat, lean_angle, gravity=gravity)
        recovery += lean
        drive += lean

        return cls(drive=float(drive), recovery=float(recovery),
                   hand_travel=float(hand_travel),
                   handle_force=float(handle_force))


def static_roll_stiffness(boat, gravity: float = 9.80665,
                          angle: float = np.radians(1.0)) -> float:
    """Net static roll stiffness in N m/rad.  **Positive means unstable.**

    Buoyancy against gravity with the crew rigid in the boat, exactly the
    configuration [D96] analyses.  A racing shell carries its centre of
    gravity above its metacentre, so this comes out positive: the boat
    does not right itself, and every newton-metre of flatness is produced
    by the crew.

    This is worth computing rather than assuming, because it sets the
    timescale the crew has to work against -- see
    :func:`roll_divergence_time`.
    """
    water = boat.water
    heave = float(boat.equilibrium_heave())
    mass, position, _, _ = boat.crew_field(0.0)
    centre_z = float((mass * position[:, 2]).sum() / boat.total_mass)
    weight = boat.total_mass * gravity

    submerged = boat.mesh.submerged(
        np.array([0.0, 0.0, heave]), np.array([angle, 0.0, 0.0]),
        rho=water.density, gravity=gravity, water_level=0.0)
    centre = np.asarray(submerged.centre_of_buoyancy, dtype=float)

    rotation = np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(angle), -np.sin(angle)],
                         [0.0, np.sin(angle), np.cos(angle)]])
    lateral_buoyancy = float((rotation @ centre)[1])
    lateral_weight = float((rotation @ np.array([0.0, 0.0, centre_z]))[1])
    return weight * (lateral_buoyancy - lateral_weight) / angle


def roll_inertia(boat) -> float:
    """Roll inertia of hull plus crew about the hull ``x`` axis, kg m^2."""
    mass, position, _, _ = boat.crew_field(0.0)
    crew = float((mass * (position[:, 1] ** 2 + position[:, 2] ** 2)).sum())
    return float(boat.hull_inertia[0, 0]) + crew


def roll_divergence_time(boat, gravity: float = 9.80665) -> float:
    """e-folding time of the uncontrolled roll instability, in seconds.

    ``sqrt(I / k)`` for the positive (destabilising) stiffness.  Compare it
    against ``boat.timing.recovery_duration``: the ratio is how many
    e-foldings of roll the crew must prevent between the finish and the
    next catch, using only the oar inertia that
    :meth:`PhaseAuthority.from_boat` quantifies.
    """
    stiffness = static_roll_stiffness(boat, gravity)
    if stiffness <= 0.0:
        return float("inf")
    return float(np.sqrt(roll_inertia(boat) / stiffness))


def trunk_lean_authority(boat, lean_angle: float = np.radians(2.0),
                         seat_height: float = 0.09,
                         gravity: float = 9.80665) -> float:
    """Roll moment available from leaning the trunk, in N m.

    The second recovery actuator, and on the evidence the stronger one.
    A rower who leans their upper body laterally moves a large mass on a
    long lever: for an eight, 704 kg of crew above seat height with its
    centroid 0.21 m up, so two degrees of lean is 7.5 mm of lateral shift
    and about 51 N m -- roughly **1.6 times** what the oars can produce
    with the blades in the air.

    This is the mechanism [D96] calls body inertia, and describes as how a
    sculler actually holds a boat that has no static stability:

        "if you set the boat up at the finish and swing straight down the
        hull, your upper torso is not going anywhere very quickly and its
        inertia can be used as a reference point to sit the boat flat"

    Two caveats, both from [D96], and both reasons this is *not* simply
    added to the oar term as though it were free:

    * it is a **transient** actuator.  Shifting mass laterally cannot fix
      a persistent list -- "You cannot correct a consistently off-flat
      racing boat by permanently leaning slightly sideways to the uphill
      side, it will tend to reinforce the tilt, not reduce it."  The boat
      re-equilibrates about the new mass distribution and, being unstable,
      diverges from that equilibrium just as readily.
    * it moves the crew centre of gravity, which is the very quantity that
      makes the boat unstable in the first place, so using it aggressively
      raises the stiffness it is fighting.

    ``seat_height`` is the seat above the waterline, 9 cm for an eight
    from [D96] Table 1; segments above it are taken as the leanable mass.
    """
    mass, position, _, _ = boat.crew_field(0.0)
    datum = float(boat.equilibrium_heave()) + float(seat_height)
    upper = position[:, 2] > datum
    if not upper.any():
        return 0.0
    upper_mass = float(mass[upper].sum())
    lever = float((mass[upper] * (position[upper, 2] - datum)).sum()
                  / upper_mass)
    return float(upper_mass * gravity * lever * np.sin(lean_angle))
