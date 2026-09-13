r"""The prescribed crew, made to follow the dynamic oar -- phase 4.1.

Why
---
The dynamic oar integrates its own angle, but the crew attached to it still
moved on the stroke clock.  Measured on a settled stroke at 380 W, before
this existed, the clock crew's hands sat up to 0.19 m off the dynamic handle
on the eight and 0.74 m on the single -- the single's oar running 48 degrees
behind the body mid-drive.  The arms are 0.7 m long; no arm solve closes
that from a shoulder still on the clock.

And the two halves of the model disagreed about the body.  The oar's
reflected inertia (:func:`~coxswain.crew.oardynamics.reflected_inertia`) is
the body's kinetic energy *as if it followed the oar*; the hull felt the same
body on the clock.

What
----
During the drive the body's stroke time is the prescribed time at which the
prescribed sweep had the current angle, ``tau = tau_p(phi)``.  The pose is
read from the rower's stroke table at ``tau``, so the hands are on the handle
by construction.  Velocity and acceleration follow by the chain rule::

    v = v_p(tau) s phi_dot
    a = a_p(tau) (s phi_dot)^2 + v_p(tau) (s' phi_dot^2 + s phi_ddot)

with ``s = d tau / d phi = 1 / phi_dot_p``.  The oar's inertia is built from
exactly these velocities, ``I(phi) = sum m |v_p|^2 s^2``, so the body's
kinetic energy is one number whether the hull or the oar is asked.

The regularisation
------------------
``phi_dot_p`` is zero at the catch and at the finish, so ``s`` diverges
there: along a settled stroke the body clock would run eighteen times fast in
the last 5% of the eight's drive.  ``|phi_dot_p|`` is floored at
``rate_floor`` times its peak -- the floor ``reflected_inertia`` already
uses.  Inside the floor the pose stays on the handle but the velocity is
capped, so it is no longer the exact derivative of the pose, and the power
balance between hull and oar does not close exactly there.  Nor does the
momentum: the acceleration is not the derivative of the velocity inside the
floor, because the pose runs through the table at the true clock rate while
the velocity is scaled by the capped one.  Using the true clock rate in the
acceleration was tried -- it reached 7e14 at the catch and the run overflowed.

The velocity also jumps twice a stroke -- at the finish, where the oar is
held and the body stops with it, and at the catch, where the retimed
recovery arrives moving.  An acceleration cannot carry a jump, so
:class:`~coxswain.sim.dynamic_oar.DynamicOarSimulator` hands each one to the
hull as an impulse that conserves the momentum of hull plus crew.  The first
wiring did not, and handed the hull +361 N s a stroke that the body never had.

**What is left, and why it cannot be fixed here.**  With the impulses, the
eight at 380 W still hands the hull +148 N s a stroke that the body does not
have, measured with RK4's own weights; the single +13.5.  It sits in the
first step off the catch and the last steps before the finish -- inside the
floor.  The cause is structural: the ergometer-fitted body is still moving
where the sweep's rate is zero, so a body slaved *kinematically* to the oar
cannot keep its hands on the handle and conserve momentum at once.  That
needs the hand-on-handle condition enforced by a constraint force, which is
the torque-driven chain of phase 4.3.

Recovery
--------
The oar is held at the finish until the next catch, so it cannot carry the
body back.  The recovery is retimed instead: the prescribed recovery is
stretched or squeezed linearly between the moment the dynamic drive finished
and the next catch.  It starts from the finish pose without a jump and
reaches the catch pose on time.

Positions are returned relative to the rower's own footboard, as the stroke
table stores them; the caller anchors them.
"""

from __future__ import annotations

import numpy as np

from .oardynamics import RATE_FLOOR, InertiaProfile


class FollowingCrew:
    """One rower's body slaved to its seat's oar angle through the drive."""

    def __init__(self, boat, seat: int, samples: int = 481,
                 rate_floor: float = RATE_FLOOR):
        timing, sweep = boat.timing, boat.oar_sweep
        rower = boat.crew[seat].rower
        self.period = float(timing.period)
        self.drive = float(timing.drive_fraction * timing.period)
        self.rate_floor = float(rate_floor)
        self.table = boat._stroke_table(rower)
        self.masses = np.asarray(rower.segment_masses, dtype=float)

        tau = np.linspace(0.0, self.drive, int(samples))
        angle = np.asarray(sweep(tau, timing), dtype=float)
        rate = np.asarray(sweep.rate(tau, timing), dtype=float)
        if not angle[0] > angle[-1] or np.any(np.diff(angle) > 0.0):
            raise ValueError(
                "the body can follow the oar only if the prescribed sweep "
                "runs monotonically from the catch to the finish")
        # A sweep with zero rate at its ends can repeat an angle to within
        # rounding; keep the first of any repeat so tau(phi) is a function.
        keep = np.concatenate([[True], np.diff(angle) < 0.0])
        tau, angle, rate = tau[keep], angle[keep], rate[keep]

        peak = float(np.max(np.abs(rate)))
        floored = -np.maximum(np.abs(rate), self.rate_floor * peak)
        slope = 1.0 / floored                      # d tau / d phi, < 0
        slope_rate = np.gradient(slope, angle)     # d s / d phi

        velocity = np.array([self.table.at(float(each))[1] for each in tau])
        energy = (self.masses[None, :]
                  * np.sum(velocity ** 2, axis=2)).sum(axis=1)
        inertia = energy * slope ** 2

        # Stored by increasing angle, for np.interp.
        self._angle = angle[::-1].copy()
        self._tau = tau[::-1].copy()
        self._slope = slope[::-1].copy()
        self._slope_rate = slope_rate[::-1].copy()
        self._inertia = inertia[::-1].copy()
        self.floored = (np.abs(rate) < self.rate_floor * peak)[::-1].copy()

    # -- the reduction -----------------------------------------------------
    @property
    def angle(self) -> np.ndarray:
        """The drive grid, increasing: finish to catch."""
        return self._angle

    def _at(self, values, phi) -> float:
        return float(np.interp(float(phi), self._angle, values))

    def stroke_time(self, phi) -> float:
        """The prescribed stroke time at which the sweep had ``phi``."""
        return self._at(self._tau, phi)

    def inertia(self, phi) -> float:
        """The body's inertia about the pin, without the oar's own."""
        return self._at(self._inertia, phi)

    def profile(self, oar_inertia: float) -> InertiaProfile:
        """The inertia the oar's balance must use for this body."""
        return InertiaProfile(angle=self._angle, inertia=self._inertia,
                              oar_inertia=float(oar_inertia))

    # -- the body ----------------------------------------------------------
    def drive_state(self, phi, rate, acceleration):
        """``(position, velocity, acceleration)``, each ``(12, 3)``."""
        tau = self.stroke_time(phi)
        slope = self._at(self._slope, phi)
        slope_rate = self._at(self._slope_rate, phi)
        position, velocity, accel = self.table.at(tau)
        clock = slope * float(rate)
        # The floored clock, not the true one.  The true clock rate
        # phi_dot / phi_dot_p is integrable at the ends but not integrable
        # by a fixed step: tried, it reached 7e14 at the catch and the run
        # overflowed.  The floor's price is a small momentum mismatch
        # inside it, measured in tests/test_crew_follows.py.
        return (np.array(position, dtype=float), velocity * clock,
                accel * clock ** 2
                + velocity * (slope_rate * float(rate) ** 2
                              + slope * float(acceleration)))

    def recovery_state(self, since_catch, drive_elapsed):
        """The retimed recovery, ``since_catch`` seconds into the stroke.

        ``drive_elapsed`` is how long the dynamic drive lasted.
        """
        drive_elapsed = float(drive_elapsed)
        left = max(self.period - drive_elapsed, 1e-9)
        clock = (self.period - self.drive) / left
        tau = self.drive + clock * max(float(since_catch) - drive_elapsed, 0.0)
        tau = min(tau, self.period)
        position, velocity, accel = self.table.at(tau)
        return np.array(position, dtype=float), velocity * clock, \
            accel * clock ** 2

    def hand(self, phi) -> np.ndarray:
        """The hand, relative to the footboard, with the body at ``phi``."""
        return np.asarray(self.table.hand_at(self.stroke_time(phi)),
                          dtype=float)
