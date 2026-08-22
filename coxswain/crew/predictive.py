"""Predictive stroke: solve for the rower's motion instead of prescribing it.

Every version of this model so far has been a **data-tracking** simulation
in the sense of [FDR21]: the crew's motion comes from measured keyframes,
the boat's response is computed, and any error in the kinematics is an
error in the answer.  Section 24 measured one such error -- the crew's
centre-of-mass velocity swing is about 1.5 times too large against
differential GPS -- and sections 23 and 25 established that four keyframes
per stroke cannot pin it down.

A **predictive** formulation does not have that failure mode, because the
kinematics are not an input.  The rower's motion becomes a decision
variable, constrained by what a rower can actually do, and the
centre-of-mass travel falls out as whatever is consistent with the physics
and the physiology.  The 1.5x error is not corrected so much as rendered
inexpressible.

It also asks a better question.  The existing solver optimises the *line*
for a fixed stroke; this optimises the *stroke*, and the two eventually
join up -- a crew chooses both.

What is solved for
------------------
The rower is planar and has three configuration variables:

``s``      seat position along the slide (m, positive towards the bow)
``phi``    trunk angle from vertical (rad, negative is forward lean)
``theta``  oar angle (rad, positive towards the bow)

They are not independent: the hands are on the handle, so the shoulder --
which follows from ``s`` and ``phi`` -- must be within arm's reach of the
handle, which follows from ``theta``.  That coupling is imposed as a path
constraint rather than solved analytically, which is the implicit
treatment [VDB11] recommends: it keeps the Jacobian sparse and avoids a
nested solve inside the dynamics.

Physiology is what stops the answer being nonsense.  Without limits the
optimiser will slide a rower through the footboard, so the constraints
are the real ones: the seat stays on its rail, the trunk stays within its
range, the arms cannot exceed their length, and the total mechanical power
stays within a budget the athlete can sustain.

Deliberate scope of this first version
--------------------------------------
Planar, one rower, straight line, surge only.  No steering, no roll, no
river.  The question it exists to answer is narrow and worth answering on
its own: **given only physics and physiological limits, does the optimiser
produce a recognisable rowing stroke, and what centre-of-mass travel does
it choose?**  If it independently lands near the measured segment split
and the measured velocity fluctuation, that is a strong validation of the
whole approach and a resolution of section 24.  If it does not, the
disagreement is informative in a way that a prescribed stroke cannot be.

References
----------
[VDB11] van den Bogert, Blana & Heinrich (2011), *Procedia IUTAM*
        2:297-316 -- implicit dynamics for direct collocation.
[FDR21] *Applied Sciences* 11(4):1450 -- data-tracking versus predictive
        forward-dynamics simulation in sport.
[CR06]  Cabrera, Ruina & Kleshnev (2006) -- slip-based blade model.
See ``docs/SOURCES.md`` sections 25 and 26.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["RowerLimits", "PredictiveStroke", "PredictiveResult"]


@dataclass(frozen=True)
class RowerLimits:
    """What the athlete can physically do.

    These are the constraints that make the answer a rowing stroke rather
    than an unbounded optimum.  Defaults are for the 1.88 m, 85 kg athlete
    the rest of the model uses.
    """

    #: Slide travel available, metres.  A racing seat runs on a rail of
    #: finite length; the rower cannot leave it.
    slide_range: tuple = (0.0, 0.70)
    #: Trunk angle from vertical, radians.  Negative is forward lean at the
    #: catch, positive is layback at the finish.  Kleshnev's on-water elite
    #: telemetry gives -24.5 deg and +26.3 deg, so the bound is a little
    #: wider than what elite crews use.
    trunk_range: tuple = (np.radians(-40.0), np.radians(35.0))
    #: Oar angle from the perpendicular, radians.
    oar_range: tuple = (np.radians(-40.0), np.radians(60.0))
    #: Arm length, shoulder to handle, metres.
    arm_length: float = 0.70
    #: How far the shoulder girdle can protract, metres.  Real rowers reach
    #: past their skeletal arm length at the catch.
    protraction: float = 0.06
    #: Peak joint accelerations, as a bound on the controls.
    max_slide_acceleration: float = 30.0      # m/s^2
    max_trunk_acceleration: float = 60.0      # rad/s^2
    max_oar_acceleration: float = 60.0        # rad/s^2
    #: Sustainable mechanical power, watts.  A trained single sculler holds
    #: roughly 300 W over a 2 km race; masters crews rather less.
    power_budget: float = 300.0
    #: How that budget is shared between the segments.  Kleshnev measures
    #: **legs 43%, trunk 33%, arms 24%** of power production in elite
    #: rowers, and the split matters: without it the optimiser sees moving
    #: 85 kg of body up the slide as simply expensive and rows with trunk
    #: and arms alone, choosing 0.22 m of slide travel where crews use
    #: 0.60-0.70.  The legs are strong, and the model has to know it.
    power_share: tuple = (0.43, 0.33, 0.24)

    def validate(self) -> None:
        if self.slide_range[1] <= self.slide_range[0]:
            raise ValueError("slide_range must be increasing")
        if self.trunk_range[1] <= self.trunk_range[0]:
            raise ValueError("trunk_range must be increasing")
        if self.arm_length <= 0.0:
            raise ValueError("arm_length must be positive")
        if self.power_budget <= 0.0:
            raise ValueError("power_budget must be positive")


@dataclass
class PredictiveResult:
    """A solved stroke."""

    time: np.ndarray
    slide: np.ndarray
    trunk: np.ndarray
    oar: np.ndarray
    boat_speed: np.ndarray
    centre_of_mass: np.ndarray
    power: float
    distance: float
    success: bool
    stats: dict = field(default_factory=dict)

    @property
    def mean_speed(self) -> float:
        return float(self.boat_speed.mean())

    @property
    def slide_travel(self) -> float:
        return float(self.slide.max() - self.slide.min())

    @property
    def com_travel(self) -> float:
        return float(self.centre_of_mass.max() - self.centre_of_mass.min())

    @property
    def com_velocity_swing(self) -> float:
        rate = np.gradient(self.centre_of_mass, self.time)
        return float(rate.max() - rate.min())

    @property
    def intracycle_variation(self) -> float:
        """Peak-to-peak boat speed as a fraction of the mean.

        The quantity section 24 measured at 37.3% on differential GPS and
        the prescribed-kinematics model puts at 54-58%.
        """
        return float((self.boat_speed.max() - self.boat_speed.min())
                     / self.boat_speed.mean())


class PredictiveStroke:
    """Optimise the rower's motion, not just the boat's response.

    States are ``(s, sdot, phi, phidot, theta, thetadot, v_boat)`` and the
    controls are the three configuration accelerations.  One stroke is
    solved with the crew states periodic, so the answer is a *cycle* the
    rower can repeat rather than a one-off transient.
    """

    def __init__(self, boat, limits: Optional[RowerLimits] = None,
                 blade=None, n_intervals: int = 40):
        from ..crew.oarlock import BladeModel

        self.boat = boat
        self.limits = limits or RowerLimits()
        self.limits.validate()
        self.blade = blade or BladeModel.sculling()
        self.n_intervals = int(n_intervals)

        lock = boat.rig.seats[0].oarlocks[0]
        self.oar = lock.oar
        self.inboard = float(self.oar.inboard)
        self.outboard = float(self.oar.length) - self.inboard
        self.span = abs(float(lock.position[1]))

        rower = boat.crew[0].rower
        self.masses = np.asarray(rower.segment_masses, dtype=float)
        self.total_crew = float(self.masses.sum())
        self.trunk_length = float(rower.trunk_length)
        # Lump the crew into three masses whose motion the three
        # configuration variables determine: legs ride with the seat, the
        # trunk swings about the hip, the arms track the handle.  de Leva's
        # segment masses set the split; section 25's amplitude check is
        # what this formulation is meant to get right without being told.
        self.leg_mass = float(self.masses[[8, 9, 10, 11]].sum())
        self.trunk_mass = float(self.masses[[0, 1, 2, 3]].sum())
        self.arm_mass = float(self.masses[[4, 5, 6, 7]].sum())
        self.trunk_com = 0.45 * self.trunk_length
        self.hull_mass = float(boat.hull_mass)

    # -- geometry ---------------------------------------------------------
    #
    # The hands are on the handle.  That is not a soft constraint to be
    # satisfied approximately -- it is what makes the trunk matter at all.
    # An earlier version made the oar angle a free variable with reach as
    # an inequality, and the optimiser immediately found the degenerate
    # answer: leave the trunk still (0.2 deg of swing) and sweep the oar
    # on its own.  A rower cannot do that.  So the configuration is
    # ``(s, phi, a)`` -- slide, trunk, arm extension -- and the oar angle
    # *follows*.
    def shoulder_x(self, s, phi, ca=None):
        """Shoulder position, from the seat and the trunk lean."""
        sin = ca.sin if ca is not None else np.sin
        return s + self.trunk_length * sin(phi)

    def hand_x(self, s, phi, a, ca=None):
        """Hand position: the shoulder, reaching ``a`` towards the stern."""
        return self.shoulder_x(s, phi, ca) - a

    def oar_angle(self, s, phi, a, ca=None):
        """Oar angle implied by where the hands are.

        The handle sits ``inboard`` from the pin, so
        ``x_handle = -inboard cos(theta)``; inverting gives theta.  The
        argument is clipped just inside +-1 because the optimiser will
        otherwise walk it to the singularity where the oar is square to
        the boat and the derivative blows up.
        """
        ratio = -self.hand_x(s, phi, a, ca) / self.inboard
        if ca is None:
            return float(np.arccos(np.clip(ratio, -0.999, 0.999)))
        return ca.acos(ca.fmax(ca.fmin(ratio, 0.999), -0.999))

    def crew_com(self, s, phi, a, ca=None):
        """Longitudinal centre of mass of the rower.

        The output section 24 is about.  Nothing here prescribes it: it
        follows from wherever the optimiser puts the configuration.
        """
        sin = ca.sin if ca is not None else np.sin
        legs = s
        trunk = s + self.trunk_com * sin(phi)
        arms = self.hand_x(s, phi, a, ca)
        return ((self.leg_mass * legs + self.trunk_mass * trunk
                 + self.arm_mass * arms) / self.total_crew)

    # -- the optimisation --------------------------------------------------
    def solve(self, period: float, initial_speed: float = 4.0,
              max_iter: int = 900, print_level: int = 0,
              effort_weight: float = 1e-4):
        """Solve one periodic stroke that covers the most ground.

        Configuration is ``(s, phi, a)`` -- slide, trunk lean, arm
        extension -- and the oar angle *follows* from where those put the
        hands.  The objective is distance per stroke; the athlete's limits
        and a power budget are what make the answer a rowing stroke rather
        than an unbounded optimum.

        This is the structure of [MSD13], which represents rower, boat and
        oars as rigid links and lets the optimisation determine the
        movement and the forces together, rather than prescribing either.
        """
        import casadi as ca

        from ..hydro.resistance import hull_resistance

        n = self.n_intervals
        dt = period / n
        limits = self.limits

        submerged = self.boat.mesh.submerged(
            np.array([0.0, 0.0, self.boat.equilibrium_heave()]),
            np.zeros(3), rho=self.boat.water.density, gravity=9.80665,
            water_level=0.0)
        probe = 4.0
        force, _ = hull_resistance(np.array([probe, 0.0, 0.0]), submerged,
                                   mean_wetted_length=self.boat.length,
                                   water=self.boat.water,
                                   coefficients=self.boat.resistance)
        drag_k = abs(float(force[0])) / probe ** 2
        total_mass = self.hull_mass + self.total_crew

        opti = ca.Opti()
        S = opti.variable(n)
        P = opti.variable(n)
        A = opti.variable(n)
        V = opti.variable(n)

        theta = [self.oar_angle(S[k], P[k], A[k], ca) for k in range(n)]
        com = [self.crew_com(S[k], P[k], A[k], ca) for k in range(n)]

        def d1(seq, k):
            return (seq[(k + 1) % n] - seq[(k - 1) % n]) / (2.0 * dt)

        def d2(seq, k):
            return (seq[(k + 1) % n] - 2.0 * seq[k]
                    + seq[(k - 1) % n]) / (dt * dt)

        leg_terms, trunk_terms, arm_terms = [], [], []
        for k in range(n):
            theta_dot = d1(theta, k)
            com_accel = d2(com, k)

            # Blade, slip-based [CR06].  In the water only while the oar
            # sweeps sternward; the gate is smoothed for the optimiser.
            slip = self.outboard * theta_dot + V[k] * ca.cos(theta[k])
            gate = 0.5 * (1.0 - ca.tanh(2.0 * theta_dot))
            blade_force = -gate * self.blade.c2 * slip * ca.fabs(slip)
            pin = blade_force * (1.0 + self.inboard / self.outboard)
            thrust = pin * ca.cos(theta[k])

            drag = -drag_k * V[k] * ca.fabs(V[k])
            reaction = -self.total_crew * com_accel
            # Trapezoidal, not a central difference.  A central difference
            # on a periodic grid leaves the odd and even nodes coupled only
            # through the dynamics, so the optimiser can put a sawtooth in
            # the speed at no cost -- the first attempt returned 203% of
            # intracycle variation that way.
            opti.subject_to((V[(k + 1) % n] - V[k]) * total_mass
                            == dt * (thrust + drag + reaction))

            # Work done on the water.  Omitting this -- as the first
            # version did -- makes thrust free, and the optimiser answered
            # with a 6.9 m/s single scull drawing 112 W and a trunk that
            # never moved.
            # Power, smoothly.
            #
            # The first version used |force x velocity| directly.  That is
            # the right physics but the wrong numerics: |a v| has a corner
            # wherever either factor crosses zero, which for a periodic
            # stroke is four times a cycle per segment, and IPOPT stalled
            # on infeasible points rather than converging.
            #
            # ``sqrt(x^2 + eps)`` is the standard smooth absolute value and
            # is the licensed kind of approximation -- a bounded smoothing
            # for solver compatibility, with the bound set by ``eps`` and
            # reported.  At eps = 1e-4 W^2 the error is under 0.01 W, which
            # is five orders below the 300 W budget.
            def smooth_abs(value, eps=1e-4):
                return ca.sqrt(value * value + eps)

            # Handle work, attributed to the segments that produce it.
            #
            # ``hand_x = s + L sin(phi) - a``, so the handle velocity is
            # the *sum* of three segment contributions:
            #
            #     v_hand = s_dot + L cos(phi) phi_dot - a_dot
            #
            # That additive split is precisely what Kleshnev measures when
            # he reports legs 43% / trunk 33% / arms 24%: shares of the
            # work delivered through the handle.
            #
            # The previous version charged each segment only for shifting
            # its own mass and then applied Kleshnev's shares to *that*.
            # It is a category error, and it starves the legs -- the
            # optimiser was billed for the largest segment mass in the
            # body and credited with none of the work that mass does on
            # the handle.  Hence 0.22-0.37 m of slide where crews use
            # 0.60-0.70.
            v_legs = d1(S, k)
            v_trunk = self.trunk_length * ca.cos(P[k]) * d1(P, k)
            v_arms = -d1(A, k)

            # Total work through the handle, taken from the oar so it
            # stays consistent with the blade model rather than being
            # recomputed from the longitudinal component alone.
            handle_work = smooth_abs(blade_force * self.outboard
                                     * theta_dot)

            # Split it by each segment's share of handle speed.  Weights
            # are smoothed absolute values, so the denominator is bounded
            # away from zero at the four instants per cycle where the
            # handle reverses, and the three shares sum to the total by
            # construction -- the attribution cannot invent or destroy
            # power.
            w_legs = smooth_abs(v_legs)
            w_trunk = smooth_abs(v_trunk)
            w_arms = smooth_abs(v_arms)
            w_total = w_legs + w_trunk + w_arms

            # Each segment also pays to accelerate its own mass.  Small
            # against the handle work, but it is what makes a violent
            # recovery cost something.
            leg_terms.append(handle_work * w_legs / w_total
                             + smooth_abs(self.leg_mass
                                          * d2(S, k) * d1(S, k)))
            trunk_terms.append(handle_work * w_trunk / w_total
                               + smooth_abs(self.trunk_mass
                                            * self.trunk_com ** 2
                                            * d2(P, k) * d1(P, k)))
            arm_terms.append(handle_work * w_arms / w_total
                             + smooth_abs(self.arm_mass
                                          * d2(A, k) * d1(A, k)))

        reach_max = limits.arm_length + limits.protraction
        for k in range(n):
            opti.subject_to(opti.bounded(limits.slide_range[0], S[k],
                                         limits.slide_range[1]))
            opti.subject_to(opti.bounded(limits.trunk_range[0], P[k],
                                         limits.trunk_range[1]))
            opti.subject_to(opti.bounded(0.25, A[k], reach_max))
            opti.subject_to(V[k] >= 0.5)
            opti.subject_to(opti.bounded(-0.95 * self.inboard,
                                         self.hand_x(S[k], P[k], A[k], ca),
                                         0.95 * self.inboard))
            opti.subject_to(opti.bounded(-limits.max_slide_acceleration,
                                         d2(S, k),
                                         limits.max_slide_acceleration))
            opti.subject_to(opti.bounded(-limits.max_trunk_acceleration,
                                         d2(P, k),
                                         limits.max_trunk_acceleration))
            opti.subject_to(opti.bounded(-limits.max_slide_acceleration,
                                         d2(A, k),
                                         limits.max_slide_acceleration))

        # Per-segment budgets, not one lump.  The handle work is now
        # distributed across the three segment terms above, so it must
        # NOT be added again here or every watt is counted twice.
        legs_power = sum(leg_terms) / n
        trunk_power = sum(trunk_terms) / n
        arms_power = sum(arm_terms) / n
        mean_power = legs_power + trunk_power + arms_power
        share = limits.power_share
        # Hard per-segment budgets.  Softening them into a penalty was
        # tried and is worse: the optimiser then buys speed by exceeding
        # the *total* budget ninefold.  The shares stay as constraints.
        budget = limits.power_budget
        opti.subject_to(legs_power / budget <= share[0])
        opti.subject_to(trunk_power / budget <= share[1])
        opti.subject_to(arms_power / budget <= share[2])
        opti.subject_to(mean_power / limits.power_budget <= 1.0)
        overflow = 0

        mean_speed = ca.sum1(V) / n
        # Scale the objective and the power constraint.  The geometry is
        # O(1) in metres and radians while the power terms are O(100) W, and
        # an unscaled objective mixing them is the same conditioning
        # mistake that cost the trajectory solver a factor of 25 in
        # iterations before it was fixed.  IPOPT convergence tolerances are
        # absolute, so the scale decides what "converged" means.
        speed_scale = max(float(initial_speed), 1.0)
        opti.minimize(-mean_speed / speed_scale
                      + 1e-4 * overflow / limits.power_budget ** 2
                      + effort_weight * (ca.sumsqr(S[1:] - S[:-1])
                                         + ca.sumsqr(P[1:] - P[:-1])
                                         + ca.sumsqr(A[1:] - A[:-1])))

        grid = np.linspace(0.0, 1.0, n, endpoint=False)
        opti.set_initial(S, 0.35 - 0.28 * np.cos(2.0 * np.pi * grid))
        opti.set_initial(P, np.radians(-8.0 - 22.0
                                       * np.cos(2.0 * np.pi * grid)))
        opti.set_initial(A, 0.55 + 0.10 * np.cos(2.0 * np.pi * grid))
        opti.set_initial(V, initial_speed)

        opti.solver("ipopt", {"print_time": False},
                    {"max_iter": max_iter, "print_level": print_level,
                     "sb": "yes", "tol": 1e-6, "acceptable_tol": 1e-4,
                     "mu_strategy": "adaptive",
                     "hessian_approximation": "limited-memory"})
        try:
            sol = opti.solve()
            ok = True
            status = 'ok'
        except RuntimeError:
            sol = opti.debug
            # IPOPT distinguishes a genuine failure from a point it is
            # satisfied with at a looser tolerance.  Collapsing both to
            # "not converged" hides a perfectly usable answer, which it did
            # here for several iterations of this model.
            status = str(opti.stats().get('return_status', 'unknown'))
            ok = status in ('Solve_Succeeded', 'Solved_To_Acceptable_Level')

        def wrap(x):
            v = np.array(sol.value(x)).ravel()
            return np.concatenate([v, v[:1]])

        times = np.arange(n + 1) * dt
        slide, trunk, arm = wrap(S), wrap(P), wrap(A)
        speed = wrap(V)
        oar = np.array([self.oar_angle(a, b, c)
                        for a, b, c in zip(slide, trunk, arm)])
        centre = np.array([self.crew_com(a, b, c)
                           for a, b, c in zip(slide, trunk, arm)])
        return PredictiveResult(
            time=times, slide=slide, trunk=trunk, oar=oar,
            boat_speed=speed, centre_of_mass=centre,
            power=float(sol.value(mean_power)),
            distance=float(sol.value(mean_speed)) * period, success=ok,
            stats={"return_status": status})
