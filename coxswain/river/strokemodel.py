"""Stroke-resolved planar dynamics, in CasADi.

Why this exists
---------------
:mod:`coxswain.river.trajectory` averages over the stroke.  That is wrong
for steering in a way that is measurable rather than theoretical: both
steering controls are strongest during the *drive*, and one of them -- the
crew pressure split -- has literally zero authority on the recovery,
because the blades are out of the water.  The rudder is 31% weaker there
too, since its authority goes as ``u^2`` and the hull is slowest through
the middle of the recovery.

So roughly 40% of every stroke can steer and 60% cannot.  A stroke-averaged
model spreads that authority evenly across the cycle, lets the boat steer
when it physically cannot, and therefore **overestimates how tightly it
holds a bend** -- on a course where 19% of the reach already demands more
than full rudder alone can deliver.

What is modelled, and what is not
---------------------------------
Planar: ``x, y, psi, u, v, r`` plus the crew's anaerobic reserve.  Heave,
pitch and roll are dropped.  They do not materially change the horizontal
path, and they are the expensive part -- the wetted-surface integral over a
hull mesh is the one piece of the 6-DOF model that genuinely resists
symbolic differentiation.  Everything that sets the *line* is kept.

How the stroke gets in
----------------------
Every stroke-periodic quantity is fitted as a **Fourier series in time**
and evaluated as harmonics of ``2 pi t / T``.  Three things follow, all of
which matter for an optimiser:

* no ``mod`` and no branch anywhere -- the piecewise drive/recovery split
  becomes smooth spectral content, so the dynamics are differentiable
  everywhere rather than at almost every point;
* the coefficients are fitted from the **existing, tested numpy model**
  rather than re-derived, so the two cannot silently diverge; and
* accuracy is a dial.  Eight harmonics reproduce the crew yaw inertia to
  0.01% of its swing.

The aggregates are what the planar equations actually need -- the summed
first moment of crew mass and its two time derivatives, the summed yaw
inertia, and the oar loads -- not the 96 individual segments.  Summing
first and fitting second keeps the symbolic expression small.

The equations
-------------
The generalised mass matrix is the planar restriction of the one in
:mod:`coxswain.core.rigid_body`, which is checked against Formaggia et al.
eq. (14).  With ``P = sum_k m_k r_k`` the absolute-frame first moment:

    M = [[ m_t,   0,    -P_y          ],
         [  0,   m_t,   +P_x          ],
         [ -P_y, +P_x,  I_z + sum m |r|^2 ]]

Deriving that by hand would be asking for a fourth sign error; it is
transcribed from the tested three-dimensional form instead, and
``tests/unit/test_strokemodel.py`` checks the transcription against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "StrokePeriodicFit",
    "StrokeAggregates",
    "HydroCoefficients",
    "planar_mass_matrix",
    "StrokeResolvedModel",
]


@dataclass(frozen=True)
class StrokePeriodicFit:
    """A stroke-periodic scalar, as a truncated Fourier series in time.

    Evaluated as ``a0 + sum_k a_k cos(k w t) + b_k sin(k w t)`` with
    ``w = 2 pi / period``.  Smooth and periodic by construction, so no
    modulo of the stroke phase is ever needed -- which is what lets a
    piecewise drive/recovery quantity enter an NLP at all.
    """

    cos: np.ndarray
    sin: np.ndarray
    period: float
    #: Largest absolute deviation from the samples this was fitted to, in
    #: the quantity's own units.  Recorded rather than assumed: a
    #: truncated series is an approximation and its error belongs in the
    #: object, not in a comment.
    max_error: float = 0.0
    #: That error as a fraction of the sample range.
    relative_error: float = 0.0

    @classmethod
    def fit(cls, samples, period: float, n_harmonics: int = 10):
        """Fit to uniformly spaced samples spanning exactly one period."""
        samples = np.asarray(samples, dtype=float)
        n = len(samples)
        if n < 2 * n_harmonics + 1:
            raise ValueError(
                f"need at least {2 * n_harmonics + 1} samples for "
                f"{n_harmonics} harmonics, got {n}")
        spectrum = np.fft.rfft(samples) / n
        cos = np.zeros(n_harmonics + 1)
        sin = np.zeros(n_harmonics + 1)
        cos[0] = spectrum[0].real
        keep = min(n_harmonics + 1, len(spectrum))
        for k in range(1, keep):
            cos[k] = 2.0 * spectrum[k].real
            sin[k] = -2.0 * spectrum[k].imag
        fit = cls(cos=cos, sin=sin, period=float(period))
        grid = np.arange(n) / n * period
        deviation = np.abs(np.asarray(fit(grid)) - samples)
        spread = float(np.ptp(samples))
        return cls(cos=cos, sin=sin, period=float(period),
                   max_error=float(deviation.max()),
                   relative_error=float(deviation.max() / spread)
                   if spread > 0 else 0.0)

    @classmethod
    def fit_to_tolerance(cls, samples, period: float,
                         relative_tolerance: float = 0.01,
                         max_harmonics: int = 128,
                         start: int = 8):
        """Refine the series until the truncation error meets a bound.

        A truncated Fourier series is an approximation, and for anything
        with a jump -- the oar force at the catch, and so the split moment
        with it -- the coefficients decay only as ``1/k`` and the error is
        much larger than intuition suggests.  Measured on the split moment:
        10 harmonics leaves 4.1% of the range, 32 leaves 0.96%.  Fitting a
        fixed number of harmonics and not looking is how that goes unnoticed.

        Doubles the harmonic count until the bound is met or
        ``max_harmonics`` is reached, and raises rather than silently
        returning something worse than asked for.
        """
        samples = np.asarray(samples, dtype=float)
        n_harmonics = int(start)
        best = None
        while n_harmonics <= max_harmonics:
            if len(samples) < 2 * n_harmonics + 1:
                break
            best = cls.fit(samples, period, n_harmonics)
            if best.relative_error <= relative_tolerance:
                return best
            n_harmonics *= 2
        if best is None:
            raise ValueError("not enough samples to fit any harmonics")
        raise ValueError(
            f"could not reach {relative_tolerance:.3%} with "
            f"{best.n_harmonics} harmonics: best was "
            f"{best.relative_error:.3%} ({best.max_error:.4g} absolute). "
            "Either loosen the tolerance, supply more samples, or model "
            "this quantity with an explicit gate instead of a series -- a "
            "jump discontinuity converges too slowly for a spectral fit.")

    @property
    def n_harmonics(self) -> int:
        return len(self.cos) - 1

    # -- evaluation ------------------------------------------------------
    def _terms(self, t, cos_fn, sin_fn):
        omega = 2.0 * np.pi / self.period
        value = self.cos[0]
        for k in range(1, len(self.cos)):
            if self.cos[k] == 0.0 and self.sin[k] == 0.0:
                continue
            angle = k * omega * t
            value = value + self.cos[k] * cos_fn(angle) \
                + self.sin[k] * sin_fn(angle)
        return value

    def __call__(self, t):
        """Evaluate with numpy."""
        return self._terms(np.asarray(t, dtype=float), np.cos, np.sin)

    def casadi(self, t):
        """Evaluate as a CasADi expression."""
        import casadi as ca

        return self._terms(t, ca.cos, ca.sin)

    def derivative(self) -> "StrokePeriodicFit":
        """Analytic time derivative, itself a Fourier fit.

        ``d/dt [a cos(k w t) + b sin(k w t)] = k w [b cos(k w t)
        - a sin(k w t)]``.  Exact, so the derivative never drifts from the
        quantity it came from.
        """
        omega = 2.0 * np.pi / self.period
        cos = np.zeros_like(self.cos)
        sin = np.zeros_like(self.sin)
        for k in range(1, len(self.cos)):
            cos[k] = k * omega * self.sin[k]
            sin[k] = -k * omega * self.cos[k]
        return StrokePeriodicFit(cos=cos, sin=sin, period=self.period,
                                 max_error=0.0, relative_error=0.0)

    @property
    def mean(self) -> float:
        return float(self.cos[0])


def _oar_load(boat, t: float, split: float = 0.0):
    """Total hull-frame oar load at one instant: ``(Fx, Fy, Fz, Mz)``.

    Calls exactly the functions the 6-DOF simulator calls, including the
    coxswain's side gain, so a split moment obtained by differencing this
    cannot disagree with what the full model applies.
    """
    from ..crew.oarlock import hull_load, oar_force
    from ..sim.control import Coxswain

    force = np.zeros(3)
    yaw = 0.0
    for seat in boat.rig.seats:
        for lock in seat.oarlocks:
            applied = oar_force(t, boat.timing, lock.side,
                                boat.force_profile, boat.oar_sweep)
            if split != 0.0:
                applied = applied * Coxswain.side_gain(split, lock.side)
            hand = lock.position + np.array([-0.5, 0.0, 0.2])
            load, torque = hull_load(applied, lock.position, hand,
                                     lock.oar.effective_gearing)
            force += load
            yaw += float(torque[2])
    return float(force[0]), float(force[1]), float(force[2]), yaw


@dataclass
class StrokeAggregates:
    """The stroke-periodic quantities the planar equations need.

    All are *sums over the crew*, in the hull frame, fitted from the numpy
    model.  Summing before fitting is deliberate: the dynamics never need
    the 96 segments individually, and carrying them symbolically would
    make the expression two orders of magnitude larger for no gain.
    """

    first_moment: StrokePeriodicFit       # sum m x            [kg m]
    first_moment_rate: StrokePeriodicFit  # sum m x_dot        [kg m/s]
    first_moment_accel: StrokePeriodicFit  # sum m x_ddot      [kg m/s2]
    yaw_inertia: StrokePeriodicFit        # sum m (x^2 + y^2)  [kg m2]
    thrust: StrokePeriodicFit             # hull-frame surge force  [N]
    yaw_per_split: StrokePeriodicFit      # yaw moment per unit split [N m]
    #: Side force per unit split, N.  A split is a pure couple in *surge*
    #: -- the x components cancel -- but not in sway: the oar force has a
    #: y component from the sweep angle, and scaling port up while scaling
    #: starboard down leaves ``s * Fy`` behind.  Dropping it made the model
    #: develop 0.04 m/s of sideslip where the 6-DOF develops 0.24, and
    #: since the weathervane acts on sideslip that cost 93% of the split's
    #: turning authority.
    sway_per_split: StrokePeriodicFit
    period: float
    crew_mass: float
    lateral_moment: float = 0.0           # sum m y, ~0 for a symmetric crew

    @classmethod
    def from_boat(cls, boat, n_harmonics: int = None, n_samples: int = 512,
                  relative_tolerance: float = 0.01):
        """Sample the validated numpy model over one stroke and fit."""
        from ..crew.oarlock import hull_load, oar_force

        period = boat.timing.period
        times = np.linspace(0.0, period, n_samples, endpoint=False)

        moment = np.empty(n_samples)
        rate = np.empty(n_samples)
        accel = np.empty(n_samples)
        inertia = np.empty(n_samples)
        surge = np.empty(n_samples)
        yaw_split = np.empty(n_samples)
        sway_split = np.empty(n_samples)

        for index, t in enumerate(times):
            mass, position, velocity, acceleration = boat.crew_field(t)
            moment[index] = float(np.sum(mass * position[:, 0]))
            rate[index] = float(np.sum(mass * velocity[:, 0]))
            accel[index] = float(np.sum(mass * acceleration[:, 0]))
            inertia[index] = float(np.sum(
                mass * (position[:, 0] ** 2 + position[:, 1] ** 2)))

            # Difference the real oar load between split = 1 and split = 0,
            # rather than deriving the couple by hand.  A first attempt
            # used -y*Fx, which looks right and is not: `hull_load` also
            # carries the sweep-rotated force components and the
            # hand-position term, and dropping them got the sign wrong
            # over the first third of the drive and the stroke mean wrong
            # by 1.6x.  The same rule as everywhere else in this module --
            # fit from the tested function, never re-derive it.
            neutral = _oar_load(boat, t, split=0.0)
            split_one = _oar_load(boat, t, split=1.0)
            surge[index] = neutral[0]
            yaw_split[index] = split_one[3] - neutral[3]
            sway_split[index] = split_one[1] - neutral[1]

        def fit(samples):
            if n_harmonics is not None:
                return StrokePeriodicFit.fit(samples, period, n_harmonics)
            return StrokePeriodicFit.fit_to_tolerance(
                samples, period, relative_tolerance)
        return cls(
            first_moment=fit(moment), first_moment_rate=fit(rate),
            first_moment_accel=fit(accel), yaw_inertia=fit(inertia),
            thrust=fit(surge), yaw_per_split=fit(yaw_split),
            sway_per_split=fit(sway_split),
            period=period, crew_mass=boat.crew_mass,
        )


@dataclass
class HydroCoefficients:
    """Linearised sway and yaw response, measured from the full model.

    Obtained by perturbing the 6-DOF force path -- hull resistance plus
    appendages -- rather than assumed.  Checked linear over the working
    range: doubling ``r`` or the rudder doubles the response to four
    figures.  Only the sway *force* is nonlinear, because hull drag goes as
    ``v|v|``, so it carries both a linear and a quadratic term.

    The signs matter and are worth stating.  ``yaw_from_sway`` is
    **positive**: a boat crabbing to port gets a moment that turns its bow
    to port, which reduces the sideslip.  That is the skeg weathervaning,
    and it is stabilising.

    It is also most of why an eight turns so badly.  Solving the steady
    turn with sideslip included, the weathervane moment very nearly cancels
    the rudder moment: rudder alone gives about 3.5 deg/s if sideslip is
    ignored, and about 1.1 deg/s once it is not.  A model without
    ``yaw_from_sway`` is both directionally unstable and wrong about
    steering authority by a factor of three.
    """

    sway_from_sway_linear: float      # Y per (u v)      -- appendages
    sway_from_sway_quadratic: float   # Y per (v |v|)    -- hull drag
    sway_from_yaw: float              # Y per (u r)
    sway_from_rudder: float           # Y per (u^2 delta)
    yaw_from_sway: float              # N per (u v)      -- weathervane
    yaw_from_yaw: float               # N per (u r)      -- damping
    yaw_from_rudder: float            # N per (u^2 delta)

    @classmethod
    def from_boat(cls, boat, speed: float = 5.2, sample_time: float = 0.35):
        """Perturb the full force path and read the slopes off."""
        from ..core.frames import abs_to_hull
        from ..core.state import State
        from ..sim.simulator import RowingSimulator

        simulator = RowingSimulator(boat)

        def loads(sway=0.0, yaw_rate=0.0, rudder=0.0):
            simulator.coxswain.rudder_override = lambda _t, _s: rudder
            state = State.create(velocity=(speed, sway, 0.0),
                                 omega=(0.0, 0.0, yaw_rate))
            breakdown = simulator.breakdown(sample_time, state)
            rotation = abs_to_hull(state.attitude)
            force = rotation @ (breakdown.resistance_force
                                + breakdown.appendage_force)
            moment = rotation @ (breakdown.resistance_moment
                                 + breakdown.appendage_moment)
            return float(force[1]), float(moment[2])

        base_y, base_n = loads()

        # sway: two points, because the hull term is quadratic
        y_one, n_one = loads(sway=0.30)
        y_two, _ = loads(sway=0.60)
        y_one -= base_y
        y_two -= base_y
        # solve a*u*v + b*v|v| at v = 0.3 and 0.6
        quadratic = ((y_two - 2.0 * y_one)
                     / (0.60 ** 2 - 2.0 * 0.30 ** 2))
        linear = (y_one - quadratic * 0.30 ** 2) / (speed * 0.30)

        y_rate, n_rate = loads(yaw_rate=0.05)
        y_rudder, n_rudder = loads(rudder=np.radians(8.0))
        rudder_scale = speed ** 2 * np.radians(8.0)

        return cls(
            sway_from_sway_linear=linear,
            sway_from_sway_quadratic=quadratic,
            sway_from_yaw=(y_rate - base_y) / (speed * 0.05),
            sway_from_rudder=(y_rudder - base_y) / rudder_scale,
            yaw_from_sway=(n_one - base_n) / (speed * 0.30),
            yaw_from_yaw=(n_rate - base_n) / (speed * 0.05),
            yaw_from_rudder=(n_rudder - base_n) / rudder_scale,
        )


def planar_mass_matrix(total_mass, yaw_inertia, moment_x, moment_y,
                       symbolic=None):
    """Planar restriction of :func:`~coxswain.core.rigid_body.assemble_mass_matrix`.

    Acting on ``[x_ddot, y_ddot, psi_ddot]`` in the absolute frame, with
    ``(moment_x, moment_y)`` the absolute-frame first mass moment of the
    crew about ``G_h``:

        [[ m,  0, -P_y ],
         [ 0,  m, +P_x ],
         [ -P_y, +P_x, I ]]

    Symmetric, as the three-dimensional form is.  ``symbolic`` is the
    CasADi module when building an expression, ``None`` for numpy.
    """
    if symbolic is None:
        matrix = np.zeros((3, 3))
        matrix[0, 0] = matrix[1, 1] = total_mass
        matrix[0, 2] = matrix[2, 0] = -moment_y
        matrix[1, 2] = matrix[2, 1] = moment_x
        matrix[2, 2] = yaw_inertia
        return matrix

    ca = symbolic
    return ca.blockcat([
        [total_mass, 0, -moment_y],
        [0, total_mass, moment_x],
        [-moment_y, moment_x, yaw_inertia],
    ])


class StrokeResolvedModel:
    """CasADi planar dynamics with the stroke resolved in time.

    State ``[x, y, psi, u, v, r, w]``: position and heading in the absolute
    frame, surge/sway in the hull frame, yaw rate, and remaining anaerobic
    capacity.  Controls ``[rudder, split, power]``.
    """

    n_states = 7
    n_controls = 3

    def __init__(self, boat, aggregates: StrokeAggregates = None,
                 n_harmonics: int = None, reference_speed: float = 5.2,
                 hydro: "HydroCoefficients" = None,
                 relative_tolerance: float = 0.01):
        self.boat = boat
        self.aggregates = (
            StrokeAggregates.from_boat(
                boat, n_harmonics, relative_tolerance=relative_tolerance)
            if aggregates is None else aggregates)
        self.hydro = (HydroCoefficients.from_boat(boat, reference_speed)
                      if hydro is None else hydro)
        self.reference_speed = float(reference_speed)
        self.hull_mass = float(boat.total_mass)
        self.hull_yaw_inertia = float(boat.hull_inertia[2, 2])
        self.period = self.aggregates.period

        # drag calibrated so the boat holds the reference speed on the mean
        # thrust the crew actually produces
        self.drag_coefficient = (self.aggregates.thrust.mean
                                 / reference_speed ** 2)
        self._yaw_inertia_rate = self.aggregates.yaw_inertia.derivative()
        self.turn_drag = 8000.0
        self.critical_power = 3040.0
        self.anaerobic_capacity = 176000.0

    # -- dynamics ---------------------------------------------------------
    def derivative(self, state, control, time, depth_lookup=None):
        """CasADi expression for the state derivative."""
        import casadi as ca

        x, y, psi, u, v, r, _ = (state[i] for i in range(7))
        rudder, split, power = control[0], control[1], control[2]
        agg = self.aggregates

        # crew aggregates at this instant in the stroke
        moment_hull = agg.first_moment.casadi(time)
        moment_rate = agg.first_moment_rate.casadi(time)
        moment_accel = agg.first_moment_accel.casadi(time)
        crew_inertia = agg.yaw_inertia.casadi(time)
        thrust = agg.thrust.casadi(time) * power
        yaw_from_split = agg.yaw_per_split.casadi(time) * split
        sway_from_split = (0.0 if agg.sway_per_split is None
                           else agg.sway_per_split.casadi(time) * split)

        # first moment in the absolute frame: the crew sits on the hull x
        # axis, so rotating it is a single heading rotation
        cos_psi, sin_psi = ca.cos(psi), ca.sin(psi)
        moment_x = moment_hull * cos_psi
        moment_y = moment_hull * sin_psi

        mass_matrix = planar_mass_matrix(
            self.hull_mass, self.hull_yaw_inertia + crew_inertia,
            moment_x, moment_y, symbolic=ca)

        # -- external forces, hull frame then rotated -----------------
        speed_sq = u * u + v * v
        shallow = 1.0
        if depth_lookup is not None:
            h = depth_lookup(ca.vertcat(x, y))
            shallow = 1.0 + 1.6 * ca.exp(-(h - 0.8) / 1.6)

        hydro = self.hydro
        surge_force = thrust - self.drag_coefficient * shallow * u * ca.fabs(u)
        sway_force = (hydro.sway_from_sway_linear * u * v
                      + hydro.sway_from_sway_quadratic * v * ca.fabs(v)
                      + hydro.sway_from_yaw * u * r
                      + hydro.sway_from_rudder * u * u * rudder
                      + sway_from_split)
        # yaw_from_sway is the weathervane and is what keeps the boat
        # directionally stable; without it this model spins up
        yaw_moment = (hydro.yaw_from_sway * u * v
                      + hydro.yaw_from_yaw * u * r
                      + hydro.yaw_from_rudder * u * u * rudder
                      + yaw_from_split)

        force_abs = ca.vertcat(
            surge_force * cos_psi - sway_force * sin_psi,
            surge_force * sin_psi + sway_force * cos_psi,
            yaw_moment,
        )

        # -- crew reaction: transport terms, absolute frame ------------
        # a_rel and the Coriolis/centrifugal parts of the moving mass,
        # exactly as moving_mass_reaction does in three dimensions
        accel_x = moment_accel * cos_psi
        accel_y = moment_accel * sin_psi
        coriolis_x = -2.0 * r * moment_rate * sin_psi
        coriolis_y = 2.0 * r * moment_rate * cos_psi
        centrifugal_x = -r * r * moment_x
        centrifugal_y = -r * r * moment_y

        transport_x = accel_x + coriolis_x + centrifugal_x
        transport_y = accel_y + coriolis_y + centrifugal_y

        # The crew's yaw reaction collapses to a single term.  Working
        # through sum_k m_k (r_k x transport_k)_z for a crew that slides
        # along the hull x axis:
        #
        #   relative acceleration  ->  -sum m y xddot = 0 for a symmetric
        #                              crew: sliding fore and aft makes no
        #                              yaw moment, which is obvious in
        #                              hindsight and easy to get wrong;
        #   centrifugal            ->  0, being parallel to r;
        #   Coriolis               ->  2 r sum m x xdot = r dI/dt.
        #
        # So all that survives is the crew's changing yaw inertia coupling
        # to the yaw rate.  Writing it as (sum m r) x (sum m a) instead --
        # which is the natural-looking vector form -- multiplies by mass
        # twice and is dimensionally wrong; it spun this model to 600 deg/s
        # before the units were checked.
        inertia_rate = self._yaw_inertia_rate.casadi(time)

        reaction = ca.vertcat(
            -transport_x,
            -transport_y,
            -r * inertia_rate,
        )

        acceleration = ca.solve(mass_matrix, force_abs + reaction)

        # absolute accelerations back to hull-frame surge/sway rates
        ax, ay, yaw_accel = acceleration[0], acceleration[1], acceleration[2]
        u_dot = ax * cos_psi + ay * sin_psi + r * v
        v_dot = -ax * sin_psi + ay * cos_psi - r * u

        drawn = self.critical_power * power ** 1.5
        return ca.vertcat(
            u * cos_psi - v * sin_psi,
            u * sin_psi + v * cos_psi,
            r,
            u_dot,
            v_dot,
            yaw_accel,
            -(drawn - self.critical_power),
        )

    def function(self, depth_lookup=None):
        """A callable CasADi ``Function`` for the derivative."""
        import casadi as ca

        state = ca.MX.sym("state", self.n_states)
        control = ca.MX.sym("control", self.n_controls)
        time = ca.MX.sym("time")
        return ca.Function(
            "stroke_dynamics", [state, control, time],
            [self.derivative(state, control, time, depth_lookup)],
            ["state", "control", "time"], ["derivative"])
