"""The full six-degree-of-freedom model, symbolically, stroke-resolved.

This carries the whole thing: surge, sway, heave, roll, pitch and yaw, the
crew as a moving mass field with its full inertia tensor, hydrostatics from
the real wetted-surface integral, nonlinear resistance, and the oar loads
with the rig's own asymmetry.  Nothing is averaged over the stroke and no
degree of freedom is dropped.

The only approximation is a **bounded smoothing** where IPOPT needs one,
and each is measured rather than assumed:

``HullSurrogate``
    The wetted-surface integral clips a triangulated hull against the
    waterline, which is a branch per triangle and has no derivative.  It
    depends on exactly three variables -- heave, pitch, roll -- so it is
    evaluated exactly on a grid and interpolated.  Worst off-node error
    1.2% of range on the transverse centre of buoyancy, which is 4 N m of
    roll moment, or 0.1% of what the crew can apply.

``StrokePeriodicFit``
    Crew aggregates and oar loads become Fourier series in time, refined
    until a relative error bound is met rather than truncated at a fixed
    order.  This is what removes the ``mod`` and the drive/recovery branch.

``tanh`` in place of ``sign``
    Exact away from zero at the scale a shell operates at.

State
-----
Thirteen: position ``(x, y, z)``, attitude ``(roll, pitch, yaw)``,
velocity and angular velocity in the **absolute** frame, and the crew's
remaining anaerobic capacity.  The absolute frame is not a preference --
it is what :mod:`coxswain.core.rigid_body` uses, and that module is
checked against Formaggia et al. eq. (14).  Writing this one in the hull
frame would mean re-deriving the transport terms by hand, which is how
this project acquired its first three sign errors.

Controls
--------
Three: rudder angle, port/starboard pressure split, and power fraction.
The crew's roll balancing is not a control but a reflex, modelled as a
saturated PD couple about the hull ``x`` axis -- crews balance without
being told to, and the bare hull is roll-unstable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strokemodel import StrokePeriodicFit, _oar_load

__all__ = ["CrewTensorFit", "SixDofModel"]

STATE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw",
               "vx", "vy", "vz", "wx", "wy", "wz", "anaerobic")


@dataclass
class CrewTensorFit:
    """The crew's mass distribution over a stroke, as Fourier series.

    The dynamics need the crew only through summed quantities: the first
    mass moment and the inertia tensor about ``G_h``, plus the first two
    time derivatives of the moment for the reaction terms.  Ninety-six
    segments are summed *before* fitting, which keeps the symbolic
    expression small without approximating anything -- the sum is exact and
    only its time dependence is fitted.

    The chain is planar in the hull frame, so ``sum m y`` vanishes for a
    symmetric crew and the tensor has no ``xy`` or ``yz`` coupling.  Those
    are carried anyway rather than assumed zero, because a per-side split
    of the crew motion would break the symmetry and the model should not
    have to be rewritten when it does.
    """

    moment_x: StrokePeriodicFit
    moment_y: StrokePeriodicFit
    moment_z: StrokePeriodicFit
    rate_x: StrokePeriodicFit
    rate_y: StrokePeriodicFit
    rate_z: StrokePeriodicFit
    accel_x: StrokePeriodicFit
    accel_y: StrokePeriodicFit
    accel_z: StrokePeriodicFit
    inertia_xx: StrokePeriodicFit
    inertia_yy: StrokePeriodicFit
    inertia_zz: StrokePeriodicFit
    inertia_xy: StrokePeriodicFit
    inertia_xz: StrokePeriodicFit
    inertia_yz: StrokePeriodicFit
    #: ``sum_k m_k (r_k x a_rel_k)`` in the hull frame, N m.  This does NOT
    #: vanish and is not ``(sum m r) x (sum m a)``.  For yaw it cancels for
    #: a symmetric crew, which is what made it easy to miss -- but the crew
    #: sits *above* the centre of mass, so a fore-and-aft acceleration at
    #: height z makes a real PITCH moment.  Leaving it out overstated the
    #: pitch swing by 54%.
    cross_accel_x: StrokePeriodicFit
    cross_accel_y: StrokePeriodicFit
    cross_accel_z: StrokePeriodicFit
    period: float
    total_mass: float

    @classmethod
    def from_boat(cls, boat, n_samples: int = 512,
                  relative_tolerance: float = 0.01):
        period = boat.timing.period
        times = np.linspace(0.0, period, n_samples, endpoint=False)

        moment = np.zeros((3, n_samples))
        rate = np.zeros((3, n_samples))
        accel = np.zeros((3, n_samples))
        inertia = np.zeros((6, n_samples))   # xx yy zz xy xz yz
        cross_accel = np.zeros((3, n_samples))

        for index, t in enumerate(times):
            mass, position, velocity, acceleration = boat.crew_field(t)
            moment[:, index] = (mass[:, None] * position).sum(axis=0)
            rate[:, index] = (mass[:, None] * velocity).sum(axis=0)
            accel[:, index] = (mass[:, None] * acceleration).sum(axis=0)

            squared = np.einsum("k,ki,ki->", mass, position, position)
            outer = np.einsum("k,ki,kj->ij", mass, position, position)
            tensor = squared * np.eye(3) - outer
            inertia[:, index] = (tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                 tensor[0, 1], tensor[0, 2], tensor[1, 2])
            cross_accel[:, index] = (
                mass[:, None] * np.cross(position, acceleration)).sum(axis=0)

        def fit(samples):
            spread = float(np.ptp(samples))
            if spread < 1e-12:                 # genuinely constant
                return StrokePeriodicFit.fit(samples, period, 1)
            return StrokePeriodicFit.fit_to_tolerance(
                samples, period, relative_tolerance)

        return cls(
            moment_x=fit(moment[0]), moment_y=fit(moment[1]),
            moment_z=fit(moment[2]),
            rate_x=fit(rate[0]), rate_y=fit(rate[1]), rate_z=fit(rate[2]),
            accel_x=fit(accel[0]), accel_y=fit(accel[1]),
            accel_z=fit(accel[2]),
            inertia_xx=fit(inertia[0]), inertia_yy=fit(inertia[1]),
            inertia_zz=fit(inertia[2]), inertia_xy=fit(inertia[3]),
            inertia_xz=fit(inertia[4]), inertia_yz=fit(inertia[5]),
            cross_accel_x=fit(cross_accel[0]),
            cross_accel_y=fit(cross_accel[1]),
            cross_accel_z=fit(cross_accel[2]),
            period=period, total_mass=float(boat.crew_mass),
        )

    def moment_hull(self, t, symbolic):
        ca = symbolic
        return ca.vertcat(self.moment_x.casadi(t), self.moment_y.casadi(t),
                          self.moment_z.casadi(t))

    def rate_hull(self, t, symbolic):
        ca = symbolic
        return ca.vertcat(self.rate_x.casadi(t), self.rate_y.casadi(t),
                          self.rate_z.casadi(t))

    def accel_hull(self, t, symbolic):
        ca = symbolic
        return ca.vertcat(self.accel_x.casadi(t), self.accel_y.casadi(t),
                          self.accel_z.casadi(t))

    def cross_accel_hull(self, t, symbolic):
        ca = symbolic
        return ca.vertcat(self.cross_accel_x.casadi(t),
                          self.cross_accel_y.casadi(t),
                          self.cross_accel_z.casadi(t))

    def inertia_hull(self, t, symbolic):
        ca = symbolic
        xx, yy, zz = (self.inertia_xx.casadi(t), self.inertia_yy.casadi(t),
                      self.inertia_zz.casadi(t))
        xy, xz, yz = (self.inertia_xy.casadi(t), self.inertia_xz.casadi(t),
                      self.inertia_yz.casadi(t))
        return ca.blockcat([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


@dataclass
class OarFit:
    """Oar force and moment over a stroke, neutral and per unit split.

    Both are taken from :func:`_oar_load`, which calls the same functions
    the 6-DOF simulator calls, so they cannot disagree with it.  The
    neutral terms are not zero for a sweep rig: port and starboard
    oarlocks sit at different stations, so their moments do not cancel.
    """

    force: tuple          # (Fx, Fy, Fz) neutral
    moment: tuple         # (Mx, My, Mz) neutral
    force_split: tuple    # per unit split
    moment_split: tuple
    #: Oar angle and its rate over the stroke.  Carried so the blade
    #: efficiency can be evaluated symbolically: it depends on the slip,
    #: which needs the oar's own motion *and* the boat's speed, so it is
    #: not a function of time alone and cannot be folded into the fits.
    angle: StrokePeriodicFit
    angle_rate: StrokePeriodicFit
    period: float
    blade_efficiency: float = 0.78

    @classmethod
    def from_boat(cls, boat, n_samples: int = 512,
                  relative_tolerance: float = 0.01):
        from ..crew.oarlock import hull_load, oar_force
        from ..sim.control import Coxswain
        from ..sim.simulator import RowingSimulator

        simulator = RowingSimulator(boat)
        period = boat.timing.period
        times = np.linspace(0.0, period, n_samples, endpoint=False)

        neutral = np.zeros((6, n_samples))
        split = np.zeros((6, n_samples))

        for index, t in enumerate(times):
            hands = simulator.hand_positions(t)
            for label, amount, target in (("n", 0.0, neutral),
                                          ("s", 1.0, split)):
                force = np.zeros(3)
                moment = np.zeros(3)
                for seat_index, seat in enumerate(boat.rig.seats):
                    hand = hands[seat_index]
                    for lock in seat.oarlocks:
                        applied = oar_force(t, boat.timing, lock.side,
                                            boat.force_profile,
                                            boat.oar_sweep)
                        if amount:
                            applied = applied * Coxswain.side_gain(
                                amount, lock.side)
                        load, torque = hull_load(
                            applied, lock.position, hand,
                            lock.oar.effective_gearing)
                        force += load
                        moment += torque
                target[0:3, index] = force
                target[3:6, index] = moment
        split -= neutral                      # per unit split
        angle = np.array([float(boat.oar_sweep(t, boat.timing))
                          for t in times])
        rate = np.array([float(boat.oar_sweep.rate(t, boat.timing))
                         for t in times])

        def fit(samples):
            spread = float(np.ptp(samples))
            if spread < 1e-12:
                return StrokePeriodicFit.fit(samples, period, 1)
            return StrokePeriodicFit.fit_to_tolerance(
                samples, period, relative_tolerance)

        return cls(
            force=tuple(fit(neutral[i]) for i in range(3)),
            moment=tuple(fit(neutral[i + 3]) for i in range(3)),
            force_split=tuple(fit(split[i]) for i in range(3)),
            moment_split=tuple(fit(split[i + 3]) for i in range(3)),
            angle=fit(angle), angle_rate=fit(rate),
            period=period,
            blade_efficiency=float(
                boat.rig.seats[0].oarlocks[0].oar.blade_efficiency),
        )


class SixDofModel:
    """The full model, as a CasADi function of state, control and time."""

    n_states = 13
    n_controls = 3

    def __init__(self, boat, surrogate=None, crew=None, oars=None,
                 relative_tolerance: float = 0.01, gravity: float = 9.80665,
                 water_level: float = 0.0, blade=None):
        from ..crew.balance import BalanceRig
        from .hullsurrogate import HullSurrogate

        self.boat = boat
        self.gravity = float(gravity)
        self.water_level = float(water_level)
        self.surrogate = (HullSurrogate.from_boat(boat) if surrogate is None
                          else surrogate)
        self.crew = (CrewTensorFit.from_boat(boat, relative_tolerance=relative_tolerance)
                     if crew is None else crew)
        self.oars = (OarFit.from_boat(boat, relative_tolerance=relative_tolerance)
                     if oars is None else oars)

        self.blade = blade
        self.hull_mass = float(boat.hull_mass) + float(boat.coxswain_mass)
        self.total_mass = float(boat.total_mass)
        self.hull_inertia = np.asarray(boat.hull_inertia, dtype=float)

        # the crew's balance reflex, matching sim.control.BalanceController
        self.balance_rig = BalanceRig.from_boat(boat)
        self.balance_stiffness = 6000.0
        self.balance_damping = 2000.0
        self.balance_limit = 4000.0
        self.critical_power = 3040.0
        self.anaerobic_capacity = 176000.0

    # -- dynamics ---------------------------------------------------------
    def derivative(self, state, control, time, depth=None):
        import casadi as ca

        from ..core import frames
        from . import hydro_casadi

        position = state[0:3]
        attitude = state[3:6]
        velocity = state[6:9]
        omega = state[9:12]
        rudder, split, power = control[0], control[1], control[2]

        roll, pitch, yaw = attitude[0], attitude[1], attitude[2]
        rotation = _rotation(roll, pitch, yaw, ca)

        # -- crew, hull frame then rotated -----------------------------
        moment_hull = self.crew.moment_hull(time, ca)
        rate_hull = self.crew.rate_hull(time, ca)
        accel_hull = self.crew.accel_hull(time, ca)
        crew_inertia_hull = self.crew.inertia_hull(time, ca)

        moment_abs = rotation @ moment_hull
        rate_abs = rotation @ rate_hull
        accel_abs = rotation @ accel_hull
        crew_inertia_abs = rotation @ crew_inertia_hull @ rotation.T

        hull_inertia_abs = rotation @ ca.DM(self.hull_inertia) @ rotation.T
        inertia_abs = hull_inertia_abs + crew_inertia_abs

        # -- generalised mass matrix, planar-free 6x6 ------------------
        coupling = _skew(moment_abs, ca)
        mass_matrix = ca.blockcat([
            [self.total_mass * ca.DM.eye(3), -coupling],
            [coupling, inertia_abs],
        ])

        # -- crew reaction: transport terms ----------------------------
        coriolis = 2.0 * ca.cross(omega, rate_abs)
        centrifugal = ca.cross(omega, ca.cross(omega, moment_abs))
        transport = accel_abs + coriolis + centrifugal
        crew_force = -transport
        crew_moment = -_crew_reaction_moment(self.crew, rotation, omega,
                                             time, ca)

        # -- hydrostatics from the exact wetted surface ----------------
        heave = position[2] - self.water_level
        hull = self.surrogate.casadi(heave, pitch, roll)
        buoyancy = self.boat.water.density * self.gravity * hull["volume"]
        centre = rotation @ ca.vertcat(hull["buoyancy_x"],
                                       hull["buoyancy_y"],
                                       hull["buoyancy_z"])
        buoyancy_force = ca.vertcat(0.0, 0.0, buoyancy)
        buoyancy_moment = ca.cross(centre, buoyancy_force)

        weight = ca.vertcat(0.0, 0.0, -self.total_mass * self.gravity)
        weight_moment = ca.cross(moment_abs, ca.vertcat(
            0.0, 0.0, -self.gravity))

        # -- resistance and appendages, hull frame ---------------------
        velocity_hull = rotation.T @ velocity
        omega_hull = rotation.T @ omega
        u, v, w = velocity_hull[0], velocity_hull[1], velocity_hull[2]

        water = self.boat.water
        coefficients = self.boat.resistance
        resistance = hydro_casadi.hull_resistance(
            u, v, w,
            wetted_area=hull["wetted_area"],
            transverse_area=hull["transverse_area"],
            plan_area=hull["plan_area"],
            lateral_area=hull["lateral_area"],
            mean_wetted_length=self.boat.length,
            depth=depth,
            density=water.density,
            kinematic_viscosity=water.kinematic_viscosity,
            shape=coefficients.shape, wave=coefficients.wave,
            friction_zero=coefficients.friction_zero,
            form_factor=coefficients.form_factor,
            cross_flow_lateral=coefficients.cross_flow_lateral,
            cross_flow_vertical=coefficients.cross_flow_vertical,
        )
        appendage_force, appendage_moment = hydro_casadi.appendage_loads(
            self.boat.appendages, u, v, omega_hull[2], rudder, water.density)

        # -- oars, including the sweep rig's own asymmetry -------------
        oar_force_hull = ca.vertcat(*[
            f.casadi(time) + s.casadi(time) * split
            for f, s in zip(self.oars.force, self.oars.force_split)])
        oar_moment_hull = ca.vertcat(*[
            m.casadi(time) + s.casadi(time) * split
            for m, s in zip(self.oars.moment, self.oars.moment_split)])
        # The blade efficiency is not a constant.  The oar loads above
        # carry the rig's fixed `blade_efficiency`, so rescale to the
        # instantaneous value from the slip model: the blade only pushes
        # what it does not let past, and that depends on the oar's own
        # sweep rate, the boat's speed, and the water around the blade.
        # Measured over a drive it runs 0.40 to 0.89 against the fixed
        # 0.78, so treating it as constant is not a small approximation.
        blade_scale = 1.0
        if self.blade is not None:
            blade_scale = _blade_efficiency(
                self.blade, self.oars, time, u, depth, ca)
            blade_scale = blade_scale / self.oars.blade_efficiency
        oar_force_hull = oar_force_hull * power * blade_scale
        oar_moment_hull = oar_moment_hull * power * blade_scale

        # -- the crew's balance reflex, applied through the riggers ----
        # The demand is a saturated PD loop on roll, but the crew cannot
        # apply a pure roll couple: they change handle height, which loads
        # the oar as a lever and puts a vertical force at the rigger.  In a
        # conventionally rigged sweep boat the port and starboard riggers
        # sit a seat apart longitudinally, so that set of forces carries a
        # pitch couple of about 0.72 of the roll couple.  Balancing a sweep
        # eight pitches it; a pure x couple cannot represent that.
        balance = -(self.balance_stiffness * roll
                    + self.balance_damping * omega_hull[0])
        balance = self.balance_limit * ca.tanh(balance / self.balance_limit)
        balance_force_t, balance_moment_t = self.balance_rig.loads(balance)
        balance_force = ca.vertcat(*balance_force_t)
        balance_moment = ca.vertcat(*balance_moment_t)

        force_hull = (resistance + appendage_force + oar_force_hull
                      + balance_force)
        moment_hull_total = (appendage_moment + oar_moment_hull
                             + balance_moment)

        force = (rotation @ force_hull + buoyancy_force + weight
                 + crew_force)
        moment = (rotation @ moment_hull_total + buoyancy_moment
                  + weight_moment + crew_moment
                  - ca.cross(omega, inertia_abs @ omega))

        acceleration = ca.solve(mass_matrix, ca.vertcat(force, moment))

        drawn = self.critical_power * power ** 1.5
        return ca.vertcat(
            velocity,
            _euler_rates(roll, pitch, yaw, rotation.T @ omega, ca),
            acceleration[0:3],
            acceleration[3:6],
            -(drawn - self.critical_power),
        )

    def function(self, depth=None):
        import casadi as ca

        state = ca.MX.sym("state", self.n_states)
        control = ca.MX.sym("control", self.n_controls)
        time = ca.MX.sym("time")
        return ca.Function(
            "sixdof", [state, control, time],
            [self.derivative(state, control, time, depth)],
            ["state", "control", "time"], ["derivative"])


# --------------------------------------------------------------------------
# small symbolic helpers
# --------------------------------------------------------------------------
def _blade_efficiency(blade, oars, time, surge, depth, ca):
    """Instantaneous blade efficiency, symbolically.

    ``1 - |slip| / |blade speed|`` from Cabrera, Ruina & Kleshnev, scaled
    by the depth of water around the blade.  The blade speed passes through
    zero at the catch and the finish, so it is floored -- the oar force is
    zero there anyway, and an unfloored division would put a pole in the
    dynamics exactly where the optimiser wants to place a node.
    """
    angle = oars.angle.casadi(time)
    rate = oars.angle_rate.casadi(time)
    blade_speed = blade.outboard * rate
    slip = blade_speed + surge * ca.cos(angle)
    magnitude = ca.fmax(ca.fabs(blade_speed), 0.25)
    efficiency = 1.0 - ca.fabs(slip) / magnitude
    efficiency = ca.fmax(efficiency, 0.05)

    # Deliberately *not* multiplied by ``immersion_factor``.  The lumped
    # 0.78 this replaces is a measured total blade efficiency at nominal
    # cover, so it already carries the ventilation and immersion losses;
    # charging for them again is the double count recorded in SOURCES
    # section 7, and it is what made an earlier attempt lose 14% of boat
    # speed.  What the fixed constant cannot carry is the *variation* --
    # slip within the stroke, and the depth of water round the blade --
    # so those are exactly what is restored here.
    factor = 1.0
    if depth is not None:
        sigma = blade.blade_width / ca.fmax(depth, blade.blade_width * 1.5)
        factor = 1.0 + blade.blockage_m * sigma * blade.blade_cd
    return efficiency * factor


def _skew(vector, ca):
    return ca.blockcat([
        [0.0, -vector[2], vector[1]],
        [vector[2], 0.0, -vector[0]],
        [-vector[1], vector[0], 0.0],
    ])


def _rotation(roll, pitch, yaw, ca):
    """Hull to absolute, matching :func:`coxswain.core.frames.hull_to_abs`."""
    cph, sph = ca.cos(roll), ca.sin(roll)
    cth, sth = ca.cos(pitch), ca.sin(pitch)
    cps, sps = ca.cos(yaw), ca.sin(yaw)
    return ca.blockcat([
        [cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps],
        [cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps],
        [-sth, sph * cth, cph * cth],
    ])


def _euler_rates(roll, pitch, yaw, omega_body, ca):
    """Matching :func:`coxswain.core.frames.euler_rates_from_body`."""
    p, q, r = omega_body[0], omega_body[1], omega_body[2]
    sph, cph = ca.sin(roll), ca.cos(roll)
    cth = ca.cos(pitch)
    common = q * sph + r * cph
    return ca.vertcat(
        p + common * ca.tan(pitch),
        q * cph - r * sph,
        common / cth,
    )


def _crew_reaction_moment(crew, rotation, omega, time, ca):
    """``sum_k m_k r_k x transport_k`` for the crew, absolute frame.

    Two pieces, and the second is easy to lose:

    * the Coriolis part, which is the rate of change of the crew's inertia
      tensor acting on the angular velocity; and
    * ``sum m (r x a_rel)``, tabulated directly because it is *not*
      ``(sum m r) x (sum m a)`` -- that multiplies by mass twice.

    The second vanishes for **yaw** with a port-starboard symmetric crew,
    which is what made it easy to drop.  It does not vanish for **pitch**:
    the crew sits above the centre of mass, so their fore-and-aft
    acceleration at height ``z`` makes a real pitching moment.  Omitting it
    overstated the pitch swing by 54%.
    """
    inertia_rate = ca.blockcat([
        [crew.inertia_xx.derivative().casadi(time),
         crew.inertia_xy.derivative().casadi(time),
         crew.inertia_xz.derivative().casadi(time)],
        [crew.inertia_xy.derivative().casadi(time),
         crew.inertia_yy.derivative().casadi(time),
         crew.inertia_yz.derivative().casadi(time)],
        [crew.inertia_xz.derivative().casadi(time),
         crew.inertia_yz.derivative().casadi(time),
         crew.inertia_zz.derivative().casadi(time)],
    ])
    inertia_rate_abs = rotation @ inertia_rate @ rotation.T
    cross_abs = rotation @ crew.cross_accel_hull(time, ca)
    return inertia_rate_abs @ omega + cross_abs
