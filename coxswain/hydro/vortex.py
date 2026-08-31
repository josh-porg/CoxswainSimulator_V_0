r"""A 2-D vortex method for the water behind a rowing boat.

The analytic model in :mod:`coxswain.hydro.wake` treats every puddle as an
independent blob decaying on its own.  That is enough to get an order of
magnitude and not enough to answer the question a coxswain actually has,
which is *where* to be.  Puddles are not independent: a train of them
advects itself, drifts sideways, and pairs up, and the induced velocity
at a point 3 m off the centreline and 20 m astern is a sum over all of
them, not a lookup on the nearest one.

Why a 2-D horizontal slice is a defensible reduction here
----------------------------------------------------------
Three things have to be true, and on this problem they are:

**The free surface stays nearly flat.**  Wave resistance is 70 N of the
261 N a masters eight carries, the Kelvin wedge is narrow, and the visible
disturbance behind a shell is swirl rather than height.  A 2-D slice
cannot represent surface waves, so it matters that there is little to
represent.

**The structure is vertical-axis.**  What a blade leaves is usually drawn
as a vortex ring, but the part that persists at the surface, and the part
a following blade meets, reads from above as a **pair of counter-rotating
vertical-axis vortices** -- which is exactly a 2-D dipole.

**Impulse is the conserved quantity, and it is already known.**  A 2-D
vortex pair of circulation ``Gamma`` separated by ``d`` carries impulse
``rho * Gamma * d`` per unit depth.  The momentum budget in
:mod:`coxswain.hydro.wake` fixes the impulse per puddle from the boat's
own drag, so **the circulation is not a free parameter** -- it is
whatever makes the dipole carry the momentum the crew put into the
water.

That leaves two constants: the initial pair separation, which is the
blade width, and an eddy viscosity, which sets how fast the cores spread.
That is one fewer free constant than the analytic model needed, and the
one that remains is the one every turbulent model has.

The method
----------
Desingularised point vortices -- vortex blobs -- with a Gaussian core::

    u_theta(r) = Gamma / (2 pi r) * (1 - exp(-r^2 / sigma^2))

which is the Lamb-Oseen profile and removes the singularity that makes
raw point vortices useless.  Blobs advect in each other's induced field
(Biot-Savart, summed directly) and their cores spread viscously::

    d(sigma^2)/dt = 4 nu_t

**Circulation is conserved, not decayed.**  It is tempting to add a decay
term to make old puddles die, and it would be wrong: impulse is conserved
for a vortex pair in unbounded water, and throwing circulation away
throws away the momentum the momentum budget just pinned.  Puddles get
weaker because their cores grow and the pair slows, not because the
vorticity evaporates.  When the cores grow past about a third of the pair
separation the dipole stops behaving like one, which is the model's own
statement of when a puddle has stopped being a puddle.

What this still cannot do
-------------------------
Surface waves, the vertical structure of the ring, and the aeration that
sets how much grip a blade loses.  It gives the induced *velocity* field
honestly; the step from there to lost blade force still runs through the
same assumptions :mod:`coxswain.hydro.wake` makes.

References
----------
.. [L32] Lamb, H. (1932) *Hydrodynamics*, 6th ed., art. 155 -- the
   impulse of a vortex pair.
.. [LC93] Leonard, A. and Chua, K. (1993) -- core spreading and vortex
   blob methods.
.. [C73] Chorin, A. J. (1973) *Numerical study of slightly viscous flow*,
   J. Fluid Mech. 57, 785-796.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["VortexField", "PuddleWake2D", "ThinBody"]

WATER_DENSITY = 1000.0


@dataclass
class VortexField:
    """A cloud of Gaussian-core vortex blobs in the horizontal plane."""

    #: Positions, shape ``(n, 2)``.
    position: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=float))
    #: Circulation of each blob, m^2/s, signed.
    circulation: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    #: Core radius of each blob, m.
    core: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    #: Eddy viscosity setting how fast cores spread, m^2/s.  Molecular
    #: water is 1e-6; a turbulent puddle is three orders above that.
    eddy_viscosity: float = 1.5e-3
    #: Age of each blob, s -- carried for diagnostics, not dynamics.
    age: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))

    def __len__(self) -> int:
        return len(self.circulation)

    def add(self, position, circulation, core) -> None:
        """Insert blobs."""
        position = np.atleast_2d(np.asarray(position, dtype=float))
        circulation = np.atleast_1d(np.asarray(circulation, dtype=float))
        core = np.atleast_1d(np.asarray(core, dtype=float))
        self.position = np.vstack([self.position, position])
        self.circulation = np.concatenate([self.circulation, circulation])
        self.core = np.concatenate([self.core, core])
        self.age = np.concatenate([self.age, np.zeros(len(circulation))])

    # -- the field --------------------------------------------------------
    def velocity_at(self, points) -> np.ndarray:
        """Induced velocity at arbitrary points, shape ``(m, 2)``.

        Direct Biot-Savart summation.  For the few thousand blobs a race
        leaves behind this is fast enough and exact; a fast multipole
        method is the fix if the count ever grows, and it would not
        change any answer.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if not len(self):
            return np.zeros_like(points)
        offset = points[:, None, :] - self.position[None, :, :]
        squared = np.einsum("ijk,ijk->ij", offset, offset)
        core = self.core[None, :] ** 2
        # Lamb-Oseen: the (1 - exp) factor removes the singularity and
        # gives a core that rotates as a solid body.
        strength = (self.circulation[None, :] / (2.0 * np.pi)
                    * (1.0 - np.exp(-squared / np.maximum(core, 1e-12)))
                    / np.maximum(squared, 1e-12))
        return np.stack([-np.einsum("ij,ij->i", strength, offset[:, :, 1]),
                         np.einsum("ij,ij->i", strength, offset[:, :, 0])],
                        axis=1)

    def step(self, dt: float) -> None:
        """Advect the blobs in their own field and spread their cores."""
        if not len(self):
            return
        # Second-order: a midpoint step.  Forward Euler lets a close pair
        # spiral outward, which for a dipole train shows up as puddles
        # that drift apart instead of running straight.
        velocity = self.velocity_at(self.position)
        midpoint = self.position + 0.5 * dt * velocity
        saved = self.position
        self.position = midpoint
        velocity = self.velocity_at(midpoint)
        self.position = saved + dt * velocity
        self.core = np.sqrt(self.core ** 2
                            + 4.0 * self.eddy_viscosity * dt)
        self.age = self.age + dt

    def impulse(self) -> np.ndarray:
        """Hydrodynamic impulse per unit depth, N s / m.

        ``rho * sum(Gamma_i * (y_i, -x_i))``.  A conserved quantity, so
        this is the model checking itself: it must stay at whatever the
        momentum budget put in.
        """
        if not len(self):
            return np.zeros(2)
        return WATER_DENSITY * np.array([
            float(np.sum(self.circulation * self.position[:, 1])),
            float(-np.sum(self.circulation * self.position[:, 0]))])


@dataclass
class PuddleWake2D:
    """A boat rowing along, shedding a dipole from each blade each stroke.

    ``drag`` and ``speed`` describe the boat being followed, exactly as in
    :class:`~coxswain.hydro.wake.PuddleWake`, so the two models are driven
    from the same momentum budget and can be compared without either
    being tuned to the other.
    """

    drag: float                     # N, the leader's total resistance
    speed: float                    # m/s
    period: float                   # s, stroke period
    n_blades: int = 8
    #: Lateral station of the blade track, m from the centreline.
    track: float = 3.15
    #: Longitudinal spacing of the seats, m.
    seat_spacing: float = 1.22
    #: Separation of the shed vortex pair, m.  The blade width: the pair
    #: is the two edges of the blade, which is where the vorticity is.
    pair_separation: float = 0.25
    #: Depth of the shed structure, m.  Enters only through converting a
    #: 3-D impulse into a 2-D one per unit depth.
    depth: float = 0.40
    #: Core radius at birth, m.
    core: float = 0.08
    eddy_viscosity: float = 1.5e-3
    density: float = WATER_DENSITY
    #: The leader's waterline offsets, for the potential-flow hull.
    offsets: object = None

    def __post_init__(self):
        self.field = VortexField(eddy_viscosity=self.eddy_viscosity)
        self.shed_count = 0
        self.hull_count = 0
        self.body = None

    def velocity_at(self, points) -> np.ndarray:
        """Everything at once: puddles, hull wake, and the hull itself."""
        total = self.field.velocity_at(points)
        if self.body is not None:
            total = total + self.body.velocity_at(points)
        return total

    @property
    def impulse_per_puddle(self) -> float:
        """N s, from the momentum budget -- the same one wake.py uses."""
        return self.drag * self.period / self.n_blades

    @property
    def circulation(self) -> float:
        """Circulation of each shed vortex, m^2/s.

        From ``I = rho * Gamma * d * h``: not fitted, derived.
        """
        return self.impulse_per_puddle / (
            self.density * self.pair_separation * self.depth)

    @property
    def self_induced_speed(self) -> float:
        """How fast a fresh pair translates, m/s.

        The consistency check on the whole construction: this should land
        near the blade slip a real eight shows, 0.5 to 1.0 m/s, because
        it is the same water seen from the other side.
        """
        return self.circulation / (2.0 * np.pi * self.pair_separation)

    #: Represent the hull's viscous wake as well as the blades' puddles.
    #: The two must sum to zero: a self-propelled body exerts no net force
    #: on the water, so whatever the blades push aft the hull drags
    #: forward.  Shedding only the puddles left the field carrying the
    #: crew's entire momentum output and none of its recovery, which is a
    #: wake no real boat has.
    hull_wake: bool = True
    #: Lateral separation of the hull's wake pair, m -- the waterline beam.
    hull_pair: float = 0.57

    @property
    def hull_circulation(self) -> float:
        """Circulation of the hull's wake pair, m^2/s.

        Sized so one stroke period of hull wake carries ``+D T`` -- the
        exact negative of what the eight blades shed in the same period.
        """
        # ``self.depth``, not a depth of its own.  A 2-D field is ONE
        # slice: momentum per unit depth is the conserved quantity, so
        # giving the hull its own depth silently breaks the cancellation
        # the whole model is anchored on.  With 0.30 m against the
        # blades' 0.40 the wake came out over-cancelled by exactly the
        # ratio of the two, 4/3, which is the tidiest possible clue.
        return (self.drag * self.period) / (
            self.density * self.hull_pair * self.depth)

    def shed_hull(self, x: float) -> None:
        """One stroke period of hull wake, on the centreline."""
        half = 0.5 * self.hull_pair
        gamma = self.hull_circulation
        # Opposite orientation to a puddle, so this water moves FORWARD.
        # Written with the same circulation ORDER as the puddles it was
        # supposed to cancel, it doubled the wake instead: the impulse
        # check went from 10140 to 48723 rather than to zero, which is
        # exactly the sort of sign error a conservation law is for.
        self.field.add([[x, -half], [x, half]], [-gamma, gamma],
                       [self.core, self.core])
        self.hull_count += 1

    def shed(self, x: float) -> None:
        """One stroke's puddles: one dipole per blade, at its own station.

        The first version dropped a single dipole per side at one
        longitudinal station -- two dipoles for eight blades.  That is
        not a rounding issue, it is a quarter of the momentum: the
        impulse check came out at a quarter of what the crew had put in
        the water and looked plausible only because nothing was compared
        against it.  Eight blades, eight dipoles, at the eight seat
        stations they are actually shed from.

        The pair is oriented so its self-induced motion is **aft** -- the
        water the blades pushed backwards to make the thrust.
        """
        half = 0.5 * self.pair_separation
        gamma = self.circulation
        seats = (np.arange(self.n_blades) - 0.5 * (self.n_blades - 1))             * self.seat_spacing
        for index, station in enumerate(seats):
            side = 1.0 if index % 2 == 0 else -1.0
            centre = side * self.track
            self.field.add([[x + station, centre - half],
                            [x + station, centre + half]],
                           [gamma, -gamma], [self.core, self.core])
            self.shed_count += 1

    def row(self, distance: float, dt: float = 0.05):
        """Row the leader ``distance`` metres, shedding as it goes.

        Returns the field, with the boat having finished at ``x = 0`` so
        that negative ``x`` is astern and the follower's gap is a
        positive number measured backwards.
        """
        self.body = ThinBody.from_offsets(self.offsets, self.speed)             if self.offsets is not None else None
        steps = int(round(distance / max(self.speed * dt, 1e-9)))
        # Shed from the first step, not from x = 0.  Initialised to zero
        # this loop compared a station that starts at -distance against a
        # threshold at the finish and shed nothing at all, silently
        # returning an empty field that every downstream number then
        # reported as "no wake".
        next_shed = -distance
        for k in range(steps):
            x = -distance + self.speed * dt * k
            if x >= next_shed - 1e-9:
                self.shed(x)
                if self.hull_wake:
                    self.shed_hull(x)
                next_shed = x + self.speed * self.period
            self.field.step(dt)
        return self.field


@dataclass
class ThinBody:
    """The hull itself, as a 2-D source distribution along its centreline.

    This is the potential-flow half of the problem, and the piece that was
    missing: until now the field contained only what the blades threw
    behind them, so the "helpful" half of a momentumless wake -- the water
    the hull drags along with it -- had no representation at all.

    Thin-body theory gives the source strength directly from the shape::

        sigma(x) = 2 U db/dx

    with ``b(x)`` the waterline half-beam.  No fitting: the hull's own
    offsets are the input, so a finer bow produces a weaker disturbance
    because it *is* a finer bow.

    Sources are attached to the body, not shed, so they translate with it
    and never enter the vortex cloud's own dynamics.
    """

    station: np.ndarray             # x along the hull, m, bow positive
    strength: np.ndarray            # source strength per unit length
    position: np.ndarray = field(   # where the hull's origin sits now
        default_factory=lambda: np.zeros(2))

    @classmethod
    def from_offsets(cls, offsets, speed: float, stations: int = 60):
        """Build from a boat's waterline offsets at a given speed."""
        x = np.asarray(offsets.station, dtype=float)
        half_beam = 0.5 * np.asarray(offsets.beam, dtype=float)
        fine = np.linspace(x.min(), x.max(), stations)
        beam = np.interp(fine, x, half_beam)
        return cls(station=fine,
                   strength=2.0 * float(speed) * np.gradient(beam, fine))

    def velocity_at(self, points) -> np.ndarray:
        """Velocity induced by the hull's displacement flow."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        sources = np.column_stack([self.station + self.position[0],
                                   np.full(len(self.station),
                                           self.position[1])])
        spacing = float(np.mean(np.diff(self.station)))
        offset = points[:, None, :] - sources[None, :, :]
        squared = np.maximum(np.einsum("ijk,ijk->ij", offset, offset), 1e-4)
        scale = (self.strength * spacing)[None, :] / (2.0 * np.pi * squared)
        return np.stack([np.einsum("ij,ij->i", scale, offset[:, :, 0]),
                         np.einsum("ij,ij->i", scale, offset[:, :, 1])],
                        axis=1)
