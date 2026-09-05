"""Rigging geometry: seats, oarlocks and oars.

All positions are in the hull frame (``x`` stern->bow, ``y`` to port,
``z`` up), measured from the hull centre of mass ``G_h``.

The oar is modelled as Formaggia et al. section 4.3 do: an ideal massless
lever of length ``L`` with its fulcrum at the blade and the oarlock a
distance ``L - r_h`` from it, where ``r_h`` is the inboard.  Force balance
on the lever (their eq. 12) gives the handle force

    F_h = -(L - r_h)/L  F_o

so the *net* force the rower-oar pair delivers to the hull is only
``(r_h / L) F_o``, not ``F_o``.  For a sweep rig that factor is about
0.31 -- the legacy code applied the full oarlock force to the hull and so
overstated the thrust by more than a factor of three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..crew.anthropometry import PORT, STARBOARD

__all__ = ["RIG_PATTERNS", 
    "Oar", "Oarlock", "Seat", "Rig", "SWEEP_OAR", "SCULLING_OAR",
    "build_sweep_rig", "build_sculling_rig",
]


@dataclass(frozen=True)
class Oar:
    """An oar, treated as an ideal lever.

    Attributes
    ----------
    length:
        ``L``, blade centre to handle end, in metres.
    inboard:
        ``r_h``, oarlock to handle centre, in metres.
    blade_area:
        Reference area of the blade, in m^2.  Not used by the ideal-lever
        model; carried for a future blade-hydrodynamics model.
    blade_efficiency:
        Fraction of the ideal-lever thrust actually delivered.  Formaggia
        et al. assume "perfect blades", i.e. the lever fulcrum sits at the
        blade and does not move.  Real blades slip through the water, and
        the paper notes the assumption "can be weakened by using a more
        detailed model of the blade action".  Published blade efficiencies
        for rowing are 0.75-0.85; without that detailed model, this single
        factor stands in for the loss.
    """

    length: float
    inboard: float
    blade_area: float = 0.11
    blade_efficiency: float = 0.78
    grip_separation: float = 0.0
    #: Oar mass in kg.  Irrelevant to propulsion -- the lever model does
    #: not need it -- but it is the *entire* mechanism by which a crew can
    #: influence roll during the recovery, when the blade is in the air and
    #: there is nothing else to push against.  Default is a composite sweep
    #: oar; Concept2's published mass for a Fat2/Skinny sweep oar is about
    #: 2.7 kg depending on layup and length.
    mass: float = 2.7

    def __post_init__(self) -> None:
        if not 0.0 < self.blade_efficiency <= 1.0:
            raise ValueError("blade_efficiency must lie in (0, 1]")
        if not 0.0 <= self.grip_separation < self.inboard:
            raise ValueError(
                f"grip_separation ({self.grip_separation}) must lie in "
                f"[0, inboard) = [0, {self.inboard})"
            )
        if not 0.0 < self.inboard < self.length:
            raise ValueError(
                f"inboard ({self.inboard}) must lie strictly between 0 and "
                f"the oar length ({self.length})"
            )

    @property
    def inertia_about_lock(self) -> float:
        """Second moment of the oar about the oarlock, kg m^2.

        Uniform slender rod of length ``L`` with the pivot ``inboard`` from
        the handle end, so the centre of mass sits ``L/2 - inboard``
        outboard of the pivot: ``m (L^2/12 + d^2)``.  A real oar is not
        uniform -- the blade end is lighter than a solid rod would be and
        modern shafts taper -- so this is an upper bound on the inertia and
        therefore a *conservative* estimate of recovery authority: a
        lighter-ended oar is easier to swing, not harder.
        """
        offset = 0.5 * self.length - self.inboard
        return self.mass * (self.length ** 2 / 12.0 + offset ** 2)

    @property
    def centre_of_mass_offset(self) -> float:
        """Distance of the oar's centre of mass outboard of the lock."""
        return 0.5 * self.length - self.inboard

    @property
    def gearing(self) -> float:
        """``r_h / L`` -- the fraction of the oarlock force reaching the hull.

        This is the coefficient on ``sum F_o`` in Formaggia eq. (14a).
        """
        return self.inboard / self.length

    @property
    def effective_gearing(self) -> float:
        """``blade_efficiency * r_h / L`` -- what actually drives the hull."""
        return self.blade_efficiency * self.gearing

    @property
    def outboard(self) -> float:
        """``L - r_h``, oarlock to blade centre."""
        return self.length - self.inboard


#: World Rowing standard sweep oar: 3.70 m overall, 1.14 m inboard.
#:
#: ``grip_separation`` is the spacing of the two hands along the handle.  A
#: sweep rower holds one oar in both hands: the outside hand takes the very
#: end of the handle and the inside hand sits about 0.30 m closer to the
#: oarlock.  It is not decoration -- with both hands modelled at the same
#: point the outside shoulder finishes 0.75 m from a 0.70 m arm, i.e. the
#: rower cannot hold their own oar.
SWEEP_OAR = Oar(length=3.70, inboard=1.14, blade_area=0.110,
                grip_separation=0.30)

#: Standard sculling oar: 2.88 m overall, 0.88 m inboard.  A sculler has
#: one hand per oar, so there is no grip separation.
SCULLING_OAR = Oar(length=2.88, inboard=0.88, blade_area=0.083)


@dataclass(frozen=True)
class Oarlock:
    """One oarlock: where it sits and which way its oar points."""

    position: np.ndarray  # (3,) hull frame
    side: int             # PORT (+1) or STARBOARD (-1)
    oar: Oar

    def __post_init__(self) -> None:
        if self.side not in (PORT, STARBOARD):
            raise ValueError(f"side must be PORT or STARBOARD, got {self.side}")
        if np.asarray(self.position).shape != (3,):
            raise ValueError("oarlock position must be a length-3 vector")


@dataclass(frozen=True)
class Seat:
    """One rowing position: a footboard station plus its oarlock(s).

    A sweep rower has one oarlock; a sculler has two, one per side.
    """

    station_x: float
    oarlocks: Tuple[Oarlock, ...]
    label: str = ""

    @property
    def is_sculling(self) -> bool:
        return len(self.oarlocks) == 2

    @property
    def rigged_side(self) -> int:
        """Which side a sweep rower's oar is on; 0 for a sculler."""
        if self.is_sculling:
            return 0
        return self.oarlocks[0].side


@dataclass(frozen=True)
class Rig:
    """The complete rigging of a boat."""

    seats: Tuple[Seat, ...]
    coxswain_position: np.ndarray = None  # (3,) or None for a coxless boat
    coxswain_mass: float = 0.0
    #: True where the coxswain lies supine in the bow rather than sitting
    #: upright in the stern.  Nearly every modern 4+ is a bow-loader, and
    #: it is not a detail: the coxswain's mass moves from behind the
    #: stern seat to ahead of the bow seat -- eight metres, a quarter of
    #: the boat's length -- which changes the trim and the pitch inertia,
    #: and it moves the only viewpoint that matters from four metres
    #: behind the crew to four metres in front of them, lying down.
    coxswain_reclined: bool = False

    @property
    def coxswain_eye_height(self) -> float:
        """Height of the coxswain's eye above their seat reference, m.

        0.70 m seated upright.  0.55 m for a bow-loader, who lies on
        their back with their head propped up on the cockpit coaming.

        Arrived at from the render rather than assumed.  At 0.25 m the
        eye sat two centimetres over a 0.25 m gunwale and the frame was
        the inside of the boat; at 0.40 m the hull's own sides still rose
        across the lower half of the view as two pale walls, because the
        hull mesh here is an open shell -- it has no foredeck, so from
        inside it you see all the way down to the keel.  A real
        bow-loader is decked over except for the cockpit, and 0.55 m puts
        the eye where the head actually is: clear of the coaming, looking
        out over the bow.

        This is why bow-loading is hard to steer from, and why the 3-D
        scene is worth having: from there the crew is behind you, the
        view is a hand's breadth off the water, and there is nothing in
        the frame to steer by but the far shore.
        """
        return 0.55 if self.coxswain_reclined else 0.70

    @property
    def n_seats(self) -> int:
        return len(self.seats)

    @property
    def n_oars(self) -> int:
        return sum(len(seat.oarlocks) for seat in self.seats)

    @property
    def is_sweep(self) -> bool:
        return all(not seat.is_sculling for seat in self.seats)

    @property
    def has_coxswain(self) -> bool:
        return self.coxswain_position is not None

    def side_balance(self) -> int:
        """Net rigged side over the crew; 0 for a balanced sweep boat."""
        return sum(seat.rigged_side for seat in self.seats)

    def oarlock_positions(self) -> np.ndarray:
        """Shape ``(n_oars, 3)`` array of every oarlock position."""
        return np.array([lock.position
                         for seat in self.seats for lock in seat.oarlocks])


#: Named sweep seating patterns, written stern to bow from the stroke seat,
#: ``+1`` port and ``-1`` starboard.  The names are the ones crews use, with
#: the caveat that usage varies between countries -- the physics cares only
#: about the side sequence, so the sequence is the definition here.
#:
#: What a pattern changes is the **stagger couple**: each oarlock sits a
#: seat's fraction ahead of its rower, so summing ``side * station`` over
#: the crew leaves a net yaw moment arm.  A standard alternating rig has
#: the largest; rigs that pair same-side seats ("buckets") can cancel it
#: almost exactly, which is the entire engineering argument for them.
RIG_PATTERNS = {
    # conventional alternating, port stroke -- what almost every club rows
    "standard":         (+1, -1, +1, -1, +1, -1, +1, -1),
    # the mirror, for a starboard-side stroke
    "starboard stroke": (-1, +1, -1, +1, -1, +1, -1, +1),
    # German (Ratzeburg): buckets at 5-4 and 3-2 -- the classic fix for the
    # stagger couple, used by the 1960 Ratzeburg eight
    "german":           (+1, -1, -1, +1, +1, -1, -1, +1),
    # Italian: stern four standard, bow four its mirror image -- the
    # arrangement the Moto Guzzi four made famous, extended to an eight
    "italian":          (+1, -1, +1, -1, -1, +1, -1, +1),
    # battleship, as the coxswain this model is built with rigs it: the
    # stern pair and bow pair on port, the middle four all on starboard.
    # Balanced in count, but the port oars sit at the ends of the boat and
    # the starboard oars amidships, so the two sides have very different
    # yaw leverage -- which is what makes it worth simulating rather than
    # eyeballing.
    "battleship":       (+1, +1, -1, -1, -1, -1, +1, +1),
    # tandem-pair variant sometimes given the same name
    "tandem":           (+1, -1, -1, +1, -1, +1, +1, -1),
}


def build_sweep_rig(n_seats: int, spacing: float, stern_station: float,
                    span: float, oarlock_height: float,
                    oar: Oar = SWEEP_OAR,
                    stroke_side: int = PORT,
                    coxswain_position=None,
                    coxswain_mass: float = 0.0,
                    coxswain_reclined: bool = False,
                    sides=None) -> Rig:
    """Lay out a sweep rig.

    Seats are numbered from the stern (seat 0 is stroke).  ``stroke_side``
    is the side stroke's oar is on -- port for a conventional rig, which
    then alternates down the boat.  ``sides`` overrides the alternating
    pattern entirely: a sequence of ``+1``/``-1`` from the stroke seat,
    most usefully one of :data:`RIG_PATTERNS`.
    """
    if sides is not None:
        if len(sides) != n_seats:
            raise ValueError("sides needs one entry per seat")
        if abs(sum(sides)) > 0:
            raise ValueError("a sweep rig needs equal port and starboard")
    seats: List[Seat] = []
    for index in range(n_seats):
        station_x = stern_station + index * spacing
        if sides is not None:
            side = int(sides[index])
        else:
            side = stroke_side if index % 2 == 0 else -stroke_side
        # the oarlock sits slightly towards the bow of the rower's station
        lock = Oarlock(
            position=np.array([station_x + 0.30, side * span, oarlock_height]),
            side=side,
            oar=oar,
        )
        label = "stroke" if index == 0 else (
            "bow" if index == n_seats - 1 else str(n_seats - index))
        seats.append(Seat(station_x=station_x, oarlocks=(lock,), label=label))

    return Rig(seats=tuple(seats),
               coxswain_position=(None if coxswain_position is None
                                  else np.asarray(coxswain_position,
                                                  dtype=float)),
               coxswain_mass=coxswain_mass,
               coxswain_reclined=coxswain_reclined)


def build_sculling_rig(n_seats: int, spacing: float, stern_station: float,
                       span: float, oarlock_height: float,
                       oar: Oar = SCULLING_OAR,
                       coxswain_position=None,
                       coxswain_mass: float = 0.0,
                       coxswain_reclined: bool = False) -> Rig:
    """Lay out a sculling rig: two oarlocks per seat, one each side."""
    seats: List[Seat] = []
    for index in range(n_seats):
        station_x = stern_station + index * spacing
        locks = tuple(
            Oarlock(
                position=np.array([station_x + 0.30, side * span,
                                   oarlock_height]),
                side=side,
                oar=oar,
            )
            for side in (PORT, STARBOARD)
        )
        label = "stroke" if index == 0 else (
            "bow" if index == n_seats - 1 else str(n_seats - index))
        seats.append(Seat(station_x=station_x, oarlocks=locks, label=label))

    return Rig(seats=tuple(seats),
               coxswain_position=(None if coxswain_position is None
                                  else np.asarray(coxswain_position,
                                                  dtype=float)),
               coxswain_mass=coxswain_mass,
               coxswain_reclined=coxswain_reclined)
