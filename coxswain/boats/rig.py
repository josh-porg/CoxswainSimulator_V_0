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

__all__ = [
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
    """

    length: float
    inboard: float
    blade_area: float = 0.11

    def __post_init__(self) -> None:
        if not 0.0 < self.inboard < self.length:
            raise ValueError(
                f"inboard ({self.inboard}) must lie strictly between 0 and "
                f"the oar length ({self.length})"
            )

    @property
    def gearing(self) -> float:
        """``r_h / L`` -- the fraction of the oarlock force reaching the hull.

        This is the coefficient on ``sum F_o`` in Formaggia eq. (14a).
        """
        return self.inboard / self.length

    @property
    def outboard(self) -> float:
        """``L - r_h``, oarlock to blade centre."""
        return self.length - self.inboard


#: World Rowing standard sweep oar: 3.70 m overall, 1.14 m inboard.
SWEEP_OAR = Oar(length=3.70, inboard=1.14, blade_area=0.110)

#: Standard sculling oar: 2.88 m overall, 0.88 m inboard.
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


def build_sweep_rig(n_seats: int, spacing: float, stern_station: float,
                    span: float, oarlock_height: float,
                    oar: Oar = SWEEP_OAR,
                    stroke_side: int = PORT,
                    coxswain_position=None,
                    coxswain_mass: float = 0.0) -> Rig:
    """Lay out a conventional alternating sweep rig.

    Seats are numbered from the stern (seat 0 is stroke).  ``stroke_side``
    is the side stroke's oar is on -- port for a conventional rig, which
    then alternates down the boat.
    """
    seats: List[Seat] = []
    for index in range(n_seats):
        station_x = stern_station + index * spacing
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
               coxswain_mass=coxswain_mass)


def build_sculling_rig(n_seats: int, spacing: float, stern_station: float,
                       span: float, oarlock_height: float,
                       oar: Oar = SCULLING_OAR,
                       coxswain_position=None,
                       coxswain_mass: float = 0.0) -> Rig:
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
               coxswain_mass=coxswain_mass)
