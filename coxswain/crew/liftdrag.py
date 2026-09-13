r"""Tier 2 blade: lift and drag resolved against the angle of attack.

What it adds to tier 1
----------------------
[CR06] Model 1 is a pure normal force from the normal component of slip, so
the blade has no tangential load and never sees the flow *along* the shaft.
Here the blade sees its whole velocity through the water, in the horizontal
plane of the hull frame::

    w   = u_lock + l phi_dot e_theta          (u_lock = v_hull + omega x r_lock)
    w_n = w . e_theta                         normal to the blade  (Model 1's slip)
    w_a = w . e_r                             along the shaft, towards the tip

and the angle of attack is the angle between that velocity and the blade's
plane, which contains the shaft::

    alpha = atan2(|w_n|, |w_a|)               in [0, pi/2]

The coefficients
----------------
Caplan & Gardner's flume fits, via a secondary source ([CG06a] in SOURCES,
provisional)::

    C_L = A_l sin(2 alpha)        C_D = A_d sin^2(alpha)

resolved onto the blade -- normal and tangential to its plane -- in the
standard flat-plate way::

    C_N = C_L cos(alpha) + C_D sin(alpha) = 2 A_l sin(a) cos^2(a) + A_d sin^3(a)
    C_T = C_D cos(alpha) - C_L sin(alpha) = (A_d - 2 A_l) sin^2(a) cos(a)

Two consequences worth stating before any run:

* **At alpha = 90 degrees it reduces to Model 1's form.**  C_T = 0 and the
  normal load is ``1/2 rho A A_d w_n^2`` -- for the rig's 0.11 m^2 Big Blade,
  114 w_n^2 N against [CR06]'s fitted 84.5 w_n^2.  Same law, 35% more grip.
* **The tangential coefficient is small and its sign is a prediction.**  For
  the Big Blade ``A_d - 2 A_l = 2.07 - 2.50 = -0.43``: the tangential load
  points AGAINST the drag direction along the shaft -- towards the tip when the
  flow runs from tip to root.  That is checkable against Grift's measured
  tangential traces, and it is the first thing tier 2 should be tested on when
  they can be obtained.

What turns the oar
------------------
Only the normal load has a moment about the pin; the tangential load acts
along the shaft, through it.  So the oar balance keeps its form and simply
reads this model's normal load, while the hull receives both components at the
blade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LiftDragBlade"]


@dataclass(frozen=True)
class LiftDragBlade:
    """A blade whose load depends on its angle of attack, not only its slip."""

    #: Where the constants came from, carried on the class so a result built
    #: on them cannot be quoted as though they were primary.
    PROVENANCE = ("[CG06a]: Caplan & Gardner's flume fits, via Atkinson's "
                  "'Oarblade Lift and Drag' -- a secondary source; the "
                  "primary is not read, so the constants are provisional.")

    #: Lift and drag amplitudes; Big Blade by default ([CG06a], provisional).
    lift_amplitude: float = 1.25
    drag_amplitude: float = 2.07
    #: Blade reference area, m^2 -- the rig's ``Oar.blade_area``.
    area: float = 0.11
    #: Outboard, pin to centre of pressure, m.
    outboard: float = 2.28
    density: float = 1000.0

    @classmethod
    def big_blade(cls, outboard: float, area: float = 0.11,
                  density: float = 1000.0) -> "LiftDragBlade":
        return cls(1.25, 2.07, area, outboard, density)

    @classmethod
    def macon(cls, outboard: float, area: float = 0.11,
              density: float = 1000.0) -> "LiftDragBlade":
        return cls(1.24, 1.90, area, outboard, density)

    # -- kinematics --------------------------------------------------------
    def relative_velocity(self, angle, rate, lock_velocity, side):
        """``(w_n, w_a)``: the blade's velocity through the water, normal to
        its plane and along the shaft, from the oarlock's water-relative
        velocity in the hull frame (horizontal components)."""
        angle = float(angle)
        normal = np.array([np.cos(angle), -side * np.sin(angle)])
        axis = np.array([np.sin(angle), side * np.cos(angle)])
        u = np.asarray(lock_velocity, dtype=float)[:2]
        w = u + self.outboard * float(rate) * normal
        return float(w @ normal), float(w @ axis)

    @staticmethod
    def attack_angle(w_n, w_a):
        return float(np.arctan2(abs(w_n), abs(w_a)))

    # -- coefficients ------------------------------------------------------
    def coefficients(self, alpha):
        """``(C_N, C_T)`` on the blade's own axes at attack angle ``alpha``."""
        s, c = np.sin(alpha), np.cos(alpha)
        lift = self.lift_amplitude * 2.0 * s * c
        drag = self.drag_amplitude * s * s
        return float(lift * c + drag * s), float(drag * c - lift * s)

    # -- loads -------------------------------------------------------------
    def loads(self, angle, rate, lock_velocity, side):
        """``(F_n, F_t)``, N: signed components along the blade normal
        ``e_theta`` and the shaft ``e_r``.

        Each opposes the corresponding component of the blade's motion through
        the water when its coefficient is positive, so the water resists the
        blade rather than driving it; ``C_T`` can be negative, and then the
        tangential load is along that motion.
        """
        w_n, w_a = self.relative_velocity(angle, rate, lock_velocity, side)
        speed2 = w_n * w_n + w_a * w_a
        if speed2 <= 0.0:
            return 0.0, 0.0
        c_n, c_t = self.coefficients(self.attack_angle(w_n, w_a))
        q = 0.5 * self.density * self.area * speed2
        return (-np.sign(w_n) * q * c_n, -np.sign(w_a) * q * c_t)
