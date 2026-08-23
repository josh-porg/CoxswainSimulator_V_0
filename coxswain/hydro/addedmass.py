"""Hydrodynamic added mass of the hull, by strip theory.

A hull accelerating through water accelerates water with it.  The
entrained water appears in the equations of motion as an addition to the
mass and inertia, and for a rowing shell it is **not** a correction --
in sway and yaw it is the same size as the boat or larger.

Leaving it out makes the boat far too easy to turn, which is exactly the
quantity a coxswain is trying to reason about.  For the double scull the
physical yaw inertia is a few hundred kg m^2 (hull plus crew) against
roughly a thousand of added yaw inertia, so a model without it responds
to rudder and to differential pressure several times too briskly, with
time constants several times too short.

Method
------
Classical strip theory.  The hull is cut into transverse stations, each
station's two-dimensional added mass is evaluated, and the sectional
values are integrated along the length:

    m22 = int a22(x) dx          sway
    m33 = int a33(x) dx          heave
    m44 = int a44(x) dx          roll
    m55 = int a33(x) x^2 dx      pitch
    m66 = int a22(x) x^2 dx      yaw

The sections here are semi-elliptical (see
:func:`coxswain.hydro.hull.parametric_offsets`), and the two-dimensional
added mass of an ellipse is classical: for a full ellipse with semi-axes
``b`` horizontal and ``T`` vertical, motion along the horizontal axis
entrains ``rho pi T^2`` per unit length, and motion along the vertical
axis ``rho pi b^2``  [Lamb 1932, art. 71].  A surface-piercing half
section is treated by the method of images with the free surface as a
rigid wall, which halves those values -- the high-frequency limit, and
the standard strip-theory starting point [Newman 1977, sec. 4.13;
Korotkin 2009].

Surge is different: a slender body moving along its own axis entrains
very little, and the appropriate figure is Lamb's inertia coefficient
``k1`` for a prolate spheroid of the same length-to-beam ratio.  At
``L/B ~ 30`` that is a couple of percent of the displaced mass, so surge
added mass is small -- but it is not zero, and it is the term that acts
on the hull's response to the crew moving up and down the slide.

Limitations, stated plainly
---------------------------
* The rigid-wall image is a high-frequency approximation.  At a stroke
  rate near 0.4 Hz a rowing shell is not obviously in that limit, and a
  frequency-dependent treatment would need a free-surface Green function.
  The error is largest in heave, which is stiff in buoyancy anyway.
* Sectional values are taken at the *design* waterline rather than the
  instantaneous one, so the added mass does not breathe with heave and
  pitch.  For a shell that trims by a couple of centimetres this is
  small; for large-amplitude motion it would not be.
* Cross-coupling terms are retained where strip theory gives them and are
  the reason a hull turns and heels together.

References
----------
Lamb, H. (1932) *Hydrodynamics*, 6th ed., art. 71.
Newman, J.N. (1977) *Marine Hydrodynamics*, sec. 4.13.
Korotkin, A.I. (2009) *Added Masses of Ship Structures*, Springer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AddedMass", "sectional_sway", "sectional_heave",
           "surge_coefficient"]


def surge_coefficient(length: float, beam: float) -> float:
    """Lamb's ``k1`` for a prolate spheroid of this slenderness.

    Tabulated by Lamb (1932) art. 71; evaluated here from the closed form
    for the prolate spheroid rather than interpolated from the table.
    """
    ratio = max(float(length) / max(float(beam), 1e-9), 1.0001)
    e2 = 1.0 - 1.0 / (ratio * ratio)
    e = float(np.sqrt(e2))
    alpha0 = (2.0 * (1.0 - e2) / (e ** 3)) * (
        0.5 * np.log((1.0 + e) / (1.0 - e)) - e)
    return float(alpha0 / (2.0 - alpha0))


def sectional_sway(depth, rho: float) -> np.ndarray:
    """2D sway added mass per unit length, kg/m.

    Half of ``rho pi T^2`` -- the ellipse result with the free surface as
    a rigid wall.  It depends on the **draft**, not the beam: a section
    swaying sideways presents its depth to the flow.
    """
    return 0.5 * rho * np.pi * np.asarray(depth, dtype=float) ** 2


def sectional_heave(beam, rho: float) -> np.ndarray:
    """2D heave added mass per unit length, kg/m.

    Half of ``rho pi b^2`` with ``b`` the section half-beam: a section
    heaving presents its width.
    """
    half = 0.5 * np.asarray(beam, dtype=float)
    return 0.5 * rho * np.pi * half ** 2


@dataclass(frozen=True)
class AddedMass:
    """The 6x6 added-mass matrix of a hull, in the hull frame.

    Ordered ``(surge, sway, heave, roll, pitch, yaw)``.
    """

    matrix: np.ndarray

    @classmethod
    def from_offsets(cls, offsets, rho: float = 1000.0,
                     roll_factor: float = 0.15) -> "AddedMass":
        """Strip-theory added mass from a table of hull offsets.

        ``roll_factor`` scales the sectional roll added inertia, which for
        a shallow round-bilged section is a small fraction of what the
        sway term alone would suggest.  It matters little either way
        because roll is dominated by the crew.
        """
        x = np.asarray(offsets.station, dtype=float)
        beam = np.asarray(offsets.beam, dtype=float)
        depth = np.asarray(offsets.depth, dtype=float)

        a22 = sectional_sway(depth, rho)
        a33 = sectional_heave(beam, rho)
        a44 = roll_factor * a22 * depth ** 2

        m = np.zeros((6, 6))
        volume = float(np.trapezoid(0.25 * np.pi * beam * depth, x))
        k1 = surge_coefficient(float(x[-1] - x[0]), float(beam.max()))
        m[0, 0] = k1 * rho * volume
        m[1, 1] = float(np.trapezoid(a22, x))
        m[2, 2] = float(np.trapezoid(a33, x))
        m[3, 3] = float(np.trapezoid(a44, x))
        m[4, 4] = float(np.trapezoid(a33 * x * x, x))
        m[5, 5] = float(np.trapezoid(a22 * x * x, x))
        m[1, 5] = m[5, 1] = float(np.trapezoid(a22 * x, x))
        m[2, 4] = m[4, 2] = -float(np.trapezoid(a33 * x, x))
        return cls(matrix=m)

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[0:3, 0:3]

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[3:6, 3:6]

    @property
    def coupling(self) -> np.ndarray:
        return self.matrix[0:3, 3:6]

    # -- velocity-dependent terms -----------------------------------------
    def coriolis(self, velocity_hull, omega_hull,
                 munk_factor: float = 0.0) -> np.ndarray:
        """Added-mass Coriolis-centripetal load, in the **hull** frame.

        The mass matrix is only half of what added mass does.  Because the
        entrained water is carried in a body-fixed frame, a hull moving
        and turning at once feels velocity-dependent forces as well, and
        the important one for a slender hull is the **Munk moment**.

        For a hull with surge added mass ``m11`` and sway added mass
        ``m22`` moving at forward speed ``u`` with sideslip ``v``, the
        yaw moment is

            N = (m11 - m22) u v

        and since ``m22`` greatly exceeds ``m11`` for anything
        boat-shaped, it acts to turn the hull **broadside** to its own
        motion.  It is destabilising, it is what makes a bare hull want to
        broach, and without it directional stability rests entirely on the
        skeg and rudder -- which flatters the boat's behaviour.

        The general form is Fossen's: with the added-mass matrix
        partitioned into ``A11, A12, A21, A22`` blocks and the generalised
        velocity into linear ``v1`` and angular ``v2`` parts,

            C_A = [[     0      , -S(A11 v1 + A12 v2)],
                   [-S(A11 v1 + A12 v2), -S(A21 v1 + A22 v2)]]

        and the load is ``-C_A(nu) nu``.  Taking the Munk moment from this
        rather than writing it down directly means the cross-coupling
        terms and the rotational contributions come out consistently.

        Why ``munk_factor`` is 0.5 and not 1.0
        --------------------------------------
        The moment above is the **ideal-flow** value, and it is computed
        from added mass evaluated in the high-frequency limit -- the
        free surface treated as a rigid wall.  That limit is the right one
        for motion at stroke rate, which is what the mass matrix is for.
        It is the wrong one for steady manoeuvring, where the free surface
        behaves as a free boundary and the effective sway added mass is
        substantially smaller.  Using the high-frequency value at zero
        frequency overstates the moment.

        For a typical section the low-frequency sway added mass is
        roughly half the rigid-wall value, which puts the factor near
        0.5.

        That estimate is corroborated by the sharpest qualitative
        evidence available: **a shell that loses its skeg becomes
        uncontrollable.**  Coxswains who have lost one describe violent
        slewing and putting an arm in the water to steer.  Any model of
        directional stability has to reproduce that, and it is a strong
        discriminator because it brackets the term from both sides.

        Eight, yawed 3 deg off course, rudder centred, ten strokes:

        =========  ===============  ===============
        factor     skeg fitted      skeg removed
        =========  ===============  ===============
        0.00       holds (-2.6)     holds (-5.1)
        0.25       holds (-5.0)     slews (+9.4)
        **0.50**   slews (+18.6)    broaches (+36.0)
        1.00       broaches (+61)   broaches (+79)
        =========  ===============  ===============

        At zero the model says losing the skeg barely matters, which
        contradicts the reported experience outright -- so the term is
        needed, and switching it off is not the safe choice it looks
        like.  At full strength the boat broaches with the skeg fitted,
        which is equally wrong.  Between them the model behaves the way
        the boat does.

        This remains a *calibration*, not a derivation.  Pinning it
        properly needs either a free-surface computation of the
        low-frequency sway added mass or a measured turning circle to fit
        an empirical ``N_v`` against, which is how ship manoeuvring codes
        handle this term.

        Reference
        ---------
        Fossen, T.I. (2011) *Handbook of Marine Craft Hydrodynamics and
        Motion Control*, Wiley, sec. 6.3.
        """
        if munk_factor == 0.0:
            return np.zeros(6)
        v1 = np.asarray(velocity_hull, dtype=float).reshape(3)
        v2 = np.asarray(omega_hull, dtype=float).reshape(3)
        a = self.matrix
        top = a[0:3, 0:3] @ v1 + a[0:3, 3:6] @ v2
        bot = a[3:6, 0:3] @ v1 + a[3:6, 3:6] @ v2

        # load = -C_A nu, written out with S(x) y = x cross y
        force = np.cross(top, v2)
        moment = np.cross(top, v1) + np.cross(bot, v2)
        return float(munk_factor) * np.concatenate([force, moment])

    def summary(self) -> str:
        d = np.diag(self.matrix)
        return ("surge %.1f kg, sway %.1f kg, heave %.1f kg; "
                "roll %.2f, pitch %.1f, yaw %.1f kg m^2"
                % (d[0], d[1], d[2], d[3], d[4], d[5]))
