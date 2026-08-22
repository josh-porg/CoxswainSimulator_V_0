"""Newton-Euler dynamics for a rigid hull carrying prescribed moving masses.

This is a direct 3-D generalisation of Formaggia et al. (2009) eq. (14),
which is written for planar (surge/heave/pitch) motion.  The paper's eq.
(14) reads, with ``R`` mapping hull -> absolute and ``O = dR/dtheta``:

    (14a)  Mt Gh_ddot + O sum(m x) theta_ddot + 2 O sum(m x_dot) theta_dot
             - R (sum m x) theta_dot^2
           = -R sum(m x_ddot) + (rh/L) sum Fo + Mt g + Fw

    (14b)  R sum(m x) x Gh_ddot + (Iyy + sum m |x|^2) theta_ddot
             + 2 sum m R x  x  O x_dot theta_dot
           = -R sum(m x) x R x_ddot + R sum(xo - xh + (rh/L) xh) x Fo
             + R sum(m x) x g + Mw

Writing ``r = R x`` for the absolute-frame offset of a body part from the
hull centre of mass ``G_h``, and using ``omega_dot x r = -S(r) omega_dot``,
the ``theta_ddot`` couplings become matrix blocks:

    A      = sum_k m_k S(r_k)                     (skew, so A^T = -A)
    B_raw  = sum_k m_k S(r_k) S(r_k)              (symmetric, negative semi-def)

    M = [[ Mt I3 ,          -A          ],
         [   A   ,  I_abs - B_raw       ]]

Note the three signs that the legacy implementation had inverted:

* the ``omega_dot`` block of the *force* equation is ``-A`` (was ``+A``);
* the ``Gh_ddot`` block of the *moment* equation is ``+A`` (was ``-A``);
* the moving-mass contribution to rotational inertia is ``-B_raw``
  (was ``+B_raw``).

The last one is the fatal one.  ``-S(r)S(r) = |r|^2 I - r r^T`` is the
familiar positive-semi-definite point-mass inertia, and reduces to the
paper's scalar ``+sum m |x|^2`` in the planar case.  With the sign flipped
the crew *subtracts* rotational inertia; for an eight with rowers spread
over +-4 m that drives the pitch and yaw inertia negative, so every moment
becomes positive feedback and the integration diverges immediately.

All quantities in this module are in the **absolute frame** unless the name
says ``_hull``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import cross3, skew

__all__ = [
    "MovingMassField",
    "first_moment",
    "coupling_matrix",
    "moving_mass_inertia",
    "assemble_mass_matrix",
    "moving_mass_reaction",
    "gyroscopic_moment",
    "solve_accelerations",
]


@dataclass(frozen=True)
class MovingMassField:
    """A cloud of point masses with prescribed motion in the hull frame.

    Attributes
    ----------
    mass:
        Shape ``(n,)``, kilograms.
    position, velocity, acceleration:
        Shape ``(n, 3)``, hull-frame offset from ``G_h`` and its first two
        time derivatives *as seen by an observer riding on the hull*
        (i.e. ``x_ij``, ``x_dot_ij``, ``x_ddot_ij`` of the paper).
    """

    mass: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self) -> None:
        n = self.mass.shape[0]
        for name in ("position", "velocity", "acceleration"):
            arr = getattr(self, name)
            if arr.shape != (n, 3):
                raise ValueError(
                    f"{name} must have shape ({n}, 3), got {arr.shape}"
                )

    @property
    def total_mass(self) -> float:
        return float(self.mass.sum())

    @classmethod
    def empty(cls) -> "MovingMassField":
        z = np.zeros((0, 3))
        return cls(mass=np.zeros(0), position=z, velocity=z, acceleration=z)

    def to_abs(self, rot_hull_to_abs: np.ndarray) -> "MovingMassField":
        """Rotate every vector into the absolute frame.

        Positions become ``r = R x``; velocities and accelerations become the
        *relative* motion resolved in absolute axes (they are still relative
        to the hull -- the transport terms are added separately by
        :func:`moving_mass_reaction`).
        """
        rot = rot_hull_to_abs
        return MovingMassField(
            mass=self.mass,
            position=self.position @ rot.T,
            velocity=self.velocity @ rot.T,
            acceleration=self.acceleration @ rot.T,
        )


def first_moment(mass: np.ndarray, position: np.ndarray) -> np.ndarray:
    """``sum_k m_k r_k`` -- the first mass moment about ``G_h``."""
    return (mass[:, None] * position).sum(axis=0)


def coupling_matrix(mass: np.ndarray, position: np.ndarray) -> np.ndarray:
    """``A = sum_k m_k S(r_k)``.

    Skew-symmetric; equals ``S(sum m r)`` because ``S`` is linear.
    """
    return skew(first_moment(mass, position))


def moving_mass_inertia(mass: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Inertia tensor about ``G_h`` of the point-mass cloud.

    ``sum_k m_k (|r_k|^2 I - r_k r_k^T) = -sum_k m_k S(r_k) S(r_k)``, which
    is symmetric positive semi-definite.  This is the 3-D form of the
    paper's ``sum m_ij |x_ij|^2`` term in eq. (14b).
    """
    if mass.size == 0:
        return np.zeros((3, 3))
    sq = np.einsum("k,ki,ki->", mass, position, position)
    outer = np.einsum("k,ki,kj->ij", mass, position, position)
    return sq * np.eye(3) - outer


def assemble_mass_matrix(total_mass: float, inertia_abs: np.ndarray,
                         mass: np.ndarray, position: np.ndarray,
                         added_mass_abs: np.ndarray = None) -> np.ndarray:
    """Build the symmetric 6x6 generalised mass matrix.

    Parameters
    ----------
    total_mass:
        Hull mass *plus* the mass of every moving part -- the paper's
        ``Mt``.  The moving masses are already inside it; the field only
        contributes the *offset* effects (coupling and added inertia).
    inertia_abs:
        Hull inertia tensor about ``G_h``, expressed in the absolute frame.
    mass, position:
        The moving-mass cloud in the absolute frame.
    added_mass_abs:
        Optional ``(6, 6)`` hydrodynamic added mass, already rotated into
        the absolute frame.  A hull accelerating through water accelerates
        water with it, and for a rowing shell the entrained water is not a
        correction: in sway and yaw it is comparable to the boat or
        larger.  Omitting it leaves the boat several times too easy to
        turn.  See :mod:`coxswain.hydro.addedmass`.

    Returns
    -------
    ``(6, 6)`` array acting on ``[Gh_ddot, omega_dot]``.
    """
    coupling = coupling_matrix(mass, position)

    matrix = np.zeros((6, 6))
    matrix[0:3, 0:3] = total_mass * np.eye(3)
    matrix[0:3, 3:6] = -coupling
    matrix[3:6, 0:3] = coupling
    matrix[3:6, 3:6] = inertia_abs + moving_mass_inertia(mass, position)
    if added_mass_abs is not None:
        matrix = matrix + np.asarray(added_mass_abs, dtype=float)
    return matrix


def moving_mass_reaction(field_abs: MovingMassField, omega: np.ndarray):
    """Reaction force and moment from prescribed motion of the crew.

    Everything here belongs on the *right-hand side*: these are the terms of
    the transport acceleration that do not multiply ``omega_dot`` (that part
    lives in the mass matrix).  For each particle the transport acceleration
    relative to the hull is

        a_t = a_rel + 2 omega x v_rel + omega x (omega x r)

    (Coriolis and centrifugal), and the reaction on the hull is ``-m a_t``
    with moment ``-m r x a_t`` about ``G_h``.

    Returns
    -------
    ``(force, moment)`` -- both ``(3,)`` arrays in the absolute frame.
    """
    mass = field_abs.mass
    if mass.size == 0:
        return np.zeros(3), np.zeros(3)

    r = field_abs.position
    v = field_abs.velocity
    a = field_abs.acceleration
    omega = np.asarray(omega, dtype=float)

    coriolis = 2.0 * cross3(omega, v)
    centrifugal = cross3(omega, cross3(omega, r))
    transport = a + coriolis + centrifugal

    force = -(mass[:, None] * transport).sum(axis=0)
    moment = -(mass[:, None] * cross3(r, transport)).sum(axis=0)
    return force, moment


def gyroscopic_moment(inertia_abs: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """``-omega x (I omega)`` -- the right-hand-side gyroscopic term."""
    omega = np.asarray(omega, dtype=float)
    return -cross3(omega, inertia_abs @ omega)


def solve_accelerations(mass_matrix: np.ndarray,
                        generalised_force: np.ndarray) -> np.ndarray:
    """Solve ``M a = f`` for ``[Gh_ddot, omega_dot]``.

    Raises
    ------
    ValueError
        If the mass matrix is singular or indefinite.  An indefinite mass
        matrix is never physical -- it means a sign error or a
        non-positive mass/inertia -- so failing loudly here is preferable
        to the legacy behaviour of silently returning zero acceleration.
    """
    try:
        cholesky = np.linalg.cholesky(mass_matrix)
    except np.linalg.LinAlgError as exc:
        eigenvalues = np.linalg.eigvalsh((mass_matrix + mass_matrix.T) / 2)
        raise ValueError(
            "generalised mass matrix is not positive definite "
            f"(eigenvalues {np.array2string(eigenvalues, precision=4)}); "
            "this indicates a sign error or a non-physical inertia, not a "
            "transient of the integration"
        ) from exc

    intermediate = np.linalg.solve(cholesky, generalised_force)
    return np.linalg.solve(cholesky.T, intermediate)
