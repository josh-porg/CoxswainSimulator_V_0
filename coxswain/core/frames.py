"""Reference frames, rotations and attitude kinematics.

Frame conventions
-----------------
Two frames are used, following Formaggia et al. (2009) section 3:

``abs``  -- inertial frame ``(O; X, Y, Z)`` fixed to the race course.
            ``X`` horizontal along the direction of progression, ``Z``
            vertical pointing *up*, ``Y = Z x X``.

``hull`` -- body frame ``(G_h; x, y, z)`` centred on the *hull* centre of
            mass (not the combined hull+crew centre of mass, which moves).
            ``x`` points stern->bow, ``z`` points bottom->top, ``y = z x x``
            (i.e. to port).

The single most important convention in this module, and the one that the
original code got wrong in three separate places, is the *direction* of the
rotation matrix.  Every function here names its direction explicitly:

    v_abs = hull_to_abs(att) @ v_hull
    v_hull = abs_to_hull(att) @ v_abs

There is deliberately no bare function called ``rotation_matrix``.

Attitude representation
-----------------------
Attitude is a length-3 array of intrinsic Z-Y-X (yaw-pitch-roll) Euler
angles, stored in the order ``[roll, pitch, yaw]``.  The index constants
:data:`ROLL`, :data:`PITCH` and :data:`YAW` are exported and should be used
instead of literal integers -- the legacy code silently disagreed with
itself about whether index 0 meant roll or yaw, which swapped the yaw and
roll derivatives.

Angular velocity
----------------
The dynamics in :mod:`coxswain.core.rigid_body` are written in the absolute
frame, matching the paper, so ``omega`` in the state vector is the angular
velocity **expressed in the absolute frame**.  Euler-angle kinematics
require *body* rates, so :func:`euler_rates` converts internally.  Do not
feed body rates to :func:`euler_rates`; use :func:`euler_rates_from_body`
if that is what you have.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ROLL",
    "PITCH",
    "YAW",
    "GIMBAL_LOCK_TOL",
    "hull_to_abs",
    "abs_to_hull",
    "cross3",
    "skew",
    "unskew",
    "euler_rates",
    "euler_rates_from_body",
    "body_rates_from_euler_rates",
    "attitude_from_components",
    "wrap_to_pi",
    "rotate_inertia_to_abs",
]

ROLL, PITCH, YAW = 0, 1, 2

#: ``cos(pitch)`` below this magnitude is treated as gimbal lock.  A rowing
#: shell pitches by ~1 degree, so reaching this is a symptom of divergence.
GIMBAL_LOCK_TOL = 1e-6


def attitude_from_components(roll: float = 0.0, pitch: float = 0.0,
                             yaw: float = 0.0) -> np.ndarray:
    """Build an attitude vector from named angles (radians)."""
    return np.array([roll, pitch, yaw], dtype=float)


def hull_to_abs(attitude: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping hull-frame vectors to the absolute frame.

    This is ``R_z(yaw) @ R_y(pitch) @ R_x(roll)`` -- the intrinsic Z-Y-X
    sequence.  Equivalent to ``R^T(theta)`` in Formaggia et al. eq. (2)-(5),
    and to the matrix written ``R(theta)`` in their eq. (8) and (14) (the
    paper switches convention between those sections).
    """
    phi, theta, psi = (float(attitude[ROLL]), float(attitude[PITCH]),
                       float(attitude[YAW]))
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)

    return np.array([
        [cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps],
        [cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps],
        [-sth,      sph * cth,                   cph * cth],
    ])


def abs_to_hull(attitude: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping absolute-frame vectors to the hull frame."""
    return hull_to_abs(attitude).T


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of vectors whose last axis has length 3.

    ``numpy.cross`` spends most of its time in axis-normalisation
    machinery that is pure overhead for a fixed 3-vector, and the
    derivative evaluation calls it ~17 times per step.  Replacing it is
    worth about a quarter of the simulation runtime.

    Broadcasts like any ufunc expression, so ``(3,) x (n, 3)`` works.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([a1 * b2 - a2 * b1,
                     a2 * b0 - a0 * b2,
                     a0 * b1 - a1 * b0], axis=-1)


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix ``S(v)`` such that ``S(v) @ w == cross(v, w)``."""
    v = np.asarray(v, dtype=float)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def unskew(matrix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew`: recover the vector from a skew matrix."""
    m = np.asarray(matrix, dtype=float)
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def rotate_inertia_to_abs(inertia_hull: np.ndarray,
                          attitude: np.ndarray) -> np.ndarray:
    """Express a hull-frame inertia tensor in the absolute frame.

    For a second-rank tensor the correct congruence is
    ``R_{hull->abs} I R_{hull->abs}^T``.  Using the opposite direction (as
    the legacy code did) happens to be harmless only when the tensor is
    isotropic in the plane of rotation, which hid the bug.
    """
    rot = hull_to_abs(attitude)
    return rot @ np.asarray(inertia_hull, dtype=float) @ rot.T


def euler_rates(attitude: np.ndarray, omega_abs: np.ndarray) -> np.ndarray:
    """Euler-angle derivatives from *absolute-frame* angular velocity.

    Returns ``[roll_dot, pitch_dot, yaw_dot]`` in the same index order as
    ``attitude``.
    """
    omega_body = abs_to_hull(attitude) @ np.asarray(omega_abs, dtype=float)
    return euler_rates_from_body(attitude, omega_body)


def euler_rates_from_body(attitude: np.ndarray,
                          omega_body: np.ndarray) -> np.ndarray:
    """Euler-angle derivatives from *body-frame* rates ``(p, q, r)``.

    ``roll_dot  = p + (q sin(phi) + r cos(phi)) tan(theta)``
    ``pitch_dot = q cos(phi) - r sin(phi)``
    ``yaw_dot   = (q sin(phi) + r cos(phi)) / cos(theta)``
    """
    phi, theta = float(attitude[ROLL]), float(attitude[PITCH])
    p, q, r = (float(omega_body[0]), float(omega_body[1]),
               float(omega_body[2]))

    cos_theta = np.cos(theta)
    if abs(cos_theta) < GIMBAL_LOCK_TOL:
        raise ValueError(
            f"gimbal lock: pitch={np.degrees(theta):.3f} deg is at +-90 deg. "
            "A rowing shell pitches by ~1 deg, so this means the integration "
            "has diverged."
        )

    sph, cph = np.sin(phi), np.cos(phi)
    common = q * sph + r * cph

    return np.array([
        p + common * np.tan(theta),
        q * cph - r * sph,
        common / cos_theta,
    ])


def body_rates_from_euler_rates(attitude: np.ndarray,
                                euler_rate: np.ndarray) -> np.ndarray:
    """Body-frame rates ``(p, q, r)`` from Euler-angle derivatives.

    Exact inverse of :func:`euler_rates_from_body`.
    """
    phi, theta = float(attitude[ROLL]), float(attitude[PITCH])
    dphi, dtheta, dpsi = (float(euler_rate[ROLL]), float(euler_rate[PITCH]),
                          float(euler_rate[YAW]))

    sph, cph = np.sin(phi), np.cos(phi)
    sth, cth = np.sin(theta), np.cos(theta)

    return np.array([
        dphi - dpsi * sth,
        dtheta * cph + dpsi * cth * sph,
        -dtheta * sph + dpsi * cth * cph,
    ])


def wrap_to_pi(angle):
    """Wrap an angle (or array of angles) to ``(-pi, pi]``."""
    return np.arctan2(np.sin(angle), np.cos(angle))
