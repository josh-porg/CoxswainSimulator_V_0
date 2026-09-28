r"""Blade loads integrated across the blade's span, not read at its centre.

Both blade laws in the project evaluate the water's velocity at one point, the blade centre
``l``, and apply the load there. But the blade spans ``[l - b/2, l + b/2]`` along the shaft,
and its normal velocity through the water varies linearly along it,

    w_n(r) = r phi_dot + v cos(phi),

because the rotation adds ``r phi_dot`` and the oarlock's motion is the same for every strip.
Every law here goes as the velocity squared, so the centre-point read is exact only when the
rotation term is negligible. Through the drive the two terms are both ~4 m/s and nearly cancel
at the centre, so the spread across the span is not small: the centre-point read understates
the load and puts it too far in. Integrating strip by strip is the same law, not a new one;
nothing here is fitted (SOURCES sec. 162).

Each strip of length ``dr`` carries the law's coefficient in proportion to its share of the
span, ``dr / b``: a rectangular blade. The one stated choice is that shape -- a Big Blade is
asymmetric -- and it is the natural default when no planform is sourced.

Returned are the total load and the centre of pressure ``r_cp = int r dF / int dF``, so a
caller can apply the load there and take its moment about the pin as ``r_cp F``.
"""
from __future__ import annotations

import numpy as np

#: Strips across the span. Trapezoid rule; a slip that changes sign part-way along the blade
#: puts a kink in the integrand, which needs more strips than a smooth one.
DEFAULT_STRIPS = 41


def _radii(centre: float, span: float, strips: int) -> np.ndarray:
    centre, span = float(centre), float(span)
    if span <= 0.0:
        raise ValueError("blade span must be positive, got %r" % (span,))
    if span >= 2.0 * centre:
        raise ValueError("a blade of span %.3f m cannot be centred %.3f m from the pin"
                         % (span, centre))
    if int(strips) < 3:
        raise ValueError("need at least 3 strips, got %r" % (strips,))
    return np.linspace(centre - 0.5 * span, centre + 0.5 * span, int(strips))


def _centre_of_pressure(r, dF, centre):
    total = float(np.trapezoid(dF, r))
    if total == 0.0:
        return 0.0, float(centre)
    return total, float(np.trapezoid(dF * r, r)) / total


def slip_strips(blade, angle: float, angular_rate: float, boat_speed: float, span: float,
                strips: int = DEFAULT_STRIPS):
    """``(F_n, r_cp)`` for the slip law (:class:`~coxswain.crew.oarlock.BladeModel`) over the span.

    Per strip, ``-sign(w) (C2 / b) w(r)^2 dr`` with ``w(r) = r phi_dot + v cos(phi)``, the
    law's own slip at radius ``r``. ``blade.outboard`` is the centre ``l``.
    """
    r = _radii(blade.outboard, span, strips)
    slip = r * float(angular_rate) + float(boat_speed) * np.cos(float(angle))
    dF = -np.sign(slip) * float(blade.c2) / float(span) * slip ** 2
    return _centre_of_pressure(r, dF, blade.outboard)


def liftdrag_strips(blade, angle: float, rate: float, lock_velocity, side: int, span: float,
                    strips: int = DEFAULT_STRIPS):
    """``(F_n, F_t, r_cp)`` for tier 2 (:class:`~coxswain.crew.liftdrag.LiftDragBlade`) over the span.

    Each strip resolves its own velocity through the water, so its own angle of attack; its
    area is ``A dr / b``. ``r_cp`` is the centre of the normal load, which is the one that
    turns the oar.
    """
    r = _radii(blade.outboard, span, strips)
    angle = float(angle)
    normal = np.array([np.cos(angle), -side * np.sin(angle)])
    axis = np.array([np.sin(angle), side * np.cos(angle)])
    u = np.asarray(lock_velocity, dtype=float)[:2]
    w_n = float(u @ normal) + r * float(rate)                # the rotation is along the normal
    w_a = np.full_like(r, float(u @ axis))
    speed2 = w_n ** 2 + w_a ** 2
    alpha = np.arctan2(np.abs(w_n), np.abs(w_a))
    s, c = np.sin(alpha), np.cos(alpha)
    lift = blade.lift_amplitude * 2.0 * s * c
    drag = blade.drag_amplitude * s * s
    c_n, c_t = lift * c + drag * s, drag * c - lift * s
    q = 0.5 * blade.density * blade.area / float(span) * speed2
    dF_n = -np.sign(w_n) * q * c_n
    dF_t = -np.sign(w_a) * q * c_t
    f_n, r_cp = _centre_of_pressure(r, dF_n, blade.outboard)
    return f_n, float(np.trapezoid(dF_t, r)), r_cp
