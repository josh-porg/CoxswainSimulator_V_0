"""Second-order forward-mode automatic differentiation (2-jets).

A :class:`Jet2` carries a value together with its first and second
derivatives with respect to a single scalar (here, always time).
Arithmetic and the elementary functions propagate all three, so any
kinematic chain built out of ``Jet2`` yields position, velocity *and*
acceleration that are exactly consistent by construction.

Why this exists
---------------
The rigid-body equations need ``x``, ``x_dot`` and ``x_ddot`` for every
body segment.  Differentiating a multi-link kinematic chain by hand is
error-prone -- and silently so, because a wrong acceleration still
*looks* like a plausible force history.  The legacy code differentiated a
``tanh(cos(...))`` composition by hand; that particular chain rule was
actually correct, but nothing in the code could have told you if it were
not.  Building the kinematics on jets removes the failure mode entirely,
and :func:`tests.unit.test_taylor` checks the propagation rules against
finite differences.

Every operation is ``numpy``-aware, so a ``Jet2`` may hold arrays and
evaluate all body segments at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Jet2", "variable", "constant"]


@dataclass(frozen=True)
class Jet2:
    """Value plus first and second derivative with respect to time."""

    value: np.ndarray
    first: np.ndarray
    second: np.ndarray

    # -- construction ----------------------------------------------------
    @staticmethod
    def _coerce(other):
        if isinstance(other, Jet2):
            return other
        arr = np.asarray(other, dtype=float)
        return Jet2(arr, np.zeros_like(arr), np.zeros_like(arr))

    # -- arithmetic ------------------------------------------------------
    def __add__(self, other) -> "Jet2":
        o = self._coerce(other)
        return Jet2(self.value + o.value, self.first + o.first,
                    self.second + o.second)

    __radd__ = __add__

    def __neg__(self) -> "Jet2":
        return Jet2(-self.value, -self.first, -self.second)

    def __sub__(self, other) -> "Jet2":
        return self + (-self._coerce(other))

    def __rsub__(self, other) -> "Jet2":
        return self._coerce(other) + (-self)

    def __mul__(self, other) -> "Jet2":
        o = self._coerce(other)
        return Jet2(
            self.value * o.value,
            self.first * o.value + self.value * o.first,
            self.second * o.value + 2.0 * self.first * o.first
            + self.value * o.second,
        )

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Jet2":
        o = self._coerce(other)
        return self * o.reciprocal()

    def __rtruediv__(self, other) -> "Jet2":
        return self._coerce(other) * self.reciprocal()

    def reciprocal(self) -> "Jet2":
        inv = 1.0 / self.value
        first = -self.first * inv ** 2
        second = (2.0 * self.first ** 2 * inv ** 3 - self.second * inv ** 2)
        return Jet2(inv, first, second)

    def __pow__(self, exponent: float) -> "Jet2":
        n = float(exponent)
        base = self.value
        value = base ** n
        first = n * base ** (n - 1) * self.first
        second = (n * (n - 1) * base ** (n - 2) * self.first ** 2
                  + n * base ** (n - 1) * self.second)
        return Jet2(value, first, second)

    # -- elementary functions -------------------------------------------
    def sin(self) -> "Jet2":
        s, c = np.sin(self.value), np.cos(self.value)
        return Jet2(s, c * self.first,
                    -s * self.first ** 2 + c * self.second)

    def cos(self) -> "Jet2":
        s, c = np.sin(self.value), np.cos(self.value)
        return Jet2(c, -s * self.first,
                    -c * self.first ** 2 - s * self.second)

    def sqrt(self) -> "Jet2":
        return self ** 0.5

    def tanh(self) -> "Jet2":
        t = np.tanh(self.value)
        sech2 = 1.0 - t ** 2
        first = sech2 * self.first
        second = sech2 * self.second - 2.0 * t * sech2 * self.first ** 2
        return Jet2(t, first, second)

    def exp(self) -> "Jet2":
        e = np.exp(self.value)
        return Jet2(e, e * self.first, e * (self.first ** 2 + self.second))

    # -- convenience -----------------------------------------------------
    def as_tuple(self):
        return self.value, self.first, self.second

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Jet2(value={self.value!r}, first={self.first!r}, "
                f"second={self.second!r})")


def variable(value) -> Jet2:
    """The independent variable: ``d/dt = 1``, ``d2/dt2 = 0``."""
    arr = np.asarray(value, dtype=float)
    return Jet2(arr, np.ones_like(arr), np.zeros_like(arr))


def constant(value) -> Jet2:
    """A time-independent quantity."""
    arr = np.asarray(value, dtype=float)
    return Jet2(arr, np.zeros_like(arr), np.zeros_like(arr))
