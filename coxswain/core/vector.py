"""Small-vector arithmetic, written out.

Why this exists
---------------
:func:`numpy.cross` spends most of its time deciding what it was asked.
Profiling one second of rowing at ``dt = 0.02``: **26,000 calls to
``np.cross`` costing 0.87 s**, of which 0.53 s was ``moveaxis`` and
0.33 s ``normalize_axis_tuple`` -- the axis bookkeeping that lets it
handle 2-vectors, stacked vectors and arbitrary axes.  The force model
never needs any of that.  It crosses a 3-vector with a 3-vector, or a
stack of them, and nothing else.

:func:`cross3` computes the same three expressions in the same order, so
for finite inputs it is **bit-identical** to ``np.cross`` -- the golden
trajectory in ``tests/data/golden_trajectory.npz`` is reproduced exactly.
It is a speed change, not a model change, and that is the whole point:
the physics is not slow because of its physics.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cross3", "clip"]


def cross3(a, b):
    """Cross product of 3-vectors, or of stacks of them.

    Accepts ``(3,)`` or ``(..., 3)`` on either side and broadcasts, which
    covers every call in the force path: ``lever x force``,
    ``arm x component``, ``centroid x panel_force`` over a hull mesh.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape == (3,) and b.shape == (3,):
        # The common case by a wide margin, and the one np.cross handles
        # worst: three multiplies and three subtracts, no dispatch.
        return np.array([a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]])
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([a1 * b2 - a2 * b1,
                     a2 * b0 - a0 * b2,
                     a0 * b1 - a1 * b0], axis=-1)


def clip(value: float, low: float, high: float) -> float:
    """Clamp a **scalar**, without going through ``np.clip``.

    ``np.clip`` on a Python float costs about 3.7 us of dispatch for
    something the CPU does in nanoseconds, and the force path called it
    98,024 times per second of rowing.  Identical for finite input; NaN
    propagates here where ``np.clip`` would also return NaN.
    """
    if value < low:
        return low
    return high if value > high else value
