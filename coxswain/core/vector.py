"""Small-vector arithmetic, written out.

Why this exists
---------------
The cross product lives in :func:`coxswain.core.frames.cross3`, which was
already here; this module holds the scalar clamp.

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

__all__ = ["clip"]


def clip(value: float, low: float, high: float) -> float:
    """Clamp a **scalar**, without going through ``np.clip``.

    ``np.clip`` on a Python float costs about 3.7 us of dispatch for
    something the CPU does in nanoseconds, and the force path called it
    98,024 times per second of rowing.  Identical for finite input.

    Anything that is not a scalar is handed straight to ``np.clip``, so
    swapping a call site can never change what it means -- several of
    these are scalar in the force path and arrays in the analysis code
    that plots the same quantity over a time base.
    """
    if not isinstance(value, (float, int)):
        return np.clip(value, low, high)
    if value < low:
        return low
    return high if value > high else value
