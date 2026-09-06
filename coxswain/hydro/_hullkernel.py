r"""The wetted-surface sweep, as one pass over the panels.

:meth:`coxswain.hydro.hull.HullMesh.submerged` is 27% of a derivative
evaluation -- 357 us per call over an 880-panel mesh -- and it is not
slow because of arithmetic.  It is fifteen numpy operations, each
allocating an ``(880,)`` or ``(880, 4)`` temporary and each walking the
array again.  Fused into a single loop the same arithmetic runs in a
fraction of the time, and Numba compiles that loop.

Why this is opt-in
------------------
**Numba is not bit-identical to numpy, and cannot be.**  ``ndarray.sum``
uses pairwise summation; a loop accumulates left to right.  The two agree
to round-off and disagree in the last bits, which is fine for a boat and
fatal for a test that asserts a trajectory has not moved.

So the numpy path stays exactly as it was and remains the default: the
studies stay bit-reproducible and ``tests/test_stepwise.py`` keeps
asserting exact equality.  The compiled path is asked for explicitly, by
the real-time loop, which cares about the frame budget and not about the
last bit.  ``tests/test_kernels.py`` holds the two to a tight tolerance
over a whole trajectory, so they cannot quietly drift apart -- the one
failure this project cannot have is a trainer that teaches a different
boat from the one the analysis describes.

Numba is an optional dependency.  Without it :data:`HAVE_NUMBA` is false,
:func:`submerged_kernel` is the plain-Python reference, and everything
still runs -- slower, and correct.
"""

from __future__ import annotations

import numpy as np

__all__ = ["HAVE_NUMBA", "submerged_kernel"]

try:                                            # pragma: no cover
    from numba import njit

    HAVE_NUMBA = True
except Exception:                               # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """No-op stand-in so the module imports without Numba."""
        def wrap(function):
            return function
        return wrap(args[0]) if args and callable(args[0]) else wrap


def _submerged(corners, centroid, normal, area, rot, position_z,
               rho, gravity, water_level):
    """One pass over the panels.

    Returns a flat tuple rather than a record so the compiled and
    interpreted paths agree on shape without Numba needing to know about
    a dataclass::

        (force(3), moment(3), volume, wetted, transverse, lateral,
         plan, centre_z, total_wet_weight)

    The rotation is applied here rather than by three matrix products
    outside, which is three fewer ``(880, 3)`` temporaries.
    """
    n = area.shape[0]
    force = np.zeros(3)
    moment = np.zeros(3)
    volume = 0.0
    wetted = 0.0
    transverse = 0.0
    lateral = 0.0
    plan = 0.0
    centre_z_weighted = 0.0
    weight_total = 0.0

    for i in range(n):
        # normal, rotated into the absolute frame
        nx = (rot[0, 0] * normal[i, 0] + rot[0, 1] * normal[i, 1]
              + rot[0, 2] * normal[i, 2])
        ny = (rot[1, 0] * normal[i, 0] + rot[1, 1] * normal[i, 1]
              + rot[1, 2] * normal[i, 2])
        nz = (rot[2, 0] * normal[i, 0] + rot[2, 1] * normal[i, 1]
              + rot[2, 2] * normal[i, 2])

        # corner depths below the still-water plane
        raw_sum = 0.0
        raw_min = 1.0e300
        raw_max = -1.0e300
        depth_sum = 0.0
        for k in range(4):
            cz = (rot[2, 0] * corners[i, k, 0] + rot[2, 1] * corners[i, k, 1]
                  + rot[2, 2] * corners[i, k, 2]) + position_z
            raw = water_level - cz
            raw_sum += raw
            if raw < raw_min:
                raw_min = raw
            if raw > raw_max:
                raw_max = raw
            depth_sum += raw if raw > 0.0 else 0.0
        mean_depth = depth_sum * 0.25
        raw_mean = raw_sum * 0.25

        spread = raw_max - raw_min
        if spread > 1e-12:
            fraction = raw_mean / spread + 0.5
            if fraction < 0.0:
                fraction = 0.0
            elif fraction > 1.0:
                fraction = 1.0
        else:
            fraction = 1.0 if raw_mean > 0.0 else 0.0

        panel_area = area[i]
        wet_area = panel_area * fraction

        # hydrostatic force: dF = -rho g q n dsigma
        scale = -(rho * gravity * mean_depth * panel_area)
        fx = scale * nx
        fy = scale * ny
        fz = scale * nz
        force[0] += fx
        force[1] += fy
        force[2] += fz

        # centroid, rotated, and its moment
        cx = (rot[0, 0] * centroid[i, 0] + rot[0, 1] * centroid[i, 1]
              + rot[0, 2] * centroid[i, 2])
        cy = (rot[1, 0] * centroid[i, 0] + rot[1, 1] * centroid[i, 1]
              + rot[1, 2] * centroid[i, 2])
        cz3 = (rot[2, 0] * centroid[i, 0] + rot[2, 1] * centroid[i, 1]
               + rot[2, 2] * centroid[i, 2])
        moment[0] += cy * fz - cz3 * fy
        moment[1] += cz3 * fx - cx * fz
        moment[2] += cx * fy - cy * fx

        volume -= mean_depth * panel_area * nz

        wetted += wet_area
        # Projections use the HULL-frame normal, not the rotated one.
        transverse += wet_area * abs(normal[i, 0])
        lateral += wet_area * abs(normal[i, 1])
        plan += wet_area * normal[i, 2]

        centre_z_weighted += wet_area * cz3
        weight_total += wet_area

    return (force, moment, volume, wetted, 0.5 * transverse,
            0.5 * lateral, abs(plan), centre_z_weighted, weight_total)


submerged_kernel = njit(cache=True, fastmath=False)(_submerged)
#: The uncompiled reference, kept for testing the compiled one against.
submerged_reference = _submerged
