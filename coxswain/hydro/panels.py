r"""Potential flow around the real hull, by source panels.

:class:`~coxswain.hydro.vortex.ThinBody` put sources on the centreline with
``sigma(x) = 2 U db/dx``.  That is thin-body theory: it assumes the hull
is a small perturbation to a uniform stream, places the singularities on
the axis rather than on the surface, and never actually enforces that
water does not flow through the boat.  For a shell of 30:1 fineness it is
a decent approximation, and "decent approximation" is exactly the thing
worth replacing when the geometry is already in hand.

This solves it properly, by the Hess-Smith constant-strength source panel
method [HS67]_:

1. Wrap the waterline in ``N`` flat panels.
2. Put an unknown constant source strength on each.
3. Require the normal velocity at every panel's control point to vanish
   -- the boundary condition thin-body theory only approximates.
4. Solve the resulting dense ``N x N`` system.

The influence coefficients are the textbook ones [KP01]_.  For a constant
source panel of unit strength lying along its own x axis from 0 to ``L``,
at a point ``(x, z)`` in panel coordinates:

.. math::

    u = \frac{1}{4\pi} \ln\frac{x^2 + z^2}{(x-L)^2 + z^2}, \qquad
    w = \frac{1}{2\pi}\left[\arctan\frac{z}{x-L}
                          - \arctan\frac{z}{x}\right]

with the self-induced normal velocity of a panel on itself equal to
``sigma / 2``, which is the jump across a source sheet and the reason the
matrix has a well-conditioned diagonal.

Three checks it has to pass
---------------------------
**A circle.**  Flow past a cylinder has the exact surface speed
``2 U sin(theta)``.  A panel method that cannot reproduce that is wrong,
and it is the cheapest possible way to find out.

**Closure.**  The source strengths of a closed body in a uniform stream
must sum to zero, weighted by panel length: a body that is not growing
cannot be a net source of water.

**The boundary condition itself.**  The residual normal velocity on the
surface should be at machine precision, because that is literally the
equation that was solved.

References
----------
.. [HS67] Hess, J. L. and Smith, A. M. O. (1967) *Calculation of
   potential flow about arbitrary bodies*, Progress in Aerospace
   Sciences 8, 1-138.
.. [KP01] Katz, J. and Plotkin, A. (2001) *Low-Speed Aerodynamics*,
   2nd ed., ch. 10 -- the constant-strength source panel influence
   coefficients used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SourcePanelBody", "waterline_from_offsets", "circle_nodes"]


def circle_nodes(radius: float = 1.0, panels: int = 120,
                 centre=(0.0, 0.0)) -> np.ndarray:
    """Closed polygon on a circle -- the validation case."""
    angle = np.linspace(0.0, 2.0 * np.pi, panels + 1)
    return np.column_stack([centre[0] + radius * np.cos(angle),
                            centre[1] + radius * np.sin(angle)])


def waterline_from_offsets(offsets, panels: int = 160,
                           tip: float = 0.004) -> np.ndarray:
    """Closed waterline polygon from a boat's own offsets.

    The hull is symmetric about its centreline, so the waterline is the
    half-beam curve mirrored: down one side from bow to stern and back up
    the other.  Cosine spacing clusters panels at the ends, where the
    curvature is and where uniform spacing would smear the stagnation
    points into a shape the solver then works hard to satisfy.
    """
    x = np.asarray(offsets.station, dtype=float)
    half_beam = 0.5 * np.asarray(offsets.beam, dtype=float)
    order = np.argsort(x)
    x, half_beam = x[order], half_beam[order]

    count = max(panels // 2, 8)
    beta = np.linspace(0.0, np.pi, count + 1)
    fine = x.min() + 0.5 * (x.max() - x.min()) * (1.0 - np.cos(beta))
    beam = np.interp(fine, x, half_beam)
    # A true cusp -- zero half-beam at both ends -- makes the upper and
    # lower panels coincide there.  Two coincident panels with opposite
    # normals give the influence matrix a near-null direction, and the
    # solution does not converge: refining 50 -> 400 panels DOUBLED the
    # peak surface speed each time instead of settling.  A real bow has a
    # few millimetres of radius; giving it that makes the body a proper
    # closed curve and the divergence goes away.
    beam[0] = beam[-1] = max(float(tip), 1e-4)

    upper = np.column_stack([fine, beam])
    lower = np.column_stack([fine[::-1], -beam[::-1]])
    nodes = np.vstack([upper, lower[1:]])
    # Drop any duplicated node so no panel has zero length.
    keep = np.concatenate([[True],
                           np.hypot(*np.diff(nodes, axis=0).T) > 1e-9])
    return nodes[keep]


@dataclass
class SourcePanelBody:
    """A closed body panelled with constant-strength sources."""

    nodes: np.ndarray                      # (N+1, 2), closed
    strength: np.ndarray = field(default=None)
    speed: float = 0.0

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)
        if np.hypot(*(self.nodes[0] - self.nodes[-1])) > 1e-9:
            self.nodes = np.vstack([self.nodes, self.nodes[:1]])
        # The normal below assumes counter-clockwise nodes.  Given a
        # clockwise polygon it points INWARD, the boundary condition gets
        # solved with the wrong sign, and the answer is not merely
        # inaccurate -- the hull came out with a peak surface speed of
        # 56 m/s in a 4.2 m/s stream.  The signed area says which way
        # round the polygon is, so ask rather than assume.
        area = 0.5 * float(np.sum(self.nodes[:-1, 0] * self.nodes[1:, 1]
                                  - self.nodes[1:, 0] * self.nodes[:-1, 1]))
        if area < 0.0:
            self.nodes = self.nodes[::-1]
        self.signed_area = abs(area)
        start, end = self.nodes[:-1], self.nodes[1:]
        delta = end - start
        self.length = np.hypot(delta[:, 0], delta[:, 1])
        self.angle = np.arctan2(delta[:, 1], delta[:, 0])
        self.control = 0.5 * (start + end)
        # Outward normal, given nodes run counter-clockwise.
        self.normal = np.column_stack([np.sin(self.angle),
                                       -np.cos(self.angle)])
        self.tangent = np.column_stack([np.cos(self.angle),
                                        np.sin(self.angle)])

    @property
    def n_panels(self) -> int:
        return len(self.length)

    # -- influence --------------------------------------------------------
    def _influence(self, points) -> np.ndarray:
        """Velocity at ``points`` per unit source strength, ``(m, n, 2)``."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        start = self.nodes[:-1]
        cos, sin = np.cos(self.angle), np.sin(self.angle)

        offset = points[:, None, :] - start[None, :, :]
        # Rotate into each panel's own frame.
        local_x = offset[:, :, 0] * cos[None, :] + offset[:, :, 1] * sin[None, :]
        local_z = -offset[:, :, 0] * sin[None, :] + offset[:, :, 1] * cos[None, :]

        r1 = local_x ** 2 + local_z ** 2
        r2 = (local_x - self.length[None, :]) ** 2 + local_z ** 2
        u = np.log(np.maximum(r1, 1e-300) / np.maximum(r2, 1e-300)) / (4.0 * np.pi)
        w = (np.arctan2(local_z, local_x - self.length[None, :])
             - np.arctan2(local_z, local_x)) / (2.0 * np.pi)

        return np.stack([u * cos[None, :] - w * sin[None, :],
                         u * sin[None, :] + w * cos[None, :]], axis=2)

    def solve(self, speed: float) -> "SourcePanelBody":
        """Find the source strengths that stop flow through the hull.

        ``speed`` is the boat's speed; in the body frame the stream runs
        the other way, which is the sign the boundary condition needs.
        """
        self.speed = float(speed)
        influence = self._influence(self.control)
        matrix = np.einsum("ijk,ik->ij", influence, self.normal)
        # A panel's own normal influence is the source-sheet jump, not the
        # numerically indeterminate self-integral above.
        np.fill_diagonal(matrix, 0.5)
        stream = np.array([-self.speed, 0.0])
        rhs = -self.normal @ stream
        self.strength = np.linalg.solve(matrix, rhs)
        return self

    # -- the field --------------------------------------------------------
    def velocity_at(self, points, freestream: bool = False) -> np.ndarray:
        """Induced velocity, in the water's frame by default.

        With ``freestream=True`` the uniform stream is added, which is the
        body-frame flow the boundary condition and the surface speed are
        expressed in.  Without it the result is the disturbance the hull
        leaves in otherwise still water -- which is what a following crew
        actually sits in, and what the vortex field must be added to.
        """
        if self.strength is None:
            raise RuntimeError("call solve() before asking for a field")
        induced = np.einsum("ijk,j->ik", self._influence(points),
                            self.strength)
        if freestream:
            induced = induced + np.array([-self.speed, 0.0])
        return induced

    # -- what has to be true ----------------------------------------------
    #: How far off the surface the diagnostics sample, as a fraction of
    #: the local panel length.  Velocity is discontinuous ACROSS a source
    #: sheet: evaluated exactly on it the influence formula returns the
    #: inner-side limit, so the residual check came back at 2.0 m/s --
    #: precisely the strength of the jump -- on a solution whose surface
    #: speed was already exact to fifteen figures.  The boundary
    #: condition lives on the outside, so sample there.
    OFFSET = 1e-3

    def _just_outside(self) -> np.ndarray:
        return self.control + (self.OFFSET * self.length)[:, None] * self.normal

    def closure(self) -> float:
        """``sum(sigma_i L_i)``, which must vanish for a closed body."""
        return float(np.sum(self.strength * self.length))

    def normal_residual(self) -> float:
        """Largest leftover flow through the surface, m/s."""
        velocity = self.velocity_at(self._just_outside(), freestream=True)
        return float(np.abs(np.einsum("ij,ij->i", velocity,
                                      self.normal)).max())

    def surface_speed(self) -> np.ndarray:
        """Tangential speed at each control point, m/s."""
        velocity = self.velocity_at(self._just_outside(), freestream=True)
        return np.einsum("ij,ij->i", velocity, self.tangent)

    def pressure_coefficient(self) -> np.ndarray:
        """``1 - (q/U)^2`` along the surface."""
        return 1.0 - (self.surface_speed() / max(self.speed, 1e-9)) ** 2
