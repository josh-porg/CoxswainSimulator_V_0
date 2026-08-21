"""A differentiable stand-in for the wetted-surface integral.

The one part of the 6-DOF model that an optimiser cannot differentiate is
:meth:`~coxswain.hydro.hull.HullMesh.submerged`, which clips a triangulated
hull against the waterline and integrates over what is left.  Clipping is a
branch per triangle; there is no derivative to take.

That single obstacle is what earlier versions of the stroke-resolved model
used to justify dropping heave, pitch and roll.  It does not justify it.
The integral depends on exactly **three** variables -- heave, pitch and
roll -- so it can be evaluated exactly on a grid and interpolated, which
gives a smooth function of the same three variables with a measurable
error.  That is a bounded smoothing of the real computation, not a
reduction of the model: no physics is dropped, and the error is reported
rather than assumed.

What is tabulated
-----------------
Everything the dynamics needs from the hull's immersion:

``volume``              buoyancy magnitude
``centre_of_buoyancy``  where it acts, all three components -- this is
                        what produces the roll and pitch restoring moments
``wetted_area``         viscous drag
``transverse_area``     shape drag
``plan_area``           wave drag and vertical cross-flow
``lateral_area``        lateral cross-flow

Accuracy
--------
:meth:`HullSurrogate.validate` re-evaluates the exact mesh at points that
are *not* grid nodes and reports the worst relative error in each
quantity.  A surrogate that has not been checked off-node has not been
checked at all -- interpolants are exact at their own nodes by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

__all__ = ["TABULATED", "HullSurrogate"]

#: Scalars taken from :class:`SubmergedProperties`, in the order they are
#: stacked in the interpolant.
TABULATED = (
    "volume",
    "wetted_area",
    "transverse_area",
    "plan_area",
    "lateral_area",
    "buoyancy_x",
    "buoyancy_y",
    "buoyancy_z",
)



def _to_hull(centre, roll, pitch):
    """Express a mesh-frame centre of buoyancy in the hull frame.

    :meth:`HullMesh.submerged` returns the centre of buoyancy in the
    **absolute** frame -- its own ``buoyancy_moment`` is ``cross(centre,
    force)`` with an unrotated vertical force, which only balances if the
    centre already carries the attitude.

    The dynamics need it in the hull frame, because they rotate it by the
    full attitude (yaw included) before crossing it with gravity.  Storing
    the absolute value and rotating it again applies roll and pitch twice.
    At one degree of heel that turned an 18.3 N m righting moment into
    2.4 N m -- it destroyed 87% of the hull's roll stiffness, and it was
    invisible at zero roll, which is where every previous check looked.
    """
    from ..core.frames import hull_to_abs

    rotation = hull_to_abs(np.array([roll, pitch, 0.0]))
    return rotation.T @ np.asarray(centre, dtype=float)


@dataclass
class HullSurrogate:
    """Submerged hull properties as a smooth function of attitude.

    Built by sampling the exact mesh computation on a regular grid in
    ``(heave, pitch, roll)`` and interpolating.  Query it exactly like the
    mesh, but with CasADi expressions.
    """

    heave: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    table: np.ndarray            # (n_quantities, nz, ntheta, nphi)
    boat: object = None
    _interpolant: object = field(default=None, repr=False)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_boat(cls, boat, heave_range: Tuple[float, float] = (-0.16, 0.06),
                  pitch_range: float = np.radians(4.0),
                  roll_range: float = np.radians(12.0),
                  n_heave: int = 23, n_pitch: int = 13, n_roll: int = 13,
                  water_level: float = 0.0, gravity: float = 9.80665):
        """Sample the exact mesh over the range the boat actually visits.

        The default ranges are generous against measured motion -- an eight
        heaves a few centimetres, pitches under a degree and rolls a degree
        or two -- because an optimiser will probe outside the physical
        envelope and the surrogate has to stay sane when it does.
        """
        heave = np.linspace(heave_range[0], heave_range[1], n_heave)
        pitch = np.linspace(-pitch_range, pitch_range, n_pitch)
        roll = np.linspace(-roll_range, roll_range, n_roll)

        table = np.empty((len(TABULATED), n_heave, n_pitch, n_roll))
        for i, z in enumerate(heave):
            for j, theta in enumerate(pitch):
                for k, phi in enumerate(roll):
                    properties = boat.mesh.submerged(
                        np.array([0.0, 0.0, z]),
                        np.array([phi, theta, 0.0]),
                        rho=boat.water.density, gravity=gravity,
                        water_level=water_level)
                    centre = _to_hull(properties.centre_of_buoyancy,
                                      phi, theta)
                    table[:, i, j, k] = (
                        properties.volume,
                        properties.wetted_area,
                        properties.transverse_area,
                        properties.plan_area,
                        properties.lateral_area,
                        centre[0], centre[1], centre[2],
                    )
        return cls(heave=heave, pitch=pitch, roll=roll, table=table,
                   boat=boat)

    # -- evaluation -------------------------------------------------------
    def _build(self):
        import casadi as ca

        if self._interpolant is None:
            grid = [self.heave.tolist(), self.pitch.tolist(),
                    self.roll.tolist()]
            # one interpolant per quantity keeps each output independent and
            # lets CasADi prune whichever the expression does not use
            self._interpolant = {
                name: ca.interpolant(
                    f"hull_{name}", "bspline", grid,
                    self.table[index].ravel(order="F").tolist())
                for index, name in enumerate(TABULATED)
            }
        return self._interpolant

    def casadi(self, heave, pitch, roll) -> Dict[str, object]:
        """Every tabulated quantity, as CasADi expressions."""
        import casadi as ca

        point = ca.vertcat(heave, pitch, roll)
        return {name: lookup(point)
                for name, lookup in self._build().items()}

    def __call__(self, heave, pitch, roll) -> Dict[str, float]:
        """Every tabulated quantity, numerically."""
        import casadi as ca

        return {name: float(ca.DM(value))
                for name, value in self.casadi(heave, pitch, roll).items()}

    # -- accuracy ---------------------------------------------------------
    def validate(self, n_samples: int = 200, seed: int = 0,
                 gravity: float = 9.80665, water_level: float = 0.0):
        """Worst relative error against the exact mesh, **off** the nodes.

        Sampling at grid nodes would report zero and mean nothing: a
        bspline reproduces its own knots exactly.  Points here are drawn
        uniformly at random inside the sampled box.
        """
        rng = np.random.default_rng(seed)
        heave = rng.uniform(self.heave[0], self.heave[-1], n_samples)
        pitch = rng.uniform(self.pitch[0], self.pitch[-1], n_samples)
        roll = rng.uniform(self.roll[0], self.roll[-1], n_samples)

        worst = {name: 0.0 for name in TABULATED}
        scale = {name: max(abs(self.table[i]).max(), 1e-9)
                 for i, name in enumerate(TABULATED)}

        for z, theta, phi in zip(heave, pitch, roll):
            exact = self.boat.mesh.submerged(
                np.array([0.0, 0.0, z]), np.array([phi, theta, 0.0]),
                rho=self.boat.water.density, gravity=gravity,
                water_level=water_level)
            centre = _to_hull(exact.centre_of_buoyancy, phi, theta)
            truth = {
                "volume": exact.volume,
                "wetted_area": exact.wetted_area,
                "transverse_area": exact.transverse_area,
                "plan_area": exact.plan_area,
                "lateral_area": exact.lateral_area,
                "buoyancy_x": centre[0],
                "buoyancy_y": centre[1],
                "buoyancy_z": centre[2],
            }
            got = self(z, theta, phi)
            for name in TABULATED:
                error = abs(got[name] - truth[name]) / scale[name]
                worst[name] = max(worst[name], error)
        return worst

    @property
    def n_points(self) -> int:
        return len(self.heave) * len(self.pitch) * len(self.roll)
