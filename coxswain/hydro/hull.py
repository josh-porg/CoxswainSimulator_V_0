"""Hull geometry and the submerged-surface integrals the dynamics need.

Formaggia et al. section 6.1 do not assume a fixed wetted surface: they
triangulate the hull, define the submersion depth

    q(x; G_z, theta) = max(0, h0 + x sin(theta) - z cos(theta) - G_z)

and integrate over the mesh at every time step to get the hydrostatic
force, the hydrostatic moment, and the three surface measures that the
resistance formulas need -- the wetted area ``|Gamma|`` and its
projections ``|Gamma_X|`` and ``|Gamma_Z|``.

That matters here because the legacy code replaced all three with fixed
closed-form guesses, and two of them were badly wrong: ``gamma_x`` used
the hull's *side* projection (length x draft) where the transverse
projection (beam x draft) was wanted, roughly 23 times too large, and
``gamma_z`` evaluated to a girth rather than an area, about 4 times too
small.  Computing them from the mesh removes the whole class of error and
is what makes different hull shapes genuinely swappable.

Deviation from the paper
------------------------
The paper writes the area measures as ``integral of q dsigma``, weighting
by the submersion *depth*.  That has units of volume, not area, and would
make the resistance scale with draft squared.  It is a slip in the paper;
the intended quantity is plainly the area of the submerged part, so that
is what is computed here.  The hydrostatic force and moment do use the
depth weighting, where it is correct (it is the pressure).

Panel geometry
--------------
Sections are semi-ellipses below the design waterline -- a good fit for a
racing shell -- continued by straight topsides up to the freeboard.  A
hull is therefore fully described by a table of waterline beam and keel
depth against longitudinal station, which is exactly what a real offsets
table provides.  :func:`parametric_offsets` generates a plausible table
from four numbers when no measured one is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.frames import hull_to_abs

__all__ = [
    "HullOffsets",
    "HullMesh",
    "SubmergedProperties",
    "parametric_offsets",
]


@dataclass(frozen=True)
class HullOffsets:
    """Waterline beam and keel depth against longitudinal station.

    ``station`` runs stern to bow in the hull frame.  ``beam`` is the full
    (port to starboard) width at the design waterline; ``depth`` is how far
    the keel sits below it.  ``freeboard`` is the height of the gunwale
    above the design waterline and bounds how much hull can ever get wet.
    """

    station: np.ndarray
    beam: np.ndarray
    depth: np.ndarray
    freeboard: float = 0.25

    def __post_init__(self) -> None:
        n = len(self.station)
        if len(self.beam) != n or len(self.depth) != n:
            raise ValueError("station, beam and depth must have equal length")
        if n < 3:
            raise ValueError("need at least 3 stations")
        if np.any(np.diff(self.station) <= 0):
            raise ValueError("stations must be strictly increasing")
        if np.any(self.beam < 0) or np.any(self.depth < 0):
            raise ValueError("beam and depth must be non-negative")
        if self.freeboard <= 0:
            raise ValueError("freeboard must be positive")

    @property
    def length(self) -> float:
        return float(self.station[-1] - self.station[0])

    @property
    def max_beam(self) -> float:
        return float(self.beam.max())

    @property
    def max_depth(self) -> float:
        return float(self.depth.max())

    def design_volume(self) -> float:
        """Displaced volume at the design waterline, in m^3.

        Semi-elliptical sections have area ``pi B D / 4``.
        """
        section_area = 0.25 * np.pi * self.beam * self.depth
        return float(np.trapezoid(section_area, self.station))

    def design_displacement(self, rho: float = 1025.0) -> float:
        """Mass the hull displaces at the design waterline, in kg."""
        return rho * self.design_volume()


def parametric_offsets(length: float, max_beam: float, max_depth: float,
                       n_stations: int = 41, fullness: float = 2.2,
                       freeboard: float = 0.25,
                       centre_fraction: float = 0.5) -> HullOffsets:
    """Generate a plausible racing-shell offsets table.

    ``fullness`` controls how much of the length carries close to the
    maximum beam: 2 gives an ellipse, larger values give the long parallel
    midbody of a shell.  ``centre_fraction`` places the maximum beam along
    the length (0.5 = amidships).
    """
    if not 1.0 < fullness <= 12.0:
        raise ValueError("fullness must lie in (1, 12]")

    station = np.linspace(-length / 2.0, length / 2.0, n_stations)
    centre = (centre_fraction - 0.5) * length
    # normalised distance from the station of maximum beam, in [-1, 1]
    half = length / 2.0
    reduced = np.clip((station - centre) / half, -1.0, 1.0)

    shape = (1.0 - np.abs(reduced) ** fullness) ** 0.5
    return HullOffsets(
        station=station,
        beam=max_beam * shape,
        depth=max_depth * shape ** 0.5,
        freeboard=freeboard,
    )


@dataclass(frozen=True)
class SubmergedProperties:
    """Everything the force models need about the current wetted surface."""

    wetted_area: float          # |Gamma|,   m^2
    transverse_area: float      # |Gamma_X|, m^2
    lateral_area: float         # |Gamma_Y|, m^2
    plan_area: float            # |Gamma_Z|, m^2
    volume: float               # displaced volume, m^3
    buoyancy_force: np.ndarray  # (3,) absolute frame, N
    buoyancy_moment: np.ndarray  # (3,) about G_h, absolute frame, N m
    centre_of_buoyancy: np.ndarray  # (3,) absolute frame offset from G_h
    submerged_fraction: float   # of total hull area, for diagnostics


class HullMesh:
    """A panelled hull surface, ready for repeated submersion queries.

    Panel centroids, corners, normals and areas are computed once in the
    hull frame; each query only rotates them and evaluates depths, so the
    cost per derivative evaluation is a handful of dense array operations.
    """

    def __init__(self, offsets: HullOffsets, n_girth: int = 16,
                 n_topside: int = 3):
        self.offsets = offsets
        self.n_girth = int(n_girth)
        self.n_topside = int(n_topside)
        if self.n_girth < 4:
            raise ValueError("need at least 4 girth panels")
        self._build()

    # -- construction ----------------------------------------------------
    def _section_points(self, beam: float, depth: float) -> np.ndarray:
        """Section outline from port gunwale round the keel to starboard.

        Returns shape ``(n_points, 2)`` of ``(y, z)`` pairs.
        """
        freeboard = self.offsets.freeboard

        # topsides: straight, port gunwale down to the waterline
        topside_z = np.linspace(freeboard, 0.0, self.n_topside + 1)[:-1]
        port_topside = np.column_stack(
            [np.full_like(topside_z, beam / 2.0), topside_z])

        # underwater: semi-ellipse, port waterline -> keel -> starboard
        psi = np.linspace(0.0, np.pi, self.n_girth + 1)
        underwater = np.column_stack(
            [(beam / 2.0) * np.cos(psi), -depth * np.sin(psi)])

        starboard_topside = np.column_stack(
            [np.full_like(topside_z, -beam / 2.0), topside_z[::-1]])

        return np.vstack([port_topside, underwater, starboard_topside])

    def _build(self) -> None:
        offsets = self.offsets
        sections = np.array([
            self._section_points(b, d)
            for b, d in zip(offsets.beam, offsets.depth)
        ])  # (n_stations, n_points, 2)

        n_stations, n_points, _ = sections.shape
        x = offsets.station

        # Quad panels between adjacent stations and adjacent girth points.
        # Corner order: (i, j), (i+1, j), (i+1, j+1), (i, j+1)
        corner_a = np.stack([
            np.repeat(x[:-1, None], n_points - 1, axis=1),
            sections[:-1, :-1, 0], sections[:-1, :-1, 1]], axis=-1)
        corner_b = np.stack([
            np.repeat(x[1:, None], n_points - 1, axis=1),
            sections[1:, :-1, 0], sections[1:, :-1, 1]], axis=-1)
        corner_c = np.stack([
            np.repeat(x[1:, None], n_points - 1, axis=1),
            sections[1:, 1:, 0], sections[1:, 1:, 1]], axis=-1)
        corner_d = np.stack([
            np.repeat(x[:-1, None], n_points - 1, axis=1),
            sections[:-1, 1:, 0], sections[:-1, 1:, 1]], axis=-1)

        corners = np.stack([corner_a, corner_b, corner_c, corner_d], axis=2)
        corners = corners.reshape(-1, 4, 3)

        centroid = corners.mean(axis=1)
        # Newell-style area vector from the two diagonals
        diagonal_1 = corners[:, 2] - corners[:, 0]
        diagonal_2 = corners[:, 3] - corners[:, 1]
        area_vector = 0.5 * np.cross(diagonal_1, diagonal_2)
        area = np.linalg.norm(area_vector, axis=1)

        keep = area > 1e-12
        self.corners = corners[keep]
        self.centroid = centroid[keep]
        area_vector = area_vector[keep]
        self.area = area[keep]

        normal = area_vector / self.area[:, None]
        # orient outward: away from the hull centreline axis (y = z = 0)
        radial = self.centroid.copy()
        radial[:, 0] = 0.0
        outward = np.sum(normal * radial, axis=1) < 0.0
        normal[outward] *= -1.0
        self.normal = normal

        self.total_area = float(self.area.sum())

    @property
    def n_panels(self) -> int:
        return len(self.area)

    # -- queries ---------------------------------------------------------
    def submerged(self, position: np.ndarray, attitude: np.ndarray,
                  rho: float = 1025.0, gravity: float = 9.81,
                  water_level: float = 0.0) -> SubmergedProperties:
        """Evaluate the wetted surface for the current hull pose.

        ``position`` is ``G_h`` in the absolute frame; ``attitude`` is the
        Euler attitude.  The still-water surface is at ``Z = water_level``.
        """
        rot = hull_to_abs(attitude)

        corners_abs = self.corners @ rot.T          # (n, 4, 3)
        centroid_abs = self.centroid @ rot.T        # (n, 3)
        normal_abs = self.normal @ rot.T            # (n, 3)

        # submersion depth at each corner: q = max(0, h0 - Z)
        corner_z = corners_abs[:, :, 2] + position[2]
        raw_depth = water_level - corner_z
        depth = np.maximum(raw_depth, 0.0)

        # Trapezoidal rule over the panel corners, as the paper does.  The
        # clipping happens per corner, so the transition through the
        # waterline is continuous rather than a step.
        mean_depth = depth.mean(axis=1)

        # Fraction of each panel below the surface: exact for a linear
        # depth field over a parallelogram, and continuous in the pose.
        spread = raw_depth.max(axis=1) - raw_depth.min(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(
                spread > 1e-12,
                np.clip(raw_depth.mean(axis=1) / spread + 0.5, 0.0, 1.0),
                (raw_depth.mean(axis=1) > 0.0).astype(float),
            )

        wet_area = self.area * fraction

        # hydrostatic force: dF = -rho g q n dsigma  (pressure acts inward)
        panel_force = -(rho * gravity * mean_depth * self.area)[:, None] \
            * normal_abs
        force = panel_force.sum(axis=0)
        moment = np.cross(centroid_abs, panel_force).sum(axis=0)

        # displaced volume by the divergence theorem: V = -(1/3) sum q_z ...
        # use the vertical flux form, which is robust for an open mesh
        volume = float(-(mean_depth * self.area * normal_abs[:, 2]).sum())

        total_force = float(np.linalg.norm(force))
        if total_force > 1e-9:
            centre_of_buoyancy = (
                (np.linalg.norm(panel_force, axis=1)[:, None] * centroid_abs
                 ).sum(axis=0) / np.linalg.norm(panel_force, axis=1).sum()
            )
        else:
            centre_of_buoyancy = np.zeros(3)

        return SubmergedProperties(
            wetted_area=float(wet_area.sum()),
            transverse_area=float(
                0.5 * (wet_area * np.abs(normal_abs[:, 0])).sum()),
            lateral_area=float(
                0.5 * (wet_area * np.abs(normal_abs[:, 1])).sum()),
            plan_area=float(abs((wet_area * normal_abs[:, 2]).sum())),
            volume=max(volume, 0.0),
            buoyancy_force=force,
            buoyancy_moment=moment,
            centre_of_buoyancy=centre_of_buoyancy,
            submerged_fraction=float(wet_area.sum() / self.total_area),
        )

    def equilibrium_heave(self, mass: float, attitude: np.ndarray = None,
                          rho: float = 1025.0, gravity: float = 9.81,
                          water_level: float = 0.0,
                          tolerance: float = 1e-8,
                          max_iterations: int = 200) -> float:
        """Find the ``G_z`` at which buoyancy balances ``mass * g``.

        Plain bisection on a monotone function -- sinking the hull can only
        increase the buoyant force -- so it is unconditionally convergent
        given a bracketing interval.
        """
        attitude = np.zeros(3) if attitude is None else attitude
        weight = mass * gravity

        def net(heave: float) -> float:
            position = np.array([0.0, 0.0, heave])
            props = self.submerged(position, attitude, rho, gravity,
                                   water_level)
            return props.buoyancy_force[2] - weight

        high = water_level + self.offsets.freeboard
        low = water_level - self.offsets.max_depth - self.offsets.freeboard
        if net(high) > 0.0:
            raise ValueError("hull floats clear of the water at this mass")
        if net(low) < 0.0:
            raise ValueError(
                f"hull cannot support {mass:.1f} kg: it submerges completely "
                f"(design displacement {self.offsets.design_displacement(rho):.1f} kg)"
            )

        # net() decreases with heave: lifting the hull removes buoyancy.
        # net(low) > 0 (too deep, excess buoyancy), net(high) < 0.
        for _ in range(max_iterations):
            middle = 0.5 * (low + high)
            if net(middle) > 0.0:
                low = middle    # still too buoyant, the hull must rise
            else:
                high = middle
            if high - low < tolerance:
                break
        return 0.5 * (low + high)
