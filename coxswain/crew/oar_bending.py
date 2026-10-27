"""Oar-shaft bending as a quasi-static hinge (a research option of the dynamic oar).

Robert, Leroyer, Barre, Queutey & Visonneau (2019) [RLBQV19], sec. 2.3.1, eq. 1, measured on a bench for a sweep oar
held at the handle and the oarlock: the blade's linear and angular deflections are proportional to the bending moment
at the oarlock, ``f = M / K_lin`` and ``phi = M / K_ang`` (K_lin = 3424.1 N, K_ang = 2948 N m; the oar's inertia
small against the fluid load, so static). Because ``f / phi = K_ang / K_lin`` for any load, the blade moves as if the
shaft were rigid and hinged ``K_ang / K_lin`` inboard of the blade root, with a spring giving ``phi = M / K_ang``.

Tested against their measured trial (SOURCES §185-186, ``research/blade/oar_bending.py``): bending from the measured
moment improves every blade law 10-30%; solved from the law's own load it is stable.

**Scull.** No bench law of this form is published for a scull. Laschowski, Nolte, Adamovsky & Alexander (2015) [LNA15]
give a sculling oar's deflection angle at the blade end, 1.18 deg at 201 N there. With the shaft as a cantilever from
the oarlock (Euler-Bernoulli, uniform EI, end load F at the tip L_t): ``theta_tip = F L_t^2 / (2 EI)`` gives EI; the
slope and deflection at the blade root ``x`` follow, so ``K_ang = 2 EI L_t / (x (2 L_t - x))`` and
``f / phi = x (3 L_t - x) / (3 (2 L_t - x))``. The same beam formula on [RLBQV19]'s oar gives f/phi = 0.935 m against
their bench's 0.861 m, so the hinge picture holds to 8%. **Derived, not measured**, and the shafts are not uniform
([LNA15]: up to 80% stiffer toward the blade).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OarBending:
    """``K_ang`` (N m per rad, oarlock moment over blade rotation) and ``hinge_inboard_of_root`` (m)."""
    k_ang: float
    hinge_inboard_of_root: float
    source: str = ""

    def __post_init__(self):
        if not self.k_ang > 0.0 or not self.hinge_inboard_of_root >= 0.0:
            raise ValueError("bending needs a positive stiffness and a hinge at or inboard of the blade root")

    def hinge_radius(self, outboard_centre: float, blade_length: float) -> float:
        """The hinge's distance from the pin, for an oar whose blade centre is ``outboard_centre`` from the pin."""
        root = float(outboard_centre) - 0.5 * float(blade_length)
        return max(root - float(self.hinge_inboard_of_root), 0.0)

    @classmethod
    def robert2019_sweep(cls) -> "OarBending":
        """[RLBQV19]'s bench law for their sweep oar on their mount: K_ang 2948 N m, K_ang/K_lin = 0.861 m."""
        return cls(2948.0, 2948.0 / 3424.1, "[RLBQV19] eq. 1, sweep oar on the rotating arm")

    @classmethod
    def laschowski2015_scull(cls, outboard_centre: float, blade_length: float,
                             tip_load: float = 201.0, tip_angle_deg: float = 1.18) -> "OarBending":
        """[LNA15]'s scull deflection through a uniform cantilever from the oarlock (derived; see the module notes)."""
        x = float(outboard_centre) - 0.5 * float(blade_length)          # blade root from the pin
        tip = float(outboard_centre) + 0.5 * float(blade_length)        # blade end from the pin
        ei = tip_load * tip ** 2 / (2.0 * np.radians(tip_angle_deg))
        k_ang = 2.0 * ei * tip / (x * (2.0 * tip - x))
        ratio = x * (3.0 * tip - x) / (3.0 * (2.0 * tip - x))
        return cls(float(k_ang), float(ratio), "[LNA15] 1.18 deg at 201 N, uniform cantilever (derived)")


def beam_hinge_ratio(root: float, tip: float) -> float:
    """``f / phi`` at the blade root of a uniform cantilever loaded at its tip (the check against [RLBQV19])."""
    return float(root * (3.0 * tip - root) / (3.0 * (2.0 * tip - root)))
