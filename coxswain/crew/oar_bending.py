"""Oar-shaft bending as a quasi-static hinge (a research option of the dynamic oar).

Robert, Leroyer, Barre, Queutey & Visonneau (2019) [RLBQV19], sec. 2.3.1, eq. 1, measured on a bench for a sweep oar
held at the handle and the oarlock: the blade's linear and angular deflections are proportional to the bending moment
at the oarlock, ``f = M / K_lin`` and ``phi = M / K_ang`` (K_lin = 3424.1 N, K_ang = 2948 N m; the oar's inertia
small against the fluid load, so static). Because ``f / phi = K_ang / K_lin`` for any load, the blade moves as if the
shaft were rigid and hinged ``K_ang / K_lin`` inboard of the blade root, with a spring giving ``phi = M / K_ang``.

Tested against their measured trial (SOURCES §185-186, ``research/blade/oar_bending.py``): bending from the measured
moment improves every blade law 10-30%; solved from the law's own load it is stable.

**Scull.** No bench law of this form is published for a scull. Laschowski's thesis (Western Ontario 2014) [L14],
Chapter 2, clamped Concept2 Skinny sculls at the handle and the sleeve and hung up to 201.04 N on the blade end of
the shaft, L_b = 1.279-1.319 m from the sleeve: the medium shafts' blade-end slope was 4.42-4.86 deg at 201.04 N
(Table 3; extra-soft 5.85-6.26 deg), so as a uniform cantilever ``EI = W L_b^2 / (2 theta)``, 2099 N m^2 for the
medium shaft at 2.68 m (Table 2's measured EI near the blade, 1992-2169 N m^2, agrees). On the boat's own scull, with
the load at the blade centre ``c`` and the blade rigid beyond its root ``x``: ``K_ang = 2 EI c / (x (2 c - x))`` and
``f / phi = x (3 c - x) / (3 (2 c - x))``. The same beam formula on [RLBQV19]'s oar gives f/phi = 0.935 m against
their bench's 0.861 m, so the hinge picture holds to 8%. **Derived from measured deflections**, and the shafts are not
uniform (most compliant near the sleeve). Until 2026-11-03 this law was built from "1.18 deg at 201 N", which is the
largest DIFFERENCE stiffness or length made to that slope (the published abstract, Laschowski et al. 2017, Sports
Biomech. 16(1)), not the slope itself: it was about nine times too stiff (SOURCES sec. 217).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _l14_ei(theta_deg: float, beam: float, load: float = 201.04) -> float:
    """``EI = W L_b^2 / (2 theta)``: a uniform end-loaded cantilever's flexural rigidity from its end slope."""
    return load * beam ** 2 / (2.0 * np.radians(theta_deg))


#: [L14] Table 3 at 2.68 m (L_b 1.299 m), mean of the port and starboard shafts: medium 4.77 / 4.49 deg, extra-soft
#: 5.98 / 6.13 deg at 201.04 N.
L14_MEDIUM_EI = _l14_ei(0.5 * (4.77 + 4.49), 1.299)
L14_EXTRA_SOFT_EI = _l14_ei(0.5 * (5.98 + 6.13), 1.299)

#: Concept2's published shaft deflections under 10 kg hung at the neck, cm (concept2.com shafts-and-flex pages, read
#: 2026-11-05): medium scull 4.5 (Skinny and Ultralight); medium sweep 3.75 (Skinny 4.0, Ultralight 3.5). The scull
#: test's span is 1.50 m from the support ([L14] citing Concept2); the sweep's is not printed, so it is scaled by the
#: blade root's distance from the pin (2.04 / 1.57 m on the catalogue oars): a stated assumption.
C2_SCULL_MEDIUM_CM, C2_SWEEP_MEDIUM_CM = 4.5, 3.75
C2_SCULL_SPAN = 1.50


def concept2_sweep_over_scull(sweep_root: float = 2.04, scull_root: float = 1.57) -> float:
    """EI of a medium sweep shaft over a medium scull shaft by Concept2's own test: ``(L_sw / L_sc)^3 d_sc / d_sw``
    for a cantilever, the spans in the ratio of the blade roots' distances from the pin. ~2.6, as a tube's EI goes as
    its radius cubed and a sweep shaft is ~1.4x the diameter (the user, 2026-11-06: "larger radius shafts")."""
    span = C2_SCULL_SPAN * float(sweep_root) / float(scull_root)
    return (span / C2_SCULL_SPAN) ** 3 * C2_SCULL_MEDIUM_CM / C2_SWEEP_MEDIUM_CM


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
    def concept2_sweep(cls, outboard_centre: float, blade_length: float) -> "OarBending":
        """A medium sweep shaft: [L14]'s measured medium scull EI times Concept2's sweep-over-scull stiffness ratio
        (``concept2_sweep_over_scull``), as a uniform cantilever on this oar (derived; SOURCES §220). [RLBQV19]'s bench
        law is ~1.65x softer than this: their K_ang was measured on their rotating-arm mount, which may add its own
        compliance to the oar's."""
        x = float(outboard_centre) - 0.5 * float(blade_length)
        ei = L14_MEDIUM_EI * concept2_sweep_over_scull(sweep_root=x)
        law = cls.laschowski2014_scull(outboard_centre, blade_length, ei=ei)
        return cls(law.k_ang, law.hinge_inboard_of_root,
                   "[L14] scull EI x Concept2 sweep/scull %.2f = %.0f N m^2, uniform cantilever (derived)"
                   % (concept2_sweep_over_scull(sweep_root=x), ei))

    @classmethod
    def laschowski2014_scull(cls, outboard_centre: float, blade_length: float,
                             ei: float = None) -> "OarBending":
        """[L14]'s measured medium scull shaft as a uniform cantilever on this scull (derived; see the module notes).
        ``ei`` overrides the flexural rigidity, N m^2 (``L14_EXTRA_SOFT_EI`` for the extra-soft shaft)."""
        ei = L14_MEDIUM_EI if ei is None else float(ei)
        c = float(outboard_centre)                                      # the load: the blade centre
        x = c - 0.5 * float(blade_length)                               # blade root from the pin
        k_ang = 2.0 * ei * c / (x * (2.0 * c - x))
        ratio = x * (3.0 * c - x) / (3.0 * (2.0 * c - x))
        return cls(float(k_ang), float(ratio), "[L14] EI %.0f N m^2, uniform cantilever (derived)" % ei)


def beam_hinge_ratio(root: float, tip: float) -> float:
    """``f / phi`` at the blade root of a uniform cantilever loaded at its tip (the check against [RLBQV19])."""
    return float(root * (3.0 * tip - root) / (3.0 * (2.0 * tip - root)))
