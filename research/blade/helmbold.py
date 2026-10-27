"""Tier 2's lift amplitude from Helmbold's low-aspect-ratio lift slope, the blade's own aspect ratio along the flow.

At small incidence a rowing blade's flow runs along its length, so the plate's aspect ratio in that flow is
height / length (0.48 for the project's sweep blade, 0.45 for [GTW21]'s); with the top edge at a free surface that
acts as a mirror, twice that. Helmbold (1942), as given by Gutierrez-Castillo, Aguilar-Cabello, Alcalde-Morales, Parras & del Pino (preprint
submitted to J. Wind Eng. Ind. Aerodyn., 2020; Univ. Malaga repository; their eq. 4):

    C_L_alpha = a0 / ( sqrt(1 + (a0 / (pi AR))^2) + a0 / (pi AR) ),   a0 = 2 pi.

[CG07]'s interpolation C_L = A_l sin(2 alpha) has small-incidence slope 2 A_l = 2.5 /rad for the Big Blade. Here A_l
is set to C_L_alpha / 2, so the lift curve keeps [CG07]'s shape and drag but starts at the low-AR slope: one sourced
change, with the surface's mirror (AR doubled) and no mirror as its bracket. A study option, not a profile.
"""
from dataclasses import dataclass

import numpy as np

from coxswain.crew.liftdrag import LiftDragBlade


def helmbold_slope(aspect):
    a0 = 2.0 * np.pi
    k = a0 / (np.pi * aspect)
    return a0 / (np.sqrt(1.0 + k * k) + k)


def blade(height, length, outboard, area, mirror):
    """A [CG07] Big Blade with its lift amplitude from Helmbold at this blade's along-flow aspect ratio."""
    ar = (2.0 if mirror else 1.0) * height / length
    return LiftDragBlade(helmbold_slope(ar) / 2.0, 2.07, area, outboard, 1000.0), ar


@dataclass(frozen=True)
class PolhamusLamarBlade(LiftDragBlade):
    """Normal load from Polhamus's leading-edge-suction analogy as Lamar extended it to side edges, capped face-on.

    Lamar (1974), NASA TR R-428, eq. 23: C_L = K_p sin a cos^2 a + K_v,tot sin^2 a cos a, with the force normal to the
    plate (Delta C_D = C_L tan a), so C_N = K_p sin a cos a + K_v sin^2 a. K_p is the potential-flow slope (Helmbold,
    above); for rectangular plates below aspect ratio 2.8, K_v,le + K_v,se is "very near pi" (his conclusion 3). The
    analogy is a moderate-incidence result; face-on it would give C_N = pi, so C_N is capped at a face-on value,
    a sourced bracket: ~1.3 (a deep AR-2 plate, [G20]) or 2.05 ([LJH12]'s dynamic measurement; [CG07] 2.07).
    The tangential coefficient keeps tier 2's resolution with Helmbold's lift (drag-like, §188): lift_amplitude is
    K_p / 2. A LiftDragBlade, so ``loads`` and the strip integration use it unchanged.
    """
    k_v: float = float(np.pi)
    cap: float = 1.3

    def coefficients(self, alpha):
        s, c = np.sin(alpha), np.cos(alpha)
        c_n = min(2.0 * self.lift_amplitude * s * c + self.k_v * s * s, self.cap)
        return float(c_n), LiftDragBlade.coefficients(self, alpha)[1]


def polhamus_lamar(height, length, area, mirror, cap, outboard=1.0, k_v=float(np.pi)):
    """The blade above for this blade's along-flow aspect ratio (doubled with the surface's mirror)."""
    ar = (2.0 if mirror else 1.0) * height / length
    return PolhamusLamarBlade(helmbold_slope(ar) / 2.0, 2.07, area, outboard, 1000.0, None, k_v, cap)
