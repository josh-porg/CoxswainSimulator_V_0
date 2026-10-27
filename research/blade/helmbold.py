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
