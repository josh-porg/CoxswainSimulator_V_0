r"""The hand as the end of a chain: the arm on its clock plus pluggable reach terms (SOURCES §209–211).

The body-generated hand (``research/crew/onwater_body.body_drive_table``) puts the grip on its arc at a reach from the
shoulders:

.. math::

    r = r_{arm}(\theta_e(x)) + \sum_i \Delta r_i(\text{context})

The arm term is always there (:mod:`coxswain.crew.arm_draw`: the held elbow, then the minimum-jerk draw, its finish
flexion solved from body and rig). Every other term is optional and named, so a mechanism is added by writing one
small class and registering it in ``TERMS``; nothing else changes:

* ``extra(ctx)``: the term's extra reach (m) at a :class:`HandContext` (phase, time fraction, handle force fraction);
* ``max_extra()``: its largest extra reach, which the chain adds to the kinematic rower's reach allowance
  (``BodyOptions.reach_allowance``) so the boat's own elbow solve accepts the hand;
* ``signature()``: for cache keys.

Registered terms:

* ``"stretch"`` (:class:`StretchTerm`): the upper-back-to-hand chain lengthening in proportion to the handle force,
  ``s0 F/F_peak`` (§210; ``arm_draw.PEAK_CHAIN_STRETCH``, measured on the test athletes, so not on by default).

Candidates that would plug in the same way: a shoulder-girdle posture term (a sourced scapular translation against
time), a wrist term (Pitto et al. 2024's wrist flexion), a trunk-rotation term for sweep rowers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import arm_draw as A


@dataclass(frozen=True)
class HandContext:
    """Where in the stroke a reach term is evaluated."""

    phase: str                      # "drive" or "recovery"
    x: np.ndarray                   # time fraction of that phase, 0-1
    force_fraction: np.ndarray      # handle force / the stroke's peak (0 through the recovery)


@dataclass(frozen=True)
class StretchTerm:
    """The upper-back-to-hand chain's stretch under the handle force (SOURCES §210)."""

    peak: float = A.PEAK_CHAIN_STRETCH
    name: str = "stretch"

    def extra(self, ctx: HandContext):
        return A.chain_stretch(ctx.force_fraction, self.peak)

    def max_extra(self) -> float:
        return float(self.peak)

    def signature(self) -> tuple:
        return (self.name, round(float(self.peak), 9))


#: reach terms by name
#: The sourced part of the stretch (SOURCES §277): the thoracic spine's extension from the catch to mid-drive in Telfer
#: et al. 2023's open data (11 rowers, rate 32: 8.7 deg, i.e. 45-53 mm of shoulder travel over a 0.30-0.35 m upper
#: trunk), which a rigid trunk cannot make; the rest of the athletes' 0.124 m (scapular glide) stays unsourced.
THORACIC_STRETCH = 0.049


@dataclass(frozen=True)
class ThoracicTerm(StretchTerm):
    """The rigid trunk's missing thoracic extension under load, as a reach term (sourced, §277)."""

    peak: float = THORACIC_STRETCH
    name: str = "thoracic"


TERMS = {"stretch": StretchTerm, "thoracic": ThoracicTerm}


@dataclass(frozen=True)
class HandChain:
    """The arm on its clock plus optional reach terms."""

    terms: tuple = ()

    def extra(self, ctx: HandContext):
        total = np.zeros_like(np.asarray(ctx.x, dtype=float))
        for term in self.terms:
            total = total + np.asarray(term.extra(ctx), dtype=float)
        return total

    def max_extra(self) -> float:
        return float(sum(term.max_extra() for term in self.terms))

    def reach_allowance(self):
        """The kinematic rower's reach allowance this chain needs (m), or None for the shipped one."""
        if not self.terms:
            return None
        from .kinematics import MAX_SHOULDER_PROTRACTION
        return float(MAX_SHOULDER_PROTRACTION + self.max_extra())

    def signature(self) -> tuple:
        return tuple(term.signature() for term in self.terms)


def hand_chain(terms=()) -> HandChain:
    """A chain from term names (``TERMS``, default parameters) or term objects."""
    return HandChain(tuple(TERMS[t]() if isinstance(t, str) else t for t in terms))
