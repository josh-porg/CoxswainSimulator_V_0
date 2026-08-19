"""Body-segment inertial parameters for the 12-segment rower model.

Formaggia et al. (2009) section 4.2 subdivide each athlete into ``p = 12``
point masses whose mass fractions come from anatomical tables (their
reference [10], NASA-STD-3000 *Man-System Integration Standards*).  Those
tables are themselves derived from the Zatsiorsky-Seluyanov gamma-ray
scanning study; the modern, joint-centre-referenced form of that data is

    P. de Leva, "Adjustments to Zatsiorsky-Seluyanov's segment inertia
    parameters", *Journal of Biomechanics* **29**(9) (1996) 1223-1230.

which is what this module tabulates, because de Leva's lengths are
referenced to *joint centres* and can therefore drive a kinematic linkage
directly (see :mod:`coxswain.crew.kinematics`).

The reference samples are 73.0 kg / 1.741 m for males and 61.9 kg /
1.735 m for females (de Leva Table 4 caption).

The 12 segments
---------------
de Leva lists 16 distinct body parts.  They are lumped into 12 by joining
forearm+hand and shank+foot, keeping port and starboard separate so that
asymmetric crew motion can generate a genuine roll moment::

    head, upper trunk, mid trunk, lower trunk      (4, on the centreline)
    upper arm, forearm+hand, thigh, shank+foot     (4 x 2 sides = 8)

Mass fractions sum to exactly 1.0 for both sexes, which
:mod:`tests.unit.test_anthropometry` checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

__all__ = [
    "PORT",
    "STARBOARD",
    "CENTRELINE",
    "SegmentSpec",
    "Segment",
    "RowerAnthropometry",
    "DE_LEVA_MALE",
    "DE_LEVA_FEMALE",
    "REFERENCE_MASS",
    "REFERENCE_STATURE",
    "N_SEGMENTS",
]

#: Lateral sense.  ``+1`` is port (hull +y), ``-1`` starboard, ``0`` centre.
PORT, STARBOARD, CENTRELINE = 1, -1, 0

N_SEGMENTS = 12

REFERENCE_MASS = {"male": 73.0, "female": 61.9}
REFERENCE_STATURE = {"male": 1.741, "female": 1.735}


@dataclass(frozen=True)
class SegmentSpec:
    """Dimensionless description of one body segment.

    Attributes
    ----------
    name:
        Segment identifier, without a side suffix.
    mass_fraction:
        Segment mass divided by whole-body mass, for *one* instance of the
        segment (i.e. one arm, not the pair).
    length_fraction:
        Joint-centre-to-joint-centre length divided by stature.
    com_fraction:
        Distance of the segment centre of mass from its proximal joint,
        divided by the segment length.
    paired:
        ``True`` if the body has a port and a starboard copy.
    """

    name: str
    mass_fraction: float
    length_fraction: float
    com_fraction: float
    paired: bool


def _spec(name, mass_pct, length_mm, com_pct, paired, stature_m):
    return SegmentSpec(
        name=name,
        mass_fraction=mass_pct / 100.0,
        length_fraction=(length_mm / 1000.0) / stature_m,
        com_fraction=com_pct / 100.0,
        paired=paired,
    )


_MALE_STATURE = REFERENCE_STATURE["male"]
_FEMALE_STATURE = REFERENCE_STATURE["female"]

#: de Leva (1996) Table 4, male column.  ``(mass %, length mm, CM %)``.
DE_LEVA_MALE: Dict[str, SegmentSpec] = {
    s.name: s for s in [
        _spec("head", 6.94, 203.3, 59.76, False, _MALE_STATURE),
        # UPT uses de Leva's alternative CERV->XYPH endpoints (not
        # SUPRA->XYPH) so that LPT + MPT + UPT sums exactly to the
        # whole-trunk length, letting the three stack into one rigid link.
        _spec("upper_trunk", 15.96, 242.1, 50.66, False, _MALE_STATURE),
        _spec("mid_trunk", 16.33, 215.5, 45.02, False, _MALE_STATURE),
        _spec("lower_trunk", 11.17, 145.7, 61.15, False, _MALE_STATURE),
        _spec("upper_arm", 2.71, 281.7, 57.72, True, _MALE_STATURE),
        _spec("forearm", 1.62, 268.9, 45.74, True, _MALE_STATURE),
        _spec("hand", 0.61, 86.2, 79.00, True, _MALE_STATURE),
        _spec("thigh", 14.16, 422.2, 40.95, True, _MALE_STATURE),
        _spec("shank", 4.33, 434.0, 44.59, True, _MALE_STATURE),
        _spec("foot", 1.37, 258.1, 44.15, True, _MALE_STATURE),
    ]
}

#: de Leva (1996) Table 4, female column.
DE_LEVA_FEMALE: Dict[str, SegmentSpec] = {
    s.name: s for s in [
        _spec("head", 6.68, 200.2, 58.94, False, _FEMALE_STATURE),
        _spec("upper_trunk", 15.45, 228.0, 50.50, False, _FEMALE_STATURE),
        _spec("mid_trunk", 14.65, 205.3, 45.12, False, _FEMALE_STATURE),
        _spec("lower_trunk", 12.47, 181.5, 49.20, False, _FEMALE_STATURE),
        _spec("upper_arm", 2.55, 275.1, 57.54, True, _FEMALE_STATURE),
        _spec("forearm", 1.38, 264.3, 45.59, True, _FEMALE_STATURE),
        _spec("hand", 0.56, 78.0, 74.74, True, _FEMALE_STATURE),
        _spec("thigh", 14.78, 368.5, 36.12, True, _FEMALE_STATURE),
        _spec("shank", 4.81, 432.3, 44.16, True, _FEMALE_STATURE),
        _spec("foot", 1.29, 228.3, 40.14, True, _FEMALE_STATURE),
    ]
}

_TABLES = {"male": DE_LEVA_MALE, "female": DE_LEVA_FEMALE}


@dataclass(frozen=True)
class Segment:
    """A dimensional body segment belonging to a specific athlete."""

    name: str
    mass: float          # kg
    length: float        # m, proximal joint to distal joint
    com_fraction: float  # of length, from the proximal joint
    side: int            # PORT / STARBOARD / CENTRELINE


class RowerAnthropometry:
    """Segment masses and lengths for one athlete, scaled from de Leva.

    Masses scale linearly with body mass; lengths scale linearly with
    stature.  The 12-segment lumping joins forearm+hand and shank+foot,
    combining their masses and summing their lengths (the lumped centre of
    mass is recomputed as the mass-weighted mean of the two).
    """

    def __init__(self, mass: float = 85.0, stature: float = 1.85,
                 sex: str = "male"):
        if mass <= 0:
            raise ValueError("body mass must be positive")
        if stature <= 0:
            raise ValueError("stature must be positive")
        if sex not in _TABLES:
            raise ValueError(f"sex must be one of {sorted(_TABLES)}, got {sex!r}")

        self.mass = float(mass)
        self.stature = float(stature)
        self.sex = sex
        self._table = _TABLES[sex]
        self._segments = self._build()

    # -- construction ----------------------------------------------------
    def _dimensional(self, name: str):
        spec = self._table[name]
        return (spec.mass_fraction * self.mass,
                spec.length_fraction * self.stature,
                spec.com_fraction)

    def _lump(self, proximal: str, distal: str):
        """Join two serial segments into one equivalent point mass."""
        m_p, l_p, c_p = self._dimensional(proximal)
        m_d, l_d, c_d = self._dimensional(distal)

        total_mass = m_p + m_d
        total_length = l_p + l_d
        # distance of each CM from the proximal joint of the pair
        d_p = c_p * l_p
        d_d = l_p + c_d * l_d
        com_distance = (m_p * d_p + m_d * d_d) / total_mass
        return total_mass, total_length, com_distance / total_length

    def _build(self) -> List[Segment]:
        segments: List[Segment] = []

        for name in ("head", "upper_trunk", "mid_trunk", "lower_trunk"):
            mass, length, com = self._dimensional(name)
            segments.append(Segment(name, mass, length, com, CENTRELINE))

        paired = [
            ("upper_arm", self._dimensional("upper_arm")),
            ("forearm_hand", self._lump("forearm", "hand")),
            ("thigh", self._dimensional("thigh")),
            ("shank_foot", self._lump("shank", "foot")),
        ]
        for name, (mass, length, com) in paired:
            for side, suffix in ((PORT, "port"), (STARBOARD, "starboard")):
                segments.append(
                    Segment(f"{name}_{suffix}", mass, length, com, side)
                )

        if len(segments) != N_SEGMENTS:
            raise AssertionError(
                f"expected {N_SEGMENTS} segments, built {len(segments)}"
            )
        return segments

    # -- accessors -------------------------------------------------------
    @property
    def segments(self) -> Sequence[Segment]:
        return tuple(self._segments)

    @property
    def segment_masses(self) -> np.ndarray:
        """Shape ``(12,)`` array of segment masses in kg."""
        return np.array([s.mass for s in self._segments])

    @property
    def total_segment_mass(self) -> float:
        return float(self.segment_masses.sum())

    def by_name(self, name: str) -> Segment:
        for segment in self._segments:
            if segment.name == name:
                return segment
        raise KeyError(f"no segment named {name!r}; have "
                       f"{[s.name for s in self._segments]}")

    def length(self, name: str) -> float:
        """Joint-centre-to-joint-centre length of a *base* segment, in m."""
        return self._table[name].length_fraction * self.stature

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"RowerAnthropometry(mass={self.mass:.1f}, "
                f"stature={self.stature:.3f}, sex={self.sex!r})")
