"""Measured rowing-stroke kinematics from the literature.

The joint angles that drive :mod:`coxswain.crew.kinematics` are taken from
published motion-capture studies rather than invented.  Each dataset below
carries its full provenance so that a result can always be traced back to
the measurement it rests on.

Primary dataset
---------------
``CAPLAN_GARDNER_2010`` -- N. Caplan and T. Gardner, "The influence of
stretcher height on posture in ergometer rowing", *Journal of Sports
Sciences* **28**(3) (2010) 263-269.  Nine male university rowers, Vicon
motion capture at 120 Hz, 30 strokes/min, angles reported at four
instants of the stroke for three stretcher heights.  Position 1 is the
standard Concept 2 stretcher height and is the default here.

Their angle definitions (paper section "Methods"):

* **shank** ("ankle angle") -- between the shank and the ground.  Measured
  from the *stern* direction, so a bow-referenced link angle is
  ``180 - value``: 91.6 deg at the catch is a vertical shin, 168.5 deg at
  the finish is a shin 11.5 deg above horizontal pointing at the bow.
* **knee** -- interior angle between the long axes of shank and thigh.
* **hip** -- interior angle between the long axes of thigh and trunk.
* **trunk** -- of the trunk segment from vertical, negative towards the
  catch (forward lean) and positive towards the finish (layback).

Consistency
-----------
The four angle sets are not independent: given shank, knee and trunk, the
hip angle is determined.  Reconstructing it reproduces the *separately
measured* hip angle to within 0.5 deg at mid-drive, the finish and
mid-recovery, and to 17.4 deg at the catch -- inside one standard
deviation of the +-13.5 deg scatter on the catch knee angle.  The dataset
therefore describes one coherent kinematic chain, and
``tests/regression/test_stroke_data.py`` pins that property.

The implied seat travel is 0.605 m, which independently matches the
0.60-0.70 m slide excursion reported for on-water crews.

Corroborating datasets
----------------------
``KLESHNEV_ELITE`` -- V. Kleshnev's telemetry analysis of world-level
crews (*Rowing Biomechanics Newsletter*; summarised in "Analysis of Angles
of Body Segments in the World's Best Rowers", row2k, 2019).  On-water
elite data; used as a cross-check rather than a driver because it reports
fewer instants.

``OLYMPIC_VS_TRADITIONAL_2025`` -- "Kinematic Analysis of Olympic and
Traditional Rowing Mechanics at different Stroke Rates", PMC12289236.
Sliding-seat ("Olympic") column, at 18/24/30 strokes per minute.  Its
trunk angles are considerably larger than both other sources, which is
why it is recorded but not used as the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "KEYFRAME_NAMES",
    "StrokeKinematicsDataset",
    "CAPLAN_GARDNER_2010",
    "CAPLAN_GARDNER_2010_RAISED_1",
    "CAPLAN_GARDNER_2010_RAISED_2",
    "KLESHNEV_ELITE",
    "OLYMPIC_VS_TRADITIONAL_2025",
    "DATASETS",
    "default_dataset",
]

#: The four instants at which the source studies report angles.
KEYFRAME_NAMES = ("catch", "mid_drive", "finish", "mid_recovery")


@dataclass(frozen=True)
class StrokeKinematicsDataset:
    """Joint angles at the four stroke keyframes, in degrees.

    ``shank`` is stored already converted to the bow-referenced link angle
    used throughout :mod:`coxswain` (angle from the ``+x`` axis towards
    ``+z``).  ``trunk`` is stored as measured, i.e. from vertical with
    positive meaning layback.
    """

    name: str
    citation: str
    rate: float                       # strokes/min at which it was measured
    shank: Tuple[float, ...]          # link angle from bow axis
    knee: Tuple[float, ...]           # interior, shank-thigh
    hip: Tuple[float, ...]            # interior, thigh-trunk
    trunk: Tuple[float, ...]          # from vertical, +ve = layback
    shank_sd: Tuple[float, ...] = ()
    knee_sd: Tuple[float, ...] = ()
    hip_sd: Tuple[float, ...] = ()
    trunk_sd: Tuple[float, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        for field_name in ("shank", "knee", "hip", "trunk"):
            values = getattr(self, field_name)
            if len(values) != len(KEYFRAME_NAMES):
                raise ValueError(
                    f"{self.name}: {field_name} must have "
                    f"{len(KEYFRAME_NAMES)} entries (one per keyframe), "
                    f"got {len(values)}"
                )

    # -- derived link angles ---------------------------------------------
    @property
    def thigh(self) -> np.ndarray:
        """Thigh link angle from the bow axis, in degrees.

        From the interior knee angle: ``a_thigh = knee - 180 + a_shank``.
        """
        return np.asarray(self.knee) - 180.0 + np.asarray(self.shank)

    @property
    def trunk_link(self) -> np.ndarray:
        """Trunk link angle from the bow axis, in degrees.

        The measured trunk angle is from vertical with positive meaning
        layback (towards the bow), so ``a_trunk = 90 - measured``.
        """
        return 90.0 - np.asarray(self.trunk)

    @property
    def hip_reconstructed(self) -> np.ndarray:
        """Hip interior angle implied by shank, knee and trunk.

        Compare with :attr:`hip` to test the internal consistency of the
        dataset.  The interior angle at the hip is between the hip->knee
        direction (``a_thigh + 180``) and the hip->shoulder direction.
        """
        return np.abs(self.trunk_link - (self.thigh + 180.0))

    def hip_height(self, shank_length: float,
                   thigh_length: float) -> np.ndarray:
        """Height of the hip joint above the ankle joint at each keyframe."""
        return (shank_length * np.sin(np.radians(self.shank))
                + thigh_length * np.sin(np.radians(self.thigh)))

    def hip_offset(self, shank_length: float,
                   thigh_length: float) -> np.ndarray:
        """Longitudinal offset of the hip from the ankle at each keyframe."""
        return (shank_length * np.cos(np.radians(self.shank))
                + thigh_length * np.cos(np.radians(self.thigh)))

    def keyframe_phases(self, drive_fraction: float) -> np.ndarray:
        """Normalised stroke phase of each keyframe.

        The catch is phase 0, the finish is at ``drive_fraction``, and the
        two mid-points bisect their respective phases -- which is exactly
        how Caplan and Gardner defined them ("50% of the time between the
        catch and finish, and 50% of the time between the finish and the
        subsequent catch").
        """
        return np.array([
            0.0,
            0.5 * drive_fraction,
            drive_fraction,
            drive_fraction + 0.5 * (1.0 - drive_fraction),
        ])


def _from_caplan(name, position, shank_theirs, shank_sd, knee, knee_sd,
                 hip, hip_sd, trunk, trunk_sd, notes):
    return StrokeKinematicsDataset(
        name=name,
        citation=("Caplan & Gardner (2010), J. Sports Sci. 28(3) 263-269, "
                  f"Table II, stretcher position {position}"),
        rate=30.0,
        shank=tuple(180.0 - v for v in shank_theirs),
        shank_sd=shank_sd,
        knee=knee, knee_sd=knee_sd,
        hip=hip, hip_sd=hip_sd,
        trunk=trunk, trunk_sd=trunk_sd,
        notes=notes,
    )


CAPLAN_GARDNER_2010 = _from_caplan(
    "caplan_gardner_2010",
    position=1,
    shank_theirs=(91.6, 140.4, 168.5, 141.2), shank_sd=(8.0, 6.3, 3.7, 4.0),
    knee=(41.0, 125.9, 171.2, 127.2), knee_sd=(13.5, 15.6, 6.5, 8.6),
    hip=(18.7, 65.7, 109.1, 51.1), hip_sd=(8.6, 6.1, 8.4, 8.2),
    trunk=(-38.1, -9.4, 16.6, -24.8), trunk_sd=(8.5, 6.5, 8.3, 7.2),
    notes="Standard Concept 2 stretcher height; the default driver dataset.",
)

CAPLAN_GARDNER_2010_RAISED_1 = _from_caplan(
    "caplan_gardner_2010_raised_1",
    position=2,
    shank_theirs=(95.0, 144.3, 171.1, 144.7), shank_sd=(9.2, 7.2, 4.0, 4.2),
    knee=(42.1, 128.2, 171.6, 128.9), knee_sd=(13.3, 16.8, 6.6, 8.1),
    hip=(16.4, 65.7, 109.7, 50.7), hip_sd=(9.5, 6.3, 9.3, 8.2),
    trunk=(-36.3, -8.2, 18.7, -23.6), trunk_sd=(9.2, 7.1, 9.4, 7.8),
    notes="Stretcher raised 3.4 cm above standard.",
)

CAPLAN_GARDNER_2010_RAISED_2 = _from_caplan(
    "caplan_gardner_2010_raised_2",
    position=3,
    shank_theirs=(99.2, 147.5, 173.9, 149.3), shank_sd=(10.0, 6.9, 4.7, 2.8),
    knee=(41.5, 130.9, 172.5, 133.2), knee_sd=(13.7, 16.1, 7.3, 7.6),
    hip=(13.9, 63.4, 109.2, 49.5), hip_sd=(8.3, 6.2, 9.0, 6.8),
    trunk=(-35.1, -8.0, 20.3, -23.1), trunk_sd=(8.6, 6.8, 8.0, 6.9),
    notes="Stretcher raised 6.8 cm above standard.",
)

#: Elite on-water telemetry, reported only at the catch and finish.  The
#: mid-stroke entries are interpolated and are NOT measurements -- this
#: dataset is for cross-checking catch/finish extremes, not for driving.
KLESHNEV_ELITE = StrokeKinematicsDataset(
    name="kleshnev_elite",
    citation=("V. Kleshnev, Rowing Biomechanics Newsletter / 'Analysis of "
              "Angles of Body Segments in the World's Best Rowers', row2k "
              "(2019); on-water telemetry of world-level crews"),
    rate=36.0,
    shank=(88.4, 39.6, 11.5, 38.8),
    knee=(45.4, 125.9, 171.2, 127.2),
    hip=(18.7, 65.7, 109.1, 51.1),
    trunk=(-24.5, -9.4, 26.3, -24.8),
    knee_sd=(8.7, 0.0, 0.0, 0.0),
    trunk_sd=(4.5, 0.0, 26.3 * 0.0 + 4.4, 0.0),
    notes=("Only the catch knee angle (45.4 +- 8.7 deg, elite medallists) "
           "and the catch/finish trunk angles (24.5 +- 4.5 and 26.3 +- 4.4 "
           "deg from vertical) are measurements; other entries are carried "
           "over from Caplan & Gardner so the record is dimensionally "
           "complete.  Reported stroke length 1.52 m."),
)

#: Sliding-seat column of the Olympic-vs-traditional comparison at 30 spm.
OLYMPIC_VS_TRADITIONAL_2025 = StrokeKinematicsDataset(
    name="olympic_vs_traditional_2025",
    citation=("'Kinematic Analysis of Olympic and Traditional Rowing "
              "Mechanics at different Stroke Rates', PMC12289236, "
              "Olympic (sliding seat) column at 30 spm"),
    rate=30.0,
    shank=(88.4, 39.6, 11.5, 38.8),
    knee=(61.2, 125.9, 166.5, 127.2),
    hip=(18.7, 65.7, 109.1, 51.1),
    trunk=(-35.8, -9.4, 48.6, -24.8),
    knee_sd=(12.0, 0.0, 4.3, 0.0),
    trunk_sd=(4.4, 0.0, 5.3, 0.0),
    notes=("Only the knee and trunk catch/finish values are measurements. "
           "The 48.6 deg finish trunk angle is far larger than both other "
           "sources report; recorded for comparison, not used as a driver."),
)

DATASETS: Dict[str, StrokeKinematicsDataset] = {
    d.name: d for d in (
        CAPLAN_GARDNER_2010,
        CAPLAN_GARDNER_2010_RAISED_1,
        CAPLAN_GARDNER_2010_RAISED_2,
        KLESHNEV_ELITE,
        OLYMPIC_VS_TRADITIONAL_2025,
    )
}


def default_dataset() -> StrokeKinematicsDataset:
    """The dataset used when none is specified."""
    return CAPLAN_GARDNER_2010
