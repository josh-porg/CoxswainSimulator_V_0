r"""Options that change how a rower's body moves, kept in one place so new pieces plug in without new plumbing.

A :class:`BodyOptions` travels from :class:`~coxswain.boats.boat.Boat` to every
:class:`~coxswain.crew.kinematics.JointDrivenRower` and into the rower's kinematics signature (so every cache keyed
on it, the matched-torque cache among them, sees it). The default reproduces the body exactly as it was before the
options existed, and adds nothing to the signature, so no existing cache is invalidated. No catalogue boat or profile
sets one; they are research options (SOURCES §211).

What it carries now:

* **Segment clocks** (:class:`SegmentClock`). A joint's *sourced* time course through the stroke, normalised 0 to 1,
  on the split clock (0 the catch, 0.5 the finish, 1 the next catch: drive and recovery each scaled to their own
  length). The rower keeps the joint's keyframe amplitudes and redistributes its motion in time so its normalised
  course follows the clock (``kinematics._retime_to_course``). Registered: ``"pitto2024_trunk"``, the upper trunk's
  inclination of 12 national-level scullers on water (Pitto et al. 2024 Fig. 2), which matches both test athletes'
  measured trunk channels better than the ergometer keyframes do (§211).
* **Reach allowance** (m). How far the hand may lie beyond a fully straight arm before the elbow solve refuses the
  rig. ``None`` keeps the shipped value (``kinematics.MAX_SHOULDER_PROTRACTION``, 6 cm). A hand model that lengthens
  the chain under load (``coxswain.crew.hand_chain``) sets it from its own largest extra reach.

To add a piece: give it a frozen dataclass with a ``signature()``, add a field here with a default that changes
nothing, and read the field where it acts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PITTO_TRUNK_FILE = os.path.join(ROOT, "data", "literature", "pitto2024_trunk_inclination.csv")

#: joints a clock may drive (the keyframe drivers in kinematics.JointAngles)
CLOCKED_JOINTS = ("shank", "thigh", "trunk")


@dataclass(frozen=True)
class SegmentClock:
    """A joint's normalised time course on the split clock (see the module notes)."""

    joint: str
    clock: tuple = field(repr=False)     # split-clock grid in [0, 1]
    course: tuple = field(repr=False)    # 0 at the course's catch-side extreme, 1 at its finish-side extreme
    source: str = ""

    def __post_init__(self):
        if self.joint not in CLOCKED_JOINTS:
            raise ValueError("a clock drives one of %s, not %r" % (CLOCKED_JOINTS, self.joint))
        c = np.asarray(self.course, float)
        if len(self.clock) != len(c) or len(c) < 8 or abs(c.min()) > 1e-9 or abs(c.max() - 1.0) > 1e-9:
            raise ValueError("a clock's course must be normalised to 0-1 on at least 8 points")

    def target(self, g):
        """The course at split-clock ``g`` (periodic)."""
        return np.interp(np.mod(np.asarray(g, float), 1.0), self.clock, self.course, period=1.0)

    def signature(self) -> tuple:
        return (self.joint, self.source, tuple(np.round(np.asarray(self.course, float)[::8], 6)))

    @classmethod
    def from_angles(cls, joint, clock, angle, source=""):
        """From an angle against the split clock (any units): normalised by its own range."""
        g = np.asarray(clock, float)
        a = np.asarray(angle, float)
        o = np.argsort(g)
        g, a = g[o], a[o]
        keep = np.concatenate([[True], np.diff(g) > 1e-9])
        g, a = g[keep], a[keep]
        return cls(joint, tuple(g), tuple((a - a.min()) / np.ptp(a)), source)

    @classmethod
    def pitto2024_trunk(cls) -> "SegmentClock":
        """The trunk link on the upper trunk's (T6) inclination, Pitto et al. 2024 Fig. 2 (digitised)."""
        rows = [l.strip().split(",") for l in open(PITTO_TRUNK_FILE, encoding="utf-8")
                if l.strip() and not l.startswith("#") and not l.startswith("segment")]
        pts = np.array([(float(g), float(v)) for s, g, v in rows if s == "trunk"])
        return cls.from_angles("trunk", pts[:, 0], pts[:, 1], "Pitto et al. 2024 Fig. 2, T6 inclination")


#: clocks by name, built on demand
CLOCKS = {"pitto2024_trunk": SegmentClock.pitto2024_trunk}


@dataclass(frozen=True)
class BodyOptions:
    """Segment clocks and the reach allowance; the default changes nothing."""

    clocks: tuple = ()
    reach_allowance: float = None

    def __post_init__(self):
        joints = [c.joint for c in self.clocks]
        if len(set(joints)) != len(joints):
            raise ValueError("one clock per joint")
        if self.reach_allowance is not None and not self.reach_allowance >= 0.0:
            raise ValueError("the reach allowance is a distance, >= 0")

    @property
    def is_default(self) -> bool:
        return not self.clocks and self.reach_allowance is None

    def clock_for(self, joint):
        return next((c for c in self.clocks if c.joint == joint), None)

    def signature(self) -> tuple:
        return (tuple(c.signature() for c in self.clocks), self.reach_allowance)


DEFAULT_BODY = BodyOptions()


def body_options(clocks=(), reach_allowance=None) -> BodyOptions:
    """``BodyOptions`` from clock names (``CLOCKS``) or clock objects."""
    built = tuple(CLOCKS[c]() if isinstance(c, str) else c for c in clocks)
    return BodyOptions(clocks=built, reach_allowance=reach_allowance)
