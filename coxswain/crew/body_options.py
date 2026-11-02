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
  measured trunk channels better than the ergometer keyframes do (§211); ``"k05_legs"``, the seat on Kleshnev
  2005's on-water women (Fig. 1), a course of the seat's position that the rower converts to its shank angle through
  its own linkage (§213).
* **Segment travel** (:class:`SegmentTravel`). How far the legs and trunk travel, from a population, scaled by the
  rower's stature; the keyframes' excursions are scaled to it (``kinematics._fit_travel``). Registered:
  ``"k05_racing"``, Kleshnev 2005's on-water women at racing rate (SOURCES §212).
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
    #: what the course describes: ``"angle"``, the joint's own angle; ``"seat"``, the hip's fore-and-aft position
    #: (for the shank: the rower converts it to the shank angle's course through its own linkage)
    channel: str = "angle"

    def __post_init__(self):
        if self.joint not in CLOCKED_JOINTS:
            raise ValueError("a clock drives one of %s, not %r" % (CLOCKED_JOINTS, self.joint))
        if self.channel not in ("angle", "seat") or (self.channel == "seat" and self.joint != "shank"):
            raise ValueError("a clock's channel is 'angle', or 'seat' for the shank")
        c = np.asarray(self.course, float)
        if len(self.clock) != len(c) or len(c) < 8 or abs(c.min()) > 1e-9 or abs(c.max() - 1.0) > 1e-9:
            raise ValueError("a clock's course must be normalised to 0-1 on at least 8 points")

    def target(self, g):
        """The course at split-clock ``g`` (periodic)."""
        return np.interp(np.mod(np.asarray(g, float), 1.0), self.clock, self.course, period=1.0)

    def signature(self) -> tuple:
        return (self.joint, self.source, self.channel, tuple(np.round(np.asarray(self.course, float)[::8], 6)))

    @classmethod
    def from_angles(cls, joint, clock, angle, source="", channel="angle"):
        """From an angle (or position) against the split clock (any units): normalised by its own range."""
        g = np.asarray(clock, float)
        a = np.asarray(angle, float)
        o = np.argsort(g)
        g, a = g[o], a[o]
        keep = np.concatenate([[True], np.diff(g) > 1e-9])
        g, a = g[keep], a[keep]
        return cls(joint, tuple(g), tuple((a - a.min()) / np.ptp(a)), source, channel)

    @classmethod
    def kleshnev2005_legs(cls, n: int = 801) -> "SegmentClock":
        """The seat on [K05]'s on-water population (Fig. 1: five women, racing rate). The legs' velocity over the
        handle's, integrated along the handle's path, gives the seat against path fraction on the drive and the
        recovery; each half is placed in time by ``dt = ds / v_handle`` with physical turning points
        (:func:`coxswain.crew.drive_law.turning_progress`), as ``research/crew/k05_body.py`` does. On the split
        clock, so the boat's own rhythm sets the drive's share. Course of the seat's position (``channel="seat"``)."""
        from .drive_law import turning_progress
        halves = _k05_half_strokes()
        x = np.linspace(0.0, 1.0, n)
        sd, hd, pd = halves["drive"]
        sr, hr, pr = halves["recovery"]
        drive = np.interp(turning_progress(sd, hd)[0](x), sd, pd) / pd[-1]
        recovery = 1.0 - np.interp(turning_progress(sr, hr)[0](x), sr, pr) / pr[-1]
        g = np.concatenate([0.5 * x, 0.5 + 0.5 * x[1:]])
        c = np.concatenate([drive, recovery[1:]])
        return cls.from_angles("shank", g, c, "Kleshnev 2005 Fig. 1, on water, racing rate", channel="seat")

    @classmethod
    def pitto2024_trunk(cls) -> "SegmentClock":
        """The trunk link on the upper trunk's (T6) inclination, Pitto et al. 2024 Fig. 2 (digitised)."""
        rows = [l.strip().split(",") for l in open(PITTO_TRUNK_FILE, encoding="utf-8")
                if l.strip() and not l.startswith("#") and not l.startswith("segment")]
        pts = np.array([(float(g), float(v)) for s, g, v in rows if s == "trunk"])
        return cls.from_angles("trunk", pts[:, 0], pts[:, 1], "Pitto et al. 2024 Fig. 2, T6 inclination")


K05_DRIVE_FILE = os.path.join(ROOT, "data", "literature", "k05_fig1_onwater.csv")
K05_RECOVERY_FILE = os.path.join(ROOT, "data", "literature", "k05_fig1_recovery.csv")
K05_DRIVE_LENGTH = 1.59           # m, [K05] Table 1


def _k05_read(path):
    import csv
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _k05_ratio(seg, handle):
    """Segment velocity over the handle's where the handle moves; held flat into the ends, which the scan does not
    resolve (research/crew/k05_body._ratio)."""
    ok = np.abs(handle) > 0.2
    idx = np.flatnonzero(ok)
    r = np.zeros_like(seg)
    r[ok] = seg[ok] / handle[ok]
    return np.interp(np.arange(seg.size), idx, r[idx])


def _k05_half_strokes(segment="legs_velocity"):
    """Per half-stroke: (handle path fraction, increasing in time; handle speed >= 0; segment position, m)."""
    d, r = _k05_read(K05_DRIVE_FILE), _k05_read(K05_RECOVERY_FILE)
    s = d["length_pct"] / 100.0
    hd = d["handle_speed"].copy()
    hd[0] = hd[-1] = 0.0
    ratio = _k05_ratio(d[segment], hd)
    pos_d = np.concatenate([[0.0], np.cumsum(0.5 * (ratio[1:] + ratio[:-1]) * np.diff(s) * K05_DRIVE_LENGTH)])
    sr = (1.0 - r["length_pct"] / 100.0)[::-1]
    hr = -r["handle_speed"][::-1]
    hr[0] = hr[-1] = 0.0
    ratio = _k05_ratio(-r[segment][::-1], hr)
    pos_r = np.concatenate([[0.0], np.cumsum(0.5 * (ratio[1:] + ratio[:-1]) * np.diff(sr) * K05_DRIVE_LENGTH)])
    return {"drive": (s, hd, pos_d), "recovery": (sr, hr, pos_r)}


#: clocks by name, built on demand
CLOCKS = {"pitto2024_trunk": SegmentClock.pitto2024_trunk, "k05_legs": SegmentClock.kleshnev2005_legs}


@dataclass(frozen=True)
class SegmentTravel:
    """How far the legs and the trunk travel, from a population, scaled to the rower's stature.

    ``legs``: the hip's fore-and-aft travel (the seat's); ``trunk``: the shoulder's travel relative to the hip; both
    in metres for rowers of ``stature``. The rower scales its keyframes' shank and trunk excursions about their means
    (``stroke_data.scale_leg_amplitude`` / ``scale_trunk_amplitude``) until it travels ``stature_rower / stature``
    times as far (``kinematics._fit_travel``). The postures' means are kept; only the range changes.
    """

    legs: float
    trunk: float
    stature: float
    source: str = ""

    def __post_init__(self):
        if not (self.legs > 0 and self.trunk > 0 and self.stature > 0):
            raise ValueError("travels and stature are positive lengths")

    def targets(self, stature: float):
        k = float(stature) / self.stature
        return self.legs * k, self.trunk * k

    def signature(self) -> tuple:
        return (round(self.legs, 9), round(self.trunk, 9), round(self.stature, 9), self.source)

    @classmethod
    def kleshnev2005_racing(cls) -> "SegmentTravel":
        """[K05] Table 1 (``data/literature/kleshnev2005_onwater_vs_machines.csv``): five women, 1.80 m, single scull
        on water at racing rate (32.3 spm): legs 0.51 m, trunk 0.48 m. Their trunk line was fixed between L1 and C7;
        the model reads its trunk at the shoulder (the caveat of SOURCES §169a)."""
        return cls(0.51, 0.48, 1.80, "Kleshnev 2005 Table 1, on water, racing rate")


#: travels by name
TRAVELS = {"k05_racing": SegmentTravel.kleshnev2005_racing}


@dataclass(frozen=True)
class BodyOptions:
    """Segment clocks, segment travel and the reach allowance; the default changes nothing."""

    clocks: tuple = ()
    reach_allowance: float = None
    travel: SegmentTravel = None

    def __post_init__(self):
        joints = [c.joint for c in self.clocks]
        if len(set(joints)) != len(joints):
            raise ValueError("one clock per joint")
        if self.reach_allowance is not None and not self.reach_allowance >= 0.0:
            raise ValueError("the reach allowance is a distance, >= 0")

    @property
    def is_default(self) -> bool:
        return not self.clocks and self.reach_allowance is None and self.travel is None

    def clock_for(self, joint):
        return next((c for c in self.clocks if c.joint == joint), None)

    def signature(self) -> tuple:
        return (tuple(c.signature() for c in self.clocks), self.reach_allowance,
                None if self.travel is None else self.travel.signature())


DEFAULT_BODY = BodyOptions()


def body_options(clocks=(), reach_allowance=None, travel=None) -> BodyOptions:
    """``BodyOptions`` from clock and travel names (``CLOCKS``, ``TRAVELS``) or objects."""
    built = tuple(CLOCKS[c]() if isinstance(c, str) else c for c in clocks)
    if isinstance(travel, str):
        travel = TRAVELS[travel]()
    return BodyOptions(clocks=built, reach_allowance=reach_allowance, travel=travel)
