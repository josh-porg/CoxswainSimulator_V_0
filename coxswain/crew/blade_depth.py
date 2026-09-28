"""Blade depth through the drive, and the force it costs or gains.

A blade's load depends on how deep it is. Grift's plate (thesis 2020, Fig. 2.4, digitised
in ``data/literature/grift2020_cd_vs_depth.csv``) has its steady drag coefficient at 1.10
with the top edge at the surface, 1.60 a fifth of a plate height down, and 1.30 deep; a
plate piercing the surface behaves like a fence (C_D about 1.1) on its submerged area.

The dynamic oar's blade force has so far been depth-blind: the fitted C2 is an average over
whatever depth the athletes it was fitted to rowed at. This module supplies a factor that
multiplies the blade force at each point of the drive:

    factor(u) = wetted(u) * C_D(cover(u)) / C_ref

where ``u`` is progress through the drive (0 at the catch angle, 1 at the finish), the
blade centre's height above the water follows from a vertical-oar-angle profile
``z_c(u) = r_b sin(V(u)) + z_0``, ``cover`` is the depth of the blade's top edge, and
``wetted`` the submerged fraction of its width.

Named choices, each marked ``chosen`` in the ledger and swept, never presented as sourced:

* ``zero_offset`` (z_0): where the vertical angle's zero puts the blade centre. Default 0,
  BioRow's convention that catch slip is the blade centre crossing the water level.
* ``reference``: ``"mean"`` normalises C_ref so the factor averages 1 over the drive,
  keeping a fitted C2's level and changing only its shape; ``"deep"`` uses Grift's deep
  1.30, so depth changes the level too.

The vertical-angle profile is the athlete's: [BR24] measures one (commercial, loaded at run
time from ``data/local``), and ``from_samples`` accepts any other.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRIFT_TABLE = os.path.join(_ROOT, "data", "literature", "grift2020_cd_vs_depth.csv")


def grift_curve(path: str = GRIFT_TABLE):
    """``(depth / plate height, C_D)`` from the digitised Fig. 2.4, sorted by depth."""
    rows = [r for r in csv.DictReader(l for l in open(path, encoding="utf-8")
                                      if not l.startswith("#"))]
    depth = np.array([float(r["depth_over_plate_height"]) for r in rows])
    cd = np.array([float(r["cd"]) for r in rows])
    o = np.argsort(depth)
    return depth[o], cd[o]


@dataclass
class BladeDepth:
    """Blade depth against drive progress, and the resulting force factor."""

    #: drive progress samples in [0, 1] and the vertical oar angle there, radians,
    #: positive = blade up (BioRow's sign)
    progress: np.ndarray
    vertical_angle: np.ndarray
    #: pin to blade centre, m, and the blade's vertical extent, m
    lever: float
    width: float
    zero_offset: float = 0.0
    reference: str = "mean"
    curve: tuple = field(default=None, repr=False)

    def __post_init__(self):
        self.progress = np.asarray(self.progress, dtype=float)
        self.vertical_angle = np.asarray(self.vertical_angle, dtype=float)
        if self.reference not in ("mean", "deep"):
            raise ValueError("reference must be 'mean' or 'deep', got %r" % (self.reference,))
        if self.curve is None:
            self.curve = grift_curve()
        self._c_ref = 1.0
        if self.reference == "deep":
            self._c_ref = float(self.curve[1][-1])
        else:
            u = np.linspace(0.0, 1.0, 201)
            raw = self._raw(u)
            wet = raw > 0
            self._c_ref = float(np.mean(raw[wet])) if wet.any() else 1.0

    @classmethod
    def from_samples(cls, progress, vertical_angle_deg, lever, width, **kw):
        return cls(np.asarray(progress, float), np.radians(np.asarray(vertical_angle_deg, float)),
                   float(lever), float(width), **kw)

    @classmethod
    def constant_cover(cls, cover: float, lever: float, width: float, **kw):
        """A blade held at one cover through the whole drive: the top edge ``cover`` metres
        below the still surface (0: at the surface; negative: piercing it)."""
        centre = -(float(cover) + 0.5 * float(width))
        if abs(centre) >= float(lever):
            raise ValueError("a cover of %.3f m is not reachable on a %.3f m lever" % (cover, lever))
        v = float(np.arcsin(centre / float(lever)))
        return cls(np.array([0.0, 1.0]), np.array([v, v]), float(lever), float(width), **kw)

    def key(self) -> tuple:
        """Everything the factor depends on, for a cache key."""
        return (np.asarray(self.progress, float).tobytes(),
                np.asarray(self.vertical_angle, float).tobytes(), float(self.lever),
                float(self.width), float(self.zero_offset), str(self.reference))

    @classmethod
    def for_oar(cls, oar, progress, vertical_angle_deg, **kw):
        """From a rig ``Oar``: lever = blade centre, width = blade area / blade length."""
        width = float(oar.blade_area) / float(oar.blade_length)
        return cls.from_samples(progress, vertical_angle_deg, oar.blade_centre_outboard, width, **kw)

    # -- geometry ----------------------------------------------------------
    def centre_height(self, u):
        """Blade centre above the water, m."""
        v = np.interp(np.clip(u, 0.0, 1.0), self.progress, self.vertical_angle)
        return self.lever * np.sin(v) + self.zero_offset

    def cover(self, u):
        """Depth of the blade's top edge below the water, m (negative: top edge out)."""
        return -(self.centre_height(u) + 0.5 * self.width)

    def wetted(self, u):
        """Submerged fraction of the blade's width."""
        bottom = self.centre_height(u) - 0.5 * self.width
        return np.clip(-bottom / self.width, 0.0, 1.0)

    # -- force -------------------------------------------------------------
    def _raw(self, u):
        depth, cd = self.curve
        c = self.cover(u) / self.width
        return self.wetted(u) * np.interp(c, depth, cd, left=cd[0], right=cd[-1])

    def factor(self, u):
        """Multiplier on the blade force at drive progress ``u``."""
        return self._raw(u) / self._c_ref


def br24_profile(path=None, oar_convention_catch_negative=True):
    """[BR24]'s vertical oar angle against drive progress (catch -> finish), from the local
    commercial file. Returns ``(progress, vertical_deg)`` over the drive, or ``None`` if the
    file is absent."""
    path = path or os.path.join(_ROOT, "data", "local", "biorow", "M1x_R32.csv")
    if not os.path.exists(path):
        return None
    d = np.genfromtxt(path, delimiter=",", names=True)[:-1]
    a = 0.5 * (d["A1"] + d["A2"])
    v = 0.5 * (d["V1"] + d["V2"])
    i0, i1 = int(np.argmin(a)), int(np.argmax(a))
    idx = np.arange(i0, i0 + ((i1 - i0) % len(a)) + 1) % len(a)
    ang = a[idx]
    u = (ang - ang[0]) / (ang[-1] - ang[0])
    u = np.maximum.accumulate(u)
    keep = np.concatenate([[True], np.diff(u) > 1e-9])
    return u[keep], v[idx][keep]
