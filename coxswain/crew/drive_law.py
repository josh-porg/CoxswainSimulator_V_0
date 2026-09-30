"""The hands' drive time law from a measured on-water population, in place of a chosen shape.

:class:`~coxswain.crew.oarlock.OarAngleSweep` moves the oar through the drive on a raised
cosine in time: a chosen shape, and the one SOURCES sec. 167 and 169 found wrong on the water
(it turns the oar too fast early and peaks at mid-drive, where on-water handles peak at 59-65%
of the drive length).

:class:`PopulationDriveSweep` takes the drive from [K05] Fig. 1 instead: five female scullers
at racing rate on the water, handle speed against drive length, digitised in
``data/literature/k05_fig1_onwater.csv`` (the digitisation reproduces [K05]'s own tabulated
segment travels). Along the handle's path, time follows from dt = ds / v(s); the oar angle is
proportional to the path (s = r_h (phi_catch - phi)), so the fraction of the arc swept is the
fraction of the drive length covered. Only the *shape* is taken from the figure: the drive's
duration is the boat's own timing (e.g. [K05]'s on-water rhythm), because the scan cannot
resolve the handle speed at the two turning points, where the time integral is most sensitive;
there the speed is set to zero. The recovery keeps the parent's raised cosine.

A research option; nothing in the shipped game uses it.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

from .oarlock import OarAngleSweep

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K05_FIG1 = os.path.join(_ROOT, "data", "literature", "k05_fig1_onwater.csv")


def k05_handle_speed(path: str = K05_FIG1):
    """``(length_fraction, handle_speed)`` of the on-water drive, turning points at zero speed."""
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    s = np.array([float(r["length_pct"]) for r in rows]) / 100.0
    v = np.array([float(r["handle_speed"]) for r in rows])
    v[0] = v[-1] = 0.0
    return s, v


def progress_in_time(s, v):
    """``(time_fraction, length_fraction)`` from dt = ds / v, midpoint speeds, normalised."""
    mid = 0.5 * (v[1:] + v[:-1])
    dt = np.diff(s) / np.maximum(mid, 1e-6)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    return t / t[-1], s


@dataclass(frozen=True)
class PopulationDriveSweep(OarAngleSweep):
    """An oar sweep whose drive follows a measured on-water handle-speed profile."""

    #: ``(time_fraction, length_fraction)``; default [K05] Fig. 1.
    profile: tuple = field(default=None, repr=False, compare=False)

    def _table(self):
        if self.profile is not None:
            return self.profile
        cached = getattr(self, "_cached", None)
        if cached is None:
            cached = progress_in_time(*k05_handle_speed())
            object.__setattr__(self, "_cached", cached)
        return cached

    def __call__(self, t, timing):
        phase = np.asarray(timing.phase(t), dtype=float)
        drive = timing.drive_fraction
        tf, pf = self._table()
        on_drive = phase < drive
        span = self.finish_angle - self.catch_angle
        drive_progress = np.clip(phase / drive, 0.0, 1.0)
        during_drive = self.catch_angle + span * np.interp(drive_progress, tf, pf)
        during_recovery = super().__call__(t, timing)
        return np.where(on_drive, during_drive, during_recovery)

    def rate(self, t, timing):
        step = 1e-4 * float(timing.period)
        return (np.asarray(self(np.asarray(t) + step, timing))
                - np.asarray(self(np.asarray(t) - step, timing))) / (2.0 * step)
