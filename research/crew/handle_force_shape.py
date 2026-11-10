"""The handle force's shape through the drive, model and measured, for rung 2 and the validation ladder (§227-§228).

One definition for both sides: the force per oar from the catch (the first wet sample) to the release, at fixed
fractions of that time, its peak and where, its impulse-weighted centre, and its least value between 10 % and 80 % of
the drive (a rower does not push the handle away mid-drive). [BR24]'s measured cycle is BioRow's M1x file (handle
forces H1, H2; the catch at the oar's least angle); the drive is the catch to the force's last positive sample.
"""
from __future__ import annotations

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRACTIONS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def model_stroke(sim, run):
    """``(t from the stroke's start, handle force per oar N, wet)`` for the run's last stroke, hands on the handle."""
    t, y = run.last_time, run.last_states
    inboard = float(sim.boat.rig.seats[sim._seats[0]].oarlocks[0].oar.inboard)
    force = np.array([-float(np.asarray(sim.handle_torques(t[k], y[:, k], sim._air_mask[:, k]))[0, 0]) / inboard
                      for k in range(t.size)])
    return t - t[0], force, ~np.asarray(sim._air_mask[0], dtype=bool)


def shape(t, force, wet):
    """The drive's force shape (see the module docstring), as a dict."""
    idx = np.flatnonzero(wet)
    i0, i1 = int(idx[0]), int(idx[-1])
    tt, ff = np.asarray(t[i0:i1 + 1], float) - float(t[i0]), np.asarray(force[i0:i1 + 1], float)
    span = float(tt[-1])
    pos = np.clip(ff, 0.0, None)
    mid = (tt >= 0.1 * span) & (tt <= 0.8 * span)
    return dict(span=span, at={"%g" % q: float(np.interp(q * span, tt, ff)) for q in FRACTIONS},
                peak=float(ff.max()), peak_at=float(tt[np.argmax(ff)] / span),
                centre=float(np.sum(pos * tt) / np.sum(pos) / span) if pos.sum() else float("nan"),
                mid_min=float(ff[mid].min()) if mid.any() else float("nan"))


def measured_br24():
    """[BR24]'s measured cycle: ``(t from the catch, mean handle force per oar N, wet)``."""
    import csv  # noqa: F401  (numpy reads it)
    path = os.path.join(ROOT, "data", "local", "biorow", "M1x_R32.csv")
    rate = 32.4093                                               # research/biorow/like_for_like.py RATE
    D = np.genfromtxt(path, delimiter=",", names=True)[:-1]
    n = len(D)
    k = int(np.argmin(0.5 * (D["A1"] + D["A2"])))
    order = np.roll(np.arange(n), -k)
    t = np.arange(n) * (60.0 / rate) / n
    force = (0.5 * (D["H1"] + D["H2"]))[order]
    peak = int(np.argmax(force))
    after = np.flatnonzero(force[peak:] <= 0.0)
    end = peak + (int(after[0]) - 1 if after.size else n - 1 - peak)
    return t, force, np.arange(n) <= end
