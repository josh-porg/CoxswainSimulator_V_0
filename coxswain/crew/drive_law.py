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

The law is smooth to its acceleration: the digitised speed, in time, is a cubic smoothing spline
weighted by the digitisation's own +-0.03 m/s (residual 0.030 m/s rms, the standard s = m
choice), and the angle is its integral. Interpolating the table linearly instead made the rate
piecewise constant and the acceleration a train of spikes at the table's nodes, harmless on a
1.2 kg oar but kilonewtons at the handle once the blade carries its entrained water.

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


#: the digitisation's stated uncertainty in handle speed, m/s (k05_fig1_onwater.csv header)
K05_SPEED_ERROR = 0.03


#: where the scan's turning points stop being resolved, fraction of the path (swept 0.02-0.05)
K05_TURNING_EDGE = 0.03


def turning_samples(s, v, edge: float = K05_TURNING_EDGE, n_end: int = 12):
    """``(time, path, speed)`` samples of a half-stroke whose turning points are physical.

    The scan resolves the handle speed between ``edge`` and ``1 - edge`` of the path; there it is
    kept as measured, placed in time by dt = ds / v. Into each turning point the handle is taken
    to start from rest (or come to rest) at constant acceleration, the simplest physical turning:
    from the edge speed v_e over the edge path s_e that takes 2 s_e / v_e. On [K05] this gives a
    0.95 s drive against the table's 1.00 s and a 0.91 s recovery against 0.86 s, where pinning
    the speed to zero and stretching the whole profile (``smooth_progress``) put 12% of the drive's
    time into the middle and slowed every speed by that much. Path in fractions, speed in
    fraction / unit time of the input."""
    s = np.asarray(s, float)
    v = np.asarray(v, float)
    lo = int(np.searchsorted(s, edge))
    hi = int(np.searchsorted(s, 1.0 - edge, side="right")) - 1
    mid = 0.5 * (v[lo + 1:hi + 1] + v[lo:hi])
    t_mid = np.concatenate([[0.0], np.cumsum(np.diff(s[lo:hi + 1]) / mid)])
    t_c = 2.0 * s[lo] / v[lo]
    t_f = 2.0 * (1.0 - s[hi]) / v[hi]
    tc = np.linspace(0.0, t_c, n_end, endpoint=False)
    a_c = v[lo] / t_c
    tf = np.linspace(0.0, t_f, n_end + 1)[1:]
    a_f = v[hi] / t_f
    t = np.concatenate([tc, t_c + t_mid, t_c + t_mid[-1] + tf])
    path = np.concatenate([0.5 * a_c * tc ** 2, s[lo:hi + 1],
                           s[hi] + v[hi] * tf - 0.5 * a_f * tf ** 2])
    speed = np.concatenate([a_c * tc, v[lo:hi + 1], v[hi] - a_f * tf])
    return t, path, speed


def turning_progress(s, v, edge: float = K05_TURNING_EDGE, error: float = K05_SPEED_ERROR):
    """As :func:`smooth_progress`, but with physical turning points (:func:`turning_samples`):
    the resolved speeds keep their measured share of the time. The speed, in time, is a cubic
    smoothing spline weighted by the digitisation's ``error``; its ends are the constant
    accelerations, so they are weighted as data, not pinned."""
    from scipy.interpolate import UnivariateSpline

    t, path, speed = turning_samples(s, v, edge)
    total = float(t[-1])
    tau = t / total
    rate = speed * total                                  # d(path fraction) / d(time fraction)
    weights = np.full(rate.size, 1.0 / (error * total))
    weights[0] = weights[-1] = 100.0 / (error * total)
    spline = UnivariateSpline(tau, rate, w=weights, k=3, s=float(rate.size))
    integral = spline.antiderivative()
    scale = 1.0 / float(integral(1.0))
    accel = spline.derivative()
    return (lambda x: scale * integral(x), lambda x: scale * spline(x), lambda x: scale * accel(x))


def smooth_progress(s, v, error: float = K05_SPEED_ERROR):
    """``(progress, speed, acceleration)`` callables of the drive's time fraction, smooth.

    ``progress`` runs 0 -> 1 as the handle covers the drive; ``speed`` and ``acceleration`` are
    its first two derivatives. The speed along the path, placed in time by dt = ds / v, is fitted
    with a cubic smoothing spline weighted by ``error`` (turning points pinned at zero), and the
    progress is its integral scaled to end at 1 (a 0.1% rescale on [K05]).
    """
    from scipy.interpolate import UnivariateSpline

    mid = 0.5 * (v[1:] + v[:-1])
    total = float(np.sum(np.diff(s) / np.maximum(mid, 1e-6)))
    tf, _pf = progress_in_time(s, v)
    rate = np.asarray(v, dtype=float) * total             # d(progress) / d(time fraction)
    weights = np.full(rate.size, 1.0 / (error * total))
    weights[0] = weights[-1] = 100.0 / (error * total)
    spline = UnivariateSpline(tf, rate, w=weights, k=3, s=float(rate.size))
    integral = spline.antiderivative()
    scale = 1.0 / float(integral(1.0))
    accel = spline.derivative()
    return (lambda x: scale * integral(x), lambda x: scale * spline(x), lambda x: scale * accel(x))


@dataclass(frozen=True)
class PopulationDriveSweep(OarAngleSweep):
    """An oar sweep whose drive follows a measured on-water handle-speed profile."""

    #: ``(length_fraction, handle_speed)``; default [K05] Fig. 1.
    profile: tuple = field(default=None, repr=False, compare=False)
    #: the speed error the smoothing is weighted by, m/s; the digitisation's by default
    speed_error: float = K05_SPEED_ERROR
    #: the turning points: ``"turning"`` (constant acceleration into and out of rest, the
    #: resolved speeds kept; the default since SOURCES sec. 174) or ``"pinned"`` (speed pinned to
    #: zero and the whole profile stretched to the drive's time, sec. 171)
    ends: str = "turning"
    turning_edge: float = K05_TURNING_EDGE

    def _law(self):
        cached = getattr(self, "_cached", None)
        if cached is None:
            profile = self.profile if self.profile is not None else k05_handle_speed()
            if self.ends == "pinned":
                cached = smooth_progress(*profile, error=self.speed_error)
            elif self.ends == "turning":
                cached = turning_progress(*profile, edge=self.turning_edge, error=self.speed_error)
            else:
                raise ValueError("ends must be 'turning' or 'pinned', got %r" % (self.ends,))
            object.__setattr__(self, "_cached", cached)
        return cached

    def __call__(self, t, timing):
        phase = np.asarray(timing.phase(t), dtype=float)
        drive = timing.drive_fraction
        progress = self._law()[0]
        on_drive = phase < drive
        span = self.finish_angle - self.catch_angle
        drive_progress = np.clip(phase / drive, 0.0, 1.0)
        during_drive = self.catch_angle + span * progress(drive_progress)
        during_recovery = super().__call__(t, timing)
        return np.where(on_drive, during_drive, during_recovery)

    def rate(self, t, timing):
        phase = np.asarray(timing.phase(t), dtype=float)
        drive = timing.drive_fraction
        speed = self._law()[1]
        span = self.finish_angle - self.catch_angle
        duration = drive * float(timing.period)
        during_drive = span * speed(np.clip(phase / drive, 0.0, 1.0)) / duration
        return np.where(phase < drive, during_drive, super().rate(t, timing))


@dataclass(frozen=True)
class BodyCoupledDriveSweep(PopulationDriveSweep):
    """The drive generated by the body: the grip one arm's reach from the shoulders (SOURCES sec. 169a, 208, 209).

    The hand is the shoulder plus the arm. The shoulders are the body's; the arm's reach is set by the elbow on its
    own clock (:mod:`coxswain.crew.arm_draw`): held near straight through the leg drive, then drawn in one
    minimum-jerk movement that ends at rest at the finish. ``drive`` tabulates the resulting oar angle against the
    drive's time fraction (built by ``research/crew/onwater_body.body_drive_table``); the sweep interpolates it with a
    C2 cubic spline and differentiates the spline for the rate. Because the elbow's draw starts with zero velocity and
    acceleration there is no hand-over between two laws. The population handle law is no longer an input to the
    drive, only its check. ``recovery`` does the same through the recovery (the arms' return, then held, while the
    body comes forward), against the recovery's time fraction; without it the recovery is the parent's. The drive
    table runs catch to finish and the recovery table finish to catch.

    (Before §209 this blended the population law with a straight-arm angle over 0.5 deg, which spiked the oar's
    acceleration to 5x the law's own peak where the arms began to draw.)
    """

    #: ``(drive_time_fraction, oar_angle_rad)``, as tuples
    drive: tuple = field(default=None, repr=False, compare=False)
    #: ``(recovery_time_fraction, oar_angle_rad)``, as tuples; None leaves the parent's recovery
    recovery: tuple = field(default=None, repr=False, compare=False)
    #: drive fraction at which the arm draw starts (reported; the table already contains it)
    onset: float = float("nan")

    def _spline(self, which):
        key = "_t_cached_" + which
        cached = getattr(self, key, None)
        if cached is None:
            from scipy.interpolate import CubicSpline
            table = self.drive if which == "drive" else self.recovery
            x = np.asarray(table[0], float)
            a = np.asarray(table[1], float)
            if not np.all(np.isfinite(a)):
                raise ValueError("the %s table has points the arm cannot reach (NaN)" % which)
            ends = [(self.catch_angle, "catch"), (self.finish_angle, "finish")]
            if which == "recovery":
                ends.reverse()
            for end, (want, name) in zip((a[0], a[-1]), ends):
                if abs(end - want) > np.radians(0.05):
                    raise ValueError("the %s table's %s angle %.2f deg is not the sweep's %.2f deg"
                                     % (which, name, np.degrees(end), np.degrees(want)))
            spline = CubicSpline(x, a)
            cached = (spline, spline.derivative())
            object.__setattr__(self, key, cached)
        return cached

    def _drive(self, x):
        s, ds = self._spline("drive")
        return s(x), ds(x)                                        # rate per unit drive fraction

    def _split(self, t, timing):
        phase = np.asarray(timing.phase(t), dtype=float)
        drive = timing.drive_fraction
        return phase, drive, np.clip(phase / drive, 0.0, 1.0), np.clip((phase - drive) / (1.0 - drive), 0.0, 1.0)

    def __call__(self, t, timing):
        phase, drive, x, y = self._split(t, timing)
        rec = OarAngleSweep.__call__(self, t, timing) if self.recovery is None else self._spline("recovery")[0](y)
        return np.where(phase < drive, self._drive(x)[0], rec)

    def rate(self, t, timing):
        phase, drive, x, y = self._split(t, timing)
        P = float(timing.period)
        rec = (OarAngleSweep.rate(self, t, timing) if self.recovery is None
               else self._spline("recovery")[1](y) / ((1.0 - drive) * P))
        return np.where(phase < drive, self._drive(x)[1] / (drive * P), rec)

    def arm_draw_onset(self) -> float:
        """The drive fraction at which the arm draw begins."""
        return float(self.onset)
