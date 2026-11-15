"""Kleshnev's population reference curves (RBN 2023/12 Fig. 1), as extracted by extract_rbn2023.py (SOURCES §237).

Per 10 % of the drive's time, >50 000 samples: handle force and handle velocity as % of their peaks, seat and
upper-body (trunk + arms) velocity as % of the peak handle velocity, the vertical oar angle, and the drive length (the
handle velocity's integral). The numbers live with the newsletter in data/local (shared by its author).
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(ROOT, "data", "local", "literature", "biorow_newsletters", "rbn2023_fig1.csv")
KEYS = ("time", "force", "vertical", "handle_v", "seat_v", "upper_v", "length")
#: the figure is vector, so the reading error is the plot's own rounding: about half a point of peak
SPEED_ERROR = 0.005


def reference():
    """The curves as a dict of arrays (all in %), keys ``KEYS``."""
    d = np.loadtxt(CSV, delimiter=",", comments="#", skiprows=6)
    return {k: d[:, i] for i, k in enumerate(KEYS)}


def drive_profile():
    """``(length_fraction, handle_speed)`` for coxswain.crew.drive_law.PopulationDriveSweep(profile=...): the
    population's handle velocity (fraction of its peak, zero at both turning points) against drive length."""
    ref = reference()
    v = np.clip(ref["handle_v"] / 100.0, 0.0, None)
    v[0] = v[-1] = 0.0
    return ref["length"] / 100.0, v


CSV2 = os.path.join(ROOT, "data", "local", "literature", "biorow_newsletters", "rbn2023_fig2.csv")
TAU = np.linspace(0.0, 1.0, 11)                                  # the samples: one per 10 % of the phase's time


def fig2(group):
    """Fig. 2's joints for ``group`` ("scull20", "scull40", "sweep20", "sweep40"): ``{(quantity, phase): (length %,
    value %)}`` with 11 samples each, in time order."""
    import csv
    rows = list(csv.DictReader(l for l in open(CSV2, encoding="utf-8") if not l.startswith("#")))
    out = {}
    for q in ("handle", "seat", "upper"):
        for ph in ("drive", "recovery"):
            sel = sorted((r for r in rows if r["group"] == group and r["quantity"] == q and r["phase"] == ph),
                         key=lambda r: int(r["k"]))
            out[(q, ph)] = (np.array([float(r["length_pct"]) for r in sel]), np.array([float(r["value_pct"]) for r in sel]))
    return out


def _progress(length, speed):
    """The handle's path fraction through one phase as a cubic Hermite spline of the phase's time fraction: through
    the joints' positions with the joints' speeds as slopes (zero at both turning points). The joints carry both, so
    the spline keeps what a ramp between speed samples loses: scullers cover 4.0 % of the drive in its first tenth
    where a steady acceleration to the same speed covers 3.5 % -- the hands turn round sharply (SOURCES §238). The
    speeds' scale (path fraction per unit time per %) comes from the eight interior spans."""
    from scipy.interpolate import CubicHermiteSpline
    s = np.asarray(length, float).copy()
    s = (s - s[0]) / (s[-1] - s[0])
    v = np.clip(np.asarray(speed, float), 0.0, None)
    v[0] = v[-1] = 0.0
    c = (s[9] - s[1]) / np.sum(0.5 * (v[1:9] + v[2:10]) * 0.1)
    return CubicHermiteSpline(TAU, s, c * v)


def scull_cycle(rate, n=201):
    """Scullers' reference cycle at ``rate`` (strokes/min), Fig. 2's rate-20 and rate-40 curves interpolated linearly
    in rate (held outside 20-40). Per phase ("drive", "recovery"): ``tau`` (time fraction, ``n`` points), ``progress``
    (handle path fraction along the phase), and ``handle``, ``seat``, ``upper`` (velocities in the phase's direction,
    fractions of the peak drive handle velocity)."""
    from scipy.interpolate import PchipInterpolator
    w = float(np.clip((float(rate) - 20.0) / 20.0, 0.0, 1.0))
    lo, hi = fig2("scull20"), fig2("scull40")
    tau = np.linspace(0.0, 1.0, int(n))
    out = {}
    for ph, sign in (("drive", 1.0), ("recovery", -1.0)):
        mix = {q: tuple((1.0 - w) * a + w * b for a, b in zip(lo[(q, ph)], hi[(q, ph)])) for q in ("handle", "seat", "upper")}
        length, hv = mix["handle"]
        path = length if ph == "drive" else 100.0 - length       # along the phase: 0 at its start
        prog = _progress(path, sign * hv)
        cols = dict(tau=tau, progress=prog(tau), spline=prog)
        for q in ("handle", "seat", "upper"):
            v = sign * mix[q][1] / 100.0
            if q == "handle":
                v = np.clip(v, 0.0, None)
                v[0] = v[-1] = 0.0
            cols[q] = PchipInterpolator(TAU, v)(tau)
        out[ph] = cols
    return out


def scull_sweep(catch_angle, finish_angle, rate):
    """The oar sweep on the scullers' reference cycle at ``rate``: drive and recovery (``scull_cycle``), as a table
    sweep (coxswain.crew.drive_law.BodyCoupledDriveSweep interpolates each table with a C2 spline)."""
    from coxswain.crew.drive_law import BodyCoupledDriveSweep
    cyc = scull_cycle(rate)
    span = float(finish_angle) - float(catch_angle)
    drive = (tuple(cyc["drive"]["tau"]), tuple(float(catch_angle) + span * cyc["drive"]["progress"]))
    recovery = (tuple(cyc["recovery"]["tau"]), tuple(float(finish_angle) - span * cyc["recovery"]["progress"]))
    return BodyCoupledDriveSweep(catch_angle=catch_angle, finish_angle=finish_angle, drive=drive, recovery=recovery)


def drive_sweep(catch_angle, finish_angle):
    """The oar sweep whose drive follows the population's handle velocity (the recovery is the parent's)."""
    from coxswain.crew.drive_law import PopulationDriveSweep
    s, v = drive_profile()
    # the first and last samples sit inside PopulationDriveSweep's default turning edge; here the turning spans are
    # the first and last 10 % of the drive's time, which the data resolve, so the edge is their path (3.2 %)
    return PopulationDriveSweep(catch_angle=catch_angle, finish_angle=finish_angle, profile=(s, v),
                                speed_error=SPEED_ERROR, turning_edge=float(s[1]) - 1e-9)
