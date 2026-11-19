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


def sculling_recovery(rate, n=201):
    """``(tau, progress)`` through the recovery from the data author's sculling averages (sculling_averages.py; rates
    32 / 36 / 40, interpolated linearly in rate, held outside): the handle's path fraction from the finish against the
    recovery's time fraction, a cubic Hermite through the samples' positions with their speeds as slopes (``_progress``;
    SOURCES §242)."""
    import sculling_averages as SA
    rates = SA.RATES
    r = float(np.clip(rate, rates[0], rates[-1]))
    j = int(np.clip(np.searchsorted(rates, r) - 1, 0, len(rates) - 2))
    w = (r - rates[j]) / (rates[j + 1] - rates[j])
    lo, hi = SA.reference(rates[j], "recovery"), SA.reference(rates[j + 1], "recovery")
    length = (1 - w) * lo["length"] + w * hi["length"]
    speed = -((1 - w) * lo["handle_v"] + w * hi["handle_v"])     # towards the stern, positive along the recovery
    prog = _progress(100.0 - length, speed)
    tau = np.linspace(0.0, 1.0, int(n))
    return tau, prog(tau)


def drive_sweep(catch_angle, finish_angle, recovery="cosine", rate=None):
    """The oar sweep whose drive follows the population's handle velocity. ``recovery``: "cosine" -- the parent sweep's
    raised-cosine recovery (as built); "sculling" -- the data author's sculling averages at ``rate`` (strokes/min;
    ``sculling_recovery``; SOURCES §242), the drive unchanged."""
    from coxswain.crew.drive_law import BodyCoupledDriveSweep, PopulationDriveSweep
    s, v = drive_profile()
    # the first and last samples sit inside PopulationDriveSweep's default turning edge; here the turning spans are
    # the first and last 10 % of the drive's time, which the data resolve, so the edge is their path (3.2 %)
    base = PopulationDriveSweep(catch_angle=catch_angle, finish_angle=finish_angle, profile=(s, v),
                                speed_error=SPEED_ERROR, turning_edge=float(s[1]) - 1e-9)
    if recovery == "cosine":
        return base
    if recovery != "sculling":
        raise ValueError("recovery is 'cosine' or 'sculling', got %r" % (recovery,))
    if rate is None:
        raise ValueError("the sculling recovery needs the stroke rate")
    from coxswain.crew.stroke import OnWaterTiming
    timing = OnWaterTiming(float(rate))
    P, D = float(timing.period), float(timing.drive_fraction)
    x = np.linspace(0.0, 1.0, 401)
    drive = (tuple(x), tuple(float(base(u * D * P, timing)) for u in x))   # the drive's angle by its time fraction
    tau, prog = sculling_recovery(rate)
    span = float(finish_angle) - float(catch_angle)
    rec = (tuple(tau), tuple(float(finish_angle) - span * prog))
    return BodyCoupledDriveSweep(catch_angle=catch_angle, finish_angle=finish_angle, drive=drive, recovery=rec)
