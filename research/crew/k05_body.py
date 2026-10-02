"""The body of [K05]'s on-water population: its legs and trunk, on the boat's clock.

[K05] Fig. 1 gives, for five women at racing rate on the water, the velocities of the legs (seat
relative to the boat), the trunk (shoulder relative to the seat) and the arms against the handle's
drive length, on the drive and (``data/literature/k05_fig1_recovery.csv``) on the recovery. Each
segment's position along the stroke is its velocity over the handle's, integrated along the
handle path; each half-stroke is placed in time by dt = ds / v_handle, smoothed within the
digitisation's error, with physical turning points (coxswain.crew.drive_law.turning_progress), and the boat's own drive fraction.

The result is two channels on the catch clock, normalised 0 -> 1, in the form
research/biorow/measured_body.py's time-warp takes (there for [BR24]'s measured seat and trunk):
the model's own postures are re-timed so its hip - ankle and shoulder - hip follow these, and the
back's travel is scaled to [K05]'s 0.48 m. The field's derivatives are exact through the warp, so
the hull's momentum books close. Nothing is fitted.

Checks (research/k05/extract_fig1*.py): drive travels legs 0.510 / trunk 0.456 m, recovery
0.507 / 0.485 m, against [K05]'s table 0.51 / 0.48.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "research", "biorow")]
import measured_body as MB                                       # noqa: E402

from coxswain.crew.drive_law import turning_progress          # noqa: E402

DRIVE = os.path.join(ROOT, "data", "literature", "k05_fig1_onwater.csv")
RECOVERY = os.path.join(ROOT, "data", "literature", "k05_fig1_recovery.csv")
K05_TRUNK_TRAVEL = 0.48            # m, table row 18, racing rate
K05_LEG_TRAVEL = 0.51              # m, table row 17, racing rate (seat on boat)
LENGTH = 1.59                      # m, drive length, table row 5


def _read(path):
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _ratio(seg, handle):
    """Segment velocity over handle velocity, where the handle moves; held flat into the ends,
    where the scan does not resolve either."""
    ok = np.abs(handle) > 0.2
    idx = np.flatnonzero(ok)
    r = np.zeros_like(seg)
    r[ok] = seg[ok] / handle[ok]
    return np.interp(np.arange(seg.size), idx, r[idx])


def half_strokes():
    """Per half-stroke: (handle path fraction s, increasing in time; handle speed >= 0;
    {segment: position along s, m})."""
    d, r = _read(DRIVE), _read(RECOVERY)
    s = d["length_pct"] / 100.0
    hd = d["handle_speed"].copy()
    hd[0] = hd[-1] = 0.0
    out = {}
    pos = {}
    for seg in ("legs_velocity", "trunk_velocity"):
        pos[seg] = np.concatenate([[0.0], np.cumsum(0.5 * (_ratio(d[seg], hd)[1:] + _ratio(d[seg], hd)[:-1])
                                                    * np.diff(s) * LENGTH)])
    out["drive"] = (s, hd, pos)
    # the recovery returns the handle 100% -> 0%: put it in its own time order
    sr = (1.0 - r["length_pct"] / 100.0)[::-1]
    hr = -r["handle_speed"][::-1]
    hr[0] = hr[-1] = 0.0
    pos = {}
    for seg in ("legs_velocity", "trunk_velocity"):
        v = -r[seg][::-1]
        ratio = _ratio(v, hr)
        pos[seg] = np.concatenate([[0.0], np.cumsum(0.5 * (ratio[1:] + ratio[:-1]) * np.diff(sr) * LENGTH)])
    out["recovery"] = (sr, hr, pos)
    return out


def channel(seg, drive_fraction, n=4001):
    """``(phase of the minimum, phase grid since it, normalised curve, travel)`` on the catch
    clock, as measured_body.measured_channel returns for [BR24]."""
    halves = half_strokes()
    sd, hd, pd = halves["drive"]
    sr, hr, pr = halves["recovery"]
    prog_d = turning_progress(sd, hd)[0]
    prog_r = turning_progress(sr, hr)[0]
    phase = np.linspace(0.0, 1.0, n, endpoint=False)
    D = float(drive_fraction)
    c = np.empty(n)
    on = phase < D
    xd = np.interp(prog_d(phase[on] / D), sd, pd[seg])
    travel_d = float(pd[seg][-1])
    xr = np.interp(prog_r((phase[~on] - D) / (1.0 - D)), sr, pr[seg])
    # close the loop: the recovery returns the segment by the drive's travel
    c[on] = xd / travel_d
    c[~on] = 1.0 - xr / float(pr[seg][-1])
    k = int(np.argmin(c))
    m = phase[k]
    c = np.roll(c, -k)
    return m, phase, (c - c.min()) / np.ptp(c), travel_d


def leg_weights(boat):
    """Per segment, the slope of its fore-aft position on the lower trunk's over the model's own
    cycle: how much of the seat's travel each segment carries (feet 0, trunk 1)."""
    P = float(boat.timing.period)
    t = np.linspace(0.0, P, 400, endpoint=False)
    x = np.array([np.asarray(boat.crew_field(float(u), exact=True)[1], float)[:, 0] for u in t])
    ref = x[:, MB.REF] - x[:, MB.REF].mean()
    w = (x - x.mean(axis=0)).T @ ref / (ref @ ref)
    return w, float(x[:, MB.REF].mean())


def body_field(boat, leg_travel=None, trunk_travel=K05_TRUNK_TRAVEL):
    """The model's body re-timed onto [K05]'s legs and trunk, back travel scaled to [K05]'s.

    ``leg_travel`` (m): also scale the seat's travel to it, each segment's fore-aft motion by its
    share of the seat's (``leg_weights``), about the cycle mean. Linear in the motion, so the
    velocities and accelerations scale with it and the momentum books still close. ``None``
    keeps the model's own leg travel (0.60 m for a 1.80 m body, against [K05]'s 0.51).
    ``trunk_travel`` (m): the back's travel, shoulder on seat; [K05]'s 0.48 by default."""
    D = float(boat.timing.drive_fraction)
    m_l, s_l, c_l, _ = channel("legs_velocity", D)
    mm_l, tau_l, cm_l, model_leg = MB.model_channel(boat, lambda j: j["hip"][0] - j["ankle"][0])
    leg_raw = MB.channel_field(boat, MB.warp(s_l, c_l, tau_l, cm_l), m_l, mm_l)
    if leg_travel is None:
        leg = leg_raw
    else:
        k_leg = float(leg_travel) / model_leg
        w, ref_mean = leg_weights(boat)

        def leg(t):
            mass, pos, vel, acc = leg_raw(t)
            pos, vel, acc = np.array(pos, float), np.array(vel, float), np.array(acc, float)
            n = pos.shape[0] - pos.shape[0] % 12
            for off in range(0, n, 12):
                r = off + MB.REF
                d, dv, da = pos[r, 0] - ref_mean, vel[r, 0], acc[r, 0]
                for i in range(12):
                    pos[off + i, 0] -= (1.0 - k_leg) * w[off + i] * d
                    vel[off + i, 0] -= (1.0 - k_leg) * w[off + i] * dv
                    acc[off + i, 0] -= (1.0 - k_leg) * w[off + i] * da
            return mass, pos, vel, acc
    m_b, s_b, c_b, _ = channel("trunk_velocity", D)
    mm_b, tau_b, cm_b, model_travel = MB.model_channel(boat, lambda j: j["shoulder"][0] - j["hip"][0])
    bk = MB.channel_field(boat, MB.warp(s_b, c_b, tau_b, cm_b), m_b, mm_b)
    K = float(trunk_travel) / model_travel

    def field(t, exact=False):
        mass, pos, vel, acc = leg(t)
        _, pb, vb, ab = bk(t)
        for off in range(0, vel.shape[0] - vel.shape[0] % 12, 12):
            idx = [off + i for i in MB.UPPER]
            r = off + MB.REF
            pos[idx] = pos[r] + K * (pb[idx] - pb[r])
            vel[idx] = vel[r] + K * (vb[idx] - vb[r])
            acc[idx] = acc[r] + K * (ab[idx] - ab[r])
        return mass, pos, vel, acc
    return field, K
