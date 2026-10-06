r"""NK CoxBox CSV exports, and lining a session up with a cox-camera video.

The CSV ("Sharing file  CoxBox <serial> <yyyymmdd> <hhmm>AM.csv", NK LiNK Logbook) has a session
header (Start Time in the CoxBox's local clock) and one row per stroke: GPS distance, elapsed time,
split /500, speed, stroke rate, stroke count, distance per stroke, latitude, longitude.

The camera's clock is not the CoxBox's: on 2026-09-25 to 10-04 the Osmo's file times ran ~13 min
fast against the CoxBox, and its file names are in Eastern time while the CoxBox is Pacific. So a
session is placed on a video by the video's own content: ``sync_by_motion`` matches the stroke
rhythm in the head camera's motion (camera_motion.py) to the CoxBox's strokes, first coarsely by
rate, then finely by stroke timing. The result is ``offset``: the video time at which the CoxBox's
elapsed time is zero.

The data are the crew's: keep them in the gitignored data/local.
"""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass

import numpy as np


@dataclass
class Session:
    start: dt.datetime            # CoxBox local clock
    t: np.ndarray                 # elapsed s, per stroke
    distance: np.ndarray          # m (GPS)
    split: np.ndarray             # s / 500 m
    speed: np.ndarray             # m/s
    rate: np.ndarray              # spm
    strokes: np.ndarray
    per_stroke: np.ndarray        # m
    lat: np.ndarray
    lon: np.ndarray
    path: str = ""

    @property
    def duration(self):
        return float(self.t[-1])

    def track_xy(self):
        """GPS track in metres east / north of its first point (equirectangular)."""
        lat0 = np.radians(np.nanmean(self.lat))
        x = np.radians(self.lon - self.lon[0]) * 6371000.0 * np.cos(lat0)
        y = np.radians(self.lat - self.lat[0]) * 6371000.0
        return np.column_stack([x, y])


def _sec(s):
    h, m, q = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(q)


def _num(s):
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_csv(path) -> Session:
    rows = list(csv.reader(open(path, encoding="utf-8", errors="replace")))
    start = next(r[1] for r in rows if r and r[0] == "Start Time:")
    i = next(k for k, r in enumerate(rows) if r and r[0] == "Per-Stroke Data:")
    head = rows[i + 2] if rows[i + 1] == [] or not any(rows[i + 1]) else rows[i + 1]
    col = {name: k for k, name in enumerate(head)}
    data = [r for r in rows[i:] if r and r[0].strip().isdigit()]
    get = lambda name, f=_num: np.array([f(r[col[name]]) for r in data])
    return Session(start=dt.datetime.strptime(start, "%m/%d/%Y %H:%M:%S"),
                   t=get("Elapsed Time", _sec), distance=get("Distance (GPS)"),
                   split=get("Split (GPS)", _sec), speed=get("Speed (GPS)"), rate=get("Stroke Rate"),
                   strokes=get("Total Strokes"), per_stroke=get("Distance/Stroke (GPS)"),
                   lat=get("GPS Lat."), lon=get("GPS Lon."), path=str(path))


# -- syncing a session to a video by the stroke rhythm in the picture ---------------------------
def windowed_rate(t, sig, win=12.0, step=2.0):
    """Stroke rate from the autocorrelation of ``sig`` in sliding windows: (centres, rate, confidence)."""
    fps = 1.0 / np.median(np.diff(t))
    n = int(win * fps)
    centres, rates, conf = [], [], []
    for a in range(0, len(sig) - n, max(int(step * fps), 1)):
        seg = sig[a:a + n] - np.mean(sig[a:a + n])
        ac = np.correlate(seg, seg, "full")[n - 1:]
        ac /= ac[0] + 1e-12
        lag = np.arange(n) / fps
        ok = (lag >= 60 / 46) & (lag <= 60 / 16)
        k = int(np.argmax(np.where(ok, ac, -9)))
        if 0 < k < n - 1:                        # parabolic refinement of the peak lag
            d = (ac[k - 1] - ac[k + 1]) / (2 * (ac[k - 1] - 2 * ac[k] + ac[k + 1]) + 1e-12)
            kk = k + float(np.clip(d, -0.5, 0.5))
        else:
            kk = k
        centres.append(t[a + n // 2])
        rates.append(60.0 * fps / kk if kk > 0 else np.nan)
        conf.append(ac[k])
    return np.array(centres), np.array(rates), np.array(conf)


def sync_by_motion(session: Session, mt, signal, guess, search=300.0, min_conf=0.4):
    """``(offset, quality)``: video time of the session's elapsed zero.

    Coarse: the offset within ``guess +- search`` s at which the CoxBox's stroke rate best matches
    the motion's windowed rate (median absolute difference, confident windows only). Fine: within
    +-3 s of that, the offset maximising the correlation of the band-passed motion with a pulse at
    each CoxBox stroke. ``quality`` reports both (rate error in spm, windows used, fine correlation).
    """
    c, r, q = windowed_rate(mt, signal)
    good = q >= min_conf
    best = None
    for off in np.arange(guess - search, guess + search, 1.0):
        cb = np.interp(c - off, session.t, session.rate, left=np.nan, right=np.nan)
        m = good & ~np.isnan(cb)
        if m.sum() < 30:
            continue
        err = float(np.median(np.abs(r[m] - cb[m])))
        if best is None or err < best[1]:
            best = (off, err, int(m.sum()))
    if best is None:
        return None, dict(reason="no confident overlap")
    coarse, err, nwin = best
    fps = 1.0 / np.median(np.diff(mt))
    # band-pass the motion around the stroke band (0.25-0.8 Hz) by differencing two smoothings
    def smooth(x, s):
        k = max(int(s * fps), 1)
        return np.convolve(x, np.ones(k) / k, mode="same")
    band = smooth(signal, 0.3) - smooth(signal, 2.5)
    best_f = None
    for off in np.arange(coarse - 3.0, coarse + 3.0, 0.05):
        st = session.t + off
        st = st[(st > mt[0] + 1) & (st < mt[-1] - 1)]
        if len(st) < 50:
            continue
        v = np.interp(st, mt, band)
        score = float(np.mean(v) / (np.std(band) + 1e-12))
        if best_f is None or abs(score) > abs(best_f[1]):
            best_f = (off, score)
    fine = best_f[0] if best_f else coarse
    return fine, dict(coarse=coarse, rate_err=err, windows=nwin, fine_score=best_f[1] if best_f else None)
