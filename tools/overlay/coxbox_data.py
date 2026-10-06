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
        err = float(np.mean(np.abs(r[m] - cb[m])))      # mean, not median: the rate changes pin the offset
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


# -- the coxswain's called splits, and the sync decision ------------------------------------------
_ONES = {"oh": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}


def _split_seconds(tokens):
    """Seconds of a spoken split's second half, and how many tokens it used: "oh two" (2),
    "sixteen" (16), "twenty one" (21), "fifty" (50), "flat" (0). Plain "three" is not a split's
    seconds (that is counting: "two, three")."""
    if not tokens:
        return None, 0
    a = tokens[0]
    if a == "flat":
        return 0, 1
    if a in ("oh", "zero") and len(tokens) > 1 and tokens[1] in _ONES and tokens[1] not in ("oh", "zero"):
        return _ONES[tokens[1]], 2
    if a in _TEENS:
        return _TEENS[a], 1
    if a in _TENS:
        if len(tokens) > 1 and tokens[1] in _ONES and tokens[1] not in ("oh", "zero"):
            return _TENS[a] + _ONES[tokens[1]], 2
        return _TENS[a], 1
    return None, 0


def called_splits(segments):
    """``[(video s, split s)]``: the splits the coxswain called, from a transcript with word times.

    Digits ("221", "2:21", "2.21") and words ("two twenty one", "two oh two", "one fifty eight",
    "two flat") both count; the recogniser writes either. Minutes must be one or two, seconds under
    sixty. Called splits are the coxswain's most time-accurate calls; called distances are rounded
    (+-500 m) and are not used."""
    import re
    words = [(float(w["start"]), w["word"].strip().lower()) for seg in segments for w in seg.get("words", [])]
    out = []
    i = 0
    while i < len(words):
        t, raw = words[i]
        tok = re.sub(r"[^a-z0-9:.]", "", raw).strip(".")
        m = re.fullmatch(r"([12])[:.]?([0-5]\d)", tok)
        if m:
            out.append((t, int(m.group(1)) * 60 + int(m.group(2))))
            i += 1
            continue
        clean = [re.sub(r"[^a-z]", "", w) for _, w in words[i + 1:i + 3]]
        # a comma between the words is a count ("one, two"), not a split
        joined = not raw.endswith(",")
        if tok in ("one", "two") and joined:
            sec, used = _split_seconds(clean)
            if sec is not None and sec < 60:
                out.append((t, (1 if tok == "one" else 2) * 60 + sec))
                i += 1 + used
                continue
        i += 1
    return out


def calls_score(session: Session, calls, offset):
    """``(median |called - CoxBox| s, lag s)`` at this offset, with one shared reading lag of 0-8 s
    (the coxswain reads the display, then speaks)."""
    best = None
    for lag in np.arange(0.0, 8.01, 0.5):
        d = [abs(v - np.interp(t - offset - lag, session.t, session.split)) for t, v in calls
             if 0 < t - offset - lag < session.duration]
        if len(d) < 5:
            continue
        sc = float(np.median(d))
        if best is None or sc < best[0]:
            best = (sc, float(lag))
    return best


def best_sync(session: Session, mt, signals: dict, guess, calls=None, search=180.0, prior_sd=None):
    """The offset (video s of the session's elapsed zero) and how it was chosen.

    Candidates come from each motion signal at three confidence thresholds, and (with called splits)
    from a direct search on the calls and motion rate together. ``prior_sd``: how far (s) the clock
    guess can be trusted, once the camera clock has been calibrated by an earlier synced video (the
    Osmo held to 4 s between two sessions on 2026-10-04); the search then stays within 3 sd and the
    choice is penalised by its distance from the guess. Without it the search is wide."""
    if prior_sd:
        search = min(search, 3.0 * prior_sd)
    # one scale for all three terms: each in units of its own typical size, the prior as a Gaussian
    # log-prior (a few seconds from the clock costs nothing, tens of seconds do)
    RATE_SCALE, CALLS_SCALE = 0.25, 0.5          # spm; s
    penalty = (lambda off: 0.5 * ((off - guess) / prior_sd) ** 2) if prior_sd else (lambda off: 0.0)
    cands = []
    # how much the picture says: in an eight the head-motion rhythm is strong (median window
    # confidence ~0.6), in a bow-loaded four weak (~0.25). Weak motion cannot move a calibrated clock
    # by tens of seconds; it only aligns the stroke phase near it.
    _c, _r, _q = windowed_rate(mt, next(iter(signals.values())))
    span = (_c >= guess) & (_c <= guess + session.duration)
    quality = float(np.median(_q[span])) if span.any() else 0.0
    motion_search = 4.0 if (prior_sd and quality < 0.35) else search
    for name, sig in signals.items():
        for mc in ((0.0,) if motion_search < search else (0.4, 0.3, 0.25)):
            off, q = sync_by_motion(session, mt, sig, guess, search=motion_search, min_conf=mc)
            if off is not None:
                cands.append(dict(offset=float(off), signal=name, min_conf=mc, rate_err=float(q["rate_err"]),
                                  windows=int(q["windows"]), motion_quality=quality))
    if calls and len(calls) >= 5:
        # the calls and the motion rate together, searched directly: a candidate near the truth even
        # when every motion-only candidate locked onto the wrong stretch of a steady piece
        c, r, q = windowed_rate(mt, next(iter(signals.values())))
        best = None
        for off in np.arange(guess - search, guess + search, 1.0):
            cs = calls_score(session, calls, off)
            if cs is None:
                continue
            cb = np.interp(c - off, session.t, session.rate, left=np.nan, right=np.nan)
            m = (q >= 0.25) & ~np.isnan(cb)
            rerr = float(np.mean(np.abs(r[m] - cb[m]))) if m.sum() >= 10 else 3.0
            score = cs[0] / CALLS_SCALE + rerr / RATE_SCALE + penalty(off)
            if best is None or score < best[0]:
                best = (score, off, rerr)
        if best is not None:
            joint = best[1]
            sig = next(iter(signals.values()))
            fine, q2 = sync_by_motion(session, mt, sig, joint, search=4.0, min_conf=0.0)
            off = fine if fine is not None else joint
            cands.append(dict(offset=float(off), signal="calls+%s" % next(iter(signals)), min_conf=0.25,
                              rate_err=float(best[2]), windows=0))
    if not cands:
        return None, dict(reason="no stroke rhythm matched the CoxBox within %d s of the clock" % search)
    if calls:
        for c in cands:
            c["calls"] = calls_score(session, calls, c["offset"])
        scored = [c for c in cands if c["calls"] is not None]
        if scored:
            best = min(scored, key=lambda c: c["calls"][0] / CALLS_SCALE + c["rate_err"] / RATE_SCALE
                       + penalty(c["offset"]))
            return best["offset"], dict(best, n_calls=len(calls), candidates=cands)
    best = min(cands, key=lambda c: c["rate_err"] / RATE_SCALE + penalty(c["offset"]))
    return best["offset"], dict(best, n_calls=len(calls or []), candidates=cands)
