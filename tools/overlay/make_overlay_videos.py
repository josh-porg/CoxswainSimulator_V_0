r"""Sync CoxBox sessions to the cox-camera videos and render the overlaid videos.

    python tools/overlay/make_overlay_videos.py --sync            # offsets for every paired video
    python tools/overlay/make_overlay_videos.py --render 0132 0134 [--layout A]
    python tools/overlay/make_overlay_videos.py --render 0124 --estimate   # no CoxBox: camera rate

Pairs (agreed with the coxswain, 2026-10-05): the five sessions in data/local/coxbox and the
videos they belong to; three race pieces of 2026-10-02 have no CoxBox and get the camera-estimated
rate. The camera clock ran ~14.4 min fast against the CoxBox (Sunday's two races agree), so that is
the starting guess; the offset itself comes from the stroke rhythm (coxbox_data.sync_by_motion) and
is checked against the coxswain's own called distances where there are any.

Everything read and written here is the crew's own: data/local only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coxbox_data as C                                          # noqa: E402
import coxbox_overlay as O                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
VIDEOS = r"C:\Users\satur\Downloads\race videos"
LOCAL = os.path.join(ROOT, "data", "local")
SYNC = os.path.join(LOCAL, "overlay", "sync.json")
OUT = os.path.join(LOCAL, "overlay", "videos")
CAMERA_FAST_MIN = 14.4          # the Osmo's clock against the CoxBox's, 2026-09/10

#: video number -> CoxBox session (file stem) or None (no CoxBox: camera-estimated rate)
PAIRS = {
    "0117": "20260925 0555AM",   # practice race 1, 25 Sep
    "0119": "20260925 0620AM",   # practice race 2, 25 Sep
    "0123": "20261002 0525AM",   # warm-up piece, 2 Oct
    "0124": None, "0125": None, "0131": None,   # race pieces, 2 Oct
    "0132": "20261004 0803AM",   # race, eight, 4 Oct
    "0134": "20261004 1237PM",   # race, four, 4 Oct
}


def video_path(num):
    return glob.glob(os.path.join(VIDEOS, "DJI_*_%s_D.MP4" % num))[0]


def video_start_local(path):
    """The camera's own start time as UTC -> Pacific (UTC-7 in Sep/Oct), before the clock fix."""
    err = subprocess.run([O.ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    ct = re.search(r"creation_time\s*:\s*(\S+)", err).group(1)
    return dt.datetime.strptime(ct[:19], "%Y-%m-%dT%H:%M:%S") - dt.timedelta(hours=7)


def motion(num):
    path = os.path.join(LOCAL, "overlay", "motion", os.path.splitext(os.path.basename(video_path(num)))[0] + ".npz")
    if not os.path.exists(path):
        raise SystemExit("no motion signal for %s: run camera_motion.py on it first" % num)
    return np.load(path)


def session(num):
    return C.load_csv(glob.glob(os.path.join(LOCAL, "coxbox", "*%s.csv" % PAIRS[num]))[0])


def called_splits(num):
    """``[(video s, split s)]``: the splits the coxswain called, from the transcript (three-digit
    numbers like "201" or "158" said as a split), or [] without a transcript."""
    path = os.path.join(LOCAL, "overlay", "calls",
                        os.path.splitext(os.path.basename(video_path(num)))[0] + ".json")
    if not os.path.exists(path):
        return []
    out = []
    for seg in json.load(open(path, encoding="utf-8")):
        for w in seg["words"]:
            d = re.sub(r"[^\d]", "", w["word"])
            if len(d) == 3 and d[0] in "12" and int(d[1:]) < 60:
                out.append((float(w["start"]), int(d[0]) * 60 + int(d[1:])))
    return out


def calls_score(s, calls, offset):
    """``(median |called - CoxBox|, lag)``: called splits against the CoxBox at this offset, with
    one shared reading lag of 0-8 s (the coxswain reads the display, then speaks)."""
    best = None
    for lag in np.arange(0.0, 8.01, 0.5):
        d = [abs(v - np.interp(t - offset - lag, s.t, s.split)) for t, v in calls if 0 < t - offset - lag < s.duration]
        if len(d) < 5:
            continue
        sc = float(np.median(d))
        if best is None or sc < best[0]:
            best = (sc, float(lag))
    return best or (99.0, 0.0)


def load_sync():
    return json.load(open(SYNC)) if os.path.exists(SYNC) else {}


def do_sync(nums):
    sync = load_sync()
    for num in nums:
        if PAIRS.get(num) is None:
            continue
        s = session(num)
        cam = video_start_local(video_path(num)) - dt.timedelta(minutes=CAMERA_FAST_MIN)
        guess = (s.start - cam).total_seconds()
        m = motion(num)
        cands = []
        for sig in ("dy", "dx", "diff"):
            for mc in (0.4, 0.3, 0.25):
                off, q = C.sync_by_motion(s, m["t"], m[sig], guess, search=180.0, min_conf=mc)
                if off is not None:
                    cands.append(dict(offset=float(off), signal=sig, min_conf=mc, rate_err=float(q["rate_err"]),
                                      windows=int(q["windows"])))
        if not cands:
            print("%s: no sync found" % num)
            continue
        calls = called_splits(num)
        for c in cands:
            c["calls"] = calls_score(s, calls, c["offset"]) if calls else None
        if calls:
            # the coxswain's own called splits arbitrate between the motion candidates
            best = min(cands, key=lambda c: (c["calls"][0], c["rate_err"]))
        else:
            best = min(cands, key=lambda c: c["rate_err"])
        sync[num] = dict(session=PAIRS[num], clock_guess=guess, n_calls=len(calls), **best,
                         candidates=cands)
        off = best["offset"]
        print("%s: CoxBox zero at video %d:%05.2f (%s >= %.2f, rate error %.2f spm over %d windows; %s; clock guess %d:%02d)"
              % (num, off // 60, off % 60, best["signal"], best["min_conf"], best["rate_err"], best["windows"],
                 "called splits %d, median %.1f s at lag %.1f s" % ((len(calls),) + tuple(best["calls"])) if calls
                 else "no called splits", guess // 60, guess % 60))
    os.makedirs(os.path.dirname(SYNC), exist_ok=True)
    json.dump(sync, open(SYNC, "w"), indent=1)


def do_render(nums, layout, estimate=False):
    os.makedirs(OUT, exist_ok=True)
    sync = load_sync()
    for num in nums:
        path = video_path(num)
        out = os.path.join(OUT, "%s_%s.mp4" % (os.path.splitext(os.path.basename(path))[0], layout))
        if PAIRS.get(num) is None or estimate:
            m = motion(num)
            at = O.estimate_moments(m["t"], m["dy"], 0.0, float(m["t"][-1]), min_conf=0.5)
        else:
            if num not in sync:
                raise SystemExit("%s is not synced yet: run --sync" % num)
            at = O.session_moments(session(num), sync[num]["offset"])
        print("rendering", out, flush=True)
        O.render_video(path, out, layout, at)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", nargs="*", help="video numbers to sync (default: all paired)")
    ap.add_argument("--render", nargs="*", help="video numbers to render")
    ap.add_argument("--layout", default="A")
    ap.add_argument("--estimate", action="store_true", help="camera-estimated rate even if CoxBox data exist")
    a = ap.parse_args()
    if a.sync is not None:
        do_sync(a.sync or [k for k, v in PAIRS.items() if v])
    if a.render:
        do_render(a.render, a.layout.upper(), a.estimate)


if __name__ == "__main__":
    main()
