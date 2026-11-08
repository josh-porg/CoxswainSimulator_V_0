r"""Head of the Lake 2026 race videos: synced CoxBox HUD, the wait before the start sped up, a drop-in a minute out.

    .venv-tools/Scripts/python.exe research/overlay/render_hotl2026.py sync
    .venv-tools/Scripts/python.exe research/overlay/render_hotl2026.py contact      # frames to pick thumbnails from
    .venv-tools/Scripts/python.exe research/overlay/render_hotl2026.py render [--thumb VIDEO=SECONDS ...]

Per video (data/local/overlay/cards_hotl2026.json: title card, CoxBox file, whether the CoxBox was started on the line):

* Sync: the overlay tool's own (stroke rhythm from the head camera, the cox's called splits, the clock as a prior).
  The prior is the known camera clock offset with the CoxBox read one hour earlier than this computer's zone: the
  clocks went back on race day and the CoxBox kept daylight time (UTC-7). The first video's measured camera clock
  then sets a tight prior for the second (SAME_DAY_PRIOR).
* Start: where the CoxBox's distance is zero for a session started on the line. For one started late, the start
  line is taken from the on-time session and the late session is extended back to it along its own heading at its
  opening speed, so the HUD's time and distance run from the line.
* Edit: everything earlier than START_LEAD before the start plays at a whole-number speed that fits it into
  TIMELAPSE_MAX (silent, marked with its speed, the title card over its first seconds); from START_LEAD before the
  start the video runs in real time with the HUD to the end of the recording.

Outputs (the crew's footage: gitignored data/local): data/local/overlay/videos_titled/hotl2026/.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import app                                                       # noqa: E402
import camera_motion as CM                                       # noqa: E402
import coxbox_data as C                                          # noqa: E402
import coxbox_overlay as O                                       # noqa: E402
import title_card as T                                           # noqa: E402

LOCAL = os.path.join(ROOT, "data", "local")
CARDS = os.path.join(LOCAL, "overlay", "cards_hotl2026.json")
COXBOX = os.path.join(LOCAL, "raceday", "hotl2026", "coxbox")
VIDEOS = os.path.join(json.load(open(os.path.join(LOCAL, "coxing", "local_paths.json"), encoding="utf-8"))["media_root"],
                      "2026 race videos (raw)")   # raw footage, kept online-only in the synced media folder (private path)
OUT = os.path.join(LOCAL, "overlay", "videos_titled", "hotl2026")
SYNC = os.path.join(OUT, "sync.json")

ZONE = -7.0              # the CoxBox clock on race day: still on daylight time (the clocks went back that morning)
#: the camera clock drifts ~2.3 s a day, so a measurement from the same morning pins the next video to a few
#: seconds; the bow-loaded four's picture carries a weak stroke rhythm and, searched widely, locked 47 s off
SAME_DAY_PRIOR = 5.0
START_LEAD = 60.0        # s of real time before the start
TIMELAPSE_MAX = 60.0     # s: the longest the sped-up wait may run
OVERLAY_FPS = 10
FPS_OUT = "30000/1001"


def cards():
    return json.load(open(CARDS, encoding="utf-8"))


def card_of(d):
    d = dict(d)
    d["lineup"] = T.parse_lineup(d["lineup"], d.pop("order", "bow"))
    return T.TitleCard.from_dict(d)


# -- sync ------------------------------------------------------------------------------------------
def _motion(path):
    os.makedirs(app.CACHE, exist_ok=True)
    mpath = os.path.join(app.CACHE, app.cache_key(path) + "_motion.npz")
    if not os.path.exists(mpath):
        print("  reading the stroke rhythm from the picture...", flush=True)
        t, dy, dx, diff = CM.motion(path)
        np.savez(mpath, t=t, dy=dy, dx=dx, diff=diff)
    m = np.load(mpath)
    return m["t"], {k: m[k] for k in ("dy", "dx", "diff")}


def _calls(path):
    import transcribe_calls as TC
    if not TC.model_available():
        return []
    cpath = os.path.join(app.CACHE, app.cache_key(path) + "_calls.json")
    if not os.path.exists(cpath):
        print("  listening to the calls...", flush=True)
        json.dump(TC.transcribe(path), open(cpath, "w", encoding="utf-8"))
    return C.called_splits(json.load(open(cpath, encoding="utf-8")))


def sync_one(name, d, clock=None):
    """``clock``: ``(minutes fast, prior s)`` measured on another video the same day, in place of the history."""
    path = os.path.join(VIDEOS, name)
    s = C.load_csv(os.path.join(COXBOX, d["coxbox"]))
    v = app.Video(path)
    mt, signals = _motion(path)
    calls = _calls(path)
    s0 = s.start - dt.timedelta(hours=ZONE)
    if clock is None:
        fast, days = app.clock_for(app.DEFAULTS["clock"], v.cam_start_utc, app.DEFAULTS["camera_fast"])
        prior = 20.0 + app.CLOCK_DRIFT * days
    else:
        fast, prior = clock
    guess = (s0 - v.true_start_utc(fast)).total_seconds()
    if clock is None:
        off, info = C.best_sync(s, mt, signals, guess, calls, prior_sd=prior)
    else:                                 # pinned by the same-day clock: the best stroke-timing match within the prior
        fits = {k: C.sync_by_motion(s, mt, sig, guess, search=prior, min_conf=0.0) for k, sig in signals.items()}
        k = min((k for k in fits if fits[k][0] is not None), key=lambda k: fits[k][1]["fine_score"])
        off, info = fits[k][0], dict(fits[k][1], calls=C.calls_score(s, calls, fits[k][0]) if calls else None)
    if off is None:
        raise SystemExit("%s: could not sync (%s)" % (name, info.get("reason")))
    measured = (v.cam_start_utc - (s0 - dt.timedelta(seconds=off))).total_seconds() / 60.0
    print("%s: CoxBox zero at video %s (clock guess %s; stroke rate agrees to %.1f spm; %d calls%s); camera %.2f min fast"
          % (name, app.mmss(off), app.mmss(guess), info["rate_err"], len(calls),
             "" if not info.get("calls") else ", called splits to %.1f s" % info["calls"][0], measured), flush=True)
    return dict(offset=float(off), guess=float(guess), rate_err=float(info["rate_err"]), camera_fast=measured,
                duration=float(v.duration))


def _line_point(s):
    """Where the session's distance is zero: its first fix taken back along the opening heading (local metres)."""
    xy = s.track_xy()
    u = xy[5] - xy[0]
    u = u / np.linalg.norm(u)
    return xy[0] - u * float(s.distance[0]), u


def late_start(s_late, s_ref):
    """``(seconds, metres)`` the late session began past the on-time session's start line."""
    lat0 = np.radians(np.nanmean(s_ref.lat))
    shift = np.array([np.radians(s_late.lon[0] - s_ref.lon[0]) * 6371000.0 * math.cos(lat0),
                      np.radians(s_late.lat[0] - s_ref.lat[0]) * 6371000.0])
    p_ref, _ = _line_point(s_ref)
    p_late, u = _line_point(s_late)
    d = float(np.dot(p_late + shift - p_ref, u))
    v0 = float(np.nanmean(s_late.speed[:5]))
    return d / v0, d


def sync():
    cs = cards()
    out = {}
    for name, d in cs.items():            # the first (the eight: the clearest stroke rhythm) sets the day's clock
        clock = None if not out else (next(iter(out.values()))["camera_fast"], SAME_DAY_PRIOR)
        out[name] = sync_one(name, d, clock)
    ref = next(C.load_csv(os.path.join(COXBOX, d["coxbox"])) for d in cs.values() if d["on_time"])
    for name, d in cs.items():
        late, dist = (0.0, 0.0)
        if not d["on_time"]:
            late, dist = late_start(C.load_csv(os.path.join(COXBOX, d["coxbox"])), ref)
            print("%s: the CoxBox started %.1f s (%.0f m) past the start line" % (name, late, dist))
        out[name].update(late=late, late_m=dist, start=out[name]["offset"] - late)
        print("%s: start at video %s of %s" % (name, app.mmss(out[name]["start"]), app.mmss(out[name]["duration"])))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(SYNC, "w"), indent=1)


def session_from_line(d, sy):
    """The session with its clock and distance running from the start line."""
    s = C.load_csv(os.path.join(COXBOX, d["coxbox"]))
    t, dist = s.t + sy["late"], s.distance + sy["late_m"]
    # a point on the line itself (time 0, distance 0), placed back along the opening heading, so the HUD's
    # distance runs up from zero instead of holding the first logged stroke's
    f = dist[0] / (dist[1] - dist[0])
    pre = lambda a, b=None: np.concatenate([[a[0] if b is None else b], a])
    return dataclasses.replace(
        s, t=pre(t, 0.0), distance=pre(dist, 0.0), split=pre(s.split), speed=pre(s.speed), rate=pre(s.rate),
        strokes=pre(s.strokes, 0.0), per_stroke=pre(s.per_stroke),
        lat=pre(s.lat, s.lat[0] - f * (s.lat[1] - s.lat[0])), lon=pre(s.lon, s.lon[0] - f * (s.lon[1] - s.lon[0])))


# -- thumbnails --------------------------------------------------------------------------------------
def contact(step=20.0, cols=4, rows=4, w=480):
    """Sheets of frames through each race (every ``step`` s), labelled with video time, to choose thumbnails."""
    sy = json.load(open(SYNC))
    for name in cards():
        path, st = os.path.join(VIDEOS, name), sy[name]["start"]
        times = np.arange(st, min(st + 1500.0, sy[name]["duration"] - 1), step)
        h = round(w * 9 / 16)
        d = os.path.join(OUT, "contact", os.path.splitext(name)[0])
        os.makedirs(d, exist_ok=True)
        for k in range(0, len(times), cols * rows):
            sheet = Image.new("RGB", (cols * w, rows * h), "black")
            for j, tv in enumerate(times[k:k + cols * rows]):
                im = O.grab_frame(path, float(tv)).resize((w, h))
                ImageDraw.Draw(im).text((8, 6), "%.0f s (race %s)" % (tv, app.mmss(tv - st)), fill="yellow",
                                        font=O.font(22))
                sheet.paste(im, ((j % cols) * w, (j // cols) * h))
            sheet.save(os.path.join(d, "sheet_%02d.jpg" % (k // (cols * rows))), quality=80)
        print(name, len(times), "frames ->", d)


def save_thumb(name, d, tv):
    """The frame at video time ``tv``, 1280 x 720, at full brightness: the title and subtitle bottom left over a
    soft gradient (the title card's wash and lineup would hide the boats)."""
    frame = T._fill(O.grab_frame(os.path.join(VIDEOS, name), tv), T.THUMB_SIZE).convert("RGBA")
    W, H = frame.size
    shade = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    a = np.clip((np.arange(H) - 0.55 * H) / (0.45 * H), 0, 1) ** 1.5 * 200
    arr = np.zeros((H, W, 4), np.uint8)
    arr[..., :3] = O.NAVY_DEEP
    arr[..., 3] = a[:, None].astype(np.uint8)
    shade = Image.fromarray(arr, "RGBA")
    img = Image.alpha_composite(frame, shade)
    dr = ImageDraw.Draw(img)
    x = 56
    dr.rectangle((x - 22, H - 150, x - 14, H - 46), fill=O.RED)
    O.text(dr, (x, H - 88), d["title"], 72, weight="Bold")
    O.text(dr, (x, H - 50), d["subtitle"], 30, fill=O.LABEL)
    path = os.path.join(OUT, "%s - thumbnail.jpg" % os.path.splitext(name)[0])
    img.convert("RGB").save(path, "JPEG", quality=90, optimize=True)
    return path


# -- render ------------------------------------------------------------------------------------------
def speed_tag(size, factor):
    """``5x speed`` in the HUD's panel style, top right, at the 1920-wide reference."""
    ref = (O.REF_WIDTH, int(round(O.REF_WIDTH * size[1] / size[0])))
    img = Image.new("RGBA", ref, (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    O.panel(dr, (ref[0] - 300, 30, ref[0] - 40, 104))
    O.text(dr, (ref[0] - 170, 82), "%d× speed" % factor, 40, anchor="ms")
    return img if ref == tuple(size) else img.resize(tuple(size), Image.LANCZOS)


def render_timelapse(path, out, upto, factor, card):
    """Video 0-``upto`` s at ``factor`` times speed, silent, the speed marked and the card over its first seconds."""
    W, H = O.probe(path)[:2]
    D = upto / factor
    cmd = [O.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-t", "%.3f" % upto, "-i", path,
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "%dx%d" % (W, H), "-r", str(OVERLAY_FPS), "-i", "-",
           "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
           "-filter_complex", "[0:v]setpts=(PTS-STARTPTS)/%d,fps=%s,setsar=1,format=yuv420p[m];"
                              "[m][1:v]overlay=0:0:format=auto[o]" % (factor, FPS_OUT),
           "-map", "[o]", "-map", "2:a", "-t", "%.3f" % D, "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    tag = speed_tag((W, H), factor)
    card_img = T.card_layer((W, H), card)
    for i in range(int(math.ceil(D * OVERLAY_FPS)) + 1):
        a = T.fade(i / OVERLAY_FPS, card.seconds)
        proc.stdin.write((Image.alpha_composite(tag, T.with_opacity(card_img, a)) if a > 0 else tag).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed on the timelapse")


def render(thumbs):
    sy = json.load(open(SYNC))
    os.makedirs(OUT, exist_ok=True)
    for name, d in cards().items():
        path, st = os.path.join(VIDEOS, name), sy[name]
        stem = os.path.splitext(name)[0]
        card = card_of(d)
        s = session_from_line(d, st)
        at = O.session_moments(s, st["start"])
        upto = st["start"] - START_LEAD
        final = os.path.join(OUT, "%s - Head of the Lake.mp4" % stem)
        if upto <= TIMELAPSE_MAX:              # the wait already fits: real time throughout, the card at the top
            print("%s: no speed-up needed (%.0f s before the drop-in)" % (name, max(upto, 0.0)), flush=True)
            O.render_video(path, final, "A", at, card=card)
        else:
            factor = int(math.ceil(upto / TIMELAPSE_MAX))
            print("%s: %.0f s of waiting at %dx (%.0f s), then real time from %s" % (
                name, upto, factor, upto / factor, app.mmss(upto)), flush=True)
            parts = [os.path.join(OUT, "%s - part1.mp4" % stem), os.path.join(OUT, "%s - part2.mp4" % stem)]
            render_timelapse(path, parts[0], upto, factor, card)
            O.render_video(path, parts[1], "A", at, start=upto)
            lst = os.path.join(OUT, "%s - parts.txt" % stem)
            open(lst, "w").write("".join("file '%s'\n" % os.path.basename(p) for p in parts))
            subprocess.run([O.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                            "-i", lst, "-c", "copy", "-movflags", "+faststart", final], check=True)
            for p in parts + [lst]:
                os.remove(p)
        print("%s: saved %s" % (name, final), flush=True)
        tv = thumbs.get(name, st["start"] + 120.0)
        print("%s: thumbnail at video %s -> %s" % (name, app.mmss(tv), save_thumb(name, d, tv)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sync", "contact", "render", "thumbs"])
    ap.add_argument("--thumb", nargs="*", default=[], help="VIDEO=SECONDS: the thumbnail's frame (video time)")
    a = ap.parse_args()
    thumbs = {k: float(v) for k, v in (x.split("=") for x in a.thumb)}
    if a.step == "sync":
        sync()
    elif a.step == "contact":
        contact()
    elif a.step == "thumbs":
        for name, d in cards().items():
            if name in thumbs:
                print(save_thumb(name, d, thumbs[name]))
    else:
        render(thumbs)


if __name__ == "__main__":
    main()
