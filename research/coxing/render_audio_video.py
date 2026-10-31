"""Render an audio-only race (a coxswain's recording synced to the CoxBox) as a video: the overlay's layout D, with a
map that follows the boat where the camera picture would be, the data column larger, and the calls as captions.

    .venv-tools/Scripts/python.exe research/coxing/render_audio_video.py <key> <coxbox csv> <audio file> [--fps 10]
        [--start 0] [--end ...] [--out ...]

Same look as tools/overlay (club colours, Bahnschrift/Barlow, the 30 s rate/split trace), built from its own drawing
pieces. Layout D sets the picture at 80 % beside a data column; with no picture to keep clear, the data column is wider
and its figures larger, and the map takes the picture's place: north up, centred on the boat, about 0.8 km across,
with the water from the surveyed reach, the HOCR corridor limits, the bridge decks and their names, the track rowed so
far, and the boat to scale. The current call runs as a caption across the foot of the map; there is no picture for it
to cover. Audio is the recording itself, muxed in. Output stays in data/local (a private recording).

Needs the sync (sync_audio_coxbox.py) and transcript (transcribe_audio.py) for <key>.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coxbox_data as C  # noqa: E402
import coxbox_overlay as O  # noqa: E402

W, H = 1920, 1080
COL = 600                                   # data column width (layout D's is 384 at 1920; the data is bigger here)
MAP_X0 = COL + 6
SCALE = 1.6                                 # px per metre on the map (~0.8 km across)
WATER = (34, 62, 92)
EDGE = (96, 140, 176)
LIMIT = (214, 36, 48)
BRIDGE = (232, 236, 244)
TRACK = (120, 196, 255)
BOAT = (255, 204, 77)


def load_course(key):
    """Course geometry from the viewer build (data/local/viewer/<key>/data.js, written by build_viewer.py), lat/lon to
    local metres about the start. Read rather than recomputed: the course model needs scipy, the render environment
    has ffmpeg, and one geometry serves both outputs."""
    txt = open(os.path.join(ROOT, "data", "local", "viewer", key, "data.js"), encoding="utf-8").read()
    g = json.loads(txt[txt.index("=") + 1:].strip().rstrip(";"))["course"]
    lat0, lon0 = g["start"]
    k = math.cos(math.radians(lat0))
    xy = lambda p: (6371000 * math.radians(p[1] - lon0) * k, 6371000 * math.radians(p[0] - lat0))
    out = {n: np.array([xy(p) for p in g[n]]) for n in ("water_port", "water_starboard", "limit_port", "limit_starboard")}
    out["bridges"] = [(b["name"], np.array([xy(b["deck"][0]), xy(b["deck"][1])])) for b in g["bridges"]]
    out["start"], out["finish"] = np.array(xy(g["start"])), np.array(xy(g["finish"]))
    return out, (lat0, lon0, k)


def draw_map(img, course, track, pos, heading, cutoff_i, frac):
    d = ImageDraw.Draw(img, "RGBA")
    cx, cy = MAP_X0 + (W - MAP_X0) / 2, H * 0.47
    to = lambda p: (cx + (p[0] - pos[0]) * SCALE, cy - (p[1] - pos[1]) * SCALE)
    d.rectangle((MAP_X0, 0, W, H), fill=O.NAVY_DEEP + (255,))
    poly = [to(p) for p in course["water_port"]] + [to(p) for p in course["water_starboard"][::-1]]
    d.polygon(poly, fill=WATER + (255,))
    d.line([to(p) for p in course["water_port"]], fill=EDGE + (255,), width=3)
    d.line([to(p) for p in course["water_starboard"]], fill=EDGE + (255,), width=3)
    for n in ("limit_port", "limit_starboard"):
        pts = [to(p) for p in course[n]]
        for a in range(0, len(pts) - 1, 2):                    # dashed
            d.line(pts[a:a + 2], fill=LIMIT + (200,), width=3)
    for name, deck in course["bridges"]:
        a, b = to(deck[0]), to(deck[1])
        d.line([a, b], fill=BRIDGE + (255,), width=9)
        m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if MAP_X0 < m[0] < W and 0 < m[1] < H:
            O.text(d, (m[0] + 18, m[1] - 14), name, 30, O.WHITE, "SemiBold")
    for p, col, lab in ((course["start"], (111, 211, 154), "start"), (course["finish"], (240, 163, 90), "finish")):
        q = to(p)
        d.ellipse((q[0] - 10, q[1] - 10, q[0] + 10, q[1] + 10), fill=col + (255,))
    done = [to(p) for p in track[:cutoff_i + 1]] + [to(pos)]
    if len(track) > 1:
        d.line([to(p) for p in track], fill=O.LABEL + (90,), width=2)
    if len(done) > 1:
        d.line(done, fill=TRACK + (255,), width=6, joint="curve")
    # the boat: a four is ~13 m long, drawn at least 60 px
    L = max(13 * SCALE, 60.0)
    hx, hy = math.cos(heading), -math.sin(heading)          # screen y down
    nx, ny = -hy, hx
    c = to(pos)
    bow = (c[0] + hx * L / 2, c[1] + hy * L / 2)
    stern = (c[0] - hx * L / 2, c[1] - hy * L / 2)
    w = L * 0.13
    d.polygon([bow, (c[0] + nx * w, c[1] + ny * w), stern, (c[0] - nx * w, c[1] - ny * w)], fill=BOAT + (255,),
              outline=O.NAVY_DEEP + (255,))
    # scale bar
    O.text(d, (W - 40, H - 30), "200 m", 24, O.LABEL, "Regular", anchor="rs")
    d.line([(W - 40 - 200 * SCALE, H - 22), (W - 40, H - 22)], fill=O.LABEL + (255,), width=4)


def caption(img, words):
    if not words:
        return
    d = ImageDraw.Draw(img, "RGBA")
    lines = textwrap.wrap(words, 52)[:2]
    h = 30 + 64 * len(lines)
    d.rectangle((MAP_X0, H - h - 60, W, H - 60), fill=O.NAVY_DEEP + (200,))
    y = H - h - 60 + 66
    for ln in lines:
        O.text(d, (MAP_X0 + (W - MAP_X0) / 2, y), ln, 50, O.WHITE, "SemiBold", anchor="ms")
        y += 64


def column(img, m, title, sub, overview, course_xy):
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle((0, 0, COL, H), fill=O.NAVY_DEEP + (255,))
    d.rectangle((COL, 0, COL + 6, H), fill=O.RED + (255,))
    O.text(d, (36, 52), title, 34, O.WHITE, "SemiBold")
    O.text(d, (36, 88), sub, 24, O.LABEL, "Light")
    if m is None:
        O.text(d, (36, 220), "before the start", 40, O.GREY, "Regular")
        return
    rate, rate_col = O.rate_figure(m)
    y = 120
    for label, value, size in (("stroke rate", rate, 132), ("split /500 m", O.fmt_split(m.split), 96)):
        O.text(d, (36, y + 26), label, 28, O.LABEL, "Light")
        y += 26 + int(size * 0.92)
        O.text(d, (36, y), value, size, rate_col if label == "stroke rate" else O.WHITE)
        y += 22
    # distance and time side by side, then metres per stroke
    O.text(d, (36, y + 26), "distance", 28, O.LABEL, "Light")
    O.text(d, (330, y + 26), "time", 28, O.LABEL, "Light")
    y += 26 + 52
    O.text(d, (36, y), O.fmt_dist(m.distance) + " m", 58, O.WHITE)
    O.text(d, (330, y), O.fmt_time(m.elapsed), 58, O.WHITE)
    y += 22
    O.text(d, (36, y + 26), "per stroke", 28, O.LABEL, "Light")
    y += 26 + 46
    O.text(d, (36, y), "–" if m.per_stroke is None else "%.1f m" % m.per_stroke, 50, O.WHITE)
    y += 18
    O.trace(d, (16, y, COL - 16, y + 170), m)
    # whole-course overview, north up
    box = (16, y + 186, COL - 16, H - 16)
    O.panel(d, box)
    tr = overview
    lo, hi = tr.min(axis=0), tr.max(axis=0)
    pad = 18
    sc = min((box[2] - box[0] - 2 * pad) / (hi[0] - lo[0]), (box[3] - box[1] - 2 * pad) / (hi[1] - lo[1]))
    cxo, cyo, mid = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, (lo + hi) / 2
    tp = lambda p: (cxo + (p[0] - mid[0]) * sc, cyo - (p[1] - mid[1]) * sc)
    d.line([tp(p) for p in tr], fill=O.WHITE + (200,), width=4, joint="curve")
    if course_xy is not None:
        q = tp(course_xy)
        d.ellipse((q[0] - 9, q[1] - 9, q[0] + 9, q[1] + 9), fill=O.RED + (255,), outline=O.WHITE + (255,), width=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("coxbox")
    ap.add_argument("audio")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--start", type=float, default=0.0, help="recording seconds")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--title", default="Head of the Charles 2026")
    ap.add_argument("--sub", default="coxswain's recording, synced to the CoxBox")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = C.load_csv(a.coxbox)
    off = float(json.load(open(os.path.join(ROOT, "data", "local", "coxing", "sync", a.key + ".json")))["offset"])
    segs = json.load(open(os.path.join(ROOT, "data", "local", "overlay", "calls", a.key + "_nocond.json"), encoding="utf-8"))
    calls = [(g["start"], g["end"], g["text"].strip()) for g in segs]
    course, (lat0, lon0, k) = load_course(a.key)
    xy = np.column_stack([6371000 * np.radians(np.asarray(s.lon) - lon0) * k, 6371000 * np.radians(np.asarray(s.lat) - lat0)])
    hd = np.array([math.atan2(*(xy[min(i + 3, len(xy) - 1)] - xy[max(i - 3, 0)])[::-1]) for i in range(len(xy))])
    moments = O.session_moments(s, off)
    dur = float(subprocess.run([O.ffmpeg_exe(), "-hide_banner", "-i", a.audio], capture_output=True, text=True).stderr
                .split("Duration: ")[1].split(",")[0].split(":")[-1]) + 60 * int(
        subprocess.run([O.ffmpeg_exe(), "-hide_banner", "-i", a.audio], capture_output=True, text=True).stderr
        .split("Duration: ")[1].split(",")[0].split(":")[1])
    t_end = a.end if a.end is not None else dur
    out = a.out or os.path.join(ROOT, "data", "local", "viewer", a.key, "race_map.mp4")
    n = int((t_end - a.start) * a.fps)
    cmd = [O.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(a.fps), "-i", "-",
           "-ss", "%.3f" % a.start, "-t", "%.3f" % (t_end - a.start), "-i", a.audio,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-shortest", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        tv = a.start + i / a.fps
        e = min(max(tv - off, s.t[0]), s.t[-1])
        kk = int(np.clip(np.searchsorted(s.t, e) - 1, 0, len(s.t) - 2))
        f = (e - s.t[kk]) / (s.t[kk + 1] - s.t[kk])
        pos = xy[kk] + f * (xy[kk + 1] - xy[kk])
        dh = (hd[kk + 1] - hd[kk] + math.pi) % (2 * math.pi) - math.pi
        img = Image.new("RGB", (W, H), O.NAVY_DEEP)
        draw_map(img, course, xy, pos, hd[kk] + f * dh, kk, f)
        cur = [c for c in calls if c[0] <= tv <= c[1] + 1.5]
        caption(img, cur[-1][2] if cur else "")
        m = moments(tv)
        column(img, m, a.title, a.sub, xy, pos)
        p.stdin.write(img.tobytes())
        if i % (a.fps * 60) == 0:
            print("  %d / %d min" % (i // (a.fps * 60), n // (a.fps * 60)), flush=True)
    p.stdin.close()
    p.wait()
    print("written", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
