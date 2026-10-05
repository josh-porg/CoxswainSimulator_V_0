r"""CoxBox data drawn over cox-camera footage: layouts A (top strip), B (top corners), D (side panel).

    python tools/overlay/coxbox_overlay.py --video "C:\...\DJI_..._D.MP4" --at 9:00 --layout A --sample \
        --race-start 3:00 --out data/local/overlay/frames

Design (agreed 2026-10-05): keep the blades and the crew clear. From a head-mounted camera at the
cox seat of an eight the blades sit across the middle and bottom of the frame on both sides and the
crew fills the centre, so layouts A and B use the sky band only; D shrinks the picture and puts the
data beside it, covering nothing. Club colours (SRA): navy panels, red accents, white figures.

Shown: stroke rate, split per 500 m, distance from the start, elapsed time, distance per stroke, a
30 s trace of rate (red) and split (white), and a course map with the boat's position. A clip with
no CoxBox data can show a stroke rate estimated from the camera's own accelerometer, drawn in amber
and marked as an estimate, with the CoxBox-only fields blanked rather than guessed.

``--sample`` draws synthetic numbers on the Head of the Lake course, stamped "sample data", to judge
the look before real data is loaded. Real data (NK CSV / FIT / GPX, synced to the video) comes next.

The footage and the CoxBox data are the crew's: outputs go to the gitignored ``data/local``.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- club colours --------------------------------------------------------------------------------
NAVY = (20, 34, 84)
NAVY_DEEP = (10, 16, 40)
RED = (214, 36, 48)
RED_BRIGHT = (240, 72, 78)
WHITE = (255, 255, 255)
LABEL = (196, 206, 232)
AMBER = (242, 184, 75)
GREY = (150, 156, 170)
PANEL_ALPHA = 190

FONT_FILE = r"C:\Windows\Fonts\bahnschrift.ttf"


def font(size, weight="SemiBold"):
    f = ImageFont.truetype(FONT_FILE, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


# -- the data for one instant ------------------------------------------------------------------
@dataclass
class Moment:
    """Everything one frame shows. ``None`` fields are blanked, never invented."""
    elapsed: float | None              # s since the start of the piece
    distance: float | None             # m from the start
    rate: float | None                 # strokes / min
    split: float | None                # s per 500 m
    per_stroke: float | None           # m per stroke
    rate_estimated: bool = False       # the rate is the camera's estimate, not the CoxBox's
    history: list = field(default_factory=list)   # [(t, rate, split)] over the last 30 s
    track: np.ndarray | None = None    # (n, 2) course or GPS track, metres, north up
    position: tuple | None = None      # the boat's (x, y) on that track
    sample: bool = False               # synthetic numbers, for the look only


def fmt_split(s):
    if s is None:
        return "–:––.–"
    m, r = divmod(s, 60.0)
    return "%d:%04.1f" % (m, r)


def fmt_time(s):
    if s is None:
        return "–:––"
    m, r = divmod(int(s), 60)
    return "%d:%02d" % (m, r)


def fmt_dist(d):
    return "–" if d is None else "{:,}".format(int(round(d)))


# -- building blocks ----------------------------------------------------------------------------
def panel(draw, box, alpha=PANEL_ALPHA, radius=10, fill=NAVY):
    draw.rounded_rectangle(box, radius=radius, fill=fill + (alpha,))


def text(draw, xy, s, size, fill=WHITE, weight="SemiBold", anchor="ls"):
    draw.text(xy, s, font=font(size, weight), fill=fill, anchor=anchor)
    return draw.textlength(s, font=font(size, weight))


def figure(draw, x, base, value, unit, size, unit_size, fill=WHITE, unit_fill=LABEL):
    """A number and its unit on one baseline; returns the x after them."""
    w = text(draw, (x, base), value, size, fill)
    if unit:
        w += 8 + text(draw, (x + w + 8, base), unit, unit_size, unit_fill, "Light")
    return x + w


def trace(draw, box, m: Moment, window=30.0):
    """Rate (red) and split (white) over the last ``window`` seconds, each in its own lane with
    its range printed, so a change after a call reads at a glance; a faster split plots higher."""
    x0, y0, x1, y1 = box
    panel(draw, box)
    pad, label_w = 12, 168
    hist = [h for h in m.history if m.elapsed is None or h[0] >= m.elapsed - window]
    rate_col = AMBER if m.rate_estimated else RED_BRIGHT
    lanes = [("rate", 1, rate_col, 4.0, lambda v: "%d" % round(v), False)]
    if not m.rate_estimated:
        lanes.append(("split", 2, WHITE, 5.0, fmt_split, True))
    text(draw, (x1 - pad, y1 - 8), "last %d s" % window, 16, LABEL, "Light", anchor="rs")
    lane_h = (y1 - y0 - 2 * pad - 14) / len(lanes)
    for k, (name, col, colour, half, fmt, invert) in enumerate(lanes):
        top = y0 + pad + k * lane_h
        bot = top + lane_h - 8
        vals = [(h[0], h[col]) for h in hist if h[col] is not None]
        base = (top + bot) / 2 + 10
        text(draw, (x0 + pad, base), name, 20, colour, "Regular")
        if len(vals) < 2:
            continue
        t = np.array([u for u, _ in vals])
        v = np.array([w for _, w in vals])
        mid = float(np.median(v))
        lo, hi = mid - half, mid + half
        text(draw, (x0 + pad + 54, base), fmt(v[-1]), 28, colour)
        gx = lambda u: x0 + pad + label_w + (u - (t[-1] - window)) / window * (x1 - x0 - 2 * pad - label_w)
        pts = []
        for u, w in zip(t, v):
            f = (w - lo) / (hi - lo)
            f = f if invert else 1 - f
            pts.append((gx(u), top + 4 + min(max(f, 0), 1) * (bot - top - 4)))
        draw.line([(x0 + pad + label_w, (top + bot) / 2), (x1 - pad, (top + bot) / 2)],
                  fill=LABEL + (70,), width=1)
        draw.line(pts, fill=colour, width=4, joint="curve")


def course_map(draw, box, m: Moment):
    """The track, north up, with the boat's position."""
    x0, y0, x1, y1 = box
    if m.track is None:
        return                                   # no GPS: no panel, rather than an empty box
    panel(draw, box)
    tr = np.asarray(m.track, float)
    pad = 16
    lo, hi = tr.min(axis=0), tr.max(axis=0)
    scale = min((x1 - x0 - 2 * pad) / (hi[0] - lo[0]), (y1 - y0 - 2 * pad) / (hi[1] - lo[1]))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    mid = (lo + hi) / 2
    to_px = lambda p: (cx + (p[0] - mid[0]) * scale, cy - (p[1] - mid[1]) * scale)
    draw.line([to_px(p) for p in tr], fill=WHITE + (210,), width=4, joint="curve")
    if m.position is not None:
        px, py = to_px(m.position)
        draw.ellipse((px - 10, py - 10, px + 10, py + 10), fill=RED, outline=WHITE, width=3)


def stamp(draw, xy, m: Moment, anchor="rs", size=20):
    if m.sample:
        text(draw, xy, "sample data", size, AMBER, "Regular", anchor=anchor)


def rate_figure(m):
    if m.rate is None:
        return "–", WHITE
    if m.rate_estimated:
        return "~%d" % round(m.rate), AMBER
    return "%d" % round(m.rate), WHITE


# -- layouts ---------------------------------------------------------------------------------------
def layout_a(frame: Image.Image, m: Moment) -> Image.Image:
    """A: one slim strip across the sky; trace and map tucked under its ends."""
    W, H = frame.size
    over = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    h = 74
    d.rectangle((0, 0, W, h), fill=NAVY + (PANEL_ALPHA,))
    d.rectangle((0, h, W, h + 5), fill=RED + (235,))
    base, big, small = 54, 46, 24
    rate, rate_col = rate_figure(m)
    cols = [(rate, "spm", rate_col),
            (fmt_split(None if m.rate_estimated else m.split), "/500 m", GREY if m.split is None else WHITE),
            (fmt_dist(m.distance), "m", GREY if m.distance is None else WHITE),
            (fmt_time(m.elapsed), "", WHITE),
            ("–" if m.per_stroke is None else "%.1f" % m.per_stroke, "m/stroke", GREY if m.per_stroke is None else WHITE)]
    x = 36
    for value, unit, col in cols:
        end = figure(d, x, base, value, unit, big, small, col)
        x = max(end + 64, x + 300)
    if m.rate_estimated:
        text(d, (x, base), "rate from camera", 24, AMBER, "Regular")
    stamp(d, (W - 28, base), m)
    trace(d, (16, h + 18, 16 + 440, h + 18 + 140), m)
    course_map(d, (W - 16 - 280, h + 18, W - 16, h + 18 + 190), m)
    return Image.alpha_composite(frame.convert("RGBA"), over).convert("RGB")


def layout_b(frame: Image.Image, m: Moment) -> Image.Image:
    """B: big rate and split top left, trace top centre, map top right."""
    W, H = frame.size
    over = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    box = (16, 16, 16 + 600, 16 + 214)
    panel(d, box)
    d.rectangle((16, 26, 24, 16 + 204), fill=RED + (240,))
    rate, rate_col = rate_figure(m)
    x = figure(d, 48, 132, rate, "spm", 112, 30, rate_col)
    if not m.rate_estimated:                     # no CoxBox: no split, rather than dashes
        figure(d, max(x + 40, 300), 132, fmt_split(m.split), "/500 m", 84, 28,
               GREY if m.split is None else WHITE)
    if m.rate_estimated:
        text(d, (48, 196), "rate estimated from the camera · no CoxBox", 28, AMBER, "Regular")
    else:
        text(d, (48, 196), "%s m   ·   %s   ·   %s m/stroke" % (
            fmt_dist(m.distance), fmt_time(m.elapsed), "–" if m.per_stroke is None else "%.1f" % m.per_stroke),
            32, LABEL, "Regular")
    trace(d, (640, 16, 640 + 540, 16 + 150), m)
    course_map(d, (W - 16 - 320, 16, W - 16, 16 + 214), m)
    stamp(d, (640 + 540, 16 + 150 + 30), m)
    return Image.alpha_composite(frame.convert("RGBA"), over).convert("RGB")


def layout_d(frame: Image.Image, m: Moment) -> Image.Image:
    """D: the picture at 80%, the data beside it; nothing in the footage is covered."""
    W, H = frame.size
    canvas = Image.new("RGBA", (W, H), NAVY_DEEP + (255,))
    s = 0.8
    vw, vh = int(W * s), int(H * s)
    canvas.paste(frame.convert("RGBA").resize((vw, vh), Image.LANCZOS), (W - vw, (H - vh) // 2))
    d = ImageDraw.Draw(canvas)
    col = W - vw
    d.rectangle((col - 6, 0, col, H), fill=RED + (255,))
    x = 32
    rate, rate_col = rate_figure(m)
    text(d, (x, 70), "stroke rate" + (" · estimate" if m.rate_estimated else ""), 26,
         AMBER if m.rate_estimated else LABEL, "Light")
    text(d, (x, 170), rate, 104, rate_col)
    text(d, (x, 230), "split /500 m", 26, LABEL, "Light")
    text(d, (x, 300), fmt_split(None if m.rate_estimated else m.split), 68,
         GREY if m.split is None or m.rate_estimated else WHITE)
    text(d, (x, 360), "distance", 26, LABEL, "Light")
    text(d, (x, 412), fmt_dist(m.distance) + (" m" if m.distance is not None else ""), 50,
         GREY if m.distance is None else WHITE)
    text(d, (x, 470), "time", 26, LABEL, "Light")
    text(d, (x, 522), fmt_time(m.elapsed), 50)
    text(d, (x, 580), "per stroke", 26, LABEL, "Light")
    text(d, (x, 630), "–" if m.per_stroke is None else "%.1f m" % m.per_stroke, 44,
         GREY if m.per_stroke is None else WHITE)
    trace(d, (16, 660, col - 22, 660 + 170), m)
    course_map(d, (16, 846, col - 22, H - 16), m)
    stamp(d, (W - 24, H - 36), m)
    return canvas.convert("RGB")


LAYOUTS = {"A": layout_a, "B": layout_b, "D": layout_d}


# -- sample data on the Head of the Lake course --------------------------------------------------
def hotl_course():
    path = os.path.join(ROOT, "data", "hotl_course.npy")
    return np.load(path) if os.path.exists(path) else None


def sample_moment(elapsed, rate=31.0, split=122.0, estimated=False):
    """Synthetic numbers for the look only: a steady piece with a rate rise 12 s ago."""
    track = hotl_course()
    hist = []
    for k in range(0, 31):
        t = elapsed - 30 + k
        wobble = (30 - k) / 30.0                 # ends exactly on the current values
        r = rate - (2.0 if k < 18 else 0.0) + 0.6 * math.sin(k * 0.9) * wobble
        sp = split + (3.0 if k < 18 else 0.0) + 0.8 * math.cos(k * 0.7) * wobble
        hist.append((t, r, None if estimated else sp))
    distance = elapsed * 500.0 / split
    pos = None
    if track is not None:
        seg = np.sqrt((np.diff(track, axis=0) ** 2).sum(axis=1))
        cum = np.concatenate([[0], np.cumsum(seg)])
        f = min(distance, cum[-1])
        pos = (np.interp(f, cum, track[:, 0]), np.interp(f, cum, track[:, 1]))
    if estimated:
        return Moment(elapsed=None, distance=None, rate=rate, split=None, per_stroke=None,
                      rate_estimated=True, history=hist, track=None, position=None, sample=True)
    return Moment(elapsed=elapsed, distance=distance, rate=rate, split=split,
                  per_stroke=(500.0 / split) * 60.0 / rate, history=hist, track=track, position=pos,
                  sample=True)


# -- frames from the video -------------------------------------------------------------------------
def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def grab_frame(video, seconds):
    """One frame at ``seconds``, as an 8-bit RGB image (the Osmo records 10-bit HEVC)."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % seconds, "-i", video,
           "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    probe = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", video], capture_output=True, text=True).stderr
    import re
    w, h = map(int, re.search(r"Video:.*? (\d{3,5})x(\d{3,5})", probe).groups())
    return Image.frombytes("RGB", (w, h), raw[:w * h * 3])


def mmss(s):
    if ":" in s:
        m, r = s.split(":")
        return 60 * float(m) + float(r)
    return float(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--at", required=True, nargs="+", help="video times, m:ss")
    ap.add_argument("--layout", default="A", help="A, B, D or several, e.g. ABD")
    ap.add_argument("--sample", action="store_true", help="synthetic numbers, for the look only")
    ap.add_argument("--estimate", action="store_true", help="show the no-CoxBox, camera-estimate case")
    ap.add_argument("--race-start", default="0:00", help="video time of the start of the piece (sample)")
    ap.add_argument("--rate", type=float, default=31.0)
    ap.add_argument("--split", default="2:02", help="sample split per 500 m, m:ss")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "local", "overlay", "frames"))
    a = ap.parse_args()
    if not a.sample:
        raise SystemExit("real CoxBox data is not wired yet: use --sample for the look")
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.video))[0]
    for at in a.at:
        t = mmss(at)
        frame = grab_frame(a.video, t)
        m = sample_moment(t - mmss(a.race_start), a.rate, mmss(a.split), a.estimate)
        for lay in a.layout.upper():
            img = LAYOUTS[lay](frame, m)
            path = os.path.join(a.out, "%s_%s_%s%s.jpg" % (stem, at.replace(":", "m"), lay,
                                                         "_est" if a.estimate else ""))
            img.save(path, quality=90)
            print("wrote", path)


if __name__ == "__main__":
    main()
