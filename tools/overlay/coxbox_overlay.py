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
#: no console window flashing up for each ffmpeg call when run from the app (Windows)
NOWIN = dict(creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def resource(*parts):
    """A file shipped with the app: next to this module from source, inside the bundle when frozen."""
    import sys
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


#: Barlow (SIL Open Font License, fonts/OFL.txt): shipped, so every platform draws the same; used
#: where Windows' Bahnschrift (the approved look, not redistributable) is absent
BARLOW = {"Light": "Barlow-Light.ttf", "Regular": "Barlow-Regular.ttf", "SemiBold": "Barlow-SemiBold.ttf"}


@__import__("functools").lru_cache(maxsize=256)
def font(size, weight="SemiBold"):
    if os.path.exists(FONT_FILE):
        f = ImageFont.truetype(FONT_FILE, size)
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
        return f
    return ImageFont.truetype(resource("fonts", BARLOW.get(weight, BARLOW["SemiBold"])), size)


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
def _blank(size):
    return Image.new("RGBA", size, (0, 0, 0, 0))


#: the HUD items a coxswain can switch on or off
FIELDS = ("rate", "split", "distance", "time", "per_stroke", "trace", "map")
FIELD_NAMES = {"rate": "Stroke rate", "split": "Split /500 m", "distance": "Distance", "time": "Elapsed time",
               "per_stroke": "Metres per stroke", "trace": "30 s trace", "map": "Course map"}


def _fields(fields):
    return set(FIELDS) if fields is None else set(fields)


def overlay_a(size, m: Moment, fields=None) -> Image.Image:
    """A: one slim strip across the sky; trace and map tucked under its ends (RGBA layer)."""
    f = _fields(fields)
    W, H = size
    over = _blank(size)
    d = ImageDraw.Draw(over)
    h = 74
    numbers = [k for k in ("rate", "split", "distance", "time", "per_stroke") if k in f]
    if numbers:
        d.rectangle((0, 0, W, h), fill=NAVY + (PANEL_ALPHA,))
        d.rectangle((0, h, W, h + 5), fill=RED + (235,))
    base, big, small = 54, 46, 24
    rate, rate_col = rate_figure(m)
    items = {"rate": (rate, "spm", rate_col),
             "split": (fmt_split(None if m.rate_estimated else m.split), "/500 m", GREY if m.split is None else WHITE),
             "distance": (fmt_dist(m.distance), "m", GREY if m.distance is None else WHITE),
             "time": (fmt_time(m.elapsed), "", WHITE if m.elapsed is not None else GREY),
             "per_stroke": ("–" if m.per_stroke is None else "%.1f" % m.per_stroke, "m/stroke",
                            GREY if m.per_stroke is None else WHITE)}
    if m.rate_estimated:                         # no CoxBox: only the rate exists
        numbers = [k for k in numbers if k == "rate"]
    x = 36
    for k in numbers:
        value, unit, col = items[k]
        end = figure(d, x, base, value, unit, big, small, col)
        x = max(end + 64, x + 300)
    if m.rate_estimated and numbers:
        text(d, (x, base), "rate from camera", 24, AMBER, "Regular")
    top = h + 18 if numbers else 16
    if numbers:
        stamp(d, (W - 28, base), m)
    if "trace" in f:
        trace(d, (16, top, 16 + 440, top + 140), m)
    if "map" in f:
        course_map(d, (W - 16 - 280, top, W - 16, top + 190), m)
    return over


def overlay_b(size, m: Moment, fields=None) -> Image.Image:
    """B: big rate and split top left, trace top centre, map top right (RGBA layer)."""
    f = _fields(fields)
    W, H = size
    over = _blank(size)
    d = ImageDraw.Draw(over)
    small = [k for k in ("distance", "time", "per_stroke") if k in f and not m.rate_estimated]
    show_split = "split" in f and not m.rate_estimated
    if "rate" in f or show_split or small or m.rate_estimated:
        box = (16, 16, 16 + 600, 16 + 214)
        panel(d, box)
        d.rectangle((16, 26, 24, 16 + 204), fill=RED + (240,))
        x = 48
        if "rate" in f:
            rate, rate_col = rate_figure(m)
            x = figure(d, 48, 132, rate, "spm", 112, 30, rate_col) + 40
        if show_split:
            figure(d, max(x, 48 if "rate" not in f else 300), 132, fmt_split(m.split), "/500 m", 84, 28,
                   GREY if m.split is None else WHITE)
        if m.rate_estimated:
            text(d, (48, 196), "rate estimated from the camera · no CoxBox", 28, AMBER, "Regular")
        elif small:
            parts = {"distance": "%s m" % fmt_dist(m.distance), "time": fmt_time(m.elapsed),
                     "per_stroke": "%s m/stroke" % ("–" if m.per_stroke is None else "%.1f" % m.per_stroke)}
            text(d, (48, 196), "   ·   ".join(parts[k] for k in small), 32, LABEL, "Regular")
    if "trace" in f:
        trace(d, (640, 16, 640 + 540, 16 + 150), m)
    if "map" in f:
        course_map(d, (W - 16 - 320, 16, W - 16, 16 + 214), m)
    stamp(d, (640 + 540, 16 + 150 + 30), m)
    return over


#: layout D: the picture at this scale, right-aligned and centred vertically
D_SCALE = 0.8


def d_geometry(size):
    W, H = size
    vw, vh = int(round(W * D_SCALE / 2) * 2), int(round(H * D_SCALE / 2) * 2)
    return vw, vh, W - vw, (H - vh) // 2


def overlay_d(size, m: Moment, fields=None) -> Image.Image:
    """D: the data column beside the picture; the picture's own area is left transparent."""
    f = _fields(fields)
    W, H = size
    vw, vh, vx, vy = d_geometry(size)
    over = Image.new("RGBA", size, NAVY_DEEP + (255,))
    d = ImageDraw.Draw(over)
    d.rectangle((vx, vy, vx + vw - 1, vy + vh - 1), fill=(0, 0, 0, 0))
    col = vx
    d.rectangle((col - 6, 0, col - 1, H), fill=RED + (255,))
    x, y = 32, 70
    rate, rate_col = rate_figure(m)
    rows = []
    if "rate" in f:
        rows.append(("stroke rate" + (" · estimate" if m.rate_estimated else ""), rate, 104, rate_col,
                     AMBER if m.rate_estimated else LABEL))
    if not m.rate_estimated:
        if "split" in f:
            rows.append(("split /500 m", fmt_split(m.split), 68, GREY if m.split is None else WHITE, LABEL))
        if "distance" in f:
            rows.append(("distance", fmt_dist(m.distance) + (" m" if m.distance is not None else ""), 50,
                         GREY if m.distance is None else WHITE, LABEL))
        if "time" in f:
            rows.append(("time", fmt_time(m.elapsed), 50, WHITE, LABEL))
        if "per_stroke" in f:
            rows.append(("per stroke", "–" if m.per_stroke is None else "%.1f m" % m.per_stroke, 44,
                         GREY if m.per_stroke is None else WHITE, LABEL))
    for label, value, size_, colour, label_colour in rows:
        text(d, (x, y), label, 26, label_colour, "Light")
        y += 8 + int(size_ * 0.95)
        text(d, (x, y), value, size_, colour)
        y += 50
    lower = 660
    if "trace" in f:
        trace(d, (16, lower, col - 22, lower + 170), m)
    if "map" in f:
        course_map(d, (16, 846, col - 22, H - 16), m)
    stamp(d, (W - 24, H - 36), m)
    return over


def blank_d(size):
    """Layout D outside the piece: the frame and column, no data."""
    W, H = size
    vw, vh, vx, vy = d_geometry(size)
    over = Image.new("RGBA", size, NAVY_DEEP + (255,))
    d = ImageDraw.Draw(over)
    d.rectangle((vx, vy, vx + vw - 1, vy + vh - 1), fill=(0, 0, 0, 0))
    d.rectangle((vx - 6, 0, vx - 1, H), fill=RED + (255,))
    return over


OVERLAYS = {"A": overlay_a, "B": overlay_b, "D": overlay_d}

#: layers are designed at this width and drawn at the video's own aspect ratio, then scaled to the
#: frame, so 720p, 1080p, 2.7K and 4K (16:9 or 4:3) all get the same proportions
REF_WIDTH = 1920


def layer(layout, size, m, fields=None):
    """The overlay layer (RGBA) for a frame of ``size``; empty (or D's frame) when ``m`` is None."""
    ref = (REF_WIDTH, int(round(REF_WIDTH * size[1] / size[0])))
    if m is None:
        img = blank_d(ref) if layout == "D" else _blank(ref)
    else:
        img = OVERLAYS[layout](ref, m, fields)
    return img if ref == tuple(size) else img.resize(tuple(size), Image.LANCZOS)


def compose(frame: Image.Image, layout: str, m: Moment | None, fields=None) -> Image.Image:
    """A still: the frame with the layout's layer over it (D shrinks the frame into its slot)."""
    size = frame.size
    if layout == "D":
        vw, vh, vx, vy = d_geometry(size)
        base = Image.new("RGBA", size, NAVY_DEEP + (255,))
        base.paste(frame.convert("RGBA").resize((vw, vh), Image.LANCZOS), (vx, vy))
        lay = layer("D", size, m, fields)
    else:
        base = frame.convert("RGBA")
        lay = layer(layout, size, m, fields)
    return Image.alpha_composite(base, lay).convert("RGB")


def layout_a(frame, m):
    return compose(frame, "A", m)


def layout_b(frame, m):
    return compose(frame, "B", m)


def layout_d(frame, m):
    return compose(frame, "D", m)


LAYOUTS = {"A": layout_a, "B": layout_b, "D": layout_d}


# -- real data: a CoxBox session placed on the video ---------------------------------------------
#: after the CoxBox session ends, its final numbers stay up this long
HOLD_AFTER = 15.0


def session_moments(session, offset, window=30.0):
    """A function of video time giving the Moment to show, or None outside the piece.

    ``offset``: the video time of the session's elapsed zero (coxbox_data.sync_by_motion). Values
    change at each logged stroke, as on the CoxBox; distance and elapsed time run continuously, and
    the boat's position on its GPS track is interpolated between strokes."""
    t = session.t
    xy = session.track_xy()
    ok = ~np.isnan(xy).any(axis=1)

    def at(tv):
        e = tv - offset
        if e < 0 or e > session.duration + HOLD_AFTER:
            return None
        e_show = min(e, session.duration)
        k = int(np.searchsorted(t, e_show, side="right")) - 1
        k = max(k, 0)
        hist_idx = np.flatnonzero((t >= e_show - window) & (t <= e_show))
        history = [(float(t[i]), float(session.rate[i]), float(session.split[i])) for i in hist_idx]
        pos = None
        if ok.any():
            pos = (float(np.interp(e_show, t[ok], xy[ok, 0])), float(np.interp(e_show, t[ok], xy[ok, 1])))
        return Moment(elapsed=e_show, distance=float(np.interp(e_show, t, session.distance)),
                      rate=float(session.rate[k]), split=float(session.split[k]),
                      per_stroke=float(session.per_stroke[k]), history=history,
                      track=xy[ok] if ok.any() else None, position=pos)
    return at


def estimate_moments(mt, signal, start, end, window=30.0, min_conf=0.3):
    """No CoxBox: the stroke rate estimated from the head camera's motion, between ``start`` and
    ``end`` (video s). Only confident windows are shown; everything else is blanked."""
    from coxbox_data import windowed_rate
    c, r, q = windowed_rate(mt, signal)

    def at(tv):
        if tv < start or tv > end:
            return None
        k = int(np.argmin(np.abs(c - tv)))
        rate = float(r[k]) if q[k] >= min_conf else None
        sel = np.flatnonzero((c >= tv - window) & (c <= tv) & (q >= min_conf))
        history = [(float(c[i]), float(r[i]), None) for i in sel]
        return Moment(elapsed=None, distance=None, rate=rate, split=None, per_stroke=None,
                      rate_estimated=True, history=history)
    return at


def probe(video):
    """``(width, height, duration s, fps, has_audio)`` of a video, from ffmpeg's own report."""
    import re
    err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", video], capture_output=True, text=True, **NOWIN).stderr
    W, H = map(int, re.search(r"Video:.*? (\d{2,5})x(\d{2,5})", err).groups())
    dur = sum(float(x) * f for x, f in zip(re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups(), (3600, 60, 1)))
    fps = re.search(r"([\d.]+) fps", err)
    return W, H, dur, float(fps.group(1)) if fps else 30.0, bool(re.search(r"Stream #.*Audio:", err))


def render_video(video, out, layout, moment_at, overlay_fps=10, start=None, end=None, encoder="libx264",
                 crf=20, preset="veryfast", fields=None, progress=None, card=None):
    """The whole clip (or ``start``-``end``) with the layout composited by ffmpeg.

    The overlay layer is drawn at ``overlay_fps`` and piped raw (RGBA) as a second input; ffmpeg
    holds each layer until the next, scales and pads the picture for D, and encodes H.264 with the
    original audio. Frames outside the piece get an empty layer (D keeps its frame).

    ``card`` (title_card.TitleCard): the race's title and lineup for its first ``card.seconds``.
    ``card.under == "broll"``: that clip comes first, sped up or slowed to fill exactly that time,
    full frame and silent, and the race video follows. ``"still"``: the race video's first frame,
    held for that time, silent. ``"video"``: the card lies over the race video's own first seconds."""
    W, H, dur, fps, has_audio = probe(video)
    t0 = 0.0 if start is None else float(start)
    t1 = dur if end is None else min(float(end), dur)
    size = (W, H)
    if card is not None and (card.empty or card.seconds <= 0):
        card = None
    under = None if card is None else card.under
    if under == "broll" and not card.broll:
        raise ValueError("the title card is set to a b-roll clip, but none was chosen")
    lead = 0.0 if card is None else card.lead      # output time before the race video starts
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", "%.3f" % t0]
    cmd += ["-t", "%.3f" % (t1 - t0), "-i", video,
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "%dx%d" % size, "-r", str(overlay_fps), "-i", "-"]
    if layout == "D":
        vw, vh, vx, vy = d_geometry(size)
        main = "[0:v]scale=%d:%d,pad=%d:%d:%d:%d:color=0x0a1028,setsar=1,format=yuv420p[m]" % (vw, vh, W, H, vx, vy)
    else:
        main = "[0:v]setsar=1,format=yuv420p[m]"
    if lead > 0:
        if under == "broll":
            bdur = probe(card.broll)[2]
            cmd += ["-i", card.broll]
            pre = ("[2:v]setpts=(PTS-STARTPTS)*%.6f,fps=%.6f,scale=%d:%d:force_original_aspect_ratio=increase,"
                   "crop=%d:%d" % (card.seconds / max(bdur, 1e-3), fps, W, H, W, H))
        else:                                      # the first frame, held, full frame even in D
            main = "[0:v]split=2[v0][v1];" + main.replace("[0:v]", "[v0]", 1)
            pre = "[v1]trim=end_frame=1,setpts=PTS-STARTPTS,fps=%.6f,scale=%d:%d" % (fps, W, H)
        graph = (main + ";" + pre + ",setsar=1,format=yuv420p,tpad=stop_mode=clone:stop_duration=%.3f,"
                 "trim=duration=%.3f,setpts=PTS-STARTPTS[b];[b][m]concat=n=2:v=1:a=0[v];"
                 "[v][1:v]overlay=0:0:format=auto[o]" % (card.seconds + 1.0, card.seconds))
        if has_audio:
            graph += ";[0:a]adelay=delays=%d:all=1[a]" % round(card.seconds * 1000)
        amap = ["-map", "[a]"] if has_audio else []
    else:
        graph = main + ";[m][1:v]overlay=0:0:format=auto[o]"
        amap = ["-map", "0:a?"]
    cmd += ["-filter_complex", graph, "-map", "[o]"] + amap + ["-c:v", encoder]
    if encoder == "libx264":
        cmd += ["-crf", str(crf), "-preset", preset]
    cmd += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, **NOWIN)
    empty = layer(layout, size, None)
    empty_bytes = empty.tobytes()
    card_img = None
    if card is not None:
        import title_card as TC
        card_img = TC.card_layer(size, card)
    n_lead = int(round(lead * overlay_fps))
    n = n_lead + int(np.ceil((t1 - t0) * overlay_fps))
    try:
        for i in range(n):
            tc = i / overlay_fps                   # output time
            if i < n_lead:                         # the b-roll or still: the card alone, full frame
                buf = TC.with_opacity(card_img, TC.fade(tc, card.seconds)).tobytes()
            else:
                tv = t0 + tc - lead
                m = moment_at(tv)
                a = TC.fade(tc, card.seconds) if (card_img is not None and lead == 0) else 0.0
                if a > 0:
                    base = empty if m is None else layer(layout, size, m, fields)
                    buf = Image.alpha_composite(base, TC.with_opacity(card_img, a)).tobytes()
                else:
                    buf = empty_bytes if m is None else layer(layout, size, m, fields).tobytes()
            proc.stdin.write(buf)
            if progress is not None and i % overlay_fps == 0:
                progress((i + 1) / n)      # may raise to cancel
            elif progress is None and i % (overlay_fps * 60) == 0:
                print("  %s: %d / %d min" % (os.path.basename(out), int(tc // 60), int((n / overlay_fps) // 60)),
                      flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding %s" % out)
    except BaseException:
        proc.kill()                        # cancelled or failed: stop the encoder, drop the partial file
        proc.wait()
        try:
            os.remove(out)
        except OSError:
            pass
        raise
    return out


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
    """The ffmpeg imageio-ffmpeg ships for this platform (bundled with the app when frozen)."""
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if os.name != "nt" and not os.access(exe, os.X_OK):
        try:                                   # a packaged copy can lose its executable bit
            os.chmod(exe, os.stat(exe).st_mode | 0o111)
        except OSError:
            pass
    return exe


def grab_frame(video, seconds):
    """One frame at ``seconds``, as an 8-bit RGB image (the Osmo records 10-bit HEVC)."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % seconds, "-i", video,
           "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True, **NOWIN).stdout
    probe = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", video], capture_output=True, text=True, **NOWIN).stderr
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
