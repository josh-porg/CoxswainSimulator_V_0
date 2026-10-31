"""Build a race viewer for one coxswain's recording: audio, CoxBox data and a map that follows the boat.

    .venv/Scripts/python.exe research/coxing/build_viewer.py <key> <coxbox csv> <audio file> --title "..."

Needs the sync from sync_audio_coxbox.py (data/local/coxing/sync/<key>.json: recording seconds at the CoxBox's
elapsed zero) and the transcript from transcribe_audio.py (data/local/overlay/calls/<key>_nocond.json).

Writes data/local/viewer/<key>/: index.html (copied from tools/race_viewer/index.html), data.js (strokes, calls
on the CoxBox clock, course geometry) and the audio. Everything stays in data/local: the page plays a private
recording and is opened locally (a static server, e.g. the training-site-preview pattern), never published.

Course geometry is the project's own: the Head of the Charles race course (start to finish), its water edges
(water half-width either side of the centreline), the HOCR corridor limits (the Cambridge shore limit and the
orange-line side), and the bridge decks from OpenStreetMap (coxswain.river.bridges.OSM_BRIDGE_DECKS).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
import coxbox_data as C  # noqa: E402
from coxswain.river.bridges import OSM_BRIDGE_DECKS  # noqa: E402
from coxswain.river.charles import CHARLES_ORIGIN, charles_course, hocr_race_course  # noqa: E402
from coxswain.river.course import inverse_tangent_plane  # noqa: E402


def course_geometry(step=10.0):
    c = hocr_race_course()
    st = np.arange(0.0, c.length + step, step).clip(0, c.length)
    ll = lambda xy: [[round(float(a), 6), round(float(b), 6)] for a, b in zip(*inverse_tangent_plane(xy[:, 0], xy[:, 1], CHARLES_ORIGIN))]
    port, starboard = c.limits_at(st)
    # the water from the whole surveyed reach, so the boat is on water before the start line and after the finish
    reach = charles_course()
    rs = np.arange(0.0, reach.length + step, step).clip(0, reach.length)
    water = reach.water_half_width_at(rs)
    return dict(
        centre=ll(c.position_at(st)),
        water_port=ll(reach.offset_position(rs, water)), water_starboard=ll(reach.offset_position(rs, -water)),
        limit_port=ll(c.offset_position(st, port)), limit_starboard=ll(c.offset_position(st, -starboard)),
        start=ll(c.position_at(np.array([0.0])))[0], finish=ll(c.position_at(np.array([c.length])))[0],
        length=float(c.length),
        bridges=[dict(name=k, deck=[list(a), list(b)]) for k, (a, b) in OSM_BRIDGE_DECKS.items()],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("coxbox")
    ap.add_argument("audio")
    ap.add_argument("--title", default="Race viewer")
    a = ap.parse_args()
    sync = json.load(open(os.path.join(ROOT, "data", "local", "coxing", "sync", a.key + ".json")))
    off = float(sync["offset"])
    s = C.load_csv(a.coxbox)
    segs = json.load(open(os.path.join(ROOT, "data", "local", "overlay", "calls", a.key + "_nocond.json"), encoding="utf-8"))
    strokes = dict(t=[round(float(x), 2) for x in s.t], lat=[float(x) for x in s.lat], lon=[float(x) for x in s.lon],
                   speed=[round(float(x), 3) for x in s.speed], split=[round(float(x), 1) for x in s.split],
                   rate=[round(float(x), 1) for x in s.rate], distance=[round(float(x), 1) for x in s.distance],
                   per_stroke=[round(float(x), 2) for x in s.per_stroke])
    calls = [dict(t0=round(g["start"] - off, 2), t1=round(g["end"] - off, 2), text=g["text"].strip()) for g in segs]
    data = dict(title=a.title, offset=off, sync=dict(method=sync.get("method", "stroke rhythm, 20-150 Hz band"),
                                                     note=sync.get("note", "")),
                start_time=s.start.isoformat(), strokes=strokes, calls=calls, course=course_geometry(),
                audio="audio" + os.path.splitext(a.audio)[1].lower())
    out = os.path.join(ROOT, "data", "local", "viewer", a.key)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.VIEWER = ")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    shutil.copy(os.path.join(ROOT, "tools", "race_viewer", "index.html"), os.path.join(out, "index.html"))
    dst = os.path.join(out, data["audio"])
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(a.audio):
        shutil.copy(a.audio, dst)
    print("viewer at %s: %d strokes, %d calls, offset %.2f s" % (os.path.relpath(out, ROOT), len(s.t), len(calls), off))


if __name__ == "__main__":
    main()
