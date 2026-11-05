"""Head of the Lake 2026 (Sunday 1 November 2026, the first coxswain) as callmodel sessions race_08 and race_09.

    .venv-tools/Scripts/python.exe research/coxing/transcribe_audio.py <video 0151> 0151     (and 0153)
    .venv-tools/Scripts/python.exe research/coxing/build_hotl2026_sessions.py [race_08 race_09] [--check]

race_08 is the women's eight (CoxBox started on the line), race_09 the men's four (bow-loaded; the CoxBox was
started about 10 s past the line). The audio is the head camera's own track, so the video sync made for the race
videos is the study's sync: research/overlay/render_hotl2026.py sync wrote data/local/overlay/videos_titled/hotl2026/
sync.json, whose ``offset`` is the video second at the CoxBox's elapsed zero (the eight by the overlay's stroke-rhythm
and called-split sync; the four pinned to the eight's camera clock the same day). That offset is copied to
data/local/coxing/sync/<key>.json (recording seconds at the CoxBox zero, as sync_audio_coxbox.py writes it) and the
session is built by build_audio_session.py. Nothing is refitted.

``--check`` reports two independent checks of that sync without changing it: the audio stroke lock of
sync_audio_coxbox.py (where the 20-150 Hz envelope locks to the CoxBox strokes) and the stroke-cycle phase of the
coxswain's counts, which are said on the catch, so both races (one coxswain, one CoxBox unit) should put them at the
same phase.

Names, lineups and audio stay in data/local; the crew keys are the video numbers.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coxbox_data as C  # noqa: E402

LOCAL = os.path.join(ROOT, "data", "local")
VSYNC = os.path.join(LOCAL, "overlay", "videos_titled", "hotl2026", "sync.json")
CARDS = os.path.join(LOCAL, "overlay", "cards_hotl2026.json")
COXBOX = os.path.join(LOCAL, "raceday", "hotl2026", "coxbox")
RACES = [  # session, crew key (video number), boat, note
    ("race_08", "0151", "8+", "Head of the Lake 2026, women's eight, 1 Nov 2026 (CoxBox started on the line)"),
    ("race_09", "0153", "4+", "Head of the Lake 2026, men's four (bow-loaded), 1 Nov 2026 (CoxBox started late, "
                              "~10 s past the line: the first strokes are not in the boat record)"),
]
COUNTS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}


def video_of(key):
    return next(v for v in json.load(open(VSYNC)) if ("_%s_" % key) in v)


def write_sync(key):
    v = video_of(key)
    vs = json.load(open(VSYNC))[v]
    card = json.load(open(CARDS, encoding="utf-8"))[v]
    out = dict(key=key, coxbox=card["coxbox"], offset=vs["offset"], method="race-video sync (render_hotl2026.py sync)",
               method_note="the race video's sync (research/overlay/render_hotl2026.py sync): "
                           + ("overlay stroke rhythm and called splits" if card["on_time"]
                              else "pinned to the same day's camera clock from the eight, best stroke-timing match within 5 s"),
               late_start_s=vs["late"], late_start_m=vs["late_m"], camera_fast_min=vs["camera_fast"],
               picture_rate_err_spm=vs["rate_err"],
               note="offset = recording seconds at the CoxBox elapsed zero; the audio is the video's own track")
    p = os.path.join(LOCAL, "coxing", "sync", key + ".json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    return out


def build(only=()):
    for sid, key, boat, note in RACES:
        if only and sid not in only:
            continue
        sy = write_sync(key)
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_audio_session.py"),
                        sid, key, os.path.join(COXBOX, sy["coxbox"]), "--note", note, "--kind", "race", "--boat", boat,
                        "--date", "1 Nov 2026", "--source", "head camera audio " + os.path.splitext(video_of(key))[0]],
                       check=True)
        m = os.path.join(LOCAL, "coxing", "sessions", sid, "meta.json")
        meta = json.load(open(m))
        meta.update(late_start_s=sy["late_start_s"], late_start_m=sy["late_start_m"])
        json.dump(meta, open(m, "w"), indent=1)


def count_phase(key, s, off):
    """Stroke-cycle phase (0 = a CoxBox stroke record, 1 = the next) of every count word in the race."""
    import rhythm_calls as RC
    segs = json.load(open(os.path.join(LOCAL, "overlay", "calls", key + "_nocond.json"), encoding="utf-8"))
    tw = np.array([w["start"] - off for g in segs for w in g.get("words", [])
                   if re.sub(r"[^a-z]", "", w["word"].lower()) in COUNTS])
    tw = tw[(tw > s.t[0]) & (tw < s.t[-1])]
    ph, _ = RC.phase_in_cycle(tw, s.t)
    return RC.circ(ph)


def check(only=()):
    import sync_audio_coxbox as SA
    for sid, key, boat, _ in RACES:
        if only and sid not in only:
            continue
        sy = json.load(open(os.path.join(LOCAL, "coxing", "sync", key + ".json")))
        s = C.load_csv(os.path.join(COXBOX, sy["coxbox"]))
        wav = os.path.join(LOCAL, "coxing", "experiment", "audio", key + ".wav")
        t, envs = SA.band_envelopes(wav)
        e = envs["20-150 Hz"]
        offs = sy["offset"] + np.arange(-6.0, 6.0, 0.02)
        sc, _ = SA.stroke_lock(t, e, s.t, offs)
        k = int(np.nanargmax(sc))
        at = float(sc[np.argmin(np.abs(offs - sy["offset"]))])
        period = float(np.median(np.diff(s.t)))
        cp = count_phase(key, s, sy["offset"])
        print("%s (%s, key %s): video sync %.2f s; audio stroke lock within +-6 s peaks at %.2f s (score %.2f; %.2f at"
              " the video sync; stroke period %.2f s)" % (sid, boat, key, sy["offset"], offs[k], sc[k], at, period))
        print("   counts in the stroke cycle at the video sync: n=%d, mean phase %.2f, R %.2f" % (
            cp.get("n", 0), cp.get("mean_phase", np.nan), cp.get("R", np.nan)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="race_08 and/or race_09 (default both)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check(a.only)
    else:
        build(a.only)


if __name__ == "__main__":
    main()
