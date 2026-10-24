"""The 2026 head-camera pieces as callmodel sessions (data/local/coxing/sessions/<id>/).

Words come from the overlay tool's local transcripts (faster-whisper small.en, word timestamps),
moved onto the CoxBox clock with each video's sync offset (video time of the CoxBox's elapsed zero,
from tools/overlay; validated against called splits). Boat rows are the CoxBox per-stroke log.
"""
import glob
import json
import os
import sys

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
import coxbox_data as C  # noqa: E402

CACHE = os.path.join(os.environ["LOCALAPPDATA"], "CoxBoxOverlay", "cache")
OUT = os.path.join(ROOT, "data", "local", "coxing", "sessions")
PIECES = [  # id, video, coxbox session, offset (s), kind, boat, note
    ("race_04", "DJI_20261004111335_0132_D", "20261004 0803AM", 275.7, "race", "8+",
     "Tail of the Lake 2026, mixed masters eight, 4 Oct 2026"),
    ("race_05", "DJI_20261004154606_0134_D", "20261004 1237PM", 315.9, "race", "4+",
     "Tail of the Lake 2026, women's masters four, 4 Oct 2026"),
    ("practice_01", "DJI_20260925090905_0117_D", "20260925 0555AM", 15.45, "practice_race", "8+",
     "practice race 1, eight, 25 Sep 2026 (sync ambiguous by one stroke: 13.4 or 15.5 s)"),
    ("practice_02", "DJI_20260925093420_0119_D", "20260925 0620AM", -13.3, "practice_race", "8+",
     "practice race 2, eight, 25 Sep 2026 (recording starts mid-piece)"),
    ("control_01", "DJI_20261002083010_0123_D", "20261002 0525AM", 587.7, "control", "4+",
     "warm-up, 2 Oct 2026: 20 on / 20 off, way-enough and row; qualitatively different, a control"),
]

for sid, vid, cb, off, kind, boat, note in PIECES:
    calls = glob.glob(os.path.join(CACHE, vid + "_*_calls.json"))
    segs = json.load(open(calls[0], encoding="utf-8"))
    s = C.load_csv(glob.glob(os.path.join(ROOT, "data", "local", "coxbox", "*%s.csv" % cb))[0])
    d = os.path.join(OUT, sid)
    os.makedirs(d, exist_ok=True)
    words = [(w["start"] - off, w["word"].strip()) for seg in segs for w in seg.get("words", []) if w["word"].strip()]
    with open(os.path.join(d, "words.jsonl"), "w", encoding="utf-8") as f:
        for t, w in words:
            f.write(json.dumps({"t": round(t, 2), "w": w}) + "\n")
    with open(os.path.join(d, "boat.csv"), "w", encoding="utf-8") as f:
        f.write("t,speed,rate,distance\n")
        for t, sp, r, dist in zip(s.t, s.speed, s.rate, s.distance):
            f.write("%.2f,%.3f,%.1f,%.1f\n" % (t, sp, r, dist))
    # the segments too, on the same clock, for coding utterances
    with open(os.path.join(d, "segments.json"), "w", encoding="utf-8") as f:
        json.dump([dict(t0=round(g["start"] - off, 2), t1=round(g["end"] - off, 2), text=g["text"].strip())
                   for g in segs], f, indent=0, ensure_ascii=False)
    json.dump(dict(id=sid, kind=kind, boat_class=boat, date=note.split(", ")[-1], source="head camera " + vid,
                   boat_source="CoxBox CBGPS per-stroke export", sync_offset_s=off, note=note,
                   t0=float(s.t[0]), t1=float(s.t[-1])), open(os.path.join(d, "meta.json"), "w"), indent=1)
    inside = sum(1 for t, _ in words if s.t[0] <= t <= s.t[-1])
    print("%-12s %5d words (%d inside the CoxBox log), %4d strokes, %s" % (sid, len(words), inside, len(s.t), note))
