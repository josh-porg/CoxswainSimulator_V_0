"""An audio-only coxswain recording as a callmodel session (the coxing study's sessions/<id>/).

    .venv-tools/Scripts/python.exe research/coxing/build_audio_session.py <session id> <key> <coxbox csv> \
        --note "..." [--kind race] [--boat 4+] [--date "17 Oct 2026"]

The same files and rules as build_hocr_session.py (race_06), from an audio recording instead of a head camera:
words from transcribe_audio.py (data/local/overlay/calls/<key>_nocond.json) moved onto the CoxBox clock with the
stroke-lock sync of sync_audio_coxbox.py (data/local/coxing/sync/<key>.json: recording seconds at the CoxBox's
elapsed zero, written to meta.json as sync_offset_s so delivery measures read the recording at phrase time +
offset); phrases by build_261007_sessions.py's rule (break at a pause over 0.6 s, at sentence or comma
punctuation, or after 9 words); boat rows from the CoxBox per-stroke log. The recording's 16 kHz WAV is
data/local/coxing/experiment/audio/<key>.wav (transcribe_audio.py).
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
import coxbox_data as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("key")
    ap.add_argument("coxbox")
    ap.add_argument("--note", required=True)
    ap.add_argument("--kind", default="race")
    ap.add_argument("--boat", default="4+")
    ap.add_argument("--date", default="")
    a = ap.parse_args()
    off = float(json.load(open(os.path.join(ROOT, "data", "local", "coxing", "sync", a.key + ".json")))["offset"])
    segs = json.load(open(os.path.join(ROOT, "data", "local", "overlay", "calls", a.key + "_nocond.json"), encoding="utf-8"))
    s = C.load_csv(a.coxbox)
    d = os.path.join(ROOT, "data", "local", "coxing", "sessions", a.sid)
    os.makedirs(d, exist_ok=True)
    words = [(w["start"] - off, w["end"] - off, w["word"].strip()) for g in segs for w in g.get("words", []) if w["word"].strip()]
    ws = [x for x in words if s.t[0] - 30 <= x[0] <= s.t[-1] + 10]
    with open(os.path.join(d, "words.jsonl"), "w", encoding="utf-8") as f:
        for t0, _t1, w in ws:
            f.write(json.dumps({"t": round(t0, 2), "w": w}) + "\n")
    with open(os.path.join(d, "boat.csv"), "w", encoding="utf-8") as f:
        f.write("t,speed,rate,distance\n")
        for t, sp, r, dist in zip(s.t, s.speed, s.rate, s.distance):
            f.write("%.2f,%.3f,%.1f,%.1f\n" % (t, sp, r, dist))
    json.dump([dict(t0=round(g["start"] - off, 2), t1=round(g["end"] - off, 2), text=g["text"].strip()) for g in segs
               if s.t[0] - 30 <= g["start"] - off <= s.t[-1] + 10], open(os.path.join(d, "segments.json"), "w", encoding="utf-8"),
              indent=0, ensure_ascii=False)
    P, cur = [], []
    for t0, t1, w in ws:
        if cur and (t0 - cur[-1][1] > 0.6 or len(cur) >= 9):
            P.append(cur)
            cur = []
        cur.append((t0, t1, w))
        if w[-1:] in ".?!,":
            P.append(cur)
            cur = []
    if cur:
        P.append(cur)
    phrases = [dict(t=round(p[0][0], 2), t_end=round(p[-1][1], 2), text=" ".join(x[2] for x in p).strip(" .,?!")) for p in P]
    phrases = [dict(i=k, **p) for k, p in enumerate(q for q in phrases if q["text"])]
    json.dump(phrases, open(os.path.join(d, "phrases.json"), "w", encoding="utf-8"), indent=0)
    open(os.path.join(d, "phrases.txt"), "w", encoding="utf-8").write("".join("%d|%d|%s\n" % (p["i"], p["t"], p["text"]) for p in phrases))
    json.dump(dict(id=a.sid, kind=a.kind, boat_class=a.boat, date=a.date, source="coxswain audio recording " + a.key,
                   boat_source="CoxBox CBGPS per-stroke export", sync_offset_s=off,
                   sync_method="stroke lock on the 20-150 Hz envelope (sync_audio_coxbox.py)", note=a.note,
                   t0=float(s.t[0]), t1=float(s.t[-1])), open(os.path.join(d, "meta.json"), "w"), indent=1)
    inside = sum(1 for p in phrases if s.t[0] <= p["t"] <= s.t[-1])
    print("%s: %d strokes, %d words, %d phrases (%d inside the race)" % (a.sid, len(s.t), len(ws), len(phrases), inside))


if __name__ == "__main__":
    main()
