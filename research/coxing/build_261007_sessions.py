"""Practice of 7 Oct 2026 (four, Lake Sammamish) as callmodel sessions: six ~800 m pieces and the warm-up.

Audio: data/local/coxing/raw_audio/261007_001.MP3, transcribed (faster-whisper small.en, word times).
Clock: recording started 08:44:55 local, fitted from 118 called splits (85 within 2 s of the CoxBox).
A word at recording time r is at session time r - (session start - recording start).
"""
import glob, json, os, re, sys
from datetime import datetime
ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
import coxbox_data as C
T0 = datetime(2026, 10, 7, 8, 44, 55).timestamp()
segs = json.load(open(os.path.join(ROOT, "data/local/overlay/calls/261007_001.json"), encoding="utf-8"))
segs = segs["segments"] if isinstance(segs, dict) else segs
words = [(w["start"], w["end"], w["word"].strip()) for g in segs for w in g.get("words", []) if w["word"].strip()]
PIECES = [("control_02", "0850", "control", "warm-up: 20 on / 20 off at 26, 28, 30 with builds; drill focus: foot-plate pressure, connection at the catch"),
          ("practice_03", "0903", "practice_race", "piece 1 of 6 (800 m): full calling, 'that's the change, hold it' calls"),
          ("practice_04", "0920", "practice_race", "piece 2 (800 m): full calling, splits only for the first ~40 s; after first-three-inches drill"),
          ("practice_05", "0936", "practice_race", "piece 3 (800 m): full calling with drill references; after dead-bugs drill; wake early on"),
          ("practice_06", "0948", "practice_race", "piece 4 (800 m): data-only calls (rate, split, distance)"),
          ("practice_07", "1003", "practice_race", "piece 5 (800 m): minimal calling, about every 10-15 strokes"),
          ("practice_08", "1014", "practice_race", "piece 6 (900 m, extra 100 m unannounced): full calling")]
OUT = os.path.join(ROOT, "data", "local", "coxing", "sessions")
for sid, hhmm, kind, note in PIECES:
    f = [x for x in glob.glob(os.path.join(ROOT, "data/local/coxbox/*20261007 %sAM.csv" % hhmm))][0]
    st = [l for l in open(f) if l.startswith("Start Time")][0].split(",")[1].strip()
    off = datetime.strptime(st, "%m/%d/%Y %H:%M:%S").timestamp() - T0      # recording time of session t=0
    s = C.load_csv(f)
    d = os.path.join(OUT, sid); os.makedirs(d, exist_ok=True)
    ws = [(a - off, b - off, w) for a, b, w in words if s.t[0] - 30 <= a - off <= s.t[-1] + 10]
    with open(os.path.join(d, "words.jsonl"), "w", encoding="utf-8") as fo:
        for a, b, w in ws:
            fo.write(json.dumps({"t": round(a, 2), "w": w}) + "\n")
    with open(os.path.join(d, "boat.csv"), "w", encoding="utf-8") as fo:
        fo.write("t,speed,rate,distance\n")
        for t, sp, r, dist in zip(s.t, s.speed, s.rate, s.distance):
            fo.write("%.2f,%.3f,%.1f,%.1f\n" % (t, sp, r, dist))
    # phrases: break at a pause > 0.6 s, at sentence or comma punctuation, or after 9 words
    P, cur = [], []
    for a, b, w in ws:
        if cur and (a - cur[-1][1] > 0.6 or len(cur) >= 9):
            P.append(cur); cur = []
        cur.append((a, b, w))
        if w[-1:] in ".?!,":
            P.append(cur); cur = []
    if cur: P.append(cur)
    phrases = [dict(i=i, t=round(p[0][0], 2), t_end=round(p[-1][1], 2), text=" ".join(x[2] for x in p).strip(" .,?!")) for i, p in enumerate(P)]
    phrases = [p for p in phrases if p["text"]]
    for k, p in enumerate(phrases): p["i"] = k
    json.dump(phrases, open(os.path.join(d, "phrases.json"), "w", encoding="utf-8"), indent=0)
    open(os.path.join(d, "phrases.txt"), "w", encoding="utf-8").write("".join("%d|%d|%s\n" % (p["i"], p["t"], p["text"]) for p in phrases))
    json.dump(dict(id=sid, kind=kind, boat_class="4+", date="7 Oct 2026", source="audio recorder 261007_001.MP3",
                   boat_source="CoxBox CBGPS per-stroke export", sync_offset_s=round(off, 2), audio="261007", note=note,
                   t0=float(s.t[0]), t1=float(s.t[-1])), open(os.path.join(d, "meta.json"), "w"), indent=1)
    inside = sum(1 for p in phrases if s.t[0] <= p["t"] <= s.t[-1])
    print("%-12s %3d strokes, %3d phrases (%d inside the piece), offset %.1f s" % (sid, len(s.t), len(phrases), inside, off))
