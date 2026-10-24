"""Head of the Charles 2026 (women's veteran four, 17 Oct 2026) as callmodel session race_06.

Words: retranscribe_hocr.py (faster-whisper small.en, word times, condition_on_previous_text off),
moved onto the CoxBox clock with the video's sync offset (CoxBox elapsed zero at video 66.5 s, from
the race-video render). Phrases: the rule of build_261007_sessions.py (break at a pause over 0.6 s,
at sentence or comma punctuation, or after 9 words). Boat rows: the CoxBox per-stroke log, which
covers the race from the start line to the finish (1449 s, 724 strokes).
Audio for delivery measures: experiment/audio/0149.wav (16 kHz mono, as the other pieces).
"""
import glob, json, os, subprocess, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
import coxbox_data as C  # noqa: E402

SID, VID, CB, OFF = "race_06", "DJI_20261017105126_0149_D", "20261017 0737AM", 66.5
NOTE = "Head of the Charles 2026, women's veteran four, 17 Oct 2026"
MEDIA = json.load(open(os.path.join(ROOT, "data", "local", "coxing", "local_paths.json"), encoding="utf-8"))["media_root"]   # private, machine-specific
VIDEO = os.path.join(MEDIA, "HOCR 2026", "%s.MP4" % VID)
CACHE = os.path.join(os.environ["LOCALAPPDATA"], "CoxBoxOverlay", "cache")
HERE = os.path.join(ROOT, "data", "local", "coxing")
d = os.path.join(HERE, "sessions", SID)
os.makedirs(d, exist_ok=True)

segs = json.load(open(os.path.join(ROOT, "data", "local", "overlay", "calls", VID + "_nocond.json"), encoding="utf-8"))  # retranscribe_hocr.py
s = C.load_csv(glob.glob(os.path.join(ROOT, "data", "local", "coxbox", "*%s.csv" % CB))[0])
words = [(w["start"] - OFF, w["end"] - OFF, w["word"].strip()) for g in segs for w in g.get("words", []) if w["word"].strip()]
ws = [x for x in words if s.t[0] - 30 <= x[0] <= s.t[-1] + 10]
with open(os.path.join(d, "words.jsonl"), "w", encoding="utf-8") as f:
    for a, b, w in ws:
        f.write(json.dumps({"t": round(a, 2), "w": w}) + "\n")
with open(os.path.join(d, "boat.csv"), "w", encoding="utf-8") as f:
    f.write("t,speed,rate,distance\n")
    for t, sp, r, dist in zip(s.t, s.speed, s.rate, s.distance):
        f.write("%.2f,%.3f,%.1f,%.1f\n" % (t, sp, r, dist))
with open(os.path.join(d, "segments.json"), "w", encoding="utf-8") as f:
    json.dump([dict(t0=round(g["start"] - OFF, 2), t1=round(g["end"] - OFF, 2), text=g["text"].strip()) for g in segs
               if s.t[0] - 30 <= g["start"] - OFF <= s.t[-1] + 10], f, indent=0, ensure_ascii=False)
P, cur = [], []
for a, b, w in ws:
    if cur and (a - cur[-1][1] > 0.6 or len(cur) >= 9):
        P.append(cur); cur = []
    cur.append((a, b, w))
    if w[-1:] in ".?!,":
        P.append(cur); cur = []
if cur:
    P.append(cur)
phrases = [dict(t=round(p[0][0], 2), t_end=round(p[-1][1], 2), text=" ".join(x[2] for x in p).strip(" .,?!")) for p in P]
phrases = [dict(i=k, **p) for k, p in enumerate(q for q in phrases if q["text"])]
json.dump(phrases, open(os.path.join(d, "phrases.json"), "w", encoding="utf-8"), indent=0)
open(os.path.join(d, "phrases.txt"), "w", encoding="utf-8").write("".join("%d|%d|%s\n" % (p["i"], p["t"], p["text"]) for p in phrases))
json.dump(dict(id=SID, kind="race", boat_class="4+", date="17 Oct 2026", source="head camera " + VID,
               boat_source="CoxBox CBGPS per-stroke export", sync_offset_s=OFF, note=NOTE,
               t0=float(s.t[0]), t1=float(s.t[-1])), open(os.path.join(d, "meta.json"), "w"), indent=1)
wav = os.path.join(HERE, "experiment", "audio", "0149.wav")
if not os.path.exists(wav):
    FF = glob.glob(os.path.join(ROOT, ".venv-tools", "Lib", "site-packages", "imageio_ffmpeg", "binaries", "ffmpeg-*.exe"))[0]
    subprocess.run([FF, "-v", "error", "-y", "-i", VIDEO, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)
inside = sum(1 for p in phrases if s.t[0] <= p["t"] <= s.t[-1])
print("%s: %d strokes, %d words, %d phrases (%d inside the race)" % (SID, len(s.t), len(ws), len(phrases), inside))
