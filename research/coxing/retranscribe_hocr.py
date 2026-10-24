"""Re-transcribe the HOCR 2026 head-camera audio for the coxing study (race_06).

The overlay's transcript of this video (cached by the render) came out with no punctuation and
word times that abut (no pauses), so phrases could not be split as for the other pieces. Same
model and prompt as tools/overlay/transcribe_calls.py, but condition_on_previous_text=False, which
stops Whisper carrying an unpunctuated style from one window to the next.

    .venv-tools/Scripts/python.exe research/coxing/retranscribe_hocr.py
Writes data/local/overlay/calls/DJI_20261017105126_0149_D_nocond.json.
"""
import json, os, sys, wave
import numpy as np
ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
import transcribe_calls as TC
from faster_whisper import WhisperModel
with wave.open(os.path.join(ROOT, "data", "local", "coxing", "experiment", "audio", "0149.wav")) as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
wm = WhisperModel("small.en", device="cpu", compute_type="int8", local_files_only=True)
segs, _ = wm.transcribe(audio, language="en", word_timestamps=True, vad_filter=True, initial_prompt=TC.PROMPT,
                        beam_size=5, condition_on_previous_text=False)
out = [dict(start=s.start, end=s.end, text=s.text.strip(),
            words=[dict(word=x.word, start=x.start, end=x.end, p=x.probability) for x in (s.words or [])]) for s in segs]
p = os.path.join(ROOT, "data", "local", "overlay", "calls", "DJI_20261017105126_0149_D_nocond.json")
json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
print("wrote", p, len(out), "segments")
