"""Transcribe a coxswain's audio recording with word times, for aligning to CoxBox data.

    .venv-tools/Scripts/python.exe research/coxing/transcribe_audio.py <audio file> <key>

Converts any audio (m4a, mp3, wav) to 16 kHz mono WAV at data/local/coxing/experiment/audio/<key>.wav, then
transcribes it with faster-whisper small.en, the model, prompt and settings of retranscribe_hocr.py
(word timestamps, VAD, condition_on_previous_text off so punctuation and pauses survive). Writes
data/local/overlay/calls/<key>_nocond.json: [{start, end, text, words: [{word, start, end, p}]}], times in
seconds of the recording. Names stay in data/local: <key> is a private crew key.
"""
import glob
import json
import os
import subprocess
import sys
import wave

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))


def to_wav(src, key):
    out = os.path.join(ROOT, "data", "local", "coxing", "experiment", "audio", key + ".wav")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ff = glob.glob(os.path.join(ROOT, ".venv-tools", "Lib", "site-packages", "imageio_ffmpeg", "binaries",
                                    "ffmpeg-*.exe"))[0]
        subprocess.run([ff, "-v", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                        out], check=True)
    return out


def main():
    src, key = sys.argv[1], sys.argv[2]
    wav = to_wav(src, key)
    import transcribe_calls as TC
    from faster_whisper import WhisperModel
    with wave.open(wav) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    wm = WhisperModel("small.en", device="cpu", compute_type="int8", local_files_only=True)
    segs, _ = wm.transcribe(audio, language="en", word_timestamps=True, vad_filter=True, initial_prompt=TC.PROMPT,
                            beam_size=5, condition_on_previous_text=False)
    out = [dict(start=s.start, end=s.end, text=s.text.strip(),
                words=[dict(word=x.word, start=x.start, end=x.end, p=x.probability) for x in (s.words or [])])
           for s in segs]
    p = os.path.join(ROOT, "data", "local", "overlay", "calls", key + "_nocond.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print("wrote", p, len(out), "segments")


if __name__ == "__main__":
    main()
