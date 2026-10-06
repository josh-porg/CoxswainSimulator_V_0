r"""The coxswain's calls, transcribed with word times, from a cox-camera video's audio.

    .venv-tools\Scripts\python tools\overlay\transcribe_calls.py "C:\...\DJI_..._D.MP4" [--model small.en]

Runs in the separate tools environment (``.venv-tools``: faster-whisper, no torch), never the
game's ``.venv``. The audio is extracted with the ffmpeg already bundled in the project (imageio-
ffmpeg, located through the game's venv if not importable here). Output:
``data/local/overlay/calls/<video>.json``, a list of segments with start, end, text and words
(word, start, end, probability). The calls are the crew's own and stay in the gitignored data/local.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "local", "overlay", "calls")


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        game = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
        return subprocess.run([game, "-c", "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
                              capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="small.en")
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", a.video, "-vn", "-ac", "1",
                        "-ar", "16000", "-y", wav], check=True)
        # read the WAV here and hand over samples: faster-whisper's own decoder (PyAV) breaks on
        # the PyAV version pip resolves alongside it
        import wave
        import numpy as np
        with wave.open(wav) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        model = WhisperModel(a.model, device="cpu", compute_type="int8")
        # rowing words help the decoder; the prompt is vocabulary, not a script
        prompt = ("Coxswain calls in a rowing race: split two oh two, rate thirty, power ten, legs, "
                  "catch, finish, sit up, length, ready all row, weigh enough, five hundred to go.")
        segments, info = model.transcribe(audio, language="en", word_timestamps=True, vad_filter=True,
                                          initial_prompt=prompt, beam_size=5)
        out = []
        for s in segments:
            out.append(dict(start=s.start, end=s.end, text=s.text.strip(),
                            words=[dict(word=w.word, start=w.start, end=w.end, p=w.probability)
                                   for w in (s.words or [])]))
    path = os.path.join(OUT, os.path.splitext(os.path.basename(a.video))[0] + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", path, len(out), "segments")


if __name__ == "__main__":
    main()
