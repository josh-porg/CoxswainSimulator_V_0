r"""The coxswain's calls, transcribed with word times, from a cox-camera video's audio.

    .venv-tools\Scripts\python tools\overlay\transcribe_calls.py "C:\...\DJI_..._D.MP4" [--model small.en]

Runs in the separate tools environment (``.venv-tools``: faster-whisper, no torch), never the
game's ``.venv``. Offline: the model is loaded from the local cache only (it was downloaded once,
2026-10-05); nothing is fetched at run time. Output (script use):
``data/local/overlay/calls/<video>.json``, segments with start, end, text and words (word, start,
end, probability). The calls are the crew's own and stay in the gitignored data/local.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "local", "overlay", "calls")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never reach for the internet
PROMPT = ("Coxswain calls in a rowing race: split two oh two, rate thirty, power ten, legs, "
          "catch, finish, sit up, length, ready all row, weigh enough, five hundred to go.")


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        game = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
        return subprocess.run([game, "-c", "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
                              capture_output=True, text=True, check=True).stdout.strip()


def bundled_model(model="small.en"):
    """The speech model shipped inside the app (a release build carries it), or None."""
    import sys
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "models", "faster-whisper-%s" % model)
    return path if os.path.exists(os.path.join(path, "model.bin")) else None


def model_available(model="small.en"):
    """Whether the model is shipped or in the local cache (no network is used to check)."""
    if bundled_model(model):
        try:
            import faster_whisper                          # noqa: F401
            return True
        except ImportError:
            return False
    try:
        from faster_whisper import WhisperModel          # noqa: F401
        from huggingface_hub import try_to_load_from_cache
        repo = "Systran/faster-whisper-%s" % model
        return all(isinstance(try_to_load_from_cache(repo, f), str)
                   for f in ("model.bin", "config.json", "tokenizer.json"))
    except Exception:
        return False


def transcribe(video, model="small.en", progress=None):
    """Segments (dicts) of the video's calls; ``progress(fraction)`` as audio is decoded."""
    import wave

    import numpy as np
    from faster_whisper import WhisperModel
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", video, "-vn", "-ac", "1",
                        "-ar", "16000", "-y", wav], check=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        # read the WAV here and hand over samples: faster-whisper's own decoder (PyAV) breaks on
        # the PyAV version pip resolves alongside it
        with wave.open(wav) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    total = len(audio) / 16000.0
    wm = WhisperModel(bundled_model(model) or model, device="cpu", compute_type="int8", local_files_only=True)
    segments, _info = wm.transcribe(audio, language="en", word_timestamps=True, vad_filter=True,
                                    initial_prompt=PROMPT, beam_size=5)
    out = []
    for s in segments:
        out.append(dict(start=s.start, end=s.end, text=s.text.strip(),
                        words=[dict(word=w.word, start=w.start, end=w.end, p=w.probability)
                               for w in (s.words or [])]))
        if progress is not None and total > 0:
            progress(min(s.end / total, 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="small.en")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = transcribe(a.video, a.model)
    path = os.path.join(OUT, os.path.splitext(os.path.basename(a.video))[0] + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", path, len(out), "segments")


if __name__ == "__main__":
    main()
