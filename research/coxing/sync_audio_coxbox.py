"""Line an audio-only coxswain recording up with the CoxBox per-stroke log.

    .venv-tools/Scripts/python.exe research/coxing/sync_audio_coxbox.py <key> <coxbox csv> [--guess 0] [--search 120]

The overlay's video sync (tools/overlay/coxbox_data.best_sync) matches the stroke rhythm seen in the picture
with the CoxBox stroke rate, then refines on the CoxBox stroke times; with a transcript it also scores the
coxswain's called splits against the CoxBox split. An audio recording has a stroke rhythm too: the catches,
the slide and the coxswain's own stroke-locked calls. The same machinery runs on the audio's loudness
envelope (50 Hz frames): broadband, and above 1 kHz (oarlock and blade transients, consonants).

Inputs: data/local/coxing/experiment/audio/<key>.wav (transcribe_audio.py writes it) and, if present,
data/local/overlay/calls/<key>_nocond.json for the called splits. Output: data/local/coxing/sync/<key>.json with
the offset (recording seconds at the CoxBox's elapsed zero) and the diagnostics.
"""
import argparse
import json
import os
import sys
import wave

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
import coxbox_data as C  # noqa: E402

FRAME = 0.02


def envelopes(wav):
    with wave.open(wav) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    n = int(FRAME * sr)
    m = len(x) // n
    broad = np.sqrt(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1))
    hp = np.diff(x, prepend=x[0])                      # first difference: a gentle high-pass (+6 dB/octave)
    high = np.sqrt(np.mean(hp[:m * n].reshape(m, n) ** 2, axis=1))
    t = (np.arange(m) + 0.5) * FRAME
    # log loudness, so the coxswain's voice does not swamp the stroke pulses
    return t, {"audio_high": np.log(high + 1e-6), "audio_broad": np.log(broad + 1e-6)}, len(x) / sr


BANDS = ((20, 150), (150, 500), (500, 1500), (1500, 4000), (4000, 7900))


def band_envelopes(wav, chunk=2.0):
    """Log RMS envelope (50 Hz frames) in each of BANDS, by FFT filtering in 2 s chunks (no scipy)."""
    with wave.open(wav) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    n = int(FRAME * sr)
    c = int(chunk * sr) // n * n
    f = np.fft.rfftfreq(c, 1.0 / sr)
    out = {b: [] for b in BANDS}
    for a in range(0, len(x) - c + 1, c):
        X = np.fft.rfft(x[a:a + c] * 1.0)
        for b in BANDS:
            y = np.fft.irfft(np.where((f >= b[0]) & (f < b[1]), X, 0), c)
            out[b].append(np.sqrt(np.mean(y.reshape(-1, n) ** 2, axis=1)))
    t = (np.arange(sum(len(v) for v in out[BANDS[0]])) + 0.5) * FRAME
    return t, {"%d-%d Hz" % b: np.log(np.concatenate(v) + 1e-7) for b, v in out.items()}


def stroke_lock(t, env, stroke_t, offsets):
    """Mean z-scored, band-passed envelope at each CoxBox stroke time, per offset: a lock on the stroke itself.
    Returns the score per offset and the stroke-phase profile at the best one."""
    fps = 1.0 / FRAME
    sm = lambda v, s: np.convolve(v, np.ones(max(int(s * fps), 1)) / max(int(s * fps), 1), mode="same")
    band = sm(env, 0.1) - sm(env, 2.5)
    band = (band - band.mean()) / (band.std() + 1e-12)
    scores = []
    for off in offsets:
        st = stroke_t + off
        st = st[(st > t[0]) & (st < t[-1])]
        scores.append(float(np.mean(np.interp(st, t, band))) if len(st) > 50 else np.nan)
    return np.array(scores), band


def lock_sync(wav, s, search=60.0, step=0.02):
    """The sync that works on audio (2026-10-30, the M VET 4+ recording): the low band (20-150 Hz) envelope,
    band-passed around the stroke, averaged at every CoxBox stroke time. The whole race adds up a per-stroke
    signal too weak for the windowed-rate method (window confidence 0.14). Neighbouring strokes are the
    ambiguity, so the margin over the best other whole-stroke shift and the agreement of the two halves of the
    race are reported."""
    t, envs = band_envelopes(wav)
    offs = np.arange(-search, search, step)
    res = {}
    for name, e in envs.items():
        sc, _ = stroke_lock(t, e, s.t, offs)
        k = int(np.nanargmax(sc))
        far = np.abs(offs - offs[k]) > 0.6
        k2 = int(np.nanargmax(np.where(far, sc, -9)))
        halves = []
        for sel in (s.t < np.median(s.t), s.t >= np.median(s.t)):
            sh, _ = stroke_lock(t, e, s.t[sel], offs)
            halves.append(float(offs[int(np.nanargmax(sh))]))
        res[name] = dict(offset=float(offs[k]), score=float(sc[k]), next_best=float(offs[k2]),
                         next_score=float(sc[k2]), halves=halves)
    best = max(res, key=lambda n: res[n]["score"])
    return res[best]["offset"], best, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("coxbox")
    ap.add_argument("--guess", type=float, default=0.0)
    ap.add_argument("--search", type=float, default=120.0)
    ap.add_argument("--method", choices=("lock", "rate"), default="lock")
    a = ap.parse_args()
    wav = os.path.join(ROOT, "data", "local", "coxing", "experiment", "audio", a.key + ".wav")
    s = C.load_csv(a.coxbox)
    if a.method == "lock":
        off, band, res = lock_sync(wav, s, search=min(a.search, 60.0))
        calls_path = os.path.join(ROOT, "data", "local", "overlay", "calls", a.key + "_nocond.json")
        segs = json.load(open(calls_path, encoding="utf-8")) if os.path.exists(calls_path) else []
        calls = C.called_splits(segs) if segs else []
        out = dict(key=a.key, coxbox=os.path.basename(a.coxbox), offset=off, method="stroke lock, " + band, bands=res,
                   coxbox_seconds=float(s.t[-1]), called_splits=len(calls),
                   calls_check=C.calls_score(s, calls, off) if len(calls) >= 5 else None,
                   note="offset = recording seconds at the CoxBox elapsed zero (the start line)")
        p = os.path.join(ROOT, "data", "local", "coxing", "sync", a.key + ".json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(out, open(p, "w"), indent=1, default=float)
        print(json.dumps(out, indent=1, default=float))
        return
    mt, sigs, dur = envelopes(wav)
    calls_path = os.path.join(ROOT, "data", "local", "overlay", "calls", a.key + "_nocond.json")
    segs = json.load(open(calls_path, encoding="utf-8")) if os.path.exists(calls_path) else []
    calls = C.called_splits(segs) if segs else None
    report = {}
    for name in sigs:
        c, r, q = C.windowed_rate(mt, sigs[name])
        report[name] = dict(median_window_confidence=float(np.median(q)))
    off, info = C.best_sync(s, mt, sigs, a.guess, calls=calls, search=a.search)
    out = dict(key=a.key, coxbox=os.path.basename(a.coxbox), audio_seconds=dur, coxbox_seconds=float(s.t[-1]),
               offset=off, called_splits=len(calls or []), signals=report,
               choice={k: v for k, v in info.items() if k != "candidates"},
               candidates=info.get("candidates"))
    if off is not None and calls:
        out["calls_check"] = C.calls_score(s, calls, off)
    p = os.path.join(ROOT, "data", "local", "coxing", "sync", a.key + ".json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("offset", "called_splits", "signals", "choice")}, indent=1, default=float))
    if "calls_check" in out:
        print("called splits vs CoxBox at this offset: median |diff| %.1f s, reading lag %.1f s" % tuple(out["calls_check"]))


if __name__ == "__main__":
    main()
