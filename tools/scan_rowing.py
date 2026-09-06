r"""Find the rowing in hours of recordings, and measure what it sounds like.

    python tools/scan_rowing.py "G:/My Drive/Cox Recordings/Practice"
    python tools/scan_rowing.py <file.mp3> --report out/audio

There are hundreds of hours here and most of them are not rowing: talking
on the water, being coached, launching and landing, sitting at the start.
Listening for the good bits by hand is not on, so this finds them by the
one property rowing has and speech does not -- **it is periodic**.

How a rowing stretch is recognised
----------------------------------
The catch is a transient, and a crew produces one every 1.5 to 3.4
seconds with a regularity nothing else in the recording has.  So the
amplitude envelope is autocorrelated over a window, and a strong peak at
a lag in that band means rowing at that rate.  Coaching has transients
too -- a voice is full of them -- but they do not repeat on a metronome,
and the voice band is notched out first anyway.

The score is the height of that peak relative to the envelope's own
energy, so it is comparable between quiet and loud recordings.

What it takes from the good stretches
-------------------------------------
For the highest-scoring windows it measures, per catch: the decay
constant of the transient, its spectral centroid, and the ratio of energy
in the first 30 ms to the tail.  Those are exactly the three numbers
:func:`coxswain.viz.strokeaudio.synthesise` currently guesses, so the
output is a recipe for replacing the guess.

Nothing here copies audio anywhere: it reports numbers, and the voice
band is removed before any of them are taken.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np

#: Notched out before anything is measured, Hz.
VOICE_BAND = (280.0, 3400.0)
#: Stroke periods to look for, seconds -- 18 to 40 strokes a minute.
PERIOD_BAND = (1.5, 3.4)
#: Envelope sample rate, Hz.  Fast enough to place a catch to 5 ms.
ENVELOPE_RATE = 200.0
#: Window over which periodicity is judged, seconds.
WINDOW = 30.0
#: A window scoring below this is not rowing.  Low, deliberately: a
#: recording is mostly not rowing and the cost of a false positive is one
#: measured window, while a false negative loses the only data there is.
ROWING_SCORE = 0.16

#: Strokes a minute is ``60 / period``.  Written out because the first
#: version used ``120 / period`` and reported every crew at twice its
#: rate -- and the unit test agreed, because it generated its expected
#: value with the same wrong formula.  A test that shares an error with
#: the code under test proves nothing.


def envelope_of(path: str, limit_minutes: float = None):
    """Amplitude envelope of a whole recording, at ``ENVELOPE_RATE``.

    Streamed in blocks: these files run to an hour and a half, which is
    four hundred million samples and not something to hold in memory.
    """
    import soundfile as sf

    info = sf.info(path)
    rate = info.samplerate
    step = max(int(rate / ENVELOPE_RATE), 1)
    want = None if limit_minutes is None else int(limit_minutes * 60 * rate)

    pieces, taken = [], 0
    for block in sf.blocks(path, blocksize=step * 2048, dtype="float32",
                           always_2d=True):
        data = block.mean(axis=1)
        if len(data) < step:
            break
        # Notch the voice out block by block, so a call is not counted as
        # a catch.  Block edges are unimportant: this feeds an envelope.
        spectrum = np.fft.rfft(data)
        freq = np.fft.rfftfreq(len(data), 1.0 / rate)
        spectrum[(freq > VOICE_BAND[0]) & (freq < VOICE_BAND[1])] = 0.0
        data = np.fft.irfft(spectrum, n=len(data))

        usable = len(data) // step * step
        pieces.append(np.abs(data[:usable]).reshape(-1, step).max(axis=1))
        taken += len(data)
        if want is not None and taken >= want:
            break
    if not pieces:
        return np.zeros(0), rate, info.duration
    return np.concatenate(pieces), rate, info.duration


def rowing_windows(envelope: np.ndarray) -> List[Tuple[float, float, float]]:
    """``(start_s, score, rate_spm)`` for every window that looks like rowing."""
    n = int(WINDOW * ENVELOPE_RATE)
    if len(envelope) < 2 * n:
        return []
    lo = int(PERIOD_BAND[0] * ENVELOPE_RATE)
    hi = int(PERIOD_BAND[1] * ENVELOPE_RATE)
    out = []
    for start in range(0, len(envelope) - n, n // 2):
        piece = envelope[start:start + n]
        piece = piece - piece.mean()
        power = float(np.dot(piece, piece))
        if power <= 1e-9:
            continue
        # Autocorrelation by FFT, normalised by the zero lag.
        spectrum = np.fft.rfft(piece, n=2 * n)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum))[:n] / power
        band = correlation[lo:hi]
        if not len(band):
            continue
        peak = int(np.argmax(band))
        score = float(band[peak])
        period = (lo + peak) / ENVELOPE_RATE
        out.append((start / ENVELOPE_RATE, score, 60.0 / period))
    return out


def measure(path: str, at: float, span: float = 20.0):
    """Catch statistics from ``span`` seconds starting at ``at``."""
    import soundfile as sf

    info = sf.info(path)
    rate = info.samplerate
    data, _ = sf.read(path, start=int(at * rate),
                      frames=int(span * rate), dtype="float32",
                      always_2d=True)
    signal = data.mean(axis=1)
    spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(len(signal), 1.0 / rate)
    spectrum[(freq > VOICE_BAND[0]) & (freq < VOICE_BAND[1])] = 0.0
    signal = np.fft.irfft(spectrum, n=len(signal))

    window = max(int(rate * 0.008), 1)
    env = np.convolve(np.abs(signal), np.ones(window) / window, mode="same")
    threshold = np.median(env) * 3.0
    loud = env > threshold
    edges = np.nonzero(loud[1:] & ~loud[:-1])[0] + 1

    catches, last = [], -1e9
    for index in edges:
        if index / rate - last >= PERIOD_BAND[0] * 0.45:
            catches.append(index)
            last = index / rate
    if len(catches) < 3:
        return None

    decays, centroids, punch = [], [], []
    for index in catches:
        end = min(index + int(0.30 * rate), len(signal))
        piece = env[index:end]
        if len(piece) < 32 or piece[0] <= 0:
            continue
        t = np.arange(len(piece)) / rate
        good = piece > piece.max() * 0.05
        if good.sum() >= 8:
            decays.append(-np.polyfit(t[good], np.log(piece[good] + 1e-12),
                                      1)[0])
        raw = signal[index:end]
        mag = np.abs(np.fft.rfft(raw * np.hanning(len(raw))))
        f = np.fft.rfftfreq(len(raw), 1.0 / rate)
        centroids.append(float((f * mag).sum() / (mag.sum() + 1e-12)))
        head = int(0.03 * rate)
        punch.append(float(np.abs(raw[:head]).sum()
                           / (np.abs(raw[head:]).sum() + 1e-9)))
    if not decays:
        return None
    return {"catches": len(catches),
            "decay": float(np.median(decays)),
            "centroid": float(np.median(centroids)),
            "punch": float(np.median(punch))}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--minutes", type=float, default=None,
                        help="only scan the first N minutes of each file")
    parser.add_argument("--top", type=int, default=3,
                        help="how many rowing windows to measure per file")
    args = parser.parse_args(argv)

    files = []
    for entry in args.paths:
        if os.path.isdir(entry):
            for pattern in ("*.mp3", "*.m4a", "*.wav", "*.flac", "*.ogg"):
                files += sorted(glob.glob(os.path.join(entry, pattern)))
        else:
            files += sorted(glob.glob(entry))
    if not files:
        print("no audio found")
        return 1

    summary = []
    for path in files:
        name = os.path.basename(path)
        try:
            env, _rate, duration = envelope_of(path, args.minutes)
        except Exception as error:
            print("%-46s unreadable (%s)" % (name[:46], str(error)[:38]))
            continue
        windows = rowing_windows(env)
        rowing = [w for w in windows if w[1] >= ROWING_SCORE]
        share = 100.0 * len(rowing) / max(len(windows), 1)
        print("%-46s %5.1f min  %4.0f%% rowing" % (name[:46],
                                                   duration / 60.0, share))
        for start, score, spm in sorted(rowing, key=lambda w: -w[1])[:args.top]:
            stats = measure(path, start)
            if stats is None:
                continue
            print("      %6.1f min  score %.2f  %4.1f spm  "
                  "decay %5.1f/s  centroid %5.0f Hz  punch %.2f"
                  % (start / 60.0, score, spm, stats["decay"],
                     stats["centroid"], stats["punch"]))
            summary.append(stats)

    if summary:
        print()
        print("ACROSS %d MEASURED STRETCHES -- feed these to strokeaudio:"
              % len(summary))
        for key, unit in (("decay", "1/s"), ("centroid", "Hz"),
                          ("punch", "")):
            values = np.array([s[key] for s in summary])
            print("   %-9s median %8.1f %-4s  (%.1f to %.1f)"
                  % (key, np.median(values), unit,
                     np.percentile(values, 10), np.percentile(values, 90)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
