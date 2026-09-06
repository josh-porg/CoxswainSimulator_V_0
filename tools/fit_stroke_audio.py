r"""Fit the synthesised stroke to real recordings of a boat.

    python tools/fit_stroke_audio.py data/raw/audio/*.wav

:mod:`coxswain.viz.strokeaudio` invents its catch and release from noise
and a decaying tone, which is enough to tell drive from recovery and not
enough to sound like a four.  This measures the real thing and reports
the numbers that shape the synthesiser, so the guess can be replaced by a
fit.

What it measures
----------------
**Onsets** -- a catch is the loudest transient in the cycle, so the
spacing between onsets is the stroke period and the spacing between
alternate onsets tells catch from release.  That gives *rate* and *drive
fraction* straight from the audio, and both are already in the model, so
they are also a check on it.

**Decay** -- how fast each transient falls away, fitted as an
exponential on the envelope.  This is ``decay`` in ``_noise_knock``.

**Spectral centroid and rolloff** -- where the energy sits.  A catch is
broadband and low; a release is quieter and higher.  These set ``tone``
and the noise fraction.

Your voice is in these recordings
---------------------------------
The tool reports **statistics only** -- times, decay constants, band
energies -- and never copies audio into the repository.  Nothing it
writes contains anything you said.  Speech also sits mostly between 300
Hz and 3 kHz, so :data:`VOICE_BAND` is notched out before any
measurement is taken; if it were not, a call would be measured as part
of the stroke.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import wave

import numpy as np

#: Notched out before measuring, Hz: this is where a coxswain's call is.
VOICE_BAND = (280.0, 3400.0)
#: A transient must exceed this many times the running median to count.
ONSET_FACTOR = 3.2
#: Minimum gap between onsets, s -- half a fast drive.
MIN_GAP = 0.35


def read_wave(path):
    """``(samples in [-1, 1], rate)`` from a PCM wav, mixed to mono."""
    with wave.open(path, "rb") as handle:
        rate = handle.getframerate()
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise SystemExit("%s: unsupported sample width %d" % (path, width))
    data = np.frombuffer(raw, dtype=dtype).astype(float)
    if dtype is np.uint8:
        data = (data - 128.0) / 128.0
    else:
        data = data / float(np.iinfo(dtype).max)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def notch_voice(signal, rate):
    """Remove the speech band, so a call is not measured as a stroke."""
    spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(len(signal), 1.0 / rate)
    spectrum[(freq > VOICE_BAND[0]) & (freq < VOICE_BAND[1])] = 0.0
    return np.fft.irfft(spectrum, n=len(signal))


def envelope(signal, rate, window: float = 0.01):
    n = max(int(rate * window), 1)
    return np.convolve(np.abs(signal), np.ones(n) / n, mode="same")


def onsets(signal, rate):
    """Times of percussive transients, seconds."""
    env = envelope(signal, rate)
    # A long running median is the noise floor; a catch stands well over it.
    step = max(int(rate * 0.5), 1)
    floor = np.median(env[:len(env) // step * step].reshape(-1, step), axis=1)
    floor = np.repeat(np.maximum(floor, 1e-6), step)
    floor = np.resize(floor, len(env))
    loud = env > ONSET_FACTOR * floor
    edges = np.nonzero(loud[1:] & ~loud[:-1])[0] + 1
    kept = []
    for index in edges:
        when = index / rate
        if not kept or when - kept[-1] >= MIN_GAP:
            kept.append(when)
    return np.asarray(kept), env


def decay_of(env, rate, at, length: float = 0.25):
    """Exponential decay constant of the transient starting at ``at``."""
    a = int(at * rate)
    b = min(a + int(length * rate), len(env))
    piece = env[a:b]
    if len(piece) < 16 or piece[0] <= 0:
        return float("nan")
    t = np.arange(len(piece)) / rate
    good = piece > piece.max() * 0.05
    if good.sum() < 8:
        return float("nan")
    slope = np.polyfit(t[good], np.log(piece[good] + 1e-12), 1)[0]
    return float(-slope)


def spectrum_of(signal, rate, at, length: float = 0.12):
    a = int(at * rate)
    piece = signal[a:min(a + int(length * rate), len(signal))]
    if len(piece) < 64:
        return float("nan"), float("nan")
    mag = np.abs(np.fft.rfft(piece * np.hanning(len(piece))))
    freq = np.fft.rfftfreq(len(piece), 1.0 / rate)
    total = mag.sum() + 1e-12
    centroid = float((freq * mag).sum() / total)
    rolloff = float(freq[np.searchsorted(np.cumsum(mag), 0.85 * total)])
    return centroid, rolloff


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*",
                        help="wav files, or a directory of them")
    args = parser.parse_args(argv)

    paths = []
    for entry in args.files:
        if os.path.isdir(entry):
            paths += sorted(glob.glob(os.path.join(entry, "*.wav")))
        else:
            paths += sorted(glob.glob(entry))
    if not paths:
        print(__doc__.split("What it measures")[0])
        print("No wav files given.  Put recordings somewhere local -- for")
        print("example data/raw/audio/ -- and pass the folder.  Convert from")
        print("m4a/mp3 first if need be:")
        print("    ffmpeg -i clip.m4a -ac 1 -ar 44100 clip.wav")
        return 1

    print("%-26s %7s %8s %8s %9s %9s"
          % ("file", "onsets", "rate spm", "drive", "decay 1/s", "centroid"))
    for path in paths:
        try:
            signal, rate = read_wave(path)
        except Exception as error:
            print("%-26s  unreadable (%s)" % (os.path.basename(path)[:26],
                                              str(error)[:40]))
            continue
        signal = notch_voice(signal, rate)
        times, env = onsets(signal, rate)
        if len(times) < 4:
            print("%-26s  %5d  (too few transients to fit)"
                  % (os.path.basename(path)[:26], len(times)))
            continue
        gaps = np.diff(times)
        # Catch-to-catch is two gaps when the release is also detected.
        period = float(np.median(gaps) * 2.0)
        drive = float(np.median(gaps[::2]) / max(period, 1e-9))
        decays = [decay_of(env, rate, t) for t in times[:40]]
        centroids = [spectrum_of(signal, rate, t)[0] for t in times[:40]]
        print("%-26s %7d %8.1f %8.2f %9.1f %9.0f"
              % (os.path.basename(path)[:26], len(times), 120.0 / period,
                 drive, np.nanmedian(decays), np.nanmedian(centroids)))
    print()
    print("Feed these into coxswain/viz/strokeaudio.py: 'decay 1/s' is the")
    print("``decay`` argument to _noise_knock and 'centroid' the ``tone``.")
    print("Rate and drive are a cross-check on StrokeTiming, not inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
