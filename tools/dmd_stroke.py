r"""Pull the stroke out from under the coxswain, by phase-locked DMD.

    python tools/dmd_stroke.py "G:/My Drive/Cox Recordings/Practice/x.mp3" --at 348

The problem this solves
-----------------------
``tools/scan_rowing.py`` measures the stroke with a notch through
280-3400 Hz, because that is where a coxswain's call sits and a call
measured as part of the catch is not a measurement of the catch.  But
that band is most of the audible spectrum, so what comes back is a
thump below 300 Hz and a splash above 3 kHz with a hole between them,
and the hole has to be interpolated.  The interpolation is a guess.

The separation that is actually available
-----------------------------------------
The stroke is **locked to the stroke cycle** and the voice is not.  A
catch happens at phase 0 of every cycle; a call happens when there is
something to say.  So if the recording is cut into cycles and each is
resampled onto a common phase grid, the stroke sound lands in the same
place in every column and the voice lands all over the place.

That is a separation into a persistent part and a transient part, which
is what **dynamic mode decomposition** is for.  Treating each
phase-normalised cycle as a snapshot :math:`c_k` and fitting
:math:`c_{k+1} \\approx A c_k`, the eigenvalues of :math:`A` sort the
structure by what it does from one cycle to the next:

* :math:`|\\lambda| \\approx 1` and :math:`\\arg\\lambda \\approx 0` --
  structure that repeats unchanged.  **That is the stroke.**
* everything else -- structure that grows, decays or oscillates between
  cycles.  That is the voice, the wind, a passing boat, a coach.

This is Grosek and Kutz's background/foreground split (2014) with the
roles the other way round: in video the static part is the background
and the interest is in the movers, here the static part is the thing
being measured and the movers are the contamination.

Why a spectrogram and not the waveform
--------------------------------------
Two catches do not have correlated waveforms -- water is noise, and the
sample-by-sample detail is different every time.  What repeats is the
**spectral envelope through the cycle**.  So the snapshots are magnitude
spectrograms, phase-normalised on the time axis, and the reconstruction
is a spectral envelope against phase rather than an audio clip.  That is
also all :mod:`coxswain.viz.strokeaudio` needs, since it synthesises by
shaping noise to an envelope.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

#: Frequency band that matters for the stroke, Hz.
BAND = (40.0, 16000.0)
#: Points on the phase grid a cycle is resampled onto.
PHASE_POINTS = 48
#: STFT window, samples, and its hop.
WINDOW = 2048
HOP = 256
#: Cycles to use.  More is better, up to the point where the crew changes
#: what it is doing.
MAX_CYCLES = 220


def load(path: str, at: float, span: float):
    import soundfile as sf

    info = sf.info(path)
    rate = info.samplerate
    data, _ = sf.read(path, start=int(at * rate), frames=int(span * rate),
                      dtype="float32", always_2d=True)
    return data.mean(axis=1), rate


def spectrogram(signal, rate):
    """``(magnitude, frequencies, frame times)`` -- plain STFT, no notch."""
    frames = 1 + (len(signal) - WINDOW) // HOP
    if frames < 4:
        raise SystemExit("too short")
    window = np.hanning(WINDOW)
    columns = np.lib.stride_tricks.sliding_window_view(
        signal, WINDOW)[::HOP][:frames]
    mag = np.abs(np.fft.rfft(columns * window, axis=1)).T
    freq = np.fft.rfftfreq(WINDOW, 1.0 / rate)
    times = (np.arange(frames) * HOP + WINDOW / 2.0) / rate
    keep = (freq >= BAND[0]) & (freq <= BAND[1])
    return mag[keep], freq[keep], times


def catches(signal, rate, period_band=(1.5, 3.4)):
    """Catch times, from the low band where the thump lives.

    Below 300 Hz there is no speech to speak of, so onsets can be found
    without touching the rest of the spectrum -- the notch is used to
    *locate* the cycle and never to measure it.
    """
    spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(len(signal), 1.0 / rate)
    spectrum[(freq < 60.0) | (freq > 300.0)] = 0.0
    low = np.fft.irfft(spectrum, n=len(signal))
    width = max(int(rate * 0.01), 1)
    env = np.convolve(np.abs(low), np.ones(width) / width, mode="same")

    # Period from the envelope's autocorrelation, then peaks near it.
    centred = env - env.mean()
    power = float(np.dot(centred, centred))
    spec = np.fft.rfft(centred, n=2 * len(centred))
    corr = np.fft.irfft(spec * np.conj(spec))[:len(centred)] / max(power, 1e-9)
    lo, hi = int(period_band[0] * rate), int(period_band[1] * rate)
    period = (lo + int(np.argmax(corr[lo:hi]))) / rate

    found, guard = [], int(0.55 * period * rate)
    index = int(np.argmax(env[:int(period * rate)]))
    while index < len(env) - guard:
        window = env[index:index + guard]
        found.append(index / rate)
        step = int(period * rate)
        nxt = index + step
        low_i, high_i = max(nxt - guard // 2, 0), min(nxt + guard // 2,
                                                      len(env) - 1)
        if high_i <= low_i:
            break
        index = low_i + int(np.argmax(env[low_i:high_i]))
    return np.asarray(found), period


def phase_normalise(mag, times, marks):
    """Stack of cycles, each resampled onto a common phase grid.

    Returns ``(cycles, bins, PHASE_POINTS)``.
    """
    grid = np.linspace(0.0, 1.0, PHASE_POINTS, endpoint=False)
    out = []
    for start, end in zip(marks[:-1], marks[1:]):
        if end - start <= 0.4:
            continue
        inside = (times >= start) & (times < end)
        if inside.sum() < 6:
            continue
        phase = (times[inside] - start) / (end - start)
        block = mag[:, inside]
        out.append(np.stack([np.interp(grid, phase, row) for row in block]))
        if len(out) >= MAX_CYCLES:
            break
    if not out:
        raise SystemExit("no usable cycles")
    return np.stack(out)


def dmd_background(cycles, rank: int = 12, tolerance: float = 0.06):
    """Split the stack into what persists and what does not.

    ``cycles`` is ``(m, bins, phases)``; each cycle is flattened into a
    snapshot.  Returns ``(background, foreground_share, eigenvalues)``
    with ``background`` shaped ``(bins, phases)``.
    """
    m = len(cycles)
    if m < 6:
        raise SystemExit("need more cycles")
    snapshots = cycles.reshape(m, -1).T           # features x snapshots
    # Work on the log magnitude: the spectrum spans six decades and a
    # linear fit is otherwise governed entirely by the loudest band.
    snapshots = np.log10(np.maximum(snapshots, 1e-8))
    # **Do not centre.**
    #
    # The first version subtracted the mean before fitting, which removes
    # exactly the thing the persistent mode is meant to find: the
    # stationary structure ends up in the mean, the fit sees only
    # cycle-to-cycle noise, every eigenvalue lands near zero and the
    # "0 persistent modes" that came back was the method reporting that
    # its target had been deleted before it started.  Grosek and Kutz fit
    # the raw snapshots and read the background off the mode with
    # |lambda| ~ 1; so does this.
    first, second = snapshots[:, :-1], snapshots[:, 1:]
    u, s, vh = np.linalg.svd(first, full_matrices=False)
    rank = int(min(rank, np.sum(s > s[0] * 1e-8)))
    u, s, vh = u[:, :rank], s[:rank], vh[:rank]
    reduced = u.T @ second @ vh.T.conj() @ np.diag(1.0 / s)
    values, vectors = np.linalg.eig(reduced)
    modes = second @ vh.T.conj() @ np.diag(1.0 / s) @ vectors

    amplitudes = np.linalg.pinv(modes) @ snapshots[:, 0]
    # Persistent: neither growing nor decaying, and not oscillating from
    # cycle to cycle.  That is the stroke.
    persistent = (np.abs(np.abs(values) - 1.0) < tolerance) & \
                 (np.abs(np.angle(values)) < tolerance * np.pi)
    if not persistent.any():
        persistent = np.abs(np.abs(values) - 1.0) == \
            np.abs(np.abs(values) - 1.0).min()

    stationary = (modes[:, persistent] @ amplitudes[persistent]).real
    background = stationary.reshape(cycles.shape[1:])
    moving = float(np.abs(amplitudes[~persistent]).sum()
                   / max(np.abs(amplitudes).sum(), 1e-12))
    return 10.0 ** background, moving, values


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--at", type=float, default=348.0,
                        help="seconds into the file to start")
    parser.add_argument("--span", type=float, default=240.0)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--out", default="out/audio")
    args = parser.parse_args(argv)

    signal, rate = load(args.path, args.at, args.span)
    mag, freq, times = spectrogram(signal, rate)
    marks, period = catches(signal, rate)
    print("%s\n  %.0f s from %.0f s, %d catches, period %.2f s (%.1f spm)"
          % (os.path.basename(args.path), args.span, args.at, len(marks),
             period, 60.0 / period))

    cycles = phase_normalise(mag, times, marks)
    print("  %d cycles phase-normalised onto %d points"
          % (len(cycles), PHASE_POINTS))

    background, moving, values = dmd_background(cycles, rank=args.rank)
    keep = np.abs(np.abs(values) - 1.0) < 0.06
    print("  DMD rank %d: %d persistent modes, %d transient; "
          "%.0f%% of the amplitude is transient (voice, wind, traffic)"
          % (len(values), int(keep.sum()), int((~keep).sum()), 100 * moving))

    # Compare with the median across cycles, which is the cheap version of
    # the same idea and a check that the DMD has not invented anything.
    median = np.median(cycles, axis=0)
    at_catch = background[:, 0]
    ref = median[:, 0]
    scale = np.median(at_catch / np.maximum(ref, 1e-9))
    error = np.abs(20 * np.log10(np.maximum(at_catch / scale, 1e-9)
                                 / np.maximum(ref, 1e-9)))
    print("  against a plain phase-locked median: %.1f dB median difference"
          % np.median(error))

    edges = np.geomspace(60.0, 16000.0, 25)
    print("\n  catch spectrum, phase-locked, NO voice notch:")
    profile = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (freq >= lo) & (freq < hi)
        if not band.any():
            continue
        level = float(at_catch[band].mean())
        profile.append((lo, hi, level))
    peak = max(v for _a, _b, v in profile)
    for lo, hi, level in profile:
        db = 20 * np.log10(max(level / peak, 1e-6))
        print("    %5.0f-%5.0f  %6.1f dB  %s"
              % (lo, hi, db, "#" * max(int(40 + db / 1.2), 0)))

    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "stroke_dmd.npz")
    np.savez(target, background=background, freq=freq,
             phase=np.linspace(0.0, 1.0, PHASE_POINTS, endpoint=False),
             period=period)
    print("\n  wrote %s (spectral envelope against stroke phase)" % target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
