r"""Build the shipped stroke envelopes by pooling many recordings.

    python tools/build_stroke_envelope.py --root "G:/My Drive/Cox Recordings"

:mod:`tools.dmd_stroke` measures one stretch of one recording.  That is
enough to show the method works and **not** enough to ship: a single
outing carries that day's water, that day's wind, that microphone
position and that crew's rate, and none of those are properties of a
coxed four.  This runs the same decomposition over every recording of a
given shell class and averages what comes back, so what gets shipped is
the part that is common to the boat rather than to the morning.

Averaging in decibels, and why
------------------------------
The envelopes are averaged as log magnitude, which is a geometric mean
of the spectra.  A recording made close to the rigger is louder in every
band than one made from the stern, and an arithmetic mean would let the
loudest tape dominate; in dB that difference is a constant offset, and
each envelope is normalised to its own peak before averaging so the
offset drops out entirely.  What survives is the **shape**.

Event phases are averaged too, but reported with their spread, because a
disagreement there is interesting: it means the crews are not putting
the same events at the same points of the cycle, and one envelope for
the class would be hiding that.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

import dmd_stroke as D
import scan_rowing as S

#: Which shell a recording is of, from its file name.
FOUR = re.compile(r"\b(4\+|4x|four)\b|4\+", re.I)
EIGHT = re.compile(r"\b(8\+|8x|eight)\b|\b8\b", re.I)
#: Seconds of a rowing stretch to decompose.
SPAN = 180.0


def shell_of_name(name: str):
    """``"four"``, ``"eight"`` or ``None`` from a file name."""
    if re.search(r"4\+|4x|\bV4\b|\bW4\b|\bMV4\b|\bMG4\b|\bMxH4\b|\bWB4\b",
                 name, re.I):
        return "four"
    if re.search(r"8\+|8x|\beight\b|\bMens 8\b|\b8\b", name, re.I):
        return "eight"
    return None


def best_window(path: str, minutes: float = None):
    """Start time, seconds, of the most rowing-like stretch, or ``None``."""
    try:
        envelope, _rate, _duration = S.envelope_of(path, minutes)
    except Exception:
        return None
    windows = S.rowing_windows(envelope)
    if not windows:
        return None
    start, score, _spm = max(windows, key=lambda w: w[1])
    return start if score >= S.ROWING_SCORE else None


def envelope_of_file(path: str, at: float):
    """``(bands, envelope, events, period)`` for one recording."""
    signal, rate = D.load(path, at, SPAN)
    mag, freq, times = D.spectrogram(signal, rate)
    marks, period = D.catches(signal, rate)
    if len(marks) < 12:
        return None
    cycles = D.phase_normalise(mag, times, marks)
    background, _moving, _values = D.dmd_background(cycles)
    scales = D.mrdmd(background)
    phases = np.linspace(0.0, 1.0, background.shape[1], endpoint=False)

    edges = np.geomspace(60.0, 16000.0, 25)
    bands, rows = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (freq >= low) & (freq < high)
        if inside.any():
            bands.append((low, high))
            rows.append(background[inside].mean(axis=0))
    envelope = np.array(rows)
    envelope = envelope / envelope.max()

    level = np.stack([s["detail"] for s in scales])[:-1].sum(axis=0)
    order = np.argsort(-level)
    picked = []
    for index in order:
        if all(abs(phases[index] - phases[j]) > 0.12 for j in picked):
            picked.append(index)
        if len(picked) == 3:
            break
    return (np.array(bands), envelope, phases[sorted(picked)], period)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--minutes", type=float, default=45.0,
                        help="how much of each file to search for rowing")
    parser.add_argument("--out", default="coxswain/data/stroke_envelope.npz")
    args = parser.parse_args(argv)

    files = []
    for folder in ("Practice", "Race", "."):
        files += sorted(glob.glob(os.path.join(args.root, folder, "*.mp3")))
    files = sorted(set(files))

    pooled = {"four": [], "eight": []}
    for path in files:
        name = os.path.basename(path)
        shell = shell_of_name(name)
        if shell is None:
            print("%-48s (shell unknown, skipped)" % name[:48])
            continue
        at = best_window(path, args.minutes)
        if at is None:
            print("%-48s %-6s no rowing found" % (name[:48], shell))
            continue
        try:
            got = envelope_of_file(path, at)
        except SystemExit:
            got = None
        except Exception as error:
            print("%-48s %-6s failed (%s)"
                  % (name[:48], shell, str(error)[:28]))
            continue
        if got is None:
            print("%-48s %-6s too few cycles" % (name[:48], shell))
            continue
        bands, envelope, events, period = got
        pooled[shell].append((bands, envelope, events, period))
        print("%-48s %-6s %5.1f min in, %.1f spm, events %s"
              % (name[:48], shell, at / 60.0, 60.0 / period,
                 " ".join("%.2f" % e for e in events)))

    blob = {}
    for shell, runs in pooled.items():
        if not runs:
            print("\nno usable %s recordings" % shell)
            continue
        bands = runs[0][0]
        # Geometric mean of the normalised envelopes: the shape, with the
        # per-recording level offset divided out.
        stack = np.stack([np.log10(np.maximum(e, 1e-8))
                          for _b, e, _v, _p in runs])
        envelope = 10.0 ** stack.mean(axis=0)
        envelope /= envelope.max()
        events = np.stack([v for _b, _e, v, _p in runs])
        spread = float(np.median(events.std(axis=0)))
        phases = np.linspace(0.0, 1.0, envelope.shape[1], endpoint=False)
        transient = np.zeros(envelope.shape[1])
        for _b, _e, v, _p in runs:
            for at in v:
                transient[int(np.argmin(np.abs(phases - at)))] += 1.0
        chosen = []
        for index in np.argsort(-transient):
            if all(abs(phases[index] - phases[j]) > 0.12 for j in chosen):
                chosen.append(index)
            if len(chosen) == 3:
                break
        chosen = sorted(chosen)
        bed_at = int(np.argmin(envelope.sum(axis=0)))
        print("\n%s: pooled %d recordings; event phases %s (spread %.2f), "
              "bed at %.2f" % (shell, len(runs),
                               " ".join("%.2f" % phases[i] for i in chosen),
                               spread, phases[bed_at]))
        blob["%s_bands" % shell] = bands
        blob["%s_envelope" % shell] = envelope.astype(np.float32)
        blob["%s_phase" % shell] = phases.astype(np.float32)
        blob["%s_events" % shell] = phases[chosen].astype(np.float32)
        blob["%s_bed" % shell] = envelope[:, bed_at].astype(np.float32)
        blob["%s_period" % shell] = float(np.median(
            [p for _b, _e, _v, p in runs]))
        blob["%s_runs" % shell] = np.array(len(runs))

    if blob:
        np.savez_compressed(args.out, **blob)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
