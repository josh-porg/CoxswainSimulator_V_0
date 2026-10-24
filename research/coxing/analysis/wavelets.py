"""Multi-resolution features of what the coxswain senses: motion and sound.

Two sensory streams, both decomposed by a discrete wavelet transform so that each
stroke is described by the energy in a set of octave bands rather than by a single
summary number.

MOTION (30 Hz, from the head IMU rotated into boat axes). Octave bands from the
stroke rhythm itself up to the highest frequency the sensor resolves:
  L6 0.23-0.47 Hz   below stroke rate      L3 1.9-3.8 Hz   check and run
  L5 0.47-0.94 Hz   stroke rate            L2 3.8-7.5 Hz   blade and slide
  L4 0.94-1.9 Hz    first harmonic         L1 7.5-15 Hz    vibration

SOUND (16 kHz). The coxswain's own voice dominates this recording, so using sound to
predict their calls is circular unless the voice is excluded. Two precautions: the
speech band (roughly 100-4000 Hz) is dropped entirely, keeping only sub-100 Hz hull
and water rumble and above-4 kHz splash and hiss; and every acoustic feature is taken
from a window that ENDS one stroke before the call it is used to predict.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np
import pywt
import scipy.io.wavfile as wav

R = np.load("race.npz")
gt = R["t"]
NS = len(gt)
EDGES = np.concatenate([[gt[0] - 2.0], (gt[:-1] + gt[1:]) / 2, [gt[-1] + 2.0]])


def band_energies(x, wavelet="db4", levels=6):
    """Band-energy SHAPE plus overall loudness: a hard catch differs from a soft one
    both in which frequencies it excites and in how much."""
    if len(x) < 2 ** levels:
        return np.full(levels + 2, np.nan)
    c = pywt.wavedec(x, wavelet, level=levels)
    e = np.array([float(np.sum(ci ** 2)) for ci in c])
    tot = e.sum()
    shape = e / tot if tot > 0 else np.full(len(e), np.nan)
    return np.append(shape, np.log10(tot + 1e-12))


def motion_features():
    from vestib import CH
    F, N = [], []
    for k, (tt, x) in CH.items():
        out = np.full((NS, 6), np.nan)
        for i in range(NS):
            m = (tt >= EDGES[i]) & (tt < EDGES[i + 1])
            if m.sum() >= 32:
                out[i] = band_energies(x[m], levels=4)
        F.append(out)
        N += ["%s L%d" % (k, j) for j in range(5)] + ["%s loud" % k]
    return np.column_stack(F), N


def sound_features(lag_strokes=1):
    sr, a = wav.read("_audio1.wav")
    a = a.astype(np.float32) / 32768.0
    # split into the two non-speech bands
    from scipy.signal import butter, sosfiltfilt
    lo = sosfiltfilt(butter(4, 100, "lowpass", fs=sr, output="sos"), a)
    hi = sosfiltfilt(butter(4, 4000, "highpass", fs=sr, output="sos"), a)
    out = np.full((NS, 16), np.nan)
    for i in range(NS):
        j = i - lag_strokes
        if j < 0:
            continue
        s0, s1 = int(EDGES[j] * sr), int(EDGES[j + 1] * sr)
        if s1 > len(a) or s1 - s0 < 512:
            continue
        out[i, :8] = band_energies(lo[s0:s1], levels=6)
        out[i, 8:] = band_energies(hi[s0:s1], levels=6)
    names = (["rumble L%d" % j for j in range(7)] + ["rumble loud"]
             + ["hiss L%d" % j for j in range(7)] + ["hiss loud"])
    return out, names


if __name__ == "__main__":
    M, MN = motion_features()
    np.save("_wav_motion.npy", M)
    print("motion wavelet features: %s, %d rows with any NaN"
          % (str(M.shape), int(np.isnan(M).any(1).sum())))
    S, SN = sound_features()
    np.save("_wav_sound.npy", S)
    print("sound wavelet features:  %s, %d rows with any NaN"
          % (str(S.shape), int(np.isnan(S).any(1).sum())))
    print()
    print("mean motion band energy (surge), L0..L6:",
          np.array2string(np.nanmean(M[:, :6], 0), precision=3))
    print("mean rumble band energy  L0..L6:",
          np.array2string(np.nanmean(S[:, :8], 0), precision=3))
    print("mean hiss band energy    L0..L6:",
          np.array2string(np.nanmean(S[:, 8:], 0), precision=3))
    import json
    json.dump({"motion": MN, "sound": SN}, open("_wav_names.json", "w"))
