"""The sound of the boat and the sound of the voice: features from the coxswain's recording.

    .venv/Scripts/python.exe research/coxing/acoustic/features.py

Plan fixed in the CODEBOOK addendum of 15 Nov 2026 ("the sound of the boat and the sound of the voice"). No
source-separation model: a frame is speech if it lies in a transcribed phrase, the band under 90 Hz is read everywhere,
and the background's spectrum by stroke phase is subtracted from speech frames before the voice is measured.

Writes data/local/coxing/acoustic/: strokes.csv (six background features per stroke, raw and standardised within the
piece), phrases.csv (ten voice features per phrase), qc.json (stroke lock of the low band, speech share, leakage check).
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, "..", "..", "callmodel"))
import call_properties as CP  # noqa: E402

LOCAL = os.path.abspath(os.path.join(CODE, "..", "..", "..", "data", "local", "coxing"))
SDIR = os.path.join(LOCAL, "sessions")
AUD = os.path.join(LOCAL, "experiment", "audio")
OUT = os.path.join(LOCAL, "acoustic")
TODAY = ["practice_%02d" % k for k in range(3, 9)]
AUDIO = {"practice_01": "0117", "practice_02": "0119", "race_04": "0132", "race_05": "0134", **{k: "261007" for k in TODAY},
         "race_06": "0149", "race_08": "0151", "race_09": "0153"}
SES = ["practice_01", "practice_02", "race_04", "race_05"] + TODAY + ["race_06", "race_08", "race_09", "race_07"]
HOP, FRAME = 160, 512                      # 10 ms hop, 32 ms window at 16 kHz
EDGES = [150, 220, 300, 400, 520, 680, 850, 1000, 1200, 1500, 1750, 2000, 2400, 2800, 3300, 4000, 4800, 5700, 6800, 7900]
NPH = 20                                    # stroke-phase bins of the background profile
BG = ["thump", "sharp", "splash", "spread", "water", "rumble"]
VOICE = ["level", "f0", "f0_range", "f0_slope", "rate", "duration", "tilt", "voiced", "ph_cos", "ph_sin"]


def rms_frames(x, win):
    """RMS of x in frames of ``win`` samples centred every HOP samples."""
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    centres = np.arange(0, len(x), HOP)
    a, b = np.clip(centres - win // 2, 0, len(x)), np.clip(centres + win // 2, 0, len(x))
    return np.sqrt((c[b] - c[a]) / np.maximum(b - a, 1))


def band_powers(x):
    """Power in the EDGES bands per 10 ms frame (Hann, 512 samples), in chunks."""
    nf = len(x) // HOP
    win = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1 / 16000)
    masks = [(freqs >= a) & (freqs < b) for a, b in zip(EDGES[:-1], EDGES[1:])]
    xp = np.concatenate([np.zeros(FRAME // 2), x, np.zeros(FRAME)])
    out = np.zeros((nf, len(masks)), np.float32)
    for s0 in range(0, nf, 20000):
        idx = np.arange(s0, min(nf, s0 + 20000))
        fr = np.lib.stride_tricks.sliding_window_view(xp, FRAME)[idx * HOP] * win
        P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        for q, m in enumerate(masks):
            out[idx, q] = P[:, m].sum(1)
    return out


def f0_track(x, sr, t0, t1):
    """Pitch (Hz) per 40 ms frame every 20 ms by autocorrelation (as call_properties.delivery); (times, f0, voiced)."""
    a, b = int(max(0, t0 * sr)), int(min(len(x), t1 * sr))
    seg = x[a:b].astype(float)
    fl = int(0.04 * sr)
    ts, f0s = [], []
    for k in range(0, len(seg) - fl, fl // 2):
        fr = seg[k:k + fl] - seg[k:k + fl].mean()
        if np.dot(fr, fr) <= 0:
            continue
        ac = np.correlate(fr, fr, "full")[fl - 1:]
        lo, hi = int(sr / 500), int(sr / 80)
        lag = lo + int(np.argmax(ac[lo:hi]))
        ts.append((a + k + fl / 2) / sr)
        f0s.append(sr / lag if ac[lag] / ac[0] > 0.5 else np.nan)
    return np.array(ts), np.array(f0s)


def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / (np.nanstd(v) or 1.0)


def one(sid, wavs):
    s = CP.load_session(os.path.join(SDIR, sid))
    key = AUDIO.get(sid) or s["meta"]["source"].split()[-1]
    if key not in wavs:
        sr, w = wavfile.read(os.path.join(AUD, key + ".wav"))
        assert sr == 16000
        wavs[key] = w.astype(np.float32) / 32768.0
    sr, off = 16000, s["meta"]["sync_offset_s"]
    ta = s["t"] + off                                              # logged strokes on the audio clock
    a0, a1 = max(0, int((ta[0] - 6) * sr)), min(len(wavs[key]), int((ta[-1] + 6) * sr))
    x = wavs[key][a0:a1]
    tf = a0 / sr + np.arange(len(x) // HOP) * HOP / sr              # frame times (audio clock)
    nf = len(tf)
    low = signal.sosfiltfilt(signal.butter(4, [20, 90], "bandpass", fs=sr, output="sos"), x)
    high = signal.sosfiltfilt(signal.butter(4, 2000, "highpass", fs=sr, output="sos"), x)
    L = rms_frames(low, 640)[:nf]                                   # 40 ms
    H = rms_frames(high, 160)[:nf] ** 2                             # 10 ms, power
    P = band_powers(x)[:nf]
    Ldb = 20 * np.log10(L + 1e-7)
    mid = np.array([150 <= e < 1500 for e in EDGES[:-1]])
    lov, hiv, vb = (np.array([a <= e < b for e in EDGES[:-1]]) for a, b in ((150, 1000), (1000, 4000), (150, 4000)))
    # speech frames: inside any transcribed phrase
    speech = np.zeros(nf, bool)
    for p in s["phrases"]:
        te = p.get("t_end") or (p["t"] + 0.4)
        speech[(tf >= p["t"] + off - 0.10) & (tf <= max(te, p["t"]) + off + 0.15)] = True
    # stroke index and phase of every frame
    k = np.searchsorted(ta, tf, side="left")                       # frame lies in the cycle ending at stroke k
    ok = (k >= 1) & (k < len(ta))
    kk = np.clip(k, 1, len(ta) - 1)
    phase = np.where(ok, (tf - ta[kk - 1]) / (ta[kk] - ta[kk - 1]), np.nan)
    pb = np.clip((phase * NPH).astype(int, copy=False), 0, NPH - 1)
    # background spectrum by stroke phase (background frames only)
    N = np.zeros((NPH, P.shape[1]))
    for b in range(NPH):
        m = ok & ~speech & (pb == b)
        N[b] = P[m].mean(0) if m.sum() > 5 else P[ok & ~speech].mean(0)
    clean = np.maximum(P - N[pb], 0.1 * P)                           # spectral subtraction, floored
    # ---- per-stroke background features
    n = len(ta)
    F = {f: np.full(n, np.nan) for f in BG}
    peak_phase, sp_frac = np.full(n, np.nan), np.full(n, np.nan)
    first = np.searchsorted(tf, ta)                                 # frame index at each stroke time
    for j in range(1, n):
        i0, i1 = first[j - 1], first[j]
        if i1 - i0 < 8:
            continue
        seg = Ldb[i0:i1]
        ip = int(np.argmax(seg))
        F["thump"][j] = seg.max() - seg.min()
        F["sharp"][j] = L[i0:i1].max() / (L[i0:i1].mean() + 1e-9)
        F["rumble"][j] = np.percentile(seg, 10)
        peak_phase[j] = ip / (i1 - i0)
        sp_frac[j] = speech[i0:i1].mean()
        bgm = ~speech[i0:i1]
        if bgm.mean() >= 0.3:
            F["water"][j] = 10 * np.log10(np.median(P[i0:i1][bgm][:, mid].sum(1)) + 1e-12)
        c = i0 + ip
        w0, w1 = max(0, c - 12), min(nf, c + 13)
        if (~speech[w0:w1]).mean() >= 0.5:
            F["splash"][j] = 10 * np.log10(H[w0:w1][~speech[w0:w1]].mean() + 1e-12)
        w0, w1 = max(0, c - 20), min(nf, c + 21)
        if (~speech[w0:w1]).mean() >= 0.7:
            wgt = np.where(~speech[w0:w1], H[w0:w1], 0.0)
            wgt = np.maximum(wgt - wgt[~speech[w0:w1]].min(), 0)
            if wgt.sum() > 0:
                tt = tf[w0:w1]
                mu = np.sum(wgt * tt) / wgt.sum()
                F["spread"][j] = 1000 * np.sqrt(np.sum(wgt * (tt - mu) ** 2) / wgt.sum())
    # stroke lock of the low-band peak (Rayleigh) and leakage check
    ph = peak_phase[~np.isnan(peak_phase)]
    R = float(np.abs(np.mean(np.exp(2j * np.pi * ph))))
    ray_p = float(np.exp(-len(ph) * R ** 2))
    talk, quiet = sp_frac > 0.7, sp_frac < 0.3
    qc = dict(strokes=n, low_peak_R=R, low_peak_rayleigh_p=ray_p, low_peak_mean_phase=float((np.angle(np.mean(np.exp(2j * np.pi * ph))) / (2 * np.pi)) % 1),
              locked=bool(ray_p < 0.05), speech_share=float(speech[ok].mean()),
              missing={f: float(np.isnan(F[f][1:]).mean()) for f in BG},
              thump_when_talking=float(np.nanmean(F["thump"][talk])) if talk.sum() > 5 else None,
              thump_when_quiet=float(np.nanmean(F["thump"][quiet])) if quiet.sum() > 5 else None)
    strokes = [dict(session=sid, k=j, t=float(s["t"][j]), speech=float(sp_frac[j]) if not np.isnan(sp_frac[j]) else "",
                    **{f: F[f][j] for f in BG}, **{f + "_z": z[j] for f, z in ((f, zs(F[f])) for f in BG)}) for j in range(n)]
    # ---- per-phrase voice features
    rows = []
    for p in s["phrases"]:
        te = max(p["t"], p.get("t_end") or p["t"])
        t0, t1 = p["t"] + off, te + off + 0.2
        m = (tf >= t0) & (tf <= max(t1, t0 + 0.4))
        r = dict(session=sid, i=p["i"], t=p["t"])
        if m.sum() >= 3:
            v, lo_, hi_ = clean[m][:, vb].sum(), clean[m][:, lov].sum(), clean[m][:, hiv].sum()
            r["level"] = 10 * np.log10(v / m.sum() + 1e-12)
            r["tilt"] = 10 * np.log10((hi_ + 1e-12) / (lo_ + 1e-12))
            ts_, f0 = f0_track(wavs[key], sr, t0, max(t1, t0 + 0.4))
            vo = ~np.isnan(f0)
            r["voiced"] = float(vo.mean()) if len(f0) else np.nan
            if vo.sum() >= 1:
                r["f0"] = 12 * np.log2(np.median(f0[vo]) / 100.0)
            if vo.sum() >= 4:
                st = 12 * np.log2(f0[vo] / 100.0)
                r["f0_range"] = float(np.percentile(st, 90) - np.percentile(st, 10))
                r["f0_slope"] = float(np.polyfit(ts_[vo] - ts_[vo][0], st, 1)[0])
            r["rate"] = len(p["text"].split()) / (te - p["t"] + 0.3)
            r["duration"] = te - p["t"]
            j = int(np.searchsorted(ta, t0))
            if 1 <= j < n:
                phs = (t0 - ta[j - 1]) / (ta[j] - ta[j - 1])
                r["ph_cos"], r["ph_sin"] = float(np.cos(2 * np.pi * phs)), float(np.sin(2 * np.pi * phs))
        rows.append(r)
    for f in VOICE:
        col = zs([r.get(f, np.nan) for r in rows]) if f not in ("ph_cos", "ph_sin") else np.array([r.get(f, np.nan) for r in rows])
        for r, zval in zip(rows, col):
            r[f + "_z"] = zval
    return strokes, rows, qc


def main():
    os.makedirs(OUT, exist_ok=True)
    wavs, S, Ph, Q = {}, [], [], {}
    for sid in SES:
        st, ph, qc = one(sid, wavs)
        S += st
        Ph += ph
        Q[sid] = qc
        print("%-12s strokes %4d  low-band lock R %.2f (p %.1e, phase %.2f) %s  speech %.0f%%  missing splash %.0f%% spread %.0f%% water %.0f%%  thump talking/quiet %s/%s dB" % (
            sid, qc["strokes"], qc["low_peak_R"], qc["low_peak_rayleigh_p"], qc["low_peak_mean_phase"], "LOCKED" if qc["locked"] else "not locked",
            100 * qc["speech_share"], 100 * qc["missing"]["splash"], 100 * qc["missing"]["spread"], 100 * qc["missing"]["water"],
            "%.1f" % qc["thump_when_talking"] if qc["thump_when_talking"] is not None else "-",
            "%.1f" % qc["thump_when_quiet"] if qc["thump_when_quiet"] is not None else "-"))
    for name, rows, cols in (("strokes.csv", S, ["session", "k", "t", "speech"] + BG + [f + "_z" for f in BG]),
                             ("phrases.csv", Ph, ["session", "i", "t"] + VOICE + [f + "_z" for f in VOICE])):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()})
    json.dump(Q, open(os.path.join(OUT, "qc.json"), "w"), indent=1)
    print("written", OUT)


if __name__ == "__main__":
    main()
