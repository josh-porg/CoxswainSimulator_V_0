"""Does the coxswain's rhythm speech follow the boat? Measured from a synced recording.

    .venv-tools/Scripts/python.exe research/coxing/rhythm_calls.py <key> <coxbox csv>

Practice (user, 2026-10-29): a coxswain's rhythm speech usually *follows* the boat; the crew are told to listen to
the sound of the strokes and take the rhythm from it; long, drawn-out calls are used now and then to nudge the boat.
Section 194 modelled the coxswain as a pacer that could lead. This script tests the practice on a recording synced
to the CoxBox (sync_audio_coxbox.py), with three measures:

1. **Phase locking.** Each word's onset is placed in the stroke cycle (0 = a CoxBox stroke record, 1 = the next).
   Rhythm words (counts, "catch", "send", "cha", "press", "swing", ...) that follow the boat land at a consistent
   phase: high resultant length R (Rayleigh test). The boat's own sound is placed the same way: the stroke-triggered
   average of the 20-150 Hz envelope gives the phase of the strongest beat ("thump" below), so a call's lead or lag
   on the sound is measured. Which beat is the catch is NOT assumed: the counts are said on the catch (user,
   2026-10-30), so they mark it. On the first recording the counts sit at phase 0.55, just after a weaker beat at ~0.45
   (the catch); the strongest beat, at the CoxBox's per-stroke record, is near the finish (SOURCES §195, correction).
2. **Tempo following.** In a count ("one, two, three ..."), the interval between successive numbers against the
   CoxBox stroke period at that moment: a follower's ratio is ~1 and tracks the period as the rate changes.
3. **Nudges.** Words held long (word duration >= 0.8 s, e.g. a stretched "swing" or "length") and the change in
   stroke rate and period over the next 2-6 strokes against the 4 strokes before, compared with every other
   stroke of the race as the baseline.

Output: data/local/coxing/rhythm/<key>.json and a printed summary. Word times come from Whisper and are good to a
few tens of milliseconds for clear words; a stretched word's end is less certain than its start.
"""
import json
import os
import re
import sys
import wave

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "overlay"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coxbox_data as C  # noqa: E402
import sync_audio_coxbox as SA  # noqa: E402

COUNTS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10}
RHYTHM = {"catch", "send", "cha", "press", "swing", "drive", "now", "go", "legs", "squeeze", "lock", "sit", "breathe",
          "push", "power", "hit", "up", "and", "ready"}


def phase_in_cycle(t, st):
    k = np.searchsorted(st, t, side="right") - 1
    ok = (k >= 0) & (k < len(st) - 1)
    ph = np.full(len(t), np.nan)
    ph[ok] = (t[ok] - st[k[ok]]) / (st[k[ok] + 1] - st[k[ok]])
    return ph, k


def circ(ph):
    ph = ph[~np.isnan(ph)]
    if len(ph) == 0:
        return dict(n=0)
    z = np.mean(np.exp(2j * np.pi * ph))
    R, n = abs(z), len(ph)
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n * n - (R * n) ** 2)) - (1 + 2 * n)))   # Rayleigh (Zar's approximation)
    return dict(n=int(n), mean_phase=float((np.angle(z) / (2 * np.pi)) % 1.0), R=float(R), rayleigh_p=min(1.0, p))


def envelope_band(wav, lo, hi, chunk=2.0):
    """Linear RMS envelope (50 Hz frames) of one frequency band, FFT-filtered in 2 s chunks."""
    with wave.open(wav) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    n = int(SA.FRAME * sr)
    c = int(chunk * sr) // n * n
    f = np.fft.rfftfreq(c, 1.0 / sr)
    out = []
    for a in range(0, len(x) - c + 1, c):
        y = np.fft.irfft(np.where((f >= lo) & (f < hi), np.fft.rfft(x[a:a + c]), 0), c)
        out.append(np.sqrt(np.mean(y.reshape(-1, n) ** 2, axis=1)))
    return np.concatenate(out)


def main():
    key, cb = sys.argv[1], sys.argv[2]
    s = C.load_csv(cb)
    sync = json.load(open(os.path.join(ROOT, "data", "local", "coxing", "sync", key + ".json")))
    off = float(sync["offset"])
    segs = json.load(open(os.path.join(ROOT, "data", "local", "overlay", "calls", key + "_nocond.json"), encoding="utf-8"))
    words = [(w["start"] - off, w["end"] - off, re.sub(r"[^a-z0-9]", "", w["word"].lower()), w.get("p", 1.0))
             for g in segs for w in g.get("words", [])]
    words = [w for w in words if w[2] and s.t[0] <= w[0] <= s.t[-1]]
    st = np.asarray(s.t, float)
    on = np.array([w[0] for w in words])
    ph, k = phase_in_cycle(on, st)

    # the boat's own sound: stroke-triggered average of the low-band envelope
    wav = os.path.join(ROOT, "data", "local", "coxing", "experiment", "audio", key + ".wav")
    t_env, envs = SA.band_envelopes(wav)
    _, band = SA.stroke_lock(t_env, envs["20-150 Hz"], st, np.array([off]))
    bins = np.linspace(0, 1, 41)
    prof = np.zeros(40)
    cnt = np.zeros(40)
    for a, b in zip(st[:-1], st[1:]):
        tt = np.linspace(a, b, 200, endpoint=False)
        v = np.interp(tt + off, t_env, band)
        idx = np.minimum((np.linspace(0, 1, 200, endpoint=False) * 40).astype(int), 39)
        np.add.at(prof, idx, v)
        np.add.at(cnt, idx, 1)
    prof /= np.maximum(cnt, 1)
    thump = float((np.argmax(prof) + 0.5) / 40)

    cats = {"counts": np.array([w[2] in COUNTS for w in words]),
            "rhythm words": np.array([w[2] in RHYTHM for w in words]),
            "all other words": np.array([(w[2] not in COUNTS) and (w[2] not in RHYTHM) for w in words])}
    res = dict(key=key, offset=off, words=len(words), strokes=len(st), thump_phase=thump,
               thump_profile=[float(x) for x in prof])
    print("%d words inside the race, %d strokes; the boat's low-band thump peaks at stroke phase %.2f" % (len(words), len(st), thump))
    res["locking"] = {}
    for name, m in cats.items():
        c = circ(ph[m])
        res["locking"][name] = c
        if c["n"]:
            lead = ((c["mean_phase"] - thump + 0.5) % 1.0) - 0.5
            c["relative_to_thump"] = float(lead)
            print("  %-16s n %4d  mean phase %.2f  R %.2f  Rayleigh p %.1e  (%+.2f of a stroke from the thump)"
                  % (name, c["n"], c["mean_phase"], c["R"], c["rayleigh_p"], lead))

    # tempo following in counts: consecutive numbers n, n+1
    ratios = []
    for (a, _, wa, _), (b, _, wb, _) in zip(words[:-1], words[1:]):
        if wa in COUNTS and wb in COUNTS and COUNTS[wb] == COUNTS[wa] + 1 and 0.8 < b - a < 4.0:
            i = int(np.searchsorted(st, a)) - 1
            if 0 <= i < len(st) - 2:
                period = float(np.median(np.diff(st[max(0, i - 1):i + 3])))
                ratios.append((b - a, period))
    if ratios:
        r = np.array(ratios)
        q = r[:, 0] / r[:, 1]
        cc = float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]) if len(r) > 3 else float("nan")
        res["count_tempo"] = dict(n=len(r), median_ratio=float(np.median(q)), iqr=[float(np.percentile(q, 25)), float(np.percentile(q, 75))],
                                  corr_with_period=cc)
        print("  counting: %d consecutive pairs; interval / stroke period median %.2f (IQR %.2f-%.2f); correlation with the period %.2f"
              % (len(r), np.median(q), np.percentile(q, 25), np.percentile(q, 75), cc))

    # nudges: long-held words
    period = np.diff(st)
    def change(i):
        if i < 4 or i + 6 >= len(period):
            return None
        before = np.mean(period[i - 4:i])
        after = np.mean(period[i + 2:i + 6])
        return 60.0 / after - 60.0 / before          # rate change, strokes per minute
    long_ = [(w[0], w[1] - w[0], w[2]) for w in words if w[1] - w[0] >= 0.8]
    nud = []
    for t0, dur, wd in long_:
        i = int(np.searchsorted(st, t0)) - 1
        dc = change(i)
        if dc is not None:
            nud.append(dict(t=float(t0), word=wd, duration=float(dur), rate_change=float(dc)))
    base = np.array([c for c in (change(i) for i in range(len(period))) if c is not None])
    res["nudges"] = nud
    res["baseline_rate_change"] = dict(mean=float(base.mean()), sd=float(base.std()))
    if nud:
        dc = np.array([x["rate_change"] for x in nud])
        print("  long-held words (>= 0.8 s): %d; rate change over the next 2-6 strokes %+.2f spm (baseline %+.2f, sd %.2f)"
              % (len(nud), dc.mean(), base.mean(), base.std()))
        for x in nud:
            print("     %6.1f s  %-10s %.2f s held  rate %+.2f spm" % (x["t"], x["word"], x["duration"], x["rate_change"]))
    # --- drawn-out calls, from the sound: unbroken voicing (speech band, 300-3000 Hz) of 0.8 s or more. Whisper's
    # word ends run on through pauses, so its durations cannot find a held vowel; the envelope can. Ordinary speech
    # dips between syllables every ~0.2 s.
    sp = envelope_band(wav, 300, 3000)
    voiced = sp > np.percentile(sp, 60)
    runs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            if (j - i) * SA.FRAME >= 0.8:
                runs.append((i * SA.FRAME - off, (j - i) * SA.FRAME))
            i = j
        else:
            i += 1
    held = []
    for t0, dur in runs:
        if s.t[0] < t0 < s.t[-1]:
            ii = int(np.searchsorted(st, t0)) - 1
            dc = change(ii)
            if dc is not None:
                w = [x[2] for x in words if x[0] <= t0 + dur and x[1] >= t0]
                held.append(dict(t=float(t0), duration=float(dur), rate_change=float(dc), words=" ".join(w[:4])))
    res["held_voicing"] = held
    if held:
        dc = np.array([x["rate_change"] for x in held])
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(base, len(dc)).mean() for _ in range(5000)])
        res["held_summary"] = dict(n=len(held), mean_rate_change=float(dc.mean()),
                                   p_two_sided=float(np.mean(np.abs(boot - base.mean()) >= abs(dc.mean() - base.mean()))))
        print("  held voicing >= 0.8 s (acoustic): %d; next 2-6 strokes %+.2f spm vs baseline %+.2f (bootstrap p %.2f)"
              % (len(held), dc.mean(), base.mean(), res["held_summary"]["p_two_sided"]))
        for x in held:
            print("     %6.1f s  %.2f s held  rate %+.2f spm   [%s]" % (x["t"], x["duration"], x["rate_change"], x["words"]))

    # --- who follows whom: stroke by stroke, the asynchrony of a rhythm word with the boat's thump (a_k) and the
    # stroke period (T_k). A follower's asynchrony is set by the boat: a longer stroke (T_k up) makes the next call
    # late relative to a prediction from the previous stroke, so a_{k+1} correlates with T_k - T_{k-1}. A leader's
    # call moves the boat: an early call (a_k down) shortens the next stroke, so T_{k+1} - T_k correlates with -a_k.
    rw = [(w[0]) for w in words if w[2] in RHYTHM or w[2] in COUNTS]
    a = {}
    for t0 in rw:
        kk = int(np.searchsorted(st, t0 + 0.5 * np.median(period))) - 1         # nearest thump
        if 1 <= kk < len(st) - 2:
            a.setdefault(kk, []).append(t0 - (st[kk] + (thump - 1.0) * period[kk]))
    ks = sorted(a)
    fol, lead = [], []
    for kk in ks:
        ak = np.mean(a[kk])
        if kk + 1 < len(period):
            lead.append((ak, period[kk + 1] - period[kk]))
        if kk + 1 in a:
            fol.append((period[kk] - period[kk - 1], np.mean(a[kk + 1])))
    def r(pairs):
        x = np.array(pairs)
        return (float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), len(x)) if len(x) > 8 else (float("nan"), len(x))
    res["direction"] = dict(follow_corr=r(fol), lead_corr=r(lead))
    print("  direction: follow (dT_k -> a_k+1) r %.2f (n %d); lead (a_k -> dT_k+1) r %.2f (n %d)"
          % (*r(fol), *r(lead)))
    p = os.path.join(ROOT, "data", "local", "coxing", "rhythm", key + ".json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, "w"), indent=1)
    print("written", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
