"""Draw the human coder's sample and write what the coding page may see (CODEBOOK addendum of 6 Nov 2026).

    .venv/Scripts/python.exe research/coxing/human_coding/pack.py

Writes data/local/coxing/human_coding/:
  pack_text.json   presentation order of 660 items (600 phrases + 60 unannounced repeats): opaque id, the phrase,
                   the three phrases before and two after it. Nothing else: no piece, time, crew or code.
  pack_voice.json  presentation order of 220 clips (200 + 20 repeats): opaque id only.
  voice/<id>.wav   the clip, low-pass filtered at 400 Hz (words unintelligible; loudness, pitch and pace kept),
                   scaled by its piece's median call loudness so loudness is relative to that coxswain's recording.
  key.json         id -> piece, phrase index, repeat_of. Read only by analyse.py, never by the page.
Re-running with the same seeds reproduces the same sample; it refuses to overwrite a pack that has codes logged.
"""
import json
import os
import secrets
import sys
import wave

import numpy as np

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "research", "callmodel"))
import call_properties as CP  # noqa: E402

LOCAL = os.path.join(ROOT, "data", "local", "coxing")
SDIR = os.path.join(LOCAL, "sessions")
EXP = os.path.join(LOCAL, "experiment")
OUT = os.path.join(LOCAL, "human_coding")
PIECES = ["race_01", "practice_01", "practice_02", "race_04", "race_05", "practice_03", "practice_04", "practice_05",
          "practice_06", "practice_07", "practice_08", "race_06", "race_07", "race_08", "race_09"]
AUDIO = {"practice_01": "0117", "practice_02": "0119", "race_04": "0132", "race_05": "0134", "race_06": "0149",
         "race_08": "0151", "race_09": "0153", **{k: "261007" for k in PIECES if k.startswith("practice_0")
                                                    and k not in ("practice_01", "practice_02")}}
N_TEXT, N_TEXT_REPEAT, MIN_GAP = 600, 60, 120
N_VOICE, N_VOICE_REPEAT = 200, 20
SEED_TEXT, SEED_VOICE = 20261106, 20261107
CUTOFF_HZ = 400.0


def load(sid):
    s = CP.load_session(os.path.join(SDIR, sid))
    codes = CP.load_codes(os.path.join(EXP, "codes_%s.txt" % sid))
    ph = [p for p in s["phrases"] if s["t"][0] <= p["t"] <= s["t"][-1]]
    for p in ph:
        p["target"] = CP.code_of(p["i"], codes)["target"]
    return s, ph, s["phrases"]


def allocate(sizes, total):
    """Largest-remainder allocation of ``total`` in proportion to ``sizes``."""
    raw = np.array(sizes, float) / sum(sizes) * total
    n = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - n))[:total - n.sum()]:
        n[k] += 1
    return n


def stratified(rng, ph, n):
    """n phrases from one piece, stratified by the primary target, at least one from every target present."""
    by = {}
    for p in ph:
        by.setdefault(p["target"], []).append(p)
    tg = sorted(by)
    if n < len(tg):
        tg = list(rng.choice(tg, n, replace=False))
    base = {t: 1 for t in tg}
    rest = allocate([len(by[t]) for t in tg], n - len(tg)) if n > len(tg) else np.zeros(len(tg), int)
    out = []
    for t, r in zip(tg, rest):
        k = min(len(by[t]), base[t] + int(r))
        out += list(rng.choice(by[t], k, replace=False))
    return out


def with_repeats(rng, items, n_rep):
    """Shuffle, then insert n_rep repeats of earlier items at least MIN_GAP positions after their first showing."""
    order = list(rng.permutation(len(items)))
    seq = [items[k] for k in order]
    rep_src = sorted(rng.choice(len(seq) - MIN_GAP, n_rep, replace=False))
    for src in rep_src[::-1]:
        pos = int(rng.integers(src + MIN_GAP, len(seq) + 1))
        seq.insert(pos, dict(seq[src], repeat_of=seq[src]["id"]))
    for x in seq:
        if "repeat_of" in x:
            x["id"] = secrets.token_hex(4)
    return seq


def lowpass(x, sr, fc=CUTOFF_HZ):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X *= np.clip((fc * 1.25 - f) / (fc * 0.25), 0, 1)        # flat to fc, linear taper to 1.25 fc
    return np.fft.irfft(X, len(x))


def main():
    os.makedirs(os.path.join(OUT, "voice"), exist_ok=True)
    if os.path.exists(os.path.join(OUT, "log_text.jsonl")):
        raise SystemExit("codes are already logged against the existing pack; not redrawing")
    data = {sid: load(sid) for sid in PIECES}
    key = {}
    # ---- text sample
    rng = np.random.default_rng(SEED_TEXT)
    quota = allocate([len(data[k][1]) for k in PIECES], N_TEXT)
    items = []
    for sid, q in zip(PIECES, quota):
        s, ph, allph = data[sid]
        idx = {p["i"]: k for k, p in enumerate(allph)}
        for p in stratified(rng, ph, int(q)):
            k = idx[p["i"]]
            iid = secrets.token_hex(4)
            key[iid] = dict(kind="text", session=sid, i=int(p["i"]))
            items.append(dict(id=iid, text=p["text"], before=[x["text"] for x in allph[max(k - 3, 0):k]],
                              after=[x["text"] for x in allph[k + 1:k + 3]]))
    seq = with_repeats(rng, items, N_TEXT_REPEAT)
    for x in seq:
        if "repeat_of" in x:
            key[x["id"]] = dict(key[x["repeat_of"]], repeat_of=x.pop("repeat_of"))
    json.dump(dict(items=seq, block=40), open(os.path.join(OUT, "pack_text.json"), "w", encoding="utf-8"), indent=0,
              ensure_ascii=False)
    print("text: %d items (%d repeats), per piece %s" % (len(seq), N_TEXT_REPEAT, dict(zip(PIECES, quota.tolist()))))
    # ---- voice sample: call phrases in pieces with a local recording
    rng = np.random.default_rng(SEED_VOICE)
    vp = [k for k in PIECES if k in AUDIO or k == "race_07"]
    calls = {k: [p for p in data[k][1] if p["target"] not in ("count", "other")] for k in vp}
    vq = allocate([len(calls[k]) for k in vp], N_VOICE)
    wavs, clips = {}, []
    for sid, q in zip(vp, vq):
        s = data[sid][0]
        akey = AUDIO.get(sid) or s["meta"]["source"].split()[-1]
        if akey not in wavs:
            with wave.open(os.path.join(EXP, "audio", akey + ".wav")) as w:
                wavs[akey] = (w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float))
        sr, x = wavs[akey]
        off = s["meta"]["sync_offset_s"]
        seg = lambda p: x[int(max(0, (p["t"] + off - 0.15) * sr)):int(min(len(x), (min(p.get("t_end") or p["t"], p["t"] + 3.5)
                                                                             + off + 0.3) * sr))]
        ref = np.median([np.sqrt(np.mean(lowpass(seg(p), sr) ** 2)) for p in calls[sid] if len(seg(p)) > sr // 5])
        for p in stratified(rng, calls[sid], int(q)):
            y = lowpass(seg(p), sr)
            if len(y) < sr // 5:
                continue
            ramp = np.minimum(1, np.minimum(np.arange(len(y)), np.arange(len(y))[::-1]) / (0.05 * sr))
            y = np.clip(y * ramp * (0.08 * 32767 / ref), -32767, 32767).astype(np.int16)
            iid = secrets.token_hex(4)
            with wave.open(os.path.join(OUT, "voice", iid + ".wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(y.tobytes())
            key[iid] = dict(kind="voice", session=sid, i=int(p["i"]))
            clips.append(dict(id=iid))
    rng2 = np.random.default_rng(SEED_VOICE + 1)
    vseq = [dict(x) for x in clips]
    vseq = [vseq[k] for k in rng2.permutation(len(vseq))]
    for src in sorted(rng2.choice(len(vseq) - 40, N_VOICE_REPEAT, replace=False))[::-1]:
        pos = int(rng2.integers(src + 40, len(vseq) + 1))
        rid = secrets.token_hex(4)
        key[rid] = dict(key[vseq[src]["id"]], repeat_of=vseq[src]["id"])
        os.link(os.path.join(OUT, "voice", vseq[src]["id"] + ".wav"), os.path.join(OUT, "voice", rid + ".wav"))
        vseq.insert(pos, dict(id=rid))
    json.dump(dict(items=vseq), open(os.path.join(OUT, "pack_voice.json"), "w"), indent=0)
    json.dump(dict(seed_text=SEED_TEXT, seed_voice=SEED_VOICE, cutoff_hz=CUTOFF_HZ, items=key),
              open(os.path.join(OUT, "key.json"), "w"), indent=0)
    print("voice: %d clips (%d repeats) from %d pieces" % (len(vseq), N_VOICE_REPEAT, len(vp)))


if __name__ == "__main__":
    main()
