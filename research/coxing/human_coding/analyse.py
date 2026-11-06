"""The human coder against the machine coders (CODEBOOK addendum of 6 Nov 2026; plan fixed before coding).

    .venv/Scripts/python.exe research/coxing/human_coding/analyse.py

Text: Cohen's kappa per field between the human and the primary coder (first showings only; the human's main target,
i.e. the first of several; fields other than target only where both coded a call, not a count or other talk), with a
95% bootstrap interval; how often the primary coder's target is among the human's targets; flags and notes; for arousal also
against the hypothesis-blind and lexical coders. Test-retest kappa per field from the unannounced repeats.
Voice: Spearman correlation of the 1-7 rating with the acoustic vocal arousal index of the same phrase (loudness,
F0 and speech rate, each z-scored within the piece over its call phrases), the mean rating by each coder's arousal
code, and test-retest correlation. The hash chain of each log is checked first. Writes analysis.json.
"""
import hashlib
import json
import os
import sys
import wave

import numpy as np

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
CODE = os.path.dirname(os.path.abspath(__file__))           # code beside this file (this checkout), data under ROOT
sys.path.insert(0, os.path.join(CODE, "..", "..", "callmodel"))
sys.path.insert(0, os.path.join(CODE, "..", "experiment"))
import call_properties as CP  # noqa: E402

DATA = os.environ.get("CODING_DATA", os.path.join(ROOT, "data", "local", "coxing", "human_coding"))
EXP = os.path.join(ROOT, "data", "local", "coxing", "experiment")
SDIR = os.path.join(ROOT, "data", "local", "coxing", "sessions")
FIELDS = CP.FIELDS


def chain_ok(log):
    prev = "0" * 16
    for r in log:
        body = {k: v for k, v in r.items() if k != "hash"}
        if r["prev"] != prev or hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16] != r["hash"]:
            return False
        prev = r["hash"]
    return True


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    ix = {c: k for k, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        M[ix[x], ix[y]] += 1
    n = M.sum()
    if n == 0:
        return np.nan
    po, pe = np.trace(M) / n, (M.sum(0) * M.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def boot(a, b, rng, n=2000):
    a, b = np.array(a), np.array(b)
    ks = [kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return [float(np.nanpercentile(ks, 2.5)), float(np.nanpercentile(ks, 97.5))]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def read(name):
    p = os.path.join(DATA, name)
    return [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []


def main():
    import blind_arousal
    import rulecoder
    key = json.load(open(os.path.join(DATA, "key.json")))["items"]
    rng = np.random.default_rng(1)
    out = {}
    lt = read("log_text.jsonl")
    out["text_chain_ok"] = chain_ok(lt)
    first = [r for r in lt if "repeat_of" not in key[r["id"]]]
    print("text: %d coded (%d first showings), hash chain %s" % (len(lt), len(first), "intact" if out["text_chain_ok"] else "BROKEN"))
    cache = {}

    def primary(sid, i):
        if sid not in cache:
            cache[sid] = (CP.load_codes(os.path.join(EXP, "codes_%s.txt" % sid)),
                          {p["i"]: p["text"] for p in json.load(open(os.path.join(SDIR, sid, "phrases.json"), encoding="utf-8"))},
                          blind_arousal.load_blind(sid))
        c, ph, bl = cache[sid]
        return CP.code_of(i, c), ph[i], bl.get(i)
    rows = []
    for r in first:
        k = key[r["id"]]
        pc, text, bl = primary(k["session"], k["i"])
        rows.append((r["codes"], pc, bl, rulecoder.code(text)["arousal"]))
    out["fields"] = {}
    for f in FIELDS:
        sel = [(h, p) for h, p, _, _ in rows if f == "target" or (h["target"] not in ("count", "other") and p["target"] not in ("count", "other"))]
        a, b = [h[f] for h, _ in sel], [p[f] for _, p in sel]
        if len(a) < 10:
            continue
        out["fields"][f] = dict(n=len(a), kappa_primary=kappa(a, b), ci=boot(a, b, rng))
        print("  %-9s n=%3d  kappa vs primary %.2f (95%% %.2f-%.2f)" % (f, len(a), out["fields"][f]["kappa_primary"], *out["fields"][f]["ci"]))
    multi = [(h.get("targets") or [h["target"]], p["target"]) for h, p, _, _ in rows]
    out["targets"] = dict(n=len(multi), several=sum(len(t) > 1 for t, _ in multi),
                          primary_among_human=float(np.mean([p in t for t, p in multi])) if multi else None)
    if multi:
        print("  targets: %d of %d phrases have several; the primary coder's target is among the human's in %.0f%%" % (
            out["targets"]["several"], len(multi), 100 * out["targets"]["primary_among_human"]))
    calls = [(h, bl, ru) for h, p, bl, ru in rows if h["target"] not in ("count", "other") and bl is not None]
    if len(calls) >= 10:
        a = [h["arousal"] for h, _, _ in calls]
        out["arousal_vs_blind"] = dict(n=len(a), kappa=kappa(a, [x[1] for x in calls]), ci=boot(a, [x[1] for x in calls], rng))
        out["arousal_vs_lexical"] = dict(n=len(a), kappa=kappa(a, [x[2] for x in calls]), ci=boot(a, [x[2] for x in calls], rng))
        print("  arousal kappa vs blind coder %.2f, vs lexical coder %.2f (n=%d)" % (
            out["arousal_vs_blind"]["kappa"], out["arousal_vs_lexical"]["kappa"], len(a)))
    byid = {r["id"]: r for r in lt}
    reps = [(byid[key[r["id"]]["repeat_of"]]["codes"], r["codes"]) for r in lt if "repeat_of" in key[r["id"]]
            and key[r["id"]]["repeat_of"] in byid]
    if reps:
        out["retest"] = {f: kappa([a[f] for a, _ in reps], [b[f] for _, b in reps]) for f in FIELDS}
        print("  test-retest kappa (%d repeats): %s" % (len(reps), ", ".join("%s %.2f" % (f, v) for f, v in out["retest"].items())))
    out["flags"] = {g: sum(g in (r.get("flags") or []) for r in lt) for g in ("fragment", "unsure", "transcript wrong", "signal")}
    out["notes"] = sum(bool(r.get("note")) for r in lt)
    print("  flags %s; %d notes" % (out["flags"], out["notes"]))
    out["seconds_per_item"] = float(np.median([r["ms"] for r in lt]) / 1000) if lt else None
    # ---- voice
    lv = read("log_voice.jsonl")
    out["voice_chain_ok"] = chain_ok(lv)
    if lv:
        firstv = [r for r in lv if "repeat_of" not in key[r["id"]]]
        idx = acoustic_index({(key[r["id"]]["session"], key[r["id"]]["i"]) for r in firstv})
        pairs = [(r["rating"], idx.get((key[r["id"]]["session"], key[r["id"]]["i"]))) for r in firstv]
        pairs = [(a, b) for a, b in pairs if b is not None and np.isfinite(b)]
        out["voice"] = dict(n=len(firstv), spearman_with_index=spearman([a for a, _ in pairs], [b for _, b in pairs]))
        for name, get in (("primary", lambda s, i: primary(s, i)[0]["arousal"]), ("blind", lambda s, i: primary(s, i)[2])):
            by = {}
            for r in firstv:
                by.setdefault(get(key[r["id"]]["session"], key[r["id"]]["i"]), []).append(r["rating"])
            out["voice"]["mean_rating_by_%s_code" % name] = {k: [float(np.mean(v)), len(v)] for k, v in by.items()}
        vb = {r["id"]: r for r in lv}
        rv = [(vb[key[r["id"]]["repeat_of"]]["rating"], r["rating"]) for r in lv if "repeat_of" in key[r["id"]]]
        if rv:
            out["voice"]["retest_spearman"] = spearman([a for a, _ in rv], [b for _, b in rv])
        print("voice: %d rated, hash chain %s; Spearman with the acoustic index %.2f; %s" % (
            len(lv), "intact" if out["voice_chain_ok"] else "BROKEN", out["voice"]["spearman_with_index"],
            json.dumps({k: v for k, v in out["voice"].items() if k.startswith("mean") or k.startswith("retest")})))
    json.dump(out, open(os.path.join(DATA, "analysis.json"), "w"), indent=1)


def acoustic_index(wanted):
    """Vocal arousal index of each wanted (session, phrase): mean of loudness, F0 and speech rate, each z-scored over
    the call phrases of its piece (CP.delivery on the 16 kHz recording)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pack
    out = {}
    for sid in sorted({s for s, _ in wanted}):
        s = CP.load_session(os.path.join(SDIR, sid))
        akey = pack.AUDIO.get(sid) or s["meta"]["source"].split()[-1]
        with wave.open(os.path.join(EXP, "audio", akey + ".wav")) as w:
            sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
        off = s["meta"]["sync_offset_s"]
        codes = CP.load_codes(os.path.join(EXP, "codes_%s.txt" % sid))
        ph = [p for p in s["phrases"] if codes.get(p["i"], {}).get("target", "count") not in ("count", "other")]
        m = np.array([CP.delivery(x, sr, p["t"] + off, max(p["t"], p.get("t_end") or p["t"]) + off, len(p["text"].split()))
                      for p in ph], float)
        z = (m - np.nanmean(m, 0)) / np.nanstd(m, 0)
        v = np.nanmean(z, 1)
        for p, vi in zip(ph, v):
            if (sid, p["i"]) in wanted:
                out[(sid, p["i"])] = float(vi)
    return out


if __name__ == "__main__":
    main()
