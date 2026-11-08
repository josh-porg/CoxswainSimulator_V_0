"""Build the outside raters' site for GitHub Pages (CODEBOOK addendum of 7 Nov 2026 and its hosting amendment).

    .venv-tools/Scripts/python.exe research/coxing/rating_site/build_site.py            # list names to review, build a test copy
    .venv-tools/Scripts/python.exe research/coxing/rating_site/build_site.py --final    # build the shareable site
        [--script-url https://script.google.com/macros/s/.../exec] [--rotate]

Everything that holds crew speech is published only encrypted (AES-256-GCM, one random key). The key travels in the
link's fragment (#k=...), which browsers never send to a server, so GitHub and Google see only ciphertext and opaque
codes. Writes data/local/coxing/rating_site/:
  site_key.json      the key, the admin key, the write token and the code -> piece/phrase map. PRIVATE: never publish.
  names_review.txt   capitalised words in any phrase that may be names, to mark "name" or "keep" (see below).
  site_build/        the folder to push to the site's GitHub repository:
    index.html, app.js, guide.html     the rater's site (from web/)
    Code.gs                            the Apps Script backend, to paste into the study lead's Google Sheet
    manifest.json      public: opaque item codes, piece codes, which items are check items or in the human sample
    pack.enc           encrypted: every phrase of the fifteen pieces (names redacted), race labels, check answers
    clips/<code>.bin   encrypted: a muffled clip (low-pass 400 Hz, 4 kHz sampling) for every call with a recording
    config.json        the Apps Script URL
Without --final the build goes to site_build_test/ and is never to be published (names may be unreviewed).
Names: the crew lineups (race-video title cards, race review) are redacted automatically; every other capitalised word
that never appears in lower case anywhere in the transcripts is listed in names_review.txt. Mark each line "name" or
"keep"; --final refuses while any line is unmarked.
"""
import argparse
import base64
import glob
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import wave

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, "..", "..", "callmodel"))
import call_properties as CP  # noqa: E402

LOCAL = os.path.join(ROOT, "data", "local")
SDIR = os.path.join(LOCAL, "coxing", "sessions")
EXP = os.path.join(LOCAL, "coxing", "experiment")
HC = os.path.join(LOCAL, "coxing", "human_coding")
OUT = os.path.join(LOCAL, "coxing", "rating_site")
KEYFILE = os.path.join(OUT, "site_key.json")
REVIEW = os.path.join(OUT, "names_review.txt")
WEB = os.path.join(CODE, "web")
PIECES = ["race_01", "practice_01", "practice_02", "race_04", "race_05", "practice_03", "practice_04", "practice_05",
          "practice_06", "practice_07", "practice_08", "race_06", "race_07", "race_08", "race_09"]
GROUPS = [  # what a rater ticks as "races I was in" (labels live only inside the encrypted pack)
    ("Pink Ribbon Row, Sept 2026 (women's masters four)", ["race_01"]),
    ("Practice races, 25 Sept 2026 (women's eight)", ["practice_01", "practice_02"]),
    ("Tail of the Lake, Oct 2026 (mixed masters eight)", ["race_04"]),
    ("Tail of the Lake, Oct 2026 (women's masters four)", ["race_05"]),
    ("Practice 800 m pieces, 7 Oct 2026 (women's masters four)", ["practice_03", "practice_04", "practice_05",
                                                                 "practice_06", "practice_07", "practice_08"]),
    ("Head of the Charles, Oct 2026 (women's veteran four)", ["race_06"]),
    ("Head of the Charles, Oct 2026 (men's veteran four)", ["race_07"]),
    ("Head of the Lake, Nov 2026 (women's eight)", ["race_08"]),
    ("Head of the Lake, Nov 2026 (men's four)", ["race_09"]),
]
AUDIO = {"practice_01": "0117", "practice_02": "0119", "race_04": "0132", "race_05": "0134", "race_06": "0149",
         "race_08": "0151", "race_09": "0153", **{k: "261007" for k in ("practice_03", "practice_04", "practice_05",
                                                                          "practice_06", "practice_07", "practice_08")}}
COMMON = set("""I I'm I'll I've I'd OK Okay Yes Yeah Yep Good Nice Great Let's Let Come Go Sit Stay Get Keep Hold Press
Push Drive Send Swing Catch Finish Long Length Ratio Rate Power Legs Body Bodies Core Chest Heels Hands Feet Starboard
Starboards Port Ports Stroke Bow That's That This There Here We We're You You're Your They They're Our And But So Now
Just All Big Even Back Up Down In On At To The A It It's Is Be Do Don't Can What Where When Who Why How Listen Remember
Ready Alright Awesome Excellent Perfect Beautiful Head Charles Lake Weeks Eliot Anderson Larz BU Cambridge Boston
Powerhouse Riverside Sculls Minute Mile Meters K Hup Cha Yap""".split())
CUTOFF_HZ, OUT_SR = 400.0, 4000


# ------------------------------------------------------------------ names
def known_names():
    names = set()
    for f in glob.glob(os.path.join(LOCAL, "overlay", "cards*.json")):
        for d in json.load(open(f, encoding="utf-8")).values():
            for line in str(d.get("lineup", "")).splitlines():
                names |= set(re.findall(r"[A-Za-z][a-z'\-]+", line))
    crews = json.load(open(os.path.join(LOCAL, "raceday", "crews.json"), encoding="utf-8"))
    names |= {c["cox"] for c in crews.get("crews", {}).values() if c.get("cox")}
    return {n for n in names if len(n) > 1}


def read_review():
    marks = {}
    if os.path.exists(REVIEW):
        for line in open(REVIEW, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and f[1] in ("name", "keep"):
                marks[f[0]] = f[1]
    return marks


def redactor(names):
    if not names:
        return lambda t: t
    rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b", re.I)
    return lambda t: rx.sub("[name]", t)


def review_names(all_phrases, names, marks):
    lower = {w for t in all_phrases for w in re.findall(r"[a-z][a-z'\-]*", t)}
    cand = {}
    for t in all_phrases:
        for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", t):
            if w[0].isupper() and w.lower() not in lower and w not in COMMON and w not in names and w not in marks:
                c = cand.setdefault(w, [0, t])
                c[0] += 1
    with open(REVIEW, "w", encoding="utf-8") as f:
        f.write("# word<TAB>name|keep<TAB>count<TAB>example. Lineup names are redacted automatically.\n")
        for w, m in sorted(marks.items()):
            f.write("%s\t%s\n" % (w, m))
        for w, (n, ex) in sorted(cand.items(), key=lambda x: -x[1][0]):
            f.write("%s\t?\t%d\t%s\n" % (w, n, ex))
    return cand


# ------------------------------------------------------------------ crypto
def b64u(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def seal(aes, data):
    nonce = secrets.token_bytes(12)
    return nonce + aes.encrypt(nonce, data, None)


def write_sealed(aes, path, data):
    """Encrypt data to path, but leave an existing file that already decrypts to the same data (so a rebuild with the
    same key changes, and re-publishes, only what changed)."""
    if os.path.exists(path):
        old = open(path, "rb").read()
        try:
            if aes.decrypt(old[:12], old[12:], None) == data:
                return
        except Exception:
            pass
    open(path, "wb").write(seal(aes, data))


def lowpass(x, sr, fc=CUTOFF_HZ):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X *= np.clip((fc * 1.25 - f) / (fc * 0.25), 0, 1)
    return np.fft.irfft(X, len(x))


def wav_bytes(y, sr):
    import io
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())
    return b.getvalue()


# ------------------------------------------------------------------ build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--script-url", default=None)
    ap.add_argument("--rotate", action="store_true", help="new key: every earlier link stops working")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(KEYFILE) and not a.rotate:
        K = json.load(open(KEYFILE))
    else:
        K = dict(key=b64u(secrets.token_bytes(32)), admin=b64u(secrets.token_bytes(18)), salt=secrets.token_hex(8))
    K["write_token"] = b64u(hashlib.sha256(("write:" + K["key"]).encode()).digest())
    aes = AESGCM(base64.urlsafe_b64decode(K["key"] + "=" * (-len(K["key"]) % 4)))
    # phrases of all pieces; the human coder's sample keeps its codes so outside, human and machine codes line up
    hkey = json.load(open(os.path.join(HC, "key.json")))["items"]
    sample = {(v["session"], v["i"]): k for k, v in hkey.items() if v["kind"] == "text" and "repeat_of" not in v}
    pcode = {sid: "P%02d" % (n + 1) for n, sid in enumerate(PIECES)}
    sess = {sid: CP.load_session(os.path.join(SDIR, sid)) for sid in PIECES}
    all_text = [p["text"] for s in sess.values() for p in s["phrases"]]
    marks = read_review()
    names = known_names() | {w for w, m in marks.items() if m == "name"}
    cand = review_names(all_text, names, marks)
    print("%d lineup names redacted; %d words reviewed; %d still to review in %s" % (len(names), len(marks), len(cand), REVIEW))
    if a.final and cand:
        raise SystemExit("not final: mark every line in names_review.txt as name or keep, then re-run with --final")
    red = redactor(names)
    items, man_items, clips_todo, codemap, checks = {}, [], [], {}, {}
    for sid in PIECES:
        s = sess[sid]
        codes = CP.load_codes(os.path.join(EXP, "codes_%s.txt" % sid))
        for idx, p in enumerate(s["phrases"]):
            if not s["t"][0] <= p["t"] <= s["t"][-1]:
                continue
            iid = sample.get((sid, p["i"])) or hashlib.sha256(("%s:%s:%d" % (K["salt"], sid, p["i"])).encode()).hexdigest()[:8]
            assert iid not in codemap, "item code collision"
            codemap[iid] = [sid, int(p["i"])]
            t = red(p["text"])
            tgt = CP.code_of(p["i"], codes)["target"]
            chk = None
            if tgt == "count" and re.fullmatch(r"(that's )?(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})", t.lower().strip(" .,!")):
                chk = "count"
            elif tgt == "info" and re.search(r"\b(you're|we're) at (a |that )?\d", t.lower()) and len(t.split()) <= 8:
                chk = "info"
            if chk:
                checks[iid] = chk
            items[iid] = [pcode[sid], idx]
            man_items.append([iid, pcode[sid], ("k" if chk else "") + ("s" if (sid, p["i"]) in sample else "")])
            if tgt not in ("count", "other") and (sid in AUDIO or sid == "race_07"):
                clips_todo.append((iid, sid, p))
    pieces = {pcode[sid]: [red(p["text"]) for p in sess[sid]["phrases"]] for sid in PIECES}
    groups = [[label, [pcode[s] for s in sids]] for label, sids in GROUPS]
    dest = os.path.join(OUT, "site_build" if a.final else "site_build_test")
    os.makedirs(os.path.join(dest, "clips"), exist_ok=True)
    plain = json.dumps(dict(v=1, pieces=pieces, items=items, groups=groups, checks=checks), ensure_ascii=False).encode()
    write_sealed(aes, os.path.join(dest, "pack.enc"), plain)
    # muffled clips, loudness relative to the piece's median call
    wavs, man_clips = {}, []
    for sid in sorted({c[1] for c in clips_todo}):
        s = sess[sid]
        akey = AUDIO.get(sid) or s["meta"]["source"].split()[-1]
        with wave.open(os.path.join(EXP, "audio", akey + ".wav")) as w:
            sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
        off = s["meta"]["sync_offset_s"]
        seg = lambda p: x[int(max(0, (p["t"] + off - 0.15) * sr)):int(min(len(x), (min(p.get("t_end") or p["t"], p["t"] + 3.5) + off + 0.3) * sr))]
        mine = [c for c in clips_todo if c[1] == sid]
        filt = {c[0]: lowpass(seg(c[2]), sr) for c in mine if len(seg(c[2])) > sr // 5}
        ref = np.median([np.sqrt(np.mean(y ** 2)) for y in filt.values()])
        step = sr // OUT_SR
        for iid, y in filt.items():
            ramp = np.minimum(1, np.minimum(np.arange(len(y)), np.arange(len(y))[::-1]) / (0.05 * sr))
            y = np.clip(y * ramp * (0.08 * 32767 / ref), -32767, 32767)[::step].astype(np.int16)
            write_sealed(aes, os.path.join(dest, "clips", iid + ".bin"), wav_bytes(y, sr // step))
            man_clips.append([iid, pcode[sid]])
    keep = {c[0] + ".bin" for c in man_clips}
    for f in os.listdir(os.path.join(dest, "clips")):        # drop clips of calls no longer in the site
        if f not in keep:
            os.remove(os.path.join(dest, "clips", f))
    json.dump(dict(v=1, items=man_items, clips=man_clips), open(os.path.join(dest, "manifest.json"), "w"), separators=(",", ":"))
    for f in ("index.html", "app.js", "guide.html", "Code.gs", "README.md") + (() if a.final else ("test.html",)):
        shutil.copyfile(os.path.join(WEB, f), os.path.join(dest, f))
    cfg_path = os.path.join(dest, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    if a.script_url:
        cfg["script"] = a.script_url
    json.dump(dict(script=cfg.get("script", ""), version=1), open(cfg_path, "w"))
    open(os.path.join(dest, "robots.txt"), "w").write("User-agent: *\nDisallow: /\n")
    open(os.path.join(dest, ".nojekyll"), "w").write("")
    K["codes"] = codemap
    K["pieces"] = {v: k for k, v in pcode.items()}
    json.dump(K, open(KEYFILE, "w"), indent=0)
    site = "https://josh-porg.github.io/coxing-ratings/"
    print("%s: %d phrases (%d in the human sample, %d check items), %d clips" % (
        dest, len(items), sum("s" in m[2] for m in man_items), len(checks), len(man_clips)))
    print("rater link:  %s#k=%s" % (site, K["key"]))
    print("admin link:  %s#k=%s&admin=%s" % (site, K["key"], K["admin"]))
    print("Apps Script properties: WRITE_TOKEN=%s  ADMIN_KEY=%s  MANIFEST_URL=%smanifest.json" % (K["write_token"], K["admin"], site))
    if not a.final:
        print("TEST BUILD ONLY (site_build_test): do not publish.")


if __name__ == "__main__":
    main()
