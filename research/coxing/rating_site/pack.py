"""Build the outside raters' pack from the human coder's sample (CODEBOOK addendum of 7 Nov 2026).

    .venv/Scripts/python.exe research/coxing/rating_site/pack.py            # build, and list possible names to review
    .venv/Scripts/python.exe research/coxing/rating_site/pack.py --final    # refuse unless every candidate is reviewed

Reads data/local/coxing/human_coding/ (pack_text.json, key.json, voice/) and writes data/local/coxing/rating_site/:
  pack.json        phrases (first showings of the 600), with names replaced by "[name]" in the phrase and its context,
                   the session each comes from (server-side only, for excluding a rater's own pieces), and check items;
                   clips (the 200 first-showing muffled clips).
  clips/<id>.wav   copies of those clips.
  names_review.txt every capitalised word inside a phrase that is not already a known name or a listed common word,
                   with a count and an example. Mark each line: "name" (redact) or "keep" (not a name), then re-run.
Known names come from the crew lineups on the race videos' title cards and the race review (data/local only).
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, "..", "..", "callmodel"))
import call_properties as CP  # noqa: E402

LOCAL = os.path.join(ROOT, "data", "local")
HC = os.path.join(LOCAL, "coxing", "human_coding")
EXP = os.path.join(LOCAL, "coxing", "experiment")
OUT = os.path.join(LOCAL, "coxing", "rating_site")
REVIEW = os.path.join(OUT, "names_review.txt")
COMMON = set("""I I'm I'll I've I'd OK Okay Yes Yeah Yep Good Nice Great Let's Let Come Go Sit Stay Get Keep Hold Press
Push Drive Send Swing Catch Finish Long Length Ratio Rate Power Legs Body Bodies Core Chest Heels Hands Feet Starboard
Starboards Port Ports Stroke Bow Two Three Four Five Six Seven Eight Nine Ten One That's That This There Here We We're
You You're Your They They're Our And But So Now Just All Big Even Back Up Down In On At To The A It It's Is Be Do Don't
Can What Where When Who Why How Listen Remember Ready Okay Alright Awesome Excellent Perfect Beautiful Head Charles
Lake Weeks Eliot Anderson Larz BU Cambridge Boston Powerhouse Riverside Sculls Minute Mile Meters K Hup Cha Yap
Breathe Relax Squeeze Hang Suspend Connect Attack Direct Clean Sharp Strong Tall Proud Excited Amazing Absolutely""".split())


def known_names():
    names = set()
    for f in glob.glob(os.path.join(LOCAL, "overlay", "cards*.json")):
        for d in json.load(open(f, encoding="utf-8")).values():
            for line in str(d.get("lineup", "")).splitlines():
                for w in re.findall(r"[A-Za-z][a-z'\-]+", line):
                    names.add(w)
    crews = json.load(open(os.path.join(LOCAL, "raceday", "crews.json"), encoding="utf-8"))
    for c in crews.get("crews", {}).values():
        if c.get("cox"):
            names.add(c["cox"])
    return {n for n in names if len(n) > 1}


def read_review():
    marks = {}
    if os.path.exists(REVIEW):
        for line in open(REVIEW, encoding="utf-8"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and f[1] in ("name", "keep"):
                marks[f[0]] = f[1]
    return marks


def redact(text, names):
    if not names:
        return text
    rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b", re.I)
    return rx.sub("[name]", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "clips"), exist_ok=True)
    key = json.load(open(os.path.join(HC, "key.json")))["items"]
    text = json.load(open(os.path.join(HC, "pack_text.json"), encoding="utf-8"))["items"]
    voice = json.load(open(os.path.join(HC, "pack_voice.json")))["items"]
    marks = read_review()
    names = known_names() | {w for w, m in marks.items() if m == "name"}
    keep = {w for w, m in marks.items() if m == "keep"}
    # candidates: capitalised words that never occur in lower case anywhere in the transcripts (names are always
    # capitalised; ordinary words also turn up lower case), not already known, common or reviewed
    lower = set()
    for f in glob.glob(os.path.join(LOCAL, "coxing", "sessions", "*", "phrases.json")):
        for p in json.load(open(f, encoding="utf-8")):
            lower |= {w for w in re.findall(r"[a-z][a-z'\-]*", p["text"])}
    cand = {}
    for it in text:
        for line in [it["text"]] + it["before"] + it["after"]:
            for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", line):
                if w[0].isupper() and w.lower() not in lower and w not in COMMON and w not in names and w not in keep:
                    c = cand.setdefault(w, [0, line])
                    c[0] += 1
    with open(REVIEW, "w", encoding="utf-8") as f:
        f.write("# word<TAB>name|keep<TAB>count<TAB>example. Known lineup names are redacted automatically.\n")
        for w, m in sorted(marks.items()):
            f.write("%s\t%s\n" % (w, m))
        for w, (n, ex) in sorted(cand.items(), key=lambda x: -x[1][0]):
            f.write("%s\t?\t%d\t%s\n" % (w, n, ex))
    print("%d known names redacted automatically; %d candidate words to review in %s" % (len(names), len(cand), REVIEW))
    if a.final and cand:
        raise SystemExit("not final: mark every candidate in names_review.txt as name or keep, then re-run")
    codes = {}
    items = []
    for it in text:
        k = key[it["id"]]
        if "repeat_of" in k:
            continue
        sid, i = k["session"], k["i"]
        if sid not in codes:
            codes[sid] = CP.load_codes(os.path.join(EXP, "codes_%s.txt" % sid))
        t = redact(it["text"], names)
        tgt = CP.code_of(i, codes[sid])["target"]
        check = None                     # unambiguous targets: a bare count, or a split readout
        if tgt == "count" and re.fullmatch(r"(that's )?(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})", t.lower().strip(" .,!")):
            check = "count"
        elif tgt == "info" and re.search(r"\b(you're|we're) at (a |that )?\d", t.lower()) and len(t.split()) <= 8:
            check = "info"
        items.append(dict(id=it["id"], session=sid, text=t, before=[redact(x, names) for x in it["before"]],
                          after=[redact(x, names) for x in it["after"]], check=check))
    clips = []
    for c in voice:
        k = key[c["id"]]
        if "repeat_of" in k:
            continue
        shutil.copyfile(os.path.join(HC, "voice", c["id"] + ".wav"), os.path.join(OUT, "clips", c["id"] + ".wav"))
        clips.append(dict(id=c["id"], session=k["session"]))
    json.dump(dict(items=items, clips=clips, final=bool(a.final)), open(os.path.join(OUT, "pack.json"), "w", encoding="utf-8"),
              indent=0, ensure_ascii=False)
    print("pack: %d phrases (%d check items), %d clips%s" % (len(items), sum(1 for x in items if x["check"]), len(clips),
                                                           "" if a.final else "  (NOT final: review names before deploying)"))


if __name__ == "__main__":
    main()
