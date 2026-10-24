"""External validation: run the SAME rule-based classifier on elite recordings.

Nugent et al. report technical 40.4%, motivational 38.6%, tactical 21.0%, at 32 calls
per minute, from eight elite recordings found on YouTube. Their search strategy is
published, so the sample is reconstructible. If the classifier used on the masters race
reproduces those proportions here, the masters numbers are credible. If it does not, the
classifier is biased and the masters comparison has to be discounted by that bias.

Nothing is tuned to make this agree.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import glob
import os
import re

import numpy as np

TACT = re.compile(r"\b(2[0-3]\d|1[89]\d)\b|\bin two\b|\btake (a )?(10|ten|five|5)\b|"
                  r"\b\d+\s*m\b|metre|meter|\blengths?\b|passing|pass a|yield|clear|"
                  r"stern deck|bow|lane|course|finish line|requalif|re-qualif|handicap|"
                  r"starboard|port\b|corner|turn\b|halfway|quarter|split|stroke rate|"
                  r"\b2[0-9]\b|per stroke|boat ahead|these other|walk on", re.I)
TECH = re.compile(r"catch|finish(es)?\b|legs?\b|bod(y|ies)|swing|suspend|hang|handle|"
                  r"blade|square|release|sequenc|reach|ratio|length|long\b|press|connect|"
                  r"seat|shoulder|sit up|chest|breathe|rhythm|together|complete|direct|"
                  r"sharp|attack|front end|hook|send|puddle|stretcher|toes|heels|hands|"
                  r"tidy|loose|prep\b", re.I)
MOTI = re.compile(r"\bgood\b|\byes\b|\bnice\b|excellent|great|that's it|you've got|"
                  r"dig deep|commit|capable|grit|power\b|strong|believe|boat ?mates|"
                  r"teammates|hurt|100%|110|doing it|proud|remember|promise|courage|"
                  r"find it in you|show (them|us|me)|we can|we have|ladies|guys", re.I)
COUNT = re.compile(r"^(that.s )?(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})"
                   r"[.,!]?( two| three)?[.,!]?$", re.I)
FILLER = re.compile(r"^(okay|ok|yeah|yep|yes|thanks|noted|right|alright|so|and|um|uh)[.,!]?$", re.I)

NAMES = {
    "G2ZrDIb6NYI": "World Champs final LM8+, 2011",
    "9RGu9kUolkE": "World U23 GB M8+, 2021",
    "ZdDz9cORKKg": "World Cup III W8+ final, 2022",
    "O_c3OfT1Wv0": "Henley Ladies Plate semi, 2022",
    "ZX2ID3SDmt8": "Henley final W varsity 8+",
    "IODivDoXZRI": "Henley semi, Thames v Barge",
}


def parse(path):
    txt = open(path, encoding="utf-8").read()
    cues = re.findall(r"(\d\d:\d\d:\d\d\.\d\d\d) --> .*?\n(.*?)(?=\n\n|\Z)", txt, re.S)
    out, recent = [], []
    for ts, body in cues:
        h, m, s = ts.split(":")
        t0 = int(h) * 3600 + int(m) * 60 + float(s)
        clean = re.sub(r"<[^>]+>", "", body).strip()
        for w in clean.split():
            recent = [(tt, ww) for tt, ww in recent if t0 - tt < 3.0]
            if any(ww == w for _, ww in recent):
                continue
            recent.append((t0, w))
            out.append((t0, w))
    calls, cur, t0 = [], [], None
    for t, w in out:
        if t0 is None:
            t0 = t
        cur.append(w)
        if w.endswith((".", "?", "!")) or len(cur) > 14:
            calls.append((t0, " ".join(cur)))
            cur, t0 = [], None
    if cur:
        calls.append((t0, " ".join(cur)))
    span = out[-1][0] - out[0][0] if out else 0.0
    return calls, span


def classify(calls):
    c = {"technical": 0, "motivational": 0, "tactical": 0, "other": 0}
    excl = 0
    for _, s in calls:
        st = s.strip()
        if FILLER.match(st):
            excl += 1
            continue
        if COUNT.match(st):
            c["tactical"] += 1
        elif TACT.search(s):
            c["tactical"] += 1
        elif TECH.search(s):
            c["technical"] += 1
        elif MOTI.search(s):
            c["motivational"] += 1
        else:
            c["other"] += 1
    return c, excl


print("%-34s %6s %7s %7s %7s %7s %7s" % ("recording", "min", "calls", "c/min", "tech", "moti", "tact"))
print("-" * 86)
tot = {"technical": 0, "motivational": 0, "tactical": 0, "other": 0}
tot_calls, tot_min = 0, 0.0
for p in sorted(glob.glob("*.en.vtt")):
    vid = os.path.basename(p).split(".")[0]
    calls, span = parse(p)
    c, excl = classify(calls)
    n = sum(c.values())
    mins = span / 60.0
    for k in tot:
        tot[k] += c[k]
    tot_calls += n
    tot_min += mins
    print("%-34s %6.1f %7d %7.1f %6.1f%% %6.1f%% %6.1f%%"
          % (NAMES.get(vid, vid)[:34], mins, n, n / max(mins, 1e-9),
             100 * c["technical"] / n, 100 * c["motivational"] / n, 100 * c["tactical"] / n))

N = sum(tot.values())
print("-" * 86)
print("%-34s %6.1f %7d %7.1f %6.1f%% %6.1f%% %6.1f%%"
      % ("POOLED ELITE (this classifier)", tot_min, tot_calls, tot_calls / tot_min,
         100 * tot["technical"] / N, 100 * tot["motivational"] / N, 100 * tot["tactical"] / N))
print("%-34s %6s %7s %7.1f %6.1f%% %6.1f%% %6.1f%%"
      % ("NUGENT ET AL., published", "-", "-", 32.0, 40.4, 38.6, 21.0))
print()
print("unclassified in the elite pool: %.1f%%" % (100 * tot["other"] / N))
print()
print("masters race, same classifier:  33.7 calls/min, tech 30.6%, moti 15.1%, tact 33.5%")
