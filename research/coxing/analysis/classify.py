"""Classify the race's calls against Nugent et al. (2025)'s taxonomy.

Their three themes, with the exemplars they give:
  TECHNICAL   (40.4%) body movements / stroke mechanics -- "legs down", "through the
              toes", "hands up", "push the finish longer", "hold the knees", "stay loose"
  MOTIVATIONAL(38.6%) praise or encouragement -- "good start", "good rhythm",
              "we can take them", "we're walking", shared history
  TACTICAL    (21%)   crew changes and their preparation -- "ready?", "in two, in one",
              "go", "this one now"; position -- "coming up on the 500 m mark"; and
              BOAT METRICS -- "you're on 1:18", "still on 1:30's"
Elite coxes averaged 32 calls/min, 94% to the whole crew.

This is a RULE-BASED approximation of a human thematic analysis, so treat the split as
indicative. Where a call could be two things the first matching rule wins, and the order
is tactical -> technical -> motivational, because metrics and moves are the least
ambiguous to spot.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import re

import numpy as np

# --- parse the VTT into word-level timings ---------------------------------------
txt = open("caps.en.vtt", encoding="utf-8").read()
cues = re.findall(r"(\d\d:\d\d:\d\d\.\d\d\d) --> .*?\n(.*?)(?=\n\n|\Z)", txt, re.S)
seen, words = set(), []
for ts, body in cues:
    h, m, s = ts.split(":")
    t0 = int(h) * 3600 + int(m) * 60 + float(s)
    clean = re.sub(r"<[^>]+>", "", body).strip()
    for w in clean.split():
        key = (round(t0, 2), w)
        if key in seen:
            continue
        seen.add(key)
        words.append((t0, w))
# collapse the rolling duplicates: keep a word only if it is new since 3 s ago
out, recent = [], []
for t0, w in words:
    recent = [(tt, ww) for tt, ww in recent if t0 - tt < 3.0]
    if any(ww == w for _, ww in recent):
        continue
    recent.append((t0, w))
    out.append((t0, w))
print("words parsed %d, span %.1f - %.1f s" % (len(out), out[0][0], out[-1][0]))

# --- rebuild sentences, each carrying the time of its first word ------------------
calls, cur, t_start = [], [], None
for t0, w in out:
    if t_start is None:
        t_start = t0
    cur.append(w)
    if w.endswith((".", "?", "!")) or len(cur) > 14:
        calls.append((t_start, " ".join(cur)))
        cur, t_start = [], None
if cur:
    calls.append((t_start, " ".join(cur)))
print("calls (sentence-ish units) %d over %.1f min = %.1f per minute"
      % (len(calls), out[-1][0] / 60.0, len(calls) / (out[-1][0] / 60.0)))

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

# counting out a burst is how the paper says a change is DELIVERED ("in two, in one",
# then "go"), so bare numbers are tactical. Bare acknowledgements are not calls to the
# crew at all and are excluded from the denominator rather than bucketed.
COUNT = re.compile(r"^(that.s )?(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})[.,!]?( two| three)?[.,!]?$", re.I)
FILLER = re.compile(r"^(okay|ok|yeah|yep|yes|thanks|noted|right|alright|so|and|um|uh)[.,!]?$", re.I)
counts = {"technical": 0, "motivational": 0, "tactical": 0, "other": 0}
excluded = 0
tagged = []
for t0, s in calls:
    st = s.strip()
    if FILLER.match(st):
        excluded += 1
        continue
    if COUNT.match(st):
        k = "tactical"
        counts[k] += 1
        tagged.append((t0, s, k))
        continue
    if TACT.search(s):
        k = "tactical"
    elif TECH.search(s):
        k = "technical"
    elif MOTI.search(s):
        k = "motivational"
    else:
        k = "other"
    counts[k] += 1
    tagged.append((t0, s, k))
n = sum(counts.values())
print("excluded as bare acknowledgements (not calls): %d" % excluded)
print()
print("%-14s %8s %9s %12s" % ("theme", "this race", "share", "Nugent 2025"))
for k, ref in (("technical", 40.4), ("motivational", 38.6), ("tactical", 21.0), ("other", None)):
    share = 100.0 * counts[k] / n
    print("%-14s %8d %8.1f%% %11s" % (k, counts[k], share, "%.1f%%" % ref if ref else "-"))
np.save("tagged_times.npy", np.array([t for t, _, _ in tagged]))
import json
json.dump([[t, s, k] for t, s, k in tagged], open("tagged.json", "w"))
