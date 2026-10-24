"""Re-run classification and the impulse-response fit on de-duplicated captions."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "elite")
import vtt

TACT = re.compile(r"\b(2[0-3]\d|1[89]\d)\b|\bin two\b|\btake (a )?(10|ten|five|5)\b|"
                  r"\b\d+\s*m\b|metre|meter|\blengths?\b|passing|pass a|yield|clear|"
                  r"stern deck|bow|lane|course|finish line|requalif|handicap|"
                  r"starboard|port\b|corner|turn\b|halfway|quarter|split|stroke rate|"
                  r"\b2[0-9]\b|per stroke|boat ahead|these other|walk on|seat", re.I)
TECH = re.compile(r"catch|finish(es)?\b|legs?\b|bod(y|ies)|swing|suspend|hang|handle|"
                  r"blade|square|release|sequenc|reach|ratio|length|long\b|press|connect|"
                  r"shoulder|sit up|chest|breathe|rhythm|together|complete|direct|"
                  r"sharp|attack|front end|hook|send|puddle|stretcher|toes|heels|hands|"
                  r"tidy|loose|prep\b|drive|flat|low\b|clean", re.I)
MOTI = re.compile(r"\bgood\b|\byes\b|\bnice\b|excellent|great|that's it|you've got|"
                  r"dig deep|commit|capable|grit|power\b|strong|believe|boat ?mates|"
                  r"teammates|hurt|100%|110|doing it|proud|remember|promise|courage|"
                  r"find it in you|show (them|us|me)|we can|we have|ladies|guys|boys|"
                  r"come on|let's go|yeah", re.I)
COUNT = re.compile(r"^(that.s )?(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})"
                   r"[.,!]?( two| three)?[.,!]?$", re.I)
FILLER = re.compile(r"^(okay|ok|yeah|yep|yes|thanks|noted|right|alright|so|and|um|uh)[.,!]?$", re.I)


def classify(calls):
    c = {"technical": 0, "motivational": 0, "tactical": 0, "other": 0}
    tagged, excl = [], 0
    for t, s in calls:
        st = s.strip()
        if FILLER.match(st):
            excl += 1
            continue
        if COUNT.match(st) or TACT.search(s):
            k = "tactical"
        elif TECH.search(s):
            k = "technical"
        elif MOTI.search(s):
            k = "motivational"
        else:
            k = "other"
        c[k] += 1
        tagged.append((t, s, k))
    return c, tagged, excl


def pct(c):
    n = sum(c.values())
    return tuple(100.0 * c[k] / n for k in ("technical", "motivational", "tactical", "other"))


print("=== elite control, de-duplicated ===")
tot = {"technical": 0, "motivational": 0, "tactical": 0, "other": 0}
tmin = 0.0
for f in sorted(glob.glob("elite/*.en.vtt")):
    w = vtt.parse_words(f)
    calls = vtt.segment(w)
    c, _, _ = classify(calls)
    n = sum(c.values())
    mins = (w[-1][0] - w[0][0]) / 60.0
    tmin += mins
    for k in tot:
        tot[k] += c[k]
    print("  %-14s %5.1f min %4d calls %6.1f/min  tech %4.1f%% moti %4.1f%% tact %4.1f%%"
          % (os.path.basename(f)[:12], mins, n, n / mins, *pct(c)[:3]))
N = sum(tot.values())
print("  %-14s %5.1f min %4d calls %6.1f/min  tech %4.1f%% moti %4.1f%% tact %4.1f%%  (other %.1f%%)"
      % ("POOLED", tmin, N, N / tmin, *pct(tot)))
print("  %-14s %5s     %4s       %6.1f/min  tech %4.1f%% moti %4.1f%% tact %4.1f%%"
      % ("NUGENT", "-", "-", 32.0, 40.4, 38.6, 21.0))

print()
print("=== masters race, de-duplicated ===")
w = vtt.parse_words("caps.en.vtt")
calls = vtt.segment(w)
c, tagged, excl = classify(calls)
mins = (w[-1][0] - w[0][0]) / 60.0
n = sum(c.values())
print("  %d calls over %.1f min = %.1f/min   tech %.1f%% moti %.1f%% tact %.1f%% other %.1f%%"
      % (n, mins, n / mins, *pct(c)))
json.dump([[t, s, k] for t, s, k in tagged], open("tagged.json", "w"))
print("  (was 595 calls, 33.7/min, tech 30.6%% moti 15.1%% tact 33.5%% before de-duplication)")
