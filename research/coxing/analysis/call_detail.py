"""Which calls worked, broken down inside each of Nugent's three themes."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import re
from collections import defaultdict

import numpy as np

D = np.load("race.npz")
t, v, sr, dps = D["t"], D["v"], D["sr"], D["dps"]
OFF, N = 1.0, 10
tag = json.load(open("tagged.json"))

SUB = {
  "technical": [
    ("ratio / rhythm",      r"ratio|rhythm"),
    ("length / send",       r"\blong\b|length|send|reach|distance per|squeeze"),
    ("catch / front end",   r"catch|front end|attack|direct|sharp|blade|prep|\bin\b\W*$"),
    ("finish / release",    r"finish|release|puddle|hook"),
    ("body / swing",        r"bod(y|ies)|swing|sit up|chest|shoulder|seat|connect|stretcher|stand"),
    ("suspend / hang",      r"suspend|hang|handle|press|fill"),
    ("together / clean",    r"together|complete|clean|tidy|one boat|sequenc"),
    ("breathe / loose",     r"breathe|loose|relax|calm"),
  ],
  "tactical": [
    ("split stated",        r"\b2[0-3]\d\b"),
    ("rate stated",         r"\b(2[4-9]|3[0-4])\b"),
    ("distance / position", r"\b\d{3,4}\s*m\b|metre|meter|halfway|quarter|finish line"),
    ("in two / take a N",   r"in two|take (a )?(10|ten|five|5)|that's one|one, two"),
    ("other crews",         r"boat|lengths|passing|pass|clear|walk|yield|them\b"),
    ("steering",            r"starboard|port\b|corner|turn|lane|course"),
  ],
  "motivational": [
    ("praise",              r"\bgood\b|\byes\b|\bnice\b|excellent|that's it|love|like it|doing it"),
    ("exhortation",         r"dig deep|find it|more\b|come on|go\b|fight|push|hurt|burn|100|110"),
    ("identity / shared",   r"remember|promise|commit|boat ?mates|teammates|together we|we are|we have"),
    ("named rower",         r"sheila|alex|marilyn|leia|francis"),
  ],
}


def resp(rt):
    i = int(np.argmin(np.abs(t - rt)))
    if i < N or i > len(t) - N - 1:
        return None
    return (v[i:i+N].mean() - v[i-N:i].mean(), dps[i:i+N].mean() - dps[i-N:i].mean(),
            sr[i:i+N].mean() - sr[i-N:i].mean())


base = np.mean([resp(x[0] + OFF)[0] for x in tag if resp(x[0] + OFF)])
print("baseline: the average call is worth %+.4f m/s over the next 10 strokes" % base)
print()
for theme, subs in SUB.items():
    rows = [x for x in tag if x[2] == theme]
    print("=== %s (%d calls) ===" % (theme.upper(), len(rows)))
    print("  %-22s %5s %11s %11s %9s" % ("sub-type", "n", "d speed", "vs baseline", "d m/str"))
    out = []
    for name, pat in subs:
        rx = re.compile(pat, re.I)
        hits = [resp(x[0] + OFF) for x in rows if rx.search(x[1])]
        hits = [h for h in hits if h]
        if len(hits) < 3:
            continue
        a = np.array(hits)
        out.append((a[:, 0].mean() - base, name, len(a), a[:, 0].mean(), a[:, 1].mean()))
    for rel, name, n, dv, dl in sorted(out, reverse=True):
        mark = "  <-- best" if rel == max(o[0] for o in out) else ("  <-- worst" if rel == min(o[0] for o in out) else "")
        print("  %-22s %5d %+11.4f %+11.4f %+9.3f%s" % (name, n, dv, rel, dl, mark))
    print()
