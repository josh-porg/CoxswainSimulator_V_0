"""Do the data calls move the boat? And what happens when the cox is silent?"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import json
import re

import numpy as np

D = np.load("race.npz")
t, v, sr, dps = D["t"], D["v"], D["sr"], D["dps"]
OFF = 1.0
tag = json.load(open("tagged.json"))

SPLIT = re.compile(r"\b(2[0-3]\d)\b")          # 2:18 - 2:39 spoken as 218..239
RATE = re.compile(r"\b(a |at a |at the )?(2[4-9]|3[0-4])\b")


def window(rt, n=10):
    i = int(np.argmin(np.abs(t - rt)))
    if i < n or i > len(t) - n - 1:
        return None
    return (v[i:i+n].mean() - v[i-n:i].mean(),
            sr[i:i+n].mean() - sr[i-n:i].mean(),
            dps[i:i+n].mean() - dps[i-n:i].mean())


def summarise(name, times):
    rows = [window(x + OFF) for x in times]
    rows = [r for r in rows if r]
    if not rows:
        print("  %-28s none" % name)
        return
    a = np.array(rows)
    se = a.std(axis=0) / np.sqrt(len(a))
    print("  %-28s n=%3d   d speed %+.4f +- %.4f   d rate %+.2f   d m/str %+.3f"
          % (name, len(a), a[:, 0].mean(), se[0], a[:, 1].mean(), a[:, 2].mean()))


split_t = [x[0] for x in tag if SPLIT.search(x[1])]
rate_t = [x[0] for x in tag if RATE.search(x[1]) and not SPLIT.search(x[1])]
tech_t = [x[0] for x in tag if x[2] == "technical"]
moti_t = [x[0] for x in tag if x[2] == "motivational"]
allc = [x[0] for x in tag]

print("boat response in the 10 strokes after a call, against the 10 before:")
summarise("every call", allc)
summarise("SPLIT stated", split_t)
summarise("RATE stated", rate_t)
summarise("technical", tech_t)
summarise("motivational", moti_t)

# --- silence ---------------------------------------------------------------------
print()
print("=== silence ===")
times = np.array(sorted(allc))
gaps = np.diff(times)
print("  %d calls, median gap %.1f s, longest %.1f s" % (len(times), np.median(gaps), gaps.max()))
for thr in (6.0, 8.0, 10.0):
    idx = np.flatnonzero(gaps >= thr)
    if not len(idx):
        continue
    dv, dr, dl, tot = [], [], [], 0.0
    for i in idx:
        a, b = times[i] + OFF, times[i + 1] + OFF
        m = (t >= a) & (t <= b)
        base = (t >= a - 30) & (t < a)
        if m.sum() < 2 or base.sum() < 3:
            continue
        dv.append(v[m].mean() - v[base].mean())
        dr.append(sr[m].mean() - sr[base].mean())
        dl.append(dps[m].mean() - dps[base].mean())
        tot += b - a
    if dv:
        print("  gaps >= %4.1f s: n=%2d, %.0f s of silence total (%.1f%% of the race)"
              % (thr, len(dv), tot, 100 * tot / (t[-1] - t[0])))
        print("      vs the 30 s before: d speed %+.4f   d rate %+.2f   d m/stroke %+.3f"
              % (np.mean(dv), np.mean(dr), np.mean(dl)))
