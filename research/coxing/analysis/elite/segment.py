"""Calibrate the segmenter so its unit matches what human coders count as a call.

The sentence-boundary segmenter returns 9.0 calls/min on elite recordings where Nugent
et al. report 32. A coxswain's speech is mostly short imperatives -- "legs", "now",
"yes", "squeeze" -- which a person codes individually and a sentence parser swallows
whole. The unit, not the classifier, is what fails first.

Calibration target is external and published: 32 calls/min on elite material. We tune
ONLY the segmenter against that number, on the elite set, and then carry the tuned
segmenter to the masters race unchanged.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import glob
import os
import re

import numpy as np


def parse_words(path):
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
    return out


def segment(words, max_words, gap):
    """Break on punctuation, on a pause of `gap` seconds, or after `max_words`."""
    calls, cur, t0, last = [], [], None, None
    for t, w in words:
        if t0 is None:
            t0 = t
        if last is not None and (t - last) >= gap and cur:
            calls.append((t0, " ".join(cur)))
            cur, t0 = [], t
        cur.append(w)
        last = t
        if w.endswith((".", "?", "!", ",", ";", ":")) or len(cur) >= max_words:
            calls.append((t0, " ".join(cur)))
            cur, t0 = [], None
    if cur:
        calls.append((t0, " ".join(cur)))
    return calls


files = sorted(glob.glob("*.en.vtt"))
data = {os.path.basename(f).split(".")[0]: parse_words(f) for f in files}
spans = {k: (v[-1][0] - v[0][0]) / 60.0 for k, v in data.items()}

print("target: 32 calls/min (Nugent et al., human coding of elite recordings)")
print()
print("%-10s %-8s %10s %12s" % ("max words", "pause s", "elite c/min", "vs target"))
best = None
for mw in (3, 4, 5, 6, 8, 14):
    for gap in (0.6, 0.8, 1.0, 1.5, 99.0):
        tot = sum(len(segment(data[k], mw, gap)) for k in data)
        rate = tot / sum(spans.values())
        err = abs(rate - 32.0)
        if best is None or err < best[0]:
            best = (err, mw, gap, rate)
        print("%-10s %-8s %10.1f %+12.1f" % (mw, "none" if gap > 50 else gap, rate, rate - 32.0))
_, MW, GAP, RATE = best
print()
print("chosen: break after %d words or a %.1f s pause -> %.1f calls/min on elite"
      % (MW, GAP, RATE))
print()
print("per recording at that setting:")
for k in sorted(data):
    n = len(segment(data[k], MW, GAP))
    print("   %-14s %5.1f min %5d calls %7.1f /min" % (k, spans[k], n, n / spans[k]))
open("seg_params.txt", "w").write("%d %.1f\n" % (MW, GAP))
