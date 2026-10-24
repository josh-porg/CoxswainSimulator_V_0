"""Correct de-duplication of YouTube rolling auto-captions.

Rolling captions repeat: each cue carries the tail of the previous line plus a few new
words, so a naive parse counts most words two or three times. The earlier "drop a word
seen in the last 3 s" heuristic removes some of it and leaves plenty -- enough to
corrupt call counts and any classification built on them.

The correct rule is suffix-prefix overlap: for each new cue, find the longest suffix of
what we already have that is a prefix of the cue, and append only what follows it.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import re


def _words(body):
    return re.sub(r"<[^>]+>", " ", body).replace("&nbsp;", " ").split()


def parse_words(path, max_overlap=40):
    txt = open(path, encoding="utf-8").read()
    cues = re.findall(r"(\d\d:\d\d:\d\d\.\d\d\d) --> .*?\n(.*?)(?=\n\n|\Z)", txt, re.S)
    words, times = [], []
    for ts, body in cues:
        h, m, s = ts.split(":")
        t0 = int(h) * 3600 + int(m) * 60 + float(s)
        new = _words(body)
        if not new:
            continue
        k = min(len(words), len(new), max_overlap)
        cut = 0
        while k > 0:
            if words[-k:] == new[:k]:
                cut = k
                break
            k -= 1
        add = new[cut:]
        if not add:
            continue
        n = len(add)
        for i, w in enumerate(add):
            words.append(w)
            times.append(t0 + 0.001 * i)
    return list(zip(times, words))


def segment(words, max_words=14, gap=99.0):
    calls, cur, t0, last = [], [], None, None
    for t, w in words:
        if t0 is None:
            t0 = t
        if last is not None and (t - last) >= gap and cur:
            calls.append((t0, " ".join(cur)))
            cur, t0 = [], t
        cur.append(w)
        last = t
        if w.endswith((".", "?", "!")) or len(cur) >= max_words:
            calls.append((t0, " ".join(cur)))
            cur, t0 = [], None
    if cur:
        calls.append((t0, " ".join(cur)))
    return [(t, s) for t, s in calls if s.strip()]
