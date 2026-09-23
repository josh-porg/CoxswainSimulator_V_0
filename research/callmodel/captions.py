"""Parse YouTube rolling auto-captions into (time, word) pairs with per-word timing.

Rolling captions repeat themselves: each cue carries the tail of the previous line plus
new words, so a naive parse counts most words two or three times. Words are recovered
by suffix-prefix overlap -- for each cue, the longest suffix of the transcript so far
that is a prefix of the cue is skipped and only the remainder appended.

Timing comes from the inline tags YouTube writes inside a cue,
    5<00:00:06.080><c> 6</c>
where the first word takes the cue's start time and each tagged word takes its own
stamp. That resolves individual words rather than whole 2-3 s cues.
"""
import re

_CUE = re.compile(r"(\d\d:\d\d:\d\d\.\d\d\d) --> (\d\d:\d\d:\d\d\.\d\d\d)[^\n]*\n(.*?)(?=\n\n|\Z)", re.S)
_TAG = re.compile(r"<(\d\d:\d\d:\d\d\.\d\d\d)><c>(.*?)</c>")


def _sec(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _timed_line(line, t0):
    """Split one caption line into (time, word) using its inline stamps."""
    out = []
    first = line.split("<", 1)[0]
    for w in first.replace("&nbsp;", " ").split():
        out.append((t0, w))
    for ts, body in _TAG.findall(line):
        for w in re.sub(r"<[^>]+>", " ", body).replace("&nbsp;", " ").split():
            out.append((_sec(ts), w))
    return out


def parse(path, max_overlap=40):
    """Return a de-duplicated list of (seconds, word)."""
    txt = open(path, encoding="utf-8").read()
    words, times = [], []
    for start, _end, body in _CUE.findall(txt):
        lines = [l for l in body.split("\n") if l.strip()]
        if not lines:
            continue
        tagged = [l for l in lines if "<c>" in l]
        line = tagged[-1] if tagged else lines[-1]
        new = _timed_line(line, _sec(start))
        if not new:
            continue
        nw = [w for _, w in new]
        k = min(len(words), len(nw), max_overlap)
        cut = 0
        while k > 0:
            if words[-k:] == nw[:k]:
                cut = k
                break
            k -= 1
        for t, w in new[cut:]:
            words.append(w)
            times.append(t)
    return list(zip(times, words))


def normalise(word):
    """Lower-case and strip punctuation, keeping apostrophes and digits."""
    return "".join(c for c in word.lower() if c.isalnum() or c == "'")
