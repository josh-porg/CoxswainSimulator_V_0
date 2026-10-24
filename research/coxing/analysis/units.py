"""A comparable unit of analysis across corpora with different caption conventions.

The masters transcript carries 26.2 sentence-end marks per 100 words; the elite
transcripts carry 0.0-0.6. Sentence segmentation therefore produces units of completely
different size on the two corpora, and every cross-corpus count or share computed that
way is an artefact of caption punctuation rather than of coxing.

Two measures survive that asymmetry:

  WORDS PER MINUTE  - punctuation-free, directly comparable, and a reasonable proxy for
                      how much a coxswain says.
  FIXED-WIDTH WINDOWS - classify every consecutive block of N words. The unit is then
                      identical across corpora by construction. Shares computed this way
                      measure what fraction of SPEECH is technical/motivational/tactical,
                      which is the comparable quantity, rather than what fraction of
                      "calls" is - since "call" is not reproducible from captions.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import glob
import os
import re
import sys

sys.path.insert(0, "elite")
import vtt


def windows(words, n=8, step=8):
    out = []
    for i in range(0, max(len(words) - n + 1, 1), step):
        chunk = words[i:i + n]
        if not chunk:
            continue
        out.append((chunk[0][0], " ".join(w for _, w in chunk)))
    return out


def corpora():
    out = {"masters": vtt.parse_words("caps.en.vtt")}
    for f in sorted(glob.glob("elite/*.en.vtt")):
        out[os.path.basename(f)[:11]] = vtt.parse_words(f)
    return out


if __name__ == "__main__":
    print("%-14s %7s %10s %9s" % ("corpus", "words", "words/min", "windows"))
    for k, w in corpora().items():
        mins = (w[-1][0] - w[0][0]) / 60.0
        print("%-14s %7d %10.1f %9d" % (k, len(w), len(w) / mins, len(windows(w))))
