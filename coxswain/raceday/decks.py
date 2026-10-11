r"""Checks on the briefing decks (R6-R8 of docs/RACEDAY.md).

The rowers' deck carries the plan and the calls and **no probabilities, odds or target times**
(discovery round 1: the rowers do not get the odds of medalling or requalifying, and do not get splits
unless the coxswain chooses to share one). :func:`rowers_violations` finds anything in a slide's text,
speaker notes included, that would break that: a percentage, a word of chance, a requalifying or medal
target, or a race time or split (``m:ss``).
"""
from __future__ import annotations

import glob
import html
import os
import re
from typing import List, Tuple

FORBIDDEN = [
    (re.compile(r"\d\s*%"), "a percentage"),
    (re.compile(r"\b(odds|chances?|probab\w*|likel(y|ihood)|unlikely)\b", re.I), "a word of chance"),
    (re.compile(r"\b(requalif\w*|medal\w*)\b", re.I), "a goal the rowers are not briefed on"),
    (re.compile(r"\b\d{1,2}:\d{2}(\.\d)?\b"), "a time or a split"),
]


def text_of(markup: str) -> str:
    """The visible text and speaker notes of a slide, without tags and attributes."""
    return html.unescape(re.sub(r"<[^>]+>", " ", markup))


def rowers_violations(text: str) -> List[Tuple[str, str]]:
    out = []
    for pattern, what in FORBIDDEN:
        for m in pattern.finditer(text):
            out.append((what, text[max(0, m.start() - 30):m.end() + 30].strip()))
    return out


def check_deck(folder: str) -> List[Tuple[str, str, str]]:
    """Every violation in a deck's slides: (slide, what, context)."""
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "project", "slides", "*.html"))):
        for what, ctx in rowers_violations(text_of(open(path, encoding="utf-8").read())):
            out.append((os.path.basename(path), what, ctx))
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(errors="replace")
    bad = check_deck(sys.argv[1])
    for slide, what, ctx in bad:
        print("%s: %s: ...%s..." % (slide, what, ctx))
    print("OK: no probabilities, odds, targets or times" if not bad else "%d problem(s)" % len(bad))
    sys.exit(1 if bad else 0)
