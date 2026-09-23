"""Call encoders: turn a session's words into per-step call channels.

An encoder maps a list of (time, word) to a fixed vocabulary of channels. Two are
provided and a third can be registered:

  keyword  fully automatic; the eight functional categories used in the paper, plus
           NUMBER (bare stroke counts and countdowns) and ANY (the coxswain is speaking)
  labels   reads per-utterance labels from a session's labels.json -- for speech-act or
           pragmatic codings produced by a human or model coder

New data works with `keyword` immediately. Richer codings plug in through `labels` by
adding the file; nothing else changes.
"""
from captions import normalise

# Functional categories. Every term occurs in the development corpus; each belongs to one
# category only, ambiguous terms assigned to their dominant sense.
FUNCTIONAL = {
    "power":    "big drive hard lean legs more power press pressure push send squeeze strength strong",
    "catch":    "catch catches direct entry front quick together",
    "finish":   "accelerate back draw finish finishes hands release tap through",
    "length":   "full length lengthen long longer out reach stretch",
    "ratio":    "breathe control controlled easy patience ratio recovery relax rhythm slide smooth swing",
    "rate":     "build rate rating settle shift sit stroke strokes time timing",
    "motivat":  "believe brilliant come dig fight go good great lovely nice now perfect well yes",
    "tactical": "bridge gone left meters metres open position seat seats their them walking water",
}
_NUMBER_WORDS = set("one two three four five six seven eight nine ten".split())

KEYWORD_CHANNELS = list(FUNCTIONAL) + ["number", "any"]
_LOOKUP = {}
for _cat, _terms in FUNCTIONAL.items():
    for _w in _terms.split():
        _LOOKUP.setdefault(_w, _cat)


def _is_number(w):
    return w.isdigit() or w in _NUMBER_WORDS


def keyword_events(words):
    """(time, channel) for every word that falls in a channel. ANY marks all speech."""
    ev = []
    for t, raw in words:
        w = normalise(raw)
        if not w:
            continue
        ev.append((t, "any"))
        if _is_number(w):
            ev.append((t, "number"))
        cat = _LOOKUP.get(w)
        if cat:
            ev.append((t, cat))
    return ev


def label_events(labels, field):
    """(time, channel) from a labels list [{t: ..., <field>: ...}, ...]."""
    return [(float(d["t"]), str(d[field])) for d in labels if d.get(field) not in (None, "")]


ENCODERS = {"keyword": KEYWORD_CHANNELS}
