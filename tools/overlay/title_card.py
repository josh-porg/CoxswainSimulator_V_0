"""The title card: the race's name and the lineup, shown for a few seconds before the racing.

Over a b-roll clip (retimed to fill exactly the card's time, then the race video follows), or, with
no b-roll, over the first seconds of the race video itself. Drawn in the HUD's SRA colours and
type, at the same 1920-wide reference as the HUD layers, then scaled to the frame.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image, ImageDraw

import coxbox_overlay as O

#: how a list of bare names is read
ORDERS = {"bow": "Bow first, cox last", "stroke": "Cox first, then stroke to bow"}
DEFAULT_SECONDS = 6.0
FADE_IN, FADE_OUT = 0.4, 0.7                    # s


@dataclass
class TitleCard:
    title: str = ""
    subtitle: str = ""
    lineup: list = field(default_factory=list)   # [(seat label, name)]
    seconds: float = DEFAULT_SECONDS
    broll: str | None = None

    @property
    def empty(self):
        return not (self.title.strip() or self.subtitle.strip() or self.lineup)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["lineup"] = [tuple(x) for x in d.get("lineup", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def seat_labels(n, coxed, order="bow"):
    """Seat names for ``n`` rowers (plus a cox), in the order the names were listed."""
    if n == 1:
        seats = [""]
    else:
        seats = ["Bow"] + [str(k) for k in range(2, n)] + ["Stroke"]          # bow to stroke
    if order == "stroke":
        return (["Cox"] if coxed else []) + seats[::-1]
    return seats + (["Cox"] if coxed else [])


_LABELLED = re.compile(r"^\s*(bow|stroke|coxswain|cox|seat\s*[1-8]|[1-8])\s*(?::|[-–.](?=\s))\s*(.+)$", re.I)


def parse_lineup(text, order="bow"):
    """``[(label, name)]`` from one name per line. A line may give its seat (``Stroke: Sam``,
    ``3 - Alex``); otherwise seats follow from the count: 9 names are an eight and its cox, 5 a four
    and its cox, 3 a pair and its cox, any other count rowers only."""
    lines = [ln.strip() for ln in str(text).replace(",", "\n").splitlines() if ln.strip()]
    if not lines:
        return []
    marked = [_LABELLED.match(ln) for ln in lines]
    if all(marked):
        out = []
        for m in marked:
            label = m.group(1).strip()
            label = {"bow": "Bow", "stroke": "Stroke", "cox": "Cox", "coxswain": "Cox"}.get(label.lower(),
                                                                                         re.sub(r"\D", "", label))
            out.append((label, m.group(2).strip()))
        return out
    names = [m.group(2).strip() if m else ln for m, ln in zip(marked, lines)]
    coxed = len(names) in (3, 5, 9)
    rowers = len(names) - (1 if coxed else 0)
    return list(zip(seat_labels(rowers, coxed, order), names))


def lineup_text(lineup):
    """Back to editable text, one ``Seat: Name`` per line."""
    return "\n".join(("%s: %s" % (s, n)) if s else n for s, n in lineup)


def fade(t, seconds):
    """The card's opacity ``t`` s into its ``seconds``."""
    if t < 0 or t >= seconds:
        return 0.0
    return float(min(1.0, t / FADE_IN, (seconds - t) / FADE_OUT))


def _fit(draw, s, size, width, weight, smallest):
    """The largest size <= ``size`` at which ``s`` fits in ``width`` px (not below ``smallest``)."""
    while size > smallest and draw.textlength(s, font=O.font(size, weight)) > width:
        size -= 2
    return size


def _wrap(draw, s, size, width, weight):
    """``s`` in at most two lines of ``width`` px at ``size``."""
    words = s.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=O.font(size, weight)) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines if len(lines) <= 2 else [lines[0], " ".join(lines[1:])]


def _draw(size, card: TitleCard):
    """The card at the reference size (RGBA): a navy scrim, the title block on the left with the
    HUD's red stripe, the lineup on the right."""
    W, H = size
    img = Image.new("RGBA", size, O.NAVY_DEEP + (165,))
    d = ImageDraw.Draw(img)
    has_lineup = bool(card.lineup)
    left = 150
    width = (1060 if has_lineup else W - 2 * left) - left
    title = card.title.strip()
    blocks = []                                   # (kind, text, size, weight, colour, line height)
    if title:
        tsize = 104
        lines = _wrap(d, title, tsize, width, "SemiBold")
        while tsize > 60 and any(d.textlength(ln, font=O.font(tsize)) > width for ln in lines):
            tsize -= 4
            lines = _wrap(d, title, tsize, width, "SemiBold")
        for ln in lines:
            blocks.append((ln, tsize, "SemiBold", O.WHITE, int(tsize * 1.08)))
    if card.subtitle.strip():
        ssize = _fit(d, card.subtitle.strip(), 42, width, "Regular", 26)
        blocks.append((card.subtitle.strip(), ssize, "Regular", O.LABEL, int(ssize * 1.5) + (14 if title else 0)))
    if blocks:
        total = sum(b[4] for b in blocks)
        y = (H - total) // 2
        top = y
        for s, sz, wt, col, lh in blocks:
            y += lh
            d.text((left + 36, y), s, font=O.font(sz, wt), fill=col + (255,), anchor="ls")
        d.rectangle((left, top + 6, left + 9, y + 10), fill=O.RED + (255,))
    if has_lineup:
        rows = card.lineup
        pitch = min(66, int((H - 220) / max(len(rows), 1)))
        name_size = min(48, int(pitch * 0.74))
        x_label, x_name, x_end = 1250, 1282, W - 90
        head = 30
        total = head + 24 + pitch * len(rows)
        y = (H - total) // 2 + head
        d.text((x_name, y), "LINEUP", font=O.font(26, "Regular"), fill=O.LABEL + (255,), anchor="ls")
        d.rectangle((x_name, y + 12, x_name + 64, y + 16), fill=O.RED + (255,))
        y += 24
        for label, name in rows:
            y += pitch
            ns = _fit(d, name, name_size, x_end - x_name, "Regular", 24)
            d.text((x_label, y), label, font=O.font(int(name_size * 0.66), "Light"),
                   fill=(O.RED_BRIGHT if label == "Cox" else O.LABEL) + (255,), anchor="rs")
            d.text((x_name, y), name, font=O.font(ns, "Regular"), fill=O.WHITE + (255,), anchor="ls")
    return img


def card_layer(size, card: TitleCard):
    """The card for a frame of ``size`` (RGBA), drawn at the HUD's reference width and scaled."""
    ref = (O.REF_WIDTH, int(round(O.REF_WIDTH * size[1] / size[0])))
    img = _draw(ref, card)
    return img if ref == tuple(size) else img.resize(tuple(size), Image.LANCZOS)


def with_opacity(img: Image.Image, a: float) -> Image.Image:
    if a >= 1.0:
        return img
    arr = np.asarray(img).copy()
    arr[..., 3] = (arr[..., 3].astype(np.float32) * a).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")
