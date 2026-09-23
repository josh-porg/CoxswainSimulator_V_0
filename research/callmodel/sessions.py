"""The session format: the plug-and-play contract between data and model.

A session is one recorded piece -- a race or a practice -- stored as a folder:

    <root>/sessions/<id>/
        meta.json     required  {"id", "title", "kind", "boat_class", "date", "source"}
        words.jsonl   optional  one {"t": seconds, "w": word} per line   (the call stream)
        boat.csv      optional  header t,speed,rate  (seconds, m/s, spm)  (the boat stream)
        labels.json   optional  [{"t": seconds, "<field>": label}, ...]  (richer call codes)

Any subset of the optional files is valid. A transcript with no boat data trains the call
model; a cox-box export with no audio trains the boat model; a session with both is
synchronised and is the only kind that can inform the coupling between them.

Every session is placed on one common time grid of DT seconds. Missing data are masked,
never filled: a grid cell with no boat reading carries a zero mask, and losses and
metrics are computed only where the mask is set.
"""
import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np

from encode import ENCODERS, KEYWORD_CHANNELS, keyword_events, label_events

DT = 2.0                       # seconds per step; one stroke at ~30 spm
BOAT_CHANNELS = ["speed", "rate"]


@dataclass
class Grid:
    """One session on the common time grid."""
    sid: str
    kind: str
    t: np.ndarray                 # (T,) cell start times
    boat: np.ndarray              # (T, 2) z-scored within session; 0 where missing
    boat_mask: np.ndarray         # (T, 2) 1 where observed
    calls: np.ndarray             # (T, K) 1 if the channel occurred in the cell
    call_mask: np.ndarray         # (T,)   1 where the call stream covers the cell
    channels: list = field(default_factory=list)

    @property
    def synchronised(self):
        return bool(self.boat_mask.any() and self.call_mask.any())

    def __len__(self):
        return len(self.t)


class Session:
    def __init__(self, path):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8"))
        self.sid = self.meta["id"]

    def _file(self, name):
        p = os.path.join(self.path, name)
        return p if os.path.exists(p) else None

    def words(self):
        p = self._file("words.jsonl")
        if not p:
            return []
        return [(float(d["t"]), d["w"]) for d in map(json.loads, open(p, encoding="utf-8"))]

    def boat(self):
        p = self._file("boat.csv")
        if not p:
            return None
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        t = np.array([float(r["t"]) for r in rows])
        cols = []
        for c in BOAT_CHANNELS:
            cols.append(np.array([float(r[c]) if r.get(c) not in (None, "", "nan") else np.nan
                                  for r in rows]))
        return t, np.column_stack(cols)

    def labels(self):
        p = self._file("labels.json")
        return json.load(open(p, encoding="utf-8")) if p else None

    def to_grid(self, encoder="keyword", label_field=None, dt=DT):
        words = self.words()
        boat = self.boat()
        if encoder == "keyword":
            channels = KEYWORD_CHANNELS
            events = keyword_events(words)
            covered = [t for t, _ in words]
        elif encoder == "labels":
            lab = self.labels()
            if lab is None:
                raise ValueError("session %s has no labels.json" % self.sid)
            events = label_events(lab, label_field)
            channels = sorted({c for _, c in events})
            covered = [t for t, _ in events]
        else:
            raise ValueError("unknown encoder %r" % encoder)

        spans = []
        if covered:
            spans += [min(covered), max(covered)]
        if boat is not None:
            spans += [float(np.nanmin(boat[0])), float(np.nanmax(boat[0]))]
        t0 = float(self.meta.get("t0", min(spans)))
        t1 = float(self.meta.get("t1", max(spans)))
        T = int(np.floor((t1 - t0) / dt)) + 1
        tgrid = t0 + dt * np.arange(T)

        calls = np.zeros((T, len(channels)))
        cidx = {c: i for i, c in enumerate(channels)}
        for t, c in events:
            i = int((t - t0) // dt)
            if 0 <= i < T and c in cidx:
                calls[i, cidx[c]] = 1.0
        call_mask = np.zeros(T)
        if covered:
            a = int(max(0, (min(covered) - t0) // dt))
            b = int(min(T - 1, (max(covered) - t0) // dt))
            call_mask[a:b + 1] = 1.0

        bvals = np.zeros((T, len(BOAT_CHANNELS)))
        bmask = np.zeros((T, len(BOAT_CHANNELS)))
        if boat is not None:
            bt, bv = boat
            acc = np.zeros((T, len(BOAT_CHANNELS)))
            cnt = np.zeros((T, len(BOAT_CHANNELS)))
            for tt, row in zip(bt, bv):
                i = int((tt - t0) // dt)
                if 0 <= i < T:
                    ok = np.isfinite(row)
                    acc[i, ok] += row[ok]
                    cnt[i, ok] += 1
            has = cnt > 0
            bvals[has] = acc[has] / cnt[has]
            bmask[has] = 1.0
            # z-score within session on observed cells only (a per-session constant, so it
            # removes crew and boat-class level differences without leaking local signal)
            for j in range(len(BOAT_CHANNELS)):
                obs = bmask[:, j] > 0
                if obs.sum() > 2:
                    mu, sd = bvals[obs, j].mean(), bvals[obs, j].std() + 1e-9
                    bvals[obs, j] = (bvals[obs, j] - mu) / sd
                bvals[~obs, j] = 0.0

        return Grid(self.sid, self.meta.get("kind", "unknown"), tgrid, bvals, bmask,
                    calls, call_mask, list(channels))


def list_sessions(root):
    base = os.path.join(root, "sessions")
    if not os.path.isdir(base):
        return []
    return [Session(os.path.join(base, d)) for d in sorted(os.listdir(base))
            if os.path.exists(os.path.join(base, d, "meta.json"))]
