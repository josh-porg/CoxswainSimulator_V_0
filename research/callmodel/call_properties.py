r"""Theory-derived call properties and the boat's response to them.

The review "What makes a call work?" derives the properties of a coxswain's call from control
theory, physiology, exercise neuroscience, psychology, education and communication science, and
proposes judging a call by the boat's response beyond its own momentum. This module turns hand
codes (one line per phrase, coded from the transcript only; see the codebook beside the data)
into call episodes with those properties, adds the relational properties computed from the boat
record, and estimates responses with a permutation null that keeps every session's call spacing.

Per session folder: ``phrases.json`` ([{i, t, t_end, text}]), ``boat.csv`` (t, speed, rate),
``meta.json``. Codes: ``codes_<session>.txt`` with lines
``i target stage level action function focus ref discrep appraisal form arousal``.
Phrases without a code line are counts (bare numbers) and become ``target = count``.

Responses are read from a precomputed matrix, so a permutation is an index shift: for every
stroke j, ``R[j, h-1] = y[j+h-1] - forecast_h(y[j-4:j])`` with an AR(4) fitted on the session.
"""
from __future__ import annotations

import csv
import json
import os
import re

import numpy as np

FIELDS = ["target", "stage", "level", "action", "function", "focus", "ref", "discrep",
          "appraisal", "form", "arousal"]
AR_P = 4
K_POST = 20
GAP = 3.0
COUNT_WORDS = set("one two three four five six seven eight nine ten eleven twelve twenty".split())


# ---------------------------------------------------------------- loading
def load_session(sdir):
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    rows = list(csv.DictReader(open(os.path.join(sdir, "boat.csv"))))
    t = np.array([float(r["t"]) for r in rows])
    sp = np.array([float(r["speed"]) for r in rows])
    ra = np.array([float(r["rate"]) for r in rows])
    phrases = json.load(open(os.path.join(sdir, "phrases.json"), encoding="utf-8"))
    return dict(meta=meta, t=t, speed=sp, rate=ra, dps=sp * 60.0 / np.maximum(ra, 1.0), phrases=phrases)


def load_codes(path):
    codes = {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) != 1 + len(FIELDS):
            raise ValueError("bad code line: %r" % line)
        codes[int(f[0])] = dict(zip(FIELDS, f[1:]))
    return codes


def code_of(i, codes):
    if i in codes:
        return codes[i]
    return dict(zip(FIELDS, ["count"] + ["-"] * (len(FIELDS) - 1)))


def normalise(text):
    """Lower case, letters only, count words and numbers removed: the wording of a cue."""
    w = [x for x in re.findall(r"[a-z']+", text.lower()) if x not in COUNT_WORDS]
    return " ".join(w)


# ---------------------------------------------------------------- episodes
def _union(codes_run):
    lead = codes_run[0]
    out = dict(lead)

    def first(field):
        for c in codes_run:
            if c[field] != "-":
                return c[field]
        return "-"

    for f in ("stage", "level", "function"):
        out[f] = lead[f] if lead[f] != "-" else first(f)
    for f in ("action", "ref", "discrep"):
        out[f] = "y" if any(c[f] == "y" for c in codes_run) else "n"
    ap = [c["appraisal"] for c in codes_run]
    out["appraisal"] = "thr" if "thr" in ap else ("chal" if "chal" in ap else "neu")
    fo = {c["focus"] for c in codes_run}
    out["focus"] = "mix" if {"int", "ext"} <= fo else ("int" if "int" in fo else ("ext" if "ext" in fo else "-"))
    fm = {c["form"] for c in codes_run}
    out["form"] = "analog" if "analog" in fm else ("expl" if "expl" in fm else "-")
    ar = {c["arousal"] for c in codes_run}
    out["arousal"] = "up" if "up" in ar else ("down" if "down" in ar else "-")
    return out


def episodes(phrases, codes, gap=GAP):
    """First phrase of each same-target run (next phrase within ``gap`` s), with the run's union
    of properties. ``other`` never starts or extends a run."""
    out, run = [], None
    for p in phrases:
        c = code_of(p["i"], codes)
        if c["target"] == "other":
            continue
        if run and c["target"] == run["target"] and p["t"] - run["t_last"] <= gap:
            run["codes"].append(c)
            run["texts"].append(p["text"])
            run["t_last"] = max(p["t"], p.get("t_end") or p["t"])
            continue
        run = dict(i=p["i"], t=p["t"], target=c["target"], codes=[c], texts=[p["text"]],
                   t_last=max(p["t"], p.get("t_end") or p["t"]))
        out.append(run)
    for e in out:
        e.update(_union(e["codes"]))
        e["words"] = sum(len(x.split()) for x in e["texts"])
        e["clauses"] = len(e["texts"])
        e["lead_norm"] = normalise(e["texts"][0])
    return out


def add_familiarity(sessions_in_order, threshold=3):
    """An episode is established if its lead wording occurred >= ``threshold`` times before it,
    over all phrases of earlier sessions and earlier in its own session."""
    seen = {}
    for s in sessions_in_order:
        lead = {e["i"]: e for e in s.get("episodes", [])}
        for p in sorted(s["phrases"], key=lambda p: p["t"]):
            key = normalise(p["text"])
            if p["i"] in lead:
                lead[p["i"]]["established"] = bool(key) and seen.get(key, 0) >= threshold
            if key:
                seen[key] = seen.get(key, 0) + 1


# ---------------------------------------------------------------- boat responses
def ar_fit(y, p=AR_P):
    X = np.column_stack([np.ones(len(y) - p)] + [y[p - j - 1:len(y) - j - 1] for j in range(p)])
    beta, *_ = np.linalg.lstsq(X, y[p:], rcond=None)
    return beta


def residual_matrix(y, beta, k=K_POST, p=AR_P):
    """R[j, h-1]: y[j+h-1] minus the h-step AR forecast from y[j-p:j]; NaN where undefined."""
    n = len(y)
    R = np.full((n, k), np.nan)
    for j in range(p, n - k + 1):
        hist = list(y[j - p:j])
        for h in range(k):
            x = np.r_[1.0, hist[::-1][:p]]
            f = float(x @ beta)
            R[j, h] = y[j + h] - f
            hist.append(f)
    return R


def prepare(session):
    """Residual matrices for speed, rate and distance per stroke, and the valid stroke range."""
    out = {}
    for ch in ("speed", "rate", "dps"):
        y = session[ch]
        out[ch] = residual_matrix(y, ar_fit(y))
    d = np.abs(np.diff(session["speed"], prepend=session["speed"][0]))
    out["absdiff"] = d
    n = len(session["t"])
    session["R"] = out
    session["valid"] = (AR_P + 5, n - K_POST)        # 5 extra strokes for the variance window
    one = out["speed"][:, 0]
    sd = np.nanstd(one)
    session["adverse"] = np.flatnonzero(one < -2 * sd)
    return session


def stroke_index(t, times):
    return np.searchsorted(t, times)


def contingency(session, j):
    """Measured speed change before the call: mean of the 2 strokes before minus the 4 before those."""
    sp = session["speed"]
    if j < 6:
        return np.nan
    return sp[j - 2:j].mean() - sp[j - 6:j - 2].mean()


def variance_change(session, j):
    d = session["R"]["absdiff"]
    return d[j:j + 5].mean() - d[j - 5:j].mean()


# ---------------------------------------------------------------- permutation inference
def shifted_indices(rng, js_by_session, valid_by_session, local=False):
    """Shift every session's episode strokes together, circularly within its valid range. Global:
    by 15-85% of the range. Local: by 10-40 strokes either way, so each call stays near its place in
    the piece (controls for trends with time in the piece, e.g. the sprint)."""
    out = []
    for js, (lo, hi) in zip(js_by_session, valid_by_session):
        span = hi - lo
        if local:
            s = int(rng.integers(10, 41)) * (1 if rng.random() < 0.5 else -1)
        else:
            s = rng.integers(int(0.15 * span), int(0.85 * span))
        out.append(lo + (js - lo + s) % span)
    return out


def gather(sessions, js_by_session, channel, h0, h1):
    """Mean residual over horizons h0..h1 (1-based, inclusive) for each episode, sessions stacked."""
    vals = []
    for s, js in zip(sessions, js_by_session):
        R = s["R"][channel]
        vals.append(np.nanmean(R[js, h0 - 1:h1], axis=1))
    return np.concatenate(vals)


def gather_var(sessions, js_by_session):
    vals = []
    for s, js in zip(sessions, js_by_session):
        d = s["R"]["absdiff"]
        vals.append(np.array([d[j:j + 5].mean() - d[j - 5:j].mean() for j in js]))
    return np.concatenate(vals)


def permutation_test(stat, sessions, js_by_session, n_perm=2000, seed=0, local=False):
    """``stat(js_by_session) -> array`` observed and under shifts; two-sided p per entry. A
    property defined from the boat's state at the call must be recomputed inside ``stat`` from
    the (shifted) strokes it is given, so the null keeps the selection it implies."""
    rng = np.random.default_rng(seed)
    valid = [s["valid"] for s in sessions]
    obs = np.asarray(stat(js_by_session), float)
    null = np.array([stat(shifted_indices(rng, js_by_session, valid, local)) for _ in range(n_perm)])
    mu = null.mean(0)
    p = (np.sum(np.abs(null - mu) >= np.abs(obs - mu), axis=0) + 1) / (n_perm + 1)
    return obs, mu, null.std(0), p


def ols_coefs(X, y):
    ok = ~np.isnan(y)
    b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return b


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, (len(p) - rank) * p[k])
        adj[k] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- delivery (audio)
def delivery(wav, sr, t_video, t_end_video, words):
    """Loudness (RMS dB), median F0 (Hz, autocorrelation, voiced frames) and speech rate (words/s)
    for one phrase window."""
    a = int(max(0, t_video * sr))
    b = int(min(len(wav), (max(t_end_video, t_video + 0.4) + 0.2) * sr))
    x = wav[a:b].astype(float)
    if len(x) < sr // 10:
        return np.nan, np.nan, np.nan
    rms = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
    f0s = []
    fl = int(0.04 * sr)
    for k in range(0, len(x) - fl, fl // 2):
        fr = x[k:k + fl] - x[k:k + fl].mean()
        e = np.dot(fr, fr)
        if e <= 0:
            continue
        ac = np.correlate(fr, fr, "full")[fl - 1:]
        lo, hi = int(sr / 500), int(sr / 80)
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] > 0.5:
            f0s.append(sr / lag)
    f0 = float(np.median(f0s)) if f0s else np.nan
    dur = max(t_end_video - t_video, 0.0) + 0.3
    return rms, f0, words / dur
