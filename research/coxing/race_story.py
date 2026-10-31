"""What one race's data and its coxswain's recording show: pacing, pushes, and how the calling compares.

    .venv/Scripts/python.exe research/coxing/race_story.py race_07 --compare race_06

Reads the coxing study's session (sessions/<id>/: boat.csv, course.csv, phrases) and the coded episodes the
experiment writes (experiment/episodes.csv). Three parts, written to data/local/coxing/experiment/story_<id>.md:

1. **Pacing** by 500 m: split, rate, metres per stroke, and the course's turn rate (bends) in each segment.
2. **Pushes**: every called "take N" / "give me N" / "N to the line", from the phrase that sets it up; the boat's speed
   over the next 10 strokes against the 10 before (mm/s), the rate change, and the same difference at every other
   stroke of the race as the yardstick (a push is judged against how much the boat varies anyway, not against zero).
   Bends confound this (the boat slows in a turn); the turn rate in each window is shown.
3. **Calling profile** against a comparison session (another coxswain on the same course and day): calls per minute,
   target shares, stage, appraisal, arousal, words per call, information share.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SDIR = os.path.join(ROOT, "data", "local", "coxing", "sessions")
EXP = os.path.join(ROOT, "data", "local", "coxing", "experiment")
PUSH = re.compile(r"\b(take|give me)\s+(a\s+)?(big\s+)?(\d+|five|ten|twenty|three)\b|\blast\s+(10|ten)\b|\b(10|ten)\s+to\s+the\s+line", re.I)


def mmss(s):
    return "%d:%04.1f" % divmod(s, 60) if np.isfinite(s) else "–"


def load(sid):
    b = pd.read_csv(os.path.join(SDIR, sid, "boat.csv"))
    c = pd.read_csv(os.path.join(SDIR, sid, "course.csv"))
    b["turn"] = c["turn"].to_numpy()
    ph = json.load(open(os.path.join(SDIR, sid, "phrases.json"), encoding="utf-8"))
    return b, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    b, ph = load(a.sid)
    L = ["# Race story: %s\n" % a.sid]
    # 1. pacing
    L.append("## Pacing by 500 m\n\n| segment | split | rate | m/stroke | turn (deg/s, median) |\n|---|---|---|---|---|")
    edges = list(range(0, int(b.distance.max()) + 500, 500))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (b.distance >= lo) & (b.distance < hi)
        if m.sum() < 5:
            continue
        seg = b[m]
        dt = seg.t.iloc[-1] - seg.t.iloc[0]
        dd = seg.distance.iloc[-1] - seg.distance.iloc[0]
        L.append("| %d–%d m | %s | %.1f | %.2f | %.2f |" % (lo, min(hi, int(b.distance.max())), mmss(500 * dt / max(dd, 1)),
                                                        seg.rate.mean(), dd / max(len(seg) - 1, 1), seg.turn.median()))
    total = b.t.iloc[-1] - b.t.iloc[0]
    L.append("\nWhole race: %s for %.0f m (average split %s), mean rate %.1f.\n" % (
        mmss(total), b.distance.iloc[-1] - b.distance.iloc[0], mmss(500 * total / (b.distance.iloc[-1] - b.distance.iloc[0])), b.rate.mean()))
    # 2. pushes
    t = b.t.to_numpy()
    sp = b.speed.to_numpy()
    rt = b.rate.to_numpy()
    tn = b.turn.to_numpy()

    def diff_at(j):
        if j < 10 or j + 10 >= len(sp):
            return None
        return (sp[j:j + 10].mean() - sp[j - 10:j].mean(), rt[j:j + 10].mean() - rt[j - 10:j].mean(),
                tn[j - 10:j].mean(), tn[j:j + 10].mean())
    base = np.array([d[0] for d in (diff_at(j) for j in range(len(sp))) if d is not None])
    pushes, last = [], -1e9
    for p in ph:
        if PUSH.search(p["text"]) and p["t"] - last > 20 and t[0] <= p["t"] <= t[-1]:
            last = p["t"]
            j = int(np.searchsorted(t, p["t"]))
            j = min(j + 1, len(t) - 1)                     # the push begins about a stroke after the set-up call
            d = diff_at(j)
            if d:
                pushes.append(dict(t=p["t"], text=p["text"], dist=float(b.distance.iloc[j]), dv=d[0], dr=d[1],
                                   turn_before=d[2], turn_after=d[3], pct=float((base < d[0]).mean())))
    L.append("## Pushes (next 10 strokes against the 10 before)\n")
    L.append("Yardstick: at every stroke of this race the same difference has sd %.0f mm/s (mean %+.0f). A push's "
             "percentile says where it falls among them.\n" % (1000 * base.std(), 1000 * base.mean()))
    L.append("| time | at | call | speed change | rate change | percentile | turn before → after |\n|---|---|---|---|---|---|---|")
    for x in pushes:
        L.append("| %s | %.0f m | %s | %+.0f mm/s | %+.1f spm | %.0f%% | %.2f → %.2f |" % (
            mmss(x["t"]), x["dist"], x["text"][:48], 1000 * x["dv"], x["dr"], 100 * x["pct"], x["turn_before"], x["turn_after"]))
    if pushes:
        dv = np.array([x["dv"] for x in pushes])
        L.append("\nMean over %d pushes: %+.0f mm/s (yardstick mean %+.0f, sd %.0f; a mean of %d random strokes has sd %.0f)."
                 % (len(dv), 1000 * dv.mean(), 1000 * base.mean(), 1000 * base.std(), len(dv), 1000 * base.std() / np.sqrt(len(dv))))
    # 3. calling profile
    ep = pd.read_csv(os.path.join(EXP, "episodes.csv"))
    sids = [a.sid] + ([a.compare] if a.compare else [])
    L.append("\n## Calling profile\n\n| | " + " | ".join(sids) + " |\n|---|" + "---|" * len(sids))

    def prof(sid):
        e = ep[(ep.session == sid) & (ep.target != "count")]
        bb, _ = load(sid)
        minutes = (bb.t.iloc[-1] - bb.t.iloc[0]) / 60
        out = {"call episodes": "%d" % len(e), "calls per minute": "%.1f" % (len(e) / minutes),
               "median words per call": "%d" % e.words.median()}
        for tg in ("power", "rate", "together", "catch", "finish", "body", "set", "length", "steer", "info", "tactic", "morale"):
            out["target: " + tg] = "%.0f%%" % (100 * (e.target == tg).mean())
        for f, lv in (("stage", "dec"), ("stage", "ass"), ("stage", "inf"), ("appraisal", "chal"), ("appraisal", "thr"),
                      ("arousal", "up"), ("arousal", "down"), ("level", "self"), ("form", "analog"), ("discrep", "y"), ("ref", "y")):
            out["%s: %s" % (f, lv)] = "%.0f%%" % (100 * (e[f] == lv).mean())
        return out
    P = [prof(s) for s in sids]
    for k in P[0]:
        L.append("| %s | %s |" % (k, " | ".join(p[k] for p in P)))
    out = os.path.join(EXP, "story_%s.md" % a.sid)
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
