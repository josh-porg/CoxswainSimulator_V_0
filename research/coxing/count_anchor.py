"""Count-anchored sync check: where each session's counts fall in the CoxBox stroke cycle, and the shift that
would put them on the catch.

    .venv/Scripts/python.exe research/coxing/count_anchor.py

The study's syncs (video stroke rhythm, called splits, the boat's sound) place the words on the CoxBox clock to
about half a stroke. Counts are said on the catch, and the CoxBox's per-stroke record sits near the finish, so in
a well-synced session the counts fall at a fixed phase of the stroke cycle after the record: the recovery's share
of the cycle at race rates, about 0.55 (0.53 in the one session synced by the boat's own sound). Phase 0 is a
CoxBox stroke record, 1 the next.

Counts are the words "one" to "ten" inside phrases the codebook leaves uncoded (counts), on each session's own
clock (words.jsonl). Sessions cut from one recording on one clock (the 7 Oct practice pieces) share one shift.
For each target phase (0.45, 0.55, 0.65) the shift is the smallest move, under half a stroke, that puts the
circular mean of the counts on the target; a group with fewer than 20 counts or no significant phase locking
(Rayleigh p >= 0.05) is not moved. Whole-stroke errors cannot be seen this way.

Output: data/local/coxing/sync/count_anchor.json {target: {session: {shift_s, shift_phase, n, phase, R, p}}};
run_experiment.py reads it with SYNC=counts COUNT_PHASE=<target>. The original syncs are not changed.
"""
import csv
import json
import os
import re

import numpy as np

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
SDIR = os.path.join(ROOT, "data", "local", "coxing", "sessions")
EXP = os.path.join(ROOT, "data", "local", "coxing", "experiment")
OUT = os.path.join(ROOT, "data", "local", "coxing", "sync", "count_anchor.json")
GROUPS = [["race_01"], ["practice_01"], ["practice_02"], ["race_04"], ["race_05"],
          ["practice_03", "practice_04", "practice_05", "practice_06", "practice_07", "practice_08"],
          ["race_06"], ["race_07"], ["race_08"], ["race_09"]]
COUNT = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
TARGETS = ("0.45", "0.55", "0.65")


def coded(sid):
    out = set()
    for line in open(os.path.join(EXP, "codes_%s.txt" % sid), encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            out.add(int(line.split()[0]))
    return out


def count_phases(sid):
    t = np.array([float(r["t"]) for r in csv.DictReader(open(os.path.join(SDIR, sid, "boat.csv")))])
    ph = json.load(open(os.path.join(SDIR, sid, "phrases.json"), encoding="utf-8"))
    cd = coded(sid)
    win = [(p["t"] - 0.05, (p.get("t_end") or p["t"]) + 0.05) for p in ph if p["i"] not in cd]
    words = [json.loads(x) for x in open(os.path.join(SDIR, sid, "words.jsonl"), encoding="utf-8")]
    tw = np.array([w["t"] for w in words if re.sub(r"[^a-z]", "", w["w"].lower()) in COUNT
                   and any(a <= w["t"] <= b for a, b in win)])
    tw = tw[(tw > t[0]) & (tw < t[-1])]
    k = np.searchsorted(t, tw, side="right") - 1
    ok = (k >= 0) & (k < len(t) - 1)
    return (tw[ok] - t[k[ok]]) / (t[k[ok] + 1] - t[k[ok]]), float(np.median(np.diff(t)))


def circ(ph):
    n = len(ph)
    z = np.mean(np.exp(2j * np.pi * ph))
    R = abs(z)
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n * n - (R * n) ** 2)) - (1 + 2 * n))) if n else 1.0
    return float((np.angle(z) / (2 * np.pi)) % 1.0), float(R), min(1.0, p)


def main():
    out = {k: {} for k in TARGETS}
    for g in GROUPS:
        per = {sid: count_phases(sid) for sid in g}
        ph = np.concatenate([v[0] for v in per.values()])
        phase, R, p = circ(ph)
        anchored = len(ph) >= 20 and p < 0.05
        for tgt in TARGETS:
            d = ((float(tgt) - phase + 0.5) % 1.0) - 0.5 if anchored else 0.0
            for sid, (_, period) in per.items():
                out[tgt][sid] = dict(shift_phase=round(d, 3), shift_s=round(d * period, 3), n=len(ph),
                                     phase=round(phase, 3), R=round(R, 3), p=p, anchored=anchored, period=period)
        print("%-40s counts %3d  phase %.2f  R %.2f  p %.1e  %s  shift to 0.55: %+.2f stroke (%+.2f s)" % (
            ",".join(g) if len(g) < 3 else "practice_03..08 (one recording)", len(ph), phase, R, p,
            "anchored" if anchored else "NOT moved", out["0.55"][g[0]]["shift_phase"], out["0.55"][g[0]]["shift_s"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
