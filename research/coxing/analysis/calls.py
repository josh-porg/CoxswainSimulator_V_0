"""Did the boat answer the call? Every explicit move, scored against the boat's response.

Clock sync from align.py: video + 1.0 s = race elapsed, verified against 18 split
readouts to a mean 1.4 s per 500. Each move is measured over the 10 strokes after the
call lands against the 10 before, so a "take 10" is judged on the 10 it asked for.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np

D = np.load("race.npz")
t, v, sr, dps, d = D["t"], D["v"], D["sr"], D["dps"], D["d"]
OFF = float(np.load("offset.npy")[0])


def vsec(s):
    m, ss = s.split(":")
    return int(m) * 60 + int(ss)


# video time the move LANDS (the "one, two" or the first stroke of the burst), and what it asked for
MOVES = [
    ("2:01", "ratio back, rate down", "ratio"),
    ("3:28", "find that ratio", "ratio"),
    ("4:23", "eyes closed five, keep the press", "focus"),
    ("5:14", "add press at the finish", "press"),
    ("5:40", "ratio, get long", "ratio"),
    ("6:43", "decide to drop the split", "press"),
    ("7:46", "suspend from the handle, five", "press"),
    ("8:34", "five strokes on sequencing", "focus"),
    ("9:32", "218, ratio", "ratio"),
    ("9:47", "distance per stroke", "length"),
    ("11:43", "front end focus, attack", "catch"),
    ("12:23", "bodies, swing, send", "length"),
    ("12:35", "stand on the stretchers", "press"),
    ("12:53", "reach long and breathe", "length"),
    ("13:21", "take 10, pass a boat", "burst"),
    ("14:10", "starboard press, three", "steer"),
    ("15:07", "take a 10, make a move", "burst"),
    ("15:45", "lengthen, get rhythm back", "ratio"),
    ("16:19", "take a 10 before the turn", "burst"),
    ("17:28", "five big ones round the corner", "burst"),
    ("18:05", "drop the splits", "press"),
    ("18:19", "send the puddles, five", "length"),
]

N = 10
print("%-7s %-34s %-7s %7s %7s %8s  %s"
      % ("video", "call", "type", "d spd", "d rate", "d m/str", "did the boat answer?"))
print("-" * 104)
score = {"yes": 0, "part": 0, "no": 0}
for vt, text, kind in MOVES:
    rt = vsec(vt) + OFF
    i = int(np.argmin(np.abs(t - rt)))
    if i < N or i > len(t) - N - 1:
        continue
    a, b = slice(i - N, i), slice(i, i + N)
    dv = v[b].mean() - v[a].mean()
    dr = sr[b].mean() - sr[a].mean()
    dl = dps[b].mean() - dps[a].mean()
    # what counts as answering depends on what was asked
    if kind in ("burst", "press", "catch"):
        ok = dv > 0.015
        partial = dv > 0.0
    elif kind in ("ratio", "length"):
        ok = dl > 0.05 and dv > -0.02
        partial = dl > 0.0
    else:
        ok = dv > 0.015 or dl > 0.05
        partial = dv > 0.0 or dl > 0.0
    verdict = "YES" if ok else ("partly" if partial else "no")
    score["yes" if ok else ("part" if partial else "no")] += 1
    print("%-7s %-34s %-7s %+7.3f %+7.1f %+8.2f  %s"
          % (vt, text[:34], kind, dv, dr, dl, verdict))

print()
print("answered %d, partly %d, not %d, of %d moves"
      % (score["yes"], score["part"], score["no"], sum(score.values())))
print()
print("by type of call:")
for kind in ("burst", "press", "ratio", "length", "catch", "focus", "steer"):
    rows = [(vt, k) for vt, _, k in MOVES if k == kind]
    if not rows:
        continue
    dvs, dls = [], []
    for vt, _ in rows:
        i = int(np.argmin(np.abs(t - (vsec(vt) + OFF))))
        if i < N or i > len(t) - N - 1:
            continue
        dvs.append(v[i:i + N].mean() - v[i - N:i].mean())
        dls.append(dps[i:i + N].mean() - dps[i - N:i].mean())
    print("   %-8s n=%d   mean d speed %+.3f m/s   mean d m/stroke %+.2f"
          % (kind, len(dvs), np.mean(dvs), np.mean(dls)))
