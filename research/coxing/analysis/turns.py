"""Steering cost done properly: the TURNS, not the wiggle.

The earlier pass measured path-length wiggle and found it was all GPS noise. That was
the wrong mechanism. On this course the steering cost is the TURNS -- rudder drag plus
the pressure turns, where one side is pulled harder and the boat yaws. The cox confirms
14:10 was exactly that: hard rudder with starboard pressure.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np

D = np.load("race.npz")
d, t, v, sr, lat, lon = D["d"], D["t"], D["v"], D["sr"], D["lat"], D["lon"]
R = 6371000.0
x = np.radians(lon - lon.mean()) * R * np.cos(np.radians(lat.mean()))
y = np.radians(lat - lat.mean()) * R


def roll(a, n=15):
    k = np.ones(n) / n
    return np.convolve(np.pad(a, (n // 2, n // 2), mode="edge"), k, mode="valid")[:len(a)]


hx, hy = roll(np.gradient(x), 9), roll(np.gradient(y), 9)
head = np.unwrap(np.arctan2(hy, hx))
turn_rate = np.degrees(np.gradient(head) / np.gradient(t))     # deg/s
tr = roll(turn_rate, 11)

thr = 1.0      # deg/s
inturn = np.abs(tr) > thr
runs, i = [], 0
while i < len(inturn):
    if inturn[i]:
        j = i
        while j + 1 < len(inturn) and inturn[j + 1]:
            j += 1
        if t[j] - t[i] > 8.0:
            runs.append((i, j))
        i = j + 1
    else:
        i += 1

base = float(np.median(v[~inturn]))
print("straight-line median speed %.3f m/s; %d strokes in turns of 8 s or more"
      % (base, sum(j - i + 1 for i, j in runs)))
print()
print("  %-9s %-9s %7s %9s %9s %9s %8s" % ("clock", "distance", "dur s", "turn deg", "peak d/s", "speed", "cost s"))
tot = 0.0
for i, j in runs:
    dur = t[j] - t[i]
    ang = np.degrees(head[j] - head[i])
    vv = v[i:j+1].mean()
    dist = d[j] - d[i]
    cost = dist / vv - dist / base
    tot += cost
    print("  %-9s %-9s %7.1f %+9.0f %9.2f %9.3f %+8.1f"
          % ("%d:%02d" % (t[i] // 60, t[i] % 60), "%.0f m" % d[i], dur, ang,
             tr[i:j+1][np.argmax(np.abs(tr[i:j+1]))], vv, cost))
print()
print("  total turning cost %+.1f s over %.0f m of turns" % (tot, sum(d[j] - d[i] for i, j in runs)))
print()
print("  For comparison, the whole-race gap to the requalification line is 66 s,")
print("  and the length fade alone is worth up to 67 s.")
