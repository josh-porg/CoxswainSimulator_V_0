"""Boat-response events: where the boat actually changed, with timestamps for the video.

I cannot hear the calls -- I have no access to the video's audio. What I CAN do is the
objective half: find every moment the boat's speed or rate moved more than its own
noise, and timestamp it. Matching those against what was called is then a lookup in
the GoPro footage, and the pairing is the analysis.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np

D = np.load("race.npz")
d, t, v, sr, dps = D["d"], D["t"], D["v"], D["sr"], D["dps"]


def clock(x):
    return "%d:%04.1f" % (int(x // 60), x % 60)


# a slow baseline: what the boat was doing over the last ~20 strokes
def roll(a, n=21):
    k = np.ones(n) / n
    return np.convolve(np.pad(a, (n // 2, n // 2), mode="edge"), k, mode="valid")[:len(a)]


v_base, sr_base = roll(v), roll(sr)
dv, dsr = v - v_base, sr - sr_base
sd_v, sd_sr = dv.std(), dsr.std()
print("stroke-to-stroke noise about the 21-stroke baseline: speed %.3f m/s, rate %.2f spm"
      % (sd_v, sd_sr))
print()

# --- sustained losses: 3+ consecutive strokes at least 1.5 sd below baseline ---------
bad = dv < -1.5 * sd_v
runs, i = [], 0
while i < len(bad):
    if bad[i]:
        j = i
        while j + 1 < len(bad) and bad[j + 1]:
            j += 1
        if j - i + 1 >= 3:
            runs.append((i, j))
        i = j + 1
    else:
        i += 1
print("=== sustained losses (3+ strokes, speed well below the boat's own trend) ===")
print("  %-10s %-10s %8s %8s %8s %9s" % ("clock", "distance", "strokes", "lost m/s", "rate", "m/stroke"))
for a, b in runs:
    print("  %-10s %-10s %8d %8.3f %8.1f %9.2f"
          % (clock(t[a]), "%.0f m" % d[a], b - a + 1, dv[a:b + 1].mean(),
             sr[a:b + 1].mean(), dps[a:b + 1].mean()))

# --- surges ---------------------------------------------------------------------
good = dv > 1.5 * sd_v
runs2, i = [], 0
while i < len(good):
    if good[i]:
        j = i
        while j + 1 < len(good) and good[j + 1]:
            j += 1
        if j - i + 1 >= 3:
            runs2.append((i, j))
        i = j + 1
    else:
        i += 1
print()
print("=== sustained surges (the boat went, and stayed gone, 3+ strokes) ===")
print("  %-10s %-10s %8s %8s %8s %9s" % ("clock", "distance", "strokes", "gain m/s", "rate", "m/stroke"))
for a, b in runs2:
    print("  %-10s %-10s %8d %8.3f %8.1f %9.2f"
          % (clock(t[a]), "%.0f m" % d[a], b - a + 1, dv[a:b + 1].mean(),
             sr[a:b + 1].mean(), dps[a:b + 1].mean()))

# --- rate moves, and whether the boat answered -------------------------------------
print()
print("=== rate moves of 1.5+ spm over 5 strokes, and what the boat did about it ===")
print("  %-10s %-10s %9s %11s %11s %s" % ("clock", "distance", "d rate", "d speed", "d m/stroke", "verdict"))
W = 5
for i in range(W, len(sr) - W):
    a = sr[i:i + W].mean() - sr[i - W:i].mean()
    if abs(a) < 1.5:
        continue
    if i > W and abs(sr[i - 1:i - 1 + W].mean() - sr[i - 1 - W:i - 1].mean()) >= abs(a):
        continue          # keep only the local peak of each move
    b = v[i:i + W].mean() - v[i - W:i].mean()
    c = dps[i:i + W].mean() - dps[i - W:i].mean()
    verdict = "boat answered" if (a > 0) == (b > 0) and abs(b) > 0.5 * sd_v else (
        "rate up, no speed" if a > 0 else "rate down, no loss")
    print("  %-10s %-10s %+9.1f %+11.3f %+11.2f %s"
          % (clock(t[i]), "%.0f m" % d[i], a, b, c, verdict))
