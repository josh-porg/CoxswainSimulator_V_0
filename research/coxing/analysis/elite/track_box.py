"""Locate the CoxBox display in every frame and save the crops.

The camera is head-mounted, so the device wanders across the frame and changes apparent
size. Matching is done at half resolution over a small scale pyramid, with the search
restricted to a neighbourhood of the previous hit so the track stays continuous, and
falling back to a full-frame search whenever the local match degrades.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import cv2
import numpy as np

V = "nat8.mp4"
FPS = 30.0
STEP = 15                      # 2 Hz
c = cv2.VideoCapture(V)
N = int(c.get(cv2.CAP_PROP_FRAME_COUNT))

c.set(cv2.CAP_PROP_POS_FRAMES, int(139 * FPS))
_, f0 = c.read()
# display panel only: tighter and far more distinctive than the bezel
TPL0 = cv2.cvtColor(f0[905:960, 1035:1125], cv2.COLOR_BGR2GRAY)
TH, TW = TPL0.shape
SC = [0.8, 0.9, 1.0, 1.12, 1.25, 1.4]
half = lambda g: cv2.resize(g, None, fx=0.5, fy=0.5)
tpls = [(s, half(cv2.resize(TPL0, None, fx=s, fy=s)).astype(np.float32)) for s in SC]

rec = []
prev = None
c.set(cv2.CAP_PROP_POS_FRAMES, 0)
for fi in range(0, N, STEP):
    c.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, fr = c.read()
    if not ok:
        break
    g = half(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    best = (-1, None, None)
    for pad in ((120,) if prev is not None else (None,)):
        if prev is None:
            sub, ox, oy = g, 0, 0
        else:
            x0 = max(0, prev[0] - pad); y0 = max(0, prev[1] - pad)
            x1 = min(g.shape[1], prev[0] + pad); y1 = min(g.shape[0], prev[1] + pad)
            sub, ox, oy = g[y0:y1, x0:x1], x0, y0
        for s, t in tpls:
            if t.shape[0] >= sub.shape[0] or t.shape[1] >= sub.shape[1]:
                continue
            r = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
            m = float(r.max())
            if m > best[0]:
                loc = np.unravel_index(r.argmax(), r.shape)
                best = (m, (loc[1] + ox, loc[0] + oy), s)
    if best[0] < 0.55 and prev is not None:        # lost: re-acquire globally
        prev = None
        g2 = g
        for s, t in tpls:
            r = cv2.matchTemplate(g2, t, cv2.TM_CCOEFF_NORMED)
            m = float(r.max())
            if m > best[0]:
                loc = np.unravel_index(r.argmax(), r.shape)
                best = (m, (loc[1], loc[0]), s)
    if best[0] >= 0.55:
        prev = best[1]
    else:
        prev = None
    rec.append((fi / FPS, best[0], best[1][0] if best[1] else -1,
                best[1][1] if best[1] else -1, best[2] or 0))
rec = np.array(rec)
np.save("_boxtrack.npy", rec)
good = rec[rec[:, 1] >= 0.60]
print("sampled %d frames at %.1f Hz; matched >=0.60 in %d (%.0f%%)"
      % (len(rec), FPS / STEP, len(good), 100 * len(good) / len(rec)))
t = good[:, 0]
runs = []; s = t[0]; p = t[0]
for x in t[1:]:
    if x - p > 1.5:
        runs.append((s, p)); s = x
    p = x
runs.append((s, p))
long = [r for r in runs if r[1] - r[0] >= 10]
print("runs >=10 s: %d, totalling %.0f s of %.0f s (%.0f%%)"
      % (len(long), sum(r[1] - r[0] for r in long), N / FPS,
         100 * sum(r[1] - r[0] for r in long) / (N / FPS)))
print("  ", [("%.0f-%.0f" % r) for r in long][:14])
