"""Track a CoxBox and emit reading montages for any of these videos.

Same machinery as race 2: multi-scale template match with temporal continuity and
global re-acquisition on loss, normalise to a canonical display, then pick the sharpest
frame per time bin. `rot180` handles boats where the unit is mounted facing the
coxswain so the camera sees it inverted.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import sys

import cv2
import numpy as np


def run(video, tpl_time, tpl_box, t0, t1, rot180=False, step_s=4.0, tag="X", fps=30.0):
    c = cv2.VideoCapture(video)
    N = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.set(cv2.CAP_PROP_POS_FRAMES, int(tpl_time * fps))
    _, f0 = c.read()
    y0, y1, x0, x1 = tpl_box
    T0 = cv2.cvtColor(f0[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    TH, TW = T0.shape
    half = lambda g: cv2.resize(g, None, fx=0.5, fy=0.5)
    SC = [0.8, 0.9, 1.0, 1.12, 1.25]
    tpls = [(s, half(cv2.resize(T0, None, fx=s, fy=s)).astype(np.float32)) for s in SC]

    rec, prev = [], None
    for fi in range(0, N, 15):
        c.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = c.read()
        if not ok:
            break
        t = fi / fps
        if not (t0 - 5 <= t <= t1 + 5):
            continue
        g = half(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)).astype(np.float32)
        best = (-1, None, None)
        for pad in ((110,) if prev is not None else (None,)):
            if prev is None:
                sub, ox, oy = g, 0, 0
            else:
                a = max(0, prev[0] - pad); b = max(0, prev[1] - pad)
                sub, ox, oy = g[b:prev[1] + pad, a:prev[0] + pad], a, b
            for s, tp in tpls:
                if tp.shape[0] >= sub.shape[0] or tp.shape[1] >= sub.shape[1]:
                    continue
                r = cv2.matchTemplate(sub, tp, cv2.TM_CCOEFF_NORMED)
                m = float(r.max())
                if m > best[0]:
                    L = np.unravel_index(r.argmax(), r.shape)
                    best = (m, (L[1] + ox, L[0] + oy), s)
        if best[0] < 0.55 and prev is not None:
            prev = None
            for s, tp in tpls:
                r = cv2.matchTemplate(g, tp, cv2.TM_CCOEFF_NORMED)
                m = float(r.max())
                if m > best[0]:
                    L = np.unravel_index(r.argmax(), r.shape)
                    best = (m, (L[1], L[0]), s)
        prev = best[1] if best[0] >= 0.55 else None
        if best[0] >= 0.58:
            X, Y = int(2 * best[1][0]), int(2 * best[1][1])
            W, H = int(TW * best[2]), int(TH * best[2])
            d = fr[max(0, Y):Y + H, max(0, X):X + W]
            if d.shape[0] > 12 and d.shape[1] > 20:
                d = cv2.cvtColor(cv2.resize(d, (360, 240)), cv2.COLOR_BGR2GRAY)
                if rot180:
                    d = cv2.rotate(d, cv2.ROTATE_180)
                rec.append((t, best[0], cv2.createCLAHE(3.0, (6, 6)).apply(d)))
    print("%s: %d usable frames between %.0f and %.0f s" % (tag, len(rec), t0, t1))

    sel = []
    for w in np.arange(t0, t1, step_s):
        cand = [i for i, r in enumerate(rec) if abs(r[0] - w) < step_s * 0.6]
        if cand:
            sel.append(max(cand, key=lambda i: cv2.Laplacian(rec[i][2], cv2.CV_64F).var()))
    sel = sorted(set(sel))
    print("   %d readings selected at ~%.0f s spacing" % (len(sel), step_s))
    rows = []
    for i in sel:
        t, m, img = rec[i]
        r = cv2.cvtColor(cv2.resize(img, (720, 480)), cv2.COLOR_GRAY2BGR)
        cv2.putText(r, "%.0fs" % t, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        rows.append(r)
    import math
    B = 4
    for b in range(math.ceil(len(rows) / B)):
        cv2.imwrite("_%s_%02d.png" % (tag, b), np.vstack(rows[b * B:(b + 1) * B]))
    np.save("_%s_times.npy" % tag, np.array([rec[i][0] for i in sel]))
    print("   wrote %d montages" % math.ceil(len(rows) / B))


if __name__ == "__main__":
    run("c4_SQQTqYmb_Dg.mp4", 240, (790, 950, 1130, 1340), 90, 340,
        rot180=True, step_s=5.0, tag="d4")
