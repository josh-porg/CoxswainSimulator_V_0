"""Segment glyphs from the LCD fields and cluster them so they can be labelled once."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import cv2
import numpy as np
import lcd_crop

GH, GW = 28, 18          # canonical glyph size


def binar(f):
    f = cv2.resize(f, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    f = cv2.GaussianBlur(f, (3, 3), 0)
    b = cv2.adaptiveThreshold(f, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY_INV, 21, 8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return b


def glyphs(field_img, min_w=4, min_h=10):
    b = binar(field_img)
    h, w = b.shape
    col = (b > 0).sum(0)
    thr = max(1, int(0.06 * h))
    on = col > thr
    runs, i = [], 0
    while i < w:
        if on[i]:
            j = i
            while j < w and on[j]:
                j += 1
            if j - i >= min_w:
                runs.append((i, j))
            i = j
        else:
            i += 1
    out = []
    for x0, x1 in runs:
        seg = b[:, x0:x1]
        row = (seg > 0).sum(1)
        ys = np.flatnonzero(row > 0)
        if len(ys) < min_h:
            continue
        g = seg[ys[0]:ys[-1] + 1]
        if g.shape[0] < min_h or g.shape[1] < min_w:
            continue
        out.append(((x0 + x1) / 2.0, (x1 - x0), g.shape[0],
                    cv2.resize(g, (GW, GH), interpolation=cv2.INTER_AREA)))
    return out


if __name__ == "__main__":
    imgs = np.load("_lcd_img.npy"); ts = np.load("_lcd_t.npy")
    keep = (ts > 105) & (ts < 335)
    imgs, ts = imgs[keep], ts[keep]
    print("%d frames in the race window (%.0f-%.0f s)" % (len(ts), ts.min(), ts.max()))
    allg, meta = [], []
    for i, im in enumerate(imgs):
        for fld in ("rate", "split", "dps", "dist"):
            for cx, gw, gh, g in glyphs(lcd_crop.cut(im, fld)):
                allg.append(g); meta.append((i, fld, cx, gw, gh))
    A = np.array(allg, np.float32) / 255.0
    print("segmented %d glyphs (%.1f per frame)" % (len(A), len(A) / len(imgs)))
    np.save("_glyphs.npy", A)
    np.save("_glyphmeta.npy", np.array([(m[0], "rate split dps dist".split().index(m[1]),
                                         m[2], m[3], m[4]) for m in meta], np.float32))
    from sklearn.cluster import KMeans
    K = 14
    lab = KMeans(K, n_init=8, random_state=0).fit_predict(A.reshape(len(A), -1))
    np.save("_glyphlab.npy", lab)
    tiles = []
    for c in range(K):
        m = lab == c
        cen = A[m].mean(0)
        t = cv2.resize((cen * 255).astype(np.uint8), (GW * 4, GH * 4),
                       interpolation=cv2.INTER_NEAREST)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        cv2.putText(t, "%d" % c, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 165, 255), 1)
        cv2.putText(t, "%d" % m.sum(), (2, GH * 4 - 4), cv2.FONT_HERSHEY_SIMPLEX, .38,
                    (0, 255, 255), 1)
        tiles.append(t)
    cv2.imwrite("_clusters.png", np.hstack(tiles))
    print("cluster sizes:", np.bincount(lab).tolist())
