"""Normalise every tracked display to a canonical size and cut it into its four fields."""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..", "data", "local", "coxing", "analysis", "elite")))   # data and outputs live here (gitignored)
import cv2
import numpy as np

W, H = 360, 220                 # canonical display size (6x the template)
TW, THg = 90, 55


def stack(video="nat8.mp4", track="_boxtrack.npy", thr=0.62, fps=30.0):
    rec = np.load(track)
    c = cv2.VideoCapture(video)
    ts, imgs = [], []
    for t, m, x, y, s in rec:
        if m < thr or x < 0:
            continue
        c.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = c.read()
        if not ok:
            continue
        X, Y = int(2 * x), int(2 * y)
        Wp, Hp = int(TW * s), int(THg * s)
        d = f[max(0, Y):Y + Hp, max(0, X):X + Wp]
        if d.shape[0] < 10 or d.shape[1] < 20:
            continue
        g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LANCZOS4)
        g = cv2.createCLAHE(3.0, (6, 6)).apply(g)
        ts.append(t); imgs.append(g)
    return np.array(ts), np.array(imgs)


# field boxes as fractions of the canonical display, read off the verified layout
FIELD = {"rate":  (0.25, 0.53, 0.37, 0.68),      # x0,x1,y0,y1 of the canonical display
         "split": (0.65, 0.99, 0.37, 0.68),
         "dps":   (0.29, 0.58, 0.69, 0.98),
         "dist":  (0.77, 0.99, 0.69, 0.98)}


def cut(img, name):
    x0, x1, y0, y1 = FIELD[name]
    return img[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]


if __name__ == "__main__":
    ts, imgs = stack()
    np.save("_lcd_t.npy", ts); np.save("_lcd_img.npy", imgs)
    print("stacked %d display crops, canonical %dx%d" % (len(ts), W, H))
    sel = np.linspace(0, len(ts) - 1, 5).astype(int)
    rows = []
    for i in sel:
        cells = [cv2.resize(cut(imgs[i], k), (170, 90)) for k in FIELD]
        rows.append(np.hstack(cells))
    out = np.vstack(rows)
    cv2.imwrite("_fields.png", out)
    print("field montage: columns are", list(FIELD), "->", out.shape)
