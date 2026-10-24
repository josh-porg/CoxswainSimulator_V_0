"""Extract a stroke-cycle signal from the GoPro footage and test it against the model.

The camera is on the cox's head, so it sees the crew's backs filling most of the frame.
Their fore-aft motion is the dominant periodic content, so a per-frame motion measure
should carry the stroke cycle. Two things are then testable:

  1. the STROKE RATE, against what the CoxBox logged at the same moment -- a check that
     the optical signal really is the stroke and not camera shake;
  2. the DRIVE FRACTION, which is a live defect in the model: the research four sits at
     0.481 and the programme's target from Hill & Fahrig is 0.395 at rate 32.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np
import imageio

FPS = 30.0
frames = []
rdr = imageio.get_reader("seg.mp4", "ffmpeg")
meta = rdr.get_meta_data()
FPS = float(meta.get("fps", 30.0))
for i, f in enumerate(rdr):
    g = f.astype(np.float32).mean(axis=2)          # greyscale
    frames.append(g)
    if i > 2800:
        break
F = np.stack(frames)
print("frames %d at %.0f fps = %.1f s, size %dx%d" % (len(F), FPS, len(F) / FPS, F.shape[2], F.shape[1]))

# motion signal: how much the frame changed, and WHERE the change sat vertically
diff = np.abs(np.diff(F, axis=0))
mag = diff.mean(axis=(1, 2))
rows = np.arange(F.shape[1])[None, :]
cent = (diff.mean(axis=2) * rows).sum(axis=1) / np.maximum(diff.mean(axis=2).sum(axis=1), 1e-9)

np.savez("video_sig.npz", mag=mag, cent=cent, fps=FPS)


def spectrum(sig, fps, lo=0.2, hi=1.5):
    s = sig - sig.mean()
    s *= np.hanning(len(s))
    P = np.abs(np.fft.rfft(s)) ** 2
    f = np.fft.rfftfreq(len(s), 1.0 / fps)
    m = (f > lo) & (f < hi)
    return f[m], P[m]


for name, sig in (("frame-difference magnitude", mag), ("vertical centroid of motion", cent)):
    f, P = spectrum(sig, FPS)
    k = int(np.argmax(P))
    print()
    print("%s:" % name)
    print("   peak at %.4f Hz = %.2f strokes/min" % (f[k], 60 * f[k]))
    order = np.argsort(P)[::-1][:4]
    print("   top peaks: %s" % ", ".join("%.3f Hz (%.1f spm)" % (f[i], 60 * f[i]) for i in order))
