"""Signed vertical optical flow, so drive and recovery can be told apart.

Magnitude alone peaks twice a cycle. A signed estimate does not: the crew moves one way
on the drive and the other on the recovery. 1-D Lucas-Kanade on the vertical gradient,
frame to frame:   shift = -sum(I_t * I_y) / sum(I_y^2)
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np
import imageio

rdr = imageio.get_reader("seg.mp4", "ffmpeg")
FPS = float(rdr.get_meta_data().get("fps", 30.0))
prev = None
shift = []
for i, f in enumerate(rdr):
    g = f.astype(np.float32).mean(axis=2)
    if prev is not None:
        Iy = 0.5 * (np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0))
        It = g - prev
        num = float(np.sum(It[2:-2] * Iy[2:-2]))
        den = float(np.sum(Iy[2:-2] ** 2))
        shift.append(-num / max(den, 1e-6))
    prev = g
s = np.asarray(shift)
np.save("flow.npy", s)
print("frames %d, signed vertical flow: mean %+.4f px, sd %.4f" % (len(s) + 1, s.mean(), s.std()))

# fundamental from the signed signal -- should now be at 1f, not 2f
w = (s - s.mean()) * np.hanning(len(s))
P = np.abs(np.fft.rfft(w)) ** 2
f = np.fft.rfftfreq(len(w), 1.0 / FPS)
m = (f > 0.25) & (f < 1.4)
k = int(np.argmax(P[m]))
f0 = f[m][k]
print("dominant frequency %.4f Hz = %.2f spm" % (f0, 60 * f0))

T = 1.0 / f0
tt = np.arange(len(s)) / FPS
ph = np.mod(tt, T) / T
nb = 36
prof = np.array([s[(ph >= i / nb) & (ph < (i + 1) / nb)].mean() for i in range(nb)])
# the drive is the faster half: the larger-magnitude excursion of the signed signal
pos, neg = prof > 0, prof < 0
# fold so the cycle starts where the signal crosses zero going into its big excursion
big = pos if abs(prof[pos].sum()) > abs(prof[neg].sum()) else neg
start = int(np.argmax(np.diff(big.astype(int)) > 0)) + 1 if np.any(np.diff(big.astype(int)) > 0) else 0
prof = np.roll(prof, -start)
big = np.roll(big, -start)
print()
print("signed profile over one cycle (catch-ish at the left):")
print("   " + "".join("#" if p > 0.4 * prof.max() else ("+" if p > 0 else ("-" if p > 0.4 * prof.min() else "=")) for p in prof))
print("   fraction of the cycle in the fast direction: %.3f" % big.mean())
print()
print("   model, research four   0.481")
print("   model, research eight  0.440")
print("   Hill & Fahrig measured 0.395 at rate 32")
print("   this crew, from video  %.3f" % big.mean())
