"""Stroke rate and drive fraction from the video, against the CoxBox and the model.

The recovery is slower than the drive, so a frame-difference magnitude signal is high
through the drive and low through the recovery. Folding it on the stroke period and
measuring the duty cycle gives the drive fraction -- a quantity the model currently
gets wrong, and one no instrument in this boat recorded.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np

S = np.load("video_sig.npz")
mag, cent, FPS = S["mag"], S["cent"], float(S["fps"])
D = np.load("race.npz")
t, sr = D["t"], D["sr"]
OFF = float(np.load("offset.npy")[0])

# the video segment is video 8:00-9:30 -> race elapsed
t0, t1 = 8 * 60 + OFF, 9 * 60 + 30 + OFF
w = (t >= t0) & (t <= t1)
box_rate = float(sr[w].mean())
print("CoxBox over the same 90 s: %.2f spm (%.4f Hz), %d strokes logged" % (box_rate, box_rate / 60.0, w.sum()))

sig = cent - cent.mean()
sig = sig * np.hanning(len(sig))
P = np.abs(np.fft.rfft(sig)) ** 2
f = np.fft.rfftfreq(len(sig), 1.0 / FPS)
m = (f > 0.7) & (f < 1.3)
k = np.argmax(P[m])
f2 = f[m][k]
# parabolic refinement on the peak
i = int(np.where(f == f2)[0][0])
a, b, c = P[i - 1], P[i], P[i + 1]
df = 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])
f2r = f2 + df
print("video second harmonic  %.4f Hz -> fundamental %.4f Hz = %.2f spm" % (f2r, f2r / 2, 30 * f2r))
print("difference from the CoxBox: %+.2f spm (%.1f%%)"
      % (30 * f2r - box_rate, 100 * (30 * f2r / box_rate - 1)))

# --- fold the motion magnitude on the stroke period -------------------------------
T = 2.0 / f2r
n = len(mag)
tt = np.arange(n) / FPS
phase = np.mod(tt, T) / T
nb = 40
prof = np.array([mag[(phase >= i / nb) & (phase < (i + 1) / nb)].mean() for i in range(nb)])
prof = (prof - prof.min()) / (prof.max() - prof.min())
# rotate so the profile starts at its minimum (the catch, where the crew is slowest)
prof = np.roll(prof, -int(np.argmin(prof)))
above = prof > 0.5
print()
print("stroke period %.3f s; motion profile folded into %d bins" % (T, nb))
print("  duty cycle above half-amplitude: %.3f of the cycle" % (above.mean()))
print()
print("  the model's drive fraction, research four   0.481")
print("  the model's drive fraction, research eight  0.440")
print("  Hill & Fahrig, measured on the water        0.395 at rate 32")
print("  this crew, from the video                   %.3f" % above.mean())
print()
print("  profile (catch at the left, one full cycle):")
bar = "".join("#" if p > 0.5 else ("+" if p > 0.25 else ".") for p in prof)
print("   " + bar)
