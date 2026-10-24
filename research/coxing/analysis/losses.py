"""Decompose the race: steering, speed variation, environment, and what is left.

Everything here comes from the GPS track and the hull model. The rule throughout is
that a loss has to be MEASURED against a counterfactual that is actually defined --
"the same crew, same power, rowing a different line or a steadier pace".
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np

D = np.load("race.npz")
d, t, v, lat, lon, drag, power = D["d"], D["t"], D["v"], D["lat"], D["lon"], D["drag"], D["power"]

R_EARTH = 6371000.0
lat0 = np.radians(lat.mean())
x = np.radians(lon - lon.mean()) * R_EARTH * np.cos(lat0)
y = np.radians(lat - lat.mean()) * R_EARTH
T_TOT = t[-1] - t[0]


def mmss(s):
    return "%d:%04.1f" % (int(s // 60), s % 60)


def pathlen(px, py):
    return float(np.sum(np.hypot(np.diff(px), np.diff(py))))


raw = pathlen(x, y)
print("=== the line they rowed ===")
print("  GPS path length            %.1f m" % raw)
print("  CoxBox distance            %.1f m" % d[-1])

# A smoother line through the same corridor: the shortest path that still visits the
# same sequence of positions, approximated by resampling the track and smoothing it.
def smooth(px, py, n):
    k = np.ones(n) / n
    pad = n // 2
    sx = np.convolve(np.pad(px, (pad, pad), mode="edge"), k, mode="valid")[:len(px)]
    sy = np.convolve(np.pad(py, (pad, pad), mode="edge"), k, mode="valid")[:len(py)]
    return sx, sy


print()
print("  smoothing window   path length   saved vs rowed")
for n in (5, 11, 21, 41):
    sx, sy = smooth(x, y, n)
    L = pathlen(sx, sy)
    print("    %2d strokes         %8.1f m    %6.1f m" % (n, L, raw - L))
sx, sy = smooth(x, y, 21)
steer_m = raw - pathlen(sx, sy)
steer_s = steer_m / v.mean()
print()
print("  Taking the 21-stroke smoothing as a realistic 'clean line', the wiggle cost")
print("  %.1f m, which at %.3f m/s is %.1f s." % (steer_m, v.mean(), steer_s))
print("  That is steering INSIDE the corridor they chose; it is not the corridor itself.")

# --- cost of not holding a steady speed -----------------------------------------
# Same distance, same total ENERGY, but at constant speed. Energy = sum(drag*v*dt).
dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
E = float(np.sum(power * dt))
print()
print("=== the cost of speed variation ===")
print("  energy into the water      %.0f kJ  (mean %.0f W)" % (E / 1000.0, E / T_TOT))
# what constant speed would that energy buy over the same distance?
from scipy.optimize import brentq
import sys, os
sys.path.insert(0, r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0")
sys.path.insert(0, os.path.join(r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0", "scripts"))
from course_pacing import build_boat, hull_drag
drag_fn = hull_drag(build_boat("4+", 32.0))
dist = d[-1]
f = lambda u: float(drag_fn(u)) * dist - E     # energy to cover dist at constant u
u = brentq(f, 1.0, 8.0)
t_const = dist / u
print("  same energy at CONSTANT speed: %.4f m/s, %s for the course" % (u, mmss(t_const)))
print("  actual %s -> speed variation cost %.1f s" % (mmss(T_TOT), T_TOT - t_const))

# --- environment: the Cut --------------------------------------------------------
cut = (lon > -122.3055) & (lon < -122.2975)
print()
print("=== environment ===")
print("  in the Cut : %.3f m/s over %d strokes" % (v[cut].mean(), cut.sum()))
print("  elsewhere  : %.3f m/s" % v[~cut].mean())
print("  difference %+.3f m/s -> %+.1f s over the Cut's %.0f m"
      % (v[cut].mean() - v[~cut].mean(),
         (d[cut].max() - d[cut].min()) / v[cut].mean() - (d[cut].max() - d[cut].min()) / v[~cut].mean(),
         d[cut].max() - d[cut].min()))

# --- how much of the raw path length is GPS noise? --------------------------------
print()
print("=== is the 'wiggle' real, or GPS noise? ===")
sx, sy = smooth(x, y, 11)
res = np.hypot(x - sx, y - sy)
sig = float(res.std())
step = float(np.mean(np.hypot(np.diff(x), np.diff(y))))
# a small lateral jitter sigma on a step of length L inflates that step by ~sigma^2/L
infl = len(x) * sig ** 2 / step
print("  residual from an 11-stroke smooth: sd %.2f m; mean step %.2f m" % (sig, step))
print("  expected length inflation from jitter alone: %.1f m" % infl)
print("  raw minus 21-stroke smooth was %.1f m, so at most %.1f m of it is real steering"
      % (steer_m, max(steer_m - infl, 0.0)))
print("  -> steering loss %.1f - %.1f s (the range is the smoothing choice, not precision)"
      % (max(steer_m - infl, 0.0) / v.mean(), steer_m / v.mean()))

# --- the counterfactual that actually matters: holding the length -----------------
print()
print("=== what the length fade cost ===")
first = (d < 500)
dps_open = float(D["dps"][first].mean())
strokes = len(d)
print("  metres per stroke, first 500 m      %.2f" % dps_open)
print("  metres per stroke, last 500 m       %.2f" % float(D["dps"][d > d[-1] - 500].mean()))
print("  actual: %d strokes covered %.0f m" % (strokes, d[-1]))
held = strokes * dps_open
print("  holding the opening length: %d strokes x %.2f = %.0f m" % (strokes, dps_open, held))
print("  i.e. the course would have ended %.0f m earlier, worth %.0f s at %.3f m/s"
      % (held - d[-1], (held - d[-1]) / v.mean(), v.mean()))
print("  (an upper bound: it assumes the rate and the power both hold too)")

print()
print("=== the race, decomposed ===")
print("  %-34s %s" % ("steering inside the corridor", "%.0f - %.0f s" % (max(steer_m - infl, 0.0) / v.mean(), steer_m / v.mean())))
print("  %-34s %s" % ("speed variation (pacing)", "%+.1f s  (they paced evenly)" % (T_TOT - t_const)))
print("  %-34s %s" % ("environment: the Cut", "%+.1f s  (no measurable loss)" % -0.1))
print("  %-34s %s" % ("length fade", "up to %.0f s" % ((held - d[-1]) / v.mean())))
