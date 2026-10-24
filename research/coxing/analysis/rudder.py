"""Reverse-engineer the rudder drag from the track.

Every turn on this course except the one pressure turn was made on the rudder. A rudder
that holds a yaw rate is carrying a side force, and a side force costs drag. So the
speed deficit against turn rate recovers the drag, without needing to know the rudder
angle: for a hull turning at rate r with speed v, the radius is R = v/r, and holding
that circle needs a side force ~ m v^2 / R = m v r. The induced drag rises as the square
of that side force, so the speed deficit should go as r^2.
"""
import os as _os
_os.chdir(_os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "analysis")))   # data and outputs live here (gitignored)
import numpy as np

D = np.load("race.npz")
t, v, d, lat, lon = D["t"], D["v"], D["d"], D["lat"], D["lon"]
R = 6371000.0
x = np.radians(lon - lon.mean()) * R * np.cos(np.radians(lat.mean()))
y = np.radians(lat - lat.mean()) * R


def roll(a, n):
    k = np.ones(n) / n
    return np.convolve(np.pad(a, (n // 2, n // 2), mode="edge"), k, mode="valid")[:len(a)]


hx, hy = roll(np.gradient(x), 9), roll(np.gradient(y), 9)
head = np.unwrap(np.arctan2(hy, hx))
r = np.abs(roll(np.gradient(head) / np.gradient(t), 11))       # rad/s, unsigned

# remove the slow fade so we compare like with like
trend = roll(v, 61)
dv = v - trend
PRESSURE = (t > 14 * 60) & (t < 14 * 60 + 40)                  # the one pressure turn
m = ~PRESSURE & np.isfinite(r)

A = np.column_stack([r[m] ** 2, np.ones(m.sum())])
coef, *_ = np.linalg.lstsq(A, dv[m], rcond=None)
k2 = coef[0]
resid = dv[m] - A @ coef
print("rudder-only strokes: %d (pressure turn excluded)" % m.sum())
print("fit  dv = k * r^2 + c :  k = %+.3f m/s per (rad/s)^2,  c = %+.4f" % (k2, coef[1]))
print("     r^2 of the fit %.3f" % (1 - resid.var() / dv[m].var()))

MASS = 4 * 68.0 + 68.0 + 50.0      # four rowers, cox, hull+riggers, kg
vbar = float(np.median(v))
for deg in (0.5, 1.0, 1.5, 2.0):
    rr = np.radians(deg)
    side = MASS * vbar * rr
    loss = k2 * rr ** 2
    Rturn = vbar / rr
    print("  turning %.1f deg/s: radius %5.0f m, side force %5.0f N, speed deficit %+.4f m/s"
          % (deg, Rturn, side, loss))

tot = float(np.sum(np.gradient(d) * (-k2 * r ** 2) / np.maximum(v, 1e-6)))
print()
print("integrated over the race, rudder use cost %+.1f s" % tot)
print("(the two big turns alone were priced at 1.6 s by the speed-deficit method;")
print(" this counts every correction, not just the turns)")

# --- how much of r is GPS noise? ---------------------------------------------------
print()
print("=== noise floor in the turn-rate estimate ===")
# the straightest tenth of the race: whatever r reads there is mostly noise
q = np.percentile(r, 10)
quiet = r <= q
print("  quietest 10%% of strokes: mean |r| %.4f deg/s" % np.degrees(r[quiet].mean()))
# heading noise from the residual of the track about a smooth line
sx, sy = roll(x, 11), roll(y, 11)
jit = float(np.hypot(x - sx, y - sy).std())
step = float(np.mean(np.hypot(np.diff(x), np.diff(y))))
dt = float(np.mean(np.diff(t)))
r_noise = jit / step / dt * np.sqrt(2) / 9        # heading noise, smoothed over 9 strokes
print("  jitter %.2f m on a %.2f m step -> heading noise ~%.4f rad/s = %.3f deg/s"
      % (jit, step, r_noise, np.degrees(r_noise)))
r_true2 = np.maximum(r ** 2 - r_noise ** 2, 0.0)
tot_corr = float(np.sum(np.gradient(d) * (-k2 * r_true2) / np.maximum(v, 1e-6)))
print()
print("  rudder cost, raw r^2            %+.1f s" % tot)
print("  rudder cost, noise-corrected    %+.1f s" % tot_corr)
print("  the two named turns alone       +1.6 s")
print()
print("  So steering costs somewhere between about 2 and %.0f s; the spread is the" % tot_corr)
print("  turn-rate noise floor, not the physics. A rate-gyro would settle it outright.")
