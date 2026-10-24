"""Race-day wind through the project's own chop and windage models.

Observed (Open-Meteo archive, 47.63 N 122.32 W, 08:00 local 20 Sep 2026):
1.7 m/s from 33 deg, gusting 4.2. Light and from the NNE.
"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from coxswain.hydro.chop import FetchLimitedSea, WalledBasin, added_resistance  # noqa: E402
from course_pacing import build_boat, hull_drag                                  # noqa: E402

U, GUST, WDIR = 1.7, 4.2, 33.0
R = np.load("race.npz")
v, d, lat, lon = R["v"], R["d"], R["lat"], R["lon"]
boat = build_boat("4+", 32.0)
BEAM, LEN = 0.60, float(boat.length)
drag_fn = hull_drag(boat)
base = float(drag_fn(float(np.median(v))))

print("=== fetch-limited chop from the observed wind ===")
print("  %-10s %10s %10s %12s %12s" % ("fetch m", "H_s m", "T_p s", "wavelength", "added R (C=1)"))
for F in (500.0, 1000.0, 2000.0, 4000.0):
    for UU, lab in ((U, "steady"), (GUST, "gust")):
        sea = FetchLimitedSea(wind=UU, fetch=F)
        H, T, L = sea.significant_height, sea.peak_period, sea.wavelength
        # the Cut and Union Bay are walled, so reflection roughly doubles it near a wall
        basin = WalledBasin(sea=sea, reflection=0.8)
        Hw = float(basin.open_water_height)
        dR = float(added_resistance(Hw, BEAM, LEN, coefficient=1.0))
        print("  %-10.0f %10.3f %10.2f %12.2f %12.3f N   (%s U=%.1f)"
              % (F, H, T, L, dR, lab, UU))

print()
print("  hull drag at race speed %.1f N, so the steady-wind chop is %.2f%% of it"
      % (base, 100 * float(added_resistance(
          WalledBasin(sea=FetchLimitedSea(wind=U, fetch=2000.0), reflection=0.8).open_water_height,
          BEAM, LEN, coefficient=1.0)) / base))

# --- windage along the track -----------------------------------------------------
print()
print("=== windage ===")
RHO_AIR, CD, AREA = 1.22, 0.60, 0.30      # crew + hull frontal area, m^2, order-of-magnitude
th = np.radians(WDIR)
wx, wy = -U * np.sin(th), -U * np.cos(th)          # vector the wind blows TOWARDS
hx, hy = np.gradient(lon), np.gradient(lat)
hn = np.hypot(hx, hy)
hx, hy = hx / np.maximum(hn, 1e-12), hy / np.maximum(hn, 1e-12)
# heading unit vector in metres
lat0 = np.radians(lat.mean())
ex, ey = hx * np.cos(lat0), hy
en = np.hypot(ex, ey)
ex, ey = ex / en, ey / en
along = wx * ex + wy * ey                  # + = tailwind
v_rel = v - along
Fa = 0.5 * RHO_AIR * CD * AREA * np.sign(v_rel) * v_rel ** 2
Fa0 = 0.5 * RHO_AIR * CD * AREA * v ** 2   # still air
extra = Fa - Fa0
seg = np.gradient(d)
lost = float(np.sum(seg * extra / max(base, 1e-6) / v))
print("  along-track wind component: mean %+.2f m/s (+ = tail), range %+.2f to %+.2f"
      % (along.mean(), along.min(), along.max()))
print("  windage in still air %.2f N; with this wind %.2f N; mean extra %+.3f N"
      % (Fa0.mean(), Fa.mean(), extra.mean()))
print("  INTEGRATED TIME COST OF THE WIND: %+.1f s" % lost)
