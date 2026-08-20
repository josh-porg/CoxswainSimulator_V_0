"""Extract Charles River isobaths from the CRAB/MIT Sea Grant KMZ.

Source
------
Charles River Alliance of Boaters + MIT Sea Grant, 2016-17 sonar survey of
the Lower Charles (New Charles River Dam to Watertown Dam, ~14.5 km).
KMZ: http://www.charlesriverallianceofboaters.org/chart/charles.kmz

Structure: each Placemark carries an attribute table in its description.
``VALUE`` is the contour depth in feet, ``MIN_CAT`` the band label, and
``MAJ_CAT`` distinguishes the filled depth areas (polygons) from the
nautical contour lines.  Depths are below the basin's normal pool, which
the New Charles River Dam holds nearly constant -- so they are depth below
the surface a boat sits on, which is exactly what the drag model wants.
"""
import os
import re
from collections import Counter

import numpy as np

d = (r"C:\Users\satur\AppData\Local\Temp\claude"
     r"\C--Users-satur-PycharmProjects"
     r"\ed74e05d-eb95-4ae0-8073-d159240c91d6\scratchpad\charles")

kml = open(os.path.join(d, "charles.kml"), encoding="utf-8").read()

placemark = re.compile(r"<Placemark\b.*?</Placemark>", re.S)
field_re = re.compile(r"<td>([\w ]+)</td>\s*<td>([^<]*)</td>", re.S)
coord_re = re.compile(r"<coordinates>(.*?)</coordinates>", re.S)

FEET = 0.3048
rows = []
levels = Counter()

for m in placemark.finditer(kml):
    body = m.group(0)
    fields = dict(field_re.findall(body))
    if fields.get("MAJ_CAT") != "Nautical":
        continue                      # keep the contour lines only
    value = fields.get("VALUE", "")
    if value == "":
        continue
    depth_ft = float(value)
    if depth_ft <= 0:
        continue                      # 0 ft is the shoreline, not a depth
    levels[depth_ft] += 1
    for cm in coord_re.finditer(body):
        for triple in cm.group(1).split():
            parts = triple.split(",")
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1]),
                             depth_ft * FEET))

print("contour levels (ft -> placemarks):")
for level, n in sorted(levels.items()):
    print(f"   {level:5.1f} ft = {level * FEET:4.2f} m   {n:5d}")

arr = np.array(rows)
print(f"\n{len(arr):,} vertices")
print(f"  lon   {arr[:, 0].min():.5f} .. {arr[:, 0].max():.5f}")
print(f"  lat   {arr[:, 1].min():.5f} .. {arr[:, 1].max():.5f}")
print(f"  depth {arr[:, 2].min():.2f} .. {arr[:, 2].max():.2f} m")

# thin: contour vertices are far denser than the field needs
keep = arr[::3]
out = os.path.join("data", "charles_isobaths.csv")
os.makedirs("data", exist_ok=True)
np.savetxt(out, keep, delimiter=",", header="lon,lat,depth_m", comments="",
           fmt="%.6f,%.6f,%.3f")
print(f"\nwrote {out}: {len(keep):,} points, "
      f"{os.path.getsize(out) / 1e6:.2f} MB")
