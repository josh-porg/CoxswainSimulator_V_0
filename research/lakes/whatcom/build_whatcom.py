"""Lake Whatcom: where the wind and the chop are, for any wind, precomputed for the live page.

    .venv/Scripts/python.exe research/lakes/whatcom/build_whatcom.py

Inputs: the OSM shoreline (osm_lake.json, relation 415479), the USGS 3DEP terrain (coxswain/data/whatcom_dem.npz,
tools/fetch_dem.py, 30 m), the two routes (routes.json, traced from the screenshots).

For every water cell on a GRID-metre grid and each of 16 wind directions (FROM):

* **fetch** F: open water upwind of the cell to the shore (islands included), up to 12 km;
* **shore shelter**: wind over water F metres downwind of a forested shore against open water at the same
  regional wind (coxswain.hydro.canopy: the internal boundary layer from a forest, z0 2 m, to water);
* **terrain shelter**: Winstral's Sx, the steepest angle up to the terrain within 2.5 km upwind, turned into a
  speed factor 1 - ALPHA min(Sx / 12 deg, 1), ALPHA fitted to the lakeside stations (calibrate.py). This is a heuristic (lee of steep ground slows the surface wind
  by up to about half); the forecast models resolve the valley only at 3 km, so it adds the local part;
* **chop**: JONSWAP fetch-limited growth (coxswain.hydro.chop), Hs = 0.0016 U sqrt(F/g) with U the wind
  averaged along the fetch, so Hs is linear in the forecast wind: kH (s) such that Hs = kH * U10;
  peak period Tp = kT * U10^(1/3).

Stored per cell and direction: kH, kT, and the local wind factor f (wind felt = f * forecast wind). The
page multiplies by the live forecast. Routes are cut into ~500 m sections with the same per-direction
numbers (mean and worst cell within 60 m of the line) and the section's heading.
Writes whatcom_model.json.
"""
import json, math, os, sys
import numpy as np
from matplotlib.path import Path

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "whatcom"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, CODE)
from coxswain.river.course import local_tangent_plane, inverse_tangent_plane
from coxswain.river.terrain import load_terrain
from coxswain.hydro import canopy
import lake_geometry as G

ORIGIN = (48.72, -122.34)
GRID = 75.0
DIRS = [22.5 * i for i in range(16)]
G0 = 9.81
FOREST = canopy.Roughness(z0=2.0, d=17.0, height=25.0, frontal_index=0.3)
SX_RANGE = 2500.0
#: how much the steepest upwind terrain takes out of the wind (f = 1 - ALPHA min(Sx/12 deg, 1)); 0.5 a priori,
#: replaced by the lakeside stations' fit (calibrate.py -> calibration.json) when one exists
_CAL = os.path.join(HERE, "calibration.json")
ALPHA = json.load(open(_CAL))["alpha"] if os.path.exists(_CAL) else 0.5
DEM_BOUNDS = (48.61994, -122.47, 48.80006, -122.18)


def main():
    outer, inner = G.rings()
    oxy = np.column_stack(local_tangent_plane(np.array([p[0] for p in outer[0]]), np.array([p[1] for p in outer[0]]), ORIGIN))
    ixy = [np.column_stack(local_tangent_plane(np.array([p[0] for p in r]), np.array([p[1] for p in r]), ORIGIN)) for r in inner]
    po, pis = Path(oxy), [Path(a) for a in ixy]
    x0, y0 = oxy.min(axis=0) - GRID; x1, y1 = oxy.max(axis=0) + GRID
    xs, ys = np.arange(x0, x1, GRID), np.arange(y0, y1, GRID)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    water = po.contains_points(pts)
    for p in pis:
        water &= ~p.contains_points(pts)
    mask = water.reshape(XX.shape)
    # a finer mask for marching (25 m) so narrow channels and the island are respected
    fine = 25.0
    fx, fy = np.arange(x0, x1, fine), np.arange(y0, y1, fine)
    FX, FY = np.meshgrid(fx, fy)
    fpts = np.column_stack([FX.ravel(), FY.ravel()])
    fw = po.contains_points(fpts)
    for p in pis:
        fw &= ~p.contains_points(fpts)
    fmask = fw.reshape(FX.shape)

    def wet(x, y):
        i = np.clip(((y - y0) / fine).astype(int), 0, fmask.shape[0] - 1)
        j = np.clip(((x - x0) / fine).astype(int), 0, fmask.shape[1] - 1)
        inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        return fmask[i, j] & inside

    terrain = load_terrain("whatcom_dem.npz", DEM_BOUNDS, ORIGIN, pool=0.0)
    # the lake's own surface: the median ground under the water cells (the tile reaches Bellingham Bay at
    # sea level, so the "flattest low ground" rule finds the wrong water)
    lake = float(np.median(terrain.at(XX[mask], YY[mask])))
    terrain.pool = lake
    print("grid %d x %d, %d water cells; lake level from the DEM %.1f m" % (len(xs), len(ys), int(mask.sum()), lake))

    cx, cy = XX[mask], YY[mask]
    n = len(cx)
    steps = np.arange(fine, 12000.0 + fine, fine)
    ow = canopy.open_water_equivalent(10.0, 1.0)
    out_kH, out_kT, out_f, out_F, out_sx = (np.zeros((16, n)) for _ in range(5))
    out_fs, out_fm = np.zeros((16, n)), np.zeros((16, n))           # shore factor here, and averaged along the fetch
    for k, d in enumerate(DIRS):
        ux, uy = math.sin(math.radians(d)), math.cos(math.radians(d))      # towards where the wind comes from
        F = np.full(n, steps[-1])
        alive = np.ones(n, bool)
        for s in steps:
            w = wet(cx + ux * s, cy + uy * s)
            hit = alive & ~w
            F[hit] = s
            alive &= w
            if not alive.any():
                break
        # terrain shelter: steepest upward angle to the ground within SX_RANGE upwind (beyond the shore)
        sx = np.full(n, -90.0)
        for s in np.arange(60.0, SX_RANGE + 1, 60.0):
            z = terrain.at(cx + ux * s, cy + uy * s) - lake - 2.0
            sx = np.maximum(sx, np.degrees(np.arctan2(z, s)))
        f_terrain = 1.0 - ALPHA * np.clip(sx / 12.0, 0.0, 1.0)
        f_shore = np.clip(canopy.sheltered_speed(10.0, np.maximum(F, 5.0), 1.0, FOREST) / ow, 0.2, 1.0)
        f = f_terrain * f_shore
        # wind averaged along the fetch: the shore factor rises from the bank to this cell; terrain as here
        fetch_mean = np.array([np.mean(np.clip(canopy.sheltered_speed(10.0, np.linspace(5.0, max(Fi, 6.0), 12), 1.0, FOREST) / ow, 0.2, 1.0))
                               for Fi in F]) * f_terrain
        out_kH[k] = 0.0016 * np.sqrt(F / G0) * fetch_mean
        out_kT[k] = 0.286 * G0 ** (-2.0 / 3.0) * F ** (1.0 / 3.0) * fetch_mean ** (1.0 / 3.0)
        out_f[k], out_F[k], out_sx[k] = f, F, sx
        out_fs[k], out_fm[k] = f_shore, fetch_mean / f_terrain
        print("  from %5.1f: fetch median %4.0f m (max %5.0f), wind factor median %.2f, Hs at 10 m/s median %.2f m, max %.2f m"
              % (d, np.median(F), F.max(), np.median(f), 10 * np.median(out_kH[k]), 10 * out_kH[k].max()))

    lat, lon = inverse_tangent_plane(cx, cy, ORIGIN)
    # routes: resample every 20 m, sections of about 500 m
    routes = json.load(open(os.path.join(HERE, "routes.json")))["routes"]
    tree_x, tree_y = cx, cy
    out_routes = []
    for r in routes:
        P = np.array(r["points"])
        rx, ry = local_tangent_plane(P[:, 0], P[:, 1], ORIGIN)
        seg = np.hypot(np.diff(rx), np.diff(ry)); cum = np.concatenate([[0], np.cumsum(seg)])
        L = cum[-1]
        s = np.arange(0, L + 1e-6, 20.0)
        sx_, sy_ = np.interp(s, cum, rx), np.interp(s, cum, ry)
        on_land = ~wet(sx_, sy_)
        nsec = max(int(round(L / 500.0)), 1)
        edges = np.linspace(0, L, nsec + 1)
        secs = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (s >= a) & (s <= b)
            px, py = sx_[m], sy_[m]
            hd = math.degrees(math.atan2(px[-1] - px[0], py[-1] - py[0])) % 360     # bearing of travel
            near = np.zeros(n, bool)
            for qx, qy in zip(px[::3], py[::3]):
                near |= (tree_x - qx) ** 2 + (tree_y - qy) ** 2 < 60.0 ** 2
            if not near.any():
                near = np.argmin((tree_x - px.mean()) ** 2 + (tree_y - py.mean()) ** 2) == np.arange(n)
            mid_lat, mid_lon = inverse_tangent_plane(np.array([px.mean()]), np.array([py.mean()]), ORIGIN)
            secs.append(dict(from_m=round(float(a)), to_m=round(float(b)), heading=round(hd, 1),
                             mid=[round(float(mid_lat[0]), 5), round(float(mid_lon[0]), 5)],
                             kH=[round(float(out_kH[k][near].mean()), 5) for k in range(16)],
                             kH_max=[round(float(out_kH[k][near].max()), 5) for k in range(16)],
                             f=[round(float(out_f[k][near].mean()), 3) for k in range(16)]))
        out_routes.append(dict(key=r["key"], name=r["name"], colour=r["colour"], length_m=round(float(L)),
                               points=r["points"], sections=secs, on_land_m=round(float(on_land.sum() * 20.0))))
        print("%s: %.0f m, %d sections, %.0f m of the traced line on land" % (r["name"], L, len(secs), on_land.sum() * 20.0))

    # the shoreline for the page, lightly thinned
    def thin(ring, step=3):
        return [[round(p[0], 5), round(p[1], 5)] for p in ring[::step]] + [[round(ring[0][0], 5), round(ring[0][1], 5)]]
    out = dict(origin=ORIGIN, grid_m=GRID, directions=DIRS, lake_level_m=round(float(lake), 1),
               cells=dict(lat=np.round(lat, 5).tolist(), lon=np.round(lon, 5).tolist()),
               kH=np.round(out_kH, 5).tolist(), kT=np.round(out_kT, 3).tolist(), f=np.round(out_f, 3).tolist(),
               fetch=np.round(out_F, -1).astype(int).tolist(),
               shore=dict(outer=thin(outer[0]), islands=[thin(r, 1) for r in inner]),
               routes=out_routes,
               method=dict(forest_z0=FOREST.z0, sx_range_m=SX_RANGE, terrain_factor="1 - 0.5 min(Sx/12 deg, 1)",
                           waves="JONSWAP fetch-limited, Hs = 0.0016 U sqrt(gF)/g, Tp = 0.286 (U/g) (gF/U^2)^(1/3)"))
    out["method"]["terrain_alpha"] = ALPHA
    json.dump(out, open(os.path.join(HERE, "whatcom_model.json"), "w"), separators=(",", ":"))
    np.savez_compressed(os.path.join(HERE, "whatcom_components.npz"), x=cx, y=cy, lat=lat, lon=lon, sx=out_sx,
                        f_shore=out_fs, f_shore_mean=out_fm, fetch=out_F, alpha=ALPHA)
    print("written whatcom_model.json (%.0f kB)" % (os.path.getsize(os.path.join(HERE, "whatcom_model.json")) / 1024))


if __name__ == "__main__":
    main()
