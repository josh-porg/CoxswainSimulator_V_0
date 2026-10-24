"""The Cambridge bank three ways: lidar (USGS 3DEP, hydro-flattened water), the CRAB chart's water edge,
the model's corridor -- and where the crews rowed.  Negative offset = Cambridge (starboard going up).

    .venv/Scripts/python.exe research/raceday/shallow_shore.py
"""
import json, os, sys
import numpy as np
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C
from coxswain.river.terrain import charles_terrain

FT = 0.3048
from review_crews import CREWS as _CREWS, alias as _alias                 # noqa: E402  private crew list
EIGHT, MSM4 = _alias("eight"), _alias("msm_four")
A = json.load(open(os.path.join(HERE, "race_review_2026.json")))
B = json.load(open(os.path.join(HERE, "line_review_%s.json" % MSM4)))
S = json.load(open(os.path.join(HERE, _CREWS[EIGHT]["json"])))
BL = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_vs_model.json")))
model = C.CourseModel.build(level_offset=A["level_ft"] * FT)
course, raster = model.course, model.raster
T = charles_terrain()
offs = np.arange(0.0, -160.0, -1.0)


def edge(s, side=-1):
    o = side * np.abs(offs)
    xy = course.offset_position(np.full(o.shape, float(s)), o)
    z = T.at(xy[:, 0], xy[:, 1]) - T.pool
    land = z > 0.5
    run = np.convolve(land.astype(int), np.ones(3, int), "same") >= 3       # 3 m of ground, not one pixel
    lidar = float(abs(o[np.argmax(run)])) if run.any() else float("nan")
    wet = np.array([bool(raster.water[raster.index_of(x, y)]) for x, y in xy])
    chart = float(abs(o[np.argmin(wet)])) if (~wet).any() else float("nan")
    nav = np.array([bool(raster.navigable[raster.index_of(x, y)]) for x, y in xy])
    navig = float(abs(o[np.argmin(nav)])) if (~nav).any() else float("nan")
    return lidar, chart, navig


def crew_off(R, s):
    st, of = np.asarray(R["track"]["station"]), np.asarray(R["track"]["offset"])
    m = np.abs(st - s) < 20
    return float(np.median(of[m])) if m.any() else float("nan")


from coxswain.river import charles
from coxswain.river.course import local_tangent_plane
RAW = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_2026.json"), encoding="utf-8"))
bl = []
for key, e in RAW["entries"].items():                    # raw tracks, unclamped
    a = np.asarray(e["rowed"], float)
    x, y = local_tangent_plane(a[:, 0], a[:, 1], charles.CHARLES_ORIGIN)
    st = np.array([course.nearest_station(p, q) for p, q in zip(x, y)])
    of = np.array([course.offset_of(p, q, t) for p, q, t in zip(x, y, st)])
    bl.append((st, of))


def bl_off(s):
    vals = []
    for st, of in bl:
        m = np.abs(st - s) < 25
        if m.any():
            vals.append(float(np.median(of[m])))
    return vals
rows = []
print(" stn   lidar  chart  nav>=1.22  corridor-hw(stbd)  water-hw | ours four_b eight | bestline crews min/med  [port: lidar chart corridor]")
for s in list(range(100, 1000, 50)) + list(range(3800, 4800, 50)):
    li, ch, nv = edge(s)
    lp, cp, _ = edge(s, +1)
    port, stbd = course.limits_at(s)
    offs_bl = bl_off(s) or [float("nan")]
    r = dict(station=s, lidar_bank=li, chart_edge=ch, navigable_edge=nv, corridor=float(stbd),
             water_hw=float(course.water_half_width_at(s)), lidar_bank_port=lp, chart_edge_port=cp, corridor_port=float(port),
             ours=crew_off(A, s), four_b=crew_off(B, s), eight=crew_off(S, s),
             bestline_min=float(np.min(offs_bl)), bestline_med=float(np.median(offs_bl)))
    rows.append(r)
    print("%5d  %5.0f  %5.0f  %6.0f   %8.0f   %8.0f        | %5.0f %5.0f %5.0f | %5.0f %5.0f   [%4.0f %4.0f %4.0f]"
          % (s, li, ch, nv, stbd, r["water_hw"], r["ours"], r["four_b"], r["eight"], r["bestline_min"], r["bestline_med"], lp, cp, port))
json.dump(rows, open(os.path.join(HERE, "shallow_shore.json"), "w"), indent=1)
