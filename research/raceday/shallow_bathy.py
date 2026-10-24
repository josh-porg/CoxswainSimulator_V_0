"""Where the crews row near the Cambridge bank, what the CRAB contours actually say there.

    .venv/Scripts/python.exe research/raceday/shallow_bathy.py

For cross-sections on the start stretch and after Eliot Bridge: modelled depth across the river, the
navigable (>=1.22 m) and water half-widths, the crews' GPS offsets, and -- for each crew position -- which
contours the depth was interpolated between and how far away their nearest vertices are.  A depth read
off a triangle whose vertices are all the shoreline contour and one deeper contour 30 m away is an
interpolation, not a sounding.
"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree, Delaunay
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C
from coxswain.river import charles

FT = 0.3048
from review_crews import CREWS as _CREWS, alias as _alias                 # noqa: E402  private crew list
EIGHT, MSM4 = _alias("eight"), _alias("msm_four")
A = json.load(open(os.path.join(HERE, "race_review_2026.json")))
B = json.load(open(os.path.join(HERE, "line_review_%s.json" % MSM4)))
S = json.load(open(os.path.join(HERE, _CREWS[EIGHT]["json"])))
model = C.CourseModel.build(level_offset=A["level_ft"] * FT)
course, raster = model.course, model.raster
pts, dep = charles.load_isobaths()
tri = Delaunay(pts)
tree = cKDTree(pts)
out = dict(sections=[], positions=[])
for s0 in (150, 300, 450, 600, 750, 900, 3950, 4100, 4250, 4400, 4600):
    offs = np.arange(-95.0, 96.0, 5.0)
    xy = course.offset_position(np.full(offs.shape, float(s0)), offs)
    d = np.asarray(course.depth(xy[:, 0], xy[:, 1]))
    wet = np.array([bool(raster.water[raster.index_of(x, y)]) for x, y in xy])
    gps = {}
    for nm, R in (("ours", A), ("four_b", B), ("eight", S)):
        st, of = np.asarray(R["track"]["station"]), np.asarray(R["track"]["offset"])
        m = np.abs(st - s0) < 15
        gps[nm] = float(np.median(of[m])) if m.any() else None
    row = dict(station=s0, hw=float(course.half_width_at(s0)), water_hw=float(course.water_half_width_at(s0)),
               gps=gps, profile=[(float(o), float(z) if w else None) for o, z, w in zip(offs, d, wet)])
    out["sections"].append(row)
    print("s=%4d  corridor hw %.0f  water hw %.0f  gps offsets %s" % (s0, row["hw"], row["water_hw"], gps))
    print("   " + " ".join("%+d:%s" % (o, ("%.1f" % z) if w else "--") for o, z, w in zip(offs, d, wet) if o <= 10))
# each crew position on the shallow stretches: which contours bound it
for nm, R in (("ours", A), ("four_b", B), ("eight", S)):
    st, of = np.asarray(R["track"]["station"]), np.asarray(R["track"]["offset"])
    m = ((st > 100) & (st < 975)) | ((st > 3900) & (st < 4800))
    xy = course.offset_position(st[m], of[m])
    simp = tri.find_simplex(xy)
    vd = dep[tri.simplices[simp]]                       # the three contour depths at each position
    vx = pts[tri.simplices[simp]]
    span = np.max(np.linalg.norm(vx - xy[:, None, :], axis=2), axis=1)
    shore = np.isclose(vd, dep.min()).any(axis=1)
    flat = np.ptp(vd, axis=1) < 1e-6
    dshore = cKDTree(pts[np.isclose(dep, dep.min())]).query(xy)[0]
    d = np.asarray(course.depth(xy[:, 0], xy[:, 1]))
    r = dict(crew=nm, n=int(m.sum()), median_depth=float(np.median(d)),
             share_triangle_touches_shoreline_contour=float(shore.mean()),
             share_flat_triangle=float(flat.mean()), median_vertex_distance_m=float(np.median(span)),
             median_distance_to_1ft_contour_m=float(np.median(dshore)),
             contour_depths_ft_median=[float(x) for x in np.median(np.sort(vd, axis=1), axis=0) / FT])
    out["positions"].append(r)
    print(r)
json.dump(out, open(os.path.join(HERE, "shallow_bathy.json"), "w"), indent=1)
