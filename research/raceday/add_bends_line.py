"""Add the bend-optimised planned line to each crew's line comparison (fields_<crew>.json), timed on
Saturday with the same breakdown as fields_2026.py, and its offsets for the straightened view."""
import json, os, sys
import numpy as np
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C
from coxswain.river import charles
from coxswain.river.route import Route
import race_card as RC

FT = 0.3048
Q_SAT, Q_SLACK, LEVEL_FT, DEEP = 16.07, 0.05, 0.06, 2.0
m_real = C.CourseModel.build(level_offset=LEVEL_FT * FT)
m_deep = C.CourseModel.build(level_offset=LEVEL_FT * FT + DEEP)
course = m_real.course
NAME = "Planned arches, bends optimised (15 m off the bank)"
for crew in ("four", "eight"):
    spec = RC.CREWS[crew]
    f = json.load(open(os.path.join(HERE, "fields_%s.json" % crew)))
    card = json.load(open(os.path.join(RC.FIELDS, "card_%s.json" % crew)))
    x0, y0 = card["map"]["origin"]
    wind = tuple(f["wind_forecast"]) if f.get("wind_forecast") else None
    pl = json.load(open(os.path.join(RC.FIELDS, "plan_%s_bends.json" % crew)))
    route = Route(np.asarray(pl["route"]["stations"]), np.asarray(pl["route"]["offsets"]))
    boat = RC.build_boat(crew, spec["rate"], profile=RC.PROFILE)   # depth on the wave term only (research, since 2026-10-25)
    k = C.RUDDER_K[spec["boat"]]
    ref = float(pl["reference_s"])

    def timed(m, q, w, kk):
        m.flow = charles.ContinuityFlow(m.course, discharge=q)
        m.route = route
        sec = C._sections(m, C._evaluate(m, boat, ref, w), w, kk)
        return sum(s.time_s for s in sec), sum(s.rudder_s for s in sec), sum(s.path_m for s in sec)

    t_deep, _, path = timed(m_deep, Q_SLACK, None, 0.0)
    t_depth, _, _ = timed(m_real, Q_SLACK, None, 0.0)
    t_cur, _, _ = timed(m_real, Q_SAT, None, 0.0)
    t_wind, _, _ = timed(m_real, Q_SAT, wind, 0.0)
    t_all, rud, _ = timed(m_real, Q_SAT, wind, k)
    s = np.linspace(0, float(course.length), 500)
    xy = course.offset_position(s, route.offset_at(s))
    base = f["lines"][0]
    row = dict(name=NAME, total_s=t_all, path_m=path,
               parts=dict(distance=t_deep, depth=t_depth - t_deep, current=t_cur - t_depth, wind=t_wind - t_cur, turning=rud),
               points=np.round(np.column_stack([xy[:, 0] - x0, xy[:, 1] - y0]), 1).tolist(),
               offsets=np.round(route.offset_at(np.asarray(f["stations"])), 1).tolist())
    row["delta_s"] = row["total_s"] - base["total_s"]
    row["delta"] = {k2: row["parts"][k2] - base["parts"][k2] for k2 in row["parts"]}
    f["lines"] = [base, row] + [L for L in f["lines"][1:] if L["name"] != NAME]
    json.dump(f, open(os.path.join(HERE, "fields_%s.json" % crew), "w"))
    d = row["delta"]
    print("%s: %s %+.1f s on Saturday (distance %+.1f, depth %+.1f, current %+.1f, wind %+.1f, turning %+.1f), path %.0f m"
          % (crew, NAME, row["delta_s"], d["distance"], d["depth"], d["current"], d["wind"], d["turning"], path))
