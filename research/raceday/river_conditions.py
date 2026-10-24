"""What Saturday's river does to each crew's line: flow (National Water Model forecast) and basin level.

Flow: the race runs upstream, so more discharge is a head current (continuity, U = Q/A).
Level: the survey is referenced to normal pool; USGS 01104715 at the dam gives the basin level
(daily mean 2025-09 to 2026-10: median 107.32 ft, 5-95% 106.88-107.78 ft).
Each crew's planned line, still air, at its requalifying pace; seconds against normal (Oct median
flow, median level). Writes river_conditions.json beside this file.
"""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C, flow as FL
from coxswain.river.route import Route
import race_card as RC

FT = 0.3048
flows = [2.8, 5.0, 10.0, 15.3, 20.0, 25.0]          # m3/s; 15.3 = NWM forecast for Sat morning (540 cfs)
levels_ft = [-0.45, -0.25, 0.0, 0.25, 0.45]          # against the median pool
out = {}
models = {lv: C.CourseModel.build(level_offset=lv * FT) for lv in levels_ft}
for crew in ("four", "eight"):
    spec = RC.CREWS[crew]
    plan = json.load(open(os.path.join(RC.FIELDS, "plan_%s.json" % crew)))
    route = Route(np.asarray(plan["route"]["stations"]), np.asarray(plan["route"]["offsets"]))
    boat = RC.build_boat(crew, spec["rate"], profile=RC.PROFILE)
    ref = plan["reference_s"]
    m0 = models[0.0]; m0.route = route
    flow_s = {q: FL.effect_s(m0, boat, ref, q) for q in flows}
    base = FL._line_time(m0, boat, ref)
    level_s = {}
    for lv, m in models.items():
        m.route = route
        level_s[lv] = FL._line_time(m, boat, ref) - base
    out[crew] = dict(flow_s=flow_s, level_s=level_s, reference_s=ref)
    print(crew, "flow:", ", ".join("%.1f m3/s %+.1f s" % (q, v) for q, v in flow_s.items()))
    print(crew, "level:", ", ".join("%+.2f ft %+.1f s" % (lv, v) for lv, v in level_s.items()))
json.dump(out, open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday")), "river_conditions.json"), "w"), indent=1)
