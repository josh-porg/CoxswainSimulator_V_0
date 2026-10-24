"""Add the straightened-course view's data to fields_<crew>.json (no model runs).

Per station of the field grid: the channel's half-width, each line's offset from the centreline
(positive = port), and the stations of the landmarks and bridges. The lines are rebuilt the way
fields_2026.py built them, in the same order, plus the bend-optimised plan when it exists.
"""
import json, os, sys
import numpy as np
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C
from coxswain.river import lines
from coxswain.river.route import Route
import race_card as RC

m = C.CourseModel.build()
course = m.course
for crew in ("four", "eight"):
    path = os.path.join(HERE, "fields_%s.json" % crew)
    f = json.load(open(path))
    st = np.asarray(f["stations"])
    f["half_width"] = np.round(course.half_width_at(st), 1).tolist()
    plan = json.load(open(os.path.join(RC.FIELDS, "plan_%s.json" % crew)))
    routes = [Route(np.asarray(plan["route"]["stations"]), np.asarray(plan["route"]["offsets"]))]
    for key in RC.ARCH_OPTIONS:                           # as fields_2026.py: the other practised line
        alt = os.path.join(RC.FIELDS, "plan_%s%s.json" % (crew, RC.option_suffix(crew, key)))
        if key != RC.arch_option_of(crew) and os.path.exists(alt):
            pl = json.load(open(alt))
            routes.append(Route(np.asarray(pl["route"]["stations"]), np.asarray(pl["route"]["offsets"])))
    an = json.load(open(os.path.join(HERE, "analysis_2026.json")))[crew]
    routes.append(Route(np.asarray(an["optimal"]["stations"]), np.asarray(an["optimal"]["offsets"])))
    routes += list(lines.candidate_lines(course, m.raster, m.gates, margin=4.0))
    for L, r in zip(f["lines"], routes):
        L["offsets"] = np.round(r.offset_at(st), 1).tolist()
    card = json.load(open(os.path.join(RC.FIELDS, "card_%s.json" % crew)))
    f["marks"] = [dict(name=s["end"], s=s["to_m"]) for s in card["sections"]]
    f["sections"] = [dict(start=s["start"], end=s["end"], a=s["from_m"], b=s["to_m"]) for s in card["sections"]]
    f["bridges"] = [dict(name=g.name, s=float(d)) for g, d in m.gates]
    json.dump(f, open(path, "w"))
    print(crew, "added half-width, %d line offsets, %d marks, %d bridges" % (len(f["lines"]), len(f["marks"]), len(f["bridges"])))
