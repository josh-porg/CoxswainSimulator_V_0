"""Re-optimise each crew's planned line with freedom through the big turn and after Eliot (15 Oct 2026).

The first optimisation only had free points before Anderson, so the planned line went round the big
turn on the outside. The candidate "inside of the bends" line is 83 m shorter (50 m of it in the big
turn) and 15-25 s faster. The course's buoy lines are not in the model, so the line is kept at least
STAND_OFF metres from the bank (half-width minus margin) as a stand-in for them; the arches stay as
planned. Still air (the plan's basis), then timed in Saturday's wind and flow. Writes plan_<crew>_bends.json.
"""
import json, os, sys
import numpy as np
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C, plan as P
from coxswain.river import charles
from coxswain.river.route import Route
import race_card as RC

STAND_OFF = 15.0
FREE = (1100.0, 1610.0, 2030.0, 2500.0, 3150.0, 3420.0, 3640.0, 3830.0, 4180.0, 4480.0)
m = C.CourseModel.build()
for crew in sys.argv[1:] or ["four", "eight"]:
    spec = RC.CREWS[crew]
    plan = json.load(open(os.path.join(RC.FIELDS, "plan_%s.json" % crew)))
    planned = Route(np.asarray(plan["route"]["stations"]), np.asarray(plan["route"]["offsets"]))
    boat = RC.build_boat(crew, spec["rate"])
    k = C.RUDDER_K[spec["boat"]]
    ref = float(plan["reference_s"])
    t0, _ = P._time(m, boat, planned, ref, None, k)
    opt = P.optimised_time(m, boat, ref, spec["arches"], k, free=FREE, margin=STAND_OFF, maxiter=900, start_route=planned)
    r = opt["route"]
    s = np.linspace(0, float(m.course.length), 600)
    hw = m.course.half_width_at(s)
    off = r.offset_at(s)
    print("%s: planned %.1f s -> %.1f s still air (%+.1f s); path %.0f m; closest to a bank %.1f m; evals %d"
          % (crew, t0, opt["time_s"], opt["time_s"] - t0, opt["path_m"], float(np.min(hw - np.abs(off))), opt["evaluations"]))
    json.dump(dict(arches=spec["arches"], reference_s=ref, stand_off_m=STAND_OFF, free=list(FREE),
                   route=dict(stations=list(map(float, np.atleast_1d(r.stations))), offsets=list(map(float, np.atleast_1d(r.offsets)))),
                   still_air_s=opt["time_s"], planned_still_air_s=t0),
              open(os.path.join(RC.FIELDS, "plan_%s_bends.json" % crew), "w"), indent=1)
