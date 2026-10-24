"""Is the planned line still right in Saturday's forecast wind? (NW 7.6 m/s at the start, 12 Oct forecast)

Each crew's planned line (optimised in still air, data/local/raceday/plan_<crew>.json) against the same
arches re-optimised in the forecast wind, both timed in that wind with the measured turning cost.
"""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C, plan as P
from coxswain.river.route import Route
import race_card as RC

WIND = (7.6, 315.0)
m = C.CourseModel.build()
out = {}
for crew in ("four", "eight"):
    spec = RC.CREWS[crew]
    boat = RC.build_boat(crew, spec["rate"])
    k = C.RUDDER_K[spec["boat"]]
    plan = json.load(open(os.path.join(RC.FIELDS, "plan_%s.json" % crew)))
    calm_line = Route(np.array(plan["route"]["stations"]), np.array(plan["route"]["offsets"]))
    ref = plan["reference_s"]
    t_calm_line, _ = P._time(m, boat, calm_line, ref, WIND, k)
    windy = P.optimised_time(m, boat, ref, spec["arches"], k, wind=WIND, start_route=calm_line)
    s = np.linspace(0, float(m.course.length), 400)
    shift = windy["route"].offset_at(s) - calm_line.offset_at(s)
    i = int(np.argmax(np.abs(shift)))
    out[crew] = dict(calm_line_in_wind=t_calm_line, wind_line_in_wind=windy["time_s"],
                     gain_s=t_calm_line - windy["time_s"], mean_shift_m=float(np.mean(np.abs(shift))),
                     max_shift_m=float(shift[i]), max_shift_at_m=float(s[i]),
                     wind_route=dict(stations=list(map(float, np.atleast_1d(windy["route"].stations))),
                                     offsets=list(map(float, np.atleast_1d(windy["route"].offsets)))))
    print(crew, "calm line in the wind %.1f s, wind line %.1f s: re-steering gains %.1f s; line moves %.1f m on average, "
          "most %+.0f m at %.0f m" % (t_calm_line, windy["time_s"], out[crew]["gain_s"], out[crew]["mean_shift_m"],
                                     out[crew]["max_shift_m"], out[crew]["max_shift_at_m"]))
json.dump(out, open(os.path.join(RC.FIELDS, "wind_line_check.json"), "w"), indent=1)
