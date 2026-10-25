"""The Charles in 2026: where each crew's time goes, and what this year's optimal line looks like.

    .venv/Scripts/python.exe research/raceday/analysis_2026.py

For the analysis session of 15 Oct 2026. Saturday's conditions as forecast on the 15th: NW wind at
each start (from the race cards), the National Water Model flow at Waltham for Saturday morning, the
basin level now (USGS 01104715). All on the corrected models: the wind charged against each boat's
own resistance, the turning cost measured on our boats, times integrated from speed (not the
evaluator's penalised ``elapsed``).

1. Decomposition per stretch, planned line, requalifying pace: depth (normal pool against 2 m more
   water), current (the forecast flow against slack), wind (forecast against calm), turning (measured).
2. This year's optimal line: the planned line re-optimised in Saturday's wind and flow (warm start),
   and how far it moves.
Writes analysis_2026.json and figures beside this file.
"""
import json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C, plan as P, flow as FL
from coxswain.river import charles
from coxswain.river.route import Route
import race_card as RC

FT = 0.3048
Q_SAT = 16.07                 # m3/s, NWM medium range for Sat 17 Oct 10:30 ET (issued 15 Oct)
Q_SLACK = 0.05
LEVEL_FT = 0.06               # basin against median pool, 15 Oct
DEEP = 2.0                    # m of extra water: the "no shallows" reference


def sections_time(model, boat, ref, wind, k):
    return C._sections(model, C._evaluate(model, boat, ref, wind), wind, k)


def with_flow(model, q):
    model.flow = charles.ContinuityFlow(model.course, discharge=q)
    return model


def main():
    out = {}
    m_lvl = C.CourseModel.build(level_offset=LEVEL_FT * FT)
    m_deep = C.CourseModel.build(level_offset=LEVEL_FT * FT + DEEP)
    for crew in ("four", "eight"):
        spec = RC.CREWS[crew]
        card = json.load(open(os.path.join(RC.FIELDS, "card_%s.json" % crew)))
        wind = (card["wind"]["speed"], card["wind"]["direction"]) if card.get("wind") else None
        plan = json.load(open(os.path.join(RC.FIELDS, "plan_%s.json" % crew)))
        route = Route(np.asarray(plan["route"]["stations"]), np.asarray(plan["route"]["offsets"]))
        boat = RC.build_boat(crew, spec["rate"], profile=RC.PROFILE)   # depth on the wave term only (research, since 2026-10-25)
        k = C.RUDDER_K[spec["boat"]]
        ref = float(plan["reference_s"])
        for m in (m_lvl, m_deep):
            m.route = route

        def run(m, q, w, kk):
            with_flow(m, q)
            return sections_time(m, boat, ref, w, kk)

        base = run(m_lvl, charles.monthly_discharge(10), None, 0.0)       # normal river, still air, no turning cost
        deep = run(m_deep, charles.monthly_discharge(10), None, 0.0)
        slack = run(m_lvl, Q_SLACK, None, 0.0)
        flowq = run(m_lvl, Q_SAT, None, 0.0)
        windy = run(m_lvl, charles.monthly_discharge(10), wind, 0.0)
        turned = run(m_lvl, charles.monthly_discharge(10), None, k)
        allin = run(m_lvl, Q_SAT, wind, k)
        rows = []
        for i, s in enumerate(base):
            rows.append(dict(start=s.start, end=s.end, metres=s.path_m, base_s=s.time_s,
                             depth_s=s.time_s - deep[i].time_s, current_s=flowq[i].time_s - slack[i].time_s,
                             wind_s=windy[i].time_s - s.time_s, turning_s=turned[i].rudder_s,
                             saturday_s=allin[i].time_s))
        tot = {key: sum(r[key] for r in rows) for key in ("base_s", "depth_s", "current_s", "wind_s", "turning_s", "saturday_s")}
        # this year's optimal line: planned arches, re-optimised in Saturday's wind and flow
        with_flow(m_lvl, Q_SAT)
        planned_sat, _ = P._time(m_lvl, boat, route, ref, wind, k)
        opt = P.optimised_time(m_lvl, boat, ref, spec["arches"], k, wind=wind, start_route=route)
        s = np.linspace(0, float(m_lvl.course.length), 300)
        shift = opt["route"].offset_at(s) - route.offset_at(s)
        out[crew] = dict(reference_s=ref, wind=wind, flow_m3s=Q_SAT, level_ft=LEVEL_FT, rows=rows, totals=tot,
                         planned_saturday_s=planned_sat, optimal_saturday_s=opt["time_s"],
                         gain_s=planned_sat - opt["time_s"], mean_shift_m=float(np.mean(np.abs(shift))),
                         max_shift_m=float(shift[int(np.argmax(np.abs(shift)))]),
                         max_shift_at_m=float(s[int(np.argmax(np.abs(shift)))]),
                         planned=dict(stations=list(map(float, route.stations)), offsets=list(map(float, route.offsets))),
                         optimal=dict(stations=list(map(float, np.atleast_1d(opt["route"].stations))),
                                      offsets=list(map(float, np.atleast_1d(opt["route"].offsets)))))
        print("%s: depth %+.1f s, current %+.1f s, wind %+.1f s, turning %+.1f s; Saturday line re-optimised gains %.1f s "
              "(moves %.1f m on average, most %+.0f m at %.0f m)"
              % (crew, tot["depth_s"], tot["current_s"], tot["wind_s"], tot["turning_s"], out[crew]["gain_s"],
                 out[crew]["mean_shift_m"], out[crew]["max_shift_m"], out[crew]["max_shift_at_m"]))
    json.dump(out, open(os.path.join(HERE, "analysis_2026.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
