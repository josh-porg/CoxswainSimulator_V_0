"""Where the water costs time on Saturday, and what the possible lines cost: for the briefing map.

    .venv/Scripts/python.exe research/raceday/fields_2026.py

1. Pace fields on a station x offset grid over the raced water (each crew, at its requalifying pace,
   Saturday's forecast flow and level and the wind used on the card): the local pace of rowing at each
   spot, split into what depth, current and wind add, and the total; each as the loss against the
   planned line's own pace at the same station (s/500). Local pace only: the distance to get there is
   in the line comparison, not the field. Raw layers too: depth (m), current along the course (m/s,
   positive = against us), wind felt at 1.5 m (m/s).
2. The possible lines (the project's candidate lines, both arch plans, this year's optimum), each timed
   on Saturday and broken down against the planned line: distance, depth, current, wind, turning.
Writes fields_<crew>.json beside this file (map coordinates as the race card's map).
"""
import json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C, plan as P
from coxswain.hydro.canopy import ShelteredWind
from coxswain.river import charles, lines
from coxswain.river.route import Route, RouteEvaluator
from coxswain.river.structures import charles_structures
import race_card as RC

FT = 0.3048
Q_SAT, Q_SLACK = 16.07, 0.05
LEVEL_FT, DEEP = 0.06, 2.0
DS, OFFSETS = 30.0, np.arange(-72.0, 72.1, 6.0)


def local_eval(model, boat, ref, wind):
    """Pace along a route without steering: the water's own effect on speed."""
    ev = RouteEvaluator(model.course, flow=model.flow, reference_speed=model.course.length / ref,
                        upstream=True, margin=0.0, minimum_depth=0.3, n_samples=700)
    if wind is not None:
        steering = C.ReducedModel()
        # wind_area is sized against the steering model's resistance (critical power / speed, ~840 N);
        # with no steering attached the evaluator divides by its 260 N default instead, which charged
        # the map's wind layers 3.2x (16 Oct). Rescale the area to the denominator actually used.
        v = ev.reference_speed
        area = C.wind_area(boat, v, steering) * float(ev.hydrodynamic_drag(v)) / (steering.critical_power / v)
        ev.with_wind(ShelteredWind(charles_structures(), model.raster, wind[0], wind[1], height=0.43), drag_area=area)
    return ev


LAG_M = 40.0                  # the boat's speed follows the water and wind with about this lag (~8 s at race pace)


def lagged(s, v, length=LAG_M):
    """Speed as the boat carries it: a first-order lag along the direction of travel. The sheltered wind
    changes within metres behind trees and buildings; the boat's momentum does not follow it point by point,
    so the unlagged local pace exaggerates the map's wind swings (to +-90 s/500, 16 Oct)."""
    out = np.array(v, float)
    for i in range(1, len(out)):
        a = 1.0 - np.exp(-abs(s[i] - s[i - 1]) / length)
        out[i] = out[i - 1] + a * (out[i] - out[i - 1])
    return out


def set_flow(m, q):
    m.flow = charles.ContinuityFlow(m.course, discharge=q)


def main():
    m_real = C.CourseModel.build(level_offset=LEVEL_FT * FT)
    m_deep = C.CourseModel.build(level_offset=LEVEL_FT * FT + DEEP)
    course = m_real.course
    L = float(course.length)
    st = np.arange(0.0, L + 1e-6, DS)
    felt_cache = {}
    for crew in ("four", "eight"):
        spec = RC.CREWS[crew]
        card = json.load(open(os.path.join(RC.FIELDS, "card_%s.json" % crew)))
        x0, y0 = card["map"]["origin"]
        wind = (card["wind"]["speed"], card["wind"]["direction"]) if card.get("wind") else None
        plan = json.load(open(os.path.join(RC.FIELDS, "plan_%s.json" % crew)))
        planned = Route(np.asarray(plan["route"]["stations"]), np.asarray(plan["route"]["offsets"]))
        boat = RC.build_boat(crew, spec["rate"])
        k = C.RUDDER_K[spec["boat"]]
        ref = float(plan["reference_s"])

        def pace(route, m, q, w):
            set_flow(m, q)
            r = local_eval(m, boat, ref, w).evaluate(route)
            v = np.interp(st, r.station, lagged(r.station, r.speed_ground))
            return 500.0 / np.maximum(v, 0.3), np.interp(st, r.station, r.depth), np.interp(st, r.station, r.current_along)

        # pace layers along the planned line, the reference for "loss"
        p_line = {}
        p_line["deep"], _, _ = pace(planned, m_deep, Q_SLACK, None)
        p_line["depth"], _, _ = pace(planned, m_real, Q_SLACK, None)
        p_line["current"], _, _ = pace(planned, m_real, Q_SAT, None)
        p_line["wind"], _, _ = pace(planned, m_real, Q_SAT, wind)
        grid = {key: [] for key in ("depth", "current", "wind", "total", "depth_m", "current_ms", "wind_ms", "x", "y", "ok")}
        if wind is not None and wind not in felt_cache:
            felt_cache[wind] = ShelteredWind(charles_structures(), m_real.raster, wind[0], wind[1], height=1.5)
        felt = felt_cache.get(wind)
        half = course.half_width_at(st)
        for off in OFFSETS:
            route = Route.constant_offset(course, float(off), n=40)
            a, _, _ = pace(route, m_deep, Q_SLACK, None)
            b, dep, _ = pace(route, m_real, Q_SLACK, None)
            c_, _, cur = pace(route, m_real, Q_SAT, None)
            d, _, _ = pace(route, m_real, Q_SAT, wind)
            # the loss against the planned line at the same station, cause by cause
            depth_loss = (b - a) - (p_line["depth"] - p_line["deep"])
            current_loss = (c_ - b) - (p_line["current"] - p_line["depth"])
            wind_loss = (d - c_) - (p_line["wind"] - p_line["current"])
            total = d - p_line["wind"]
            xy = course.offset_position(st, np.full(len(st), float(off)))
            ok = (np.abs(off) <= half - 3.0) & (dep >= 0.6)
            grid["depth"].append(np.round(depth_loss, 2)); grid["current"].append(np.round(current_loss, 2))
            grid["wind"].append(np.round(wind_loss, 2)); grid["total"].append(np.round(total, 2))
            grid["depth_m"].append(np.round(dep, 2)); grid["current_ms"].append(np.round(-cur, 3))
            grid["wind_ms"].append(np.round([felt.speed_at(px, py) if felt is not None else 0.0 for px, py in xy], 2))
            grid["x"].append(np.round(xy[:, 0] - x0, 1)); grid["y"].append(np.round(xy[:, 1] - y0, 1))
            grid["ok"].append(ok.astype(int))
        fields = {key: np.asarray(v).tolist() for key, v in grid.items()}
        fields.update(stations=st.tolist(), offsets=OFFSETS.tolist(), wind_forecast=wind, flow_m3s=Q_SAT, level_ft=LEVEL_FT)   # not "wind": that is the loss layer

        # the possible lines, timed on Saturday with the turning cost, and why they differ
        cands = lines.candidate_lines(course, m_real.raster, m_real.gates, margin=4.0)
        others = {c: json.load(open(os.path.join(RC.FIELDS, "plan_%s.json" % c))) for c in ("four", "eight")}
        own = RC.arch_option_of(crew)
        named = [("Planned line: " + RC.ARCH_OPTIONS[own]["name"].replace("Street", "St").replace("Avenue", "Ave"), planned)]
        for key, opt in RC.ARCH_OPTIONS.items():           # the other practised line, optimised for this crew
            alt = os.path.join(RC.FIELDS, "plan_%s%s.json" % (crew, RC.option_suffix(crew, key)))
            if key != own and os.path.exists(alt):
                pl = json.load(open(alt))
                named.append(("Other practised line: " + opt["name"].replace("Street", "St").replace("Avenue", "Ave"),
                              Route(np.asarray(pl["route"]["stations"]), np.asarray(pl["route"]["offsets"]))))
        an = json.load(open(os.path.join(HERE, "analysis_2026.json")))[crew]
        named.append(("This year's optimum (wind and current)", Route(np.asarray(an["optimal"]["stations"]), np.asarray(an["optimal"]["offsets"]))))
        label = {"centreline": "Down the middle", "through the arches": "Arch to arch (straight)", "widest legal arch": "Widest arches",
                 "shortest": "Shortest legal line", "inside of the bends": "Inside of the bends"}
        for r in cands:
            named.append((label.get(r.name, r.name.capitalize()), r))

        def timed(route, m, q, w, kk):
            set_flow(m, q)
            m.route = route
            sec = C._sections(m, C._evaluate(m, boat, ref, w), w, kk)
            return sum(s.time_s for s in sec), sum(s.rudder_s for s in sec), sum(s.path_m for s in sec)

        rows = []
        for name, route in named:
            t_deep, _, path = timed(route, m_deep, Q_SLACK, None, 0.0)
            t_depth, _, _ = timed(route, m_real, Q_SLACK, None, 0.0)
            t_cur, _, _ = timed(route, m_real, Q_SAT, None, 0.0)
            t_wind, _, _ = timed(route, m_real, Q_SAT, wind, 0.0)
            t_all, rud, _ = timed(route, m_real, Q_SAT, wind, k)
            s = np.linspace(0, L, 500)
            xy = course.offset_position(s, route.offset_at(s))
            rows.append(dict(name=name, total_s=t_all, path_m=path, parts=dict(distance=t_deep, depth=t_depth - t_deep,
                             current=t_cur - t_depth, wind=t_wind - t_cur, turning=rud),
                             points=np.round(np.column_stack([xy[:, 0] - x0, xy[:, 1] - y0]), 1).tolist()))
        base = rows[0]
        for r in rows:
            r["delta_s"] = r["total_s"] - base["total_s"]
            r["delta"] = {k2: r["parts"][k2] - base["parts"][k2] for k2 in r["parts"]}
        fields["lines"] = rows
        json.dump(fields, open(os.path.join(HERE, "fields_%s.json" % crew), "w"))
        print(crew, "field %dx%d;" % (len(OFFSETS), len(st)), "; ".join("%s %+.1f s" % (r["name"], r["delta_s"]) for r in rows))


if __name__ == "__main__":
    main()
