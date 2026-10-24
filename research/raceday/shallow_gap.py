"""What the line gap becomes when depth moves only the wave term (research profile, Sretenskii).

    .venv/Scripts/python.exe research/raceday/shallow_gap.py [ours|four_b|eight ...]

For each crew, in its calibrated conditions (wind and flow exactly as race_review_opt.py / line_review.py
used): the pace is re-set so the crew's own line takes its GPS time under each physics, then the optimum
through the crew's own arches is found by the same Powell search (a free point every 150 m, 4 m inside
the corridor, warm-started from the crew's line).  Two physics:

  default   -- the review evaluator as it ran (Schlichting factor on the whole resistance);
  research  -- the boat with physics.resolve("research") applied, so card._evaluate hands it to the
               evaluator and depth moves the Sretenskii wave term only.

Writes shallow_gap.json (merged per crew).  A few minutes per crew per physics.
"""
import json, os, sys
import numpy as np
from scipy.optimize import minimize

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, CODE)
from coxswain import physics                                                  # noqa: E402
from coxswain.raceday import card as C, plan as P                            # noqa: E402
from coxswain.river import charles, lines                                    # noqa: E402
from coxswain.river.route import Route                                       # noqa: E402
import race_card as RC                                                       # noqa: E402
import shallow_check as SC                                                   # noqa: E402

FT = 0.3048
MARGIN, STEP = 4.0, 150.0


def setup(name):
    c = {x["name"].split()[0]: x for x in SC.crews()}[name]
    R = c["R"]
    if name == "four_b":
        own = R["routes"]["line"]; race_s = R["model_course_s"]
    else:
        own = R["ours"]; race_s = R["race_s"]
    return c, R, Route(np.asarray(own["stations"], float), np.asarray(own["offsets"], float), name="own line"), race_s


def run(name, profile):
    c, R, own, race_s = setup(name)
    model = C.CourseModel.build(level_offset=c["level"] * FT)
    course = model.course
    L = float(course.length)
    model.flow = charles.ContinuityFlow(course, discharge=c["q"])
    boat = RC.build_boat(c["crew"], RC.CREWS[c["crew"]]["rate"])
    k = C.RUDDER_K[c["boat"]]
    if c["ref"] is None:
        c["wind"], c["ref"] = SC.calibrate_eight(model, boat, k, R, c["q"])
    wind = tuple(c["wind"])
    if profile == "research":
        physics.resolve("research").apply(boat)
    ref = c["ref"]
    for _ in range(4):
        t, _ = P._time(model, boat, own, ref, wind, k)
        ref *= race_s / t
    t_own, _ = P._time(model, boat, own, ref, wind, k)

    pinned = lines.legalise(lines.pinned_arch_route(course, model.raster, model.gates, R["pins"], margin=MARGIN),
                            course, model.raster, model.gates, margin=MARGIN)
    st0, off0 = np.asarray(pinned.stations, float), np.asarray(pinned.offsets, float)
    free = np.array([s for s in np.arange(150.0, L - 100.0, STEP) if np.min(np.abs(st0 - s)) > 60.0])
    limit = np.array([max(course.half_width_at(s) - MARGIN, 0.0) for s in free])

    def route_of(x):
        st = np.concatenate([st0, free]); off = np.concatenate([off0, np.clip(x, -limit, limit)])
        o = np.argsort(st)
        return Route(st[o], off[o], name="optimum")

    def cost(x):
        r = route_of(x)
        legal = lines.legalise(r, course, model.raster, model.gates, margin=MARGIN)
        if len(np.atleast_1d(legal.stations)) != len(np.atleast_1d(r.stations)):
            return 1e6
        t, _ = P._time(model, boat, r, ref, wind, k)
        return t + 100.0 * P._over_turn(model, r)

    x0 = np.clip(own.offset_at(free), -limit, limit)
    c0 = cost(x0)
    res = minimize(cost, x0, method="Powell", bounds=list(zip(-limit, limit)),
                   options=dict(maxfev=5000, xtol=0.3, ftol=1e-4))
    best = route_of(res.x)
    t_best, _ = P._time(model, boat, best, ref, wind, k)
    # how much of each line's time is depth, under this physics: the same line in water 2 m deeper
    deep = C.CourseModel.build(level_offset=c["level"] * FT + 2.0)
    deep.flow = charles.ContinuityFlow(deep.course, discharge=c["q"])
    d_own = t_own - P._time(deep, boat, own, ref, wind, k)[0]
    d_best = t_best - P._time(deep, boat, best, ref, wind, k)[0]
    s = np.linspace(0, L, 60)
    out = dict(profile=profile, ref_s=ref, own_s=t_own, optimum_s=t_best, gap_s=t_own - t_best,
               warm_start_s=c0, depth_own_s=d_own, depth_optimum_s=d_best,
               optimum_offsets_60=[round(float(v), 1) for v in best.offset_at(s)],
               own_offsets_60=[round(float(v), 1) for v in own.offset_at(s)],
               stations_60=[round(float(v), 0) for v in s],
               optimum=dict(stations=list(map(float, best.stations)), offsets=list(map(float, best.offsets))))
    print("%-7s %-9s pace %.0f s  own %.1f  optimum %.1f  gap %+.1f s   depth: own %.1f optimum %.1f"
          % (name, profile, ref, t_own, t_best, t_own - t_best, d_own, d_best), flush=True)
    return out


def main():
    names = sys.argv[1:] or ["ours", "four_b", "eight"]
    path = os.path.join(HERE, "shallow_gap.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for n in names:
        out[n] = {p: run(n, p) for p in ("default", "research")}
        json.dump(out, open(path, "w"), indent=1)
    print("written", path)


if __name__ == "__main__":
    main()
