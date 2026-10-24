"""Why does the review optimiser end up slower than bestline's shortest legal line?  (our four, HOCR 2026)

    .venv/Scripts/python.exe research/raceday/optimiser_check.py

Times, in the four's calibrated conditions on the research boat, with the optimiser's own checks:
the shortest line as published; the same line projected onto the optimiser's parametrisation (arch
points fixed at the arch centres, start and finish on the centreline, a free point every 150 m); and the
projection with each restriction lifted in turn.  The gap between the rows is what each restriction costs.
"""
import json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C, plan as P                            # noqa: E402
from coxswain.river import charles, lines, bridges                           # noqa: E402
from coxswain.river.route import Route                                       # noqa: E402
import race_card as RC                                                       # noqa: E402

FT, MARGIN, STEP = 0.3048, 4.0, 150.0
R = json.load(open(os.path.join(HERE, "race_review_2026.json")))
BL = json.load(open(os.path.join(ROOT, "data", "local", "bestline", "bestline_vs_model.json")))
cal = R["calibrated"]
model = C.CourseModel.build(level_offset=R["level_ft"] * FT)
course = model.course
L = float(course.length)
model.flow = charles.ContinuityFlow(course, discharge=R["flow_cfs"] * 0.0283168466)
boat = RC.build_boat("four", RC.CREWS["four"]["rate"], profile="research")
k = C.RUDDER_K["4+"]
wind = (cal["wind"]["speed"], cal["wind"]["direction"])
ref = cal["ref_s"]


def report(name, route):
    t, _ = P._time(model, boat, route, ref, wind, k)
    model.route = route
    ev = C._evaluate(model, boat, ref, wind)
    legal = lines.legalise(route, course, model.raster, model.gates, margin=MARGIN)
    moved = len(np.atleast_1d(legal.stations)) != len(np.atleast_1d(route.stations))
    print("%-58s %7.1f s  path %5.0f m  over-turn %.2f deg/s  illegal arches %d  aground %.3f  legalise moves it: %s"
          % (name, t, P.route_length(model, route), P._over_turn(model, route), ev.illegal_arches,
             ev.fraction_aground, moved))
    return t


short = next(x for x in BL["lines"] if x["name"] == "their shortest line")
sh = Route(np.asarray(short["stations"]), np.asarray(short["offsets"]), name="shortest")
best = Route(np.asarray(cal["best"]["stations"]), np.asarray(cal["best"]["offsets"]), name="optimum")
ours = Route(np.asarray(cal["ours"]["stations"]) if "stations" in cal["ours"] else np.asarray(R["ours"]["stations"]),
             np.asarray(R["ours"]["offsets"]), name="ours")
print("arches the shortest line takes:")
for gate, metres in model.gates:
    arches = bridges.candidate_arches(gate, model.raster)
    here = float(sh.offset_at(np.array([metres]))[0])
    along = gate.station_of(course.offset_position(np.array([metres]), np.array([here]))[0])
    inside = [a.label for a in arches if a.interval[0] <= along <= a.interval[1]]
    print("  %-18s at %5.0f m: offset %+6.1f, inside %s; arches %s" % (gate.name, metres, here, inside or "-",
          [(a.label, round(a.interval[0], 1), round(a.interval[1], 1), round(a.centre, 1)) for a in arches]))
print("our pins:", R["pins"])
if "--fixed" not in sys.argv:
    report("bestline shortest legal line", sh)
    report("review optimum (as saved)", best)
    report("our GPS line", ours)

    pinned = lines.legalise(lines.pinned_arch_route(course, model.raster, model.gates, R["pins"], margin=MARGIN),
                            course, model.raster, model.gates, margin=MARGIN)
    st0, off0 = np.asarray(pinned.stations, float), np.asarray(pinned.offsets, float)
    print("pinned points:", [(round(s), round(o, 1)) for s, o in zip(st0, off0)])
    free = np.array([s for s in np.arange(150.0, L - 100.0, STEP) if np.min(np.abs(st0 - s)) > 60.0])
    limit = np.array([max(course.half_width_at(s) - MARGIN, 0.0) for s in free])


    def build(st_fixed, off_fixed, st_free, off_free):
        st = np.concatenate([st_fixed, st_free]); off = np.concatenate([off_fixed, off_free])
        o = np.argsort(st)
        return Route(st[o], off[o])


    x = np.clip(sh.offset_at(free), -limit, limit)
    report("projection: optimiser's params (arch centres, ends at 0)", build(st0, off0, free, x))
    # lift 1: arch points where the shortest line passes them
    off_arch = off0.copy()
    inner = (st0 > 1.0) & (st0 < L - 1.0)
    off_arch[inner] = sh.offset_at(st0[inner])
    report("  + arch points free within the arch", build(st0, off_arch, free, x))
    # lift 2: start and finish free too
    off_ends = off_arch.copy()
    off_ends[~inner] = sh.offset_at(np.clip(st0[~inner], 20.0, L - 20.0))
    report("  + start and finish free", build(st0, off_ends, free, x))
    # lift 3: denser free points (every 50 m)
    free50 = np.array([s for s in np.arange(50.0, L - 30.0, 50.0) if np.min(np.abs(st0 - s)) > 25.0])
    lim50 = np.array([max(course.half_width_at(s) - MARGIN, 0.0) for s in free50])
    report("  + a free point every 50 m", build(st0, off_ends, free50, np.clip(sh.offset_at(free50), -lim50, lim50)))


# -- after the fixes (2026-10-22): corrected arch rules and plan.optimise_line ---------------------
if __name__ == "__main__" and "--fixed" in sys.argv:
    import time
    print("\nwith the corrected arch rules:")
    report("bestline shortest legal line", sh)
    report("our GPS line", ours)
    t0 = time.time()
    out = P.optimise_line(model, boat, ref, R["pins"], k, wind, starts=(ours, sh))
    print("optimise_line: %.1f s from warm start %r (%s) in %d evaluations, %.0f s"
          % (out["time_s"], out["warm_start"], {n: round(c, 1) for n, c in out["warm_starts"].items()},
             out["evaluations"], time.time() - t0))
    report("new optimum", out["route"])
