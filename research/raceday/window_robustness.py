"""HOCR 2026 club coxswains: do the skill rankings hold along the course, or is one stretch driving them?

    .venv/Scripts/python.exe research/raceday/window_robustness.py

Uses skill_separation.py's measures, windowed:
* windows of WIN m sliding by STEP m along the course (stations 0 to the finish);
* line choice per window: each crew's intended line (line_smoothness.py, wavelet 160 m) timed through the
  window against the setting's best line through the same window, from cumulative time along the course
  (the full model plus the measured turning loss, as card._sections computes it): in the crew's own
  setting, and averaged over every setting;
* steering per window: each crew's mean residual heading wobble and weave per boat length over the 100 m
  segments in the window, after the externals (wind, gust, bend, width, bridge, traffic) at the
  coefficients fitted on the whole course (skill_separation.py), so every window is on the same scale;
* stability: per measure, how often each crew ranks first and last across windows, and Kendall's W (the
  agreement of the windows' rankings, 1 = every window ranks the crews the same);
* leave-one-out: drop each 500 m block in turn, recompute each crew's whole-course score, report any
  block whose removal changes the ranking, and which;
* bad patches: windows where a crew's gap is more than 2 robust SD above its own median.
Writes window_robustness.json.
"""
import json, os, sys
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C                                       # noqa: E402
from coxswain.river import charles                                           # noqa: E402
from coxswain.river.route import Route                                       # noqa: E402
import race_card as RC                                                       # noqa: E402
from review_crews import CREWS                                               # noqa: E402

WIN, STEP, BLOCK = 1000.0, 250.0, 500.0
FT = 0.3048
from review_crews import COX                                 # noqa: E402  names: data/local/raceday/crews.json (private)


def cum_time(model, boat, ref, wind, k, route):
    """Station and cumulative time along a route: the full model's ground speed, slowed by the measured
    turning loss (dv/v = k omega^2), as card._sections does."""
    model.route = route
    res = C._evaluate(model, boat, ref, wind)
    st = np.asarray(res.station)
    pts = model.course.offset_position(st, route.offset_at(st))
    seg = np.hypot(*np.diff(pts, axis=0).T)
    v = np.asarray(res.speed_ground)
    om = C.turn_rate(pts, v)
    vt = v * (1.0 - np.clip(k * om ** 2, 0.0, 0.25))
    dt = seg / np.maximum(0.5 * (vt[:-1] + vt[1:]), 0.05)
    return st, np.concatenate([[0.0], np.cumsum(dt)])


def kendall_w(R):
    """R: windows x crews ranks. Kendall's coefficient of concordance."""
    m, n = R.shape
    S = np.sum((R.sum(0) - m * (n + 1) / 2.0) ** 2)
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def ranks(v):
    return np.argsort(np.argsort(v)) + 1


def main():
    course = C.CourseModel.build().course
    L = float(course.length)
    keys = [k for k, c in CREWS.items() if os.path.exists(os.path.join(HERE, c["json"]))]
    SM = json.load(open(os.path.join(HERE, "line_smoothness.json")))
    SK = json.load(open(os.path.join(HERE, "skill_separation.json")))
    starts = np.arange(0.0, L - WIN + 1, STEP)
    windows = [(float(a), float(a + WIN)) for a in starts]
    # ---- line choice, windowed
    gaps = {}                         # gaps[setting][crew] = array over windows (s)
    for setk in keys:
        R = json.load(open(os.path.join(HERE, CREWS[setk]["json"])))
        cal = R["calibrated"]
        model = C.CourseModel.build(level_offset=R["level_ft"] * FT)
        model.flow = charles.ContinuityFlow(model.course, discharge=R["flow_cfs"] * 0.0283168466)
        spec = RC.CREWS[CREWS[setk]["crew"]]
        boat = RC.build_boat(CREWS[setk]["crew"], spec["rate"], profile="research")
        kk = C.RUDDER_K[spec["boat"]]
        wind = (cal["wind"]["speed"], cal["wind"]["direction"])
        best = Route(np.asarray(cal["best"]["stations"], float), np.asarray(cal["best"]["offsets"], float), name="best")
        sb, tb = cum_time(model, boat, cal["ref_s"], wind, kk, best)
        gaps[setk] = {}
        for k in keys:
            it = SM["crews"][k]["intended"]["wavelet_%d" % SM["main_cutoff"]]
            st = np.concatenate([[0.0], it["st"], [L]]); of = np.concatenate([[it["off"][0]], it["off"], [it["off"][-1]]])
            hw = course.half_width_at(st)
            r = Route(st, np.clip(of, -(hw - 1.0), hw - 1.0), name=k)
            s_, t_ = cum_time(model, boat, cal["ref_s"], wind, kk, r)
            gaps[setk][k] = np.array([(np.interp(b, s_, t_) - np.interp(a, s_, t_)) - (np.interp(b, sb, tb) - np.interp(a, sb, tb)) for a, b in windows])
    own = {k: gaps[k][k] for k in keys}
    common = {k: np.mean([gaps[s][k] for s in keys], axis=0) for k in keys}
    # ---- steering, windowed: residuals after the externals at the whole-course coefficients
    rows = SK["segments"]; covs = SK["covariates"]
    steer = {}
    for y in ("wobble", "weave_L"):
        beta = SK["steering"][y]["coefficients"]
        steer[y] = {}
        for k in keys:
            rk = [r for r in rows if r["crew"] == k]
            resid = np.array([r[y] - sum(beta[c] * r[c] for c in covs) for r in rk]); stn = np.array([r["station"] for r in rk])
            steer[y][k] = np.array([float(np.mean(resid[(stn >= a) & (stn < b)])) if np.any((stn >= a) & (stn < b)) else np.nan for a, b in windows])
    out = dict(win_m=WIN, step_m=STEP, windows=windows, measures={})
    MEAS = {"line_own": own, "line_common": common, "steer_wobble": steer["wobble"], "steer_weave": steer["weave_L"]}
    for name, M in MEAS.items():
        A = np.column_stack([M[k] for k in keys])                     # windows x crews
        ok = np.all(np.isfinite(A), 1)
        Rk = np.array([ranks(row) for row in A[ok]])
        first = {k: round(float(np.mean(Rk[:, i] == 1)), 3) for i, k in enumerate(keys)}
        last = {k: round(float(np.mean(Rk[:, i] == len(keys))), 3) for i, k in enumerate(keys)}
        meanrank = {k: round(float(np.mean(Rk[:, i])), 2) for i, k in enumerate(keys)}
        W = kendall_w(Rk)
        # bad patches: windows more than 2 robust SD above the crew's own median (relative to the field that window)
        rel = A - np.nanmedian(A, 1, keepdims=True)
        patches = {}
        for i, k in enumerate(keys):
            r = rel[:, i]; med = np.nanmedian(r); sd = 1.4826 * np.nanmedian(np.abs(r - med)) + 1e-9
            bad = np.flatnonzero(r > med + 2 * sd)
            patches[k] = [dict(window=windows[j], excess=round(float(r[j] - med), 3)) for j in bad]
        out["measures"][name] = dict(values={k: np.round(M[k], 3).tolist() for k in keys}, rank_first=first, rank_last=last, mean_rank=meanrank,
                                     kendall_w=round(W, 3), bad_patches=patches)
        print("%-12s Kendall W %.2f | mean rank %s | first %s | last %s" % (name, W, meanrank, first, last))
        for k in keys:
            if patches[k]:
                print("     %-7s bad patches: %s" % (k, ", ".join("%d-%d m (+%.2f)" % (p["window"][0], p["window"][1], p["excess"]) for p in patches[k])))
    # ---- leave one 500 m block out: whole-course scores and their ranking
    blocks = np.arange(0.0, L, BLOCK)
    nonover = [(a, min(a + BLOCK, L)) for a in blocks]
    loo = {}
    # line choice whole-course gap by block (exact additive split of the windows' time)
    for name, src in (("line_own", "own"), ("line_common", "common")):
        per_block = {}
        for k in keys:
            vals = []
            for a, b in nonover:
                if src == "own":
                    vals.append(block_gap(k, k, a, b))
                else:
                    vals.append(np.mean([block_gap(s, k, a, b) for s in keys]))
            per_block[k] = np.array(vals)
        full = {k: float(np.sum(per_block[k])) for k in keys}
        base = list(np.array(keys)[np.argsort([full[k] for k in keys])])
        changes = []
        for j, (a, b) in enumerate(nonover):
            dropped = {k: full[k] - per_block[k][j] for k in keys}
            order = list(np.array(keys)[np.argsort([dropped[k] for k in keys])])
            if order != base:
                changes.append(dict(block=[a, b], order=order, contributions={k: round(float(per_block[k][j]), 2) for k in keys}))
        loo[name] = dict(full={k: round(full[k], 2) for k in keys}, order=base, changes=changes,
                         per_block={k: np.round(per_block[k], 2).tolist() for k in keys}, blocks=nonover)
        print("%-12s whole course %s; order %s; %d of %d blocks change the order when dropped%s" % (
            name, {k: round(full[k], 1) for k in keys}, base, len(changes), len(nonover),
            "".join("\n     drop %d-%d m -> %s" % (c["block"][0], c["block"][1], c["order"]) for c in changes)))
    for y, name in (("wobble", "steer_wobble"), ("weave_L", "steer_weave")):
        beta = SK["steering"][y]["coefficients"]
        def score(rows_):
            return {k: float(np.mean([r[y] - sum(beta[c] * r[c] for c in covs) for r in rows_ if r["crew"] == k])) for k in keys}
        full = score(rows); base = sorted(keys, key=lambda k: full[k])
        changes = []
        for a, b in nonover:
            sc = score([r for r in rows if not (a <= r["station"] < b)])
            order = sorted(keys, key=lambda k: sc[k])
            if order != base:
                changes.append(dict(block=[a, b], order=order))
        loo[name] = dict(full={k: round(full[k], 3) for k in keys}, order=base, changes=changes)
        print("%-12s order %s; %d of %d blocks change the order when dropped%s" % (name, base, len(changes), len(nonover),
              "".join("\n     drop %d-%d m -> %s" % (c["block"][0], c["block"][1], c["order"]) for c in changes)))
    out["leave_one_out"] = loo
    json.dump(out, open(os.path.join(HERE, "window_robustness.json"), "w"), indent=1)
    print("written window_robustness.json")


_CACHE = {}


def block_gap(setk, k, a, b):
    """A crew's intended line against the setting's best line over [a, b), cached cumulative times."""
    key = (setk, k)
    if key not in _CACHE:
        course = C.CourseModel.build().course
        L = float(course.length)
        R = json.load(open(os.path.join(HERE, CREWS[setk]["json"])))
        cal = R["calibrated"]
        model = C.CourseModel.build(level_offset=R["level_ft"] * FT)
        model.flow = charles.ContinuityFlow(model.course, discharge=R["flow_cfs"] * 0.0283168466)
        spec = RC.CREWS[CREWS[setk]["crew"]]
        boat = RC.build_boat(CREWS[setk]["crew"], spec["rate"], profile="research")
        kk = C.RUDDER_K[spec["boat"]]
        wind = (cal["wind"]["speed"], cal["wind"]["direction"])
        if ("best", setk) not in _CACHE:
            best = Route(np.asarray(cal["best"]["stations"], float), np.asarray(cal["best"]["offsets"], float), name="best")
            _CACHE[("best", setk)] = cum_time(model, boat, cal["ref_s"], wind, kk, best)
        SM = json.load(open(os.path.join(HERE, "line_smoothness.json")))
        it = SM["crews"][k]["intended"]["wavelet_%d" % SM["main_cutoff"]]
        st = np.concatenate([[0.0], it["st"], [L]]); of = np.concatenate([[it["off"][0]], it["off"], [it["off"][-1]]])
        hw = course.half_width_at(st)
        _CACHE[key] = cum_time(model, boat, cal["ref_s"], wind, kk, Route(st, np.clip(of, -(hw - 1.0), hw - 1.0), name=k))
    s_, t_ = _CACHE[key]; sb, tb = _CACHE[("best", setk)]
    return (np.interp(b, s_, t_) - np.interp(a, s_, t_)) - (np.interp(b, sb, tb) - np.interp(a, sb, tb))


if __name__ == "__main__":
    main()
