"""HOCR 2026 club coxswains: how well did each crew adapt to the conditions? A Monte Carlo of nominal runs.

    .venv/Scripts/python.exe research/raceday/adaptation.py

1. Each crew's speed per 100 m of course, measured (CoxBox), against the course model's speed on the crew's
   own track in calm water (current, depth, the boat; card._evaluate) at the crew's own pace. The physics of
   the wind on the boat comes off too: the model's d log v / d(felt headwind) for that boat, from the model run
   in the race's wind against calm. What is left is the crew's response to the conditions.
2. That response is regressed per segment on the conditions - felt headwind (beyond the physics), felt
   crosswind (scaled by 1/speed^2, how far it pushes the boat), gusts, passing, being passed, a crew within
   25 m - with each crew's own slopes, shrunk towards the pooled slopes (ridge, strength by leave-block-out
   cross-validation; partial pooling, so one crew's few traffic segments cannot run away). Course terms
   (bridges, bend, width) and the stroke rate (effort, each crew's own slope) are controls, not scored.
3. Nominal runs: N synthetic races, each a compilation of conditions drawn Monte Carlo style - weather from
   a random start time in the day's racing (one-minute ASOS at Logan, Bedford and Norwood, felt on the
   course centreline with the bank shelter), and traffic from a real HOCR 2026 field (one of the four
   events' recorded splits) with the crew at a random bow and a random pace within that field. Every crew
   rows the same draws (common random numbers), at a common reference pace, so the comparison is of
   responses only.
4. Adaptation: each crew's time lost to the conditions over the nominal runs - mean (how much the conditions
   cost it) and spread (how much its time varies from run to run) - with 95% intervals from a 500 m block
   bootstrap of the fit, and how often each crew beats each other on the same draw. Also the realised
   picture: in its own race's conditions, what its own response cost against the pooled (typical) response.
Writes adaptation.json (crew keys; compare_page.py anonymises).

MODE (env ADAPT_MODE): "steer" (default, published) or "speed". The speed route above cannot separate the
response from its causes (a crew passes because it is faster; effort rises into the wind), and with one race
each its slopes are not identified (some crews would "gain" 20 s from the conditions). The steer route
measures the response where cause and effect run one way: the conditions make the coxswain steer more,
and the extra steering costs time by the measured turning loss, dv/v = k omega^2 (card.RUDDER_K, the
four's k for every crew, so the seconds compare coxswains, not boats). y = mean squared turn rate of the
steering part (deg/s)^2 per 100 m; time lost = segment time x k x the conditions' share of y.
Writes adaptation_speed.json in speed mode.
"""
import json, os, sys
from datetime import timedelta

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C                                       # noqa: E402
from coxswain.river import charles                                           # noqa: E402
from coxswain.river.route import Route                                       # noqa: E402
import race_card as RC                                                       # noqa: E402
import wind_perturbation as WP                                               # noqa: E402
import skill_separation as SS                                                # noqa: E402
from review_crews import CREWS                                               # noqa: E402

FT = 0.3048
SEG = 100.0
START_SKIP = 250.0                     # the start's acceleration is not a condition
COND = ["cross_drift", "gust", "passing", "passed", "near"]
# headwind is a control, not scored: after the model's wind physics the residual headwind slope is positive
# (crews held speed into the wind better than the drag says), which is effort and rowing, not steering
CTRL = ["head", "bridges", "bend", "width"]
N_RUNS = 4000
N_BOOT = 300
DETREND_SEG = 4.0                      # each crew's slow trend (Gaussian, segments of 100 m) comes off speed and conditions alike
REF_TIME = 1140.0                      # common reference pace for the seconds: 19:00 over the course
RNG = np.random.default_rng(7)
MODE = os.environ.get("ADAPT_MODE", "steer")
K_TURN = 0.0071                        # dv/v per (deg/s)^2, the four's (card.RUDDER_K["4+"])


def flags_on_grid(our_t, grid, field, mybow, gap, L):
    """Traffic along a station grid: someone within 25 m (near), we pass someone (passing), someone passes
    us (passed), from the field's recorded splits (the official split stations) and the start order."""
    near = np.zeros(len(grid)); pas = np.zeros(len(grid)); pd = np.zeros(len(grid))
    for c in field:
        if c["bow"] is None or c["bow"] == mybow or not c["sp"] or c["sp"][3] <= 0:
            continue
        off0 = (c["bow"] - mybow) * gap
        sp = c["sp"]
        kt = [0.0] + [x for x in sp[:3] if x > 0] + [sp[3]]
        ks = [0.0] + [s for s, x in zip(SS.SPLIT_STATIONS, sp[:3]) if x > 0] + [L]
        their = np.interp(our_t - off0, kt, ks, left=np.nan, right=np.nan)
        g = their - grid
        ok = np.isfinite(g)
        near[ok] += np.abs(g[ok]) < 25.0
        sg = np.sign(np.where(ok, g, 0.0))
        for i in np.flatnonzero((sg[1:] * sg[:-1] < 0) & ok[1:] & ok[:-1]):
            m = np.abs(grid - grid[i]) <= 30.0              # a pass touches the 30 m either side
            (pas if sg[i] > 0 else pd)[m] += 1               # they were ahead, now behind: we passed them
    return near > 0, pas > 0, pd > 0


def seg_mean(st, val, edges):
    idx = np.digitize(st, edges) - 1
    return np.array([np.nanmean(val[idx == i]) if np.any(idx == i) else np.nan for i in range(len(edges) - 1)])


def model_speed(k, course, track_st, track_off, wind):
    R = json.load(open(os.path.join(HERE, CREWS[k]["json"])))
    cal = R["calibrated"]
    model = C.CourseModel.build(level_offset=R["level_ft"] * FT)
    model.flow = charles.ContinuityFlow(model.course, discharge=R["flow_cfs"] * 0.0283168466)
    spec = RC.CREWS[CREWS[k]["crew"]]
    boat = RC.build_boat(CREWS[k]["crew"], spec["rate"], profile="research")
    g = np.arange(0.0, course.length + 1e-6, 25.0)
    off = np.array([np.nanmedian(track_off[np.abs(track_st - s) < 12.5]) if np.any(np.abs(track_st - s) < 12.5) else np.nan for s in g])
    ok = np.isfinite(off)
    off = np.interp(g, g[ok], off[ok])
    hw = course.half_width_at(g)
    model.route = Route(g, np.clip(off, -(hw - 1.0), hw - 1.0), name=k)
    calm = C._evaluate(model, boat, cal["ref_s"], None)
    windy = C._evaluate(model, boat, cal["ref_s"], (cal["wind"]["speed"], cal["wind"]["direction"]))
    return (np.asarray(calm.station), np.asarray(calm.speed_ground), np.asarray(windy.station), np.asarray(windy.speed_ground),
            (cal["wind"]["speed"], cal["wind"]["direction"]))


def design(rows, keys, scale):
    """Columns: per crew [intercept, rate, conditions...]; then pooled controls. Conditions standardised."""
    nk = len(keys); p = 2 + len(COND)
    X = np.zeros((len(rows), nk * p + len(CTRL)))
    for i, r in enumerate(rows):
        j = keys.index(r["crew"]) * p
        X[i, j] = 1.0; X[i, j + 1] = r["lograte"]
        for c, name in enumerate(COND):
            X[i, j + 2 + c] = r[name] / scale[name]
        for c, name in enumerate(CTRL):
            X[i, nk * p + c] = r[name]
    return X


def fit(rows, keys, scale, lam):
    """Crew slopes on the conditions shrunk to the pooled slopes: minimise |y - Xb|^2 + lam sum_c |b_c - b_bar|^2
    (b_bar the crews' mean slope), solved jointly with b_bar as free parameters."""
    nk = len(keys); p = 2 + len(COND); q = len(COND)
    X = design(rows, keys, scale); y = np.array([r["y"] for r in rows])
    ncol = X.shape[1]
    Xa = np.hstack([X, np.zeros((len(rows), q))])          # last q columns: b_bar
    pen = []
    for c in range(nk):
        for j in range(q):
            row = np.zeros(ncol + q); row[c * p + 2 + j] = np.sqrt(lam); row[ncol + j] = -np.sqrt(lam)
            pen.append(row)
    A = np.vstack([Xa, np.array(pen)]); b = np.concatenate([y, np.zeros(len(pen))])
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    slopes = {k: sol[i * p + 2: i * p + 2 + q] / np.array([scale[n] for n in COND]) for i, k in enumerate(keys)}
    pooled = sol[ncol:] / np.array([scale[n] for n in COND])
    resid = y - X @ sol[:ncol]
    return slopes, pooled, sol, resid


def nominal_conditions(course, shelter, W, fields, gap, speeds, edges):
    """N_RUNS x segments x conditions, without the crew-specific speed scaling of the crosswind."""
    L = float(course.length)
    mid = 0.5 * (edges[1:] + edges[:-1])
    xy = course.offset_position(mid, np.zeros(len(mid)))
    hd = course.heading_at(mid)
    bearing = (90.0 - np.degrees(hd)) % 360.0
    minutes = sorted(t for t in W if 8 <= t.hour < 17)
    grid = np.arange(0.0, L + 1e-6, 5.0)
    fine = np.digitize(grid, edges) - 1
    out = np.zeros((N_RUNS, len(mid), 6))                     # head, cross, gust, passing, passed, near
    ev_keys = list(fields)
    for r in range(N_RUNS):
        t0 = minutes[RNG.integers(len(minutes) - 25)]
        el = mid / (L / REF_TIME)
        for i in range(len(mid)):
            a, d, g = WP.minute_wind(W, t0 + timedelta(seconds=float(el[i])))
            if not np.isfinite(d):
                continue
            f = shelter(xy[i, 0], xy[i, 1], d)
            rel = np.radians(d - bearing[i])
            out[r, i, 0] = f * a * np.cos(rel); out[r, i, 1] = abs(f * a * np.sin(rel)); out[r, i, 2] = f * max(g - a, 0.0)
        field = fields[ev_keys[RNG.integers(len(ev_keys))]]
        fin = np.array([c["sp"][3] for c in field if c["bow"] is not None and c["sp"] and c["sp"][3] > 0])
        T = float(np.percentile(fin, RNG.uniform(10, 90)))
        bows = [c["bow"] for c in field if c["bow"] is not None]
        mybow = bows[RNG.integers(len(bows))]
        near, pas, pd = flags_on_grid(grid / (L / T), grid, field, mybow, gap, L)
        for j, arr in ((3, pas), (4, pd), (5, near)):
            out[r, :, j] = [np.mean(arr[fine == i]) if np.any(fine == i) else 0.0 for i in range(len(mid))]
    return out


def loss_seconds(conds, slope, v_crew, edges):
    """Seconds lost to the conditions at the common reference pace: sum over segments of
    len / (v_ref exp(dlogv)) - len / v_ref."""
    c = conds.copy()
    c[..., 0] = c[..., 0] * (3.8 / v_crew) ** 2
    dl = c @ slope
    seg = np.diff(edges)
    vref = edges[-1] / REF_TIME
    if MODE == "steer":
        return np.sum(seg / vref * K_TURN * dl, axis=-1)
    return np.sum(seg / vref * (np.exp(-dl) - 1.0), axis=-1)


def main():
    CM = C.CourseModel.build()
    course = CM.course
    L = float(course.length)
    W = WP.asos()
    shelter = WP.Shelter(CM.raster)
    keys = [k for k, c in CREWS.items() if os.path.exists(os.path.join(HERE, c["json"]))]
    D = {k: WP.crew_series(k, course, W, shelter) for k in keys}
    EV = json.load(open(os.path.join(HERE, "results", "hocr2026_event_crews.json")))["events"]
    _, gap = SS.traffic(D, keys, L)
    starts = SS.start_crossings(D, keys)
    SM = json.load(open(os.path.join(HERE, "line_smoothness.json")))
    F = SS.OCP.SmoothFrame(course)
    bridges = [m for _, m in CM.gates]
    edges = np.arange(START_SKIP, L - 20.0, SEG); edges = np.append(edges, L - 20.0)
    rows, speeds, realised = [], {}, {}
    sigma = {}
    for k in keys:
        Z = D[k]
        st = Z["st"]
        s_c, v_c, s_w, v_w, cw = model_speed(k, course, st, Z["off"], None)
        vc = np.interp(st, s_c, v_c); vw = np.interp(st, s_w, v_w)
        # the model's wind physics for this boat: d log v / d(felt headwind), felt under the race's constant wind
        brg = (90.0 - np.degrees(course.heading_at(st))) % 360.0
        hd_cal = np.array([shelter(Z["x"][i], Z["y"][i], cw[1]) * cw[0] * np.cos(np.radians(cw[1] - brg[i])) for i in range(len(st))])
        dlv = np.log(vw / vc)
        sig = float(np.sum(dlv * hd_cal) / np.sum(hd_cal ** 2))
        sigma[k] = sig
        v = Z["v"]; rate = Z["rate"]
        y = np.log(np.maximum(v, 0.5)) - np.log(vc) - sig * Z["head"]
        speeds[k] = SM["crews"][k]["own"]["speed"]
        # traffic on a fine grid from the GPS clock
        my0 = starts[k]
        t_rel = np.array([(CREWS[k]["start_et"] + timedelta(seconds=float(t)) - my0).total_seconds() for t in Z["t"]])
        o = np.argsort(st)
        grid = np.arange(0.0, L + 1e-6, 5.0)
        our_t = np.interp(grid, np.maximum.accumulate(st[o]), t_rel[o])
        near, pas, pd = flags_on_grid(our_t, grid, EV[CREWS[k]["event"]]["crews"], CREWS[k]["bow"], gap, L)
        fs, _ = F.from_course(st, np.zeros(len(st)))
        kc = np.abs(np.interp(fs, F.s, F.curvature)) * 1000.0
        nb = np.array([np.min(np.abs(np.array(bridges) - s)) < 75.0 for s in st]).astype(float)
        oo = np.argsort(Z["t"])
        om = np.empty(len(st)); om[oo] = np.gradient(np.degrees(Z["fast"][oo]), Z["t"][oo])
        if MODE == "steer":
            y = om ** 2
        cols = dict(y=y, lograte=np.log(np.maximum(rate, 10.0)), head=Z["head"], cross=np.abs(Z["cross"]),
                    gust=Z["f"] * np.maximum(Z["gust"] - Z["wind"], 0.0), bend=kc, width=2 * course.water_half_width_at(st) / 100.0, bridges=nb)
        S_ = {n: seg_mean(st, a, edges) for n, a in cols.items()}
        fine = np.digitize(grid, edges) - 1
        for n, a in (("passing", pas), ("passed", pd), ("near", near)):
            S_[n] = np.array([np.mean(a[fine == i]) if np.any(fine == i) else np.nan for i in range(len(edges) - 1)])
        S_["lograte"] = S_["lograte"] - np.nanmedian(S_["lograte"])
        S_["cross_drift"] = S_["cross"] * (3.8 / speeds[k]) ** 2
        # only the local response is fitted: race plan, fatigue and where on the course a crew was (which
        # sets the wind's angle) are slow; passing is caused by being faster, but the pass itself is local
        from scipy.ndimage import gaussian_filter1d
        raw = {n: S_[n].copy() for n in COND}
        for n in (["y", "lograte"] + COND + CTRL if MODE == "speed" else []):
            a = S_[n]; ok = np.isfinite(a)
            if ok.sum() > 3:
                S_[n] = a - gaussian_filter1d(np.interp(np.arange(len(a)), np.flatnonzero(ok), a[ok]), DETREND_SEG, mode="nearest")
        for i in range(len(edges) - 1):
            r = {n: float(S_[n][i]) for n in ["y", "lograte"] + COND + CTRL}
            r.update({"raw_" + n: float(raw[n][i]) for n in COND})
            if all(np.isfinite(list(r.values()))):
                r.update(crew=k, station=float(0.5 * (edges[i] + edges[i + 1])), block=int(edges[i] // 500))
                rows.append(r)
        print("%-7s %d segments; wind physics d log v / d headwind %.4f per m/s; %d passing, %d passed segments"
              % (k, sum(1 for r in rows if r["crew"] == k), sig, sum(1 for r in rows if r["crew"] == k and r["passing"] > 0),
                 sum(1 for r in rows if r["crew"] == k and r["passed"] > 0)))
    scale = {n: float(np.std([r[n] for r in rows])) or 1.0 for n in COND}
    # shrinkage strength by leave-one-block-out cross-validation
    blocks = sorted(set(r["block"] for r in rows))
    cv = {}
    for lam in (0.3, 1, 3, 10, 30, 100, 300, 1000):
        err = 0.0
        for b in blocks:
            tr = [r for r in rows if r["block"] != b]; te = [r for r in rows if r["block"] == b]
            _, _, sol, _ = fit(tr, keys, scale, lam)
            err += float(np.sum((np.array([r["y"] for r in te]) - design(te, keys, scale) @ sol[:design(te, keys, scale).shape[1]]) ** 2))
        cv[lam] = err
    lam = min(cv, key=cv.get)
    print("shrinkage by cross-validation:", {k_: round(v_, 4) for k_, v_ in cv.items()}, "-> lambda", lam)
    slopes, pooled, sol, resid = fit(rows, keys, scale, lam)
    y_all = np.array([r["y"] for r in rows])
    r2 = 1 - np.var(resid) / np.var(y_all - np.array([np.mean([q["y"] for q in rows if q["crew"] == r["crew"]]) for r in rows]))
    print("R^2 within crews: %.3f" % r2)
    # nominal runs
    fields = {e: EV[e]["crews"] for e in EV}
    conds_all = nominal_conditions(course, shelter, W, fields, gap, speeds, edges)
    conds = conds_all[..., 1:]                                # head (index 0) is a control
    edges_full = edges
    loss = {k: loss_seconds(conds, slopes[k], speeds[k], edges_full) for k in keys}
    loss_pool = {k: loss_seconds(conds, pooled, speeds[k], edges_full) for k in keys}
    # bootstrap the fit (500 m blocks within each crew), same draws
    rel = {k: loss[k] - loss_pool[k] for k in keys}
    boot = {k: [] for k in keys}; boot_sd = {k: [] for k in keys}; boot_rel = {k: [] for k in keys}
    for _ in range(N_BOOT):
        samp = []
        for k in keys:
            rk = [r for r in rows if r["crew"] == k]
            bl = sorted(set(r["block"] for r in rk))
            for b in RNG.choice(bl, len(bl)):
                samp += [r for r in rk if r["block"] == b]
        sl, pl, _, _ = fit(samp, keys, scale, lam)
        for k in keys:
            ls = loss_seconds(conds[:800], sl[k], speeds[k], edges_full)
            boot[k].append(float(np.mean(ls))); boot_sd[k].append(float(np.std(ls)))
            boot_rel[k].append(float(np.mean(ls - loss_seconds(conds[:800], pl, speeds[k], edges_full))))
    # realised: own conditions, own response vs the pooled response
    for k in keys:
        rk = [r for r in rows if r["crew"] == k]
        cc = np.array([[r["raw_" + n] for n in COND] for r in rk])   # the conditions met (cross_drift scaled for this crew's speed)
        e = np.array([r["station"] for r in rk])
        seg = np.full(len(rk), SEG); vref = L / REF_TIME
        cost = lambda b_: float(np.sum(seg / vref * (K_TURN * (cc @ b_) if MODE == "steer" else np.exp(-(cc @ b_)) - 1.0)))   # noqa: E731
        own, typ = cost(slopes[k]), cost(pooled)
        per = {n: cost(np.where(np.arange(len(COND)) == j, slopes[k], 0.0)) for j, n in enumerate(COND)}
        realised[k] = dict(own_response_s=round(own, 2), typical_response_s=round(typ, 2), adapted_s=round(typ - own, 2),
                           by_condition_s={n: round(v_, 2) for n, v_ in per.items()},
                           segments=[dict(station=r["station"], resid=round(float(r_), 4)) for r, r_ in zip(rk, resid[[i for i, r in enumerate(rows) if r["crew"] == k]])],
                           stations=[float(s) for s in e])
    pct = lambda a, q: round(float(np.percentile(a, q)), 2)                    # noqa: E731
    allv = np.concatenate([loss[k] for k in keys])
    he = np.linspace(np.floor(np.percentile(allv, 0.5)), np.ceil(np.percentile(allv, 99.5)), 41)
    beats = {a: {b: round(float(np.mean(loss[a] < loss[b])), 3) for b in keys if b != a} for a in keys}
    out = dict(n_runs=N_RUNS, n_boot=N_BOOT, ref_time_s=REF_TIME, lam=lam, cv=cv, r2_within=round(float(r2), 3), conditions=COND, controls=CTRL,
               wind_physics=sigma, start_interval_s=round(gap, 2),
               pooled={n: round(float(v_), 5) for n, v_ in zip(COND, pooled)},
               slopes={k: {n: round(float(v_), 5) for n, v_ in zip(COND, slopes[k])} for k in keys},
               nominal={k: dict(mean_s=round(float(np.mean(loss[k])), 2), sd_s=round(float(np.std(loss[k])), 2),
                                p10_s=pct(loss[k], 10), p50_s=pct(loss[k], 50), p90_s=pct(loss[k], 90),
                                mean_ci95=[pct(boot[k], 2.5), pct(boot[k], 97.5)], sd_ci95=[pct(boot_sd[k], 2.5), pct(boot_sd[k], 97.5)],
                                typical_mean_s=round(float(np.mean(loss_pool[k])), 2),
                                rel_mean_s=round(float(np.mean(rel[k])), 2), rel_sd_s=round(float(np.std(rel[k])), 2),
                                rel_ci95=[pct(boot_rel[k], 2.5), pct(boot_rel[k], 97.5)],
                                hist=np.histogram(loss[k], bins=he)[0].tolist()) for k in keys},
               hist_edges=he.tolist(), beats=beats, realised=realised,
               conditions_summary=dict(cross_ms=[pct(conds_all[..., 1].mean(1), 10), pct(conds_all[..., 1].mean(1), 90)],
                                       gust_ms=[pct(conds_all[..., 2].mean(1), 10), pct(conds_all[..., 2].mean(1), 90)],
                                       passes_per_race=[pct((np.diff((conds_all[..., 3] > 0).astype(int), axis=1) == 1).sum(1), 10), pct((np.diff((conds_all[..., 3] > 0).astype(int), axis=1) == 1).sum(1), 90)],
                                       passed_per_race=[pct((np.diff((conds_all[..., 4] > 0).astype(int), axis=1) == 1).sum(1), 10), pct((np.diff((conds_all[..., 4] > 0).astype(int), axis=1) == 1).sum(1), 90)],
                                       near_share=round(float(np.mean(conds_all[..., 5])), 3)))
    for k in sorted(keys, key=lambda kk: out["nominal"][kk]["mean_s"]):
        n_ = out["nominal"][k]
        print("%-7s vs typical response %+.1f s [%+.1f, %+.1f] (spread %.1f)" % (k, n_["rel_mean_s"], *n_["rel_ci95"], n_["rel_sd_s"]))
        print("%-7s nominal runs: lost %.1f s [%.1f, %.1f], spread %.1f s (P10 %.1f, P90 %.1f); typical response %.1f; realised: own %.1f vs typical %.1f"
              % (k, n_["mean_s"], *n_["mean_ci95"], n_["sd_s"], n_["p10_s"], n_["p90_s"], n_["typical_mean_s"], realised[k]["own_response_s"], realised[k]["typical_response_s"]))
    print("slopes per condition (d log v per unit):", out["pooled"])
    out["mode"] = MODE; out["k_turn"] = K_TURN
    name = "adaptation.json" if MODE == "steer" else "adaptation_speed.json"
    json.dump(out, open(os.path.join(HERE, name), "w"), indent=1)
    print("written", name)


if __name__ == "__main__":
    main()
