"""HOCR 2026: separating the coxswains' skill from what happened to them.

    .venv/Scripts/python.exe research/raceday/skill_separation.py

Three skill measures, each with the externals that act on it taken out as far as the data allow.

1. LINE CHOICE (planning). The intended line (line_smoothness.py, changes over > 160 m) is timed with the
   pace fixed (crew strength out), in the same boat per setting (boat out), in the same conditions per
   setting (weather out) - line_compare.py / line_smoothness.py. Skill: the gap to the setting's best line
   (Radau optimum), (a) in the crew's own conditions (did they choose well for what they had?) and (b)
   averaged over every crew's conditions.
2. STEERING (execution). The heading wobble and sideways weave around the intended line, per 100 m of
   course, regressed on what varies along the course and in time, with a fixed effect per crew:
     felt crosswind and its gust part (ASOS one-minute wind x shelter at the boat; wind_perturbation.py),
       the crosswind also scaled by (3.8 / speed)^2 (how far a crosswind pushes a boat goes as 1/speed^2);
     the course: the smoothed centreline's curvature (bends), the water's width (narrows), a bridge within
       75 m (arches force alignment);
     traffic: crews of the same event within 25 m along the course, and passes (from the start order and
       every crew's official splits; the start interval from two club crews' GPS starts in one event).
   The crew effect is the steering score at common externals; 95% intervals from a block bootstrap
   (blocks of 500 m within each crew).
   What a regression cannot separate from the coxswain because it is fixed per crew - the boat (only one
   eight), the crew's speed, the GPS unit - is handled by physics instead: weave per boat length, drift
   scaled by speed^2, turning priced in a common boat (line_smoothness.py), and a noise-floor check: if all
   of the finest wavelet level (< 20 m) were GPS noise, white noise would put only (0.200^2 + 0.086^2) /
   0.890^2 of that energy into the steering band (B3 starlet detail ratios, Starck & Murtagh).
3. PLAN SMOOTHNESS: the intended line's turning in a common boat, and SPARC (line_smoothness.py).
Writes skill_separation.json.
"""
import json, os, sys
from datetime import timedelta
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C                                       # noqa: E402
from coxswain.raceday import line_ocp as OCP                                 # noqa: E402
import race_card as RC                                                       # noqa: E402
import wind_perturbation as WP                                               # noqa: E402
import line_smoothness as LS                                                 # noqa: E402
from review_crews import CREWS                                               # noqa: E402

SPLIT_STATIONS = (1169.0, 2667.0, 3978.0)
RNG = np.random.default_rng(1)
NOISE_RATIO = (0.200 ** 2 + 0.086 ** 2) / 0.890 ** 2


def start_crossings(D, keys):
    """Each club crew's start-line crossing, ET."""
    out = {}
    for k in keys:
        Z = D[k]
        i0 = int(np.argmax(Z["st"] > 5.0))
        t0 = float(np.interp(0.0, Z["st"][max(i0 - 3, 0):i0 + 3], Z["t"][max(i0 - 3, 0):i0 + 3])) if i0 > 0 else float(Z["t"][0])
        out[k] = CREWS[k]["start_et"] + timedelta(seconds=t0)
    return out


def traffic(D, keys, L):
    EV = json.load(open(os.path.join(HERE, "results", "hocr2026_event_crews.json")))["events"]
    starts = start_crossings(D, keys)
    # the start interval, from two club crews in one event (bows 4 and 7 in event 27)
    pair = [k for k in keys if CREWS[k]["event"] == "27"]
    if len(pair) >= 2:
        a, b = sorted(pair, key=lambda k: CREWS[k]["bow"])[:2]
        gap = (starts[b] - starts[a]).total_seconds() / (CREWS[b]["bow"] - CREWS[a]["bow"])
    else:
        gap = 10.0
    print("start interval from two GPS starts in one event: %.1f s per bow" % gap)
    out = {}
    for k in keys:
        ev = EV[CREWS[k]["event"]]
        mybow = CREWS[k]["bow"]
        my0 = starts[k]
        Z = D[k]
        t_et = np.array([(CREWS[k]["start_et"] + timedelta(seconds=float(t)) - my0).total_seconds() for t in Z["t"]])  # s since our start
        near = np.zeros(len(t_et)); passes = np.zeros(len(t_et))
        for c in ev["crews"]:
            if c["bow"] is None or c["bow"] == mybow:
                continue
            off0 = (c["bow"] - mybow) * gap                      # their start, s after ours
            sp = c["sp"]
            knots_t = [0.0] + [x for x in sp[:3] if x > 0] + [sp[3]]
            knots_s = [0.0] + [s for s, x in zip(SPLIT_STATIONS, sp[:3]) if x > 0] + [L]
            their_s = np.interp(t_et - off0, knots_t, knots_s, left=np.nan, right=np.nan)
            g = their_s - Z["st"]
            ok = np.isfinite(g)
            near[ok] += np.abs(g[ok]) < 25.0
            sg = np.sign(np.where(ok, g, 0.0))
            ch = np.flatnonzero((sg[1:] * sg[:-1] < 0) & ok[1:] & ok[:-1])
            for i in ch:
                passes[max(i - 3, 0):i + 4] += 1                # a pass touches the strokes around it
        out[k] = dict(near=near, passes=passes, n_passes=int(np.sum(np.diff((passes > 0).astype(int)) == 1)))
    return out, gap


def main():
    course = C.CourseModel.build().course
    L = float(course.length)
    W = WP.asos()
    shelter = WP.Shelter(C.CourseModel.build().raster)
    keys = [k for k, c in CREWS.items() if os.path.exists(os.path.join(HERE, c["json"]))]
    D = {k: WP.crew_series(k, course, W, shelter) for k in keys}
    TR, gap = traffic(D, keys, L)
    F = OCP.SmoothFrame(course)
    bridges = [m for _, m in C.CourseModel.build().gates]
    SM = json.load(open(os.path.join(HERE, "line_smoothness.json")))
    rows = []
    for k in keys:
        Z, T = D[k], TR[k]
        v = SM["crews"][k]["own"]["speed"]; Lb = SM["crews"][k]["own"]["boat_length"]
        fs, _ = F.from_course(Z["st"], np.zeros(len(Z["st"])))
        kc = np.abs(np.interp(fs, F.s, F.curvature))
        width = 2 * course.water_half_width_at(Z["st"])
        nearb = np.array([np.min(np.abs(np.array(bridges) - s)) < 75.0 for s in Z["st"]])
        seg = np.floor(Z["st"] / 100.0)
        for sgi in np.unique(seg):
            m = seg == sgi
            if m.sum() < 8:
                continue
            cross = float(np.sqrt(np.nanmean(Z["cross"][m] ** 2)))
            rows.append(dict(crew=k, station=float(sgi * 100 + 50), block=int(sgi // 5),
                             wobble=float(np.sqrt(np.mean(np.degrees(Z["fast"][m]) ** 2))),
                             weave_L=float(np.sqrt(np.mean(Z["d"][m] ** 2)) / Lb),
                             cross=cross, cross_drift=cross * (3.8 / v) ** 2,
                             gust=float(np.sqrt(np.nanmean(Z["gust_c"][m] ** 2))),
                             bend=float(np.mean(kc[m])) * 1000.0, width=float(np.mean(width[m])) / 100.0,
                             bridge=float(np.mean(nearb[m])), near=float(np.mean(T["near"][m] > 0)), passing=float(np.mean(T["passes"][m] > 0))))
    covs = ["cross_drift", "gust", "bend", "width", "bridge", "near", "passing"]
    out = dict(start_interval_s=round(gap, 2), covariates=covs, traffic={k: dict(passes=TR[k]["n_passes"], share_near=round(float(np.mean(TR[k]["near"] > 0)), 3)) for k in keys},
               steering={}, noise={}, line_choice={}, plan={})
    for k in keys:
        print("%-7s traffic: %d passes, a crew within 25 m on %.0f%% of strokes" % (k, TR[k]["n_passes"], 100 * np.mean(TR[k]["near"] > 0)))

    def fit(rows_, y):
        X = np.column_stack([[1.0 if r["crew"] == kk else 0.0 for r in rows_] for kk in keys] + [[r[c] for r in rows_] for c in covs])
        yv = np.array([r[y] for r in rows_])
        ok = np.all(np.isfinite(X), 1) & np.isfinite(yv)
        b = np.linalg.lstsq(X[ok], yv[ok], rcond=None)[0]
        resid = yv[ok] - X[ok] @ b
        r2_ext = 1 - np.var(yv[ok] - X[ok][:, :len(keys)] @ np.linalg.lstsq(X[ok][:, :len(keys)], yv[ok], rcond=None)[0]) / np.var(yv[ok])
        r2_all = 1 - np.var(resid) / np.var(yv[ok])
        xbar = np.nanmean(X[ok][:, len(keys):], 0)
        adj = {kk: float(b[i] + b[len(keys):] @ xbar) for i, kk in enumerate(keys)}
        raw = {kk: float(np.mean(yv[ok][X[ok][:, i] > 0])) for i, kk in enumerate(keys)}
        return b, adj, raw, r2_all, r2_ext
    for y in ("wobble", "weave_L"):
        b, adj, raw, r2_all, r2_crew = fit(rows, y)
        # block bootstrap: resample 500 m blocks within each crew
        boots = []
        for _ in range(1000):
            samp = []
            for k in keys:
                rk = [r for r in rows if r["crew"] == k]
                blocks = sorted(set(r["block"] for r in rk))
                pick = RNG.choice(blocks, len(blocks))
                for bl in pick:
                    samp += [r for r in rk if r["block"] == bl]
            boots.append(fit(samp, y)[1])
        ci = {k: [round(float(np.percentile([bb[k] for bb in boots], 2.5)), 3), round(float(np.percentile([bb[k] for bb in boots], 97.5)), 3)] for k in keys}
        coef = {c: round(float(b[len(keys) + i]), 4) for i, c in enumerate(covs)}
        out["steering"][y] = dict(adjusted={k: round(adj[k], 3) for k in keys}, raw={k: round(raw[k], 3) for k in keys}, ci95=ci, coefficients=coef,
                                  r2_crews_only=round(float(r2_crew), 3), r2_crews_and_externals=round(float(r2_all), 3))
        print("%s: R^2 crews only %.3f, with externals %.3f; coefficients %s" % (y, r2_crew, r2_all, coef))
        for k in sorted(keys, key=lambda kk: adj[kk]):
            print("   %-7s raw %.3f  adjusted %.3f  [%.3f, %.3f]" % (k, raw[k], adj[k], ci[k][0], ci[k][1]))
    # noise floor: the share of the steering band that GPS noise could be, at most
    for k in keys:
        bands = SM["crews"][k]["bands"]
        l1 = bands[0]["heading_rms_deg"] ** 2
        steer_band = bands[1]["heading_rms_deg"] ** 2 + bands[2]["heading_rms_deg"] ** 2
        out["noise"][k] = dict(max_noise_share=round(float(NOISE_RATIO * l1 / steer_band), 3))
    print("noise floor (upper bound on the GPS-noise share of the steering band):", {k: v["max_noise_share"] for k, v in out["noise"].items()})
    # line choice: own conditions and every setting, intended line vs the setting's best line
    for k in keys:
        own = SM["timed"][k]
        out["line_choice"][k] = dict(own_conditions_s=round(own[k]["intended"]["wavelet_%d" % LS.MAIN] - own["best"], 2),
                                     all_settings_s=round(float(np.mean([SM["timed"][s][k]["intended"]["wavelet_%d" % LS.MAIN] - SM["timed"][s]["best"] for s in keys])), 2),
                                     gps_line_own_s=round(own[k]["raw"] - own["best"], 2))
        bs = SM.get("by_setting", {})
        out["plan"][k] = dict(turning_common_boat_s=round(float(np.mean([bs[s]["lines"][k]["plan_turn_s"] for s in keys])), 2) if bs else None,
                              sparc=SM["crews"][k]["own"]["steer"]["sparc"])
        print("%-7s line choice: own conditions %+.1f s, all settings %+.1f s behind the best line; plan turning (common boat) %.2f s"
              % (k, out["line_choice"][k]["own_conditions_s"], out["line_choice"][k]["all_settings_s"], out["plan"][k]["turning_common_boat_s"] or float("nan")))
    out["segments"] = rows
    json.dump(out, open(os.path.join(HERE, "skill_separation.json"), "w"), indent=1)
    print("written skill_separation.json")


if __name__ == "__main__":
    main()
