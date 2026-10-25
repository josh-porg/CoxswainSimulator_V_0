"""Is the shallow-water cost real?  HOCR 2026 per-stroke GPS speed against the model's depth term.

    .venv/Scripts/python.exe research/raceday/shallow_check.py

The race reviews (race_review_opt.py, line_review.py) put ~45 s of the faster crews' 32-33 s gap to the
optimum on depth: the Cambridge-bank water from the start to Powerhouse and after Eliot Bridge, modelled at
1.0-1.5 m.  If that is right, every crew that rowed there should have been measurably slower there than its
own pace elsewhere.  This tests it, stroke by stroke, for the three Sammamish CoxBox tracks:

* the model's speed along each crew's OWN line, in that crew's calibrated conditions (wind scaled to the day
  factor, observed flow and basin level, the crew's pace), twice: with the depth term, and with it off.
  Wind, current and turning are identical in both, so their ratio is the depth term alone, at the depth
  under the boat's actual GPS position (not the corridor-clamped line);
* regress  log(v_gps / v_model_without_depth)  on  log(depth factor)  with log(stroke rate) and crew
  constants.  Slope 1: the boats lost what the model says.  Slope 0: depth cost nothing measurable.
  Standard errors are Newey-West (strokes are serially correlated: gusts, pushes, steering).

Also: the same factor computed three ways -- as the review evaluator applies it (Schlichting factor times
TOTAL resistance), with the factor on the wave term only (what hull_resistance does, and what [D11] says),
and the research profile's Sretenskii finite-depth Michell -- so the size of each is on the table.
Writes shallow_check.json beside this file.
"""
import json, os, sys
import numpy as np
from scipy.optimize import brentq

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, CODE)
from coxswain.raceday import card as C, gusts as GU                          # noqa: E402
from coxswain.river import charles                                           # noqa: E402
from coxswain.river.route import Route                                       # noqa: E402
import race_card as RC                                                       # noqa: E402

FT = 0.3048
G = 9.81
EIGHT_DAY = 1181.1 / 1149.7 * 1.02            # review_crews.py


def track_route(st, off, L, bin_m=25.0):
    """The crew's line as a Route: 25 m bins, median offset (as bestline_vs_model.py)."""
    bins = np.arange(0.0, L + bin_m, bin_m)
    so, oo = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (st >= lo) & (st < hi)
        if m.sum():
            so.append(float(st[m].mean())); oo.append(float(np.median(off[m])))
    so, oo = np.array(so), np.array(oo)
    keep = np.concatenate([[True], np.diff(so) > 1.0])
    so, oo = so[keep], oo[keep]
    if so[0] > 0.5:
        so, oo = np.concatenate([[0.0], so]), np.concatenate([[oo[0]], oo])
    if so[-1] < L - 0.5:
        so, oo = np.concatenate([so, [L]]), np.concatenate([oo, [oo[-1]]])
    return Route(so, oo, name="own line")


def evaluator(model, boat, ref_s, wind):
    """C._evaluate's evaluator, returned rather than run, so its depth table can be read and switched."""
    from coxswain.raceday.card import RouteEvaluator, ReducedModel, ShelteredWind, charles_structures, wind_area
    ev = RouteEvaluator(model.course, flow=model.flow, reference_speed=model.course.length / ref_s,
                        upstream=True, margin=4.0, minimum_depth=1.2, n_samples=900)
    steering = ReducedModel()
    ev.with_steering(steering, raster=model.raster, gates=model.gates)
    ev.with_exertion()
    if wind is not None and wind[0] > 0.3:
        ev.with_wind(ShelteredWind(charles_structures(), model.raster, wind[0], wind[1], height=0.43),
                     drag_area=wind_area(boat, ev.reference_speed, steering))
    return ev


def speeds_along(model, ev, route, k):
    """speed_ground along the route with the rudder loss of C._sections."""
    model.route = route
    res = ev.evaluate(route)
    pts = model.course.offset_position(res.station, res.route.offset_at(res.station))
    om = C.turn_rate(pts, res.speed_ground)
    return res.station, res.speed_ground * (1.0 - np.clip(k * om ** 2, 0.0, 0.25)), res


def calibrate_eight(model, boat, k, R, q):
    T = json.load(open(os.path.join(RC.FIELDS, "wind_table_eight.json")))
    bundle = json.load(open(os.path.join(RC.FIELDS, "site_bundle.json")))["crews"]["eight"]
    river = 1.0 + float(np.interp(q, bundle["flow"]["discharge_m3s"], bundle["flow"]["effect_fraction"]))
    d = R["wind"]["direction"]
    f = lambda v: GU.table_at(T, v, d)["factor"] * river - EIGHT_DAY          # noqa: E731
    v_cal = brentq(f, 0.0, 12.0) if f(0.0) < 0 < f(12.0) else 0.0
    wind = (float(v_cal), float(d))
    line = Route(np.asarray(R["ours"]["stations"]), np.asarray(R["ours"]["offsets"]), name="her line")
    ref = R["race_s"] / EIGHT_DAY
    for _ in range(4):
        model.route = line
        secs = C._sections(model, C._evaluate(model, boat, ref, wind), wind, k)
        ref *= R["race_s"] / sum(z.time_s for z in secs)
    return wind, ref


from review_crews import CREWS as _CREWS, alias as _alias                 # noqa: E402  private crew list
EIGHT, MSM4 = _alias("eight"), _alias("msm_four")
def crews():
    a = json.load(open(os.path.join(HERE, "race_review_2026.json")))
    b = json.load(open(os.path.join(HERE, "line_review_%s.json" % MSM4)))
    s = json.load(open(os.path.join(HERE, _CREWS[EIGHT]["json"])))
    return [
        dict(name="ours (W VET 4+)", crew="four", boat="4+", R=a, level=a["level_ft"], q=a["flow_cfs"] * 0.0283168466,
             wind=(a["calibrated"]["wind"]["speed"], a["calibrated"]["wind"]["direction"]), ref=a["calibrated"]["ref_s"]),
        dict(name="four_b (W MSM 4+)", crew="four", boat="4+", R=b, level=b["conditions"]["level_ft"], q=b["conditions"]["q"],
             wind=(b["calibrated"]["wind"]["speed"], b["calibrated"]["wind"]["direction"]), ref=b["calibrated"]["ref_s"]),
        dict(name="eight (W GM 8+)", crew="eight", boat="8+", R=s, level=s["level_ft"], q=s["flow_cfs"] * 0.0283168466,
             wind=None, ref=None),
    ]


def hac(X, y, lag=15):
    """OLS with Newey-West standard errors."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        G_ = u[l:].T @ u[:-l]
        S += w * (G_ + G_.T)
    cov = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.diag(cov)), e


def wave_only_speed(boat, v_ref, depths, shallow_template):
    """Speed at fixed power with the Schlichting factor on the WAVE term only (hull_resistance's split)."""
    from coxswain.hydro.resistance import hull_resistance
    from coxswain.hydro.shallow import ShallowWaterModel
    sub = boat.mesh.submerged(np.array([0.0, 0.0, boat.equilibrium_heave()]), np.zeros(3),
                              rho=boat.water.density, gravity=9.80665, water_level=0.0)

    def parts(v, h):
        sh = ShallowWaterModel(depth=float(h)) if np.isfinite(h) else None
        _, br = hull_resistance(np.array([float(v), 0.0, 0.0]), sub, mean_wetted_length=boat.length,
                                water=boat.water, coefficients=boat.resistance, shallow=sh,
                                wave_table=getattr(boat, "wave_table", None))
        return br

    deep = parts(v_ref, np.inf)
    total_key = [k for k in deep if "total" in k.lower()]
    tot = lambda br: float(abs(br[total_key[0]])) if total_key else float(sum(abs(x) for x in br.values()))  # noqa: E731
    target = tot(deep) * v_ref
    out = []
    for h in depths:
        f = lambda v: tot(parts(v, h)) * v - target                            # noqa: E731
        out.append(v_ref if f(v_ref) <= 0 else brentq(f, 0.3 * v_ref, v_ref, xtol=1e-6))
    return np.array(out), deep


def main():
    out = dict(crews=[], method=__doc__.split("\n\n")[1])
    rows = []
    for c in crews():
        R = c["R"]
        model = C.CourseModel.build(level_offset=c["level"] * FT)
        course = model.course
        L = float(course.length)
        model.flow = charles.ContinuityFlow(course, discharge=c["q"])
        boat = RC.build_boat(c["crew"], RC.CREWS[c["crew"]]["rate"])
        k = C.RUDDER_K[c["boat"]]
        if c["ref"] is None:
            c["wind"], c["ref"] = calibrate_eight(model, boat, k, R, c["q"])
        tr = R["track"]
        st, off = np.asarray(tr["station"], float), np.asarray(tr["offset"], float)
        v, rate = np.asarray(tr["speed"], float), np.asarray(tr["rate"], float)
        ok = (st > 30.0) & (st < L - 30.0) & (v > 1.0) & (rate > 10.0)
        st, off, v, rate = st[ok], off[ok], v[ok], rate[ok]
        xy = course.offset_position(st, off)
        depth = np.asarray(course.depth(xy[:, 0], xy[:, 1]), float)
        hw = course.half_width_at(st)
        whw = course.water_half_width_at(st)

        route = track_route(st, off, L)
        ev = evaluator(model, boat, c["ref"], c["wind"])
        v_ref = ev.reference_speed
        s_full, v_full, res = speeds_along(model, ev, route, k)
        f_eval = ev.speed_through_water(depth) / v_ref                         # as the reviews apply it
        grid, _ = ev._speed_table
        ev._speed_table = (grid, np.full_like(grid, v_ref))                    # depth term off
        s_nd, v_nd, _ = speeds_along(model, ev, route, k)
        v_nodepth = np.interp(st, s_nd, v_nd)
        ev_calm = evaluator(model, boat, c["ref"], None)
        ev_calm._speed_table = (grid, np.full_like(grid, v_ref))
        s_c, v_c, _ = speeds_along(model, ev_calm, route, k)
        v_calm = np.interp(st, s_c, v_c)                                       # no depth, no wind
        v_model = v_nodepth * f_eval                                           # depth at the GPS position

        # the same depth term, physically split: factor on the wave term only
        hgrid = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0, 6.0])
        v_wave, br = wave_only_speed(boat, v_ref, hgrid, None)
        f_wave = np.interp(depth, hgrid, v_wave / v_ref, left=v_wave[0] / v_ref, right=1.0)
        # research profile: Sretenskii finite-depth Michell
        try:
            from coxswain import physics
            rb = RC.build_boat(c["crew"], RC.CREWS[c["crew"]]["rate"])
            physics.resolve("research").apply(rb)
            from coxswain.river.route import RouteEvaluator
            evr = RouteEvaluator(course, boat=rb, reference_speed=v_ref, upstream=True)
            f_sret = evr.speed_through_water(depth) / v_ref
            sret_ok = True
        except Exception as e:                                                 # pragma: no cover
            print("  research profile path failed:", repr(e)); f_sret = np.full_like(depth, np.nan); sret_ok = False

        crew_rows = dict(name=c["name"], crew=c["crew"], v_ref=v_ref, ref_s=c["ref"], wind=list(c["wind"]),
                         level_ft=c["level"], n=int(len(st)), resistance_breakdown_deep={kk: float(vv) for kk, vv in br.items()})
        fr = v / np.sqrt(G * depth)
        crew_rows["depth_profile"] = dict(
            shallow_share=float(np.mean(depth < 1.6)), median_depth=float(np.median(depth)),
            p10_depth=float(np.percentile(depth, 10)), max_froude=float(fr.max()),
            share_froude_over_0_8=float(np.mean(fr > 0.8)),
            outside_corridor=float(np.mean(np.abs(off) > hw)), outside_clamp=float(np.mean(np.abs(off) > hw - 2.0)),
            outside_water=float(np.mean(np.abs(off) > whw)))
        crew_rows["model_depth_seconds"] = dict(
            evaluator=float(np.sum(np.gradient(st) / v_model) - np.sum(np.gradient(st) / v_nodepth)),
            wave_only=float(np.sum(np.gradient(st) / (v_nodepth * f_wave)) - np.sum(np.gradient(st) / v_nodepth)),
            sretenskii=float(np.nansum(np.gradient(st) / (v_nodepth * f_sret)) - np.sum(np.gradient(st) / v_nodepth)) if sret_ok else None)
        print("%-20s v_ref %.2f  ref %.0f s  n %d  depth<1.6 m %.0f%%  max Fr_h %.2f  depth cost: eval %.1f s, wave-only %.1f s, Sretenskii %s s"
              % (c["name"], v_ref, c["ref"], len(st), 100 * crew_rows["depth_profile"]["shallow_share"], fr.max(),
                 crew_rows["model_depth_seconds"]["evaluator"], crew_rows["model_depth_seconds"]["wave_only"],
                 crew_rows["model_depth_seconds"]["sretenskii"]))

        # stretch table: observed against model, with and without depth
        stretches = []
        edges = [0.0, 975.0, 1600.0, 2400.0, 3200.0, 3900.0, 4400.0, L]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (st >= lo) & (st < hi)
            if m.sum() < 5:
                continue
            stretches.append(dict(from_m=lo, to_m=hi, n=int(m.sum()), depth_med=float(np.median(depth[m])),
                                  offset_med=float(np.median(off[m])), rate=float(np.mean(rate[m])),
                                  obs_over_nodepth=float(np.exp(np.mean(np.log(v[m] / v_nodepth[m])))),
                                  model_depth_factor=float(np.exp(np.mean(np.log(f_eval[m])))),
                                  wave_only_factor=float(np.exp(np.mean(np.log(f_wave[m]))))))
        crew_rows["stretches"] = stretches
        for z in stretches:
            print("   %5.0f-%5.0f  depth %.1f  off %+5.0f  rate %4.1f   gps/model-no-depth %.3f   model depth factor %.3f   wave-only %.3f"
                  % (z["from_m"], z["to_m"], z["depth_med"], z["offset_med"], z["rate"], z["obs_over_nodepth"],
                     z["model_depth_factor"], z["wave_only_factor"]))
        out["crews"].append(crew_rows)
        rows.append(dict(name=c["name"], st=st, off=off, v=v, rate=rate, depth=depth, f=f_eval, fw=f_wave,
                         fs=f_sret, vnd=v_nodepth, vcalm=v_calm, L=L))

    # -- the regression --------------------------------------------------------------------------
    def fit(sel_rows, which="f", extra_phase=True, label="", free_wind=False, trend=False):
        X, y = [], []
        for i, r in enumerate(sel_rows):
            n = len(r["st"])
            cols = [np.ones(n) if j == i else np.zeros(n) for j in range(len(sel_rows))]
            cols.append(np.log(r[which]))
            cols.append(np.log(r["rate"]) - np.mean(np.log(r["rate"])))
            if extra_phase:
                cols.append((r["st"] < 250.0).astype(float))
                cols.append((r["st"] > r["L"] - 400.0).astype(float))
            if free_wind:   # the model's wind term with its own coefficient (0 = wind model right)
                cols.append(np.log(r["vnd"] / r["vcalm"]))
            if trend:       # per-crew fade/build through the race
                for jj in range(len(sel_rows)):
                    cols.append((r["st"] / r["L"] - 0.5) if jj == i else np.zeros(n))
            X.append(np.column_stack(cols)); y.append(np.log(r["v"] / r["vnd"]))
        X, y = np.vstack(X), np.concatenate(y)
        beta, se, e = hac(X, y)
        j = len(sel_rows)
        res = dict(label=label, which=which, slope=float(beta[j]), se=float(se[j]), rate_elasticity=float(beta[j + 1]),
                   rate_se=float(se[j + 1]), n=int(len(y)), resid_sd=float(np.std(e)),
                   spread_of_regressor=float(np.std(X[:, j])))
        if extra_phase:
            res.update(start_dummy=float(beta[j + 2]), finish_dummy=float(beta[j + 3]))
        return res

    fits = []
    for which, lab in (("f", "evaluator factor (Schlichting x total R)"), ("fw", "Schlichting x wave term only"),
                       ("fs", "research: Sretenskii finite depth")):
        if any(not np.all(np.isfinite(r[which])) for r in rows):
            continue
        fits.append(fit(rows, which, True, "pooled, " + lab))
        fits.append(fit(rows, which, False, "pooled, no phase dummies, " + lab))
        fits.append(fit(rows, which, True, "pooled, wind rescaled + fade, " + lab, free_wind=True, trend=True))
        for r in rows:
            fits.append(fit([r], which, True, r["name"] + ", " + lab))
    # within the start stretch only (same wind, same bend, depth varies with the line)
    sub = []
    for r in rows:
        m = (r["st"] > 250.0) & (r["st"] < 975.0)
        sub.append({kk: (vv[m] if isinstance(vv, np.ndarray) else vv) for kk, vv in r.items()})
    fits.append(fit(sub, "f", False, "start stretch 250-975 m only, evaluator factor"))
    fits.append(fit(sub, "fw", False, "start stretch 250-975 m only, wave-only factor"))
    fits.append(fit(sub, "fs", False, "start stretch 250-975 m only, Sretenskii"))
    sub = []
    for r in rows:
        m = (r["st"] > 3800.0) & (r["st"] < r["L"] - 300.0)
        sub.append({kk: (vv[m] if isinstance(vv, np.ndarray) else vv) for kk, vv in r.items()})
    fits.append(fit(sub, "f", False, "Eliot stretch 3800 m to finish-300 only, evaluator factor"))
    fits.append(fit(sub, "fs", False, "Eliot stretch 3800 m to finish-300 only, Sretenskii"))
    out["fits"] = fits
    print()
    for z in fits:
        print("%-75s slope %+.2f +- %.2f (1 = model, 0 = no effect)   rate elasticity %.2f   n %d   regressor sd %.4f"
              % (z["label"], z["slope"], z["se"], z["rate_elasticity"], z["n"], z["spread_of_regressor"]))
    json.dump(out, open(os.path.join(HERE, "shallow_check.json"), "w"), indent=1)
    print("written shallow_check.json")


if __name__ == "__main__":
    main()
