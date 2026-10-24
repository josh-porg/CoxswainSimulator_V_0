"""HOCR 2026: was the lateral movement steering, or the gusty wind? And did shore lines shelter crews?

    .venv/Scripts/python.exe research/raceday/wind_perturbation.py

Inputs per crew (review_crews.CREWS): the per-stroke CoxBox track and its intended-line / steering split
(line_smoothness.py: the fast heading and the sideways weave d around the intended line, in the boat's
own frame), and the one-minute ASOS wind at KBOS, KBED and KOWD (IEM asos1min: two-minute mean speed,
direction, and the minute's peak five-second gust), results/asos1min_*_20261017.csv.

Per stroke:
* the felt wind: the minute's wind (station mean speed and gust, KBOS direction) times the shelter factor
  at the boat's position for that direction (coxswain.hydro.canopy.ShelteredWind at 1.5 m: fetch to the
  upwind bank, the bank's buildings and trees), split into headwind and crosswind against the boat's
  intended heading. Crosswind is signed + from starboard, so a crosswind pushes the boat to port (+d);
* gustiness: the minute's gust minus its mean speed, felt (times the shelter factor);
* the boat as an anemometer: the stroke's speed against a smooth trend and the stroke rate (a robust
  regression); a residual below -2.5 robust SD is a "hit" not explained by the crew's rate.

Tests:
A. Shelter: each line's mean shelter factor in a common wind (330 deg) and in its own; shelter against the
   distance to the nearest bank.
B. Direction: the correlation of the sideways weave d with the felt crosswind (both high-passed like the
   steering), against a null of circular shifts of d (keeps its own structure). Wind-driven drift has the
   crosswind's sign; steering has no reason to.
C. Perturbation share: d regressed on the felt crosswind and its gust term, lags 0-3 strokes; the R^2 is the
   share of the weave the wind explains (the rest is steering, and GPS noise).
D. Hits: the signed weave after speed hits (signed by the crosswind side) against the null.
E. Segments of 100 m pooled over crews: heading wobble against felt crosswind and gustiness, with crew
   fixed effects (does a more sheltered stretch let a coxswain steer straighter?).
Limits: the stations are 5-25 km away; they give each crew's gust climate by the minute, not the gust that
hit a particular stroke. Stroke-level timing rests on the boat's own speed (test D) and the wind's sign (B).
Writes wind_perturbation.json and wind_perturbation.png.
"""
import csv, json, os, sys
from datetime import datetime, timedelta, timezone
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CODE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from coxswain.raceday import card as C                                       # noqa: E402
from coxswain.river import charles                                           # noqa: E402
from coxswain.river.course import local_tangent_plane                        # noqa: E402
from coxswain.river.structures import charles_structures                     # noqa: E402
from coxswain.hydro.canopy import ShelteredWind                              # noqa: E402
import race_review_2026 as RR                                                # noqa: E402
import line_smoothness as LS                                                 # noqa: E402
from review_crews import CREWS                                               # noqa: E402

EDT = timezone(timedelta(hours=-4))
KT = 0.514444
from review_crews import COX                                 # noqa: E402  names: data/local/raceday/crews.json (private)
RNG = np.random.default_rng(0)


def asos():
    out = {}
    for st in ("BOS", "BED", "OWD"):
        p = os.path.join(HERE, "results", "asos1min_%s_20261017.csv" % st)
        for r in csv.DictReader(open(p)):
            try:
                t = datetime.strptime(r["valid(America/New_York)"], "%Y-%m-%d %H:%M").replace(tzinfo=EDT)
                out.setdefault(t, {})[st] = (float(r["sknt"]) * KT, float(r["drct"]), float(r["gust_sknt"]) * KT)
            except (ValueError, KeyError):
                pass
    return out


def minute_wind(W, t):
    m = t.replace(second=0, microsecond=0)
    for dt in (0, -1, 1, -2, 2):
        v = W.get(m + timedelta(minutes=dt))
        if v:
            sp = np.mean([a[0] for a in v.values()]); gu = np.mean([a[2] for a in v.values()])
            d = v["BOS"][1] if "BOS" in v else np.mean([a[1] for a in v.values()])
            return sp, d, gu
    return np.nan, np.nan, np.nan


class Shelter:
    """Shelter factor (felt / open wind at 10 m) at a point for a wind direction, cached per 10 degrees."""
    def __init__(self, raster):
        self.raster, self.s, self.models = raster, charles_structures(), {}

    def __call__(self, x, y, wind_from):
        b = int(round(wind_from / 10.0)) * 10 % 360
        if b not in self.models:
            self.models[b] = ShelteredWind(self.s, self.raster, 1.0, float(b), height=1.5)
        return float(self.models[b].speed_at(float(x), float(y)))


def hp(sig, cut=LS.MAIN):
    """High-pass on the stroke series: minus its starlet smooth at the steering cut-off (strokes ~ DS)."""
    return sig - LS.starlet(sig, LS.LEVELS[cut])[0]


def crew_series(k, course, W, shelter):
    cfg = CREWS[k]
    T = LS.track(k, course)
    slow, fast, d, xi, yi = LS.decompose(T, LS.MAIN, "wavelet")
    RR.COXBOX = cfg["coxbox"]
    S = RR.strokes()
    x, y = local_tangent_plane(S[:, 4], S[:, 5], charles.CHARLES_ORIGIN)
    st = np.array([course.nearest_station(a, b) for a, b in zip(x, y)])
    race = (st > 2.0) & (st < course.length - 2.0)
    S, x, y, st = S[race], x[race], y[race], st[race]
    from scipy.spatial import cKDTree
    _, idx = cKDTree(np.column_stack([T["x"], T["y"]])).query(np.column_stack([x, y]))
    heading = slow[idx]                                     # maths angle of the intended line
    bearing = (90.0 - np.degrees(heading)) % 360.0
    t_et = [cfg["start_et"] + timedelta(seconds=float(t)) for t in S[:, 0]]
    sp, wd, gu, f, head, cross, gust_c, gust_h = ([] for _ in range(8))
    for i in range(len(S)):
        a, dirn, g = minute_wind(W, t_et[i])
        fi = shelter(x[i], y[i], dirn) if np.isfinite(dirn) else np.nan
        rel = np.radians(dirn - bearing[i])                 # wind from, relative to the boat's heading
        sp.append(a); wd.append(dirn); gu.append(g); f.append(fi)
        head.append(fi * a * np.cos(rel)); cross.append(fi * a * np.sin(rel))
        gust_h.append(fi * (g - a) * np.cos(rel)); gust_c.append(fi * (g - a) * np.sin(rel))
    hw = course.water_half_width_at(st)
    off = np.array([course.offset_of(a, b, s) for a, b, s in zip(x, y, st)])
    # the boat as an anemometer: speed against a smooth trend and the stroke rate
    from scipy.ndimage import gaussian_filter1d
    v, rate = S[:, 2], S[:, 3]
    vt, rt = gaussian_filter1d(v, 8.0, mode="nearest"), gaussian_filter1d(rate, 8.0, mode="nearest")
    A = np.column_stack([np.ones(len(v)), rate - rt])
    beta = np.linalg.lstsq(A, v - vt, rcond=None)[0]
    res = v - vt - A @ beta
    sd = 1.4826 * np.median(np.abs(res - np.median(res)))
    hits = np.flatnonzero(res < -2.5 * sd)
    return dict(st=st, x=x, y=y, t=S[:, 0], d=d[idx], fast=fast[idx], f=np.array(f), head=np.array(head), cross=np.array(cross),
                gust_h=np.array(gust_h), gust_c=np.array(gust_c), wind=np.array(sp), gust=np.array(gu), wdir=np.array(wd),
                bank=hw - np.abs(off), off=off, res=res, hits=hits, v=v, rate=rate, rate_beta=float(beta[1]), res_sd=float(sd),
                window=[t_et[0].strftime("%H:%M"), t_et[-1].strftime("%H:%M")])


def shift_null(stat, a, b, n=2000):
    obs = stat(a, b)
    null = np.array([stat(np.roll(a, RNG.integers(30, len(a) - 30)), b) for _ in range(n)])
    return float(obs), float(np.mean(np.abs(null) >= abs(obs))), float(np.std(null))


def main():
    course = C.CourseModel.build().course
    W = asos()
    shelter = Shelter(C.CourseModel.build().raster)
    keys = [k for k, c in CREWS.items() if os.path.exists(os.path.join(HERE, c["json"]))]
    out = dict(crews={}, pooled={})
    D = {}
    for k in keys:
        Z = crew_series(k, course, W, shelter)
        D[k] = Z
        ok = np.isfinite(Z["cross"])
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])                   # noqa: E731
        dh, ch, gh = hp(Z["d"]), hp(np.nan_to_num(Z["cross"])), hp(np.nan_to_num(Z["gust_c"]))
        r_dir, p_dir, sd_dir = shift_null(corr, dh, ch)
        r_gust, p_gust, _ = shift_null(corr, dh, gh)
        # perturbation share: d on crosswind and crosswind gust, lags 0-3 strokes
        X = [np.ones(len(dh))]
        for lag in range(4):
            X += [np.roll(ch, lag), np.roll(gh, lag)]
        X = np.column_stack(X)[4:]; yv = dh[4:]
        coef = np.linalg.lstsq(X, yv, rcond=None)[0]
        r2 = 1 - np.var(yv - X @ coef) / np.var(yv)
        null_r2 = []
        for _ in range(300):
            sh = RNG.integers(30, len(dh) - 30)
            Xs = [np.ones(len(dh))]
            for lag in range(4):
                Xs += [np.roll(np.roll(ch, sh), lag), np.roll(np.roll(gh, sh), lag)]
            Xs = np.column_stack(Xs)[4:]
            cs = np.linalg.lstsq(Xs, yv, rcond=None)[0]
            null_r2.append(1 - np.var(yv - Xs @ cs) / np.var(yv))
        # hits: signed weave after a speed hit (signed by the crosswind side) against shifted hits
        side = np.sign(np.nan_to_num(Z["cross"]))

        def after(hits, lags=range(1, 7)):
            vals = [dh[min(h + L, len(dh) - 1)] * side[h] for h in hits for L in lags]
            return float(np.mean(vals)) if vals else np.nan
        a_obs = after(Z["hits"])
        a_null = [after(np.sort((Z["hits"] + RNG.integers(20, len(dh) - 20)) % len(dh))) for _ in range(1000)]
        # does a felt headwind gust show in the speed? (the boat as an anemometer)
        r_speed, p_speed, _ = shift_null(corr, Z["res"], hp(np.nan_to_num(Z["gust_h"])))
        out["crews"][k] = dict(
            cox=COX.get(k, k), window_et=Z["window"], mean_wind=round(float(np.nanmean(Z["wind"])), 2), mean_gust=round(float(np.nanmean(Z["gust"])), 2),
            gust_factor=round(float(np.nanmean(Z["gust"]) / np.nanmean(Z["wind"])), 3), mean_dir=round(float(np.nanmean(Z["wdir"])), 0),
            shelter_mean=round(float(np.nanmean(Z["f"])), 3), felt_cross_rms=round(float(np.sqrt(np.nanmean(Z["cross"] ** 2))), 3),
            felt_gust_cross_rms=round(float(np.sqrt(np.nanmean(Z["gust_c"] ** 2))), 3),
            direction=dict(r=round(r_dir, 3), p=round(p_dir, 4), null_sd=round(sd_dir, 3)),
            gust_direction=dict(r=round(r_gust, 3), p=round(p_gust, 4)),
            perturbation_share=dict(r2=round(float(r2), 3), null_mean=round(float(np.mean(null_r2)), 3), null_95=round(float(np.percentile(null_r2, 95)), 3),
                                    excess=round(float(r2 - np.mean(null_r2)), 3)),
            hits=dict(n=int(len(Z["hits"])), per_km=round(len(Z["hits"]) / (np.ptp(Z["st"]) / 1000), 2), weave_after_m=round(a_obs, 3),
                      null_sd=round(float(np.std(a_null)), 3), p=round(float(np.mean(np.abs(np.array(a_null) - np.mean(a_null)) >= abs(a_obs - np.mean(a_null)))), 4)),
            speed_vs_gust=dict(r=round(r_speed, 3), p=round(p_speed, 4)))
        c = out["crews"][k]
        print("%-7s %s-%s ET  wind %.1f m/s gust %.1f (x%.2f) from %03.0f | shelter %.2f | felt crosswind rms %.2f m/s | weave~crosswind r=%+.3f p=%.3f | gust r=%+.3f p=%.3f | wind explains %.1f%% of weave (null %.1f%%, 95%% %.1f%%) | %d hits, weave after %+.3f m (sd %.3f, p %.3f) | speed~headwind gust r=%+.3f p=%.3f"
              % (k, c["window_et"][0], c["window_et"][1], c["mean_wind"], c["mean_gust"], c["gust_factor"], c["mean_dir"], c["shelter_mean"], c["felt_cross_rms"],
                 r_dir, p_dir, r_gust, p_gust, 100 * r2, 100 * np.mean(null_r2), 100 * np.percentile(null_r2, 95), len(Z["hits"]), a_obs, np.std(a_null),
                 c["hits"]["p"], r_speed, p_speed))
    # A. shelter of each line in a common wind (330 deg), and against the distance to the bank
    common = {}
    for k in keys:
        Z = D[k]
        fc = np.array([shelter(a, b, 330.0) for a, b in zip(Z["x"], Z["y"])])
        common[k] = round(float(np.mean(fc)), 3)
    allb = np.concatenate([D[k]["bank"] for k in keys]); allf = np.concatenate([D[k]["f"] for k in keys])
    okb = np.isfinite(allf) & np.isfinite(allb)
    bins = [0, 10, 20, 30, 45, 60, 200]
    by_bank = [dict(from_m=a, to_m=b, shelter=round(float(np.mean(allf[okb][(allb[okb] >= a) & (allb[okb] < b)])), 3),
                    n=int(np.sum((allb[okb] >= a) & (allb[okb] < b)))) for a, b in zip(bins[:-1], bins[1:])]
    out["pooled"]["shelter_common_wind_330"] = common
    out["pooled"]["shelter_by_bank_distance"] = by_bank
    print("shelter in a common 330 deg wind:", common)
    print("shelter by distance to the nearest bank:", [(b["from_m"], b["to_m"], b["shelter"], b["n"]) for b in by_bank])
    # E. 100 m segments pooled: heading wobble vs felt crosswind and gust, crew fixed effects
    rows = []
    for k in keys:
        Z = D[k]
        segs = np.floor(Z["st"] / 100.0)
        for sgi in np.unique(segs):
            m = segs == sgi
            if m.sum() < 8:
                continue
            rows.append((k, float(np.sqrt(np.mean(np.degrees(Z["fast"][m]) ** 2))), float(np.nanmean(np.abs(Z["cross"][m]))),
                         float(np.nanmean(np.abs(Z["gust_c"][m]))), float(np.nanmean(Z["f"][m])), float(np.nanmean(Z["bank"][m]))))
    ks = sorted(set(r[0] for r in rows))
    y = np.array([r[1] for r in rows])
    Xf = np.column_stack([[1.0 if r[0] == kk else 0.0 for r in rows] for kk in ks])
    res = {}
    for name, col in (("felt_crosswind", 2), ("felt_gust_crosswind", 3), ("shelter_factor", 4), ("distance_to_bank", 5)):
        xv = np.array([r[col] for r in rows])
        ok = np.isfinite(xv)
        Xm = np.column_stack([Xf[ok], xv[ok]])
        b = np.linalg.lstsq(Xm, y[ok], rcond=None)[0]
        resid = y[ok] - Xm @ b
        sigma2 = np.sum(resid ** 2) / (ok.sum() - Xm.shape[1])
        se = float(np.sqrt(sigma2 * np.linalg.inv(Xm.T @ Xm)[-1, -1]))
        res[name] = dict(slope=round(float(b[-1]), 4), se=round(se, 4), t=round(float(b[-1]) / se, 2), n=int(ok.sum()))
    out["pooled"]["segments_wobble"] = res
    print("100 m segments, heading wobble (deg) vs:", res)
    json.dump(out, open(os.path.join(HERE, "wind_perturbation.json"), "w"), indent=1)
    figure(D, keys, out)


def figure(D, keys, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(keys), 1, figsize=(11, 2.0 * len(keys)), sharex=True, constrained_layout=True)
    for a, k in zip(np.atleast_1d(ax), keys):
        Z = D[k]
        a.plot(Z["st"], hp(Z["d"]), color="#1f4e79", lw=0.9, label="sideways weave (m)")
        a2 = a.twinx()
        a2.plot(Z["st"], hp(np.nan_to_num(Z["cross"])), color="#c0392b", lw=0.8, alpha=0.7, label="felt crosswind, high-passed (m/s)")
        a.plot(Z["st"][Z["hits"]], np.zeros(len(Z["hits"])), "v", color="#e67e22", ms=4, label="speed hit")
        a.set_ylabel(out["crews"][k]["cox"])
    np.atleast_1d(ax)[0].legend(loc="upper left", fontsize=7, frameon=False)
    np.atleast_1d(ax)[-1].set_xlabel("course station (m)")
    fig.savefig(os.path.join(HERE, "wind_perturbation.png"), dpi=120)
    print("written wind_perturbation.json and wind_perturbation.png")


if __name__ == "__main__":
    main()
