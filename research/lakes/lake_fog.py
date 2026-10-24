"""Lake fog on Lake Sammamish: a water-aware model, labelled on the lake itself.

    .venv/Scripts/python.exe research/lakes/lake_fog.py data/local/sammamish/lake.json

Why: the page's fog model (fog_calibrate.py) is trained on Renton airport's fog. In autumn the lake is warmer
than the night air (16-17 C water under 5-8 C air in late October), and cold air from the hills and the river
valley pools over it: steam (evaporation) fog forms on the water while the airports stay clear. 21-22 October
2026 were two such mornings: no fog at Renton or Boeing Field; saturated air and a dimmed sun at the buoy.

Label (lake fog), from the King County buoy in the open water, in daylight only (sun at least 4 degrees up,
the first four hours after sunrise): fog = the buoy's air at 95% humidity or more AND its sunlight at most 55%
of clear sky (Haurwitz) AND neither airport (KRNT, KBFI) reporting a ceiling below 5000 ft (so the dimming is
not a general overcast), or an airport reporting fog. Clear = sunlight at least 75% of clear sky, or humidity
under 90%. Everything else (dim under an overcast, middling) is left out. Before sunrise the label cannot be
seen, so the model is fitted on daylight hours and applied to the early hours through the same physics.

Features, all known ahead (the forecast as issued at the lake point, and the water as known the evening before):
* water minus air temperature, and Saunders' mixing excess (hPa): the most supersaturated point on the mixing
  line between the forecast air and air saturated at the water's temperature (steam fog);
* forecast dew-point depression, wind, low and total cloud, visibility, the evening's dew-point depression
  (clear, damp, calm nights: radiation fog and cold-air pooling);
* the airport model's own score (regional fog);
* hours after sunrise (clamped at 0: the sun lifts fog).
Logistic regression, fitted on 2023-2024, tested on 2025-2026 against the airport model and climatology.
Writes fog/lake_fog_model.json and, after refitting on all years, lake.json "lake_fog_model".
"""
import json, math, os, sys
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lake_fog_data as LD                                                   # noqa: E402

CFG_PATH = os.path.abspath(sys.argv[1]); CFG = json.load(open(CFG_PATH))
D = os.path.join(os.path.dirname(CFG_PATH), "fog")
NAMES = ["dT", "mix", "dpd", "wind", "low_cloud", "cloud", "log_vis", "dpd_eve", "airport_z", "from_sunrise", "mix_calm", "inv925", "inv850"]


def airport_z(f, t, eve_dpd):
    """The page's existing (airport-trained) model's linear score for this hour."""
    m = CFG["fog_model"]
    doy = t.timetuple().tm_yday
    x = [math.log10(min(max(f["visibility"], 50.0), 30000.0)), min(max(f["temperature_2m"] - f["dew_point_2m"], 0.0), 10.0), f["relative_humidity_2m"] / 100.0,
         min((f["wind_speed_10m"] or 0) / 3.6, 15.0), (f["cloud_cover_low"] or 0) / 100.0, 1.0 if f["weather_code"] in (45, 48) else 0.0,
         t.hour - 6.0, ((t.hour - 6.0) / 3.0) ** 2, math.cos(2 * math.pi * doy / 365.25), math.sin(2 * math.pi * doy / 365.25),
         min(max(eve_dpd if eve_dpd is not None else 5.0, 0.0), 10.0)]
    return m["intercept"] + sum(c * (v - mu) / sd for c, v, mu, sd in zip(m["coef"], x, m["mean"], m["sd"]))


def sunrise(day):
    t = datetime(day.year, day.month, day.day, 4, 0)
    while LD.sun(t)[0] < -0.833 and t.hour < 11:
        t += timedelta(minutes=2)
    return t


def feats(r, sr):
    f = r["f"]; Ta, Td, Tw = f["temperature_2m"], f["dew_point_2m"], r["tw"]
    hs = min(max((r["t"] + timedelta(minutes=30) - sr).total_seconds() / 3600.0, -5.0), 4.0)
    return [Tw - Ta, LD.mixing_excess(Ta, Td, Tw), min(max(Ta - Td, 0.0), 10.0), min((f["wind_speed_10m"] or 0) / 3.6, 15.0),
            (f["cloud_cover_low"] or 0) / 100.0, (f.get("cloud_cover") or 0) / 100.0, math.log10(min(max(f["visibility"], 50.0), 30000.0)),
            min(max(r["eve_dpd"] if r["eve_dpd"] is not None else 5.0, 0.0), 10.0), airport_z(f, r["t"], r["eve_dpd"]), hs,
            LD.mixing_excess(Ta, Td, Tw) / (1.0 + min((f["wind_speed_10m"] or 0) / 3.6, 15.0)),
            # the inversion over the valley: warm air aloft holds the cold pool (and its fog) on the lake
            min(max((f.get("temperature_925hPa") if f.get("temperature_925hPa") is not None else Ta) - Ta, -5.0), 15.0),
            min(max((f.get("temperature_850hPa") if f.get("temperature_850hPa") is not None else Ta - 8) - Ta, -15.0), 15.0)]


def label(r, sr):
    b, a = r["b"], r["a"]
    tm = r["t"] + timedelta(minutes=30)
    el, ghi = LD.sun(tm)
    if b is None or el < 4.0 or (tm - sr).total_seconds() > 4 * 3600 or ghi < 20:
        return None
    k = b["solar"] / ghi
    if any(v["fog"] for v in a.values()):
        return 1
    overcast = any(v["ceil"] < 5000 for v in a.values())
    if b["rh"] >= 95.0 and k <= 0.55 and not overcast and a:
        return 1
    if k >= 0.75 or b["rh"] < 90.0:
        return 0
    return None


NOW = ["rh_def", "buoy_wind", "dT_buoy"]


def now_feats(r, sat):
    """What the buoy itself says about fog on the water now: humidity below the sensor's saturation reading
    (its 99.5th percentile that year, about 98%), wind on the water, water minus the buoy's air temperature."""
    b = r["b"]
    return [max(sat[r["t"].year] - b["rh"], 0.0), b["wind"], r["tw"] - b["tair"]]


def fit_best(X, y, w, tr, te, Cs=(0.03, 0.1, 0.3, 1.0, 3.0)):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    best = None
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=3000).fit((X[tr] - mu) / sd, y[tr], sample_weight=w[tr])
        p = m.predict_proba((X[te] - mu) / sd)[:, 1]
        bs = float(np.average((p - y[te]) ** 2, weights=w[te]))
        if best is None or bs < best[0]:
            best = (bs, C, m, p, mu, sd)
    return best


def main():
    R = LD.table(CFG_PATH)
    sat = {}
    for yv in range(2023, 2027):
        rh = [r["b"]["rh"] for r in R if r["b"] and r["t"].year == yv]
        sat[yv] = float(np.percentile(rh, 99.5)) if rh else 98.0
    srs = {}
    rows = []
    for r in R:
        if not np.isfinite(r["tw"]) or r["f"]["visibility"] is None:
            continue
        day = r["t"].date()
        if day not in srs:
            srs[day] = sunrise(r["t"])
        rows.append(dict(r=r, x=feats(r, srs[day]), lab=label(r, srs[day]), now=now_feats(r, sat) if r["b"] else None))
    # ---- stage A: the buoy's own readings against the daylight label
    A = [q for q in rows if q["lab"] is not None and q["now"] is not None]
    XA = np.array([q["now"] for q in A]); yA = np.array([q["lab"] for q in A]); yrA = np.array([q["r"]["t"].year for q in A])
    print("stage A, daylight hours with a label: %d, lake fog %d (%.1f%%), %d mornings" % (len(yA), yA.sum(), 100 * yA.mean(), len({q["r"]["t"].date() for q in A if q["lab"]})))
    bsA, CA, mA, pA, muA, sdA = fit_best(XA, yA, np.ones(len(yA)), yrA <= 2024, yrA >= 2025)
    print("  buoy detector on 2025-26: AUC %.3f, Brier %.4f (base %.4f); coefficients %s" % (roc_auc_score(yA[yrA >= 2025], pA), bsA, np.mean((yA[yrA >= 2025] - yA[yrA <= 2024].mean()) ** 2),
          {n: round(float(c), 2) for n, c in zip(NOW, mA.coef_[0])}))
    mA = LogisticRegression(C=CA, max_iter=3000).fit((XA - XA.mean(0)) / (XA.std(0) + 1e-9), yA); muA, sdA = XA.mean(0), XA.std(0) + 1e-9
    # ---- the detector at every practice hour, dark or light
    B = [q for q in rows if q["now"] is not None and 3 <= q["r"]["t"].hour <= 10]
    pobs = mA.predict_proba((np.array([q["now"] for q in B]) - muA) / sdA)[:, 1]
    rained = 0
    for i, q in enumerate(B):
        a_ = q["r"]["a"]
        if any(v.get("rain") for v in a_.values()) and not any(v["fog"] for v in a_.values()):
            pobs[i] = 0.0; rained += 1                      # rain saturates the buoy's air too: not fog
        q["pobs"] = float(pobs[i])
    print("practice hours with rain at an airport (detector set to 0): %d of %d" % (rained, len(B)))
    # ---- stage B: the forecast (and the water) against the detector, weighted soft labels
    XB = np.array([q["x"] for q in B]); yrB = np.array([q["r"]["t"].year for q in B])
    X2 = np.vstack([XB, XB]); y2 = np.concatenate([np.ones(len(B)), np.zeros(len(B))]); w2 = np.concatenate([pobs, 1 - pobs]); yr2 = np.concatenate([yrB, yrB])
    bsB, CB, mB, pB, muB, sdB = fit_best(X2, y2, w2, yr2 <= 2024, yr2 >= 2025)
    te = yrB >= 2025
    pB = mB.predict_proba((XB[te] - muB) / sdB)[:, 1]
    p_air = 1 / (1 + np.exp(-XB[te][:, NAMES.index("airport_z")]))
    tgt = pobs[te]
    clim = np.full(te.sum(), pobs[~te].mean())
    bs = lambda p: float(np.mean((p - tgt) ** 2))                    # noqa: E731
    out = dict(stage_A=dict(C=CA, auc_test=round(float(roc_auc_score(yA[yrA >= 2025], pA)), 3), features=NOW, mean=muA.round(4).tolist(), sd=sdA.round(4).tolist(),
                            coef=mA.coef_[0].round(4).tolist(), intercept=round(float(mA.intercept_[0]), 4), saturation_rh=sat),
               stage_B=dict(C=CB, brier=dict(lake_model=round(bs(pB), 5), airport_model=round(bs(p_air), 5), climatology=round(bs(clim), 5)),
                            skill_vs_climatology=round(1 - bs(pB) / bs(clim), 3), skill_vs_airport_model=round(1 - bs(pB) / bs(p_air), 3),
                            fog_hours_test=round(float(tgt.sum()), 1)))
    # hard daylight label check for stage B
    D_ = [q for q in B if q["lab"] is not None and q["r"]["t"].year >= 2025]
    if D_:
        yd = np.array([q["lab"] for q in D_]); xd = np.array([q["x"] for q in D_])
        pd_ = mB.predict_proba((xd - muB) / sdB)[:, 1]; pa_ = 1 / (1 + np.exp(-xd[:, NAMES.index("airport_z")]))
        out["stage_B"]["daylight_label"] = dict(n=len(yd), fog=int(yd.sum()), auc_lake=round(float(roc_auc_score(yd, pd_)), 3), auc_airport=round(float(roc_auc_score(yd, pa_)), 3),
                                                 brier_lake=round(float(brier_score_loss(yd, pd_)), 5), brier_airport=round(float(brier_score_loss(yd, pa_)), 5))
    # practice mornings: a morning is foggy if the detector says >= 50% at any hour 04-07
    mornings = {}
    for q, p in zip(B, pobs):
        if 4 <= q["r"]["t"].hour <= 7:
            d = q["r"]["t"].date(); mornings[d] = max(mornings.get(d, 0), p)
    out["foggy_practice_mornings_per_year"] = {yv: sum(1 for d, p in mornings.items() if d.year == yv and p >= 0.5) for yv in range(2023, 2027)}
    print(json.dumps(out, indent=1))
    print("stage B coefficients (standardised):", {n: round(float(c), 3) for n, c in zip(NAMES, mB.coef_[0])})
    # refit stage B on all years
    muF, sdF = XB.mean(0), XB.std(0) + 1e-9
    mF = LogisticRegression(C=CB, max_iter=3000).fit((X2 - muF) / sdF, y2, sample_weight=w2)
    print("recent practice hours: buoy detector / new forecast model / airport model")
    for q in rows:
        t = q["r"]["t"]
        if t >= datetime(2026, 10, 17) and 4 <= t.hour <= 8:
            pf = mF.predict_proba(((np.array(q["x"]) - muF) / sdF)[None])[0, 1]
            pt = mB.predict_proba(((np.array(q["x"]) - muB) / sdB)[None])[0, 1]     # trained on 2023-24 only
            po = q.get("pobs", float("nan"))
            print("  %s  %3.0f%%  %3.0f%% (2023-24 fit %3.0f%%)  %3.0f%%   dT %.1f mix %.2f wind %.1f" % (t.strftime("%m-%d %H:00"), 100 * po, 100 * pf, 100 * pt, 100 / (1 + math.exp(-q["x"][NAMES.index("airport_z")])), q["x"][0], q["x"][1], q["x"][3]))
    model = dict(features=NAMES, mean=muF.round(5).tolist(), sd=sdF.round(5).tolist(), coef=mF.coef_[0].round(5).tolist(), intercept=round(float(mF.intercept_[0]), 5), test=out,
                 label="fog on the water at the buoy: a detector from the buoy's humidity (against its saturation reading), wind and water-air temperature, "
                       "calibrated in daylight on sun dimmed to <= 55% of clear sky with no airport ceiling below 5000 ft (or airport fog), applied to every practice hour")
    json.dump(model, open(os.path.join(D, "lake_fog_model.json"), "w"), indent=1)
    print("written fog/lake_fog_model.json")


if __name__ == "__main__":
    main()
