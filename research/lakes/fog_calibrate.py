"""Morning fog model for a lake: probability of fog at each early hour, calibrated on the airport's history.

    .venv/Scripts/python.exe research/lakes/fog_calibrate.py data/local/<lake>/lake.json

Label: fog at the airport (lake.json "fog"), visibility under 1 km (0.62 mile, the meteorological definition of
fog) or FG reported, at each hour 03-10 local, 2023 to now. Features: the forecast as issued for that hour
(fog_history.py): log visibility, dew-point depression, relative humidity, wind, low cloud, the fog weather code,
the hour, the season, and the dew-point depression at 21:00 the evening before (a clear, near-saturated evening is
how radiation fog starts). Logistic regression, fitted on 2023-2024 and tested on 2025-2026 against:
the climatology (fog rate by month and hour), the rule "forecast visibility under 1 km", and the forecast's own
fog code. Both the airport's and the lake's forecast point are tried as the feature source; the better is kept.
Also: how well it times the lifting (the first hour after the launch with less than 30% chance).
Writes fog/fog_model.json and lake.json "fog_model" (refitted on all years) for the page.
"""
import csv, json, math, os, sys
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

CFG_PATH = os.path.abspath(sys.argv[1]); CFG = json.load(open(CFG_PATH))
D = os.path.join(os.path.dirname(CFG_PATH), "fog")
FOG_MI = 0.62
HOURS = range(3, 11)
NAMES = ["log_vis", "dpd", "rh", "wind", "low_cloud", "fog_code", "hour", "hour2", "season_c", "season_s", "dpd_eve"]


def observed():
    lab = {}
    for r in csv.DictReader(open(os.path.join(D, "obs_%s.csv" % CFG["fog"]["airport"]))):
        try:
            t = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        k = (t + timedelta(minutes=30)).replace(minute=0)                    # :53 report -> the next hour
        try:
            v = float(r["vsby"])
        except ValueError:
            continue
        fog = v < FOG_MI or "FG" in (r["wxcodes"] or "")
        lab[k] = lab.get(k, False) or fog
    return lab


def features(where):
    h = json.load(open(os.path.join(D, "forecast_%s.json" % where)))
    T = [datetime.fromisoformat(t) for t in h["time"]]
    idx = {t: i for i, t in enumerate(T)}
    X = {}
    for i, t in enumerate(T):
        if t.hour not in HOURS:
            continue
        vis, tt, td, rh = h["visibility"][i], h["temperature_2m"][i], h["dew_point_2m"][i], h["relative_humidity_2m"][i]
        if None in (vis, tt, td, rh):
            continue
        eve = idx.get((t - timedelta(days=1 if t.hour < 21 else 0)).replace(hour=21))
        dpd_eve = (h["temperature_2m"][eve] - h["dew_point_2m"][eve]) if eve is not None and h["temperature_2m"][eve] is not None else 5.0
        doy = t.timetuple().tm_yday
        X[t] = [math.log10(min(max(vis, 50.0), 30000.0)), min(max(tt - td, 0.0), 10.0), rh / 100.0,
                min((h["wind_speed_10m"][i] or 0) / 3.6, 15.0), (h["cloud_cover_low"][i] or 0) / 100.0,
                1.0 if h["weather_code"][i] in (45, 48) else 0.0, t.hour - 6.0, (t.hour - 6.0) ** 2 / 10.0,
                math.cos(2 * math.pi * doy / 365.25), math.sin(2 * math.pi * doy / 365.25), min(max(dpd_eve, 0.0), 10.0),
                vis]
    return X


def evaluate(where, lab):
    X = features(where)
    keys = sorted(k for k in X if k in lab)
    A = np.array([X[k][:-1] for k in keys]); y = np.array([lab[k] for k in keys], int); vis = np.array([X[k][-1] for k in keys])
    yr = np.array([k.year for k in keys]); tr, te = yr <= 2024, yr >= 2025
    mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9
    m = LogisticRegression(C=0.5, max_iter=2000).fit((A[tr] - mu) / sd, y[tr])
    p = m.predict_proba((A[te] - mu) / sd)[:, 1]
    # climatology by month and hour (from the training years)
    clim = {}
    for k, v in zip([keys[i] for i in np.where(tr)[0]], y[tr]):
        c = clim.setdefault((k.month, k.hour), [0, 0]); c[0] += v; c[1] += 1
    pc = np.array([clim.get((keys[i].month, keys[i].hour), [0, 1])[0] / max(clim.get((keys[i].month, keys[i].hour), [0, 1])[1], 1) for i in np.where(te)[0]])
    rule_rate = y[tr][vis[tr] < 1000].mean() if (vis[tr] < 1000).any() else 0.5
    pr = np.where(vis[te] < 1000, rule_rate, y[tr][vis[tr] >= 1000].mean())
    code = A[te][:, NAMES.index("fog_code")]
    code_rate = y[tr][A[tr][:, NAMES.index("fog_code")] > 0].mean() if (A[tr][:, NAMES.index("fog_code")] > 0).any() else 0.5
    pcode = np.where(code > 0, code_rate, y[tr][A[tr][:, NAMES.index("fog_code")] == 0].mean())
    res = dict(where=where, n_train=int(tr.sum()), n_test=int(te.sum()), base_rate_test=float(y[te].mean()),
               brier=dict(model=brier_score_loss(y[te], p), climatology=brier_score_loss(y[te], pc), vis_rule=brier_score_loss(y[te], pr), fog_code=brier_score_loss(y[te], pcode)),
               auc=dict(model=roc_auc_score(y[te], p), vis_forecast=roc_auc_score(y[te], -vis[te])))
    res["brier_skill_vs_climatology"] = 1 - res["brier"]["model"] / res["brier"]["climatology"]
    # reliability: predicted vs observed in bins
    bins = [0, .05, .1, .2, .35, .5, .7, 1.01]
    res["reliability"] = [dict(p_lo=a, p_hi=b, n=int(((p >= a) & (p < b)).sum()), observed=float(y[te][(p >= a) & (p < b)].mean()) if ((p >= a) & (p < b)).any() else None)
                          for a, b in zip(bins[:-1], bins[1:])]
    # lifting: on test mornings foggy at 5 or 6, predicted vs observed first clear hour (p < 0.3 / no fog) from 6 on
    tk = [keys[i] for i in np.where(te)[0]]; pk = dict(zip(tk, p)); yk = dict(zip(tk, y[te]))
    errs = []
    for day in sorted({k.date() for k in tk}):
        h5 = datetime.combine(day, datetime.min.time()).replace(hour=5)
        if not (yk.get(h5) or yk.get(h5 + timedelta(hours=1))):
            continue
        obs_clear = next((h for h in range(6, 11) if yk.get(h5.replace(hour=h)) == 0), 11)
        pred_clear = next((h for h in range(6, 11) if pk.get(h5.replace(hour=h), 1) < 0.3), 11)
        errs.append(pred_clear - obs_clear)
    res["lifting"] = dict(foggy_mornings=len(errs), mean_abs_error_h=float(np.mean(np.abs(errs))) if errs else None,
                          within_1h=float(np.mean(np.abs(np.array(errs)) <= 1)) if errs else None)
    return res, (A, y, keys)


def main():
    lab = observed()
    print("%s: %d hours observed, fog in %.1f%% of morning hours (03-10)" % (CFG["fog"]["airport"], len(lab),
          100 * np.mean([v for k, v in lab.items() if k.hour in HOURS])))
    out = {}
    for where in ("airport", "lake"):
        r, data = evaluate(where, lab)
        out[where] = (r, data)
        b = r["brier"]
        print("features at the %-7s: test n=%d (fog %.1f%%); Brier model %.4f, climatology %.4f, vis<1km rule %.4f, fog code %.4f; skill vs climatology %+.0f%%; AUC %.3f (forecast visibility alone %.3f); lifting: %d foggy mornings, mean error %.1f h, within 1 h %.0f%%"
              % (where, r["n_test"], 100 * r["base_rate_test"], b["model"], b["climatology"], b["vis_rule"], b["fog_code"], 100 * r["brier_skill_vs_climatology"],
                 r["auc"]["model"], r["auc"]["vis_forecast"], r["lifting"]["foggy_mornings"], r["lifting"]["mean_abs_error_h"] or float("nan"), 100 * (r["lifting"]["within_1h"] or 0)))
        print("   reliability:", [(x["p_lo"], x["n"], None if x["observed"] is None else round(x["observed"], 2)) for x in r["reliability"]])
    where = min(out, key=lambda w: out[w][0]["brier"]["model"])
    r, (A, y, keys) = out[where]
    mu, sd = A.mean(0), A.std(0) + 1e-9
    m = LogisticRegression(C=0.5, max_iter=2000).fit((A - mu) / sd, y)
    model = dict(where=where, point=CFG["fog"]["airport_latlon"] if where == "airport" else CFG["forecast_points"][1], features=NAMES,
                 mean=mu.round(5).tolist(), sd=sd.round(5).tolist(), coef=m.coef_[0].round(5).tolist(), intercept=round(float(m.intercept_[0]), 5),
                 airport=CFG["fog"]["airport"], fog_definition="visibility under 1 km (0.62 mile) or FG reported",
                 test=dict(years="2025-2026", brier_skill_vs_climatology=round(r["brier_skill_vs_climatology"], 3), auc=round(r["auc"]["model"], 3),
                           base_rate=round(r["base_rate_test"], 3), lifting=r["lifting"]))
    print("kept: features at the %s; coefficients %s" % (where, dict(zip(NAMES, model["coef"]))))
    json.dump(dict(results={w: out[w][0] for w in out}, model=model), open(os.path.join(D, "fog_model.json"), "w"), indent=1, default=float)
    CFG["fog_model"] = model
    json.dump(CFG, open(CFG_PATH, "w"), indent=1)


if __name__ == "__main__":
    main()
