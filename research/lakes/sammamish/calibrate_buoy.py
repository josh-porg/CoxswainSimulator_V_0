"""Calibrate the Lake Sammamish model against King County's buoy: wind measured on open water, 2023-2026.

    .venv/Scripts/python.exe research/lakes/sammamish/calibrate_buoy.py

The CWOP home stations sit among trees and could not pin the model down. The buoy (47.5817 N, 122.0900 W, in the
open southern half) measures the wind the lake itself gets, every 15 minutes, for years. Against the forecast as
issued (Open-Meteo historical forecast archive, gfs_seamless, hourly, buoy/forecast_buoy.json):

* time base: the buoy's local timestamps are tested as PST all year, PDT all year and local clock time (with
  daylight saving) by their correlation with the forecast; the best is used;
* models, fitted on 2023-2024 and tested on 2025-2026 (RMS error, m/s, hours with a forecast of at least 1 m/s):
    M0  buoy = c U                                   (the forecast, scaled)
    M1  buoy = c f(dir; alpha) U                     (the lake model's shelter at the buoy, alpha fitted)
    M2  buoy = c f(dir; alpha) U^beta                (+ the forecast's speed bias)
    M3  M2 x s(sector)                               (+ the forecast's bias by direction, 8 sectors, shrunk to 1)
  The chosen model becomes the page's forecast correction: the 10 m open-water wind over the lake is
  k * s(sector) * U^beta (k = c x 1.125, the log-law step from the buoy's anemometer, taken as 3 m above the
  water, to 10 m; any error in that height is absorbed in c and cancels in the page's use);
* skill of the buoy as a live station: its last reading (published about 1.5 h late) as a correction for the
  next hours, as for the CWOP stations.
Writes buoy/calibration_buoy.json and updates lake.json's forecast_correction and alpha.
"""
import csv, glob, json, math, os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
from scipy.optimize import minimize

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "sammamish"))   # its data and outputs (gitignored)
B = os.path.join(HERE, "buoy")
BUOY = (47.58167, -122.09)
HEIGHT_TO_10M = math.log(10 / 2e-4) / math.log(3 / 2e-4)


def interp16(arr16, d):
    x = (d % 360) / 22.5; i = int(math.floor(x)) % 16; w = x - math.floor(x)
    return arr16[i] * (1 - w) + arr16[(i + 1) % 16] * w


def load_buoy():
    rows = []
    for p in sorted(glob.glob(os.path.join(B, "met_*.csv"))):
        for r in csv.DictReader(open(p, encoding="utf-8", errors="replace"), delimiter="\t"):
            try:
                t = datetime.strptime(r["Date"].strip(), "%m/%d/%Y %I:%M:%S %p")
                s = float(r["Wind Speed (m/sec)"]); d = float(r["Wind Direction (degrees)"])
            except (ValueError, KeyError, AttributeError):
                continue
            if (r.get("ParametersOutOfRange") or "").strip() or not (0 <= s < 40):
                continue
            rows.append((t, s, d))
    return rows


def hourly(rows, mode):
    tz = ZoneInfo("America/Los_Angeles")
    acc = {}
    for t, s, d in rows:
        if mode == "PST":
            u = t + timedelta(hours=8)
        elif mode == "PDT":
            u = t + timedelta(hours=7)
        else:
            u = t.replace(tzinfo=tz).astimezone(timezone.utc).replace(tzinfo=None)
        k = u.strftime("%Y-%m-%dT%H:00")
        a = acc.setdefault(k, [0.0, 0.0, 0.0, 0])
        a[0] += s; a[1] += s * math.sin(math.radians(d)); a[2] += s * math.cos(math.radians(d)); a[3] += 1
    return {k: (a[0] / a[3], math.degrees(math.atan2(a[1], a[2])) % 360) for k, a in acc.items() if a[3] >= 2}


def main():
    fc = json.load(open(os.path.join(B, "forecast_buoy.json")))["hourly"]
    F = {t: (s, d) for t, s, d in zip(fc["time"], fc["wind_speed_10m"], fc["wind_direction_10m"]) if s is not None}
    raw = load_buoy()
    print("buoy readings: %d (%s to %s)" % (len(raw), raw[0][0], raw[-1][0]))
    best = None
    for mode in ("PST", "PDT", "local"):
        H = hourly(raw, mode)
        ks = [k for k in H if k in F]
        r = np.corrcoef([H[k][0] for k in ks], [F[k][0] for k in ks])[0, 1]
        print("time base %-5s: %d matched hours, correlation with the forecast %.3f" % (mode, len(ks), r))
        if best is None or r > best[0]:
            best = (r, mode, H)
    _, mode, H = best
    C = np.load(os.path.join(HERE, "components.npz"))
    cell = int(np.argmin(np.hypot((C["lat"] - BUOY[0]) * 111000, (C["lon"] - BUOY[1]) * 75000)))
    sx, fs = C["sx"][:, cell], C["f_shore"][:, cell]
    print("buoy's model cell: fetch by direction (m)", [int(x) for x in C["fetch"][:, cell][::2]], "Sx (deg)", [round(float(x), 1) for x in sx[::2]])
    keys = sorted(k for k in H if k in F and F[k][0] >= 1.0)
    obs = np.array([H[k][0] for k in keys]); U = np.array([F[k][0] for k in keys]); D = np.array([F[k][1] for k in keys])
    year = np.array([int(k[:4]) for k in keys]); train, test = year <= 2024, year >= 2025
    sector = (np.round(D / 45.0).astype(int)) % 8

    def f_of(alpha):
        f16 = fs * (1 - alpha * np.clip(sx / 12.0, 0, 1))
        return np.array([interp16(f16, d) for d in D])

    def predict(p, model):
        c, alpha, beta = math.exp(p[0]), min(max(p[1], 0.0), 1.0), p[2]
        base = f_of(alpha) if model >= 1 else np.ones(len(U))
        pred = c * base * (U ** (beta if model >= 2 else 1.0))
        if model >= 3:
            s = np.exp(np.asarray(p[3:11]))
            pred = pred * s[sector]
        return pred

    results = {}
    for model, npar in ((0, 3), (1, 3), (2, 3), (3, 11)):
        def loss(p):
            e = obs[train] - predict(p, model)[train]
            pen = 0.0 if model < 3 else 20.0 * float(np.sum(np.square(p[3:11])))        # shrink sectors to 1
            return float(np.mean(e * e)) + pen / len(e)
        p0 = np.array([math.log(0.5), 0.5, 1.0] + ([0.0] * 8 if model == 3 else []))
        if model == 0:
            p0[1] = 0.0
        res = minimize(loss, p0, method="Nelder-Mead" if model < 3 else "Powell", options=dict(maxiter=20000, xatol=1e-4, fatol=1e-6))
        p = res.x
        rm = lambda m: float(np.sqrt(np.mean(np.square(obs[m] - predict(p, model)[m]))))     # noqa: E731
        results[model] = dict(params=p.tolist(), rms_train=rm(train), rms_test=rm(test), c=math.exp(p[0]),
                              alpha=float(min(max(p[1], 0), 1)) if model >= 1 else None, beta=float(p[2]) if model >= 2 else 1.0,
                              sectors=[round(math.exp(x), 3) for x in p[3:11]] if model == 3 else None)
        print("M%d: test RMS %.3f m/s (train %.3f); c %.3f%s%s%s" % (model, results[model]["rms_test"], results[model]["rms_train"], results[model]["c"],
              "" if model < 1 else ", alpha %.2f" % results[model]["alpha"], "" if model < 2 else ", beta %.2f" % results[model]["beta"],
              "" if model < 3 else ", sectors N..NW " + " ".join("%.2f" % x for x in results[model]["sectors"])))
    raw_rms = float(np.sqrt(np.mean(np.square(obs[test] - U[test]))))
    print("the raw forecast as the buoy's wind: test RMS %.3f m/s; mean buoy %.2f vs forecast %.2f m/s" % (raw_rms, obs.mean(), U.mean()))
    # high-wind skill (what matters for rowing): hours with forecast >= 6 m/s
    hi = test & (U >= 6.0)
    for model in results:
        p = np.array(results[model]["params"])
        e = obs[hi] - predict(p, model)[hi]
        results[model]["rms_test_windy"] = float(np.sqrt(np.mean(e * e))) if hi.any() else None
        results[model]["bias_test_windy"] = float(np.mean(e)) if hi.any() else None
    print("windy test hours (forecast >= 6 m/s): n=%d; RMS by model %s; bias %s" % (int(hi.sum()),
          [round(results[m]["rms_test_windy"], 2) for m in results], [round(results[m]["bias_test_windy"], 2) for m in results]))
    chosen = min(results, key=lambda m: results[m]["rms_test"])
    R = results[chosen]
    print("chosen: M%d" % chosen)
    # the buoy as a live station: its value 2 h ago (published ~1.5 h late) as a ratio for the next hours
    p = np.array(R["params"]); pred_all = predict(p, chosen)
    idx = {k: i for i, k in enumerate(keys)}
    gains = {}
    for ahead in (1, 2, 3, 6):
        a, b = [], []
        for k, i in idx.items():
            if not test[i]:
                continue
            k0 = (datetime.fromisoformat(k) - timedelta(hours=2 + ahead)).strftime("%Y-%m-%dT%H:00")
            if k0 in idx and U[idx[k0]] >= 2.5:
                r = min(max(obs[idx[k0]] / max(pred_all[idx[k0]], 0.2), 0.3), 3.0)
                w = math.exp(-(2 + ahead) / 3.0)
                a.append(obs[i] - pred_all[i]); b.append(obs[i] - pred_all[i] * (1 + (r - 1) * w))
        gains[ahead] = round(1 - math.sqrt(np.mean(np.square(b))) / math.sqrt(np.mean(np.square(a))), 3) if a else None
    print("buoy as a live correction (data 2 h old), error cut by hours ahead:", gains)
    out = dict(time_base=mode, hours=len(keys), test_years="2025-2026", train_years="2023-2024", results={"M%d" % m: v for m, v in results.items()},
               chosen="M%d" % chosen, raw_forecast_rms_test=raw_rms, live_gain=gains, height_to_10m=HEIGHT_TO_10M, buoy=BUOY, cell=cell)
    json.dump(out, open(os.path.join(B, "calibration_buoy.json"), "w"), indent=1)
    cfg_path = os.path.join(HERE, "lake.json"); cfg = json.load(open(cfg_path))
    cfg["forecast_correction"] = dict(source="King County Lake Sammamish buoy 2023-2026, model M%d" % chosen,
                                      k=round(R["c"] * HEIGHT_TO_10M, 4), beta=round(R["beta"], 4),
                                      sectors=R["sectors"] or [1.0] * 8, buoy_f=None)
    # the correction is relative to the lake model's factor at the buoy: page multiplies forecast U by k s U^(beta-1)
    if R["alpha"] is not None:
        cfg["alpha"] = round(R["alpha"], 3)
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    print("lake.json updated: forecast_correction k=%.3f beta=%.2f, alpha %s" % (cfg["forecast_correction"]["k"], cfg["forecast_correction"]["beta"], cfg.get("alpha")))


if __name__ == "__main__":
    main()
