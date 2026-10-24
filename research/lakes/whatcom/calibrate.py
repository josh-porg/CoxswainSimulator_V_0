"""Calibrate the Lake Whatcom wind model against the lakeside CWOP stations' history.

    .venv/Scripts/python.exe research/lakes/whatcom/calibrate.py

Each lake-level station is matched to the nearest water cell of the model (whatcom_components.npz). For every
hour with a forecast wind of at least MIN_MS (history/forecast_<call>.csv, the forecast as issued) the
station's mean wind is compared with

    c_s * f(dir; alpha) * U_forecast,     f = shore factor * (1 - alpha * min(Sx / 12 deg, 1))

where c_s is the station's own siting constant (a garden anemometer among trees reads low whatever the
wind does) and alpha the strength of the terrain shelter, the one physical parameter in question. c_s is
least squares per station; alpha is chosen to minimise the pooled RMS error. Baselines: no shelter at all
(f = 1) and the shore alone (alpha = 0). The hillside station (90 m above the lake) is reported but not fitted.
Writes calibration.json (alpha, c_s, errors, the per-sector picture) and the nowcast constants the page uses.
"""
import csv, json, math, os
from datetime import datetime
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "whatcom"))   # its data and outputs (gitignored)
H = os.path.join(HERE, "history")
MIN_MS = 2.5
PRIOR = 0.5
TAU_H = 3.0          # how fast a live correction fades into the forecast, hours
MPH = 0.44704
LAKE = {"DW0884": "Agate Bay", "FW1160": "North shore, middle basin", "FW6124": "Geneva shore"}
HILL = {"EW7158": "Sudden Valley hillside"}
POS = {"DW0884": (48.76116, -122.36255), "FW1160": (48.75083, -122.3545), "FW6124": (48.745, -122.3955), "EW7158": (48.7035, -122.32317)}


def hourly_obs(call):
    acc = {}
    for r in csv.DictReader(open(os.path.join(H, call + ".csv"))):
        try:
            t = datetime.strptime(r["t"], "%Y%m%d%H%M%S"); s = float(r["mph"]) * MPH; d = float(r["dir"])
        except ValueError:
            continue
        k = t.strftime("%Y-%m-%dT%H:00")
        a = acc.setdefault(k, [0.0, 0.0, 0.0, 0])
        a[0] += s; a[1] += s * math.sin(math.radians(d)); a[2] += s * math.cos(math.radians(d)); a[3] += 1
    return {k: (a[0] / a[3], math.degrees(math.atan2(a[1], a[2])) % 360, a[3]) for k, a in acc.items() if a[3] >= 2}


def forecast(call):
    out = {}
    for r in csv.DictReader(open(os.path.join(H, "forecast_%s.csv" % call))):
        try:
            out[r["t"]] = (float(r["speed_ms"]), float(r["dir"]))
        except ValueError:
            pass
    return out


def interp16(arr16, d):
    x = (d % 360) / 22.5; i = int(math.floor(x)) % 16; w = x - math.floor(x)
    return arr16[i] * (1 - w) + arr16[(i + 1) % 16] * w


def main():
    C = np.load(os.path.join(HERE, "whatcom_components.npz"))
    lat, lon = C["lat"], C["lon"]
    data = {}
    for call in list(LAKE) + list(HILL):
        la, lo = POS[call]
        dist = np.hypot((lat - la) * 111000, (lon - lo) * 111000 * math.cos(math.radians(48.72)))
        cell = int(np.argmin(dist))
        ob, fc = hourly_obs(call), forecast(call)
        rows = [(ob[k][0], fc[k][0], fc[k][1]) for k in ob if k in fc and fc[k][0] >= MIN_MS]
        data[call] = dict(cell=cell, cell_m=float(dist[cell]), rows=np.array(rows),
                          sx=C["sx"][:, cell], fs=C["f_shore"][:, cell])
        print("%s: nearest water cell %.0f m away; %d hours with forecast >= %.1f m/s" % (call, dist[cell], len(rows), MIN_MS))

    def evaluate(alpha, calls, shelter=True):
        err, n, cs = 0.0, 0, {}
        for call in calls:
            D = data[call]; R = D["rows"]
            if not len(R):
                continue
            if shelter:
                f = np.array([interp16(D["fs"] * (1 - alpha * np.clip(D["sx"] / 12.0, 0, 1)), d) for d in R[:, 2]])
            else:
                f = np.ones(len(R))
            p = f * R[:, 1]
            c = float(np.sum(R[:, 0] * p) / np.sum(p * p))                   # least squares siting constant
            cs[call] = c
            err += float(np.sum((R[:, 0] - c * p) ** 2)); n += len(R)
        return math.sqrt(err / n), cs, n

    lake = list(LAKE)
    base_none, cs_none, n = evaluate(0, lake, shelter=False)
    base_shore, _, _ = evaluate(0.0, lake)
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    scores = [(evaluate(a, lake)[0], a) for a in grid]
    best_err, alpha_fit = min(scores)
    prior_err = evaluate(PRIOR, lake)[0]
    # adopt the fit only if it is clearly better than the prior (the error curve is often nearly flat)
    alpha = alpha_fit if (prior_err - best_err) / prior_err > 0.02 else PRIOR
    print("alpha: fit %.2f (RMS %.3f) against the prior %.2f (RMS %.3f): using %.2f" % (alpha_fit, best_err, PRIOR, prior_err, alpha))
    _, cs, _ = evaluate(alpha, lake)
    _, cs_hill, _ = evaluate(alpha, list(HILL))
    print("RMS error over %d station-hours (m/s): no shelter %.3f, shore only %.3f, shore + terrain (alpha %.2f) %.3f"
          % (n, base_none, base_shore, alpha, evaluate(alpha, lake)[0]))
    print("   alpha scan:", " ".join("%.2f:%.3f" % (a, e) for e, a in scores))
    # per sector: the station/forecast ratio against the model's factor, 8 sectors
    sectors = {}
    for call in lake + list(HILL):
        D = data[call]; R = D["rows"]
        tab = []
        for s in range(8):
            m = ((R[:, 2] - s * 45 + 22.5) % 360) < 45 if len(R) else np.array([], bool)
            if m.sum() >= 5:
                ratio = float(np.mean(R[m, 0]) / np.mean(R[m, 1]))
                model = float(np.mean([interp16(D["fs"] * (1 - alpha * np.clip(D["sx"] / 12.0, 0, 1)), d) for d in R[m, 2]]))
                tab.append(dict(sector=s * 45, n=int(m.sum()), obs_over_forecast=round(ratio, 3), model_factor=round(model, 3)))
        sectors[call] = tab
        print("%-7s c=%.2f " % (call, (cs.get(call) or cs_hill.get(call) or float("nan"))) +
              "  ".join("%03d:%d obs/fc %.2f model %.2f" % (t["sector"], t["n"], t["obs_over_forecast"], t["model_factor"]) for t in tab))
    # does the shape agree? correlation across sectors of obs/forecast with the model factor, per lake station
    shape = {}
    for call in lake:
        t = sectors[call]
        if len(t) >= 3:
            a = np.array([x["obs_over_forecast"] for x in t]); b = np.array([x["model_factor"] for x in t])
            shape[call] = round(float(np.corrcoef(a, b)[0, 1]), 2)
    print("sector-shape correlation (station vs model):", shape)
    # nowcast skill: the last hour's ratio (station / expected) applied to the next hours, fading with TAU_H
    weights = {}
    for call in lake:
        D = data[call]; ob, fc = hourly_obs(call), forecast(call)
        f16 = D["fs"] * (1 - alpha * np.clip(D["sx"] / 12.0, 0, 1))
        a, b = [], []
        for k in sorted(ob):
            if k not in fc or fc[k][0] < MIN_MS:
                continue
            e = cs[call] * interp16(f16, fc[k][1]) * fc[k][0]
            r = min(max(ob[k][0] / max(e, 0.2), 0.3), 3.0)
            k2 = str(np.datetime64(k) + np.timedelta64(1, "h"))[:16]
            if k2 in ob and k2 in fc and fc[k2][0] >= MIN_MS:
                e2 = cs[call] * interp16(f16, fc[k2][1]) * fc[k2][0]
                w = math.exp(-1.0 / TAU_H)
                a.append(ob[k2][0] - e2); b.append(ob[k2][0] - e2 * (1 + (r - 1) * w))
        gain = 1 - math.sqrt(np.mean(np.square(b))) / math.sqrt(np.mean(np.square(a))) if a else 0.0
        weights[call] = round(max(gain, 0.0), 3)
    print("nowcast weights (error cut at +1 h):", weights)
    out = dict(fitted=datetime.now().astimezone().isoformat(timespec="minutes"), min_forecast_ms=MIN_MS, station_hours=n,
               alpha=float(alpha), alpha_fit=float(alpha_fit), alpha_prior=PRIOR, tau_h=TAU_H, nowcast_weights=weights, rms_ms=dict(no_shelter=base_none, shore_only=base_shore, fitted=best_err),
               alpha_scan=[dict(alpha=float(a), rms=e) for e, a in scores],
               stations={c: dict(name=(LAKE.get(c) or HILL.get(c)), lat=POS[c][0], lon=POS[c][1], siting=round(cs.get(c, cs_hill.get(c, float("nan"))), 3),
                                 cell=data[c]["cell"], cell_m=round(data[c]["cell_m"]), hours=int(len(data[c]["rows"])),
                                 factor16=[round(float(x), 3) for x in data[c]["fs"] * (1 - alpha * np.clip(data[c]["sx"] / 12.0, 0, 1))])
                         for c in lake + list(HILL)},
               sectors=sectors, shape_correlation=shape)
    json.dump(out, open(os.path.join(HERE, "calibration.json"), "w"), indent=1)
    print("written calibration.json")


if __name__ == "__main__":
    main()
