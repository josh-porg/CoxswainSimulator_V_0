"""What a hard rudder costs: the Tail of the Lake 2026 per-stroke GPS, speed against turn rate.

    .venv/Scripts/python.exe research/raceday/rudder_totl.py

Both squad boats raced Tail of the Lake on 4 Oct 2026 with the CBGPS (per-stroke speed, rate and
position). The coxswain of the four called the turn ("I'm about to start our turn", "press from all
of you to make this turn") and the speed fell through it. Here the heading comes from the GPS track,
the turn rate from its change per second, and the speed through each stroke is compared with the
crew's own pace either side (a rolling median over +-12 strokes, turns excluded) at the same rate.

Output: the speed lost per stroke against turn rate, a fitted loss ``dv/v = k * omega^2`` (omega in
deg/s), and the summary of each turn.
"""
import csv
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "local", "raceday"))   # its data and outputs (gitignored)
COXBOX = os.path.join(HERE, "..", "coxbox")
FILES = {
    "four (women's masters 4+)": "Sharing file  CoxBox 011009704 20261004 1237PM.csv",
    "eight (mixed masters 8+)": "Sharing file  CoxBox 011009704 20261004 0803AM.csv",
}
R = 6371000.0


def load(path):
    rows = list(csv.reader(open(path, encoding="utf-8", errors="ignore")))
    i = next(k for k, r in enumerate(rows) if r and r[0].startswith("Per-Stroke Data"))
    out = []
    for r in rows[i + 3:]:
        if len(r) < 24 or not r[0].strip():
            continue
        try:
            h, m, s = r[3].split(":")
            t = int(h) * 3600 + int(m) * 60 + float(s)
            out.append((t, float(r[1]), float(r[5]), float(r[8]), float(r[22]), float(r[23])))
        except ValueError:
            continue
    a = np.array(out)
    return dict(t=a[:, 0], d=a[:, 1], v=a[:, 2], rate=a[:, 3], lat=a[:, 4], lon=a[:, 5])


def heading_rate(s, half=2):
    """Heading (deg from north) and turn rate (deg/s, + = to starboard) from the track."""
    lat0 = np.radians(s["lat"].mean())
    x = np.radians(s["lon"]) * R * np.cos(lat0)
    y = np.radians(s["lat"]) * R
    n = len(x)
    hd = np.full(n, np.nan)
    for i in range(n):
        a, b = max(0, i - half), min(n - 1, i + half)
        hd[i] = np.degrees(np.arctan2(x[b] - x[a], y[b] - y[a]))
    u = np.degrees(np.unwrap(np.radians(hd)))
    om = np.gradient(u, s["t"])
    return hd % 360, om, u


def per_stroke(path):
    """Each stroke's time, speed, rate, heading (unwrapped, deg), turn rate and loss against pace."""
    s = load(path)
    hd, om, u = heading_rate(s)
    n = len(s["v"])
    turning = np.abs(om) > 1.0
    base = np.full(n, np.nan)
    rate_base = np.full(n, np.nan)
    for i in range(n):
        a, b = max(0, i - 12), min(n, i + 13)
        w = np.arange(a, b)
        w = w[~turning[w]]
        if len(w) >= 6:
            base[i] = np.median(s["v"][w])
            rate_base[i] = np.median(s["rate"][w])
    # speed at the crew's own rate-pace: v scales about as rate^(1/3) at fixed run (small correction)
    expected = base * (s["rate"] / rate_base) ** (1 / 3)
    loss = 1.0 - s["v"] / expected
    ok = np.isfinite(loss) & (s["t"] > 30) & (s["t"] < s["t"].max() - 20)
    return s, om, u, turning, expected, loss, ok


def analyse(name, path):
    s, om, u, turning, expected, loss, ok = per_stroke(path)
    n = len(s["v"])
    # fit dv/v = c0 + k * omega^2 over all strokes (c0 absorbs the baseline's small bias)
    X = np.column_stack([np.ones(ok.sum()), om[ok] ** 2])
    coef, *_ = np.linalg.lstsq(X, loss[ok], rcond=None)
    resid = loss[ok] - X @ coef
    cov = np.linalg.inv(X.T @ X) * resid.var(ddof=2)
    bins = [0, 0.5, 1, 1.5, 2, 3, 5]
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ok & (np.abs(om) >= lo) & (np.abs(om) < hi)
        if m.sum():
            table.append(dict(omega=f"{lo}-{hi}", strokes=int(m.sum()), loss_pct=100 * float(np.median(loss[m]))))
    # turns: runs of |omega| > 1 deg/s lasting >= 4 strokes
    turns = []
    i = 0
    while i < n:
        if turning[i] and ok[i]:
            j = i
            while j + 1 < n and turning[j + 1]:
                j += 1
            if j - i + 1 >= 4:
                seg = slice(i, j + 1)
                lost_s = float(np.nansum((s["v"][seg] < expected[seg]) * 0 + (1 / s["v"][seg] - 1 / expected[seg])
                                         * np.gradient(s["d"])[seg]))
                turns.append(dict(t0=float(s["t"][i]), t1=float(s["t"][j]), d0=float(s["d"][i]),
                                  degrees=float(u[j] - u[i]), metres=float(s["d"][j] - s["d"][i]),
                                  mean_omega=float(np.mean(np.abs(om[seg]))),
                                  peak_omega=float(np.max(np.abs(om[seg]))),
                                  speed=float(np.mean(s["v"][seg])), pace_speed=float(np.nanmean(expected[seg])),
                                  loss_pct=100 * float(np.nanmean(loss[seg])), seconds_lost=lost_s))
            i = j + 1
        else:
            i += 1
    return dict(name=name, strokes=int(ok.sum()), k=float(coef[1]), k_se=float(np.sqrt(cov[1, 1])),
                c0=float(coef[0]), table=table, turns=turns, mean_speed=float(np.mean(s["v"][ok])))


def main():
    out = []
    for name, f in FILES.items():
        r = analyse(name, os.path.join(COXBOX, f))
        out.append(r)
        print("\n%s: %d strokes, mean %.2f m/s" % (name, r["strokes"], r["mean_speed"]))
        print("  dv/v = %.4f + %.4f (+-%.4f) * omega^2   [omega deg/s]" % (r["c0"], r["k"], r["k_se"]))
        for b in r["table"]:
            print("  |omega| %-7s %4d strokes  loss %+5.1f%%" % (b["omega"], b["strokes"], b["loss_pct"]))
        for t in r["turns"]:
            print("  turn at %4.0f-%4.0f s (%4.0f m): %+4.0f deg over %3.0f m, mean %.1f / peak %.1f deg/s,"
                  " %.2f vs %.2f m/s (%.1f%%), %.1f s lost"
                  % (t["t0"], t["t1"], t["d0"], t["degrees"], t["metres"], t["mean_omega"], t["peak_omega"],
                     t["speed"], t["pace_speed"], t["loss_pct"], t["seconds_lost"]))
    json.dump(out, open(os.path.join(HERE, "rudder_totl.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
