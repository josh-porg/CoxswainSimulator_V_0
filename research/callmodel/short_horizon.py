r"""Short-horizon call responses: what the boat does in the 1-8 strokes after a call.

    python research/callmodel/short_horizon.py --root data/local/coxing --out short_horizon.json

Coxswains judge a call by whether it takes within three to five strokes; the earlier analyses
looked for slow responses over tens of strokes against a 61-stroke trend. Here each call episode
is compared with the four strokes before it, stroke by stroke, and against two references:

* the null: the same statistic with the episode times circularly shifted within each session
  (keeps each session's call spacing and boat structure, destroys their alignment);
* the boat's own momentum: an autoregression on the previous four strokes, fitted on the whole
  session, predicts each post-call stroke; the call effect is what is left. A coxswain who calls
  what they can already feel is answered by this term, not by the call.

A session needs ``boat.csv`` (per stroke: t, speed, rate), ``phrases.json`` ([{i, t, text}]) and
``labels_phrases.txt`` ("index function valence" per line). An episode is the first phrase of a
run with the same function, the next starting more than 3 s after the last; counts (C) and
other talk (O) never start one. The ``control`` session kind is reported apart and never pooled.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

K_POST = 8           # strokes after the call
K_PRE = 4            # strokes before it, the baseline
GAP = 3.0            # s between phrases that still belong to one episode
AR_P = 4             # strokes of the boat's own past


def load(path):
    rows = list(csv.DictReader(open(os.path.join(path, "boat.csv"))))
    t = np.array([float(r["t"]) for r in rows])
    sp = np.array([float(r["speed"]) for r in rows])
    ra = np.array([float(r["rate"]) for r in rows])
    phrases = json.load(open(os.path.join(path, "phrases.json"), encoding="utf-8"))
    lab = {}
    for line in open(os.path.join(path, "labels_phrases.txt"), encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        i, f, v = line.split()[:3]
        lab[int(i)] = (f, v)
    meta = json.load(open(os.path.join(path, "meta.json")))
    return meta, t, sp, ra, phrases, lab


def episodes(phrases, lab):
    """[(t, function, valence, text)]: the first phrase of each same-function run."""
    out, last_f, last_t = [], None, -1e9
    for p in phrases:
        f, v = lab.get(p["i"], ("O", "0"))
        if f in ("C", "O"):
            continue
        if f == last_f and p["t"] - last_t <= GAP:
            last_t = p["t"]
            continue
        out.append((p["t"], f, v, p["text"]))
        last_f, last_t = f, p["t"]
    return out


def ar_fit(y):
    """Least-squares AR(AR_P) with intercept, on a whole session's strokes."""
    X = np.column_stack([np.ones(len(y) - AR_P)] + [y[AR_P - j - 1:len(y) - j - 1] for j in range(AR_P)])
    beta, *_ = np.linalg.lstsq(X, y[AR_P:], rcond=None)
    return beta


def ar_forecast(beta, past, k):
    """k-step forecast from the last AR_P values (oldest first)."""
    hist = list(past)
    out = []
    for _ in range(k):
        x = np.r_[1.0, hist[::-1][:AR_P]]
        nxt = float(x @ beta)
        out.append(nxt)
        hist.append(nxt)
    return np.array(out)


def responses(t, y, beta, ev_times):
    """(raw, beyond-momentum) responses, each (n_events, K_POST); NaN where strokes run out."""
    raw = np.full((len(ev_times), K_POST), np.nan)
    res = np.full((len(ev_times), K_POST), np.nan)
    for n, tc in enumerate(ev_times):
        j = int(np.searchsorted(t, tc))          # first stroke at or after the call
        if j - K_PRE < 0 or j + K_POST > len(t):
            continue
        base = y[j - K_PRE:j].mean()
        raw[n] = y[j:j + K_POST] - base
        res[n] = y[j:j + K_POST] - ar_forecast(beta, y[j - AR_P:j], K_POST)
    return raw, res


def summarise(sessions, select, n_null=2000, seed=0):
    """Mean response per stroke after the selected episodes, pooled over sessions, with a
    circular-shift null on the same sessions."""
    rng = np.random.default_rng(seed)
    obs = {"speed": [], "rate": [], "speed_beyond": [], "rate_beyond": []}
    per = []
    for s in sessions:
        ev = [e for e in s["episodes"] if select(e)]
        if not ev:
            continue
        times = np.array([e[0] for e in ev])
        sr, sres = responses(s["t"], s["speed"], s["beta_s"], times)
        rr, rres = responses(s["t"], s["rate"], s["beta_r"], times)
        obs["speed"].append(sr); obs["rate"].append(rr)
        obs["speed_beyond"].append(sres); obs["rate_beyond"].append(rres)
        per.append((s, times))
    if not per:
        return None
    stat = {k: np.nanmean(np.vstack(v), axis=0) for k, v in obs.items()}
    n = int(np.sum(~np.isnan(np.vstack(obs["speed"])[:, 0])))
    null = {k: [] for k in obs}
    for _ in range(n_null):
        acc = {k: [] for k in obs}
        for s, times in per:
            span = s["t"][-1] - s["t"][0]
            shifted = s["t"][0] + (times - s["t"][0] + rng.uniform(0.15, 0.85) * span) % span
            sr, sres = responses(s["t"], s["speed"], s["beta_s"], shifted)
            rr, rres = responses(s["t"], s["rate"], s["beta_r"], shifted)
            acc["speed"].append(sr); acc["rate"].append(rr)
            acc["speed_beyond"].append(sres); acc["rate_beyond"].append(rres)
        for k in obs:
            null[k].append(np.nanmean(np.vstack(acc[k]), axis=0))
    out = dict(n=n)
    for k in obs:
        nk = np.array(null[k])
        # two-sided p per stroke, and for the 3-5 stroke window the coxswain watches
        p = (np.sum(np.abs(nk - nk.mean(0)) >= np.abs(stat[k] - nk.mean(0)), axis=0) + 1) / (len(nk) + 1)
        w = slice(2, 5)                                         # strokes 3, 4, 5
        wobs = stat[k][w].mean()
        wnull = nk[:, w].mean(1)
        pw = (np.sum(np.abs(wnull - wnull.mean()) >= abs(wobs - wnull.mean())) + 1) / (len(wnull) + 1)
        out[k] = dict(mean=stat[k].round(4).tolist(), null_mean=nk.mean(0).round(4).tolist(),
                      null_sd=nk.std(0).round(4).tolist(), p=p.round(4).tolist(),
                      window_3_5=round(float(wobs), 4), window_null_sd=round(float(wnull.std()), 4),
                      p_window=round(float(pw), 4))
    return out


def load_sessions(root, ids):
    out = []
    for sid in ids:
        path = os.path.join(root, "sessions", sid)
        meta, t, sp, ra, phrases, lab = load(path)
        out.append(dict(id=sid, kind=meta["kind"], boat=meta["boat_class"], t=t, speed=sp, rate=ra,
                        beta_s=ar_fit(sp), beta_r=ar_fit(ra), episodes=episodes(phrases, lab)))
    return out


GROUPS = {
    "effort (E)": lambda e: e[1] == "E",
    "rate, rhythm, ratio, length (R)": lambda e: e[1] == "R",
    "technical (T)": lambda e: e[1] == "T",
    "praise (P)": lambda e: e[1] == "P",
    "motivation, positive (M+)": lambda e: e[1] == "M" and e[2] == "+",
    "competitor, tactical (K)": lambda e: e[1] == "K",
    "steering, turn press (S)": lambda e: e[1] == "S",
    "split and race information (I)": lambda e: e[1] == "I",
    "any positive valence": lambda e: e[2] == "+",
    "any negative valence": lambda e: e[2] == "-",
    "effort or motivation, positive": lambda e: e[1] in "EM" and e[2] == "+",
    "effort or motivation, negative": lambda e: e[1] in "EM" and e[2] == "-",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sessions", nargs="+", default=["race_04", "race_05", "practice_01", "practice_02"])
    ap.add_argument("--control", default="control_01")
    ap.add_argument("--nulls", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sess = load_sessions(a.root, a.sessions)
    report = {}
    print("sessions: " + ", ".join("%s (%s %s, %d episodes, %d strokes)" % (s["id"], s["kind"], s["boat"],
                                   len(s["episodes"]), len(s["t"])) for s in sess))
    for name, sel in GROUPS.items():
        r = summarise(sess, sel, a.nulls)
        if r is None:
            continue
        report[name] = r
        sp, spb, ra = r["speed"], r["speed_beyond"], r["rate"]
        print("\n%-34s n=%d" % (name, r["n"]))
        print("   speed, mm/s vs the 4 strokes before:  " + " ".join("%+6.0f" % (1000 * x) for x in sp["mean"]))
        print("   null mean (shifted times):            " + " ".join("%+6.0f" % (1000 * x) for x in sp["null_mean"]))
        print("   beyond the boat's own momentum:       " + " ".join("%+6.0f" % (1000 * x) for x in spb["mean"]))
        print("   strokes 3-5: speed %+.0f mm/s (null sd %.0f, p %.3f); beyond momentum %+.0f (p %.3f);"
              " rate %+.2f spm (p %.3f)"
              % (1000 * sp["window_3_5"], 1000 * sp["window_null_sd"], sp["p_window"],
                 1000 * spb["window_3_5"], spb["p_window"], ra["window_3_5"], ra["p_window"]))
    if a.control:
        csess = load_sessions(a.root, [a.control])
        report["control"] = {}
        print("\nCONTROL (%s): the transitions, each against its own four strokes before" % a.control)
        for name, sel in (("build (rate up)", lambda e: e[1] == "B"), ("paddle (rate down)", lambda e: e[1] == "D"),
                          ("way enough (stop)", lambda e: e[1] == "W"), ("row (start)", lambda e: e[1] == "G")):
            r = summarise(csess, sel, a.nulls)
            if r is None:
                continue
            report["control"][name] = r
            print("   %-20s n=%d  rate: %s spm   speed: %s mm/s   strokes 3-5 rate p %.3f, speed p %.3f"
                  % (name, r["n"], " ".join("%+5.1f" % x for x in r["rate"]["mean"]),
                     " ".join("%+5.0f" % (1000 * x) for x in r["speed"]["mean"]),
                     r["rate"]["p_window"], r["speed"]["p_window"]))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
