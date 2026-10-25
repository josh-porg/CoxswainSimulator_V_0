"""OKID/ERA: the boat's response to each kind of call, as an identified state-space model.

    .venv/Scripts/python.exe research/callmodel/okid.py [--root data/local/coxing] [--charles-only]

Observer/Kalman filter IDentification (Juang, Phan, Horta & Longman 1993) with the Eigensystem
Realization Algorithm (Juang & Pappa 1985), per stroke:

* outputs y_k: boat speed and stroke rate, each minus its own slow trend (a centred 41-stroke running mean),
  so the race plan (start, settle, sprint) is not what is fitted;
* inputs u_k: call onsets in stroke k, one channel per call target (CODEBOOK.md via codes_<session>.txt):
  power; rhythm (rate, length, together); technique (catch, finish, body, set); morale; information and
  tactics; steering. An onset is the first phrase of a same-target run (a new target, or a gap over 3 s);
  count phrases carry no target and are not inputs;
* the observer form y_k = D u_k + sum_{i=1..p} [b_i u_{k-i} + a_i y_{k-i}] + e_k, fitted by least squares on
  every session at once without crossing a session's boundary. Its impulse response gives the system
  Markov parameters Y_0 = D, Y_k = b_k + sum_{i=1..min(k,p)} a_i Y_{k-i}; ERA on their Hankel matrix gives a
  minimal (A, B, C, D) whose eigenvalues are the response's modes (time constants in strokes).

The observer form needs no assumption that calls are independent of the boat: a coxswain who calls
because of what the boat just did is a call driven by past outputs, which the regression conditions on.
It cannot separate a call from something unrecorded that drove both (a crew's own surge the coxswain
heard coming).

Checks, all on held-out data:
* leave one session out: does the call history improve the one-step prediction of speed beyond the
  boat's own past (the same observer form without inputs)?
* null: each session's inputs circularly shifted (calls keep their spacing, lose their alignment);
* intervals: moving-block bootstrap (40-stroke blocks within sessions) of the whole fit.
Writes data/local/coxing/okid.json and okid.png. --charles-only fits the HOCR 2026 race alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CODE, "..", ".."))
sys.path.insert(0, CODE)
import call_properties as CP                                                  # noqa: E402

SESSIONS = ["practice_01", "practice_02", "practice_03", "practice_04", "practice_05", "practice_06", "practice_07",
            "practice_08", "race_01", "race_04", "race_05", "race_06"]
CHARLES = "race_06"
GROUPS = {"power": {"power"}, "rhythm": {"rate", "length", "together"}, "technique": {"catch", "finish", "body", "set"},
          "morale": {"morale"}, "info/tactic": {"info", "tactic"}, "steer": {"steer"}}
OUTPUTS = ["speed", "rate"]
TREND = 41
GAP = 3.0
P_LAGS = (2, 3, 4, 6, 8)
HORIZON = 12
RNG = np.random.default_rng(11)


def running_mean(x, w):
    k = np.ones(w) / w
    pad = np.r_[np.full(w // 2, x[0]), x, np.full(w // 2, x[-1])]
    return np.convolve(pad, k, mode="valid")[:len(x)]


def load(root, sid):
    d = os.path.join(root, "sessions", sid)
    rows = list(csv.DictReader(open(os.path.join(d, "boat.csv"))))
    t = np.array([float(r["t"]) for r in rows])
    y = np.column_stack([np.array([float(r[o]) for r in rows]) for o in OUTPUTS])
    ok = np.all(np.isfinite(y), 1) & (y[:, 0] > 0.5)
    t, y = t[ok], y[ok]
    y = y - np.column_stack([running_mean(y[:, j], TREND) for j in range(y.shape[1])])
    phrases = json.load(open(os.path.join(d, "phrases.json"), encoding="utf-8"))
    codes = CP.load_codes(os.path.join(root, "experiment", "codes_%s.txt" % sid))
    group_of = {tg: g for g, tgs in GROUPS.items() for tg in tgs}
    u = np.zeros((len(t), len(GROUPS)))
    last = (None, -1e9)
    edges = np.r_[t, t[-1] + np.median(np.diff(t))]
    for p in sorted(phrases, key=lambda q: q["t"]):
        g = group_of.get(CP.code_of(p["i"], codes)["target"])
        if g is None:
            continue
        if g == last[0] and p["t"] - last[1] <= GAP:
            last = (g, p["t"]); continue
        last = (g, p["t"])
        k = int(np.searchsorted(edges, p["t"], side="right")) - 1     # the stroke the call falls in
        if 0 <= k < len(t):
            u[k, list(GROUPS).index(g)] += 1.0
    return dict(id=sid, t=t, y=y, u=u)


def design(S, p, with_inputs=True, sl=None):
    """Rows of the observer form for one session (no row crosses the session's start)."""
    y, u = S["y"], S["u"]
    n = len(y)
    idx = np.arange(p, n) if sl is None else np.array([k for k in sl if k >= p])
    cols = []
    if with_inputs:
        cols.append(u[idx])                                        # D u_k
    for i in range(1, p + 1):
        if with_inputs:
            cols.append(u[idx - i])
        cols.append(y[idx - i])
    X = np.hstack(cols + [np.ones((len(idx), 1))])
    return X, y[idx]


def fit(sessions, p, with_inputs=True, ridge=1e-3):
    X = np.vstack([design(S, p, with_inputs)[0] for S in sessions]); Y = np.vstack([design(S, p, with_inputs)[1] for S in sessions])
    G = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(G, X.T @ Y)                             # (n_reg, q)


def markov(theta, p, m, q, horizon=HORIZON):
    """System Markov parameters Y_0..Y_horizon (q x m each) from the observer-form coefficients."""
    D = theta[:m].T
    b, a = [], []
    off = m
    for i in range(p):
        b.append(theta[off:off + m].T); off += m
        a.append(theta[off:off + q].T); off += q
    Y = [D]
    for k in range(1, horizon + 1):
        Yk = b[k - 1].copy() if k <= p else np.zeros((q, m))
        for i in range(1, min(k, p) + 1):
            Yk = Yk + a[i - 1] @ Y[k - i]
        Y.append(Yk)
    return np.array(Y)                                             # (horizon+1, q, m)


def era(Y, n_order, r=5):
    """Eigensystem Realization from Y_1..Y_{2r}: (A, B, C), singular values."""
    q, m = Y.shape[1], Y.shape[2]
    H0 = np.block([[Y[1 + i + j] for j in range(r)] for i in range(r)])
    H1 = np.block([[Y[2 + i + j] for j in range(r)] for i in range(r)])
    U, s, Vt = np.linalg.svd(H0)
    U, S, Vt = U[:, :n_order], np.diag(s[:n_order]), Vt[:n_order]
    Si = np.diag(1 / np.sqrt(np.diag(S))); Sh = np.diag(np.sqrt(np.diag(S)))
    A = Si @ U.T @ H1 @ Vt.T @ Si
    B = (Sh @ Vt)[:, :m]
    C = (U @ Sh)[:q, :]
    return A, B, C, s


def one_step_rmse(theta, S, p, with_inputs):
    X, Y = design(S, p, with_inputs)
    return float(np.sqrt(np.mean((Y[:, 0] - X @ theta[:, 0]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "local", "coxing"))
    ap.add_argument("--charles-only", action="store_true")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--nulls", type=int, default=400)
    a = ap.parse_args()
    ids = [CHARLES] if a.charles_only else SESSIONS
    S = [load(a.root, s) for s in ids]
    m, q = len(GROUPS), len(OUTPUTS)
    print("sessions: " + ", ".join("%s (%d strokes, %d calls)" % (s["id"], len(s["t"]), int(s["u"].sum())) for s in S))
    print("calls by target: " + ", ".join("%s %d" % (g, int(sum(s["u"][:, j].sum() for s in S))) for j, g in enumerate(GROUPS)))
    # observer order by leave-one-session-out (or, for one session, leave-one-third-out) one-step speed error
    folds = [(tr, te) for te in range(len(S)) for tr in [[s for i, s in enumerate(S) if i != te]]] if len(S) > 1 else None
    cv = {}
    for p in P_LAGS:
        if folds:
            e_in = [one_step_rmse(fit(tr, p, True), S[te], p, True) for tr, te in folds]
            e_no = [one_step_rmse(fit(tr, p, False), S[te], p, False) for tr, te in folds]
        else:
            n = len(S[0]["t"]); thirds = np.array_split(np.arange(n), 3)
            e_in, e_no = [], []
            for th in thirds:
                tr = dict(S[0]); keep = np.setdiff1d(np.arange(n), th)
                tr = dict(id="tr", t=S[0]["t"][keep], y=S[0]["y"][keep], u=S[0]["u"][keep])
                te = dict(id="te", t=S[0]["t"][th], y=S[0]["y"][th], u=S[0]["u"][th])
                e_in.append(one_step_rmse(fit([tr], p, True), te, p, True)); e_no.append(one_step_rmse(fit([tr], p, False), te, p, False))
        cv[p] = dict(with_calls=float(np.mean(e_in)), boat_only=float(np.mean(e_no)),
                     calls_help_in=int(np.sum(np.array(e_in) < np.array(e_no))), folds=len(e_in))
    p = min(cv, key=lambda k: cv[k]["with_calls"])
    print("observer order by held-out one-step speed error (m/s): " +
          "; ".join("p=%d: with calls %.4f, boat only %.4f (calls better in %d of %d)" % (k, v["with_calls"], v["boat_only"], v["calls_help_in"], v["folds"]) for k, v in cv.items()))
    theta = fit(S, p, True)
    Y = markov(theta, p, m, q)
    # ERA: order from the Hankel singular values (keep 99% of the energy, at most 8)
    A0, B0, C0, sv = era(Y, 1)
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    n_ord = int(min(8, np.searchsorted(energy, 0.99) + 1))
    A, B, C, _ = era(Y, n_ord)
    lam = np.linalg.eigvals(A)
    modes = [dict(abs=float(abs(z)), angle_deg=float(np.degrees(np.angle(z))), tau_strokes=float(-1 / np.log(abs(z))) if 0 < abs(z) < 1 else None) for z in lam]
    # check the realisation reproduces the Markov parameters
    Yr = np.array([theta[:m].T] + [C @ np.linalg.matrix_power(A, k - 1) @ B for k in range(1, HORIZON + 1)])
    fit_err = float(np.linalg.norm(Yr[1:] - Y[1:]) / np.linalg.norm(Y[1:]))
    # bootstrap: moving blocks of 40 strokes within sessions, refit, Markov parameters
    boots = []
    for _ in range(a.boot):
        rs = []
        for s in S:
            n = len(s["t"]); nb = max(1, n // 40)
            starts = RNG.integers(0, max(1, n - 40), nb)
            sel = np.concatenate([np.arange(st, min(n, st + 40)) for st in starts])
            rs.append((s, sel))
        X = np.vstack([design(s, p, True, sl)[0] for s, sl in rs]); Yv = np.vstack([design(s, p, True, sl)[1] for s, sl in rs])
        th = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Yv)
        boots.append(markov(th, p, m, q))
    boots = np.array(boots)
    # null: inputs circularly shifted within each session
    nulls = []
    for _ in range(a.nulls):
        Sn = [dict(s, u=np.roll(s["u"], int(RNG.integers(20, len(s["t"]) - 20)), axis=0)) for s in S]
        nulls.append(markov(fit(Sn, p, True), p, m, q))
    nulls = np.array(nulls)
    win = slice(3, 6)                                              # strokes 3-5 after the call, as short_horizon.py
    out = dict(sessions=ids, strokes=int(sum(len(s["t"]) for s in S)), observer_order=p, cv=cv, era_order=n_ord,
               hankel_singular_values=sv[:12].round(5).tolist(), realisation_rel_error=fit_err, modes=modes, groups=list(GROUPS), responses={})
    print("\nERA order %d (Hankel energy %.3f); realisation reproduces the Markov parameters to %.1f%%; modes: %s"
          % (n_ord, energy[n_ord - 1], 100 * fit_err, ", ".join("|z| %.2f%s" % (md["abs"], " (tau %.1f strokes)" % md["tau_strokes"] if md["tau_strokes"] else "") for md in modes)))
    for j, g in enumerate(GROUPS):
        sp = Y[:, 0, j] * 1000; lo, hi = np.percentile(boots[:, :, 0, j] * 1000, [2.5, 97.5], axis=0)
        w = float(np.mean(Y[win, 0, j]) * 1000); wn = np.mean(nulls[:, win, 0, j], axis=1) * 1000
        pval = float(np.mean(np.abs(wn - wn.mean()) >= abs(w - wn.mean())))
        rate_w = float(np.mean(Y[win, 1, j]))
        out["responses"][g] = dict(speed_mm_s=sp.round(1).tolist(), ci_lo=lo.round(1).tolist(), ci_hi=hi.round(1).tolist(),
                                   window_3_5_mm_s=round(w, 1), null_sd=round(float(wn.std()), 1), p=round(pval, 3),
                                   rate_window_3_5_spm=round(rate_w, 3), n_calls=int(sum(s["u"][:, j].sum() for s in S)))
        print("%-12s n=%3d  speed response (mm/s), strokes 0-8: %s | strokes 3-5 %+5.1f (null sd %.1f, p %.3f); rate %+.2f spm"
              % (g, out["responses"][g]["n_calls"], " ".join("%+5.0f" % x for x in sp[:9]), w, wn.std(), pval, rate_w))
    name = "okid_charles.json" if a.charles_only else "okid.json"
    json.dump(out, open(os.path.join(a.root, name), "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        k = np.arange(HORIZON + 1)
        for ax, (j, g) in zip(axs.flat, enumerate(GROUPS)):
            lo, hi = np.percentile(boots[:, :, 0, j] * 1000, [2.5, 97.5], axis=0)
            nlo, nhi = np.percentile(nulls[:, :, 0, j] * 1000, [2.5, 97.5], axis=0)
            ax.fill_between(k, nlo, nhi, color="0.85", label="shifted-call null")
            ax.fill_between(k, lo, hi, color="C0", alpha=0.3, label="95% bootstrap")
            ax.plot(k, Y[:, 0, j] * 1000, "C0-o", ms=3); ax.axhline(0, color="k", lw=0.6)
            ax.set_title("%s (n=%d)" % (g, out["responses"][g]["n_calls"])); ax.set_ylabel("speed, mm/s")
        for ax in axs[1]:
            ax.set_xlabel("strokes after the call")
        axs[0, 0].legend(fontsize=7)
        fig.suptitle("OKID/ERA speed response to one call (%s)" % ("HOCR 2026 race only" if a.charles_only else "%d sessions" % len(S)))
        fig.tight_layout(); fig.savefig(os.path.join(a.root, name.replace(".json", ".png")), dpi=110)
    except ImportError:
        pass
    print("written", os.path.join(a.root, name))


if __name__ == "__main__":
    main()
