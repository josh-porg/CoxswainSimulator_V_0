"""Robustness and inference checks from the examiner's review (CODEBOOK addenda of 14, 15 and 16 Nov 2026, items 1-15;
item 10, the forecast order, is call_properties.forecast_order with run_experiment.py AR_ORDER=4 as the sensitivity run).

    .venv/Scripts/python.exe research/coxing/experiment/examiner_checks.py

All post hoc, specified in the addendum before they were run. Reads the sessions, the primary codes and the study's
results.json; writes data/local/coxing/experiment/review/checks.json and fig_robust.pdf/png beside the paper. No earlier
result file is changed.

1 Benjamini-Hochberg over the secondary (hypothesis-specific) tests      6 specification curve for the power-call response
2 random-effects meta-analysis over nine crew-occasions                  7 forecast diagnostics (order, out-of-sample R2)
3 event study with the forecast made ten strokes before the call         8 minimum detectable effects for the nulls
4 stroke-level regression with neighbouring calls                        9 sample size with between-occasion variation
5 step-response template for the plateau gain
"""
import json
import os
import sys
from itertools import product

import numpy as np
from scipy import stats

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, "..", "..", "callmodel"))
import call_properties as CP  # noqa: E402

EXP = os.path.abspath(os.path.join(CODE, "..", "..", "..", "data", "local", "coxing", "experiment"))
SDIR = os.path.join(EXP, "..", "sessions")
OUTDIR = os.path.join(EXP, "review")
SES = ["race_01", "practice_01", "practice_02", "race_04", "race_05", "practice_03", "practice_04", "practice_05",
       "practice_06", "practice_07", "practice_08", "race_06", "race_08", "race_09", "race_07"]
OCC = {"race_01": "Pink Ribbon Row, four", "practice_01": "Practice races, eight", "practice_02": "Practice races, eight",
       "race_04": "Tail of the Lake, eight", "race_05": "Tail of the Lake, four",
       **{"practice_%02d" % k: "800 m pieces, four" for k in range(3, 9)},
       "race_06": "Head of the Charles, four", "race_07": "Head of the Charles, four (2nd coxswain)",
       "race_08": "Head of the Lake, eight", "race_09": "Head of the Lake, four"}
TECH = {"catch", "finish", "body", "set", "together"}
CLASSES = ["power", "rate", "length", "tech", "steer", "info", "morale", "tactic", "other", "count"]
K = CP.K_POST
NPERM = 2000
TEMPLATE = 1 - np.exp(-np.arange(1, 11) / 3.6)
ANCHOR = json.load(open(os.path.join(EXP, "..", "sync", "count_anchor.json")))["0.55"]


def call_class(e):
    t, dec = e["target"], e["stage"] == "dec"
    if t == "count":
        return "count"
    if t in ("power", "rate", "length") and dec:
        return t
    if t in TECH and dec:
        return "tech"
    if t in ("steer", "info", "morale", "tactic"):
        return t
    return "other"


class Data:
    """Sessions with residual matrices cached by (AR order, course) and episodes cached by (gap, sync)."""

    def __init__(self):
        self.s = {sid: CP.load_session(os.path.join(SDIR, sid)) for sid in SES}
        self.codes = {sid: CP.load_codes(os.path.join(EXP, "codes_%s.txt" % sid)) for sid in SES}
        self._R, self._E, self._O = {}, {}, {}

    def order(self, sid):
        """The piece's main forecast order (BIC on its speed series, or AR_ORDER from the environment)."""
        if sid not in self._O:
            self._O[sid] = CP.forecast_order(self.s[sid])
        return self._O[sid]

    def valid(self, sid):
        return max(self.order(sid), CP.AR_P) + 5, self.n(sid) - K

    def R(self, sid, ch="speed", p=None, course=False):
        p = p or self.order(sid)
        key = (sid, ch, p, course)
        if key not in self._R:
            s = self.s[sid]
            ex = CP.course_exog(s) if course else None
            self._R[key] = CP.residual_matrix(s[ch], CP.ar_fit(s[ch], p, ex), K, p, ex)
        return self._R[key]

    def episodes(self, sid, gap=3.0, anchored=False):
        key = (sid, gap, anchored)
        if key not in self._E:
            s = self.s[sid]
            d = ANCHOR.get(sid, {}).get("shift_s", 0.0) if anchored else 0.0
            ph = [dict(p, t=p["t"] + d, t_end=(p["t_end"] + d) if p.get("t_end") is not None else None) for p in s["phrases"]]
            ep = CP.episodes(ph, self.codes[sid], gap)
            for e in ep:
                e["j"] = int(np.searchsorted(s["t"], e["t"]))
                e["cls"] = call_class(e)
            self._E[key] = ep
        return self._E[key]

    def n(self, sid):
        return len(self.s[sid]["t"])


def window_stat(R, kind):
    """Per-stroke statistic of a call hypothetically made at each stroke: a window mean or the template amplitude."""
    if kind == "template":
        return R[:, :10] @ TEMPLATE / (TEMPLATE @ TEMPLATE)
    h0, h1 = kind
    return np.nanmean(R[:, h0 - 1:h1], axis=1)


def shift(rng, js, lo, hi, local):
    span = hi - lo
    s = int(rng.integers(10, 41)) * (1 if rng.random() < 0.5 else -1) if local else int(rng.integers(int(0.15 * span), int(0.85 * span)))
    return lo + (js - lo + s) % span


def perm(V, JS, RNG, local, seed, n_perm=NPERM, agg=None):
    """V, JS, RNG: per piece, the per-stroke statistic (1-D or 2-D: strokes x outputs), the call strokes and the valid
    range. Mean over all calls (or agg) observed and under shifts; returns est (obs - null mean), null sd, two-sided p."""
    rng = np.random.default_rng(seed)
    agg = agg or (lambda parts: np.nanmean(np.concatenate(parts), axis=0))
    obs = np.atleast_1d(agg([v[j] for v, j in zip(V, JS)]))
    null = np.array([np.atleast_1d(agg([v[shift(rng, j, lo, hi, local)] for v, j, (lo, hi) in zip(V, JS, RNG)])) for _ in range(n_perm)])
    mu = null.mean(0)
    p = (np.sum(np.abs(null - mu) >= np.abs(obs - mu), axis=0) + 1) / (n_perm + 1)
    return obs - mu, null.std(0), p


def both(V, JS, RNG, seed):
    eg, sg, pg = perm(V, JS, RNG, False, seed)
    el, sl, pl = perm(V, JS, RNG, True, seed + 1000)
    f = lambda x: x.tolist() if len(x) > 1 else float(x[0])
    return dict(n=int(sum(len(j) for j in JS)), est=f(eg), sd=f(sg), p=f(pg), est_local=f(el), sd_local=f(sl), p_local=f(pl))


def calls_of(D, sid, cls, lo, hi, gap=3.0, anchored=False, any_power=False):
    ep = D.episodes(sid, gap, anchored)
    if any_power:
        js = [e["j"] for e in ep if e["target"] == "power"]
    else:
        js = [e["j"] for e in ep if e["cls"] == cls]
    js = np.array(js, int)
    return js[(js >= lo) & (js < hi)]


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    run = 1.0
    for rank, k in enumerate(o[::-1]):
        run = min(run, p[k] * len(p) / (len(p) - rank))
        q[k] = run
    return q


def random_effects(theta, se):
    theta, se = np.asarray(theta, float), np.asarray(se, float)
    k, w = len(theta), 1 / se ** 2
    fixed = np.sum(w * theta) / w.sum()
    Q = float(np.sum(w * (theta - fixed) ** 2))
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    mu = float(np.sum(ws * theta) / ws.sum())
    se_hk = float(np.sqrt(np.sum(ws * (theta - mu) ** 2) / ((k - 1) * ws.sum())))
    t = stats.t.ppf(0.975, k - 1)
    pi = stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + se_hk ** 2) if k > 2 else np.nan
    return dict(k=k, fixed=float(fixed), mu=mu, se=se_hk, ci=[mu - t * se_hk, mu + t * se_hk], p=float(2 * stats.t.sf(abs(mu / se_hk), k - 1)),
                tau=float(np.sqrt(tau2)), I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, Q=Q,
                prediction=[float(mu - pi), float(mu + pi)])


# ------------------------------------------------------------------ 1
def check1(out):
    H = json.load(open(os.path.join(EXP, "results.json")))["hypotheses"]
    names, pg, pl = [], [], []
    for k_, v in H.items():
        p_, q_ = np.atleast_1d(v["p"]), np.atleast_1d(v["p_local"])
        if len(p_) > 4:            # response curves are not scalar tests
            continue
        terms = v.get("terms") or ([""] if len(p_) == 1 else [str(i) for i in range(len(p_))])
        for t, a, b in zip(terms, p_, q_):
            names.append(k_ + (": " + t if t else ""))
            pg.append(float(a))
            pl.append(float(b))
    qg, ql = bh(pg), bh(pl)
    rows = [dict(test=n_, p=a, q=float(b), p_local=c, q_local=float(d)) for n_, a, b, c, d in zip(names, pg, qg, pl, ql)]
    out["bh"] = dict(n_tests=len(rows), survive_global=[r["test"] for r in rows if r["q"] < 0.05],
                     survive_local=[r["test"] for r in rows if r["q_local"] < 0.05],
                     survive_both=[r["test"] for r in rows if r["q"] < 0.05 and r["q_local"] < 0.05], rows=rows)
    print("1. BH over %d secondary tests; q<0.05 under both nulls: %s" % (len(rows), out["bh"]["survive_both"]))
    r = next(x for x in rows if x["test"] == "H7_power_calls_speed_3_5")
    print("   power calls, speed 3-5: p %.4f q %.3f | local p %.4f q %.3f" % (r["p"], r["q"], r["p_local"], r["q_local"]))


# ------------------------------------------------------------------ 2
def check2(D, out):
    res = {}
    specs = [("power_speed_3_5", "power", "speed", (3, 5), False), ("power_speed_3_5_course", "power", "speed", (3, 5), True),
             ("rate_rate_1_3", "rate", "rate", (1, 3), False), ("length_dps_3_5", "length", "dps", (3, 5), False)]
    for name, cls, ch, win, course in specs:
        piece = {}
        for sid in SES:
            lo, hi = D.valid(sid)
            js = calls_of(D, sid, cls, lo, hi)
            if len(js) < 5:
                continue
            piece[sid] = both([window_stat(D.R(sid, ch, None, course), win)], [js], [(lo, hi)], 300 + SES.index(sid))
        r = dict(pieces=piece)
        for null, e_, s_ in (("global", "est", "sd"), ("local", "est_local", "sd_local")):
            occ = {}
            for sid, v in piece.items():
                occ.setdefault(OCC[sid], []).append((v[e_], v[s_], v["n"]))
            labels = list(occ)
            th, se, nn = [], [], []
            for lab in labels:
                e, s, n_ = (np.array(x) for x in zip(*occ[lab]))
                w = 1 / s ** 2
                th.append(float(np.sum(w * e) / w.sum()))
                se.append(float(np.sqrt(1 / w.sum())))
                nn.append(int(n_.sum()))
            m = random_effects(th, se)
            loo = [random_effects(np.delete(th, i), np.delete(se, i))["mu"] for i in range(len(th))]
            pos = int(np.sum(np.array(th) > 0))
            m.update(occasions=[dict(label=a, est=b, se=c, n=d) for a, b, c, d in zip(labels, th, se, nn)],
                     loo_range=[float(min(loo)), float(max(loo))], loo={a: float(b) for a, b in zip(labels, loo)},
                     n_positive=pos, sign_p=float(stats.binomtest(pos, len(th), 0.5).pvalue))
            for tag, drop in (("without_pink_ribbon", "Pink Ribbon Row, four"), ("without_hotl_four", "Head of the Lake, four")):
                keep = [i for i, a in enumerate(labels) if a != drop]
                x = random_effects(np.array(th)[keep], np.array(se)[keep])
                m[tag] = dict(mu=x["mu"], ci=x["ci"], p=x["p"])
            r[null] = m
        res[name] = r
        g = r["global"]
        sc = 1000 if ch != "rate" else 1
        print("2. %-24s occasions %d: RE %+.1f [%+.1f, %+.1f] p %.3f, tau %.1f, I2 %.0f%%, PI [%+.1f, %+.1f], LOO %+.1f..%+.1f, positive %d/%d (sign p %.3f) | local RE %+.1f [%+.1f, %+.1f] p %.3f" % (
            name, g["k"], sc * g["mu"], sc * g["ci"][0], sc * g["ci"][1], g["p"], sc * g["tau"], 100 * g["I2"], sc * g["prediction"][0], sc * g["prediction"][1],
            sc * g["loo_range"][0], sc * g["loo_range"][1], g["n_positive"], g["k"], g["sign_p"],
            sc * r["local"]["mu"], sc * r["local"]["ci"][0], sc * r["local"]["ci"][1], r["local"]["p"]))
        if name.startswith("power"):
            print("     without Pink Ribbon Row %+.1f (p %.3f); without Head of the Lake four %+.1f (p %.3f)" % (
                sc * g["without_pink_ribbon"]["mu"], g["without_pink_ribbon"]["p"], sc * g["without_hotl_four"]["mu"], g["without_hotl_four"]["p"]))
    # pooled call-level estimate without each of the two pieces (the paper's estimator)
    for tag, drop in (("pooled_all", None), ("pooled_without_pink_ribbon", "race_01"), ("pooled_without_hotl_four", "race_09")):
        sel = [s_ for s_ in SES if s_ != drop]
        rng_ = [D.valid(s_) for s_ in sel]
        res[tag] = both([window_stat(D.R(s_), (3, 5)) for s_ in sel], [calls_of(D, s_, "power", lo, hi) for s_, (lo, hi) in zip(sel, rng_)], rng_, 350)
        print("     %-28s n=%d %+.1f (sd %.1f) p %.4f | local %+.1f p %.4f" % (tag, res[tag]["n"], 1000 * res[tag]["est"], 1000 * res[tag]["sd"], res[tag]["p"],
                                                                         1000 * res[tag]["est_local"], res[tag]["p_local"]))
    out["meta"] = res


# ------------------------------------------------------------------ 3
def check3(D, out):
    res = {}
    for course in (False, True):
        V, JS, RNG = [], [], []
        for sid in SES:
            lo, hi = D.valid(sid)[0] + 10, D.n(sid) - K
            R = D.R(sid, "speed", None, course)
            V.append(np.vstack([np.full((10, K), np.nan), R[:-10]]))        # row j holds the residuals from origin j-10
            JS.append(calls_of(D, sid, "power", lo, hi))
            RNG.append((lo, hi))
        curve = both(V, JS, RNG, 400)
        agg = lambda parts: np.array([np.nanmean(np.concatenate(parts)[:, 5:8]), np.nanmean(np.concatenate(parts)[:, 8:10]),
                                      np.nanmean(np.concatenate(parts)[:, 12:15]), np.nanmean(np.concatenate(parts)[:, 15:20])])
        eg, sg, pg = perm(V, JS, RNG, False, 410, agg=agg)
        el, sl, pl = perm(V, JS, RNG, True, 411, agg=agg)
        res["course" if course else "momentum"] = dict(
            strokes=list(range(-10, 0)) + list(range(1, 11)), curve=curve,      # stroke 1 = the first stroke after the call, as in the paper
            windows={w: dict(est=float(eg[i]), sd=float(sg[i]), p=float(pg[i]), est_local=float(el[i]), sd_local=float(sl[i]), p_local=float(pl[i]))
                     for i, w in enumerate(("placebo_-5_-3", "pre_-2_-1", "post_3_5", "post_6_10_unplanned"))})
        w = res["course" if course else "momentum"]["windows"]
        print("3. event study (%s), origin 10 strokes before, n=%d: placebo -5..-3 %+.1f (sd %.1f) p %.3f, local p %.3f | -2..-1 %+.1f p %.3f | +3..+5 %+.1f (sd %.1f) p %.3f, local %+.1f p %.3f" % (
            "course" if course else "momentum", curve["n"], 1000 * w["placebo_-5_-3"]["est"], 1000 * w["placebo_-5_-3"]["sd"], w["placebo_-5_-3"]["p"], w["placebo_-5_-3"]["p_local"],
            1000 * w["pre_-2_-1"]["est"], w["pre_-2_-1"]["p"], 1000 * w["post_3_5"]["est"], 1000 * w["post_3_5"]["sd"], w["post_3_5"]["p"],
            1000 * w["post_3_5"]["est_local"], w["post_3_5"]["p_local"]))
        u = w["post_6_10_unplanned"]
        print("     unplanned follow-up, strokes 6-10: %+.1f (sd %.1f) p %.4f | local %+.1f (sd %.1f) p %.4f" % (
            1000 * u["est"], 1000 * u["sd"], u["p"], 1000 * u["est_local"], 1000 * u["sd_local"], u["p_local"]))
    out["event_study"] = res


# ------------------------------------------------------------------ 4
def check4(D, out):
    res = {}
    for course in (False, True):
        Y, base, DD, spans = [], [], [], []
        for q, sid in enumerate(SES):
            s, n = D.s[sid], D.n(sid)
            lo, hi = D.valid(sid)
            y = window_stat(D.R(sid, "speed", None, course), (3, 5))[lo:hi]
            pre = np.array([CP.contingency(s, j) for j in range(lo, hi)])
            fe = np.zeros((hi - lo, len(SES)))
            fe[:, q] = 1
            base.append(np.column_stack([fe, np.arange(lo, hi) / n - 0.5, pre]))
            d = np.zeros((hi - lo, len(CLASSES)))
            for e in D.episodes(sid):
                if lo <= e["j"] < hi:
                    d[e["j"] - lo, CLASSES.index(e["cls"])] += 1
            Y.append(y)
            DD.append(d)
            spans.append(hi - lo)
        Yc, Bc = np.concatenate(Y), np.vstack(base)
        ok = ~np.isnan(Yc)

        def fit(shifts, neighbours):
            parts = []
            for d, sft in zip(DD, shifts):
                d = np.roll(d, sft, axis=0)
                cols = [d]
                if neighbours in ("both", "before"):
                    cols.append(sum(np.roll(d, l, axis=0) for l in range(1, 6)))
                if neighbours in ("both", "after"):
                    cols.append(sum(np.roll(d, -l, axis=0) for l in range(1, 5)))
                parts.append(np.hstack(cols))
            X = np.hstack([Bc, np.vstack(parts)])
            b, *_ = np.linalg.lstsq(X[ok], Yc[ok], rcond=None)
            return b[Bc.shape[1]:Bc.shape[1] + len(CLASSES)]

        r = {}
        for neighbours in ("alone", "both", "before", "after"):      # before/after separately: unplanned follow-up
            obs = fit([0] * len(SES), neighbours)
            x = {}
            for null, local in (("global", False), ("local", True)):
                rng = np.random.default_rng(500 + local)       # same shifts for every variant
                draws = np.array([fit([(int(rng.integers(10, 41)) * (1 if rng.random() < 0.5 else -1)) if local else int(rng.integers(int(0.15 * sp), int(0.85 * sp)))
                                       for sp in spans], neighbours) for _ in range(NPERM)])
                mu = draws.mean(0)
                p = (np.sum(np.abs(draws - mu) >= np.abs(obs - mu), axis=0) + 1) / (NPERM + 1)
                x[null] = {c: dict(est=float(obs[i] - mu[i]), sd=float(draws.std(0)[i]), p=float(p[i])) for i, c in enumerate(CLASSES)}
            r[{"alone": "alone", "both": "with_neighbours", "before": "before_only_unplanned", "after": "after_only_unplanned"}[neighbours]] = x
        r["n_strokes"] = int(ok.sum())
        r["n_calls"] = {c: int(sum(d[:, i].sum() for d in DD)) for i, c in enumerate(CLASSES)}
        res["course" if course else "momentum"] = r
        for c in ("power", "tech", "steer", "rate", "length"):
            a, b = r["alone"], r["with_neighbours"]
            print("4. stroke-level (%s) %-7s n=%4d alone %+.1f (sd %.1f) p %.3f / local %+.1f p %.3f | with neighbours %+.1f (sd %.1f) p %.3f / local %+.1f p %.3f" % (
                "course" if course else "momentum", c, r["n_calls"][c], 1000 * a["global"][c]["est"], 1000 * a["global"][c]["sd"], a["global"][c]["p"],
                1000 * a["local"][c]["est"], a["local"][c]["p"], 1000 * b["global"][c]["est"], 1000 * b["global"][c]["sd"], b["global"][c]["p"],
                1000 * b["local"][c]["est"], b["local"][c]["p"]))
        for tag in ("before_only_unplanned", "after_only_unplanned"):
            g, l_ = r[tag]["global"]["power"], r[tag]["local"]["power"]
            print("     power, %s: %+.1f (sd %.1f) p %.3f / local %+.1f p %.3f" % (tag, 1000 * g["est"], 1000 * g["sd"], g["p"], 1000 * l_["est"], l_["p"]))
    out["stroke_level"] = res


# ------------------------------------------------------------------ 5
def check5(D, out):
    res = {}
    for course in (False, True):
        RNG = [D.valid(s_) for s_ in SES]
        JS = [calls_of(D, s_, "power", lo, hi) for s_, (lo, hi) in zip(SES, RNG)]
        res["course" if course else "momentum"] = both([window_stat(D.R(s_, "speed", None, course), "template") for s_ in SES], JS, RNG, 600)
        r = res["course" if course else "momentum"]
        print("5. template plateau (%s): %+.1f (sd %.1f) p %.4f | local %+.1f (sd %.1f) p %.4f" % (
            "course" if course else "momentum", 1000 * r["est"], 1000 * r["sd"], r["p"], 1000 * r["est_local"], 1000 * r["sd_local"], r["p_local"]))
    out["template"] = res


# ------------------------------------------------------------------ 6
def check6(D, out):
    dims = dict(ar=[2, 4, 6, 8, "bic"], course=[False, True], null=["global", "local"], window=[(3, 5), (2, 6), (4, 8), "template"],
                definition=["directive", "any power"], gap=[2.0, 3.0, 5.0], sync=["recorded", "counts"], start=["all", "no first 20"])
    paper = dict(ar="bic", course=False, null="global", window=(3, 5), definition="directive", gap=3.0, sync="recorded", start="all")
    rows, cache = [], {}
    for q, vals in enumerate(product(*dims.values())):
        sp = dict(zip(dims, vals))
        lo0 = 20 if sp["start"] == "no first 20" else 13          # 13 = the longest AR order's first valid stroke
        RNG = [(lo0, D.n(s_) - K) for s_ in SES]
        kV = (sp["ar"], sp["course"], sp["window"])
        if kV not in cache:
            cache[kV] = [window_stat(D.R(s_, "speed", None if sp["ar"] == "bic" else sp["ar"], sp["course"]), sp["window"]) for s_ in SES]
        JS = [calls_of(D, s_, "power", lo, hi, sp["gap"], sp["sync"] == "counts", sp["definition"] == "any power") for s_, (lo, hi) in zip(SES, RNG)]
        e, s, p = perm(cache[kV], JS, RNG, sp["null"] == "local", 7000 + q, 1000)
        rows.append(dict(sp, window="template" if sp["window"] == "template" else "%d-%d" % sp["window"], n=int(sum(len(j) for j in JS)),
                         est=float(e[0]), sd=float(s[0]), p=float(p[0]), paper=all(sp[k_] == v for k_, v in paper.items())))
    est, pv = np.array([r["est"] for r in rows]), np.array([r["p"] for r in rows])
    by = {}
    for d_ in dims:
        for v in sorted({str(r[d_]) for r in rows}):
            m = np.array([str(r[d_]) == v for r in rows])
            by["%s=%s" % (d_, v)] = dict(n=int(m.sum()), median=float(np.median(est[m])), share_sig=float(np.mean(pv[m] < 0.05)), share_pos=float(np.mean(est[m] > 0)))
    tm = np.array([r["window"] == "template" for r in rows])
    out["spec_curve"] = dict(n=len(rows), median=float(np.median(est)), q05=float(np.percentile(est, 5)), q95=float(np.percentile(est, 95)),
                             min=float(est.min()), max=float(est.max()), share_positive=float(np.mean(est > 0)), share_sig=float(np.mean(pv < 0.05)),
                             share_sig_positive=float(np.mean((pv < 0.05) & (est > 0))),
                             windows_only=dict(n=int((~tm).sum()), median=float(np.median(est[~tm])), min=float(est[~tm].min()), max=float(est[~tm].max()),
                                               share_sig=float(np.mean(pv[~tm] < 0.05))),
                             paper=next(r for r in rows if r["paper"]), by_dimension=by, rows=rows)
    sc = out["spec_curve"]
    print("6. specification curve: %d specs, median %+.1f, 5-95%% %+.1f..%+.1f, range %+.1f..%+.1f, positive %.0f%%, p<0.05 %.0f%% (window specs only: median %+.1f, p<0.05 %.0f%%)" % (
        sc["n"], 1000 * sc["median"], 1000 * sc["q05"], 1000 * sc["q95"], 1000 * sc["min"], 1000 * sc["max"], 100 * sc["share_positive"], 100 * sc["share_sig"],
        1000 * sc["windows_only"]["median"], 100 * sc["windows_only"]["share_sig"]))
    for k_, v in by.items():
        print("     %-22s median %+5.1f  p<0.05 %3.0f%%" % (k_, 1000 * v["median"], 100 * v["share_sig"]))


# ------------------------------------------------------------------ 7
def check7(D, out):
    res = {}
    for sid in SES:
        s = D.s[sid]
        y, n = s["speed"], D.n(sid)
        bic = {}
        for p in range(1, 9):
            cols = [np.ones(n - 8)] + [y[8 - j - 1:n - j - 1] for j in range(p)]
            Xp = np.column_stack(cols)
            b, *_ = np.linalg.lstsq(Xp, y[8:], rcond=None)
            rss = float(np.sum((y[8:] - Xp @ b) ** 2))
            bic[p] = (n - 8) * np.log(rss / (n - 8)) + (p + 1) * np.log(n - 8)
        r = dict(n=n, bic_order=int(min(bic, key=bic.get)))
        for tag_, p_ in (("ar4", 4), ("selected", D.order(sid))):
            e = D.R(sid, "speed", p_)[:, 0]
            e = e[~np.isnan(e)]
            e = e - e.mean()
            ac = np.array([np.sum(e[k_:] * e[:-k_]) / np.sum(e * e) for k_ in range(1, 13)])
            Q = len(e) * (len(e) + 2) * np.sum(ac ** 2 / (len(e) - np.arange(1, 13)))
            r["ljung_box_p_" + tag_] = float(stats.chi2.sf(Q, 12 - p_))          # lag 12, so that order eight keeps 4 df
            r["resid_sd_" + tag_] = float(e.std())
        for course, p in ((False, 4), (True, 4), (False, D.order(sid)), (True, D.order(sid))):
            ex = CP.course_exog(s) if course else None
            if course and ex is None:
                continue
            e1, e35, t1, t35 = [], [], [], []
            edges = np.linspace(p, n, 6).astype(int)
            for a, b_ in zip(edges[:-1], edges[1:]):
                rows_ = np.array([t for t in range(p, n) if not a <= t < b_])
                cols = [np.ones(len(rows_))] + [y[rows_ - j - 1] for j in range(p)]
                if ex is not None:
                    cols += list(ex[rows_].T)
                beta, *_ = np.linalg.lstsq(np.column_stack(cols), y[rows_], rcond=None)
                mean_tr = y[rows_].mean()
                for j in range(a, b_ - 5):
                    hist = list(y[j - p:j])
                    f = []
                    for h in range(5):
                        x = np.r_[1.0, hist[::-1][:p]]
                        if ex is not None:
                            x = np.r_[x, ex[j + h]]
                        f.append(float(x @ beta))
                        hist.append(f[-1])
                    e1.append(y[j] - f[0])
                    t1.append(y[j] - mean_tr)
                    e35.append(np.mean(y[j + 2:j + 5]) - np.mean(f[2:5]))
                    t35.append(np.mean(y[j + 2:j + 5]) - mean_tr)
            tag = ("course" if course else "momentum") + ("_ar4" if p == 4 else "")
            r["oos_r2_1_" + tag] = float(1 - np.sum(np.square(e1)) / np.sum(np.square(t1)))
            r["oos_r2_3_5_" + tag] = float(1 - np.sum(np.square(e35)) / np.sum(np.square(t35)))
        res[sid] = r
    med = lambda k_: float(np.median([v[k_] for v in res.values() if k_ in v]))
    rng_ = lambda k_: [float(min(v[k_] for v in res.values() if k_ in v)), float(max(v[k_] for v in res.values() if k_ in v))]
    summ = dict(bic_orders={str(o): int(sum(v["bic_order"] == o for v in res.values())) for o in range(1, 9)},
                ljung_box_rejected_ar4=int(sum(v["ljung_box_p_ar4"] < 0.05 for v in res.values())),
                ljung_box_rejected_selected=int(sum(v["ljung_box_p_selected"] < 0.05 for v in res.values())),
                resid_sd_ar4=med("resid_sd_ar4"), resid_sd_selected=med("resid_sd_selected"),
                **{k_: dict(median=med(k_), range=rng_(k_)) for k_ in ("oos_r2_1_momentum", "oos_r2_3_5_momentum", "oos_r2_1_course", "oos_r2_3_5_course",
                                                                       "oos_r2_1_momentum_ar4", "oos_r2_3_5_momentum_ar4")})
    out["forecast"] = dict(pieces=res, summary=summ)
    print("7. forecast: BIC orders %s; Ljung-Box (lag 12) p<0.05 in %d of %d pieces at order four, %d at the selected order; out-of-sample R2 one-step median %.2f (%.2f..%.2f), 3-5 strokes %.2f (%.2f..%.2f); order four %.2f and %.2f; with course %.2f and %.2f" % (
        {k_: v for k_, v in summ["bic_orders"].items() if v}, summ["ljung_box_rejected_ar4"], len(res), summ["ljung_box_rejected_selected"],
        summ["oos_r2_1_momentum"]["median"], *summ["oos_r2_1_momentum"]["range"],
        summ["oos_r2_3_5_momentum"]["median"], *summ["oos_r2_3_5_momentum"]["range"], summ["oos_r2_1_momentum_ar4"]["median"], summ["oos_r2_3_5_momentum_ar4"]["median"],
        summ["oos_r2_1_course"]["median"], summ["oos_r2_3_5_course"]["median"]))


# ------------------------------------------------------------------ 8, 9
def check8(out):
    R = json.load(open(os.path.join(EXP, "results.json")))
    mde = {r["label"]: dict(n=r["n"], sd=r["sd"], mde80=2.80 * r["sd"], sd_local=r["sd_local"], mde80_local=2.80 * r["sd_local"]) for r in R["primary"]}
    cd = json.load(open(os.path.join(EXP, "coxswain_difference.json")))
    cox = {}
    for k_, v in cd.items():
        if isinstance(v, dict) and "diff_sd" in v:
            cox[k_] = dict(diff=v["diff"], sd=v["diff_sd"], mde80=2.80 * v["diff_sd"])
    out["mde"] = dict(primary=mde, coxswain_difference=cox)
    print("8. MDE (80%% power): primary contrasts %.0f..%.0f mm/s; e.g. focus %.0f, established %.0f, analogy %.0f" % (
        1000 * min(v["mde80"] for v in mde.values()), 1000 * max(v["mde80"] for v in mde.values()),
        1000 * mde["H11 internal minus external focus"]["mde80"], 1000 * mde["H17 established cue"]["mde80"], 1000 * mde["H18 analogy minus explicit"]["mde80"]))
    if cox:
        print("   coxswain differences:", {k_: round(v["mde80"], 3) for k_, v in cox.items()})


def check9(out):
    R = json.load(open(os.path.join(EXP, "results.json")))["heterogeneity"]
    Rc = json.load(open(os.path.join(EXP, "results_course.json")))["heterogeneity"]
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    res = {}
    for tag, sigma, delta, key in (("momentum", R["sd_calls"], 0.015, "power_speed_3_5"), ("course", Rc["sd_calls"], 0.010, "power_speed_3_5_course")):
        tau = out["meta"][key]["global"]["tau"]
        per_arm_iid = 2 * (z * sigma / delta) ** 2
        occ = {}
        for n_arm in (20, 40, 80):
            var_one = tau ** 2 + 2 * sigma ** 2 / n_arm           # variance of one occasion's estimated effect around the mean effect
            occ[str(n_arm)] = float(np.ceil(z ** 2 * var_one / delta ** 2))
        res[tag] = dict(sigma=sigma, delta=delta, tau=tau, calls_per_arm_no_clustering=float(per_arm_iid), occasions_needed_by_calls_per_arm=occ,
                        occasions_floor_from_tau=float(np.ceil(z ** 2 * tau ** 2 / delta ** 2)))
        print("9. sample size (%s): sigma %.0f, delta %.0f, tau %.1f mm/s: %d calls per arm ignoring clustering; occasions needed at 20/40/80 calls per arm per occasion: %s; floor from tau alone %d" % (
            tag, 1000 * sigma, 1000 * delta, 1000 * tau, per_arm_iid, "/".join("%d" % occ[k_] for k_ in ("20", "40", "80")), res[tag]["occasions_floor_from_tau"]))
    out["sample_size"] = res


# ------------------------------------------------------------------ 11-13 (CODEBOOK addendum of 15 Nov 2026)
def check11(D, out):
    """Cross-fitted window: each occasion is scored on the window chosen from the other eight."""
    WINS = [(1, 2), (3, 5), (2, 6), (4, 8), (6, 10)]
    occs = list(dict.fromkeys(OCC[s_] for s_ in SES))
    res = {}
    for tag, course, local in (("momentum_global", False, False), ("course_local", True, True)):
        RNG = {s_: D.valid(s_) for s_ in SES}
        JS = {s_: calls_of(D, s_, "power", *RNG[s_]) for s_ in SES}
        V = {w: {s_: window_stat(D.R(s_, "speed", None, course), w) for s_ in SES} for w in WINS}
        rows = []
        for q, occ in enumerate(occs):
            train = [s_ for s_ in SES if OCC[s_] != occ]
            test = [s_ for s_ in SES if OCC[s_] == occ and len(JS[s_])]
            zs = {}
            for w in WINS:
                e, sd, _ = perm([V[w][s_] for s_ in train], [JS[s_] for s_ in train], [RNG[s_] for s_ in train], local, 1100 + q, 1000)
                zs[w] = float(e[0] / sd[0])
            w = max(zs, key=zs.get)
            e, sd, p_ = perm([V[w][s_] for s_ in test], [JS[s_] for s_ in test], [RNG[s_] for s_ in test], local, 1200 + q, NPERM)
            rows.append(dict(occasion=occ, window="%d-%d" % w, train_z=zs[w], est=float(e[0]), sd=float(sd[0]), p=float(p_[0]),
                             n=int(sum(len(JS[s_]) for s_ in test))))
        m = random_effects([r["est"] for r in rows], [r["sd"] for r in rows])
        z = float(sum(r["est"] / r["sd"] for r in rows) / np.sqrt(len(rows)))
        m.update(rows=rows, n_positive=int(sum(r["est"] > 0 for r in rows)), stouffer_z=z, stouffer_p=float(2 * stats.norm.sf(abs(z))),
                 windows_chosen={w_: int(sum(r["window"] == w_ for r in rows)) for w_ in sorted({r["window"] for r in rows})})
        res[tag] = m
        print("11. cross-fitted window (%s): windows chosen %s; held-out RE %+.1f [%+.1f, %+.1f] p %.3f, tau %.1f; positive %d/%d; Stouffer z %.2f p %.4f" % (
            tag, m["windows_chosen"], 1000 * m["mu"], 1000 * m["ci"][0], 1000 * m["ci"][1], m["p"], 1000 * m["tau"], m["n_positive"], m["k"], z, m["stouffer_p"]))
    out["crossfit"] = res


def check12(D, out):
    """Power calls identified without the primary coder: lexical rule coder, practitioner guide, and both coders."""
    sys.path.insert(0, CODE)
    import guide_validation as GV
    import rulecoder as RC
    from sklearn.metrics.pairwise import cosine_similarity
    guide, _ = GV.load_guide()
    coded = {s_: [p for p in D.s[s_]["phrases"] if p["i"] in D.codes[s_]] for s_ in SES}      # every phrase that is not a bare count
    vec = GV.vectoriser([g["phrase"] for g in guide] + [p["text"] for s_ in SES for p in coded[s_]])
    G = vec([g["phrase"] for g in guide])
    drive = np.array([g["section"] == "Drive" for g in guide])
    blank = dict(stage="-", level="-", action="n", function="-", focus="-", ref="n", discrep="n", appraisal="neu", form="-", arousal="-")
    JS = {k_: {} for k_ in ("primary", "rule", "guide", "both")}
    for s_ in SES:
        st = D.s[s_]
        lo, hi = D.valid(s_)
        keep = lambda js, lo=lo, hi=hi: np.array(sorted(j for j in js if lo <= j < hi), int)
        JS["primary"][s_] = calls_of(D, s_, "power", lo, hi)
        rule = {p["i"]: dict(blank, target=RC.code(p["text"])["target"]) for p in coded[s_]}
        JS["rule"][s_] = keep(int(np.searchsorted(st["t"], e["t"])) for e in CP.episodes(st["phrases"], rule) if e["target"] == "power")
        S = cosine_similarity(vec([p["text"] for p in coded[s_]]), G)
        lab = [p for p, row in zip(coded[s_], S) if drive[row.argmax()] and row.max() >= 0.5]
        starts, last = [], -1e9
        for p in lab:                                             # an episode: a run of such phrases within 3 s
            if p["t"] - last > 3.0:
                starts.append(p["t"])
            last = max(p["t"], p.get("t_end") or p["t"])
        JS["guide"][s_] = keep(int(np.searchsorted(st["t"], t_)) for t_ in starts)
        JS["both"][s_] = keep(e["j"] for e in D.episodes(s_) if e["cls"] == "power" and any(RC.code(t_)["target"] == "power" for t_ in e["texts"]))
    res = {}
    RNG = [D.valid(s_) for s_ in SES]
    for lab_, js in JS.items():
        prim = JS["primary"]
        tot = sum(len(js[s_]) for s_ in SES)
        near = float(np.mean([np.min(np.abs(prim[s_] - j)) <= 1 if len(prim[s_]) else False for s_ in SES for j in js[s_]])) if tot else float("nan")
        r = dict(n=int(tot), share_at_a_primary_power_call=near)
        for tag, course in (("momentum", False), ("course", True)):
            for wname, w in (("3_5", (3, 5)), ("template", "template")):
                r["%s_%s" % (tag, wname)] = both([window_stat(D.R(s_, "speed", None, course), w) for s_ in SES], [js[s_] for s_ in SES], RNG, 1300)
        res[lab_] = r
        a, b_, c_, d_ = r["momentum_3_5"], r["course_3_5"], r["momentum_template"], r["course_template"]
        print("12. power calls by %-8s n=%3d (%.0f%% at a primary power call): 3-5 %+.1f (sd %.1f) p %.3f / local %+.1f p %.3f | course %+.1f p %.3f / local %+.1f p %.3f | template %+.1f p %.4f / local %+.1f p %.4f | course template %+.1f p %.4f" % (
            lab_, r["n"], 100 * near, 1000 * a["est"], 1000 * a["sd"], a["p"], 1000 * a["est_local"], a["p_local"], 1000 * b_["est"], b_["p"], 1000 * b_["est_local"], b_["p_local"],
            1000 * c_["est"], c_["p"], 1000 * c_["est_local"], c_["p_local"], 1000 * d_["est"], d_["p"]))
    out["labellers"] = res


def check13(out):
    """Human agreement so far: the human coding tool's own analysis (analysis.json beside its log)."""
    import subprocess
    hc = os.path.join(CODE, "..", "human_coding", "analyse.py")
    r = subprocess.run([sys.executable, hc], check=True, capture_output=True, text=True)
    a = json.load(open(os.path.join(EXP, "..", "human_coding", "analysis.json")))
    out["human_agreement"] = a
    print("13. human agreement so far:\n" + "\n".join("    " + l for l in r.stdout.splitlines()[:16]))


# ------------------------------------------------------------------ 14, 15 (CODEBOOK addendum of 16 Nov 2026)
def fast_residual_matrix(y, beta, p, exog=None):
    """call_properties.residual_matrix, vectorised over forecast origins (same values)."""
    n = len(y)
    R = np.full((n, K), np.nan)
    J = np.arange(p, n - K + 1)
    H = np.column_stack([y[J - l - 1] for l in range(p)])            # most recent stroke first
    for h in range(K):
        f = beta[0] + H @ beta[1:1 + p]
        if exog is not None:
            f = f + exog[J + h] @ beta[1 + p:]
        R[J, h] = y[J + h] - f
        H = np.column_stack([f, H[:, :-1]])
    return R


def placebo_draw(rng, pool, matched=True):
    """One dip-matched placebo set: for every power call a stroke of the same piece, at least ten strokes from any
    power call, within 15% of the piece's length, among the five nearest in pre-call speed change; none used twice."""
    out, gap = {}, []
    for sid, (js, cand, cont, n) in pool.items():
        used, pick = set(), []
        for j in rng.permutation(js):
            c = [k for k in cand if abs(k - j) <= 0.15 * n and k not in used]
            if not c:
                continue
            if matched:
                c = sorted(c, key=lambda k: abs(cont[k] - cont[j]))[:5]
            k = int(c[int(rng.integers(len(c)))])
            used.add(k)
            pick.append(k)
            gap.append(cont[k] - cont[j])
        out[sid] = np.array(sorted(pick), int)
    return out, float(np.mean(gap))


def placebo_pool(D, clear=10):
    pool = {}                                    # clear: placebo strokes lie more than this many strokes from every power call
    for sid in SES:
        lo, hi = D.valid(sid)
        js = calls_of(D, sid, "power", lo, hi)
        s = D.s[sid]
        cont = np.array([CP.contingency(s, j) if j >= 6 else np.nan for j in range(D.n(sid))])
        cand = [k for k in range(max(lo, 6), hi) if len(js) == 0 or np.min(np.abs(js - k)) > clear]
        if len(js):
            pool[sid] = (js, cand, cont, D.n(sid))
    return pool


def check14(D, out):
    pool = placebo_pool(D)
    res = {}
    for tag, course in (("momentum", False), ("course", True)):
        V = {w: {sid: window_stat(D.R(sid, "speed", None, course), w) for sid in pool} for w in ((3, 5), (6, 10), "template")}
        real = {w: float(np.nanmean(np.concatenate([V[w][sid][pool[sid][0]] for sid in pool]))) for w in V}
        rng = np.random.default_rng(1400 + course)
        draws, gaps, nplac = {w: [] for w in V}, [], []
        for _ in range(500):
            pl, g = placebo_draw(rng, pool)
            gaps.append(g)
            nplac.append(sum(len(v) for v in pl.values()))
            for w in V:
                draws[w].append(float(np.nanmean(np.concatenate([V[w][sid][pl[sid]] for sid in pool]))))
        r = dict(n_power=int(sum(len(v[0]) for v in pool.values())), n_placebo_mean=float(np.mean(nplac)), match_gap=float(np.mean(gaps)),
                 real_pre_change=float(np.nanmean(np.concatenate([pool[sid][2][pool[sid][0]] for sid in pool]))))
        for w, name in (((3, 5), "3_5"), ((6, 10), "6_10"), ("template", "template")):
            d = np.array(draws[w])
            r[name] = dict(real=real[w], placebo_mean=float(d.mean()), placebo_lo=float(np.percentile(d, 2.5)), placebo_hi=float(np.percentile(d, 97.5)),
                           p_real_vs_placebo=float((np.sum(d >= real[w]) + 1) / (len(d) + 1)))
        res[tag] = r
        print("14. dip-matched placebo (%s): %d power calls, %.0f placebo strokes per draw, pre-call change %.1f (match gap %+.1f) mm/s" % (
            tag, r["n_power"], r["n_placebo_mean"], 1000 * r["real_pre_change"], 1000 * r["match_gap"]))
        for name in ("3_5", "6_10", "template"):
            x = r[name]
            print("      %-8s real %+.1f | placebo %+.1f (95%% of draws %+.1f to %+.1f) | p(real <= placebo) %.3f" % (
                name, 1000 * x["real"], 1000 * x["placebo_mean"], 1000 * x["placebo_lo"], 1000 * x["placebo_hi"], x["p_real_vs_placebo"]))
    # unplanned diagnostics (CODEBOOK notes of 16 Nov): unmatched placebo, and dip-matched with a one-stroke exclusion
    diag = {}
    span = lambda pl_: sum(D.valid(sid)[1] - max(D.valid(sid)[0], 6) for sid in pl_)
    for name, clear, matched in (("unmatched_clear10", 10, False), ("matched_clear1", 1, True), ("unmatched_clear1", 1, False)):
        pl_pool = placebo_pool(D, clear)
        elig = sum(len(v[1]) for v in pl_pool.values()) / span(pl_pool)
        for tag, course in (("momentum", False), ("course", True)):
            V = {w: {sid: window_stat(D.R(sid, "speed", None, course), w) for sid in pl_pool} for w in ((3, 5), (6, 10), "template")}
            rng = np.random.default_rng(1450 + clear + 7 * matched + course)
            draws, gaps, npl = {w: [] for w in V}, [], []
            for _ in range(500):
                pl, g = placebo_draw(rng, pl_pool, matched)
                gaps.append(g)
                npl.append(sum(len(v) for v in pl.values()))
                for w in V:
                    draws[w].append(float(np.nanmean(np.concatenate([V[w][sid][pl[sid]] for sid in pl_pool]))))
            real = {w: float(np.nanmean(np.concatenate([V[w][sid][pl_pool[sid][0]] for sid in pl_pool]))) for w in V}
            x = dict(eligible_share=float(elig), n_placebo_mean=float(np.mean(npl)), match_gap=float(np.mean(gaps)))
            for w, nm in (((3, 5), "3_5"), ((6, 10), "6_10"), ("template", "template")):
                d = np.array(draws[w])
                x[nm] = dict(real=real[w], placebo_mean=float(d.mean()), placebo_lo=float(np.percentile(d, 2.5)), placebo_hi=float(np.percentile(d, 97.5)),
                             p_real_vs_placebo=float((np.sum(d >= real[w]) + 1) / 501))
            diag["%s_%s" % (name, tag)] = x
            print("14 (unplanned) %-18s %-8s eligible %.0f%% of strokes, %.0f placebo, match gap %+.1f: 3-5 %+.1f (%+.1f to %+.1f) | 6-10 %+.1f (%+.1f to %+.1f) | template %+.1f | p(real<=placebo) %.3f/%.3f" % (
                name, tag, 100 * elig, x["n_placebo_mean"], 1000 * x["match_gap"], 1000 * x["3_5"]["placebo_mean"], 1000 * x["3_5"]["placebo_lo"], 1000 * x["3_5"]["placebo_hi"],
                1000 * x["6_10"]["placebo_mean"], 1000 * x["6_10"]["placebo_lo"], 1000 * x["6_10"]["placebo_hi"], 1000 * x["template"]["placebo_mean"],
                x["3_5"]["p_real_vs_placebo"], x["6_10"]["p_real_vs_placebo"]))
    res["eligible_share_clear10"] = float(sum(len(v[1]) for v in pool.values()) / span(pool))
    res["diagnostics_unplanned"] = diag
    out["placebo"] = res


def check15(D, out):
    # Injection at random strokes at least 25 apart; recovery = the estimate with the injection minus the estimate at
    # the same strokes without it (CODEBOOK notes of 16 Nov 2026: the first version let injections overlap).
    A = 0.030
    step = np.r_[1 - np.exp(-np.arange(1, K + 1) / 3.6)]
    truth35 = A * float(step[2:5].mean())
    res = {}
    for tag, course in (("momentum", False), ("course", True)):
        rng = np.random.default_rng(1500 + course)
        base = {sid: {w: window_stat(D.R(sid, "speed", None, course), w) for w in ((3, 5), "template")} for sid in SES}
        rec35, recT, nin = [], [], []
        for _ in range(200):
            d35, dT, n_ = [], [], 0
            for sid in SES:
                s, p = D.s[sid], D.order(sid)
                lo, hi = D.valid(sid)
                js, k = [], lo + int(rng.integers(0, 25))
                while k < hi:
                    js.append(k)
                    k += 25 + int(rng.integers(0, 11))
                js = np.array(js, int)
                n_ += len(js)
                ex = CP.course_exog(s) if course else None
                y = s["speed"].copy()
                for q in js:
                    m = min(K, len(y) - q)
                    y[q:q + m] += A * step[:m]
                R = fast_residual_matrix(y, CP.ar_fit(y, p, ex), p, ex)
                for w, acc in (((3, 5), d35), ("template", dT)):
                    st = window_stat(R, w)
                    acc.append((st[js] - np.nanmean(st[lo:hi])) - (base[sid][w][js] - np.nanmean(base[sid][w][lo:hi])))
            rec35.append(np.nanmean(np.concatenate(d35)))
            recT.append(np.nanmean(np.concatenate(dT)))
            nin.append(n_)
        res[tag] = dict(injected_plateau=A, injected_3_5=truth35, n_injected_mean=float(np.mean(nin)), recovered_3_5=float(np.mean(rec35)),
                        recovered_3_5_sd=float(np.std(rec35)), recovered_template=float(np.mean(recT)), recovered_template_sd=float(np.std(recT)),
                        share_3_5=float(np.mean(rec35) / truth35), share_template=float(np.mean(recT) / A))
        r = res[tag]
        print("15. injection and recovery (%s): %.0f strokes per draw; injected plateau %.0f (3-5 mean %.1f) mm/s; recovered 3-5 %.1f (sd %.1f) = %.0f%%; template %.1f (sd %.1f) = %.0f%%" % (
            tag, r["n_injected_mean"], 1000 * A, 1000 * truth35, 1000 * r["recovered_3_5"], 1000 * r["recovered_3_5_sd"], 100 * r["share_3_5"],
            1000 * r["recovered_template"], 1000 * r["recovered_template_sd"], 100 * r["share_template"]))
    out["injection"] = res


# ------------------------------------------------------------------ figure
def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    INK, ACC, GREY = "#1f2a36", "#b0442c", "#8c96a0"
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    m = out["meta"]["power_speed_3_5"]["global"]
    occ = m["occasions"]
    yy = np.arange(len(occ))[::-1] + 1.5
    for y_, o in zip(yy, occ):
        ax[0].plot([1000 * (o["est"] - 1.96 * o["se"]), 1000 * (o["est"] + 1.96 * o["se"])], [y_, y_], color=GREY, lw=1)
        ax[0].plot(1000 * o["est"], y_, "s", color=INK, ms=3)
    ax[0].plot([1000 * m["ci"][0], 1000 * m["ci"][1]], [0.3, 0.3], color=ACC, lw=1.6)
    ax[0].plot(1000 * m["mu"], 0.3, "D", color=ACC, ms=4.5)
    ax[0].axvline(0, color=INK, lw=0.5)
    ax[0].set_yticks(list(yy) + [0.3])
    ax[0].set_yticklabels(["%s (%d)" % (o["label"].replace(" (2nd coxswain)", "*"), o["n"]) for o in occ] + ["Random effects"], fontsize=6.3)
    ax[0].set_xlabel("power calls, strokes 3-5 (mm/s)")
    ax[0].set_title("By crew-occasion", fontsize=8.5)
    e = out["event_study"]["momentum"]
    x = np.arange(-10, 10) + 0.5          # plotted between stroke -1 and stroke 1 at the call
    c, s = 1000 * np.array(e["curve"]["est"]), 1000 * np.array(e["curve"]["sd"])
    ax[1].fill_between(x, -1.96 * s, 1.96 * s, color=GREY, alpha=0.25, lw=0)
    ax[1].plot(x, c, color=ACC, lw=1.5)
    ax[1].plot(x, 1000 * np.array(e["curve"]["est_local"]), color=INK, lw=1, ls="--")
    ax[1].axhline(0, color=INK, lw=0.5)
    ax[1].axvline(0, color=INK, lw=0.5, ls=":")
    ax[1].set_xticks([-9.5, -4.5, 0.5, 4.5, 9.5])
    ax[1].set_xticklabels(["-10", "-5", "1", "5", "10"])
    ax[1].set_xlabel("stroke relative to the call")
    ax[1].set_ylabel("speed beyond a forecast made\n10 strokes before the call (mm/s)", fontsize=7.5)
    ax[1].set_title("Before and after the call", fontsize=8.5)
    rows = sorted(out["spec_curve"]["rows"], key=lambda r: r["est"])
    est = 1000 * np.array([r["est"] for r in rows])
    sig = np.array([r["p"] < 0.05 for r in rows])
    k_ = np.arange(len(rows))
    ax[2].scatter(k_[~sig], est[~sig], s=2, color=GREY, lw=0)
    ax[2].scatter(k_[sig], est[sig], s=2, color=ACC, lw=0)
    pk = next(i for i, r in enumerate(rows) if r["paper"])
    ax[2].plot(pk, est[pk], "o", mfc="none", mec=INK, ms=7, mew=1)
    ax[2].axhline(0, color=INK, lw=0.5)
    ax[2].set_xlabel("specification (sorted)")
    ax[2].set_ylabel("power-call response (mm/s)")
    ax[2].set_title("%d specifications" % len(rows), fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(EXP, "fig_robust." + ext), dpi=200)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if "--figure" in sys.argv:              # redraw the figure from the saved results
        return figure(json.load(open(os.path.join(OUTDIR, "checks.json"))))
    if "--controls" in sys.argv:            # only the control tests of 16 Nov, added to the saved results
        D, out = Data(), json.load(open(os.path.join(OUTDIR, "checks.json")))
        check14(D, out)
        check15(D, out)
        return json.dump(out, open(os.path.join(OUTDIR, "checks.json"), "w"), indent=1)
    if "--confirm" in sys.argv:             # only the confirmation analyses of 15 Nov, added to the saved results
        D, out = Data(), json.load(open(os.path.join(OUTDIR, "checks.json")))
        check11(D, out)
        check12(D, out)
        check13(out)
        return json.dump(out, open(os.path.join(OUTDIR, "checks.json"), "w"), indent=1)
    D, out = Data(), {}
    check1(out)
    check2(D, out)
    check3(D, out)
    check4(D, out)
    check5(D, out)
    check6(D, out)
    check7(D, out)
    check8(out)
    check9(out)
    check11(D, out)
    check12(D, out)
    check13(out)
    check14(D, out)
    check15(D, out)
    json.dump(out, open(os.path.join(OUTDIR, "checks.json"), "w"), indent=1)
    figure(out)
    print("written", os.path.join(OUTDIR, "checks.json"))


if __name__ == "__main__":
    main()
