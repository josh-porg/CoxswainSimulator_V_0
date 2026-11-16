"""The sound of the boat and the sound of the voice: the tests (CODEBOOK addendum of 15 Nov 2026; exploratory).

    .venv/Scripts/python.exe research/coxing/acoustic/tests.py

A1 does the background sound track the boat?   V1 does the voice alone tell the kind of call?
A2 what precedes a call?                       V2 does the voice follow the boat's state?
A3 does a call change the sound?               V3 does the voice predict the boat's response?

Reads acoustic/strokes.csv and phrases.csv (features.py), the sessions and the study's episodes.csv; writes
acoustic/results.json. Shift nulls as in the study; Benjamini-Hochberg within each family under each null.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, "..", "experiment"))
import examiner_checks as X  # noqa: E402  (shift permutations, BH)
import features as FT  # noqa: E402
CP = X.CP

LOCAL = os.path.abspath(os.path.join(CODE, "..", "..", "..", "data", "local", "coxing"))
EXP, OUT = os.path.join(LOCAL, "experiment"), os.path.join(LOCAL, "acoustic")
K, NPERM = CP.K_POST, 2000
BG, VOICE = FT.BG, FT.VOICE
TECH = {"catch", "finish", "body", "set", "together"}
CLASSES = {"power": lambda d: (d.target == "power") & (d.stage == "dec"), "rate": lambda d: (d.target == "rate") & (d.stage == "dec"),
           "length": lambda d: (d.target == "length") & (d.stage == "dec"), "catch": lambda d: d.target == "catch",
           "timing": lambda d: d.target == "together", "set": lambda d: d.target == "set", "steering": lambda d: d.target == "steer",
           "praise": lambda d: (d.stage == "ass") & (d.appraisal == "chal")}
EXPECT = {("catch", "spread"): "+", ("timing", "spread"): "+", ("catch", "sharp"): "-", ("timing", "sharp"): "-", ("power", "thump"): "-"}
EXPECT3 = {("catch", "spread"): "-", ("timing", "spread"): "-", ("catch", "sharp"): "+", ("timing", "sharp"): "+", ("power", "thump"): "+"}


def zs(v):
    return (v - np.nanmean(v)) / (np.nanstd(v) or 1.0)


def load():
    qc = json.load(open(os.path.join(OUT, "qc.json")))
    st = pd.read_csv(os.path.join(OUT, "strokes.csv"))
    ph = pd.read_csv(os.path.join(OUT, "phrases.csv"))
    S = {}
    for sid in FT.SES:
        s = CP.prepare(CP.load_session(os.path.join(LOCAL, "sessions", sid)))
        g = st[st.session == sid].sort_values("k")
        sp = g.speech.fillna(0).to_numpy()
        F = {}
        for f in BG:
            v = g[f + "_z"].to_numpy(float)
            if f in ("thump", "sharp", "rumble"):                    # leakage rule: residual on the stroke's speech share
                ok = ~np.isnan(v)
                A = np.column_stack([np.ones(ok.sum()), sp[ok]])
                v = v.copy()
                v[ok] = zs(v[ok] - A @ np.linalg.lstsq(A, v[ok], rcond=None)[0])
            F[f] = v
        s.update(F=F, n=len(s["t"]), locked=qc[sid]["locked"], rng=(13, len(s["t"]) - K))
        S[sid] = s
    ep = pd.read_csv(os.path.join(EXP, "episodes.csv"))
    ep = ep[ep.session.isin(FT.SES) & (ep.target != "count")].reset_index(drop=True)
    bl = pd.read_csv(os.path.join(EXP, "episodes_blindarousal_all.csv"))[["session", "i", "arousal"]].rename(columns={"arousal": "arousal_blind"})
    ph = ph[["session", "i"] + [f + "_z" for f in VOICE]].rename(columns={f + "_z": "v_" + f for f in VOICE})      # v_: not the study's own loud_z, f0_z
    ep = ep.merge(bl, on=["session", "i"], how="left").merge(ph, on=["session", "i"], how="left")
    return S, ep, qc


def js_of(ep, sids, mask, S):
    out = []
    for sid in sids:
        lo, hi = S[sid]["rng"]
        j = ep.loc[(ep.session == sid) & mask, "j"].to_numpy()
        out.append(j[(j >= lo) & (j < hi)])
    return out


def family(rows):
    for key, q in (("p", "q"), ("p_local", "q_local")):
        for r, v in zip(rows, X.bh([r[key] for r in rows])):
            r[q] = float(v)
    return rows


def a1(S, sids, out):
    rng = np.random.default_rng(11)

    def sse(sid, shift):
        s = S[sid]
        y, n, p = s["speed"], s["n"], s["ar_order"]
        G = np.column_stack([np.nan_to_num(np.roll(s["F"][f], shift)) for f in BG])
        t = np.arange(max(p, 2), n)
        base = np.column_stack([np.ones(len(t))] + [y[t - l - 1] for l in range(p)])
        full = np.column_stack([base] + [G[t - l] for l in range(3)])
        edges = np.linspace(0, len(t), 6).astype(int)
        eb = ef = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            tr = np.r_[0:a, b:len(t)]
            for Xm, acc in ((base, "b"), (full, "f")):
                beta = np.linalg.lstsq(Xm[tr], y[t][tr], rcond=None)[0]
                e = float(np.sum((y[t][a:b] - Xm[a:b] @ beta) ** 2))
                if acc == "b":
                    eb += e
                else:
                    ef += e
        return eb, ef

    def gain(shifts):
        tot = np.sum([sse(sid, sh) for sid, sh in zip(sids, shifts)], axis=0)
        return 1 - tot[1] / tot[0]

    obs = gain([0] * len(sids))
    null = np.array([gain([int(rng.integers(30, S[sid]["n"] - 30)) for sid in sids]) for _ in range(500)])
    res = dict(oos_gain=float(obs), null_mean=float(null.mean()), null_sd=float(null.std()), p=float((np.sum(null >= obs) + 1) / 501))
    print("A1 out-of-sample gain in speed forecast from the background sound: %.4f (null %.4f, sd %.4f), p %.3f" % (obs, null.mean(), null.std(), res["p"]))
    # partial correlations with the one-step speed error and with stroke rate
    tgt = {sid: (zs(S[sid]["R"]["speed"][:, 0]), zs(S[sid]["rate"])) for sid in sids}

    def corrs(local):
        r = np.random.default_rng(12 + local)

        def stat(shifts):
            o = []
            for q in range(2):
                for f in BG:
                    a = np.concatenate([np.roll(S[sid]["F"][f], sh) for sid, sh in zip(sids, shifts)])
                    b = np.concatenate([tgt[sid][q] for sid in sids])
                    ok = ~np.isnan(a) & ~np.isnan(b)
                    o.append(np.corrcoef(a[ok], b[ok])[0, 1])
            return np.array(o)
        obs_ = stat([0] * len(sids))
        nl = np.array([stat([(int(r.integers(10, 41)) * (1 if r.random() < 0.5 else -1)) if local else int(r.integers(30, S[sid]["n"] - 30)) for sid in sids])
                       for _ in range(NPERM)])
        return obs_, (np.sum(np.abs(nl - nl.mean(0)) >= np.abs(obs_ - nl.mean(0)), axis=0) + 1) / (NPERM + 1)

    (og, pg), (ol, pl) = corrs(False), corrs(True)
    rows = family([dict(feature=f, with_=w, r=float(og[q * 6 + i]), p=float(pg[q * 6 + i]), p_local=float(pl[q * 6 + i]))
                   for q, w in enumerate(("speed error", "stroke rate")) for i, f in enumerate(BG)])
    for r in rows:
        print("   %-7s with %-11s r %+.3f p %.4f (q %.3f) | local p %.4f (q %.3f)" % (r["feature"], r["with_"], r["r"], r["p"], r["q"], r["p_local"], r["q_local"]))
    res["correlations"] = rows
    out["A1"] = res


def a23(S, ep, sids, out):
    RNG = [S[sid]["rng"] for sid in sids]
    pre = {f: [np.array([np.nan] * 3 + [np.nanmean(S[sid]["F"][f][j - 3:j]) if not np.all(np.isnan(S[sid]["F"][f][j - 3:j])) else np.nan
                                        for j in range(3, S[sid]["n"])]) for sid in sids] for f in BG}
    post = {}
    for f in BG:
        post[f] = []
        for sid in sids:
            v, n = S[sid]["F"][f], S[sid]["n"]
            if f in ("thump", "sharp", "rumble"):
                y = np.nan_to_num(v)
                p = CP.bic_order(y)
                post[f].append(np.nanmean(CP.residual_matrix(y, CP.ar_fit(y, p), K, p)[:, :5], axis=1))
            else:
                d = np.full(n, np.nan)
                for j in range(5, n - 5):
                    a, b = v[j:j + 5], v[j - 5:j]
                    if not (np.all(np.isnan(a)) or np.all(np.isnan(b))):
                        d[j] = np.nanmean(a) - np.nanmean(b)
                post[f].append(d)
    for name, V, exp in (("A2", pre, EXPECT), ("A3", post, EXPECT3)):
        rows = []
        for q, (cls, mk) in enumerate(CLASSES.items()):
            JS = js_of(ep, sids, mk(ep), S)
            for i, f in enumerate(BG):
                r = X.both(V[f], JS, RNG, 2000 + 100 * q + i + (5000 if name == "A3" else 0))
                rows.append(dict(cls=cls, feature=f, expected=exp.get((cls, f), ""), **r))
        family(rows)
        out[name] = rows
        hits = [r for r in rows if r["q"] < 0.05 and r["q_local"] < 0.05]
        print("%s: %d tests; q<0.05 under both nulls: %s" % (name, len(rows), [(r["cls"], r["feature"], round(r["est"], 2)) for r in hits] or "none"))
        for r in rows:
            if r["expected"] or min(r["p"], r["p_local"]) < 0.01:
                print("   %-8s %-7s n=%3d %+.3f sd (null sd %.3f) p %.4f q %.3f | local %+.3f p %.4f q %.3f  %s" % (
                    r["cls"], r["feature"], r["n"], r["est"], r["sd"], r["p"], r["q"], r["est_local"], r["p_local"], r["q_local"],
                    ("expected " + r["expected"]) if r["expected"] else ""))


def v1(ep, out, feats=None, key="V1"):
    """feats=None: the ten planned voice features. The unplanned follow-up passes the six that do not depend on how
    many words the phrase has (no speech rate, duration or onset phase)."""
    feats = feats or VOICE
    d = ep.dropna(subset=["v_level"]).reset_index(drop=True)
    Xv = np.nan_to_num(d[["v_" + f for f in feats]].to_numpy(float))
    grp = d.target.map(lambda t: "technique" if t in TECH else ("rate_length" if t in ("rate", "length") else t))
    tasks = {"target group": grp.to_numpy(), "directive": (d.stage == "dec").to_numpy().astype(int),
             "blind arousal raised": (d.arousal_blind == "up").to_numpy().astype(int)}
    pieces = d.session.to_numpy()
    rng = np.random.default_rng(21)
    res = {}

    def score(y):
        pred = np.empty(len(y), dtype=object)
        for sid in np.unique(pieces):
            te = pieces == sid
            m = LogisticRegression(max_iter=300, class_weight="balanced").fit(Xv[~te], y[~te])
            pred[te] = m.predict(Xv[te])
        return balanced_accuracy_score(y.astype(str), pred.astype(str))

    for name, y in tasks.items():
        obs = score(y)
        null = []
        for _ in range(1000):
            yp = y.copy()
            for sid in np.unique(pieces):
                m = pieces == sid
                yp[m] = rng.permutation(yp[m])
            null.append(score(yp))
        null = np.array(null)
        res[name] = dict(n=int(len(y)), classes=int(len(np.unique(y))), balanced_accuracy=float(obs), null_mean=float(null.mean()), null_sd=float(null.std()),
                         p=float((np.sum(null >= obs) + 1) / 1001))
        print("%s %-22s n=%d, %d classes: balanced accuracy %.3f (null %.3f, sd %.3f) p %.4f" % (key, name, len(y), res[name]["classes"], obs, null.mean(), null.std(), res[name]["p"]))
    # which features differ (descriptive): mean z by directive and by blind arousal
    res["means"] = {name: {str(c): {f: float(np.nanmean(d.loc[y == c, "v_" + f])) for f in VOICE} for c in np.unique(y)}
                    for name, y in tasks.items() if name != "target group"}
    res["means"]["target group"] = {str(c): {f: float(np.nanmean(d.loc[grp == c, "v_" + f])) for f in VOICE} for c in sorted(set(grp))}
    out[key] = res


def v23(S, ep, out):
    sids = list(FT.SES)
    d = ep.dropna(subset=["v_level"]).reset_index(drop=True)
    d = d[[S[s_]["rng"][0] <= j < S[s_]["rng"][1] for s_, j in zip(d.session, d.j)]].reset_index(drop=True)
    Y = np.nan_to_num(d[["v_" + f for f in VOICE]].to_numpy(float))
    fe = np.column_stack([(d.session == s_).to_numpy(float) for s_ in sids])
    state = {}
    for s_ in sids:
        s = S[s_]
        state[s_] = np.column_stack([[CP.contingency(s, j) if j >= 6 else np.nan for j in range(s["n"])],
                                     np.r_[np.nan, s["rate"][:-1]] / np.median(s["rate"]) - 1, np.arange(s["n"]) / s["n"] - 0.5])
    JS = [d.loc[d.session == s_, "j"].to_numpy() for s_ in sids]
    order = np.concatenate([np.flatnonzero((d.session == s_).to_numpy()) for s_ in sids])

    def coefs(jj):
        St = np.full((len(d), 3), 0.0)
        St[order] = np.nan_to_num(np.vstack([state[s_][j] for s_, j in zip(sids, jj)]))
        St = St / (St.std(0) + 1e-12)
        B = np.linalg.lstsq(np.column_stack([fe, St]), Y, rcond=None)[0]
        return B[-3:].ravel()                                       # 3 states x 10 features

    res = []
    obs = coefs(JS)
    P = {}
    for null, local in (("global", False), ("local", True)):
        r = np.random.default_rng(31 + local)
        nl = np.array([coefs([X.shift(r, j, *S[s_]["rng"], local) for s_, j in zip(sids, JS)]) for _ in range(NPERM)])
        P[null] = ((np.sum(np.abs(nl - nl.mean(0)) >= np.abs(obs - nl.mean(0)), axis=0) + 1) / (NPERM + 1), obs - nl.mean(0))
    for a, st in enumerate(("speed change before", "stroke rate", "time in piece")):
        for b, f in enumerate(VOICE):
            q = a * len(VOICE) + b
            res.append(dict(state=st, feature=f, est=float(P["global"][1][q]), p=float(P["global"][0][q]), est_local=float(P["local"][1][q]), p_local=float(P["local"][0][q])))
    family(res)
    out["V2"] = res
    hits = [r for r in res if r["q"] < 0.05 and r["q_local"] < 0.05]
    print("V2: %d tests (n=%d calls); q<0.05 under both nulls: %s" % (len(res), len(d), [(r["state"], r["feature"], round(r["est"], 3)) for r in hits] or "none"))
    for r in res:
        if r["q"] < 0.05 or r["q_local"] < 0.05:
            print("   %-20s %-9s %+.3f sd per sd, p %.4f q %.3f | local %+.3f p %.4f q %.3f" % (r["state"], r["feature"], r["est"], r["p"], r["q"], r["est_local"], r["p_local"], r["q_local"]))
    # V3: the ten voice features in the joint model of the 3-5-stroke speed response
    cols = [fe, (d.time_frac - 0.5).to_numpy()[:, None]]
    for tg in ["power", "rate", "length", "catch", "finish", "body", "set", "together", "steer", "morale", "tactic"]:
        cols.append((d.target == tg).to_numpy(float)[:, None])
    for c, vals in (("stage", ["ass", "dec"]), ("level", ["proc", "sreg", "self"]), ("appraisal", ["chal", "thr"]), ("form", ["analog", "expl"]), ("arousal", ["up", "down"])):
        for v in vals:
            cols.append((d[c] == v).to_numpy(float)[:, None])
    cols += [(d.action == "y").to_numpy(float)[:, None], d.focus.isin(["int", "mix"]).to_numpy(float)[:, None], d.focus.isin(["ext", "mix"]).to_numpy(float)[:, None],
             (d.ref == "y").to_numpy(float)[:, None], (d.discrep == "y").to_numpy(float)[:, None], d.established.to_numpy(float)[:, None],
             (np.log(d.words) - np.log(d.words).mean()).to_numpy()[:, None]]
    Bm = np.column_stack(cols)
    Fm = np.column_stack([Bm, Y])
    V35 = [np.nanmean(S[s_]["R"]["speed"][:, 2:5], axis=1) for s_ in sids]

    def gain(jj):
        y = np.empty(len(d))
        y[order] = np.concatenate([v[j] for v, j in zip(V35, jj)])
        ok = ~np.isnan(y)
        sst = np.sum((y[ok] - y[ok].mean()) ** 2)
        e = [np.sum((y[ok] - M[ok] @ np.linalg.lstsq(M[ok], y[ok], rcond=None)[0]) ** 2) for M in (Bm, Fm)]
        return (e[0] - e[1]) / sst

    obs = gain(JS)
    v3 = dict(n=int(len(d)), r2_gain=float(obs))
    for null, local in (("global", False), ("local", True)):
        r = np.random.default_rng(41 + local)
        nl = np.array([gain([X.shift(r, j, *S[s_]["rng"], local) for s_, j in zip(sids, JS)]) for _ in range(NPERM)])
        v3["null_mean_" + null], v3["p_" + null] = float(nl.mean()), float((np.sum(nl >= obs) + 1) / (NPERM + 1))
    out["V3"] = v3
    print("V3 voice features in the joint model: R2 gain %.4f (null %.4f) p %.3f | local null %.4f p %.3f" % (
        obs, v3["null_mean_global"], v3["p_global"], v3["null_mean_local"], v3["p_local"]))


def main():
    S, ep, qc = load()
    if "--v1b" in sys.argv:                 # unplanned follow-up: pitch, level and voice quality only
        out = json.load(open(os.path.join(OUT, "results.json")))
        v1(ep, out, ["level", "f0", "f0_range", "f0_slope", "tilt", "voiced"], "V1_sound_only_unplanned")
        return json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    locked = [s_ for s_ in FT.SES if S[s_]["locked"]]
    out = dict(pieces_background=locked, pieces_voice=list(FT.SES), n_calls=int(len(ep)), qc=qc)
    print("background analyses on %d locked pieces; voice on %d; %d calls" % (len(locked), len(FT.SES), len(ep)))
    a1(S, locked, out)
    a23(S, ep, locked, out)
    v23(S, ep, out)
    v1(ep, out)
    json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("written", os.path.join(OUT, "results.json"))


if __name__ == "__main__":
    main()
