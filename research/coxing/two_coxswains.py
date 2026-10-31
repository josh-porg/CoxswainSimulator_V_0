"""What a second coxswain makes answerable: three analyses over the coded call study (both coxswains).

    .venv/Scripts/python.exe research/coxing/two_coxswains.py

1. Calling policy. What boat state comes before each kind of call, per coxswain: the boat's surprise over the three
   strokes before the call (one-step speed residual against its own AR forecast), the measured speed change before
   it (contingency: 2 strokes minus the 4 before), and the turn rate at the call. Against the same statistics at
   shifted strokes (global shift null, keeps each session's call spacing). A coxswain who follows the boat should
   show it here: praise after gains, power calls after losses, steering on bends.
2. Pushes anchored at their start. An announced push ("in two, let's take a 10") starts on the count, and counting is
   on the catch, so the push's first stroke is the stroke at the first execution word ("one", "go", "now") after the
   announcement, from the word-level transcript. The response is then read from that stroke: if announced pushes
   work, the gain should sit at strokes 1-5 from the start, and the delay from announcement to start should explain
   the strokes 8-12 response seen from the announcement.
3. Gearing. How speed scales with rate, d ln v / d ln r: (a) across 10-stroke blocks within each piece (session
   fixed effects, turn rate held), which mixes rate and effort, and (b) for rate calls alone, where the crew changes
   rate on request: the ratio of the relative speed response to the relative rate response at strokes 1-5.
   Constant force per stroke (work per stroke fixed, drag power ~ v^3) predicts 1/3; constant power predicts ~0.

Coxswain labels are neutral: A (sessions of the first coxswain), B (race_07). Data and outputs in
data/local/coxing/experiment (two_coxswains.json, two_coxswains.log).
"""
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "callmodel"))
import call_properties as CP  # noqa: E402

EXP = os.path.join(ROOT, "data", "local", "coxing", "experiment")
warnings.filterwarnings("ignore", category=RuntimeWarning)          # race_01 has no track: empty turn means
SDIR = os.path.join(ROOT, "data", "local", "coxing", "sessions")
COX_B = ["race_07"]
NPERM = 2000
ANN = re.compile(r"in two|take a\b|take (?:\d+|five|ten|twenty|three)|\bready\b|give me (?:\d+|five|ten)|last 10|last ten", re.I)
EXEC_WORDS = {"one", "go", "now"}
LOG = os.path.join(EXP, "two_coxswains.log")
OUT = {}


def log(*a):
    print(*a)
    with open(LOG, "a", encoding="utf-8") as f:
        print(*a, file=f)


def load(sid):
    s = CP.prepare(CP.load_session(os.path.join(SDIR, sid)))
    s["id"] = sid
    n = len(s["t"])
    s["turn_abs"] = np.abs(s["course"]["turn"]) if "course" in s else np.full(n, np.nan)
    wp = os.path.join(SDIR, sid, "words.jsonl")
    s["words"] = [json.loads(x) for x in open(wp, encoding="utf-8")] if os.path.exists(wp) else []
    return s


def by_session(df, sids):
    return [df.loc[df.session == k, "j"].to_numpy() for k in sids]


# ------------------------------------------------------------------ 1. calling policy
def pre_state(S):
    def stat(jj):
        surprise, cont, turn, ahead = [], [], [], []
        for s, js in zip(S, jj):
            r1 = s["R"]["speed"][:, 0]
            for j in js:
                surprise.append(np.nanmean(r1[max(j - 3, 0):j]))
                cont.append(CP.contingency(s, j))
                turn.append(s["turn_abs"][j])
                ahead.append(np.nanmean(s["turn_abs"][j + 2:j + 9]))
        return np.array([np.nanmean(surprise), np.nanmean(cont), np.nanmean(turn), np.nanmean(ahead)])
    return stat


CLASSES = {
    "power (directive)": lambda d: (d.target == "power") & (d.stage == "dec"),
    "praise (challenge assessment)": lambda d: (d.stage == "ass") & (d.appraisal == "chal"),
    "steering": lambda d: d.target == "steer",
    "technique (directive)": lambda d: d.target.isin(["catch", "finish", "body", "set", "together"]) & (d.stage == "dec"),
    "rate (directive)": lambda d: (d.target == "rate") & (d.stage == "dec"),
    "arousal raised": lambda d: d.arousal == "up",
    "information": lambda d: d.target == "info",
}


def policy(calls, sess):
    res = {}
    log("\n1. CALLING POLICY: boat state before the call, minus the same at shifted strokes")
    log("   surprise = one-step speed residual, mean of the 3 strokes before (mm/s); contingency = speed change before"
        " (mm/s); turn = |turn rate| at the call (deg/s)")
    for cox, sids in (("A", [k for k in sess if k not in COX_B]), ("B", COX_B)):
        S = [sess[k] for k in sids]
        res[cox] = {}
        for lab, f in CLASSES.items():
            d = calls[calls.session.isin(sids) & f(calls)]
            if len(d) < 5:
                continue
            og, mg, sg, pg = CP.permutation_test(pre_state(S), S, by_session(d, sids), NPERM, 5)
            e = og - mg
            res[cox][lab] = dict(n=len(d), surprise=float(e[0]), surprise_sd=float(sg[0]), surprise_p=float(pg[0]),
                                 contingency=float(e[1]), contingency_sd=float(sg[1]), contingency_p=float(pg[1]),
                                 turn=float(e[2]), turn_sd=float(sg[2]), turn_p=float(pg[2]),
                                 turn_ahead=float(e[3]), turn_ahead_sd=float(sg[3]), turn_ahead_p=float(pg[3]))
            log("   %s %-30s n=%3d  surprise %+4.0f (sd %2.0f, p %.3f)  contingency %+4.0f (sd %2.0f, p %.3f)"
                "  turn %+5.2f (p %.3f)  turn ahead %+5.2f (p %.3f)" % (
                    cox, lab, len(d), 1000 * e[0], 1000 * sg[0], pg[0], 1000 * e[1], 1000 * sg[1], pg[1], e[2], pg[2],
                    e[3], pg[3]))
    return res


# ------------------------------------------------------------------ 2. pushes anchored at their start
def execution_word(s, t0, t_end, horizon=15.0):
    """First execution word strictly after the announcement phrase starts (within ``horizon`` s); None if none."""
    for w in s["words"]:
        if t0 + 0.3 < w["t"] <= t0 + horizon:
            tok = re.sub(r"[^a-z]", "", w["w"].lower())
            if tok in EXEC_WORDS:
                return w["t"]
    return None


def pushes(calls, sess):
    log("\n2. ANNOUNCED PUSHES, RESPONSE FROM THE PUSH'S FIRST STROKE (the first 'one'/'go'/'now', counting on the catch)")
    rows = []
    for sid, s in sess.items():
        if not s["words"]:
            continue
        pw = calls[(calls.session == sid) & (calls.target == "power") & (calls.stage == "dec") & calls.text.str.contains(ANN)]
        lo, hi = s["valid"]
        for _, r in pw.iterrows():
            te = execution_word(s, r.t, r.t)
            if te is None:
                rows.append(dict(session=sid, j_ann=int(r.j), j_exec=-1, delay=np.nan))
                continue
            je = int(CP.stroke_index(s["t"], [te])[0])
            rows.append(dict(session=sid, j_ann=int(r.j), j_exec=je if lo <= je < hi else -1, delay=je - int(r.j)))
    P = pd.DataFrame(rows)
    found = P[P.j_exec >= 0]
    log("   %d announcements; %d with an execution word inside the valid strokes; delay announcement -> start: median %.0f"
        " strokes (range %d-%d)" % (len(P), len(found), found.delay.median(), found.delay.min(), found.delay.max()))
    res = dict(n_announce=len(P), n_found=len(found), delay_median=float(found.delay.median()),
               delay_counts={int(k): int(v) for k, v in found.delay.value_counts().sort_index().items()})
    wins = [(1, 3), (4, 6), (7, 10)]
    for lab, sel in (("all", found), ("A", found[~found.session.isin(COX_B)]), ("B", found[found.session.isin(COX_B)])):
        if len(sel) < 4:
            continue
        sids = sorted(sel.session.unique())
        S = [sess[k] for k in sids]
        for anchor in ("j_exec", "j_ann"):
            jj = [sel.loc[sel.session == k, anchor].to_numpy() for k in sids]
            st = lambda q: np.array([np.nanmean(CP.gather(S, q, "speed", a, b)) for a, b in wins] +
                                    [np.nanmean(CP.gather(S, q, "rate", 1, 5))])
            og, mg, sg, pg = CP.permutation_test(st, S, jj, NPERM, 7)
            e = og - mg
            curve = [1000 * np.nanmean(CP.gather(S, jj, "speed", h, h)) for h in range(1, 13)]
            res["%s_%s" % (lab, anchor)] = dict(n=len(sel), windows=[list(w) for w in wins], speed=(1000 * e[:3]).tolist(),
                                                speed_sd=(1000 * sg[:3]).tolist(), speed_p=pg[:3].tolist(),
                                                rate_1_5=float(e[3]), rate_sd=float(sg[3]), rate_p=float(pg[3]), curve=curve)
            log("   %-3s from %-13s n=%2d  speed 1-3 %+4.0f (sd %3.0f, p %.2f)  4-6 %+4.0f (sd %3.0f, p %.2f)  7-10 %+4.0f"
                " (sd %3.0f, p %.2f)  rate 1-5 %+.2f spm (p %.2f)" % (
                    lab, "start" if anchor == "j_exec" else "announcement", len(sel), 1000 * e[0], 1000 * sg[0], pg[0],
                    1000 * e[1], 1000 * sg[1], pg[1], 1000 * e[2], 1000 * sg[2], pg[2], e[3], pg[3]))
            log("        curve mm/s, strokes 1-12: " + " ".join("%+.0f" % c for c in curve))
    return res


# ------------------------------------------------------------------ 3. gearing
def gearing(calls, sess):
    log("\n3. GEARING: d ln(speed) / d ln(rate)")
    res = {}
    blocks = []
    for sid, s in sess.items():
        n = len(s["t"])
        lo, hi = 10, n - 10                       # drop the start and the last strokes of the record
        for b0 in range(lo, hi - 10, 10):
            sl = slice(b0, b0 + 10)
            blocks.append(dict(session=sid, cox="B" if sid in COX_B else "A", lv=np.log(s["speed"][sl].mean()),
                               lr=np.log(s["rate"][sl].mean()), turn=np.nanmean(s["turn_abs"][sl]),
                               frac=(b0 + 5) / n))
    B = pd.DataFrame(blocks)
    for lab, d in (("all", B), ("A", B[B.cox == "A"]), ("B", B[B.cox == "B"])):
        d = d.copy()
        d["turn"] = d.turn.fillna(d.turn.mean())
        for c in ("lv", "lr", "turn", "frac"):    # within-session (fixed effects)
            d[c + "_w"] = d[c] - d.groupby("session")[c].transform("mean")
        X = np.column_stack([d.lr_w, d.turn_w, d.frac_w])
        b, *_ = np.linalg.lstsq(X, d.lv_w, rcond=None)
        rng = np.random.default_rng(3)
        boot = []
        sids = d.session.unique()
        for _ in range(2000):                     # resample pieces; one piece: resample runs of 3 blocks inside it
            if len(sids) >= 3:
                dd = pd.concat([d[d.session == k] for k in rng.choice(sids, len(sids))])
            else:
                starts = rng.integers(0, len(d) - 2, len(d) // 3)
                dd = d.iloc[np.concatenate([np.arange(s0, s0 + 3) for s0 in starts])]
            bb, *_ = np.linalg.lstsq(np.column_stack([dd.lr_w, dd.turn_w, dd.frac_w]), dd.lv_w, rcond=None)
            boot.append(bb[0])
        lo_, hi_ = np.percentile(boot, [2.5, 97.5])
        res["blocks_" + lab] = dict(n_blocks=len(d), n_pieces=len(sids), elasticity=float(b[0]), ci=[float(lo_), float(hi_)])
        log("   blocks %-3s %3d blocks in %2d pieces: elasticity %.2f (95%% CI %.2f to %.2f, resampled)" % (
            lab, len(d), len(sids), b[0], lo_, hi_))
    # rate calls: relative speed response over relative rate response, strokes 1-5
    for lab, sids in (("all", list(sess)), ("A", [k for k in sess if k not in COX_B]), ("B", COX_B)):
        S = [sess[k] for k in sids]
        d = calls[calls.session.isin(sids) & (calls.target == "rate") & (calls.stage == "dec")]
        if len(d) < 5:
            continue

        def st(jj):
            dv = np.concatenate([CP.gather([s], [q], "speed", 1, 5) / s["speed"].mean() for s, q in zip(S, jj)])
            dr = np.concatenate([CP.gather([s], [q], "rate", 1, 5) / s["rate"].mean() for s, q in zip(S, jj)])
            return np.array([np.nanmean(dv), np.nanmean(dr)])
        og, mg, sg, pg = CP.permutation_test(st, S, by_session(d, sids), NPERM, 9)
        e = og - mg
        # elasticity with a delta-method band from the null spreads (ratio of two shifted means)
        el = e[0] / e[1] if abs(e[1]) > 0 else np.nan
        el_sd = abs(el) * np.sqrt((sg[0] / e[0]) ** 2 + (sg[1] / e[1]) ** 2) if e[0] != 0 else np.nan
        res["rate_calls_" + lab] = dict(n=len(d), dv_rel=float(e[0]), dv_sd=float(sg[0]), dv_p=float(pg[0]),
                                        dr_rel=float(e[1]), dr_sd=float(sg[1]), dr_p=float(pg[1]),
                                        elasticity=float(el), elasticity_sd=float(el_sd))
        log("   rate calls %-3s n=%3d: rate %+.2f%% (p %.3f), speed %+.2f%% (p %.3f) -> elasticity %.2f (sd ~%.2f)" % (
            lab, len(d), 100 * e[1], pg[1], 100 * e[0], pg[0], el, el_sd))
    log("   reference: constant work per stroke (drag power ~ v^3) 0.33; constant power ~0")
    return res


def main():
    open(LOG, "w").close()
    D = pd.read_csv(os.path.join(EXP, "episodes.csv"))
    calls = D[D.target != "count"].reset_index(drop=True)
    sids = list(dict.fromkeys(D.session))
    sess = {k: load(k) for k in sids}
    log("sessions: %s; calls %d (A %d, B %d)" % (", ".join(sids), len(calls), (~calls.session.isin(COX_B)).sum(),
                                                 calls.session.isin(COX_B).sum()))
    OUT["policy"] = policy(calls, sess)
    OUT["pushes"] = pushes(calls, sess)
    OUT["gearing"] = gearing(calls, sess)
    json.dump(OUT, open(os.path.join(EXP, "two_coxswains.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
