"""Stage 1 (observational replication) and Stage 2 (control) of the review's testing plan, on the
five synchronised pieces: which theory-derived properties of a call predict the boat's response.

    .venv/Scripts/python.exe research/coxing/experiment/run_experiment.py

Codes come from codes_<session>.txt (transcript only, CODEBOOK.md). Everything relational is
computed here from the boat record and the audio. Inference: shift of each session's episode
strokes (keeps call spacing and boat structure, breaks alignment), 2000 shifts; global (anywhere
in the piece) and local (10-40 strokes, keeps place in the piece). Relational properties are
recomputed at the shifted strokes.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import wavfile

ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "research", "callmodel"))
import call_properties as CP  # noqa: E402

CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "experiment"))   # its data and outputs (gitignored)
SDIR = os.path.join(ROOT, "data", "local", "coxing", "sessions")
TODAY = ["practice_03", "practice_04", "practice_05", "practice_06", "practice_07", "practice_08"]
# race_06: Head of the Charles, 17 Oct 2026 (added 21 Oct 2026; coded under the same codebook)
SES = ["race_01", "practice_01", "practice_02", "race_04", "race_05"] + TODAY + ["race_06"]
ORDER = ["race_01", "practice_01", "practice_02", "control_01", "race_04", "race_05", "control_02"] + TODAY + ["race_06"]
AUDIO = {"practice_01": "0117", "practice_02": "0119", "race_04": "0132", "race_05": "0134", **{k: "261007" for k in TODAY},
         "race_06": "0149"}
# COURSE=1: the boat's own forecast also carries the course (turn rate; heading where the piece turns),
# a robustness run written with the suffix _course. Without it the analysis is the pre-registered one.
COURSE = os.environ.get("COURSE") == "1"
# ALT_AROUSAL=race_06|all: arousal from the lexical rule coder (rulecoder.py) instead of the hand codes,
# for those sessions; a sensitivity check on the arousal contrast (race_06 was coded after the first results).
ALT_AROUSAL = os.environ.get("ALT_AROUSAL", "")
SUF = ("_course" if COURSE else "") + ("_rulearousal_" + ALT_AROUSAL if ALT_AROUSAL else "")
# drilled cues (CODEBOOK addendum): regex on the lead phrase, piece number from which the drill had been taught
import re
sys.path.insert(0, CODE)
DRILLS = [(r"connect|stand on it|feet? (connection|pressure)|pressure on (the|your) (feet|foot|stretcher)", 1),
          (r"hang|suspend|rushing catch", 2), (r"send (the|that) puddle|complete (the stroke|all the way)", 3),
          (r"^legs|sequenc|legs,? body", 4)]
WAVS = {}
NPERM = 2000
OUT = {}
LOG = os.path.join(HERE, "run_experiment%s.log" % SUF)
open(LOG, "w").close()


def log(*a):
    print(*a)
    with open(LOG, "a", encoding="utf-8") as f:
        print(*a, file=f)


# ------------------------------------------------------------------ build
sessions = {}
for sid in ORDER:
    s = CP.load_session(os.path.join(SDIR, sid))
    s["id"] = sid
    s["episodes"] = []
    if sid in SES:
        CP.prepare(s, course=COURSE)
        codes = CP.load_codes(os.path.join(HERE, "codes_%s.txt" % sid))
        if ALT_AROUSAL in (sid, "all"):
            import rulecoder
            for p in s["phrases"]:
                if p["i"] in codes and codes[p["i"]]["target"] not in ("other", "count"):
                    codes[p["i"]]["arousal"] = rulecoder.code(p["text"])["arousal"]
        s["episodes"] = CP.episodes(s["phrases"], codes)
    sessions[sid] = s
CP.add_familiarity([sessions[k] for k in ORDER])
S = [sessions[k] for k in SES]

rows = []
for sid in SES:
    s = sessions[sid]
    n = len(s["t"])
    lo, hi = s["valid"]
    s["med_rate"] = np.median(s["rate"])
    wav = None
    if sid in AUDIO:
        if AUDIO[sid] not in WAVS:
            WAVS[AUDIO[sid]] = wavfile.read(os.path.join(HERE, "audio", AUDIO[sid] + ".wav"))
        sr, wav = WAVS[AUDIO[sid]]
        off = s["meta"]["sync_offset_s"]
    last_by_target = {}
    for e in s["episodes"]:
        j = int(CP.stroke_index(s["t"], [e["t"]])[0])
        interval = j - last_by_target.get(e["target"], -10 ** 6)
        last_by_target[e["target"]] = j
        if not lo <= j < hi:
            continue
        r = dict(session=sid, kind=s["meta"]["kind"], boat=s["meta"]["boat_class"], i=e["i"], t=e["t"], j=j,
                 text=" / ".join(e["texts"])[:90], words=e["words"], clauses=e["clauses"],
                 established=int(e.get("established", False)), time_frac=j / n,
                 intensity=s["rate"][j - 1] / s["med_rate"], contingency=CP.contingency(s, j), interval=interval)
        for f in CP.FIELDS:
            r[f] = e[f]
        r["drilled"] = 0
        if sid in TODAY:
            piece = TODAY.index(sid) + 1
            lead = e["texts"][0].lower()
            r["drilled"] = int(any(re.search(rx, lead) and piece >= p0 for rx, p0 in DRILLS))
            r["hold"] = int(bool(re.search(r"hold|keep it like this|maintain|the change", " ".join(e["texts"]).lower())))
        if wav is not None:
            p0 = next(p for p in s["phrases"] if p["i"] == e["i"])
            te = max(p0["t"], p0.get("t_end") or p0["t"])
            r["loud"], r["f0"], r["srate"] = CP.delivery(wav, sr, p0["t"] + off, te + off, len(p0["text"].split()))
        rows.append(r)
D = pd.DataFrame(rows)
D["sorder"] = D.session.map(SES.index)
D = D.sort_values(["sorder", "t"]).reset_index(drop=True)
for c in ("loud", "f0"):
    D[c + "_z"] = D.groupby("session")[c].transform(lambda x: (x - x.mean()) / x.std())
D.to_csv(os.path.join(HERE, "episodes%s.csv" % SUF), index=False)
calls = D[D.target != "count"].reset_index(drop=True)
counts = D[D.target == "count"].reset_index(drop=True)
log("episodes: %d calls and %d count runs; per piece %s" % (len(calls), len(counts), ", ".join(
    "%s %d" % (k, (calls.session == k).sum()) for k in SES)))
OUT["n"] = dict(calls=len(calls), counts=len(counts), per_session={k: int((calls.session == k).sum()) for k in SES})
log("property shares among calls:")
shares = {}
for f in CP.FIELDS:
    vc = calls[f].value_counts(normalize=True)
    shares[f] = {k: float(v) for k, v in vc.items()}
    log("  %-9s %s" % (f, "  ".join("%s %.0f%%" % (k, 100 * v) for k, v in vc.items())))
shares["established"] = float(calls.established.mean())
OUT["shares"] = shares
log("  established cue %.0f%%; median words %d" % (100 * calls.established.mean(), calls.words.median()))


def js_of(df):
    return [df.loc[df.session == k, "j"].to_numpy() for k in SES]


OUTCOMES = {"speed_1_2": ("speed", 1, 2), "speed_3_5": ("speed", 3, 5), "speed_6_10": ("speed", 6, 10),
            "speed_11_20": ("speed", 11, 20), "rate_1_3": ("rate", 1, 3), "dps_3_5": ("dps", 3, 5)}


def outcome(js, key):
    ch, h0, h1 = OUTCOMES[key]
    return CP.gather(S, js, ch, h0, h1)


def design(df, extra=()):
    cols, names = [], []

    def add(name, v):
        cols.append(np.asarray(v, float))
        names.append(name)

    add("const", np.ones(len(df)))
    for k in SES[1:]:
        add("S:" + k, df.session == k)
    add("time_frac", df.time_frac - 0.5)
    for tg in ["power", "rate", "length", "catch", "finish", "body", "set", "together", "steer", "morale", "tactic"]:
        add("target:" + tg, df.target == tg)
    add("stage:ass", df.stage == "ass")
    add("stage:dec", df.stage == "dec")
    add("level:proc", df.level == "proc")
    add("level:sreg", df.level == "sreg")
    add("level:self", df.level == "self")
    add("action", df.action == "y")
    add("focus:int", df.focus.isin(["int", "mix"]))
    add("focus:ext", df.focus.isin(["ext", "mix"]))
    add("ref", df.ref == "y")
    add("discrep", df.discrep == "y")
    add("appraisal:chal", df.appraisal == "chal")
    add("appraisal:thr", df.appraisal == "thr")
    add("form:analog", df.form == "analog")
    add("form:expl", df.form == "expl")
    add("arousal:up", df.arousal == "up")
    add("arousal:down", df.arousal == "down")
    add("established", df.established)
    add("log_words", np.log(df.words) - np.log(df.words).mean())
    for name, v in extra:
        add(name, v)
    return np.column_stack(cols), names


# ------------------------------------------------------------------ joint model
X, names = design(calls)
P = np.linalg.pinv(X)
keys = list(OUTCOMES)
stat_all = lambda jj: np.concatenate([P @ outcome(jj, o) for o in keys])
k = len(names)
main = {}
for nullname, local in (("global", False), ("local", True)):
    obs, mu, sd, p = CP.permutation_test(stat_all, S, js_of(calls), NPERM, 1, local)
    main[nullname] = {o: {nm: dict(b=float(obs[a * k + q] - mu[a * k + q]), sd=float(sd[a * k + q]), p=float(p[a * k + q]))
                          for q, nm in enumerate(names)} for a, o in enumerate(keys)}
OUT["joint"] = main
log("\nJOINT MODEL (Equation 3): %d calls, %d columns, condition number %.0f. Coefficient minus null mean;"
    " speed mm/s, rate 0.01 spm, dps mm. p global / local null." % (len(calls), k, np.linalg.cond(X)))
log("%-16s %s" % ("", "".join("%22s" % o for o in keys)))
for nm in names:
    if nm.startswith("S:") or nm == "const":
        continue
    cells = []
    for o in keys:
        g, l_ = main["global"][o][nm], main["local"][o][nm]
        scale = 100 if o.startswith("rate") else 1000
        cells.append("%+7.1f %.3f/%.3f" % (scale * g["b"], g["p"], l_["p"]))
    log("%-16s %s" % (nm, "".join("%22s" % c for c in cells)))

# ------------------------------------------------------------------ primary contrasts
PRIMARY = [("H5 action specified", {"action": 1}, "speed_3_5"),
           ("H6 self level", {"level:self": 1}, "speed_3_5"),
           ("H11 internal minus external focus", {"focus:int": 1, "focus:ext": -1}, "speed_3_5"),
           ("H12 named reference (strokes 6-10)", {"ref": 1}, "speed_6_10"),
           ("H17 established cue", {"established": 1}, "speed_3_5"),
           ("H18 analogy minus explicit", {"form:analog": 1, "form:expl": -1}, "speed_3_5"),
           ("H19 arousal raised", {"arousal:up": 1}, "speed_3_5"),
           ("H19 arousal lowered", {"arousal:down": 1}, "speed_3_5"),
           ("appraisal: threat", {"appraisal:thr": 1}, "speed_3_5"),
           ("appraisal: challenge", {"appraisal:chal": 1}, "speed_3_5"),
           ("H9 discrepancy reported", {"discrep": 1}, "speed_3_5"),
           ("H8 load (per log-word)", {"log_words": 1}, "speed_3_5")]
prim = []
for lab, w, key in PRIMARY:
    c = np.zeros(k)
    for nm, v in w.items():
        c[names.index(nm)] = v
    st = lambda jj, c=c, key=key: np.array([c @ (P @ outcome(jj, key))])
    og, mg, sg, pg = CP.permutation_test(st, S, js_of(calls), NPERM, 2)
    ol, ml, sl, pl = CP.permutation_test(st, S, js_of(calls), NPERM, 3, local=True)
    pos = [names.index(nm) for nm, v in w.items() if v > 0]
    prim.append(dict(label=lab, outcome=key, est=float(og[0] - mg[0]), sd=float(sg[0]), p=float(pg[0]),
                     est_local=float(ol[0] - ml[0]), sd_local=float(sl[0]), p_local=float(pl[0]),
                     n=int((X[:, pos] != 0).any(axis=1).sum()) if lab[:3] != "H8 " else len(calls)))
for r, a, b in zip(prim, CP.holm([r["p"] for r in prim]), CP.holm([r["p_local"] for r in prim])):
    r["p_holm"], r["p_local_holm"] = float(a), float(b)
OUT["primary"] = prim
log("\nPRIMARY CONTRASTS (joint model, minus null mean), mm/s; Holm over the twelve")
for r in prim:
    log("  %-36s n=%4d  %+6.1f (sd %4.1f) p %.3f Holm %.2f | local %+6.1f (sd %4.1f) p %.3f Holm %.2f" % (
        r["label"], r["n"], 1000 * r["est"], 1000 * r["sd"], r["p"], r["p_holm"], 1000 * r["est_local"],
        1000 * r["sd_local"], r["p_local"], r["p_local_holm"]))

# ------------------------------------------------------------------ hypothesis-specific tests
H = {}


def both(stat, js, seed):
    og, mg, sg, pg = CP.permutation_test(stat, S, js, NPERM, seed)
    ol, ml, sl, pl = CP.permutation_test(stat, S, js, NPERM, seed + 100, local=True)
    return dict(est=(og - mg).tolist(), sd=sg.tolist(), p=pg.tolist(), est_local=(ol - ml).tolist(),
                sd_local=sl.tolist(), p_local=pl.tolist())


def mean_of(df, key):
    return lambda jj: np.array([np.nanmean(outcome(jj, key))])


def sub(df, mask):
    return df[mask.to_numpy()].reset_index(drop=True)


# H1: response curves of directive power calls
pw = sub(calls, (calls.target == "power") & (calls.stage == "dec"))
curve = lambda ch: (lambda jj: np.nanmean(np.column_stack([CP.gather(S, jj, ch, h, h) for h in range(1, 21)]), 0))
H["H1_power_speed_curve"] = dict(n=len(pw), **both(curve("speed"), js_of(pw), 10))
H["H1_power_rate_curve"] = dict(n=len(pw), **both(curve("rate"), js_of(pw), 12))
# the same curve for every call, for comparison
H["H1_all_calls_speed_curve"] = dict(n=len(calls), **both(curve("speed"), js_of(calls), 14))

# H2 spacing (directive calls): mean 3-5 response when the same target was called <=4 strokes before
dec = sub(calls, calls.stage == "dec")
short = (dec.interval <= 4).to_numpy()
H["H2_spacing"] = dict(n_short=int(short.sum()), n_long=int((~short).sum()), **both(
    lambda jj: np.array([np.nanmean(outcome(jj, "speed_3_5")[short]) - np.nanmean(outcome(jj, "speed_3_5")[~short])]), js_of(dec), 20))

# H3 contingency, recomputed at the shifted strokes
ev = sub(calls, (calls.stage == "ass") & calls.appraisal.isin(["chal", "thr"]))
sign = np.where(ev.appraisal == "chal", 1.0, -1.0)


def contingency_stat(jj):
    pre = np.concatenate([[CP.contingency(s, j) for j in js] for s, js in zip(S, jj)])
    cont = (sign * pre) > 0
    y = outcome(jj, "speed_3_5")
    return np.array([np.nanmean(y[cont]) - np.nanmean(y[~cont])])


obs_pre = np.array([CP.contingency(sessions[s_], j) for s_, j in zip(ev.session, ev.j)])
H["H3_contingency"] = dict(n=len(ev), n_contingent=int(((sign * obs_pre) > 0).sum()), **both(contingency_stat, js_of(ev), 30))

# H4 payback: power calls, strokes 11-20, by half of the piece (local null keeps the half)
for half, m in (("first", pw.time_frac < 0.5), ("second", pw.time_frac >= 0.5)):
    d_ = sub(pw, m)
    H["H4_power_11_20_%s_half" % half] = dict(n=len(d_), **both(mean_of(d_, "speed_11_20"), js_of(d_), 40))
    H["H4_power_3_5_%s_half" % half] = dict(n=len(d_), **both(mean_of(d_, "speed_3_5"), js_of(d_), 42))

# H7 target channels
for lab, m, key in (("rate_calls_rate_1_3", (calls.target == "rate") & (calls.stage == "dec"), "rate_1_3"),
                    ("rate_calls_dps_3_5", (calls.target == "rate") & (calls.stage == "dec"), "dps_3_5"),
                    ("rate_calls_speed_3_5", (calls.target == "rate") & (calls.stage == "dec"), "speed_3_5"),
                    ("length_calls_dps_3_5", (calls.target == "length") & (calls.stage == "dec"), "dps_3_5"),
                    ("length_calls_speed_3_5", (calls.target == "length") & (calls.stage == "dec"), "speed_3_5"),
                    ("power_calls_rate_1_3", (calls.target == "power") & (calls.stage == "dec"), "rate_1_3"),
                    ("power_calls_speed_3_5", (calls.target == "power") & (calls.stage == "dec"), "speed_3_5"),
                    ("technique_calls_speed_3_5", calls.target.isin(["catch", "finish", "body", "set", "together"]) & (calls.stage == "dec"), "speed_3_5"),
                    ("steer_calls_speed_3_5", (calls.target == "steer"), "speed_3_5")):
    d_ = sub(calls, m)
    H["H7_" + lab] = dict(n=len(d_), **both(mean_of(d_, key), js_of(d_), 50))


# interactions inside the joint model; intensity recomputed at shifted strokes for H17
def interaction(name, v, seed):
    Xi, ni = design(calls, [(name, v)])
    Pi = np.linalg.pinv(Xi)
    q = ni.index(name)
    return dict(n=len(calls), **both(lambda jj: np.array([(Pi @ outcome(jj, "speed_3_5"))[q]]), js_of(calls), seed))


lw = np.log(calls.words) - np.log(calls.words).mean()
H["H8_load_x_time"] = interaction("logwords_x_time", lw * (calls.time_frac - 0.5), 60)
H["H18_analog_x_time"] = interaction("analog_x_time", (calls.form == "analog") * (calls.time_frac - 0.5), 62)
H["H9_discrep_x_time"] = interaction("discrep_x_time", (calls.discrep == "y") * (calls.time_frac - 0.5), 64)
est = calls.established.to_numpy()


def established_x_intensity(jj):
    inten = np.concatenate([s["rate"][js - 1] / s["med_rate"] for s, js in zip(S, jj)])
    v = est * (inten - inten.mean())
    Xi, ni = design(calls, [("est_x_int", v), ("intensity", inten - inten.mean())])
    return np.array([(np.linalg.pinv(Xi) @ outcome(jj, "speed_3_5"))[ni.index("est_x_int")]])


H["H17_established_x_intensity"] = dict(n=len(calls), **both(established_x_intensity, js_of(calls), 66))

# H20 delivery, sessions with audio
aud = sub(calls, calls.session.isin(list(AUDIO)) & calls.loud_z.notna())
Xa, na = design(aud, [("loud_z", aud.loud_z), ("loud_x_time", aud.loud_z * (aud.time_frac - 0.5)),
                      ("f0_z", aud.f0_z.fillna(0)), ("srate", aud.srate - aud.srate.mean())])
Pa = np.linalg.pinv(Xa)
ix = [na.index(c) for c in ("loud_z", "loud_x_time", "f0_z", "srate")]
H["H20_delivery"] = dict(n=len(aud), terms=["loud_z", "loud_x_time", "f0_z", "srate"],
                         **both(lambda jj: (Pa @ outcome(jj, "speed_3_5"))[ix], js_of(aud), 70))

# H14 counts and variability; H19 arousal and variability
H["H14_counts_variance"] = dict(n=len(counts), **both(lambda jj: np.array([np.nanmean(CP.gather_var(S, jj))]), js_of(counts), 80))
up = sub(calls, calls.arousal == "up")
H["H19_arousal_up_variance"] = dict(n=len(up), **both(lambda jj: np.array([np.nanmean(CP.gather_var(S, jj))]), js_of(up), 82))
dn = sub(calls, calls.arousal == "down")
H["H19_arousal_down_variance"] = dict(n=len(dn), **both(lambda jj: np.array([np.nanmean(CP.gather_var(S, jj))]), js_of(dn), 84))

# H22 adverse events: the coxswain's call rate around them, and responses to calls just after them
adv = [s["adverse"][(s["adverse"] >= s["valid"][0]) & (s["adverse"] < s["valid"][1])] for s in S]
call_js = js_of(calls)


def call_rate_around(adv_by_session):
    after, before = [], []
    for js, a_ in zip(call_js, adv_by_session):
        for a in a_:
            after.append(((js > a) & (js <= a + 10)).sum())
            before.append(((js > a - 10) & (js <= a)).sum())
    return np.array([np.mean(after) - np.mean(before), np.mean(after)])


H["H22_call_rate_after_adverse"] = dict(n_events=int(sum(len(a) for a in adv)), **both(call_rate_around, adv, 90))


def after_adverse_stat(jj):
    near = np.concatenate([np.array([any(0 < j - b <= 10 for b in a_) for j in js], bool) for js, a_ in zip(jj, adv)])
    y = outcome(jj, "speed_3_5")
    return np.array([np.nanmean(y[near]) - np.nanmean(y[~near])])


H["H22_calls_after_adverse"] = dict(n=len(calls), **both(after_adverse_stat, call_js, 92))
OUT["hypotheses"] = H


def show(name, v, scale=1000):
    e, s_, p, el, sl, pl = (np.atleast_1d(v[x]) for x in ("est", "sd", "p", "est_local", "sd_local", "p_local"))
    if len(e) > 3:
        return
    nn = {a: b for a, b in v.items() if a.startswith("n")}
    log("  %-34s %s  %s" % (name, " ".join("%s=%s" % (a, b) for a, b in nn.items()), " | ".join(
        "%+.1f (sd %.1f) p %.3f; local %+.1f (sd %.1f) p %.3f" % (scale * a, scale * b, c, scale * d, scale * f, g)
        for a, b, c, d, f, g in zip(e, s_, p, el, sl, pl))))


log("\nHYPOTHESIS-SPECIFIC (minus null mean; speed mm/s, dps mm, rate spm x1000 shown as 0.001 spm, counts per 10 strokes x1000)")
for name, v in H.items():
    show(name, v)
c = H["H1_power_speed_curve"]
log("  H1 power-call speed curve (mm/s): " + " ".join("%+.0f" % (1000 * x) for x in c["est"]))
log("     global null sd:               " + " ".join("%.0f" % (1000 * x) for x in c["sd"]))
log("     local estimate:               " + " ".join("%+.0f" % (1000 * x) for x in c["est_local"]))
c = H["H1_power_rate_curve"]
log("  H1 power-call rate curve (spm):  " + " ".join("%+.2f" % x for x in c["est"]))

# ------------------------------------------------------------------ heterogeneity and crossover
w = outcome(js_of(calls), "speed_3_5")
rng = np.random.default_rng(0)
noise = np.concatenate([np.nanmean(s["R"]["speed"][s["valid"][0]:s["valid"][1], 2:5], 1) for s in S])
B = np.array([np.var(rng.choice(w, len(w)), ddof=1) - np.var(rng.choice(noise, len(noise)), ddof=1) for _ in range(4000)])
d = np.var(w, ddof=1) - np.var(noise, ddof=1)
OUT["heterogeneity"] = dict(n=len(w), sd_calls=float(np.std(w, ddof=1)), sd_noise=float(np.std(noise, ddof=1)),
                            true_sd=float(np.sqrt(max(d, 0))), ci_var=[float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))],
                            reliability=float(max(d, 0) / np.var(w, ddof=1)),
                            reliability_upper=float(max(np.percentile(B, 97.5), 0) / np.var(w, ddof=1)))
h = OUT["heterogeneity"]
log("\nHETEROGENEITY: per-call sd %.1f vs call-free %.1f mm/s; true-effect sd %.1f mm/s; reliability %.3f (upper %.3f)" % (
    1000 * h["sd_calls"], 1000 * h["sd_noise"], 1000 * h["true_sd"], h["reliability"], h["reliability_upper"]))

cross = {}
for k_ in ("practice_01", "practice_02"):
    s = sessions[k_]
    c_ = calls[calls.session == k_]
    cross[k_] = dict(strokes=len(s["t"]), mean_speed=float(s["speed"].mean()), mean_rate=float(s["rate"].mean()),
                     calls=len(c_), calls_per_min=float(len(c_) / ((s["t"][-1] - s["t"][0]) / 60)),
                     thr=int((c_.appraisal == "thr").sum()), chal=int((c_.appraisal == "chal").sum()),
                     self_level=int((c_.level == "self").sum()), arousal_up=int((c_.arousal == "up").sum()),
                     power=int((c_.target == "power").sum()), established=float(c_.established.mean()))
OUT["crossover"] = cross
log("\nCROSSOVER practice_01 vs practice_02:")
for k_, v in cross.items():
    log("  %s %s" % (k_, ", ".join("%s %s" % (a, ("%.3g" % b) if isinstance(b, float) else b) for a, b in v.items())))

# ------------------------------------------------------------------ 7 Oct 2026: drills, "hold it", calling density
T = {}
tod = sub(calls, calls.session.isin(TODAY))
tech = tod.target.isin(["catch", "finish", "body", "set", "together", "length", "power"]) & (tod.stage == "dec") & (tod.level == "proc")
dr = sub(tod, tech)
m = (dr.drilled == 1).to_numpy()
T["drilled_vs_other_technical"] = dict(n_drilled=int(m.sum()), n_other=int((~m).sum()), **both(
    lambda jj: np.array([np.nanmean(outcome(jj, "speed_3_5")[m]) - np.nanmean(outcome(jj, "speed_3_5")[~m])]), js_of(dr), 200))
pr = sub(tod, (tod.appraisal == "chal") & (tod.stage == "ass") | (tod.hold == 1))
h = (pr.hold == 1).to_numpy()
for key in ("speed_3_5", "speed_6_10"):
    T["hold_vs_plain_praise_" + key] = dict(n_hold=int(h.sum()), n_plain=int((~h).sum()), **both(
        lambda jj, key=key: np.array([np.nanmean(outcome(jj, key)[h]) - np.nanmean(outcome(jj, key)[~h])]), js_of(pr), 210))
pwt = sub(tod, (tod.target == "power") & (tod.stage == "dec"))
T["power_calls_today_3_5"] = dict(n=len(pwt), **both(mean_of(pwt, "speed_3_5"), js_of(pwt), 220))
pieces = {}
for k in TODAY:
    s_ = sessions[k]
    c_ = calls[calls.session == k]
    one = s_["R"]["speed"][:, 0]
    d = s_["meta"]
    dist = s_["speed"].sum() * 0 + (np.sum(np.diff(s_["t"]) * s_["speed"][1:]))
    pieces[k] = dict(note=d.get("note", ""), strokes=len(s_["t"]), minutes=float((s_["t"][-1] - s_["t"][0]) / 60),
                     calls=len(c_), calls_per_min=float(len(c_) / ((s_["t"][-1] - s_["t"][0]) / 60)),
                     info_share=float((c_.target == "info").mean()) if len(c_) else 0.0,
                     mean_speed=float(s_["speed"].mean()), avg_split=float(500 / s_["speed"].mean()),
                     mean_rate=float(s_["rate"].mean()), resid_sd=float(np.nanstd(one)), rate_sd=float(np.std(s_["rate"][5:-5])))
T["pieces"] = pieces
OUT["today"] = T
log("\n7 OCT 2026 PRACTICE (six pieces, a four)")
for k in ("drilled_vs_other_technical", "hold_vs_plain_praise_speed_3_5", "hold_vs_plain_praise_speed_6_10", "power_calls_today_3_5"):
    show(k, T[k])
for k, v in pieces.items():
    log("  %s %-60s calls/min %4.1f info %3.0f%%  split %.1f s  rate %.1f  resid sd %.0f mm/s  rate sd %.2f" % (
        k, v["note"][:60], v["calls_per_min"], 100 * v["info_share"], v["avg_split"], v["mean_rate"], 1000 * v["resid_sd"], v["rate_sd"]))

OUT["course_terms"] = {k: sessions[k].get("course_terms", 0) for k in SES}
json.dump(OUT, open(os.path.join(HERE, "results%s.json" % SUF), "w"), indent=1)
log("\nwritten results.json, episodes.csv")
