"""Directive power calls, speed beyond momentum over strokes 3-5, piece by piece (global and local nulls)."""
import json, os, sys
import numpy as np, pandas as pd
ROOT = r"C:\Users\satur\PycharmProjects\CoxswainSimulator_V_0"
sys.path.insert(0, os.path.join(ROOT, "research", "callmodel"))
import call_properties as CP
CODE = os.path.dirname(os.path.abspath(__file__))   # this script (committed, research/)
HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "local", "coxing", "experiment"))   # its data and outputs (gitignored)
D = pd.read_csv(os.path.join(HERE, "episodes.csv"))
pw = D[(D.target == "power") & (D.stage == "dec")]
out = {}
for sid in ["race_01", "practice_01", "practice_02", "race_04", "race_05", "practice_03", "practice_04", "practice_05", "practice_06", "practice_07", "practice_08", "race_06", "race_07"]:
    s = CP.prepare(CP.load_session(os.path.join(ROOT, "data", "local", "coxing", "sessions", sid)))
    js = pw.loc[pw.session == sid, "j"].to_numpy()
    st = lambda jj: np.array([np.nanmean(CP.gather([s], jj, "speed", 3, 5))])
    og, mg, sg, pg = CP.permutation_test(st, [s], [js], 2000, 1)
    ol, ml, sl, pl = CP.permutation_test(st, [s], [js], 2000, 2, local=True)
    out[sid] = dict(n=len(js), est=float(og[0] - mg[0]), sd=float(sg[0]), p=float(pg[0]), est_local=float(ol[0] - ml[0]), sd_local=float(sl[0]), p_local=float(pl[0]))
    print("%-12s n=%3d  %+5.1f (sd %4.1f) p %.3f | local %+5.1f (sd %4.1f) p %.3f" % (sid, len(js), 1000*out[sid]["est"], 1000*sg[0], pg[0], 1000*out[sid]["est_local"], 1000*sl[0], pl[0]))
json.dump(out, open(os.path.join(HERE, "per_piece.json"), "w"), indent=1)
