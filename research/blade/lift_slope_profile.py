"""The catch's build-up, tank first: does a single lift slope fit all three measured loads? (§231)

The cross-flow law's lift term is Helmbold's slope at the blade's along-shaft aspect ratio (C_N = C_La sin a cos a +
C_Dc sin^2 a; coxswain.crew.liftdrag.LiftDragBlade.crossflow). The early drive is lift-dominated ([GTW21] sec. 4.2.4)
and the law is short there on [GTW21] and on [CR06] (§190-191, §228). As the face-on value was profiled per dataset
(§191), this profiles a multiplier k on the lift slope per dataset at that dataset's own face-on value, scored by each
dataset's own measure. A sensitivity, not a fit that enters the model: if the three agree on one k (within each one's
10 %-of-least band) the value is tank-sourced and may go to the athletes; if they disagree, the lift slope is not the
missing piece.

    python research/blade/lift_slope_profile.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
from coxswain.crew.liftdrag import CrossflowBlade                 # noqa: E402
from coxswain.hydro.appendages import polhamus_helmbold_slope     # noqa: E402

KS = np.round(np.arange(0.6, 2.61, 0.1), 2)


def blade(area, length, face_on, k):
    return CrossflowBlade(k * polhamus_helmbold_slope(area / length ** 2) / 2.0, 2.07, area, 1.0, 1000.0, None,
                          face_on, True)


def gtw21(k):
    import grift2021_quasisteady as G
    g, th, thd, vtip = G.motion()
    star = g / G.T_DRIVE
    fn_meas, _ = G.measured(star)
    law = G.liftdrag_law(blade(G.SPAN * G.HEIGHT, G.SPAN, 2.2, k))
    F = np.array([G.strip_loads(a, b, c, law) for a, b, c in zip(th, thd, vtip)])
    r = G.score(star, -F[:, 0], fn_meas)
    return r["rms"], r["fifths"][:4]


def rlbqv19(k):
    import robert2019_quasisteady as R
    D = R.load()
    K = R.kinematics(D, rigid=False, n=300)
    meas = np.stack([np.interp(K["psi_deg"], *D[q].T) for q in ("Fx_N", "Fy_N", "Mz_Nm")], 1)
    r = R.score(K, R.loads(K, R.liftdrag_law(blade(R.AREA, R.SPAN, 1.3, k))), meas)
    return r["Fx"]["rms"], r["Fx"]["windows"]


def ljh12(k, _cache={}):
    import leroyer_jh12_quasisteady as J
    D = J.load()
    theta = np.radians(np.linspace(1.0, 179.0, 357))
    td = np.degrees(theta)
    _cl, _cd, _inc, fx, _fy = J.reduce(J.run(J.liftdrag_law(blade(J.AREA, J.SPAN, 2.05, k)), theta), theta)
    mfx = np.interp(td, *D["Fx_N"].T)
    if "sign" not in _cache:
        _cache["sign"] = np.sign(np.corrcoef(fx, mfx)[0, 1])
    fx = _cache["sign"] * fx
    w = [float(np.mean(fx[(td >= a) & (td < b)] - mfx[(td >= a) & (td < b)]) / np.max(np.abs(mfx)))
         for a, b in ((0, 45), (45, 90), (90, 135), (135, 180))]
    return float(np.sqrt(np.mean((fx - mfx) ** 2))), w


def main():
    import json
    rows = {}
    for name, f in (("GTW21", gtw21), ("RLBQV19", rlbqv19), ("LJH12", ljh12)):
        res = [(k,) + f(k) for k in KS]
        rms = np.array([r[1] for r in res])
        best = int(np.argmin(rms))
        band = KS[rms <= 1.1 * rms[best]]
        rows[name] = dict(k=KS.tolist(), rms=rms.tolist(), best=float(KS[best]), band=[float(band.min()), float(band.max())],
                          at1=res[int(np.argmin(abs(KS - 1.0)))][2], atbest=res[best][2])
        print("%-8s least rms %.1f N at k %.1f (within 10 %%: %.1f-%.1f); k 1: rms %.1f, early/late %s; best: %s"
              % (name, rms[best], KS[best], band.min(), band.max(), rms[np.argmin(abs(KS - 1.0))],
                 " ".join("%+.2f" % v for v in rows[name]["at1"]), " ".join("%+.2f" % v for v in res[best][2])))
    lo = max(r["band"][0] for r in rows.values())
    hi = min(r["band"][1] for r in rows.values())
    print("\ncommon k (every dataset within 10 %% of its least rms): %s"
          % ("%.1f-%.1f" % (lo, hi) if lo <= hi else "none"))
    json.dump(rows, open(os.path.join(ROOT, "data", "local", "blade", "lift_slope_profile.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
