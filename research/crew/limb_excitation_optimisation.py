"""The optimisation step on the legs and arms: excitations solved for the knee and elbow, validated against EMG (§275).

As trunk_excitation_optimisation.py (§274) for the trunk, here for the two other joints of the multi-clock body. Over
one stroke the agonist and antagonist excitations e(t) in [0, 1] minimise the joint-moment error plus a small effort
term through first-order activation dynamics (tau 25 ms, periodic):

* knee: extensors and flexors, Anderson et al. 2007 (KE, KF; passive terms on) on the knee flexion and its rate;
* elbow: flexors and extensors, Frey-Law et al. 2012.

The moments are rung 2's logged joint demands (muscle_capacity.py --rung2; data/local/crew/muscle_capacity_rung2_s253.json),
i.e. today's kinematic body (§255: it loads its joints 0.1-0.3 s before rowers' muscles fire). Nothing is fitted to
EMG; the solved timing is compared with Vieira et al. 2020's vasti (ergometer) for the knee extensors and Pitto et
al. 2024's arm pullers and triceps for the elbow. The comparison therefore tests the body's demand timing, not the
optimiser: a solved excitation that leads its EMG is the kinematic body's early loading, seen from the muscles.

    python research/crew/limb_excitation_optimisation.py
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import lsq_linear

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
import trunk_excitation_optimisation as TEO                       # noqa: E402

from coxswain.crew import joint_muscles as J                      # noqa: E402
from coxswain.crew.excitation import Modules, PITTO2024_FINISH    # noqa: E402

DEMANDS = os.path.join(ROOT, "data", "local", "crew", "muscle_capacity_rung2_s253.json")
GROUP = {"cr06": "F18-25", "br24": "M18-25"}
SEX = {"cr06": "F", "br24": "M"}
N = 240


def solve_joint(t, y, th, w, T_ag, T_an, P_pas, P):
    """Solve agonist / antagonist excitations for one joint over one stroke (periodic), as §274."""
    tau = np.linspace(0.0, P, N, endpoint=False)
    yk = np.interp(tau, t, y, period=P)
    thk, wk = np.interp(tau, t, th, period=P), np.interp(tau, t, w, period=P)
    TA = np.array([T_ag(a, b) for a, b in zip(thk, wk)])
    TN = np.array([T_an(a, b) for a, b in zip(thk, wk)])
    PS = np.array([P_pas(a) for a in thk])
    L = TEO.activation_matrix(N, P / N, TEO.TAU_ACT)
    s = max(np.std(yk), 1.0)
    eps = 1e-3
    A = np.vstack([np.hstack([TA[:, None] * L, -TN[:, None] * L]) / s, np.sqrt(eps) * np.eye(2 * N)])
    b = np.concatenate([(yk - PS) / s, np.zeros(2 * N)])
    res = lsq_linear(A, b, bounds=(0.0, 1.0), method="trf", lsmr_tol="auto", max_iter=2000)
    eA, eN = res.x[:N], res.x[N:]
    fit = (L @ eA) * TA - (L @ eN) * TN + PS
    r2 = 1.0 - np.sum((yk - fit) ** 2) / np.sum((yk - yk.mean()) ** 2)
    return tau, eA, eN, float(r2), float((L @ eA).max()), float((L @ eN).max())


def compare(tau, e, emg, D, P):
    on = tau <= D
    x = tau[on] / D
    cen = float(np.sum(x * e[on]) / max(np.sum(e[on]), 1e-9))
    cene = float(np.sum(x * emg[on]) / max(np.sum(emg[on]), 1e-9))
    best = (-2.0, 0)
    for lag in range(-N // 4, N // 4 + 1):
        c = np.corrcoef(np.roll(emg, lag), e)[0, 1]
        if np.isfinite(c) and c > best[0]:
            best = (float(c), lag)
    return dict(peak=float(x[int(np.argmax(e[on]))]), centroid=cen, emg_peak=float(x[int(np.argmax(emg[on]))]),
                emg_centroid=cene, r_zero=float(np.corrcoef(emg, e)[0, 1]), r_best=best[0],
                lag_ms=1000.0 * best[1] * P / N)


def main(path=None):
    path = path or (sys.argv[1] if len(sys.argv) > 1 else DEMANDS)
    d = json.load(open(path))
    print("demands from", os.path.basename(path))
    mods = Modules.pitto2024()
    res = {}
    for a in ("cr06", "br24"):
        s = {k: np.asarray(v, float) for k, v in d[a]["series"].items()}
        t = s["t"]
        F = s["F"]
        D = float(t[np.flatnonzero(F > 0.05 * F.max())[-1]])          # the drive, as sequencing_vs_emg.py
        P = D / PITTO2024_FINISH                                       # the cycle on the EMG's drive fraction
        ok = np.isfinite(s["knee"]) & np.isfinite(s["elbow"])
        t, s = t[ok], {k: v[ok] for k, v in s.items()}
        mass, stature = d[a]["mass"], d[a]["stature"]
        KE = J.anderson("KE", GROUP[a], mass, stature)
        KF = J.anderson("KF", GROUP[a], mass, stature)
        EF, EE = J.elbow("flexion", SEX[a]), J.elbow("extension", SEX[a])
        clip_k = lambda th: float(np.clip(th, 0.0, 1.8))               # Anderson's tested knee range (B1's clamp)
        knee = solve_joint(t, s["knee"], s["th_knee"], s["w_knee"],
                           lambda th, w: float(np.ravel(KE.max_active(clip_k(th), -w))[0]),
                           lambda th, w: float(np.ravel(KF.max_active(clip_k(th), w))[0]),
                           lambda th: float(np.ravel(KE.passive(clip_k(th)))[0] - np.ravel(KF.passive(clip_k(th)))[0]), P)
        elbow = solve_joint(t, s["elbow"], s["th_elbow"], s["w_elbow"],
                            lambda th, w: float(np.ravel(EF.max_active(th, w))[0]),
                            lambda th, w: float(np.ravel(EE.max_active(th, -w))[0]),
                            lambda th: 0.0, P)
        out = {}
        for joint, (tau, eA, eN, r2, aA, aN), pairs in (
                ("knee", knee, (("extensors", "leg_extensors", 1),)),
                ("elbow", elbow, (("flexors", "arm_pullers", 1), ("extensors", "arm_extensors", 2)))):
            out[joint] = dict(r2=r2, peak_activation=(aA, aN))
            print("%s %-5s: moment reproduced r^2 %.3f; peak activations %.2f / %.2f" % (a, joint, r2, aA, aN))
            for label, mod, which in pairs:
                e = eA if which == 1 else eN
                emg = mods(mod, tau / P)
                v = compare(tau, e, emg, D, P)
                out[joint][label] = v
                print("   %-10s solved peak %3.0f %% / centroid %3.0f %%; EMG (%s) peak %3.0f %% / centroid %3.0f %%; "
                      "r %.2f at zero lag, %.2f at %+.0f ms" % (label, 100 * v["peak"], 100 * v["centroid"], mod,
                                                               100 * v["emg_peak"], 100 * v["emg_centroid"],
                                                               v["r_zero"], v["r_best"], v["lag_ms"]))
        res[a] = out
    out_path = os.path.join(ROOT, "data", "local", "crew", "limb_excitation_optimisation_%s" % os.path.basename(path))
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
